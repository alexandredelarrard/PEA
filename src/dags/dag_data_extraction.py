# pyright: reportAttributeAccessIssue=false, reportMissingImports=false, reportUnusedExpression=false
"""
dag_data_extraction.py  (src/dags/dag_data_extraction.py)
---------------------------------------------------------
Nightly DATA-EXTRACTION DAG. One task PER SOURCE (fetcher, or one step sub-run), so parallelism is tuned by
LOAD, not by big group: the light sources fan out freely, while the heavy / long / rate-limited ones are
capped by Airflow POOLS (created in airflow-init):

  * sec_bulk (2 slots)  — the SEC bulk zips: zips_download (Form 3/4/5, Notes and FTD
                          zips + the SEC current-tickers GET), zips_ingest_insider (the Form 3/4/5 zip
                          parse) and zips_ingest_datasets (notes, FTD; disk + SEC bandwidth bound)
  * sec_api  (2 slots)  — per-ticker EDGAR API (shared 10 req/s); each task consumes both
                          slots, so only one EDGAR walk runs at a time (edgar_index and insider_edgar included)
  * default             — light / fast: macro, short_interest, earnings_surprises, identity,
                          superinvestors, earnings calls  (+ the one heavy yfinance pull:
                          price_history)

Flow: seed_universe -> (fetchers, with source dependencies) -> extraction_status -> identity_check -> trigger
the data_aggregation DAG. Every task has `retries: 3`, so four attempts in all. No source blocks
the night: `extraction_status` and `identity_check` run on ALL_DONE, so they start once every fetcher
has finished, failed or not. The aggregation trigger runs only when `identity_check` succeeds (the
hard identity gate); extraction itself never waits for it. Identity stage: zips_download -> identity
(captures, symbol_tenure + entity_lineage + security_master, propagation; offline) -> every task that reads
the lineage, the two zip ingests included (price_history waits for identity: it reads the master's current
secondary classes); a failed identity run leaves those consumers unrun. identity and price_history run on
ALL_DONE: a failed download leaves the build on its cached evidence, and a failed build leaves prices on the
stored master. edgar_index refreshes the EDGAR filing index once after identity; the six EDGAR document walks
wait for it and skip a refresh already done today (the insider walk still rebuilds its own inventory), and
insider_edgar also waits for the insider zip ingest. After every fetcher, `extraction_status` prints the
per-table freshness report, logs a WARNING per RED table and exits 0; only `modelling predict` refuses stale
inputs. Then `identity_check` (`python -m src validate identity`) fails on rows filed by a CIK outside the
ticker's entity or a broken lineage invariant, which holds the aggregation trigger.

Every command is `/opt/pipeline/bin/python -m src data_extract <cmd>` (the pipeline's isolated venv),
run from the mounted repo. Fetchers are incremental, so a nightly run only pulls new data.
"""

from datetime import datetime, timedelta

from airflow.operators.bash import BashOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.utils.trigger_rule import TriggerRule

from airflow import DAG

PROJECT = "/opt/airflow/project"  # the repo, bind-mounted
CONFIGS = f"{PROJECT}/configs"
PIPE_PY = "/opt/pipeline/bin/python"  # pipeline's isolated venv interpreter
PIPE = f"{PIPE_PY} -m src data_extract"

default_args = {
    "owner": "pea",
    "depends_on_past": False,
    "retries": 3,
    "retry_delay": timedelta(minutes=3),
    "email_on_failure": False,
}

dag = DAG(
    dag_id="data_extraction",
    default_args=default_args,
    description="Refresh every raw data source (one task per fetcher) before the nightly cube build.",
    schedule="0 1 * * *",  # 01:00 daily
    start_date=datetime(2026, 10, 11),
    catchup=False,
    max_active_tasks=4,
    tags=["pea", "extraction"],
)


def fetch(
    cmd: str,
    pool: str = "default_pool",
    task_id: str | None = None,
    trigger_rule: str = TriggerRule.ALL_SUCCESS,
) -> BashOperator:
    """Run one retryable extraction command from the pipeline venv."""
    return BashOperator(
        task_id=task_id or cmd.replace("-", "_"),
        bash_command=f"{PIPE} {cmd} -c {CONFIGS}",
        cwd=PROJECT,
        pool=pool,
        pool_slots=2 if pool == "sec_api" else 1,
        trigger_rule=trigger_rule,
        dag=dag,
    )


# 0) universe seed — everything downstream resolves the universe from sp500_tickers
seed_universe = fetch("seed-universe")

# 1) LIGHT / fast — fan out in the default pool
macro = fetch("macro")
short_interest = fetch("short-interest")
earnings_surprises = fetch("earnings-surprises")  # StepExtractPrices.run_earnings_surprises

# yfinance: one download writes prices, prices_dividends and prices_splits (a new split re-pulls its ticker)
price_history = fetch("price-history", trigger_rule=TriggerRule.ALL_DONE)  # the stored master is enough when identity fails

# 2) identity stage: cache the Form 3/4/5, Notes and FTD zips and snapshot SEC's current
#    tickers (StepExtractZips download phase); then StepExtractIdentity offline: capture the Notes cover-page
#    symbols and the raw in-scope FTD lines, build symbol_tenure + entity_lineage + security_master, and carry
#    lineage changes into the stored rows
zips_download = fetch("zips-download", pool="sec_bulk")
identity = fetch("identity", trigger_rule=TriggerRule.ALL_DONE)  # builds offline from cached evidence

# 3) SEC bulk zip ingests (StepExtractZips ingest phase) — capped to 2 concurrent (disk + SEC bandwidth); the
#    insider quarters alone, so insider_edgar never waits for the heavy notes ingest
zips_ingest_insider = fetch("zips-ingest --only insider", pool="sec_bulk", task_id="zips_ingest_insider")
zips_ingest_datasets = fetch("zips-ingest --only notes --only ftd", pool="sec_bulk", task_id="zips_ingest_datasets")  # VERY heavy
thirteen_f = fetch("thirteen-f", pool="sec_api")
# new tickers' 13F history (cached data-set ZIPs + one EDGAR walk); a no-op on nights with no new ticker
thirteen_f_backfill = fetch("thirteen-f-backfill", pool="sec_api")
superinvestors = fetch("superinvestors")  # light, needs 13F
thirteen_f_managers = fetch("thirteen-f-managers", pool="sec_api")  # roster books, needs roster
#   ^ institutionals step: thirteen_f, superinvestors, thirteen_f_managers, short-interest

# 4) per-ticker EDGAR API — capped to 2 (shared SEC 10 req/s)
edgar_index = fetch("edgar-index", pool="sec_api")  # one index refresh per night, before the EDGAR walks
fundamentals = fetch("fundamentals", pool="sec_api")
insider_edgar = fetch("insider-edgar", pool="sec_api")  # Form 3/4/5 EDGAR walk, after the zip half
fundamentals_employees = fetch("fundamentals-employees", pool="sec_api")
fundamentals_sharadar = fetch("fundamentals-sharadar")  # vendor tables + merged consumer history
sharadar_actions = fetch("sharadar-actions")  # market-wide Sharadar actions, independent of the SEC walk
def14a = fetch("def14a", pool="sec_api")  # + LLM
def14a_edgar = fetch("def14a-edgar", pool="sec_api")  # deterministic PVP XBRL
sec_8k_items = fetch("sec-8k-items", pool="sec_api")  # 8-K item codes (structured)
sec_8k_votes = fetch("sec-8k-votes", pool="sec_api")  # parses stored Item 5.07 narratives
sec_13d = fetch("sec-13d", pool="sec_api")  # SC 13D activist filings
sec_13g = fetch("sec-13g", pool="sec_api")  # SC 13G passive 5%+ stakes
filing_text = fetch("filing-text", pool="sec_api")  # 10-K Item 1A + Item 7 text
#   ^ EDGAR step: insider_edgar, sec_13d, sec_13g, sec_8k_items, filing_text, def14a_edgar;
#     LLM step: def14a, sec_8k_votes, fundamentals_employees

# 5) earnings calls: HuggingFace defeatbeta parquet -> earnings_call_sections (incremental)
extract_earnings_calls = fetch("extract-earnings-calls")

# 6) the schema-driven freshness report (exit 0), then the identity check: rows filed by a CIK outside the
#    ticker's entity, or a broken lineage invariant, fail it (the items needing a manual decision are logged,
#    not failed). Both run whatever the fetchers' outcome; aggregation is triggered only on the check's success.
identity_check = BashOperator(
    task_id="identity_check",
    bash_command=f"{PIPE_PY} -m src validate identity -o {PROJECT}/reports/validate/identity-nightly -c {CONFIGS}",
    cwd=PROJECT,
    trigger_rule=TriggerRule.ALL_DONE,
    dag=dag,
)
extraction_status = fetch("extraction-status", task_id="extraction_status", trigger_rule=TriggerRule.ALL_DONE)

trigger_aggregation = TriggerDagRunOperator(
    task_id="trigger_data_aggregation",
    trigger_dag_id="data_aggregation",
    wait_for_completion=False,
    reset_dag_run=True,
    trigger_rule=TriggerRule.ALL_SUCCESS,
    dag=dag,
)

# --- wiring ---
all_fetchers = [
    macro,
    short_interest,
    earnings_surprises,
    price_history,
    zips_download,
    identity,
    zips_ingest_insider,
    zips_ingest_datasets,
    thirteen_f,
    thirteen_f_backfill,
    edgar_index,
    insider_edgar,
    fundamentals,
    fundamentals_employees,
    fundamentals_sharadar,
    sharadar_actions,
    def14a,
    def14a_edgar,
    sec_8k_items,
    sec_8k_votes,
    sec_13d,
    sec_13g,
    filing_text,
    extract_earnings_calls,
    superinvestors,
    thirteen_f_managers,
]

identity_consumers = [
    zips_ingest_insider,
    zips_ingest_datasets,
    short_interest,
    fundamentals,
    fundamentals_employees,
    def14a,
    def14a_edgar,
    sec_8k_items,
    sec_13d,
    sec_13g,
    filing_text,
    insider_edgar,
]
edgar_walks = [insider_edgar, sec_13d, sec_13g, sec_8k_items, filing_text, def14a_edgar]

seed_universe >> all_fetchers
identity >> price_history  # the fetch list adds the security master's current secondary classes
zips_download >> identity >> identity_consumers
identity >> edgar_index >> edgar_walks  # the walks plan on tonight's index and skip their own refresh
zips_ingest_insider >> insider_edgar  # zips fill first; EDGAR resumes after their last quarter and replaces each filing it re-reads
thirteen_f >> thirteen_f_backfill  # one EDGAR walk at a time: after the nightly walk
thirteen_f >> superinvestors  # roster gate reads the filers' 13F activity
superinvestors >> thirteen_f_managers  # roster IS the walk scope
fundamentals >> fundamentals_sharadar  # headcount is read by the cube, not merged
[sec_8k_items, def14a] >> sec_8k_votes

# every source done (failed or not) -> freshness report -> identity check -> aggregation only when the check passed
all_fetchers >> extraction_status >> identity_check >> trigger_aggregation
