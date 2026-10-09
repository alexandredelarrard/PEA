# pyright: reportAttributeAccessIssue=false, reportMissingImports=false, reportUnusedExpression=false
"""
dag_data_extraction.py  (src/dags/dag_data_extraction.py)
---------------------------------------------------------
Nightly DATA-EXTRACTION DAG. One task PER SOURCE (fetcher), so parallelism is tuned by LOAD, not by
big group: the light sources fan out freely, while the heavy / long / rate-limited ones are capped by
Airflow POOLS (created in airflow-init):

  * sec_bulk (2 slots)  — big SEC zip downloads: insider_download, notes_download, ftd_download,
                          sec_tickers (one GET), fails_to_deliver,
                          financial_statements, insider_zip (the zip parse), financial_notes
                          (disk + SEC bandwidth bound)
  * sec_api  (2 slots)  — per-ticker EDGAR API (shared 10 req/s); each task consumes both
                          slots, so only one EDGAR walk runs at a time (insider_edgar included)
  * default             — light / fast: macro, short_interest, earnings_surprises,
                          superinvestors, earnings calls  (+ the one heavy yfinance pull:
                          price_history)

Flow: seed_universe -> (fetchers, with source dependencies) -> extraction_status -> identity_check -> trigger
the data_aggregation DAG. Every task has `retries: 3`, so four attempts in all. No source blocks
the night: `extraction_status` and `identity_check` run on ALL_DONE, so they start once every fetcher
has finished, failed or not. The aggregation trigger runs only when `identity_check` succeeds (the
hard identity gate); extraction itself never waits for it. Identity stage: the Form 3/4/5, Notes and FTD
zip downloads and the SEC current-tickers snapshot -> identity_tables (tenure, lineage, security
master) -> identity_propagate -> every task that reads the lineage (price_history waits for
identity_tables only: it reads the master's current secondary classes); a failed build or propagation
leaves those consumers unrun. identity_tables and price_history run on ALL_DONE: a failed download leaves
the build on its cached evidence, and a failed build leaves prices on the stored master. After every fetcher, `extraction_status` prints the per-table freshness
report, logs a WARNING per RED table and exits 0; only `modelling predict` refuses stale inputs. Then
`identity_check` (`python -m src validate identity`) fails on rows filed by a CIK outside the ticker's
entity or a broken lineage invariant, which holds the aggregation trigger.

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
    start_date=datetime(2026, 11, 11),
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
earnings_surprises = fetch("earnings-surprises")

# yfinance: one download writes prices, prices_dividends and prices_splits (a new split re-pulls its ticker)
price_history = fetch("price-history", trigger_rule=TriggerRule.ALL_DONE)  # the stored master is enough when identity_tables fails

# 2) identity stage: cache the Form 3/4/5 zips, the Notes zips (+ cover-page dei symbols) and the FTD zips
#    (+ raw in-scope lines), snapshot SEC's current tickers, build symbol_tenure + entity_lineage +
#    security_master offline, then carry lineage changes into the stored rows
insider_download = fetch("insider-download", pool="sec_bulk")
notes_download = fetch("notes-download", pool="sec_bulk")
ftd_download = fetch("ftd-download", pool="sec_bulk")
sec_tickers = fetch("sec-tickers", pool="sec_bulk")
identity_tables = fetch("identity-tables", trigger_rule=TriggerRule.ALL_DONE)  # builds offline from cached evidence
identity_propagate = fetch("identity-propagate")

# 3) SEC bulk zips — capped to 2 concurrent (disk + SEC bandwidth)
fails_to_deliver = fetch("fails-to-deliver", pool="sec_bulk")
thirteen_f = fetch("thirteen-f", pool="sec_api")
# new tickers' 13F history (cached data-set ZIPs + one EDGAR walk); a no-op on nights with no new ticker
thirteen_f_backfill = fetch("thirteen-f-backfill", pool="sec_api")
financial_statements = fetch("financial-statements", pool="sec_bulk")
insider_zip = fetch("insider-zip", pool="sec_bulk")  # the zip half; the EDGAR half is insider_edgar (sec_api)
financial_notes = fetch("financial-notes", pool="sec_bulk")  # VERY heavy
superinvestors = fetch("superinvestors")  # light, needs 13F
thirteen_f_managers = fetch("thirteen-f-managers", pool="sec_api")  # roster books, needs roster
#   ^ institutionals step: thirteen_f, insider_zip + insider_edgar, fails_to_deliver,
#     superinvestors, short-interest, sec_8k_items, sec_13d and sec_13g (below)

# 4) per-ticker EDGAR API — capped to 2 (shared SEC 10 req/s)
fundamentals = fetch("fundamentals", pool="sec_api")
insider_edgar = fetch("insider-edgar", pool="sec_api")  # Form 3/4/5 EDGAR walk, after the zip half
fundamentals_employees = fetch("fundamentals-employees", pool="sec_api")
fundamentals_sharadar = fetch("fundamentals-sharadar")  # vendor tables + merged consumer history
def14a = fetch("def14a", pool="sec_api")  # + LLM
def14a_edgar = fetch("def14a-edgar", pool="sec_api")  # deterministic PVP XBRL
sec_8k_items = fetch("sec-8k-items", pool="sec_api")  # 8-K item codes (structured)
sec_8k_votes = fetch("sec-8k-votes", pool="sec_api")  # parses stored Item 5.07 narratives
sec_13d = fetch("sec-13d", pool="sec_api")  # SC 13D activist filings
sec_13g = fetch("sec-13g", pool="sec_api")  # SC 13G passive 5%+ stakes
filing_text = fetch("filing-text", pool="sec_api")  # 10-K Item 1A + Item 7 text

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
    fails_to_deliver,
    thirteen_f,
    thirteen_f_backfill,
    financial_statements,
    insider_zip,
    insider_edgar,
    financial_notes,
    insider_download,
    notes_download,
    ftd_download,
    sec_tickers,
    identity_tables,
    identity_propagate,
    fundamentals,
    fundamentals_employees,
    fundamentals_sharadar,
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
    insider_zip,
    short_interest,
    fails_to_deliver,
    financial_statements,
    financial_notes,
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

seed_universe >> all_fetchers
identity_tables >> price_history  # the fetch list adds the security master's current secondary classes
[insider_download, notes_download, ftd_download, sec_tickers] >> identity_tables >> identity_propagate >> identity_consumers
insider_zip >> insider_edgar  # zips fill first; EDGAR resumes after their last quarter and replaces each filing it re-reads
thirteen_f >> thirteen_f_backfill  # one EDGAR walk at a time: after the nightly walk
thirteen_f >> superinvestors  # roster gate reads the filers' 13F activity
superinvestors >> thirteen_f_managers  # roster IS the walk scope
fundamentals >> fundamentals_sharadar  # headcount is read by the cube, not merged
[sec_8k_items, def14a] >> sec_8k_votes

# every source done (failed or not) -> freshness report -> identity check -> aggregation only when the check passed
all_fetchers >> extraction_status >> identity_check >> trigger_aggregation
