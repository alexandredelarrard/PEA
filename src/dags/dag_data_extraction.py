# pyright: reportAttributeAccessIssue=false, reportMissingImports=false, reportUnusedExpression=false
"""
dag_data_extraction.py  (src/dags/dag_data_extraction.py)
---------------------------------------------------------
Nightly DATA-EXTRACTION DAG. One task PER SOURCE (fetcher), so parallelism is tuned by LOAD, not by
big group: the light sources fan out freely, while the heavy / long / rate-limited ones are capped by
Airflow POOLS (created in airflow-init):

  * sec_bulk (2 slots)  — big SEC zip downloads: fails_to_deliver, financial_statements,
                          insider_transactions, financial_notes  (disk + SEC bandwidth bound)
  * sec_api  (2 slots)  — per-ticker EDGAR API (shared 10 req/s); each task consumes both
                          slots, so only one EDGAR walk runs at a time
  * default             — light / fast: macro, short_interest, earnings_surprises,
                          superinvestors, earnings calls  (+ the one heavy yfinance pull:
                          price_history)

Flow: seed_universe -> (fetchers, with source dependencies) -> extraction_status -> trigger
the data_aggregation DAG. Fetchers and the schema-driven freshness gate each get three attempts;
the final gate is a hard block.

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
    "retry_delay": timedelta(minutes=10),
    "email_on_failure": False,
}

dag = DAG(
    dag_id="data_extraction",
    default_args=default_args,
    description="Refresh every raw data source (one task per fetcher) before the nightly cube build.",
    schedule="0 1 * * *",  # 01:00 daily
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_tasks=4,
    tags=["pea", "extraction"],
)


def fetch(
    cmd: str,
    pool: str = "default_pool",
    task_id: str | None = None,
) -> BashOperator:
    """Run one retryable extraction command from the pipeline venv."""
    return BashOperator(
        task_id=task_id or cmd.replace("-", "_"),
        bash_command=f"{PIPE} {cmd} -c {CONFIGS}",
        cwd=PROJECT,
        pool=pool,
        pool_slots=2 if pool == "sec_api" else 1,
        dag=dag,
    )


# 0) universe seed — everything downstream resolves the universe from sp500_tickers
seed_universe = fetch("seed-universe")

# 1) LIGHT / fast — fan out in the default pool
macro = fetch("macro")
short_interest = fetch("short-interest")
earnings_surprises = fetch("earnings-surprises")

# yfinance sources: splits must finish before prices so a new event triggers a full re-pull
splits = fetch("splits")
price_history = fetch("price-history")
dividends = fetch("dividends")

# 2) SEC bulk zips — capped to 2 concurrent (disk + SEC bandwidth)
fails_to_deliver = fetch("fails-to-deliver", pool="sec_bulk")
thirteen_f = fetch("thirteen-f", pool="sec_api")
financial_statements = fetch("financial-statements", pool="sec_bulk")
insider_transactions = fetch("insider-transactions", pool="sec_bulk")
financial_notes = fetch("financial-notes", pool="sec_bulk")  # VERY heavy
identity_tables = fetch("identity-tables")
superinvestors = fetch("superinvestors")  # light, needs 13F
thirteen_f_managers = fetch("thirteen-f-managers", pool="sec_api")  # roster books, needs roster
#   ^ institutionals step: thirteen_f, insider_transactions, fails_to_deliver,
#     superinvestors, short-interest, sec_8k_items, sec_13d and sec_13g (below)

# 3) per-ticker EDGAR API — capped to 2 (shared SEC 10 req/s)
fundamentals = fetch("fundamentals", pool="sec_api")
fundamentals_employees = fetch("fundamentals-employees", pool="sec_api")
fundamentals_sharadar = fetch("fundamentals-sharadar")  # vendor tables + merged consumer history
def14a = fetch("def14a", pool="sec_api")  # + LLM
def14a_edgar = fetch("def14a-edgar", pool="sec_api")  # deterministic PVP XBRL
sec_8k_items = fetch("sec-8k-items", pool="sec_api")  # 8-K item codes (structured)
sec_8k_votes = fetch("sec-8k-votes", pool="sec_api")  # parses stored Item 5.07 narratives
sec_13d = fetch("sec-13d", pool="sec_api")  # SC 13D activist filings
sec_13g = fetch("sec-13g", pool="sec_api")  # SC 13G passive 5%+ stakes
filing_text = fetch("filing-text", pool="sec_api")  # 10-K Item 1A + Item 7 text

# 4) earnings calls: HuggingFace defeatbeta parquet -> earnings_call_sections (incremental)
extract_earnings_calls = fetch("extract-earnings-calls")

# 5) final schema-driven freshness gate; a red gate retries and never permits aggregation.
extraction_status = fetch("extraction-status", task_id="extraction_status")

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
    splits,
    price_history,
    dividends,
    fails_to_deliver,
    thirteen_f,
    financial_statements,
    insider_transactions,
    financial_notes,
    identity_tables,
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
]

seed_universe >> all_fetchers
splits >> price_history
insider_transactions >> identity_tables >> identity_consumers
thirteen_f >> superinvestors  # roster reads the 13F holdings
superinvestors >> thirteen_f_managers  # roster IS the walk scope
[fundamentals, fundamentals_employees] >> fundamentals_sharadar
[sec_8k_items, def14a] >> sec_8k_votes

# all sources refreshed -> schema freshness hard gate -> aggregation
all_fetchers >> extraction_status >> trigger_aggregation
