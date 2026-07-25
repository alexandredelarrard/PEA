"""
dag_data_extraction.py  (src/dags/dag_data_extraction.py)
---------------------------------------------------------
Nightly DATA-EXTRACTION DAG. One task PER SOURCE (fetcher), so parallelism is tuned by LOAD, not by
big group: the light sources fan out freely, while the heavy / long / rate-limited ones are capped by
Airflow POOLS (created in airflow-init):

  * sec_bulk (2 slots)  — big SEC zip downloads: fails_to_deliver, thirteen_f, financial_statements,
                          insider_transactions, financial_notes  (disk + SEC bandwidth bound)
  * sec_api  (2 slots)  — per-ticker EDGAR API (shared 10 req/s): fundamentals, employees, def14a
  * scrape   (2 slots)  — external rate-limited scraping: wiki_pageviews, google_trends,
                          download_earnings_calls -> ingest_earnings_calls
  * default             — light / fast: market_prices, macro, macro_assets, short_interest,
                          earnings_surprises, superinvestors  (+ the one heavy yfinance pull: price_history)

Flow: seed_universe -> (all fetchers in parallel, pool-throttled) -> extraction_complete ->
check_data_freshness (data-drift/gap gate: verifies every source is up to date for its cadence
daily..yearly, pushes the latest date per source to XCom, turns RED when not as expected) -> trigger
the data_aggregation DAG. The gate is a visible WARNING, not a hard block (trigger_rule=ALL_DONE), so
aggregation still runs on a red gate; flip to ALL_SUCCESS to hard-stop prediction on stale data.

Every command is `/opt/pipeline/bin/python -m src data_extract <cmd>` (the pipeline's isolated venv),
run from the mounted repo. Fetchers are incremental, so a nightly run only pulls new data.
"""
import json
import subprocess
from datetime import datetime, timedelta

from airflow import DAG
from airflow.exceptions import AirflowFailException
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.utils.trigger_rule import TriggerRule

PROJECT = "/opt/airflow/project"                 # the repo, bind-mounted
CONFIGS = f"{PROJECT}/configs"
PIPE_PY = "/opt/pipeline/bin/python"             # pipeline's isolated venv interpreter
PIPE = f"{PIPE_PY} -m src data_extract"

default_args = {
    "owner": "pea",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
    "email_on_failure": False,
}

dag = DAG(
    dag_id="data_extraction",
    default_args=default_args,
    description="Refresh every raw data source (one task per fetcher) before the nightly cube build.",
    schedule="0 1 * * *",                        # 01:00 daily
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_tasks=8,
    tags=["pea", "extraction"],
)


def fetch(cmd: str, pool: str = "default_pool", task_id: str | None = None) -> BashOperator:
    """A BashOperator that runs one extraction command from the pipeline venv."""
    return BashOperator(
        task_id=task_id or cmd.replace("-", "_"),
        bash_command=f"{PIPE} {cmd} -c {CONFIGS}",
        cwd=PROJECT,
        pool=pool,
        dag=dag,
    )


# 0) universe seed — everything downstream resolves the universe from sp500_tickers
seed_universe = fetch("seed-universe")

# 1) LIGHT / fast — fan out in the default pool
light = [
    fetch("market-prices"),
    fetch("macro"),
    fetch("macro-assets"),
    fetch("short-interest"),
    fetch("earnings-surprises"),
]
# the one heavy yfinance pull (own host -> default pool, not the SEC pools)
price_history = fetch("price-history")

# 2) SEC bulk zips — capped to 2 concurrent (disk + SEC bandwidth)
fails_to_deliver = fetch("fails-to-deliver", pool="sec_bulk")
thirteen_f = fetch("thirteen-f", pool="sec_bulk")
financial_statements = fetch("financial-statements", pool="sec_bulk")
insider_transactions = fetch("insider-transactions", pool="sec_bulk")
financial_notes = fetch("financial-notes", pool="sec_bulk")           # VERY heavy
superinvestors = fetch("superinvestors")                              # light, needs 13F

# 3) per-ticker EDGAR API — capped to 2 (shared SEC 10 req/s)
fundamentals = fetch("fundamentals", pool="sec_api")
employees = fetch("employees", pool="sec_api")
def14a = fetch("def14a", pool="sec_api")                              # + LLM

# 4) external scraping — capped to 2 (site rate limits)
wiki_pageviews = fetch("wiki-pageviews", pool="scrape")
google_trends = fetch("google-trends", pool="scrape")                # slow
# earnings calls split in two: DOWNLOAD to disk (HF 1.8GB one-time + MF HTML) -> INGEST to DB
download_earnings_calls = fetch("download-earnings-calls", pool="scrape")
ingest_earnings_calls = fetch("ingest-earnings-calls", pool="scrape")
download_earnings_calls >> ingest_earnings_calls

extraction_complete = EmptyOperator(task_id="extraction_complete", dag=dag)


def _freshness_check(**context) -> None:
    """Data-drift / gap gate. Shells to the pipeline venv (`check-freshness`), captures the JSON
    report, pushes it to XCom (the whole report under `freshness` + the latest date per source under
    `latest_<source>`), and RAISES when anything is not up to date so the task goes RED. XCom is
    pushed BEFORE raising, so the per-source latest dates are visible even on a red run."""
    proc = subprocess.run(
        [PIPE_PY, "-m", "src", "data_extract", "check-freshness", "-c", CONFIGS],
        cwd=PROJECT, capture_output=True, text=True)
    report = None
    for line in reversed((proc.stdout or "").strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                report = json.loads(line)
                break
            except json.JSONDecodeError:
                continue
    ti = context["ti"]
    if report is None:                            # the check itself failed to produce a report
        ti.xcom_push(key="freshness", value={
            "ok": False, "error": "no report parsed", "returncode": proc.returncode,
            "stdout_tail": (proc.stdout or "")[-800:], "stderr_tail": (proc.stderr or "")[-800:]})
        raise AirflowFailException(f"freshness check produced no report (rc={proc.returncode})")
    ti.xcom_push(key="freshness", value=report)
    for label, info in report.get("sources", {}).items():
        ti.xcom_push(key=f"latest_{label}", value=info.get("latest"))
    # which tickers got a new fundamentals filing (new earnings) since the last run
    ti.xcom_push(key="new_fundamentals", value=report.get("new_fundamentals"))
    if not report.get("ok", False):
        raise AirflowFailException(
            "Data NOT up to date (RED) — stale/gapped sources: "
            + ", ".join(report.get("stale", [])))


# RED when any source is not up to date; XCom carries the latest date per source either way.
freshness_check = PythonOperator(
    task_id="check_data_freshness", python_callable=_freshness_check, dag=dag)

# aggregation still runs even if the freshness gate is RED (it is a visible WARNING, not a hard
# block); flip this to TriggerRule.ALL_SUCCESS to make stale data hard-stop the prediction build.
trigger_aggregation = TriggerDagRunOperator(
    task_id="trigger_data_aggregation",
    trigger_dag_id="data_aggregation",
    wait_for_completion=False,
    reset_dag_run=True,
    trigger_rule=TriggerRule.ALL_DONE,
    dag=dag,
)

# --- wiring ---
all_fetchers = light + [price_history, fails_to_deliver, thirteen_f, financial_statements,
                        insider_transactions, financial_notes, fundamentals, employees, def14a,
                        wiki_pageviews, google_trends, download_earnings_calls]
seed_universe >> all_fetchers
thirteen_f >> superinvestors                                         # roster reads the 13F holdings
download_earnings_calls >> ingest_earnings_calls                     # ingest parses the downloaded files
# all sources refreshed -> freshness/gap gate (XCom + RED) -> trigger aggregation
(all_fetchers + [superinvestors, ingest_earnings_calls]) >> extraction_complete
extraction_complete >> freshness_check >> trigger_aggregation
