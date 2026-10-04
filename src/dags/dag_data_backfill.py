# pyright: reportAttributeAccessIssue=false, reportMissingImports=false, reportUnusedExpression=false
"""
dag_data_backfill.py  (src/dags/dag_data_backfill.py)
-----------------------------------------------------
Daily BACKFILL DAG, outside the 01:00 extraction chain: one task that fills `sec13f_hr` history for
tickers newly added to the universe from the SEC Form 13F data sets (cached ZIPs, downloaded on first
use). With no new ticker the task does nothing. It runs in the `sec_bulk` pool, so it never overlaps
more than one other bulk download, and it triggers nothing downstream.

The command is `/opt/pipeline/bin/python -m src data_extract thirteen-f-backfill` (the pipeline's isolated venv).
"""

from datetime import datetime, timedelta

from airflow.operators.bash import BashOperator

from airflow import DAG

PROJECT = "/opt/airflow/project"  # the repo, bind-mounted
CONFIGS = f"{PROJECT}/configs"
PIPE_PY = "/opt/pipeline/bin/python"  # pipeline's isolated venv interpreter
PIPE = f"{PIPE_PY} -m src data_extract"

default_args = {
    "owner": "pea",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=30),
    "email_on_failure": False,
}

dag = DAG(
    dag_id="data_backfill",
    default_args=default_args,
    description="13F history for tickers new to the universe, from the cached SEC 13F data sets.",
    schedule="0 12 * * *",  # 12:00 daily, away from the 01:00 extraction chain
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["pea", "extraction", "backfill"],
)

thirteen_f_backfill = BashOperator(
    task_id="thirteen_f_backfill",
    bash_command=f"{PIPE} thirteen-f-backfill -c {CONFIGS}",
    cwd=PROJECT,
    pool="sec_bulk",
    dag=dag,
)
