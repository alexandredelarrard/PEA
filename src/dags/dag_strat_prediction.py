# pyright: reportAttributeAccessIssue=false, reportMissingImports=false, reportUnusedExpression=false
"""Daily full-history model prediction; triggered after the nightly cube refresh."""

from datetime import datetime, timedelta

from airflow.operators.bash import BashOperator

from airflow import DAG

PROJECT = "/opt/airflow/project"  # the repo, bind-mounted
CONFIGS = f"{PROJECT}/configs"

default_args = {
    "owner": "pea",
    "depends_on_past": False,
    # prediction is cheap and idempotent, so a
    # transient DB / disk hiccup is worth retrying — unlike the training DAG.
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}

dag = DAG(
    dag_id="strat_prediction",
    default_args=default_args,
    description="DAILY: predict the latest cube date (long-format predictions_latest).",
    schedule="0 6 * * *",  # every day at 06:00, after the nightly aggregation
    start_date=datetime(2024, 1, 1),
    catchup=False,  # only ever the latest cube date matters
    max_active_runs=1,  # keep prediction publications serial
    max_active_tasks=1,
    tags=["pea", "prediction", "strategy", "daily"],
)


def run(pkg_cmd: str, task_id: str) -> BashOperator:
    """A BashOperator running one pipeline command from the isolated venv."""
    return BashOperator(
        task_id=task_id,
        bash_command=f"/opt/pipeline/bin/python -m src {pkg_cmd} -c {CONFIGS}",
        cwd=PROJECT,
        dag=dag,
    )


# 1) score the latest cube date with the weekly-trained ensemble -> predictions_latest (long)
predict = run("modelling predict", task_id="predict")
