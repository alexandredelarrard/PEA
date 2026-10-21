# pyright: reportAttributeAccessIssue=false, reportMissingImports=false, reportUnusedExpression=false
"""Weekly original-fold/cutoff training, saved-model assessment, portfolio backtest and final refit.

The daily prediction DAG reads only complete production generations. Assessment failure blocks
portfolio and full-history fitting; every task uses the same filesystem-safe dated run identity.
"""

from datetime import datetime, timedelta

from airflow.operators.bash import BashOperator

from airflow import DAG

PROJECT = "/opt/airflow/project"
CONFIGS = f"{PROJECT}/configs"

default_args = {
    "owner": "pea",
    "depends_on_past": False,
    "retries": 0,
    "retry_delay": timedelta(minutes=10),
    "email_on_failure": False,
}

dag = DAG(
    dag_id="modelling",
    default_args=default_args,
    description="Weekly: five-fold/cutoff training, saved-model diagnostics, portfolio backtest, then full-history production refit.",
    schedule="0 2 * * 6",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    max_active_tasks=1,
    tags=["pea", "modelling", "backtest", "weekly"],
)


def run(pkg_cmd: str, task_id: str) -> BashOperator:
    """Capture the fresh training ID and pin every downstream task to that successful result."""
    selected_run = "" if task_id == "train_model" else " --run-id '{{ ti.xcom_pull(task_ids=\"train_model\") }}'"
    return BashOperator(
        task_id=task_id,
        bash_command=f"/opt/pipeline/bin/python -m src {pkg_cmd} -c {CONFIGS}{selected_run}",
        do_xcom_push=task_id == "train_model",
        cwd=PROJECT,
        dag=dag,
    )


train_model = run("modelling train", task_id="train_model")
backtest_modelling = run("modelling assess", task_id="backtest_modelling")
backtest_portfolio = run("portfolio backtest", task_id="backtest_portfolio")
full_train = run("modelling full-train", task_id="full_train")

train_model >> backtest_modelling >> backtest_portfolio >> full_train
