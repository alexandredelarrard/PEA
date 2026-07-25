"""
dag_modelling.py  (src/dags/dag_modelling.py)
---------------------------------------------
MODELLING & BACKTEST DAG — train the cross-sectional long/short ensemble for the configured window,
backtest it, THEN retrain on all history and emit live predictions for the allocation.

    train_model ──▶ backtest_portfolio ──▶ full_train ──▶ predict

  * full_train  — `python -m src modelling full-train`: PRODUCTION retrain on ALL history up to the
                  latest cube date (no train_end cutoff, no OOS holdout).
  * predict     — `python -m src modelling predict`: score the latest cube date per horizon
                  (pred_h<h>) + the blended signal -> `predictions_latest`, the table the future
                  allocation DAG consumes.

  * train_model       — `python -m src modelling train` : trains one per-horizon ensemble
                        (elasticnet + LightGBM + random_forest) on the cube for the
                        modellling.yml `train.start_date`/`train.end_date` window, saves the model
                        artifacts + metadata.json (the backtest reads them back), the
                        `predictions` / `cube_signal` tables and the per-run diagnostics pictures.
  * backtest_portfolio — `python -m src portfolio backtest` : blends the configured sleeves
                        (portfolio.yml; the L/S sleeve is OOS from the model's train_end),
                        risk-parity/ERC-weights them to the global vol target, reports
                        per-strategy vs global Sharpe and saves the backtest pictures
                        (equity vs SP500, dynamic sleeve weights, per-sleeve analysis) + tables
                        under data/output/portfolio/ (the ./data bind mount -> host disk).

schedule=None: trigger manually, or chain after the nightly aggregation once the cube has been
rebuilt. Each command runs in the pipeline's isolated venv (/opt/pipeline) from the mounted repo.
"""
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

PROJECT = "/opt/airflow/project"                 # the repo, bind-mounted
CONFIGS = f"{PROJECT}/configs"

default_args = {
    "owner": "pea",
    "depends_on_past": False,
    "retries": 0,                                # long GPU/CPU train — don't silently re-run
    "retry_delay": timedelta(minutes=10),
    "email_on_failure": False,
}

dag = DAG(
    dag_id="modelling",
    default_args=default_args,
    description="Train the long/short ensemble (config train window) then run the portfolio backtest.",
    schedule=None,                               # triggered manually or after the cube rebuild
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_tasks=1,                          # train then backtest, strictly sequential
    tags=["pea", "modelling", "backtest"],
)


def run(pkg_cmd: str, task_id: str) -> BashOperator:
    """A BashOperator running one pipeline command from the isolated venv."""
    return BashOperator(
        task_id=task_id,
        bash_command=f"/opt/pipeline/bin/python -m src {pkg_cmd} -c {CONFIGS}",
        cwd=PROJECT,
        dag=dag,
    )


# 1) train the ensemble on the configured train window -> saves models + metadata.json
train_model = run("modelling train", task_id="train_model")

# 2) once trained, run the portfolio backtest (reads the model artifacts) -> saves pictures
backtest_portfolio = run("portfolio backtest", task_id="backtest_portfolio")

# 3) PRODUCTION retrain on ALL history up to the latest cube date (no OOS holdout)
full_train = run("modelling full-train", task_id="full_train")

# 4) predict the latest cube date per horizon + blended -> predictions_latest (for the allocation DAG)
predict = run("modelling predict", task_id="predict")

train_model >> backtest_portfolio >> full_train >> predict
