"""
dag_modelling.py  (src/dags/dag_modelling.py)
---------------------------------------------
MODELLING & BACKTEST DAG — the WEEKLY (weekend) retrain. Fit the cross-sectional long/short
ensemble on the holdout window, backtest the portfolio over that out-of-sample period, then refit
on ALL history so the production model has learnt from every available observation.

    train_model ──▶ backtest_portfolio ──▶ full_train

  * train_model       — `python -m src modelling train` : trains one per-horizon ensemble
                        (elasticnet + LightGBM `lgbm` + random_forest) on the cube for the
                        modellling.yml `train.start_date`/`train.end_date` window, saves the model
                        artifacts + metadata.json (the backtest reads them back), the
                        `predictions` / `cube_signal` tables and the per-run diagnostics
                        (per horizon: SHAP values, PDPs and kpis.json per booster member).
  * backtest_portfolio — `python -m src portfolio backtest` : blends the configured sleeves
                        (portfolio.yml; the L/S sleeve is OOS from the model's train_end),
                        risk-parity/ERC-weights them to the global vol target, reports
                        per-strategy vs global Sharpe and saves the backtest pictures
                        (equity vs SP500, dynamic sleeve weights, per-sleeve analysis) + tables
                        under data/output/portfolio/.
  * full_train        — `python -m src modelling full-train` : PRODUCTION refit on ALL history up
                        to the latest cube date (NO train_end cutoff, NO holdout), so nothing is
                        withheld from the model that actually trades. This is the artifact the
                        daily prediction reads back.

The holdout `train_model` + `backtest_portfolio` deliberately run BEFORE the full refit: the
per-horizon IR blend weights and the backtest are only meaningful measured out-of-sample. Fitting
on everything first would leave both in-sample, inflating the reported Sharpe and mis-weighting
the horizon blend that production then uses.

WEEKLY, Saturday 02:00 — a retrain is a weekly concern and the weekend has the machine to itself.
PREDICTION IS NOT IN THIS DAG: the daily `strat_prediction` DAG owns it (and is what the nightly
`data_aggregation` DAG triggers), so `predictions_latest` has exactly one writer and a fresh cube
is scored every day without waiting for a retrain. Each command runs in the pipeline's isolated
venv (/opt/pipeline) from the mounted repo.
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
    description="WEEKLY (Sat): train the long/short ensemble on the holdout window, backtest the "
                "portfolio OOS, then refit on ALL history for production.",
    schedule="0 2 * * 6",                        # Saturday 02:00 — weekend retrain
    start_date=datetime(2024, 1, 1),
    catchup=False,                               # a missed week is covered by the next run
    max_active_runs=1,                           # never two trains writing the same artifacts
    max_active_tasks=1,                          # train then backtest, strictly sequential
    tags=["pea", "modelling", "backtest", "weekly"],
)


def run(pkg_cmd: str, task_id: str) -> BashOperator:
    """A BashOperator running one pipeline command from the isolated venv."""
    return BashOperator(
        task_id=task_id,
        bash_command=f"/opt/pipeline/bin/python -m src {pkg_cmd} -c {CONFIGS}",
        cwd=PROJECT,
        dag=dag,
    )


# 1) train the ensemble on the configured HOLDOUT window -> models + metadata.json + diagnostics
train_model = run("modelling train", task_id="train_model")

# 2) backtest the sleeve blend over that out-of-sample window (reads the model artifacts)
backtest_portfolio = run("portfolio backtest", task_id="backtest_portfolio")

# 3) PRODUCTION refit on ALL history up to the latest cube date — no holdout, nothing withheld.
#    The daily `strat_prediction` DAG scores with these artifacts.
full_train = run("modelling full-train", task_id="full_train")

train_model >> backtest_portfolio >> full_train
