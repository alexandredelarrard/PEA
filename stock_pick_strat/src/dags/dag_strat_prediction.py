"""
dag_strat_prediction.py  (src/dags/dag_strat_prediction.py)
-----------------------------------------------------------
STRATEGY PREDICTION DAG — the DAILY production run: score the freshest cube with the
already-trained model, then turn the resulting book into the concrete trades to place.

    predict ──▶ strategy_moves

  * predict         — `python -m src modelling predict`: loads the full-trained ensemble
                      artifacts from disk (NO training here) and scores the latest cube date for
                      every horizon, writing `predictions_latest` in LONG form — one row per
                      (as-of date, ticker, horizon, model) with `predicted_at` (when this run
                      produced it) and `predicts_for` (the date that row is about, = as-of +
                      horizon business days). `model` covers each ensemble member, plus
                      'ensemble' per horizon and 'blended' across horizons.
  * strategy_moves  — `python -m src portfolio strategy-moves`: runs the configured sleeves,
                      blends them (ERC/risk-parity + one global vol target) and re-sizes each
                      sleeve's book to its dynamic ERC allocation x leverage, so the notional is
                      the real dollars the portfolio would deploy. Every (day, sleeve, ticker)
                      move is FIFO-matched into round trips and upserted to the `strategy` table
                      with entry price, exit price and realized P&L.

Runs DAILY and is triggered by the nightly `data_aggregation` DAG once the cube is fresh — it
replaced `modelling` as that DAG's downstream, because prediction is a daily concern while
(re)training is a weekly one (see dag_modelling.py, now weekend-only). Model artifacts therefore
come from the last weekly `modelling` run; this DAG never trains.

`strategy_moves` recomputes the WHOLE ledger each day rather than appending today's rows: a BUY
placed weeks ago only learns its exit price and P&L on the day it closes, so past rows must be
rewritten. That also makes the DAG self-healing — a missed day, or a model retrained over the
weekend, corrects itself on the next run.
"""
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

PROJECT = "/opt/airflow/project"                 # the repo, bind-mounted
CONFIGS = f"{PROJECT}/configs"

default_args = {
    "owner": "pea",
    "depends_on_past": False,
    # prediction is cheap and idempotent (both tasks fully rewrite what they own), so a
    # transient DB / disk hiccup is worth retrying — unlike the training DAG.
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}

dag = DAG(
    dag_id="strat_prediction",
    default_args=default_args,
    description="DAILY: predict the latest cube date (long-format predictions_latest) then write "
                "the tradeable strategy ledger (`strategy`).",
    schedule="0 6 * * *",                        # every day at 06:00, after the nightly aggregation
    start_date=datetime(2024, 1, 1),
    catchup=False,                               # only ever the latest cube date matters
    max_active_runs=1,                           # the ledger is a full rewrite — never concurrently
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

# 2) the trades to actually place, with per-position P&L -> `strategy`
strategy_moves = run("portfolio strategy-moves", task_id="strategy_moves")

predict >> strategy_moves
