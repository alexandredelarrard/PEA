"""Removing the obsolete ledger leaves the daily forecasting contract intact."""

from pathlib import Path

from tests.dags.dag_harness import load_dag


def test_daily_prediction_survives_ledger_removal() -> None:
    dag = load_dag(Path("src/dags/dag_strat_prediction.py"))
    assert dag.kwargs["dag_id"] == "strat_prediction"
    assert dag.kwargs["schedule"] == "0 6 * * *"
    assert set(dag.tasks) == {"predict"}
    assert "-m src modelling predict" in dag.tasks["predict"].kwargs["bash_command"]
    print("SANITY: daily model prediction and schedule remain; no removed portfolio command is invoked.")
