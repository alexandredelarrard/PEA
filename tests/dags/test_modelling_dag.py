"""Existing weekly training order and command contracts, without executing Airflow jobs."""

from pathlib import Path

from tests.dags.dag_harness import load_dag


def test_modelling_dag_keeps_cutoff_backtest_then_production_refit() -> None:
    dag = load_dag(Path(__file__).resolve().parents[2] / "src/dags/dag_modelling.py")
    assert list(dag.tasks) == ["train_model", "backtest_portfolio", "full_train"]
    assert dag.tasks["backtest_portfolio"].upstream == {"train_model"}
    assert dag.tasks["full_train"].upstream == {"backtest_portfolio"}
    for name, command in {"train_model": "modelling train", "backtest_portfolio": "portfolio backtest", "full_train": "modelling full-train"}.items():
        assert f"-m src {command} -c " in dag.tasks[name].kwargs["bash_command"]
    assert dag.kwargs["max_active_runs"] == 1
    print("SANITY: existing three tasks remain sequential; full-train reuses the train selection and cannot precede evaluation.")
