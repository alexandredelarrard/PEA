"""Existing weekly training order and command contracts, without executing Airflow jobs."""

from pathlib import Path

from tests.dags.dag_harness import load_dag, replay


def test_modelling_dag_keeps_cutoff_backtest_then_production_refit() -> None:
    dag = load_dag(Path(__file__).resolve().parents[2] / "src/dags/dag_modelling.py")
    assert list(dag.tasks) == ["train_model", "backtest_modelling", "backtest_portfolio", "full_train"]
    assert dag.tasks["backtest_modelling"].upstream == {"train_model"}
    assert dag.tasks["backtest_portfolio"].upstream == {"backtest_modelling"}
    assert dag.tasks["full_train"].upstream == {"backtest_portfolio"}
    for name, command in {
        "train_model": "modelling train",
        "backtest_modelling": "modelling assess",
        "backtest_portfolio": "portfolio backtest",
        "full_train": "modelling full-train",
    }.items():
        assert f"-m src {command} -c " in dag.tasks[name].kwargs["bash_command"]
        assert "--run-id '{{ ts_nodash }}_{{ dag_run.id }}'" in dag.tasks[name].kwargs["bash_command"]
        assert "dag_run.run_id" not in dag.tasks[name].kwargs["bash_command"]
    states = replay(dag, failed=["backtest_modelling"])
    assert states["backtest_portfolio"] == states["full_train"] == "upstream_failed"
    assert dag.kwargs["max_active_runs"] == 1
    assert dag.kwargs["max_active_tasks"] == 1 and dag.kwargs["default_args"]["retries"] == 0
    print("SANITY: all four tasks share a safe dated run ID, remain sequential and block portfolio/full refit if model assessment fails.")
