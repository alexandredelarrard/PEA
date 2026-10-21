"""Existing weekly training order and command contracts, without executing Airflow jobs."""

import shlex
from pathlib import Path
from types import SimpleNamespace

from jinja2 import Environment, StrictUndefined

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
        command_template = dag.tasks[name].kwargs["bash_command"]
        assert "--resume" not in command_template
        if name == "train_model":
            assert "--run-id" not in command_template and dag.tasks[name].kwargs["do_xcom_push"]
        else:
            task_instance = SimpleNamespace(xcom_pull=lambda task_ids: "generated_first" if task_ids == "train_model" else None)
            rendered = Environment(undefined=StrictUndefined).from_string(command_template).render(ti=task_instance)
            argv = shlex.split(rendered)
            assert argv[-2:] == ["--run-id", "generated_first"]
            assert not dag.tasks[name].kwargs["do_xcom_push"]
        assert "dag_run.run_id" not in dag.tasks[name].kwargs["bash_command"]
    states = replay(dag, failed=["backtest_modelling"])
    assert states["backtest_portfolio"] == states["full_train"] == "upstream_failed"
    assert dag.kwargs["max_active_runs"] == 1
    assert dag.kwargs["max_active_tasks"] == 1 and dag.kwargs["default_args"]["retries"] == 0
    print("SANITY: all four tasks share a safe dated run ID, remain sequential and block portfolio/full refit if model assessment fails.")


def test_rendered_handoff_pins_successful_train_across_downstream_retry() -> None:
    dag = load_dag(Path(__file__).resolve().parents[2] / "src/dags/dag_modelling.py")
    xcom = {"train_model": "generated_first"}
    task_instance = SimpleNamespace(xcom_pull=lambda task_ids: xcom[task_ids])
    renderer = Environment(undefined=StrictUndefined)
    template = dag.tasks["backtest_modelling"].kwargs["bash_command"]
    first = renderer.from_string(template).render(ti=task_instance)
    concurrent_run = {"train_model": "unrelated_newer"}
    assert concurrent_run != xcom
    assert renderer.from_string(template).render(ti=task_instance) == first
    xcom["train_model"] = "generated_after_train_retry"
    assert shlex.split(renderer.from_string(template).render(ti=task_instance))[-1] == "generated_after_train_retry"
    failed = replay(dag, failed=["train_model"])
    assert all(failed[name] == "upstream_failed" for name in ["backtest_modelling", "backtest_portfolio", "full_train"])
    print(
        "SANITY: rendered downstream retries retain this task instance's successful training ID; a retried train changes it, and failure blocks every successor."
    )
