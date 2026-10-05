"""Load a DAG file against stub Airflow modules and replay one night's task states.

Airflow is not installed in the test venv, so the DAG module runs against minimal stand-ins
that record each task's trigger rule and `>>` / `chain` edges. `replay` applies Airflow's
ALL_SUCCESS / ALL_DONE semantics to a night where the given tasks fail.
"""

from __future__ import annotations

import runpy
import sys
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any

ALL_SUCCESS = "all_success"
ALL_DONE = "all_done"


class StubDag:
    """A DAG: its keyword arguments and its tasks by id, in definition order."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.tasks: dict[str, StubTask] = {}


class StubTask:
    """Any operator: its id, trigger rule, keyword arguments and upstream task ids."""

    def __init__(self, task_id: str, dag: StubDag, trigger_rule: str = ALL_SUCCESS, **kwargs: Any) -> None:
        self.task_id = task_id
        self.trigger_rule = str(trigger_rule)
        self.kwargs = kwargs
        self.upstream: set[str] = set()
        dag.tasks[task_id] = self

    def __rshift__(self, other: StubTask | list[StubTask]) -> StubTask | list[StubTask]:
        for task in other if isinstance(other, list) else [other]:
            task.upstream.add(self.task_id)
        return other

    def __rrshift__(self, other: list[StubTask]) -> StubTask:
        for task in other:
            self.upstream.add(task.task_id)
        return self


class _TriggerRule:
    ALL_SUCCESS = ALL_SUCCESS
    ALL_DONE = ALL_DONE


def _chain(*tasks: StubTask) -> None:
    for upstream, downstream in zip(tasks, tasks[1:], strict=False):
        upstream.__rshift__(downstream)


def _module(name: str, **attrs: Any) -> ModuleType:
    module = ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


@contextmanager
def _stub_airflow() -> Iterator[None]:
    stubs = {
        "airflow": _module("airflow", DAG=StubDag),
        "airflow.operators": _module("airflow.operators"),
        "airflow.operators.bash": _module("airflow.operators.bash", BashOperator=StubTask),
        "airflow.operators.python": _module("airflow.operators.python", PythonOperator=StubTask),
        "airflow.operators.trigger_dagrun": _module("airflow.operators.trigger_dagrun", TriggerDagRunOperator=StubTask),
        "airflow.utils": _module("airflow.utils"),
        "airflow.utils.trigger_rule": _module("airflow.utils.trigger_rule", TriggerRule=_TriggerRule),
        "airflow.models": _module("airflow.models"),
        "airflow.models.baseoperator": _module("airflow.models.baseoperator", chain=_chain),
        "airflow.exceptions": _module("airflow.exceptions", AirflowFailException=RuntimeError),
    }
    saved = {name: sys.modules.get(name) for name in stubs}
    sys.modules.update(stubs)
    try:
        yield
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def load_dag(path: Path) -> StubDag:
    """Run the DAG file against the stubs and return its DAG."""
    with _stub_airflow():
        namespace = runpy.run_path(str(path))
    return next(value for value in namespace.values() if isinstance(value, StubDag))


def replay(dag: StubDag, failed: Iterable[str]) -> dict[str, str]:
    """Each task's final state on a night where the `failed` tasks fail when they run."""
    failing = set(failed)
    states: dict[str, str] = {}
    pending = dict(dag.tasks)
    while pending:
        ready = [task for task in pending.values() if task.upstream <= states.keys()]
        assert ready, f"cycle among {sorted(pending)}"
        for task in ready:
            upstream_bad = any(states[name] in ("failed", "upstream_failed") for name in task.upstream)
            if task.trigger_rule == ALL_SUCCESS and upstream_bad:
                states[task.task_id] = "upstream_failed"
            else:
                states[task.task_id] = "failed" if task.task_id in failing else "success"
            del pending[task.task_id]
    return states
