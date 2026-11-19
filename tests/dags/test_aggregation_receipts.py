"""The receipts the aggregation DAG hands to `assemble-cube`: only successful upstream results, each failure named."""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pandas as pd
import pytest
from jinja2 import Environment

from src.context import Context
from src.data_aggregate.utils.common.assembly import read_receipts, write_receipt
from src.data_store.schema import Table, Tables
from src.utils.window import Window, WindowPlan, WindowRun
from tests.dags.dag_harness import load_dag
from tests.support.config import fixture_config
from tests.support.universe import universe_of

DAG_FILE = Path(__file__).resolve().parents[2] / "src" / "dags" / "dag_data_aggregation.py"
INTERVAL = Window(pd.Timestamp("2026-11-03"), pd.Timestamp("2026-11-10"))
PARTS = {"build_prices": Tables.cube_part_prices, "build_momentum": Tables.cube_part_momentum}


def _context(store: Any) -> Context:
    store.save(Tables.cube_part_prices, pd.DataFrame({"ticker": "AAPL", "date": pd.bdate_range(INTERVAL.start, INTERVAL.end), "close": 100.0}))
    return cast(
        Context,
        SimpleNamespace(
            store=store,
            config=fixture_config(),
            universe=universe_of(["AAPL"]),
            run=WindowRun("incremental", "DAG receipt hand-off", pd.Timestamp("2026-11-10T15:00:00Z")),
        ),
    )


def _receipt(context: Context, table: Table) -> str:
    plan = WindowPlan(INTERVAL, INTERVAL, INTERVAL, context.run, None, (("ticker", ("AAPL",)),))
    return write_receipt(context, {table: plan})


def _rendered_argument(xcom: dict[str, str]) -> str:
    """The `CUBE_COMPONENT_WINDOWS` value Airflow 2.10 builds: `xcom_pull` of a task list is ordered by the list, None when a task pushed nothing."""
    template = load_dag(DAG_FILE).tasks["assemble_cube"].kwargs["env"]["CUBE_COMPONENT_WINDOWS"]
    pull = lambda task_ids: [xcom.get(task_id) for task_id in task_ids]  # noqa: E731
    return Environment().from_string(template).render(ti=SimpleNamespace(xcom_pull=pull))


def test_cleared_run_keeps_the_cutoff_its_receipts_were_written_with() -> None:
    from datetime import datetime

    tasks = load_dag(DAG_FILE).tasks
    template = re.search(r'--as-of "([^"]*)"', tasks["assemble_cube"].kwargs["bash_command"]).group(1)
    logical = datetime.fromisoformat("2026-11-19T16:12:00.123456+00:00")

    def cutoff(start_date: datetime) -> str:
        run = SimpleNamespace(conf={}, start_date=start_date, logical_date=logical)
        return Environment().from_string(template).render(dag_run=run, macros=SimpleNamespace(datetime=datetime))

    first, after_clear = cutoff(logical), cutoff(datetime.fromisoformat("2026-11-19T21:30:00+00:00"))

    assert first == after_clear == logical.isoformat()
    parts = [tasks[name.replace("-", "_")] for name in ("build-prices", "build-target", "build-governance")]
    assert all(re.search(r'--as-of "([^"]*)"', task.kwargs["bash_command"]).group(1) == template for task in parts)
    print("SANITY: a re-run assemble_cube (new DagRun.start_date, 5 hours later) renders the same cutoff as the part tasks that wrote the receipts.")


def test_a_part_task_that_pushed_nothing_is_named_in_the_refusal(sqlite_store: Any) -> None:
    context = _context(sqlite_store)
    xcom = {task: _receipt(context, table) for task, table in PARTS.items()}

    argument = _rendered_argument(xcom)

    assert len(json.loads(argument)) == 8 and json.loads(argument).count(None) == 6
    with pytest.raises(ValueError, match=r"root run or mode: receipt 1 is not a receipt object: None"):
        read_receipts(context, argument)
    print("SANITY: the 6 part tasks that failed or pushed nothing arrive as null and the refusal names the first one by position.")


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [("as_of", "as_of"), ("instant", "instant"), ("mode", "mode full != incremental"), ("fields", "fields")],
)
def test_root_run_refusal_names_the_receipt_and_the_field(sqlite_store: Any, mutation: str, expected: str) -> None:
    context = _context(sqlite_store)
    good = json.loads(_receipt(context, Tables.cube_part_prices))
    bad = json.loads(_receipt(context, Tables.cube_part_momentum))
    if mutation == "as_of":
        bad["as_of"] = "2026-11-09"
    elif mutation == "instant":
        bad["instant"] = "2026-11-10T22:00:00+00:00"
    elif mutation == "mode":
        bad["mode"] = "full"
    else:
        bad.pop("configuration")

    with pytest.raises(ValueError, match=rf"root run or mode: receipt 1 .*{expected}"):
        read_receipts(context, json.dumps([good, bad]))
    print(f"SANITY: a receipt with a wrong {mutation} is refused by position and field instead of one catch-all message.")
