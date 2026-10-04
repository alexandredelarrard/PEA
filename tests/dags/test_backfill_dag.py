"""Static contract for the backfill DAG (Airflow is not installed in the test venv, so the file is parsed)."""

from __future__ import annotations

import ast
from pathlib import Path

DAGS = Path(__file__).resolve().parents[2] / "src" / "dags"
BACKFILL_DAG = DAGS / "dag_data_backfill.py"


def _calls(tree: ast.AST, name: str) -> list[ast.Call]:
    return [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == name]


def _kwargs(call: ast.Call) -> dict[str, object]:
    return {k.arg: ast.literal_eval(k.value) for k in call.keywords if k.arg and isinstance(k.value, ast.Constant)}


def test_backfill_dag_is_one_noon_task_outside_the_extraction_chain():
    source = BACKFILL_DAG.read_text(encoding="utf-8")
    tree = ast.parse(source)
    dag = _kwargs(_calls(tree, "DAG")[0])
    tasks = _calls(tree, "BashOperator")
    task = _kwargs(tasks[0])

    assert dag["dag_id"] == "data_backfill" and dag["schedule"] == "0 12 * * *"
    assert len(tasks) == 1 and task["task_id"] == "thirteen_f_backfill" and task["pool"] == "sec_bulk"
    assert "thirteen-f-backfill" in source
    assert not _calls(tree, "TriggerDagRunOperator") and "from src" not in source and "import src" not in source
    for other in sorted(DAGS.glob("dag_*.py")):
        if other != BACKFILL_DAG:
            text = other.read_text(encoding="utf-8")
            assert "thirteen-f-backfill" not in text and "data_backfill" not in text, other.name

    print("\n=== SANITY CHECK: backfill DAG ===")
    print("  data_backfill: one thirteen_f_backfill task in the sec_bulk pool at 12:00, triggering nothing;")
    print("  no other DAG names it, so it is not wired into the 01:00 extraction chain. Validated.")
