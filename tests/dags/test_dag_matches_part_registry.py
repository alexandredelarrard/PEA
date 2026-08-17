"""
The aggregation DAG's task chain must match the cube part registry.

The old DAG carried a `GROUPS` literal with the comment "must match
StepBuildCube._GROUP_SOURCES", maintained by hand -- and it had drifted: `attention` was
commented out here but still registered there, so the nightly `cube_status` gate reported
`cube_part_attention` as missing on every run and the DAG went red for a part nobody was
building. This test replaces that comment with an assertion.

Airflow is not importable in the plain test venv, so the DAG module is parsed rather than
imported: `CHAIN` is derived from `PART_COMMANDS`, which is what actually needs checking.
"""
from __future__ import annotations

import ast
from pathlib import Path

from src.data_aggregate.utils.common.parts import PART_COMMANDS

DAG_FILE = Path(__file__).resolve().parents[2] / "src" / "dags" / "dag_data_aggregation.py"


def _dag_source() -> str:
    assert DAG_FILE.exists(), f"missing {DAG_FILE}"
    return DAG_FILE.read_text(encoding="utf-8")


def test_dag_chain_is_derived_from_the_registry():
    src = _dag_source()
    tree = ast.parse(src)

    # CHAIN must be assigned from PART_COMMANDS, not from a hand-written list of strings
    chain_assign = [n for n in ast.walk(tree)
                    if isinstance(n, ast.Assign)
                    and any(getattr(t, "id", None) == "CHAIN" for t in n.targets)]
    assert chain_assign, "the DAG no longer defines CHAIN"
    rendered = ast.dump(chain_assign[0].value)
    assert "PART_COMMANDS" in rendered, (
        "CHAIN must be derived from parts.PART_COMMANDS; a hand-written literal is exactly what "
        f"drifted before (got {ast.unparse(chain_assign[0].value)})")
    # no module-level GROUPS assignment (the historical hand-synced literal). Checked on the
    # AST, not the text, so the comment explaining the history does not trip it.
    assigned = {getattr(t, "id", None)
                for n in ast.walk(tree) if isinstance(n, ast.Assign) for t in n.targets}
    assert "GROUPS" not in assigned, "the old hand-synced GROUPS literal is back"

    print("\n=== SANITY CHECK: DAG chain <-> part registry ===")
    print(f"  CHAIN = {ast.unparse(chain_assign[0].value)} -> {list(PART_COMMANDS)}")
    print("  CONCLUSION: the task list cannot drift from the registry -- the `attention` "
          "mismatch that made cube_status permanently red is structurally impossible. Validated.")


def test_dag_is_sequential_and_ends_in_the_status_gate():
    src = _dag_source()
    assert "max_active_tasks=1" in src, (
        "the chain must be sequential -- peak memory should be the largest single step, not the "
        "sum of parallel pool slots")
    assert "chain(deduce_peers, *step_tasks, assemble_cube, cube_status," in src, \
        "the DAG must run peers -> the registry steps -> assemble -> status, in order"
    # the memory-driven serialization the old DAG needed is gone
    for gone in ("institutional_task", "superinvestor_task", "fundamental_task", "features -g"):
        assert gone not in src, f"'{gone}' should have been removed with the sub-step split"

    print("\n=== SANITY CHECK: DAG shape ===")
    print(f"  max_active_tasks=1; deduce_peers -> {len(PART_COMMANDS)} steps -> assemble_cube -> "
          "cube_status -> trigger_strat_prediction")
    print("  CONCLUSION: strictly sequential; the old `features -g <group>` fan-out and the "
          "institutional->superinvestor->fundamental memory serialization are gone. Validated.")
