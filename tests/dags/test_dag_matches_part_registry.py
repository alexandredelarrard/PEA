"""
The aggregation DAG's task chain must match the cube part registry.

The old DAG carried a `GROUPS` literal with the comment "must match
StepBuildCube._GROUP_SOURCES", maintained by hand -- and it had drifted: `attention` was
commented out here but still registered there, so the nightly `cube_status` gate reported
`cube_part_attention` as missing on every run and the DAG went red for a part nobody was
building. This test replaces that comment with an assertion.

Airflow is not importable in the plain test venv, so the DAG module is parsed rather than
imported. `CHAIN` has to be a literal: the Airflow DAG processor parses `/opt/airflow/dags`
with no project on `sys.path`, so `from src...` there raises ModuleNotFoundError and the whole
DAG fails to load. The drift guard is therefore this equality, checked at test time.
"""

from __future__ import annotations

import ast
from pathlib import Path

from src.data_aggregate.utils.common.parts import PART_COMMANDS
from tests.dags.dag_harness import ALL_DONE, load_dag, replay

DAG_FILE = Path(__file__).resolve().parents[2] / "src" / "dags" / "dag_data_aggregation.py"


def _dag_source() -> str:
    assert DAG_FILE.exists(), f"missing {DAG_FILE}"
    return DAG_FILE.read_text(encoding="utf-8")


def test_dag_chain_matches_the_registry():
    src = _dag_source()
    tree = ast.parse(src)

    chain_assign = [n for n in ast.walk(tree) if isinstance(n, ast.Assign) and any(getattr(t, "id", None) == "CHAIN" for t in n.targets)]
    assert chain_assign, "the DAG no longer defines CHAIN"
    assert "from src" not in src and "import src" not in src, "the DAG processor cannot import `src`; keep CHAIN a literal"
    chain = ast.literal_eval(chain_assign[0].value)
    assert list(chain) == list(PART_COMMANDS), (
        f"CHAIN drifted from parts.PART_COMMANDS: DAG {list(chain)} vs registry {list(PART_COMMANDS)}; update the DAG literal"
    )
    # no module-level GROUPS assignment (the historical hand-synced literal). Checked on the
    # AST, not the text, so the comment explaining the history does not trip it.
    assigned = {getattr(t, "id", None) for n in ast.walk(tree) if isinstance(n, ast.Assign) for t in n.targets}
    assert "GROUPS" not in assigned, "the old hand-synced GROUPS literal is back"

    print("\n=== SANITY CHECK: DAG chain <-> part registry ===")
    print(f"  CHAIN literal {list(chain)} == registry {list(PART_COMMANDS)}")
    print(
        "  CONCLUSION: the DAG's literal task list equals the registry, so the `attention` "
        "mismatch that made cube_status permanently red would fail this test. Validated."
    )


def test_dag_is_sequential_and_ends_in_the_status_gate():
    src = _dag_source()
    assert "max_active_tasks=1" in src, (
        "the chain must be sequential -- peak memory should be the largest single step, not the sum of parallel pool slots"
    )
    assert "chain(deduce_peers, *step_tasks, assemble_cube, cube_status," in src, (
        "the DAG must run peers -> the registry steps -> assemble -> status, in order"
    )
    # the memory-driven serialization the old DAG needed is gone
    for gone in ("institutional_task", "superinvestor_task", "fundamental_task", "features -g"):
        assert gone not in src, f"'{gone}' should have been removed with the sub-step split"

    print("\n=== SANITY CHECK: DAG shape ===")
    print(f"  max_active_tasks=1; deduce_peers -> {len(PART_COMMANDS)} steps -> assemble_cube -> cube_status -> trigger_strat_prediction")
    print(
        "  CONCLUSION: strictly sequential; the old `features -g <group>` fan-out and the "
        "institutional->superinvestor->fundamental memory serialization are gone. Validated."
    )


def test_every_step_and_the_prediction_trigger_run_all_done():
    dag = load_dag(DAG_FILE)
    steps = [command.replace("-", "_") for command in PART_COMMANDS]
    gated = [*steps, "assemble_cube", "cube_status", "trigger_strat_prediction"]
    assert {task_id: dag.tasks[task_id].trigger_rule for task_id in gated} == dict.fromkeys(gated, ALL_DONE)
    assert dag.tasks["trigger_strat_prediction"].upstream == {"cube_status"}

    for failed in ([steps[0]], ["build_text", "assemble_cube"], ["cube_status"]):
        states = replay(dag, failed)
        assert states["trigger_strat_prediction"] == "success", (failed, states)
        assert all(states[task_id] in ("success", "failed") for task_id in gated), "no step is skipped as upstream_failed"

    print("\n=== SANITY CHECK: non-blocking aggregation DAG (AC-015) ===")
    print(f"  {len(steps)} build steps, assemble_cube, cube_status and trigger_strat_prediction all run ALL_DONE")
    print("  replay: build_prices failed / build_text + assemble_cube failed / cube_status RED -> later steps still run and prediction is triggered")
    print("  (predict itself refuses stale inputs: tests/modelling/test_predict_freshness.py). Validated.")
