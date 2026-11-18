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
from tests.dags.dag_harness import ALL_DONE, StubDag, load_dag, replay

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


def _descendants(dag: StubDag, task_id: str) -> set[str]:
    """Every task that runs after `task_id`."""
    found: set[str] = set()
    frontier = {task_id}
    while frontier:
        frontier = {name for name, task in dag.tasks.items() if task.upstream & frontier} - found
        found |= frontier
    return found


def test_dag_is_sequential_except_the_manager_profile_beside_momentum():
    src = _dag_source()
    dag = load_dag(DAG_FILE)
    assert dag.kwargs["max_active_tasks"] == 2 and "max_active_tasks=2" in src, "two slots: the profile task and one cube step"
    assert "chain(deduce_peers, *step_tasks" not in src, "the one long chain is split into task-to-task pairs"
    tasks = dag.tasks
    assert tasks["build_manager_profile"].upstream == {"build_fundamentals"}
    assert tasks["build_momentum"].upstream == {"build_fundamentals"}
    assert tasks["build_cluster"].upstream == {"build_manager_profile"}
    assert tasks["build_text"].upstream == {"build_momentum", "build_cluster"}
    assert tasks["build_institutionals"].upstream == {"build_text"}, "the profile task is finished before the 13F parts start"
    assert tasks["build_prices"].upstream == {"deduce_peers"} and tasks["assemble_cube"].upstream == {"build_governance"}
    others = set(tasks) - {"build_manager_profile"}
    parallel = {
        name for name in others if name not in _descendants(dag, "build_manager_profile") and "build_manager_profile" not in _descendants(dag, name)
    }
    assert parallel == {"build_momentum"}, "the profile task overlaps build_momentum only"
    cluster_parallel = {
        name
        for name in set(tasks) - {"build_cluster"}
        if name not in _descendants(dag, "build_cluster") and "build_cluster" not in _descendants(dag, name)
    }
    assert cluster_parallel == {"build_momentum"}, "the cluster task overlaps build_momentum only"
    # the memory-driven serialization the old DAG needed is gone
    for gone in ("institutional_task", "superinvestor_task", "fundamental_task", "features -g"):
        assert gone not in src, f"'{gone}' should have been removed with the sub-step split"

    print("\n=== SANITY CHECK: DAG shape (AC-106) ===")
    print(
        f"  max_active_tasks=2; build_fundamentals -> [build_momentum, build_manager_profile] -> build_text; parallel with the profile: {sorted(parallel)}"
    )
    print("  CONCLUSION: the profile task overlaps build_momentum only and ends before build_institutionals and build_sec_thirteen. Validated.")


def test_every_step_and_the_prediction_trigger_run_all_done():
    dag = load_dag(DAG_FILE)
    steps = [command.replace("-", "_") for command in PART_COMMANDS]
    gated = [*steps, "build_manager_profile", "build_cluster", "assemble_cube", "cube_status", "trigger_strat_prediction"]
    assert {task_id: dag.tasks[task_id].trigger_rule for task_id in gated} == dict.fromkeys(gated, ALL_DONE)
    assert dag.tasks["trigger_strat_prediction"].upstream == {"cube_status"}

    for failed in ([steps[0]], ["build_text", "assemble_cube"], ["cube_status"], ["build_manager_profile"], ["build_cluster"]):
        states = replay(dag, failed)
        assert states["trigger_strat_prediction"] == "success", (failed, states)
        assert all(states[task_id] in ("success", "failed") for task_id in gated), "no step is skipped as upstream_failed"

    print("\n=== SANITY CHECK: non-blocking aggregation DAG (AC-015) ===")
    print(f"  {len(steps)} build steps, assemble_cube, cube_status and trigger_strat_prediction all run ALL_DONE")
    print("  replay: build_prices failed / build_text + assemble_cube failed / cube_status RED -> later steps still run and prediction is triggered")
    print("  (predict itself refuses stale inputs: tests/modelling/test_predict_freshness.py). Validated.")
