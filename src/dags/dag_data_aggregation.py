"""
dag_data_aggregation.py  (src/dags/dag_data_aggregation.py)
-----------------------------------------------------------
Nightly DATA-AGGREGATION DAG — the cube build as seven sequential, memory-bounded steps.
Triggered by the extraction DAG once ALL sources have refreshed (schedule=None).

    deduce_peers ─▶ build_prices ─▶ build_target ─▶ build_fundamentals ─▶ build_momentum
                         │                                                        │
                         │  (normalizes `prices` ONCE into cube_part_prices /     ▼
                         │   every later step reads that part                  build_text
                         │   back, projected to the fields it needs)              │
                         │                                                        ▼
                         │                                                   build_extras
                         ▼                                                        │
                    assemble_cube ◀────────────────────────────────────────────────┘
                         │
                         ├─▶ writes the `cube` table
                         ▼
                    cube_status (XCom: max date/rows per part; RED if a part is behind)
                         └──▶ trigger `strat_prediction` (daily)

Each step is `/opt/pipeline/bin/python -m src data_aggregate <cmd>` (the pipeline's isolated venv).
STRICTLY SEQUENTIAL, on purpose: peak memory is the largest single step rather than the sum of two
parallel pool slots, and each step keeps its heavy frames local so they are freed when it returns.
`assemble_cube` merges the parts (+ composites + betas + peers + targets) into the cube. Peers are
computed once up front and cached; `build_prices` folds them into a persisted sector-return column.
"""
import json
import subprocess
from datetime import datetime, timedelta

from airflow import DAG
from airflow.exceptions import AirflowFailException
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.models.baseoperator import chain
from airflow.utils.trigger_rule import TriggerRule

PROJECT = "/opt/airflow/project"
CONFIGS = f"{PROJECT}/configs"
PIPE_PY = "/opt/pipeline/bin/python"
AGG = f"{PIPE_PY} -m src data_aggregate"
PEERS = f"{PIPE_PY} -m src data_peers"

# The ordered cube sub-steps. Imported from the part registry rather than hand-listed: the
# old GROUPS literal was documented as "must match StepBuildCube._GROUP_SOURCES" and had
# already drifted (`attention` was commented out here but still registered there, so the
# status gate reported cube_part_attention missing on every run).
# tests/dags/test_dag_matches_part_registry.py asserts the two stay in step.
from src.data_aggregate.utils.common.parts import PART_COMMANDS  # noqa: E402

CHAIN = list(PART_COMMANDS)

default_args = {
    "owner": "pea",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
    "email_on_failure": False,
}

dag = DAG(
    dag_id="data_aggregation",
    default_args=default_args,
    description="Build the cube from the DB in seven sequential, memory-bounded steps.",
    schedule=None,                                   # triggered by the extraction DAG when it finishes
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_tasks=1,               # sequential: peak memory = the largest single step
    tags=["pea", "aggregation"],
)


def run(cmd: str, base: str = AGG, pool: str = "aggregate", task_id: str | None = None) -> BashOperator:
    return BashOperator(
        task_id=task_id or cmd.split()[0].replace("-", "_"),
        bash_command=f"{base} {cmd} -c {CONFIGS}",
        cwd=PROJECT,
        pool=pool,
        dag=dag,
    )


# 1) peers once (cached) — build-prices turns them into the persisted sector-return column
deduce_peers = run("deduce-peers", base=PEERS, pool="default_pool", task_id="deduce_peers")

# 2) the seven sub-steps, STRICTLY SEQUENTIAL. Peak memory is now the largest single step
#    rather than the sum of two parallel pool slots, so the `aggregate` pool is unnecessary
#    and the old institutional -> superinvestor -> fundamental serialization (which existed
#    only to keep those three off each other's memory) is gone: they are `build-extras` and
#    `build-fundamentals`, sequential by construction.
step_tasks = [run(cmd, pool="default_pool", task_id=cmd.replace("-", "_")) for cmd in CHAIN]

# 3) assemble the cube from the persisted parts
assemble_cube = run("assemble-cube", pool="default_pool", task_id="assemble_cube")

def _cube_status(**context) -> None:
    """Push the max date + row count of every cube part (+ cube / predictions) to XCom, so drift is
    visible; RED when a part is missing or more than one build behind the cube. XCom is pushed
    BEFORE raising, so the per-part status is available even on a red run."""
    proc = subprocess.run([PIPE_PY, "-m", "src", "data_aggregate", "cube-status", "-c", CONFIGS],
                          cwd=PROJECT, capture_output=True, text=True)
    report = None
    for line in reversed((proc.stdout or "").strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                report = json.loads(line)
                break
            except json.JSONDecodeError:
                continue
    ti = context["ti"]
    if report is None:
        ti.xcom_push(key="cube_status", value={"ok": False, "error": "no report",
                                               "stderr_tail": (proc.stderr or "")[-800:]})
        raise AirflowFailException(f"cube-status produced no report (rc={proc.returncode})")
    ti.xcom_push(key="cube_status", value=report)
    for name, info in report.get("parts", {}).items():
        ti.xcom_push(key=f"max_{name}", value=info.get("max_date"))
    if not report.get("ok", False):
        raise AirflowFailException("Cube parts behind/missing (RED): " + ", ".join(report.get("behind", [])))


# 4) status gate: latest date per cube part -> XCom (RED if a part is behind); visible, not blocking
cube_status = PythonOperator(task_id="cube_status", python_callable=_cube_status, dag=dag)

# 5) kick off the DAILY prediction DAG (predict -> strategy ledger) once the cube is fresh.
#    NOT `modelling`: (re)training is weekly (Saturday, see dag_modelling.py) while a freshly
#    rebuilt cube should be SCORED every night, so the nightly downstream is prediction only.
trigger_strat_prediction = TriggerDagRunOperator(
    task_id="trigger_strat_prediction", trigger_dag_id="strat_prediction",
    wait_for_completion=False, reset_dag_run=True, trigger_rule=TriggerRule.ALL_DONE, dag=dag)

chain(deduce_peers, *step_tasks, assemble_cube, cube_status, trigger_strat_prediction)
