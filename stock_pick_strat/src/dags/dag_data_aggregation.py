"""
dag_data_aggregation.py  (src/dags/dag_data_aggregation.py)
-----------------------------------------------------------
Nightly DATA-AGGREGATION DAG — the cube build EXPLODED into small, memory-light, parallel steps.
Triggered by the extraction DAG once ALL sources have refreshed (schedule=None).

    deduce_peers ──▶ ┌ build_target                    ┐
                     ├ features:price                  │  (parallel, capped by the `aggregate` pool
                     ├ features:fundamental             │   so peak memory stays bounded — each task
                     ├ features:sector ... earnings_call┘   loads only prices + peers + its source)
                                       │
                                       ▼
                                 assemble_cube  ──▶ cube_status (XCom: max date/rows per part; RED if
                                       │             a part is missing / behind)
                                       │                     └──▶ trigger `strat_prediction` (daily)
                                       ▼
                                 writes the `cube` table

Each step is `/opt/pipeline/bin/python -m src data_aggregate <cmd>` (the pipeline's isolated venv).
No step loads all source tables at once; the heavy feature computation is split across the parallel
`features:*` tasks, each of which persists a compact `cube_part_<group>`; `assemble_cube` merges the
parts (+ composites + betas + peers + targets) into the cube. Peers are computed once up front and
cached, so every downstream step reuses them.
"""
import json
import subprocess
from datetime import datetime, timedelta

from airflow import DAG
from airflow.exceptions import AirflowFailException
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.utils.trigger_rule import TriggerRule

PROJECT = "/opt/airflow/project"
CONFIGS = f"{PROJECT}/configs"
PIPE_PY = "/opt/pipeline/bin/python"
AGG = f"{PIPE_PY} -m src data_aggregate"
PEERS = f"{PIPE_PY} -m src data_peers"

# feature groups (must match StepBuildCube._GROUP_SOURCES). Earnings calls are TWO tasks:
# FinBERT/LM sentiment vs OpenAI-embedding analysis.
GROUPS = ["price", "sector", "earnings", "governance", "employee", "dividend",
          "insider", "short_interest", # "attention", 
          "earnings_call_sentiment", "earnings_call_embedding"]

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
    description="Build the cube from the DB in exploded, memory-light, parallel steps.",
    schedule=None,                                   # triggered by the extraction DAG when it finishes
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_tasks=2,
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


# 1) peers once (cached) — every feature/target step reads the peer dict it writes
deduce_peers = run("deduce-peers", base=PEERS, pool="default_pool", task_id="deduce_peers")

# 2) target + one task per feature group, in parallel (memory-capped by the `aggregate` pool)
build_target = run("build-target", task_id="build_target")
feature_tasks = [run(f"features -g {g}", task_id=f"features_{g}") for g in GROUPS]
fundamental_task =  run("features -g fundamental", task_id="features_fundamental")
institutional_task =  run("features -g institutional", task_id="features_institutional")
superinvestor_task =  run("features -g superinvestor", task_id="features_superinvestor")

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

deduce_peers >> [build_target, *feature_tasks] >> institutional_task >> superinvestor_task >> fundamental_task >> assemble_cube >> cube_status >> trigger_strat_prediction
