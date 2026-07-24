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
                                 assemble_cube  ──▶  writes the `cube` table

Each step is `/opt/pipeline/bin/python -m src data_aggregate <cmd>` (the pipeline's isolated venv).
No step loads all source tables at once; the heavy feature computation is split across the parallel
`features:*` tasks, each of which persists a compact `cube_part_<group>`; `assemble_cube` merges the
parts (+ composites + betas + peers + targets) into the cube. Peers are computed once up front and
cached, so every downstream step reuses them.
"""
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

PROJECT = "/opt/airflow/project"
CONFIGS = f"{PROJECT}/configs"
AGG = "/opt/pipeline/bin/python -m src data_aggregate"
PEERS = "/opt/pipeline/bin/python -m src data_peers"

# feature groups (must match StepBuildCube._GROUP_SOURCES)
GROUPS = ["price", "fundamental", "sector", "earnings", "governance", "employee", "dividend",
          "attention", "institutional", "superinvestor", "insider", "short_interest", "earnings_call"]

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
    max_active_tasks=8,
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

# 3) assemble the cube from the persisted parts
assemble_cube = run("assemble-cube", pool="default_pool", task_id="assemble_cube")

deduce_peers >> [build_target, *feature_tasks] >> assemble_cube
