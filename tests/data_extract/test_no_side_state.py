"""No resume decision reads side state under `data/` (AC-001): a source scan of `src/` and `configs/`."""

from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
CONFIGS = ROOT / "configs"
RESUME = SRC / "data_extract" / "utils" / "common" / "resume.py"
#: The run manifest, the bulk-archive sidecars and their helpers, retired for the `schema.Resume` planners.
_SIDE_STATE_TOKENS = (
    "run_manifest",
    "manifest_window",
    "record_run",
    "extraction_manifest",
    "_universe.json",
    "pending_periods",
    "mark_processed",
    "_processed_scope",
)
#: Config keys that only the retired side state read.
_RETIRED_CONFIG_KEYS = ("manifest_full_rescan_days", "reconcile_days", "extraction_manifest")
#: A fetcher's planning function: what decides a run's work list or window.
_PLANNER_NAME = re.compile(r"plan|worklist|window|frontier")


def _sources() -> list[Path]:
    return sorted(SRC.rglob("*.py"))


def test_no_source_reads_or_writes_side_state() -> None:
    hits = [
        f"{path.relative_to(ROOT)}:{n}: {line.strip()}"
        for path in _sources()
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
        if any(token in line for token in _SIDE_STATE_TOKENS)
    ]
    assert not hits, "side-state references left in src/:\n" + "\n".join(hits)
    print("\n=== SANITY CHECK: no run manifest or sidecar in src/ ===")
    print(f"  {len(_sources())} source files scanned for {len(_SIDE_STATE_TOKENS)} tokens: no hit. Validated.")


def test_no_config_carries_a_retired_side_state_key() -> None:
    configs = sorted(CONFIGS.rglob("*.yml"))
    hits = [
        f"{path.relative_to(ROOT)}:{n}: {line.strip()}"
        for path in configs
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
        if any(key in line for key in _RETIRED_CONFIG_KEYS)
    ]
    assert not hits, "retired side-state config keys:\n" + "\n".join(hits)
    print("\n=== SANITY CHECK: no retired config key ===")
    print(f"  {len(configs)} YAML files scanned for {_RETIRED_CONFIG_KEYS}: no hit. Validated.")


def test_no_planner_reads_the_data_store_directory() -> None:
    scanned: list[str] = []
    hits: list[str] = []
    if "DATA_STORE" in RESUME.read_text(encoding="utf-8"):
        hits.append(f"{RESUME.relative_to(ROOT)}: reads DATA_STORE")
    for path in sorted((SRC / "data_extract").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and _PLANNER_NAME.search(node.name):
                scanned.append(f"{path.stem}.{node.name}")
                if "DATA_STORE" in ast.unparse(node):
                    hits.append(f"{path.relative_to(ROOT)}:{node.lineno}: {node.name} reads DATA_STORE")
    assert scanned, "the planner-name pattern matched no function"
    assert not hits, "planning code reads the data directory:\n" + "\n".join(hits)
    print("\n=== SANITY CHECK: planners read the DB, not data/ ===")
    print(f"  resume.py plus {len(scanned)} planning functions (e.g. {', '.join(sorted(scanned)[:4])}): no DATA_STORE read. Validated.")
