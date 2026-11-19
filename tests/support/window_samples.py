"""Pinned real-row capture and one shared deadline for timing validation."""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest
from omegaconf import DictConfig, OmegaConf

from src.context import Context, get_config_context
from src.data_aggregate.utils.common.incremental import plan_window
from src.data_aggregate.utils.common.inputs import input_windows
from src.data_store.schema import ALL, KIND_PART, Table, Tables
from src.utils.config import read_config
from src.utils.window import Window
from src.validate.checks.window_sample import SampleSelection, capture_window_selection, check_window_sample, timing_sample_tables
from src.validate.result import CheckResult, jsonable

_STARTED = pytest.StashKey[float]()


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("window samples")
    group.addoption("--window-capture", type=Path, help="Create a new pinned real-row sample manifest")
    group.addoption("--window-baseline", type=Path, help="Reuse a pinned real-row sample manifest")
    group.addoption("--window-out", type=Path, help="Write per-table timing/sample results")


def pytest_sessionstart(session: pytest.Session) -> None:
    session.config.stash[_STARTED] = time.perf_counter()


@dataclass
class WindowSamples:
    """Evidence shared by every table case, with no per-table budget reset."""

    context: Context | None
    baseline: Path
    out: Path
    started: float
    budget: float
    capturing: bool
    manifest: dict[str, Any] = field(default_factory=dict)
    results: dict[str, Any] = field(default_factory=dict)
    lineage: dict[str, Any] = field(default_factory=dict)
    planning: dict[str, Any] = field(default_factory=dict)
    identity: dict[str, Any] = field(default_factory=dict)

    def publish(self) -> None:
        # Save partial evidence atomically after each completed table; the watchdog never writes.
        elapsed = time.perf_counter() - self.started
        payload = {
            "elapsed_seconds": elapsed,
            "budget_seconds": self.budget,
            "tables": self.results,
            "lineage": self.lineage,
            "planning": self.planning,
            "identity": self.identity,
        }
        reports = [(self.out / "samples.json", payload)]
        if self.capturing:
            reports.append((self.baseline / "manifest.json", self.manifest))
        for path, report in reports:
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(jsonable(report), indent=2), encoding="utf-8")
            for attempt in range(5):
                try:
                    temporary.replace(path)
                    break
                except PermissionError:
                    if attempt == 4:
                        raise
                    # Windows indexers can briefly hold the previous evidence file open.
                    time.sleep(0.1 * (attempt + 1))

    def measure(self, table: Table) -> CheckResult:
        # Reuse the immutable pin, or capture the configured number of scoped groups.
        if table not in timing_sample_tables():
            raise ValueError(f"{table.name}: model and portfolio outputs are excluded from timing validation")
        before = time.perf_counter()
        if self.context is None:
            raise RuntimeError("The sample source context has not been constructed")
        entry: dict[str, Any] | None = self.manifest["tables"].get(table.name)
        if entry is None and self.capturing:
            entry = self.capture(table)
        if entry is None:
            entry = {"reason": "Table absent from the immutable baseline manifest"}
        if "selection" not in entry:
            result = CheckResult.abstained("window_sample", table.name, entry["reason"])
        else:
            selection_data: dict[str, Any] = entry["selection"]
            selection = SampleSelection(**selection_data)
            result = check_window_sample(self.context, table, selection=selection)
        if self.capturing:
            entry["baseline"] = result.to_json()
        elif result.status == "pass" and result.metrics["sha256"] != entry.get("baseline", {}).get("metrics", {}).get("sha256"):
            baseline_metrics = entry.get("baseline", {}).get("metrics", {})
            old_rows = {tuple(str(row.get(key)) for key in table.pk): row for row in baseline_metrics.get("records", [])}
            changes = []
            for row in result.metrics["records"]:
                previous = old_rows.get(tuple(str(row.get(key)) for key in table.pk), {})
                changed = {key: {"old": previous.get(key), "new": value} for key, value in row.items() if previous.get(key) != value}
                if changed:
                    metadata = {
                        key: row[key]
                        for key in ("first_filed_date", "available_at", "updated_at", "source", "source_period", "accession_number")
                        if key in row
                    }
                    changes.append({"pk": {key: row[key] for key in table.pk}, "fields": changed, "source_metadata": metadata})
            result = CheckResult(
                "window_sample",
                table.name,
                "abstain",
                "Pinned source cells changed; this is not an unchanged-source replay",
                scope=result.scope,
                metrics={
                    "expected_sha256": baseline_metrics.get("sha256"),
                    "observed_sha256": result.metrics["sha256"],
                    "changes": changes,
                    "expected_dtypes": dict(
                        zip(
                            baseline_metrics.get("identity", {}).get("columns", []),
                            baseline_metrics.get("identity", {}).get("dtypes", []),
                            strict=True,
                        )
                    ),
                    "observed_dtypes": dict(zip(result.metrics["identity"]["columns"], result.metrics["identity"]["dtypes"], strict=True)),
                },
            )

        # Freeze actual cells/nulls and observed work time, without certifying producers.
        measured = result.to_json()
        measured["elapsed_seconds"] = time.perf_counter() - before
        self.results[table.name] = measured
        self.publish()
        return result

    def capture(self, table: Table) -> dict[str, Any]:
        # The production selector owns fixed-year, grouped and inherited-clock key planning.
        if self.context is None:
            raise RuntimeError("The sample source context has not been constructed")
        entry = capture_window_selection(self.context, table)
        entry.update({"kind": table.kind, "pk": list(table.pk), "clock": table.date_col, "producer_replay": "unmeasured"})
        self.manifest["tables"][table.name] = entry
        return entry


@pytest.fixture(scope="session")
def window_samples(request: pytest.FixtureRequest) -> Iterator[WindowSamples]:
    # Start the single timer before connecting or capturing; missing options run no DB work.
    capture = cast(Path | None, request.config.getoption("--window-capture"))
    baseline = cast(Path | None, request.config.getoption("--window-baseline"))
    out = cast(Path | None, request.config.getoption("--window-out"))
    if (capture is None and baseline is None) or out is None:
        pytest.skip("Pass --window-capture/--window-baseline and --window-out for the real sample run")
    if capture is not None and baseline is not None:
        raise pytest.UsageError("Choose capture or baseline, not both")
    path = capture or baseline
    assert path is not None and out is not None
    with window_sample_session(path, out, capture is not None, request.config.stash[_STARTED]) as run:
        yield run


@contextmanager
def window_sample_session(path: Path, out: Path, capturing: bool, started: float) -> Iterator[WindowSamples]:
    """One readonly setup/deadline shared by pytest and the direct complete-sample worker."""
    config = read_config("configs")
    budget = float(config.validate.window_samples.budget_seconds)
    if budget <= 0:
        raise pytest.UsageError("The configured complete sample budget must be positive")
    # Enforce read-only PostgreSQL at connection level; public producer tests use SQLite.
    previous = os.environ.get("PGOPTIONS")
    os.environ["PGOPTIONS"] = (
        f"{previous or ''} -c default_transaction_read_only=on -c default_transaction_isolation=serializable -c statement_timeout={int(budget * 1000)}"
    )
    run = WindowSamples(
        None,
        path,
        out,
        started,
        budget,
        capturing,
        {
            "schema_version": 3,
            "year": int(config.validate.window_samples.year),
            "keys_per_table": int(config.validate.window_samples.keys_per_table),
            "tables": {},
        },
    )
    run.results = {table.name: {"status": "abstain", "reason": "Not measured before interruption/deadline"} for table in timing_sample_tables()}
    timer = threading.Timer(max(0, budget - (time.perf_counter() - started)), os._exit, args=(124,))
    timer.daemon = True
    with ExitStack() as cleanup:
        cleanup.callback(timer.cancel)
        if previous is None:
            cleanup.callback(os.environ.pop, "PGOPTIONS", None)
        else:
            cleanup.callback(os.environ.__setitem__, "PGOPTIONS", previous)
        timer.start()
        path.mkdir(parents=True, exist_ok=True)
        out.mkdir(parents=True, exist_ok=True)
        manifest_path = path / "manifest.json"
        if capturing and manifest_path.exists():
            raise ValueError("Capture refuses to overwrite a pinned baseline; reuse --window-baseline")
        if not capturing:
            run.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        run.identity = _sample_identity(config, None if capturing else manifest_path)
        run.publish()
        _, run.context = get_config_context("configs", use_cache=True, save=False)
        yield run
        run.publish()


def _sample_identity(config: DictConfig, manifest: Path | None) -> dict[str, Any]:
    """Record current pipeline policy beside immutable source pins, without rewriting historical metadata."""
    policy = {name: OmegaConf.to_container(config[name], resolve=True) for name in ("data", "data_extract", "data_aggregate")}
    root = Path(__file__).resolve().parents[2]
    code = hashlib.sha256()
    paths = [
        *(root / "src").rglob("*.py"),
        *[
            root / "tests" / "support" / name
            for name in ("window_samples.py", "window_lineage.py", "window_lineage_sources.py", "run_window_samples.py")
        ],
    ]
    for path in sorted(paths):
        code.update(path.relative_to(root).as_posix().encode("utf-8") + b"\0" + path.read_bytes() + b"\0")
    encoded = json.dumps(policy, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return {
        "pipeline_code_sha256": code.hexdigest(),
        "pipeline_configuration_sha256": hashlib.sha256(encoded).hexdigest(),
        "pipeline_configuration": policy,
        "source_manifest_sha256": None if manifest is None else hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "policy_scope": "Current extraction, cube input history and source publication rules; historical source manifest remains immutable",
    }


def cube_input_windows(context: Context, part: Table, sessions: pd.DatetimeIndex) -> dict[Table, Window | None]:
    """Reuse the actual producer's plan and declared payload scopes for source-state replay."""
    plan = plan_window(context, part, sessions)
    if plan is None:
        raise ValueError(f"{part.name}: no applicable actual producer window for source replay")
    return input_windows(context, part, plan.write)


def assert_lineage_results(results: dict[str, CheckResult], planning: dict[str, Any]) -> None:
    """Identical required-source and complete actual-planner acceptance in both entry points."""
    failed = [result.summary() for result in results.values() if result.status == "fail"]
    assert not failed, "\n".join(failed)
    required = [result for result in results.values() if result.metrics.get("required_routes")]
    assert required and all(result.status == "pass" for result in required), "Required semantic lineage routes are incomplete"
    expected = {table.name for table in ALL if table.kind == KIND_PART} | {Tables.cube.name}
    assert set(planning) == expected and all(result["status"] == "pass" for result in planning.values()), (
        "All ten actual production planners must pass"
    )
