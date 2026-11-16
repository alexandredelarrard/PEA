"""Pinned real-row capture and one shared deadline for timing validation."""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Iterator
from contextlib import ExitStack
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest

from src.context import Context, get_config_context
from src.data_store.schema import ALL, Table
from src.utils.config import read_config
from src.validate.checks.window_sample import SampleSelection, check_window_sample
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

    def publish(self) -> None:
        # Save partial evidence atomically after each completed table; the watchdog never writes.
        elapsed = time.perf_counter() - self.started
        payload = {"elapsed_seconds": elapsed, "budget_seconds": self.budget, "tables": self.results}
        reports = [(self.out / "samples.json", payload)]
        if self.capturing:
            reports.append((self.baseline / "manifest.json", self.manifest))
        for path, report in reports:
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(jsonable(report), indent=2), encoding="utf-8")
            temporary.replace(path)

    def measure(self, table: Table) -> CheckResult:
        # Reuse the pinned selection, or discover one deterministic complete primary key.
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
            result = CheckResult.abstained("window_sample", table.name, "Pinned source cells changed; this is not an unchanged-source replay")

        # Freeze actual cells/nulls and observed work time, without certifying producers.
        measured = result.to_json()
        measured["elapsed_seconds"] = time.perf_counter() - before
        self.results[table.name] = measured
        self.publish()
        return result

    def capture(self, table: Table) -> dict[str, Any]:
        # Inspect only this table, then request one indexed primary key, never a global max.
        if self.context is None:
            raise RuntimeError("The sample source context has not been constructed")
        columns = self.context.store.columns(table)
        entry: dict[str, Any]
        if not columns:
            entry = {"reason": "Source table absent"}
        elif not set(table.pk) <= set(columns):
            entry = {"reason": "Live schema lacks the registered primary key"}
        else:
            df_key = self.context.store.load(table, columns=table.pk, order_by=table.pk, limit=1, markers=True, optional=True)
            entry = self.selection(table, columns, df_key)
        entry.update({"kind": table.kind, "pk": list(table.pk), "clock": table.date_col, "producer_replay": "unmeasured"})
        self.manifest["tables"][table.name] = entry
        return entry

    @staticmethod
    def selection(table: Table, columns: list[str], df_key: pd.DataFrame | None) -> dict[str, Any]:
        # Exact keys bound every subsequent read; vectors remain explicit provenance gaps.
        if df_key is None or df_key.empty:
            return {"reason": "Source table empty"}
        if df_key.isna().any(axis=None):
            return {"reason": "Selected source primary key contains NULL"}
        projection = tuple(col for col in columns if col != table.vector_col)
        where = jsonable(df_key.iloc[0].to_dict())
        return {"selection": asdict(SampleSelection(projection, where)), "excluded_vector": table.vector_col}


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
    started = request.config.stash[_STARTED]
    config = read_config("configs")
    budget = float(config.validate.window_samples.budget_seconds)
    if budget <= 0:
        raise pytest.UsageError("The configured complete sample budget must be positive")
    # Enforce read-only PostgreSQL at connection level; public producer tests use SQLite.
    previous = os.environ.get("PGOPTIONS")
    os.environ["PGOPTIONS"] = f"{previous or ''} -c default_transaction_read_only=on -c statement_timeout={int(budget * 1000)}"
    run = WindowSamples(None, path, out, started, budget, capture is not None, {"tables": {}})
    run.results = {table.name: {"status": "abstain", "reason": "Not measured before interruption/deadline"} for table in ALL}
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
        if capture is not None and manifest_path.exists():
            raise pytest.UsageError("Capture refuses to overwrite a pinned baseline; reuse --window-baseline")
        if baseline is not None:
            run.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        run.publish()
        _, run.context = get_config_context("configs", use_cache=True, save=False)
        yield run
        run.publish()
