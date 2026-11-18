"""Protocol controls for the direct complete-sample entry point."""

import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from importlib.util import find_spec
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.data_store.schema import Table, Tables
from src.validate.checks.window_sample import timing_sample_tables
from src.validate.result import CheckResult
from tests.support import run_window_samples as worker
from tests.support import window_samples as support


def test_direct_sample_worker_is_permanent() -> None:
    assert find_spec("tests.support.run_window_samples") is not None, (
        "Complete source validation needs a durable entry point without pytest collection"
    )
    print("SANITY: the direct worker is a permanent entry point; pytest remains responsible for focused known-truth controls.")


def test_direct_worker_runs_every_scalar_then_lineage_despite_abstention(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    tables = timing_sample_tables()
    events = []
    run = SimpleNamespace(
        results={},
        lineage={},
        planning={},
        manifest={"tables": {table.name: {"applicability": "inactive" if table == Tables.notes_embedding else "active"} for table in tables}},
    )

    def measure(table: Table) -> CheckResult:
        events.append(table)
        result = CheckResult("window_sample", table.name, "abstain" if table in {Tables.sec_13f_manager_details, Tables.notes_embedding} else "pass")
        run.results[table.name] = result.to_json()
        return result

    @contextmanager
    def session(path: Path, out: Path, capturing: bool, started: float) -> Iterator[SimpleNamespace]:
        assert not capturing and started == worker._PROCESS_STARTED and started <= time.perf_counter()
        events.append("context")
        yield run

    def lineage(actual: SimpleNamespace) -> dict[str, CheckResult]:
        assert actual is run and events[2:] == list(tables)
        events.append("lineage")
        return {}

    run.measure = measure
    monkeypatch.setattr(worker.live_guard, "install", lambda: events.append("guard"))
    monkeypatch.setattr(worker.live_guard, "TRIPS", [])
    monkeypatch.setattr(worker, "window_sample_session", session)
    monkeypatch.setattr(worker, "measure_window_lineage", lineage)
    monkeypatch.setattr(worker, "assert_lineage_results", lambda results, planning: events.append("assertions"))
    monkeypatch.setattr(sys, "argv", ["run_window_samples", "--window-baseline", str(tmp_path), "--window-out", str(tmp_path)])
    assert worker.main() == 3
    assert events[:2] == ["guard", "context"] and events[-2:] == ["lineage", "assertions"]
    assert len([event for event in events if not isinstance(event, str)]) == len(tables) == 59
    print("SANITY: the direct worker keeps all59 scalar calls, required lineage/planner assertions and guard, even when a pinned source abstains.")


def test_shared_session_keeps_baseline_and_readonly_setup(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    baseline, out = tmp_path / "baseline", tmp_path / "out"
    baseline.mkdir()
    manifest = baseline / "manifest.json"
    original = '{"tables": {}}'
    manifest.write_text(original, encoding="utf-8")

    def context(*args: object, **kwargs: object) -> tuple[None, object]:
        assert "default_transaction_read_only=on" in support.os.environ["PGOPTIONS"]
        assert kwargs == {"use_cache": True, "save": False}
        return None, SimpleNamespace()

    monkeypatch.setenv("PGOPTIONS", "-c statement_timeout=123")
    monkeypatch.setattr(support, "get_config_context", context)
    with support.window_sample_session(baseline, out, False, time.perf_counter()) as run:
        assert len(run.results) == 59 and run.context is not None and not run.capturing
    assert manifest.read_text(encoding="utf-8") == original
    assert support.os.environ["PGOPTIONS"] == "-c statement_timeout=123"
    print("SANITY: pytest and direct entry share one readonly setup/deadline; immutable source pins and previous environment are preserved.")


def test_shared_acceptance_requires_every_actual_planner() -> None:
    result = CheckResult("window_lineage", "source", "pass", metrics={"required_routes": [{}]})
    with pytest.raises(AssertionError, match="ten actual"):
        support.assert_lineage_results({"source": result}, {})
    print("SANITY: a direct run cannot accept source evidence while omitting actual planner checks.")
