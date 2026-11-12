"""`StepExtractZips`: phase order (the 13F data sets download last), one Notes period list, options reaching each source, per-source failures, CLI `--only`,
the 13F ingest outside `run`."""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest
from click.testing import CliRunner

from src.data_extract import cli as cli_mod
from src.data_extract.transformers import step_extract_zips as module
from src.data_extract.transformers.step_extract_zips import StepExtractZips, ZipsOptions
from tests.support.universe import universe_of

DOWNLOADS = ("insider_download", "notes_download", "ftd_download", "13f_download")
INGESTS = ("insider_ingest", "notes_ingest", "ftd_ingest")
METHODS = {
    "insider_download": "run_download_insider",
    "notes_download": "run_download_financial_notes",
    "ftd_download": "run_download_fails_to_deliver",
    "13f_download": "run_download_13f",
    "insider_ingest": "run_ingest_insider",
    "notes_ingest": "run_ingest_financial_notes",
    "ftd_ingest": "run_ingest_fails_to_deliver",
}


def _config() -> SimpleNamespace:
    paths = SimpleNamespace(insider_transactions="insider", financial_notes="notes", fails_deliver="ftd")
    return SimpleNamespace(data_extract=SimpleNamespace(years_history=15), local=SimpleNamespace(paths=paths))


def _stub(monkeypatch: pytest.MonkeyPatch, calls: list[tuple[str, tuple, dict]], boom: str | None = None) -> list[int]:
    scrapes: list[int] = []

    def mark(name: str, result: object = 0):
        def call(*args: Any, **kwargs: Any) -> object:
            calls.append((name, args, kwargs))
            if name == boom:
                raise RuntimeError(f"simulated {name} failure")
            return result

        return call

    def periods(context: Any, years_history: int, today: pd.Timestamp) -> list[str]:
        scrapes.append(years_history)
        return ["2026q1"]

    monkeypatch.setattr(module, "cache_dir", lambda context, name: Path("cache") / name)
    monkeypatch.setattr(module, "notes_periods", periods)
    monkeypatch.setattr(module, "download_insider_transactions", mark("insider_download", []))
    monkeypatch.setattr(module, "download_notes_periods", mark("notes_download"))
    monkeypatch.setattr(module, "download_fails_to_deliver", mark("ftd_download"))
    monkeypatch.setattr(module, "download_13f_datasets", mark("13f_download"))
    monkeypatch.setattr(module, "fetch_insider_transactions", mark("insider_ingest"))
    monkeypatch.setattr(module, "ingest_financial_notes", mark("notes_ingest"))
    monkeypatch.setattr(module, "repair_notes_availability", mark("notes_repair"))
    monkeypatch.setattr(module, "fetch_fails_to_deliver", mark("ftd_ingest"))
    monkeypatch.setattr(module, "ingest_13f_datasets", mark("13f_ingest"))
    return scrapes


def _context() -> SimpleNamespace:
    return SimpleNamespace(config=_config(), config_dir="configs", log=logging.getLogger(__name__), universe=universe_of(["AAA", "BBB"]))


def test_run_downloads_then_ingests_with_one_notes_period_list(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, tuple, dict]] = []
    scrapes = _stub(monkeypatch, calls)
    StepExtractZips(_context(), _config()).run()  # type: ignore[arg-type]

    order = [name for name, _, _ in calls]
    assert order == [*DOWNLOADS, *INGESTS], order
    assert scrapes == [16], "the Notes landing page is scraped once per step, for the window plus one year"
    notes_download = next(args for name, args, _ in calls if name == "notes_download")
    notes_ingest = next(args for name, args, _ in calls if name == "notes_ingest")
    assert notes_download[2] == notes_ingest[3] == ["2026q1"], "download and ingest share one period list"

    print("\n=== SANITY CHECK: StepExtractZips.run order ===")
    print(f"  download: {' -> '.join(DOWNLOADS)}")
    print(f"  ingest:   {' -> '.join(INGESTS)}")
    print(f"  Notes landing scrapes: {len(scrapes)}")
    print("  OK: a plain sequence, and both Notes phases read one period list")


def test_options_reach_every_source(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, tuple, dict]] = []
    _stub(monkeypatch, calls)
    as_of = pd.Timestamp("2026-10-30")
    StepExtractZips(_context(), _config(), ZipsOptions(tickers=["AAA"], full=True, as_of=as_of)).run_ingest()  # type: ignore[arg-type]

    seen = {name: (args, kwargs) for name, args, kwargs in calls}
    insider_args, insider_kwargs = seen["insider_ingest"]
    assert insider_args[1:] == (["AAA"], 15) and insider_kwargs == {"reparse": True, "as_of": as_of}
    scope = seen["notes_ingest"][0][2]
    assert (scope.tickers, scope.run_date, scope.full) == (["AAA"], as_of, True)
    ftd_args, ftd_kwargs = seen["ftd_ingest"]
    assert ftd_args[1:] == (True,) and ftd_kwargs == {}

    print("\n=== SANITY CHECK: StepExtractZips options ===")
    print("  tickers, full and run date declared once in the init reach all three ingests")


def test_repair_availability_only_repairs_the_notes_clocks(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, tuple, dict]] = []
    scrapes = _stub(monkeypatch, calls)
    StepExtractZips(_context(), _config(), ZipsOptions(repair_availability=True)).run_ingest_financial_notes()  # type: ignore[arg-type]

    assert [name for name, _, _ in calls] == ["notes_repair"] and scrapes == []

    print("\n=== SANITY CHECK: --repair-availability ===")
    print("  metadata-only clock repair: no landing scrape, no parse")


@pytest.mark.parametrize("boom", list(METHODS))
def test_a_failing_source_raises_from_its_own_method_only(monkeypatch: pytest.MonkeyPatch, boom: str) -> None:
    calls: list[tuple[str, tuple, dict]] = []
    _stub(monkeypatch, calls, boom=boom)
    step = StepExtractZips(_context(), _config())  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match=boom):
        getattr(step, METHODS[boom])()
    for name, method in METHODS.items():
        if name != boom:
            getattr(step, method)()
    with pytest.raises(RuntimeError, match=boom):
        step.run()

    phase = DOWNLOADS if boom in DOWNLOADS else (*DOWNLOADS, *INGESTS)
    last_run = [name for name, _, _ in calls][-(phase.index(boom) + 1) :]
    assert last_run == list(phase[: phase.index(boom) + 1]), last_run

    print(f"\n=== SANITY CHECK: {boom} failure ===")
    print(f"  {METHODS[boom]} raised alone; the other {len(METHODS) - 1} methods ran; run() stopped at it")


def test_cli_only_runs_the_named_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, tuple, dict]] = []
    _stub(monkeypatch, calls)
    monkeypatch.setattr(cli_mod, "get_config_context", lambda *args, **kwargs: (_config(), _context()))

    runner = CliRunner()
    ingest = runner.invoke(cli_mod.cli, ["zips-ingest", "--only", "notes", "--only", "ftd"])
    download = runner.invoke(cli_mod.cli, ["zips-download", "--only", "insider"])
    thirteen_f = runner.invoke(cli_mod.cli, ["zips-download", "--only", "13f"])
    everything = runner.invoke(cli_mod.cli, ["zips-ingest"])
    all_downloads = runner.invoke(cli_mod.cli, ["zips-download"])
    ingest_13f = runner.invoke(cli_mod.cli, ["zips-ingest", "--only", "13f"])

    results = (ingest, download, thirteen_f, everything, all_downloads, ingest_13f)
    assert all(r.exit_code == 0 for r in results), [r.output for r in results]
    order = [name for name, _, _ in calls]
    assert order == ["notes_ingest", "ftd_ingest", "insider_download", "13f_download", *INGESTS, *DOWNLOADS, "13f_ingest"], order

    print("\n=== SANITY CHECK: CLI --only (D-16) ===")
    print("  zips-ingest --only notes --only ftd -> 2; zips-download --only insider / --only 13f -> 1 each; zips-ingest -> 3; zips-download -> 4")
    print("  zips-ingest --only 13f -> the 13F ingest alone; the bare zips-ingest never runs it")


def test_13f_ingest_runs_by_name_only(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, tuple, dict]] = []
    _stub(monkeypatch, calls)
    step = StepExtractZips(_context(), _config(), ZipsOptions(full=True))  # type: ignore[arg-type]
    step.run()
    step.run_ingest()
    in_run = [name for name, _, _ in calls]
    step.run_ingest_13f()

    assert "13f_ingest" not in in_run, in_run
    assert calls[-1][0] == "13f_ingest" and calls[-1][2] == {"full": True}, calls[-1]

    print("\n=== SANITY CHECK: 13F ingest outside run() ===")
    print(f"  run() + run_ingest() -> {len(in_run)} calls, none the 13F ingest; run_ingest_13f() passes full=True alone")
