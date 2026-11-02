"""`StepExtractIdentity`: captures before the builders, one Form 3/4/5 scan, per-stage failures, and the D-15 cold start."""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest
from click.testing import CliRunner

from src.data_extract import cli as cli_mod
from src.data_extract.transformers import step_extract_identity as module
from src.data_extract.transformers.step_extract_identity import IdentityOptions, StepExtractIdentity
from src.data_extract.utils.zips import fetch_fails_to_deliver as ftd
from src.data_store.schema import Tables
from tests.canonical_schema import create_canonical
from tests.data_extract.institutionals.test_fails_to_deliver import _HEADER, _scope_lineage

STAGES = ("dei", "ftd", "scan", "tenure", "lineage", "master", "refresh", "propagate")


def _config() -> SimpleNamespace:
    paths = SimpleNamespace(insider_transactions="insider", financial_notes="notes", fails_deliver="sec_fails_to_deliver")
    return SimpleNamespace(local=SimpleNamespace(paths=paths), data_extract=SimpleNamespace(years_history=1, redundant_ticks=["GOOG"]))


def _stub_stages(monkeypatch: pytest.MonkeyPatch, order: list[str], boom: str | None = None) -> None:
    """Every stage's operation records its call; `boom` raises."""
    scan = SimpleNamespace(owner_pairs=pd.DataFrame())

    def mark(name: str, result: object = None):
        def call(*args: Any, **kwargs: Any) -> object:
            order.append(name)
            if name == boom:
                raise RuntimeError(f"simulated {name} failure")
            return result

        return call

    monkeypatch.setattr(module, "cache_dir", lambda context, name: Path(name))
    monkeypatch.setattr(module, "load_universe_tickers", lambda context: ["AAA", "BBB"])
    monkeypatch.setattr(module, "capture_pending_dei", mark("dei", 0))
    monkeypatch.setattr(module, "capture_ftd_cusips", mark("ftd", 0))
    monkeypatch.setattr(module, "scan_form345_cache", mark("scan", scan))
    monkeypatch.setattr(module, "build_symbol_tenure", mark("tenure", pd.DataFrame()))
    monkeypatch.setattr(module, "build_entity_lineage", mark("lineage", pd.DataFrame()))
    monkeypatch.setattr(module, "build_security_master", mark("master", pd.DataFrame()))
    monkeypatch.setattr(module, "load_identity", mark("refresh", object()))
    monkeypatch.setattr(module, "propagate_identity", mark("propagate"))
    monkeypatch.setattr(module, "load_stored_lineage", lambda context: pd.DataFrame())
    monkeypatch.setattr(module, "load_built_tenure", lambda context: pd.DataFrame())


def _context(store: Any = None, tmp_path: Path | None = None) -> SimpleNamespace:
    return SimpleNamespace(store=store, log=logging.getLogger("test.identity_step"), paths={"DATA_STORE": tmp_path}, config_dir="configs")


def test_run_captures_first_then_builds_and_propagates_with_one_scan(monkeypatch: pytest.MonkeyPatch) -> None:
    order: list[str] = []
    _stub_stages(monkeypatch, order)

    StepExtractIdentity(_context(), _config()).run()  # type: ignore[arg-type]

    assert order == list(STAGES), order
    assert order.count("scan") == 1, "tenure and lineage share one Form 3/4/5 scan"
    print("\n=== SANITY CHECK: StepExtractIdentity.run order ===")
    print(f"  {' -> '.join(order)}; one scan shared by tenure and lineage")
    print("  OK: the captures land before the builders read their evidence, and propagation runs on the rebuilt tables")


@pytest.mark.parametrize("boom", ["dei", "ftd", "tenure", "lineage", "master", "propagate"])
def test_a_failing_stage_raises_from_its_own_method_only(monkeypatch: pytest.MonkeyPatch, boom: str) -> None:
    methods = {
        "dei": "run_capture_dei_symbols",
        "ftd": "run_capture_ftd_cusips",
        "tenure": "run_symbol_tenure",
        "lineage": "run_entity_lineage",
        "master": "run_security_master",
        "propagate": "run_propagate",
    }
    order: list[str] = []
    _stub_stages(monkeypatch, order, boom=boom)
    step = StepExtractIdentity(_context(), _config())  # type: ignore[arg-type]

    for name, method in methods.items():
        if name == boom:
            with pytest.raises(RuntimeError, match=boom):
                getattr(step, method)()
        else:
            getattr(step, method)()
    with pytest.raises(RuntimeError, match=boom):
        StepExtractIdentity(_context(), _config()).run()  # type: ignore[arg-type]

    print(f"\n=== SANITY CHECK: {boom} failure isolation ===")
    print(f"  {methods[boom]} raised alone; the other five sub-runs completed when called alone; run() stopped at it")


def test_options_reach_the_propagation_and_the_lineage_build(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, dict[str, Any]] = {}
    _stub_stages(monkeypatch, [])
    monkeypatch.setattr(module, "build_entity_lineage", lambda *args, **kwargs: calls.setdefault("lineage", kwargs))
    monkeypatch.setattr(
        module, "propagate_identity", lambda context, tickers, **kwargs: calls.setdefault("propagate", {"tickers": tickers, **kwargs})
    )
    options = IdentityOptions(tickers=["AAA"], full=True, dry_run=True, as_of=pd.Timestamp("2026-10-30"), approved_rekeys=frozenset({("E1", "E2")}))
    step = StepExtractIdentity(_context(), _config(), options)  # type: ignore[arg-type]

    step.run_entity_lineage()
    step.run_propagate()

    assert calls["lineage"]["approved_rekeys"] == {("E1", "E2")} and calls["lineage"]["redundant_symbols"] == {"GOOG"}
    propagate = calls["propagate"]
    assert propagate["tickers"] == ["AAA"] and propagate["all_tickers"] == ["AAA", "BBB"]
    assert propagate["dry_run"] and propagate["every_ticker"] and propagate["as_of"] == pd.Timestamp("2026-10-30")
    print("\n=== SANITY CHECK: identity options ===")
    print("  --approve-rekey and redundant_ticks reach the lineage; -t, --full, --dry-run and --as-of reach the propagation")


def test_cold_start_sequence_captures_ftd_once_a_lineage_exists(sqlite_store: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """D-15: `identity` (no lineage: the FTD capture is a no-op) -> `identity --only capture-ftd -F` -> `identity --only master`."""
    tables = [Tables.entity_lineage, Tables.security_master, Tables.sec_fails_to_deliver_security, Tables.sec_fails_to_deliver, Tables.symbol_tenure]
    create_canonical(sqlite_store.engine, tables)
    context, config = _context(sqlite_store, tmp_path), _config()
    raw_by_period = {"201501b": _HEADER + "20150120|000111AAA|ABC|100|ABC CORP|10.0\n20150120|999999ZZZ|ZZZ|9|OTHER CO|3.0\n"}
    read: list[str] = []
    monkeypatch.setattr(ftd, "ftd_periods", lambda *args, **kwargs: sorted(raw_by_period))
    monkeypatch.setattr(ftd, "cached_ftd_periods", lambda cache: set(raw_by_period))
    monkeypatch.setattr(ftd, "read_zip_text", lambda path, log=None: read.append(path.name) or raw_by_period[path.stem.removeprefix("cnsfails")])
    monkeypatch.setattr(module, "load_universe_tickers", lambda ctx: ["ABC"])
    monkeypatch.setattr(module, "scan_form345_cache", lambda cache: SimpleNamespace(owner_pairs=pd.DataFrame()))
    monkeypatch.setattr(module, "build_symbol_tenure", lambda *args: pd.DataFrame())
    monkeypatch.setattr(
        module,
        "build_entity_lineage",
        lambda ctx, *args, **kwargs: (sqlite_store.replace(Tables.entity_lineage, _scope_lineage()), _scope_lineage())[1],
    )
    masters: list[tuple[pd.DataFrame, int]] = []

    def master(ctx: Any, lineage: pd.DataFrame, config_dir: str) -> pd.DataFrame:
        lines = ctx.store.load(Tables.sec_fails_to_deliver_security, optional=True)
        masters.append((lineage, 0 if lines is None else len(lines)))
        return pd.DataFrame()

    monkeypatch.setattr(module, "build_security_master", master)
    monkeypatch.setattr(module, "load_identity", lambda ctx, refresh=False: object())
    monkeypatch.setattr(module, "propagate_identity", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli_mod, "get_config_context", lambda path, **kwargs: (config, context))

    StepExtractIdentity(context, config).run()  # type: ignore[arg-type]
    assert read == [] and masters[0][1] == 0, "without a lineage the FTD capture reads nothing and the master sees no line"

    assert CliRunner().invoke(cli_mod.cli, ["identity", "--only", "capture-ftd", "-F"]).exit_code == 0
    stored = sqlite_store.load(Tables.sec_fails_to_deliver_security)
    assert stored["cusip"].tolist() == ["000111AAA"] and set(read) == {"cnsfails201501b.zip"}

    result = CliRunner().invoke(cli_mod.cli, ["identity", "--only", "master"])
    assert result.exit_code == 0, result.output
    lineage, n_lines = masters[-1]
    assert n_lines == 1 and set(lineage["entity_id"]) == {"E0000000777"}
    assert pd.api.types.is_datetime64_any_dtype(lineage["valid_from"]), "a lone master stage reads the stored lineage with dates parsed"

    print("\n=== SANITY CHECK: D-15 cold start ===")
    print(f"  identity: lineage built, FTD capture read {0} zip(s), master saw 0 line(s)")
    print(f"  identity --only capture-ftd -F: read {read}, stored {stored['cusip'].tolist()} (ZZZ out of scope)")
    print(f"  identity --only master: built from the stored lineage ({len(lineage)} row(s)) and {n_lines} FTD line(s)")
    print("  OK: the documented three-command sequence fills the master on a cold store")
