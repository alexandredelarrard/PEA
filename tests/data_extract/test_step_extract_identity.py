"""`StepExtractIdentity`: `run()` is the eight-stage flow, the captures land before the builders with one Form 3/4/5 scan,
per-stage failures, and the D-15 cold start."""

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
from tests.support.universe import seed_universe, universe_of

FLOW = (
    "run_capture_evidence",
    "run_company_anchor",
    "run_cik_history",
    "run_symbol_history",
    "run_cusip_history",
    "run_lifecycle",
    "run_publish",
    "run_propagate",
)


def _config() -> SimpleNamespace:
    paths = SimpleNamespace(insider_transactions="insider", financial_notes="notes", fails_deliver="sec_fails_to_deliver")
    return SimpleNamespace(local=SimpleNamespace(paths=paths), data_extract=SimpleNamespace(years_history=1))


def _context(store: Any = None, tmp_path: Path | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        store=store if store is not None else SimpleNamespace(load=lambda table, **kwargs: None),
        log=logging.getLogger("test.identity_step"),
        paths={"DATA_STORE": tmp_path},
        config_dir="configs",
        universe=universe_of(["AAA", "BBB"]),
    )


def _stub_operations(monkeypatch: pytest.MonkeyPatch, order: list[str], boom: str | None = None) -> None:
    """Every operation behind the stages records its call (`boom` raises); the anchor and CIK history are stubbed whole."""
    scan = SimpleNamespace(owner_pairs=pd.DataFrame())

    def mark(name: str, result: object = None):
        def call(*args: Any, **kwargs: Any) -> object:
            order.append(name)
            if name == boom:
                raise RuntimeError(f"simulated {name} failure")
            return result

        return call

    monkeypatch.setattr(module, "cache_dir", lambda context, name: Path(name))
    monkeypatch.setattr(module, "pending_dei_periods", mark("dei_periods", ["2026q1"]))
    monkeypatch.setattr(module, "capture_dei_period", mark("dei", 0))
    monkeypatch.setattr(module, "capture_ftd_lines", mark("ftd", (pd.DataFrame(), [])))
    monkeypatch.setattr(module, "save_ftd_lines", lambda context, rows, periods: 0)
    monkeypatch.setattr(module, "scan_form345_cache", mark("scan", scan))
    monkeypatch.setattr(module, "build_symbol_tenure", mark("tenure", pd.DataFrame()))
    monkeypatch.setattr(StepExtractIdentity, "run_company_anchor", lambda self: order.append("anchor"))

    def cik_history(self: StepExtractIdentity) -> object:
        order.append("cik_history")
        self._cik_history = history = object()  # type: ignore[assignment]
        return history

    monkeypatch.setattr(StepExtractIdentity, "run_cik_history", cik_history)

    def symbols(self: StepExtractIdentity) -> object:
        mark("symbols")()
        self._symbols = SimpleNamespace(rows=pd.DataFrame())  # type: ignore[assignment]
        return self._symbols

    monkeypatch.setattr(StepExtractIdentity, "run_symbol_history", symbols)
    monkeypatch.setattr(StepExtractIdentity, "run_publish", lambda self: order.append("publish"))
    monkeypatch.setattr(StepExtractIdentity, "run_cusip_history", lambda self: mark("cusips", pd.DataFrame())())
    monkeypatch.setattr(module, "load_identity", mark("refresh", object()))
    monkeypatch.setattr(module, "propagate_identity", mark("propagate"))
    monkeypatch.setattr(module, "load_stored_lineage", lambda context: pd.DataFrame())


def test_run_is_the_flow_of_eight_named_stages(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-009: `run()` calls the eight sub-runs by name, in order, and nothing else."""
    calls: list[str] = []
    monkeypatch.setattr(module, "cache_dir", lambda context, name: Path(name))
    for name in FLOW:
        monkeypatch.setattr(StepExtractIdentity, name, lambda self, n=name: calls.append(n))

    StepExtractIdentity(_context(), _config()).run()  # type: ignore[arg-type]

    assert tuple(calls) == FLOW, calls
    print("\n=== SANITY CHECK: StepExtractIdentity.run is the flow ===")
    print(f"  {' -> '.join(calls)}")
    print("  OK: evidence, anchor, CIK, symbol and CUSIP history, lifecycle, publish, propagate -- in the REQ-009 order")


def test_run_captures_first_then_builds_and_propagates_with_one_scan(monkeypatch: pytest.MonkeyPatch) -> None:
    order: list[str] = []
    _stub_operations(monkeypatch, order)

    StepExtractIdentity(_context(), _config()).run()  # type: ignore[arg-type]

    expected = ["dei_periods", "dei", "ftd", "scan", "tenure", "anchor", "cik_history", "symbols", "cusips", "publish", "refresh", "propagate"]
    assert order == expected, order
    assert order.count("scan") == 1, "the tenure and the CIK history share one Form 3/4/5 scan"
    print("\n=== SANITY CHECK: StepExtractIdentity.run operations ===")
    print(f"  {' -> '.join(order)}; one scan shared by the tenure and the CIK history")
    print("  OK: the captures land before the builders read their evidence, and propagation runs on the rebuilt tables")


@pytest.mark.parametrize("boom", ["ftd", "symbols", "cusips", "propagate"])
def test_a_failing_stage_raises_from_its_own_method_only(monkeypatch: pytest.MonkeyPatch, boom: str) -> None:
    owner = {"ftd": "run_capture_evidence", "symbols": "run_symbol_history", "cusips": "run_cusip_history", "propagate": "run_propagate"}
    order: list[str] = []
    _stub_operations(monkeypatch, order, boom=boom)
    step = StepExtractIdentity(_context(), _config())  # type: ignore[arg-type]

    for method in FLOW:
        if method == owner[boom]:
            with pytest.raises(RuntimeError, match=boom):
                getattr(step, method)()
        else:
            getattr(step, method)()
    with pytest.raises(RuntimeError, match=boom):
        StepExtractIdentity(_context(), _config()).run()  # type: ignore[arg-type]

    print(f"\n=== SANITY CHECK: {boom} failure isolation ===")
    print(f"  {owner[boom]} raised alone; the other seven sub-runs completed when called alone; run() stopped at it")


def test_options_reach_the_capture_and_the_propagation(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, dict[str, Any]] = {}
    _stub_operations(monkeypatch, [])
    monkeypatch.setattr(module, "pending_dei_periods", lambda context, cache, full: calls.setdefault("dei", {"full": full}) and [])
    monkeypatch.setattr(
        module, "propagate_identity", lambda context, tickers, **kwargs: calls.setdefault("propagate", {"tickers": tickers, **kwargs})
    )
    options = IdentityOptions(tickers=["AAA"], full=True, dry_run=True, as_of=pd.Timestamp("2026-10-30"))
    step = StepExtractIdentity(_context(), _config(), options)  # type: ignore[arg-type]

    step.run_capture_evidence()
    step.run_propagate()

    assert calls["dei"] == {"full": True}
    propagate = calls["propagate"]
    assert propagate["tickers"] == ["AAA"] and "all_tickers" not in propagate
    assert propagate["dry_run"] and propagate["every_ticker"] and propagate["as_of"] == pd.Timestamp("2026-10-30")
    print("\n=== SANITY CHECK: identity options ===")
    print("  --full reaches the capture; -t, --full, --dry-run and --as-of reach the propagation")


def test_cold_start_sequence_captures_ftd_once_a_lineage_exists(sqlite_store: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """D-15: `identity` (no lineage: the FTD capture is a no-op) -> `identity --only capture-evidence -F` -> `--only cusip-history`."""
    tables = [
        Tables.universe_cik_symbol,
        Tables.universe_cusip,
        Tables.sec_fails_to_deliver_security,
        Tables.sec_fails_to_deliver,
        Tables.symbol_tenure,
    ]
    create_canonical(sqlite_store.engine, tables)
    context, config = _context(sqlite_store, tmp_path), _config()
    context.universe = seed_universe(sqlite_store, ["ABC"], rows={"ABC": {"cik": "0000000777"}})
    raw_by_period = {"201501b": _HEADER + "20150120|000111AAA|ABC|100|ABC CORP|10.0\n20150120|999999ZZZ|ZZZ|9|OTHER CO|3.0\n"}
    read: list[str] = []
    monkeypatch.setattr(ftd, "ftd_periods", lambda *args, **kwargs: sorted(raw_by_period))
    monkeypatch.setattr(ftd, "cached_ftd_periods", lambda cache: set(raw_by_period))
    monkeypatch.setattr(ftd, "read_zip_text", lambda path, log=None: read.append(path.name) or raw_by_period[path.stem.removeprefix("cnsfails")])
    monkeypatch.setattr(module, "pending_dei_periods", lambda *args: [])
    monkeypatch.setattr(module, "scan_form345_cache", lambda cache: SimpleNamespace(owner_pairs=pd.DataFrame()))
    monkeypatch.setattr(module, "build_symbol_tenure", lambda *args: pd.DataFrame())
    monkeypatch.setattr(StepExtractIdentity, "run_company_anchor", lambda self: None)
    monkeypatch.setattr(StepExtractIdentity, "run_cik_history", lambda self: setattr(self, "_cik_history", object()))
    monkeypatch.setattr(
        StepExtractIdentity,
        "run_symbol_history",
        lambda self: (
            sqlite_store.replace(Tables.universe_cik_symbol, _scope_lineage()),
            setattr(self, "_symbols", SimpleNamespace(rows=_scope_lineage())),
        ),
    )
    monkeypatch.setattr(StepExtractIdentity, "run_publish", lambda self: None)
    lines_seen: list[int] = []
    lineages: list[pd.DataFrame] = []
    prepare, view = module.prepare_observations, module.lineage_view
    monkeypatch.setattr(module, "prepare_observations", lambda observations: lines_seen.append(len(observations)) or prepare(observations))
    monkeypatch.setattr(module, "lineage_view", lambda lineage, roster, tape: lineages.append(lineage) or view(lineage, roster, tape))
    monkeypatch.setattr(module, "load_identity", lambda ctx, refresh=False: object())
    monkeypatch.setattr(module, "propagate_identity", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli_mod, "get_config_context", lambda path, **kwargs: (config, context))

    StepExtractIdentity(context, config).run()  # type: ignore[arg-type]
    assert read == [] and lines_seen == [], "without a lineage the FTD capture reads nothing and the CUSIP stage sees no line"

    assert CliRunner().invoke(cli_mod.cli, ["identity", "--only", "capture-evidence", "-F"]).exit_code == 0
    stored = sqlite_store.load(Tables.sec_fails_to_deliver_security)
    assert stored["cusip"].tolist() == ["000111AAA"] and set(read) == {"cnsfails201501b.zip"}

    result = CliRunner().invoke(cli_mod.cli, ["identity", "--only", "cusip-history"])
    assert result.exit_code == 0, result.output
    lineage, master = lineages[-1], sqlite_store.load(Tables.universe_cusip)
    assert lines_seen == [1] and set(lineage["company_id"]) == {"E0000000777"}
    assert pd.api.types.is_datetime64_any_dtype(lineage["valid_from"]), "a lone CUSIP stage reads the stored lineage with dates parsed"
    assert master[["cusip", "canonical_company", "lineage_role"]].values.tolist() == [["000111AAA", "ABC", "canonical_current"]], master

    print("\n=== SANITY CHECK: D-15 cold start ===")
    print("  identity: lineage built, FTD capture read 0 zip(s), the CUSIP stage saw no line (table left unchanged)")
    print(f"  identity --only capture-evidence -F: read {read}, stored {stored['cusip'].tolist()} (ZZZ out of scope)")
    print(
        f"  identity --only cusip-history: built from the stored lineage ({len(lineage)} row(s)) and 1 FTD line -> {master['cusip'].tolist()} canonical for ABC"
    )
    print("  OK: the documented three-command sequence fills universe_cusip on a cold store")
