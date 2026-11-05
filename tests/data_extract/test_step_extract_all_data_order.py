"""`StepExtractAllData` runs the spec's decision-1 order (AC-001): zip downloads -> `StepExtractIdentity` (captures, anchor,
CIK / symbol / CUSIP history, propagation) -> zip ingests -> prices (+ earnings surprises) -> institutionals -> Sharadar -> fundamentals -> `StepExtractEdgar`
-> `StepExtractLlm` (built only then; DEF 14A, 8-K votes; employees stay on their own command) -> behavioral."""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pandas as pd
import pytest

from src.data_extract import step_extract_all_data as all_data_module
from src.data_extract.step_extract_all_data import StepExtractAllData
from src.data_extract.transformers import step_extract_identity as identity_module
from src.data_extract.transformers import step_extract_universe as universe_module
from src.data_extract.transformers import step_extract_zips as zips_module
from tests.support.propagation import stub_propagation
from tests.support.universe import seed_universe

PRE_EDGAR_STEPS = ("prices", "institutionals", "fundamentals_sharadar", "fundamentals")
EDGAR_AND_LLM = ("edgar", "llm_init", "def14a_llm", "votes_llm")
DOMAIN_STEPS = (*PRE_EDGAR_STEPS, *EDGAR_AND_LLM, "behavioral")
UNIVERSE_STAGE = ("sec_tickers", "sharadar_tickers", "sharadar_actions", "sec_submissions", "universe_sync")
IDENTITY_STAGE = (
    "insider_download",
    "notes_download",
    "ftd_download",
    "dei_capture",
    "ftd_capture",
    "form345_scan",
    "symbol_tenure",
    "company_anchor",
    "cik_history",
    "universe_cik_symbol",
    "universe_cusip",
    "identity_refresh",
    "propagate",
)
ZIP_INGESTS = ("insider_ingest", "notes_ingest", "ftd_ingest")


def _step(monkeypatch, store, order: list[str], *, boom: str | None = None) -> Any:
    scan = SimpleNamespace(owner_pairs=object())
    tenure = object()

    def mark(name: str, result: object = None):
        def call(*args, **kwargs):
            order.append(name)
            if name == boom:
                raise RuntimeError(f"simulated {name} failure")
            return result

        return call

    monkeypatch.setattr(universe_module, "download_sec_company_tickers", mark("sec_tickers", 0))
    monkeypatch.setattr(universe_module, "fetch_sharadar_tickers", mark("sharadar_tickers"))
    monkeypatch.setattr(universe_module, "fetch_sharadar_actions", mark("sharadar_actions", 0))
    monkeypatch.setattr(universe_module, "refresh_sec_cik_profile", mark("sec_submissions", 0))
    monkeypatch.setattr(universe_module, "cache_dir", lambda context, name: Path("cache") / name)
    monkeypatch.setattr(universe_module, "sync", mark("universe_sync", 0))
    monkeypatch.setattr(zips_module, "cache_dir", lambda context, name: Path("cache") / name)
    monkeypatch.setattr(zips_module, "notes_periods", lambda context, years_history, today: ["2026q1"])
    monkeypatch.setattr(zips_module, "download_insider_transactions", mark("insider_download", []))
    monkeypatch.setattr(zips_module, "download_notes_periods", mark("notes_download", 0))
    monkeypatch.setattr(zips_module, "download_fails_to_deliver", mark("ftd_download", 0))
    monkeypatch.setattr(zips_module, "fetch_insider_transactions", mark("insider_ingest", 0))
    monkeypatch.setattr(zips_module, "ingest_financial_notes", mark("notes_ingest", 0))
    monkeypatch.setattr(zips_module, "fetch_fails_to_deliver", mark("ftd_ingest", 0))
    monkeypatch.setattr(identity_module, "cache_dir", lambda context, name: Path("cache") / name)
    monkeypatch.setattr(identity_module, "pending_dei_periods", lambda context, cache, full: mark("dei_capture", [])())
    monkeypatch.setattr(identity_module, "capture_ftd_lines", lambda context, plan, scope, changed: mark("ftd_capture", (pd.DataFrame(), []))())

    def form345_scan(cache):
        assert cache == Path("cache") / "sec_insider_transactions"
        return mark("form345_scan", scan)()

    def symbol_tenure(context, given_scan, manual, rejected):
        assert given_scan is scan
        return mark("symbol_tenure", tenure)()

    lineage = object()
    history = object()

    def cik_history(step):
        assert step._tenure is tenure and step._scan is scan, "the CIK history reads this run's tenure and Form 3/4/5 scan"
        step._cik_history = history
        return mark("cik_history", history)()

    def universe_cik_symbol(step):
        assert step._cik_history is history
        step._symbols = SimpleNamespace(rows=lineage)
        return mark("universe_cik_symbol", step._symbols)()

    def universe_cusip(step):
        assert step._symbols.rows is lineage, "the CUSIP history is derived from the lineage just built"
        return mark("universe_cusip")()

    def refresh(context):
        # the identity stage reloads the resolver from the tables it just published
        inputs = SimpleNamespace(lineage=None, tenure=None, master=None, co_registrants=frozenset(), mergers=())
        return mark("identity_refresh", inputs)()

    def propagate(scopes, as_of, every):
        today = pd.Timestamp.today().normalize()
        assert sorted(scopes) == ["AAA", "BBB"] and not every and as_of == today
        return mark("propagate", [])()

    monkeypatch.setattr(identity_module, "scan_form345_cache", form345_scan)
    monkeypatch.setattr(identity_module, "build_symbol_tenure", symbol_tenure)
    monkeypatch.setattr(identity_module.StepExtractIdentity, "run_company_anchor", lambda step: mark("company_anchor")())
    monkeypatch.setattr(identity_module.StepExtractIdentity, "run_cik_history", cik_history)
    monkeypatch.setattr(identity_module.StepExtractIdentity, "run_symbol_history", universe_cik_symbol)
    monkeypatch.setattr(identity_module.StepExtractIdentity, "run_lifecycle", lambda step: None)
    monkeypatch.setattr(identity_module.StepExtractIdentity, "run_publish", lambda step: None)
    monkeypatch.setattr(identity_module.StepExtractIdentity, "run_cusip_history", universe_cusip)
    stub_propagation(monkeypatch, identity_module)
    monkeypatch.setattr(identity_module, "identity_inputs", refresh)
    monkeypatch.setattr(identity_module, "recheck_tickers", propagate)

    config = SimpleNamespace(
        data_extract=SimpleNamespace(years_history=15, sharadar_years_history=31),
        local=SimpleNamespace(
            paths=SimpleNamespace(
                insider_transactions="sec_insider_transactions",
                financial_notes="notes",
                fails_deliver="sec_fails_to_deliver",
                sec_submissions="sec_submissions",
            )
        ),
    )
    step = cast(Any, object.__new__(StepExtractAllData))
    step._context = SimpleNamespace(config=config, config_dir="./configs", store=store, universe=seed_universe(store, ["AAA", "BBB"]))
    step._config = config
    step._log = logging.getLogger("test_step_extract_all_data_order")
    for name in (*PRE_EDGAR_STEPS, "behavioral"):
        setattr(step, f"_{name}", SimpleNamespace(run=mark(name)))

    def edgar_step(context, config, options):
        assert options.tickers == ["AAA", "BBB"]
        return SimpleNamespace(run=mark("edgar"))

    monkeypatch.setattr(all_data_module, "StepExtractEdgar", edgar_step)

    def llm_step(context, config, options):
        assert options.tickers == ["AAA", "BBB"]
        order.append("llm_init")
        return SimpleNamespace(run_def14a=mark("def14a_llm"), run_8k_votes=mark("votes_llm"))

    monkeypatch.setattr(all_data_module, "StepExtractLlm", llm_step)
    return step


def test_identity_stage_runs_before_every_domain_step(monkeypatch, sqlite_store):
    order: list[str] = []
    _step(monkeypatch, sqlite_store, order).run()

    assert order == [*UNIVERSE_STAGE, *IDENTITY_STAGE, *ZIP_INGESTS, *DOMAIN_STEPS], order

    print("\n=== SANITY CHECK: StepExtractAllData order (AC-011, AC-038, AC-040) ===")
    print(f"  universe: {' -> '.join(UNIVERSE_STAGE)}, then identity stage: {' -> '.join(IDENTITY_STAGE)}")
    print(f"  then zip ingests: {' -> '.join(ZIP_INGESTS)}")
    print(f"  then domain steps: {' -> '.join(DOMAIN_STEPS)}")
    print("  OK: decision-1 order; the LLM step is built after the EDGAR step; every consumer reads the propagated lineage")


@pytest.mark.parametrize("boom", ["universe_cik_symbol", "universe_cusip", "propagate"])
def test_identity_or_propagation_failure_stops_before_any_consumer(monkeypatch, sqlite_store, boom):
    order: list[str] = []
    with pytest.raises(RuntimeError, match=boom):
        _step(monkeypatch, sqlite_store, order, boom=boom).run()

    assert order[-1] == boom and not set(DOMAIN_STEPS + ZIP_INGESTS) & set(order), order

    print(f"\n=== SANITY CHECK: {boom} failure ===")
    print(f"  {len(order)} call(s) ran, the last one {boom}; no zip ingest or domain step started")
    print("  OK: a failed identity stage raises out of the run instead of feeding consumers a stale lineage")
