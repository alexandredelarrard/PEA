"""`StepExtractAllData` runs zip downloads -> `StepExtractIdentity` (captures, build, propagation) -> zip ingests before every
domain step; `StepExtractEdgar` runs after fundamentals, then the DEF 14A LLM pass and the 8-K votes, then behavioral."""

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
from src.data_extract.transformers import step_extract_zips as zips_module

PRE_EDGAR_STEPS = ("prices", "institutionals", "fundamentals_sharadar", "fundamentals")
EDGAR_AND_LLM = ("edgar", "def14a_llm", "votes_llm")
DOMAIN_STEPS = (*PRE_EDGAR_STEPS, *EDGAR_AND_LLM, "behavioral")
IDENTITY_STAGE = (
    "insider_download",
    "notes_download",
    "statements_download",
    "ftd_download",
    "sec_tickers",
    "dei_capture",
    "ftd_capture",
    "form345_scan",
    "symbol_tenure",
    "entity_lineage",
    "security_master",
    "identity_refresh",
    "propagate",
)
ZIP_INGESTS = ("insider_ingest", "notes_ingest", "statements_ingest", "ftd_ingest")


def _step(monkeypatch, order: list[str], *, boom: str | None = None) -> Any:
    identity = object()
    scan = SimpleNamespace(owner_pairs=object())
    tenure = object()

    def mark(name: str, result: object = None):
        def call(*args, **kwargs):
            order.append(name)
            if name == boom:
                raise RuntimeError(f"simulated {name} failure")
            return result

        return call

    monkeypatch.setattr(zips_module, "cache_dir", lambda context, name: Path("cache") / name)
    monkeypatch.setattr(zips_module, "load_universe_tickers", lambda context: ["AAA", "BBB"])
    monkeypatch.setattr(zips_module, "notes_periods", lambda context, years_history, today: ["2026q1"])
    monkeypatch.setattr(zips_module, "download_insider_transactions", mark("insider_download", []))
    monkeypatch.setattr(zips_module, "download_notes_periods", mark("notes_download", 0))
    monkeypatch.setattr(zips_module, "download_financial_statements", mark("statements_download", 0))
    monkeypatch.setattr(zips_module, "download_fails_to_deliver", mark("ftd_download", 0))
    monkeypatch.setattr(zips_module, "download_sec_company_tickers", mark("sec_tickers", 0))
    monkeypatch.setattr(zips_module, "fetch_insider_transactions", mark("insider_ingest", 0))
    monkeypatch.setattr(zips_module, "ingest_financial_notes", mark("notes_ingest", 0))
    monkeypatch.setattr(zips_module, "fetch_financial_statements", mark("statements_ingest", 0))
    monkeypatch.setattr(zips_module, "fetch_fails_to_deliver", mark("ftd_ingest", 0))
    monkeypatch.setattr(identity_module, "cache_dir", lambda context, name: Path("cache") / name)
    monkeypatch.setattr(identity_module, "load_universe_tickers", lambda context: ["AAA", "BBB"])
    monkeypatch.setattr(identity_module, "capture_pending_dei", lambda context, cache, full: mark("dei_capture")())
    monkeypatch.setattr(identity_module, "capture_ftd_cusips", lambda context, cache, years, full, as_of: mark("ftd_capture")())

    def form345_scan(cache):
        assert cache == Path("cache") / "sec_insider_transactions"
        return mark("form345_scan", scan)()

    def symbol_tenure(context, given_scan, config_dir):
        assert given_scan is scan
        return mark("symbol_tenure", tenure)()

    lineage = object()

    def entity_lineage(context, given_tenure, owner_pairs, config_dir, approved_rekeys=frozenset(), redundant_symbols=frozenset()):
        assert given_tenure is tenure and owner_pairs is scan.owner_pairs and redundant_symbols == {"GOOG"}
        return mark("entity_lineage", lineage)()

    def security_master(context, given_lineage, config_dir):
        assert given_lineage is lineage, "the security master is derived from the lineage just built"
        return mark("security_master")()

    def refresh(context, refresh=False):
        assert refresh, "the identity stage must reload the resolver it just rebuilt"
        return mark("identity_refresh", identity)()

    def propagate(context, tickers, **kwargs):
        today = pd.Timestamp.today().normalize()
        assert list(tickers) == ["AAA", "BBB"] and kwargs["identity"] is identity and kwargs["all_tickers"] == ["AAA", "BBB"]
        assert not kwargs["dry_run"] and not kwargs["every_ticker"] and kwargs["as_of"] == today
        return mark("propagate")()

    monkeypatch.setattr(identity_module, "scan_form345_cache", form345_scan)
    monkeypatch.setattr(identity_module, "build_symbol_tenure", symbol_tenure)
    monkeypatch.setattr(identity_module, "build_entity_lineage", entity_lineage)
    monkeypatch.setattr(identity_module, "build_security_master", security_master)
    monkeypatch.setattr(identity_module, "load_identity", refresh)
    monkeypatch.setattr(identity_module, "propagate_identity", propagate)

    config = SimpleNamespace(
        data_extract=SimpleNamespace(years_history=15, redundant_ticks=["GOOG"]),
        local=SimpleNamespace(
            paths=SimpleNamespace(
                insider_transactions="sec_insider_transactions",
                financial_notes="notes",
                financial_statements="statements",
                fails_deliver="sec_fails_to_deliver",
            )
        ),
    )
    step = cast(Any, object.__new__(StepExtractAllData))
    step._context = SimpleNamespace(config=config, config_dir="./configs")
    step._config = config
    step._log = logging.getLogger("test_step_extract_all_data_order")
    step._resolve_tickers = lambda: ["AAA", "BBB"]
    for name in (*PRE_EDGAR_STEPS, "behavioral"):
        setattr(step, f"_{name}", SimpleNamespace(run=mark(name)))

    def edgar_step(context, config, options):
        assert options.tickers == ["AAA", "BBB"]
        return SimpleNamespace(run=mark("edgar"))

    monkeypatch.setattr(all_data_module, "StepExtractEdgar", edgar_step)

    def llm_step(context, config, options):
        assert options.tickers == ["AAA", "BBB"]
        return SimpleNamespace(run_def14a=mark("def14a_llm"), run_8k_votes=mark("votes_llm"))

    monkeypatch.setattr(all_data_module, "StepExtractLlm", llm_step)
    return step


def test_identity_stage_runs_before_every_domain_step(monkeypatch):
    order: list[str] = []
    _step(monkeypatch, order).run()

    assert order == [*IDENTITY_STAGE, *ZIP_INGESTS, *DOMAIN_STEPS], order

    print("\n=== SANITY CHECK: StepExtractAllData order (AC-011, AC-038, AC-040) ===")
    print(f"  identity stage: {' -> '.join(IDENTITY_STAGE)}")
    print(f"  then zip ingests: {' -> '.join(ZIP_INGESTS)}")
    print(f"  then domain steps: {' -> '.join(DOMAIN_STEPS)}")
    print("  OK: every consumer, and the derived tables they rebuild, run on the propagated lineage")


@pytest.mark.parametrize("boom", ["entity_lineage", "security_master", "propagate"])
def test_identity_or_propagation_failure_stops_before_any_consumer(monkeypatch, boom):
    order: list[str] = []
    with pytest.raises(RuntimeError, match=boom):
        _step(monkeypatch, order, boom=boom).run()

    assert order[-1] == boom and not set(DOMAIN_STEPS + ZIP_INGESTS) & set(order), order

    print(f"\n=== SANITY CHECK: {boom} failure ===")
    print(f"  {len(order)} call(s) ran, the last one {boom}; no zip ingest or domain step started")
    print("  OK: a failed identity stage raises out of the run instead of feeding consumers a stale lineage")
