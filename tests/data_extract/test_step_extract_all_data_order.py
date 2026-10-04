"""`StepExtractAllData` runs downloads -> identity build -> propagation before every domain step."""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from src.data_extract import step_extract_all_data as module
from src.data_extract.step_extract_all_data import StepExtractAllData

DOMAIN_STEPS = ("prices", "institutionals", "fundamentals_sharadar", "fundamentals", "structure", "behavioral")
IDENTITY_STAGE = ("insider_download", "notes_download", "form345_scan", "symbol_tenure", "entity_lineage", "identity_refresh", "propagate")


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

    monkeypatch.setattr(module, "download_insider_transactions", mark("insider_download"))
    monkeypatch.setattr(module, "download_financial_notes", mark("notes_download"))
    monkeypatch.setattr(module, "cache_dir", lambda context, name: Path("cache") / name)

    def form345_scan(cache):
        assert cache == Path("cache") / "sec_insider_transactions"
        return mark("form345_scan", scan)()

    def symbol_tenure(context, given_scan, config_dir):
        assert given_scan is scan
        return mark("symbol_tenure", tenure)()

    def entity_lineage(context, given_tenure, owner_pairs, config_dir, redundant_symbols=frozenset()):
        assert given_tenure is tenure and owner_pairs is scan.owner_pairs and redundant_symbols == {"GOOG"}
        return mark("entity_lineage")()

    def refresh(context, refresh=False):
        assert refresh, "the identity stage must reload the resolver it just rebuilt"
        return mark("identity_refresh", identity)()

    def propagate(context, tickers, **kwargs):
        assert list(tickers) == ["AAA", "BBB"] and kwargs == {"identity": identity}
        return mark("propagate")()

    monkeypatch.setattr(module, "scan_form345_cache", form345_scan)
    monkeypatch.setattr(module, "build_symbol_tenure", symbol_tenure)
    monkeypatch.setattr(module, "build_entity_lineage", entity_lineage)
    monkeypatch.setattr(module, "load_identity", refresh)
    monkeypatch.setattr(module, "propagate_identity", propagate)

    config = SimpleNamespace(
        data_extract=SimpleNamespace(years_history=15, redundant_ticks=["GOOG"]),
        local=SimpleNamespace(paths=SimpleNamespace(insider_transactions="sec_insider_transactions")),
    )
    step = cast(Any, object.__new__(StepExtractAllData))
    step._context = SimpleNamespace(config=config, config_dir="./configs")
    step._config = config
    step._log = logging.getLogger("test_step_extract_all_data_order")
    step._resolve_tickers = lambda: ["AAA", "BBB"]
    for name in DOMAIN_STEPS:
        setattr(step, f"_{name}", SimpleNamespace(run=mark(name)))
    return step


def test_identity_stage_runs_before_every_domain_step(monkeypatch):
    order: list[str] = []
    _step(monkeypatch, order).run()

    assert order == [*IDENTITY_STAGE, *DOMAIN_STEPS], order

    print("\n=== SANITY CHECK: StepExtractAllData order (AC-011, AC-038, AC-040) ===")
    print(f"  identity stage: {' -> '.join(IDENTITY_STAGE)}")
    print(f"  then domain steps: {' -> '.join(DOMAIN_STEPS)}")
    print("  OK: every consumer, and the derived tables they rebuild, run on the propagated lineage")


@pytest.mark.parametrize("boom", ["entity_lineage", "propagate"])
def test_identity_or_propagation_failure_stops_before_any_consumer(monkeypatch, boom):
    order: list[str] = []
    with pytest.raises(RuntimeError, match=boom):
        _step(monkeypatch, order, boom=boom).run()

    assert order[-1] == boom and not set(DOMAIN_STEPS) & set(order), order

    print(f"\n=== SANITY CHECK: {boom} failure ===")
    print(f"  {len(order)} call(s) ran, the last one {boom}; no domain step started")
    print("  OK: a failed identity stage raises out of the run instead of feeding consumers a stale lineage")
