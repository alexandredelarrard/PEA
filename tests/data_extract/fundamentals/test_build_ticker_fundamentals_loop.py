"""`build_ticker_fundamentals`'s per-filing loop had NO test, and that is how three
`NameError`s shipped.

Phase 9b replaced the two-CIK `cutover` object with the N-segment register but left three call
sites referring to the old name. Two were in this function -- `cutover.cik_for(...)` on every
filing, and `cutover.predecessor_cik` in the dedup guard -- and one was latent in the DEF 14A
lister. `NameError` is in `PROGRAMMING_ERRORS`, which `run_per_ticker` re-raises by design, so
`fundamentals-facts` aborted the whole 16-ticker run on its first filing, twice, before anyone
noticed. Nothing caught it because no test ever entered this loop.

These tests are deliberately shallow: they patch the walk and the row builder and assert only
what the loop itself owns -- the CIK stamped on a row, and the dedup guard. That is enough to
turn "this function is never executed in CI" into "it is".
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any, cast

import pandas as pd
import pytest

import src.data_extract.utils.fundamentals.fetch_fundamentals_sec as mod
from src.data_extract.utils.common.edgar_driver import EdgarScope
from src.data_extract.utils.common.registrant import Registrant, Segment


def _filing(accession: str, cik: str, date: str, form: str = "10-Q"):
    return SimpleNamespace(accession_number=accession, cik=cik, form=form, filing_date=pd.Timestamp(date).date())


def _xbrl_filing(accession: str, *, error: Exception | None = None):
    def xbrl():
        if error is not None:
            raise error
        return None

    return SimpleNamespace(
        accession_number=accession,
        cik="0000320193",
        form="10-Q",
        filing_date=pd.Timestamp("2008-07-31").date(),
        xbrl=xbrl,
    )


def _registrants() -> dict[str, Registrant]:
    return {
        "GOOGL": Registrant(
            ticker="GOOGL",
            kind="reorganisation",
            segments=(
                Segment(cik="0001288776", valid_from=None, valid_to=pd.Timestamp("2015-10-02"), evidence="Google Inc"),
                Segment(cik="0001652044", valid_from=pd.Timestamp("2015-10-02"), valid_to=None, evidence="Alphabet Inc"),
            ),
        )
    }


@pytest.fixture
def patched(monkeypatch):
    """Patch the walk and the row builder; the loop body is what is under test."""

    def _install(filings, rows_per_filing):
        monkeypatch.setattr(mod, "resolve_registrant_filings", lambda *a, **k: list(filings))
        monkeypatch.setattr(mod, "filing_rows", lambda ticker, stamp, cat, gics, **kwargs: rows_per_filing(ticker, stamp.cik, stamp.filing))

    return _install


def _row(ticker, cik, filing, field="totalRevenue", period_end="2014-12-31"):
    return {
        "ticker": ticker,
        "accession_number": filing.accession_number,
        "field": field,
        "duration_type": "duration",
        "period_end": pd.Timestamp(period_end),
        "cik": cik,
    }


def test_each_row_carries_the_cik_that_filed_it_not_the_roster(patched):
    """The regression. `filing_cik` was `cutover.cik_for(filing.filing_date)` -> NameError on
    the FIRST filing of the FIRST ticker."""
    pre = _filing("0001-pre", "0001288776", "2014-04-24")  # Google Inc, pre-boundary
    post = _filing("0001-post", "0001652044", "2016-04-21")  # Alphabet, post-boundary
    patched([pre, post], lambda t, c, f: [_row(t, c, f, period_end=str(f.filing_date))])

    out = mod.build_ticker_fundamentals("GOOGL", "0001652044", catalogue=cast(Any, None), gics_by_ticker={}, scope=EdgarScope(None, _registrants()))[
        mod.Tables.fundamentals_facts
    ]

    assert list(out["cik"]) == ["0001288776", "0001652044"]
    print("\n=== SANITY: provenance survives the boundary ===")
    print(f"  {len(out)} rows, ciks={list(out['cik'])} -- the pre-boundary row keeps Google Inc's CIK even though the roster says Alphabet.")


def test_a_ticker_with_no_register_entry_still_stamps_a_cik(patched):
    f = _filing("0001-a", "0000320193", "2024-02-01")
    patched([f], lambda t, c, fl: [_row(t, c, fl)])
    out = mod.build_ticker_fundamentals("AAPL", "0000320193", catalogue=cast(Any, None), gics_by_ticker={}, scope=EdgarScope(None, {}))[
        mod.Tables.fundamentals_facts
    ]
    assert list(out["cik"]) == ["0000320193"]


def test_the_dedup_overlap_guard_cannot_fire(patched):
    """The second stale site -- and writing this test showed the guard is unfalsifiable.

    It compares `accession_number.nunique()` before and after `drop_duplicates(subset=PK)`,
    but `accession_number` is ITSELF part of the PK
    (`ticker, accession_number, field, duration_type, period_end`), so dedup can only ever
    collapse rows WITHIN one accession and can never remove an accession. The guard's own
    comment says it exists to prove the two segment walks are disjoint; it cannot observe
    that, and it never could -- it was dead code carrying a `NameError` in its message.

    The real check lives where the overlap would actually happen: the SPLIT branch of
    `resolve_registrant_filings` warns when one accession is kept by two segments. This test
    pins the dead branch so nobody re-derives confidence from it.
    """
    from src.data_store.schema import Tables

    assert "accession_number" in Tables.fundamentals_facts.pk

    a = _filing("0001-dup", "0001288776", "2014-04-24")
    b = _filing("0001-dup", "0001652044", "2016-04-21")  # same accession, both segments
    patched([a, b], lambda t, c, f: [_row(t, c, f)])  # identical PK -> dedup drops one

    out = mod.build_ticker_fundamentals("GOOGL", "0001652044", catalogue=cast(Any, None), gics_by_ticker={}, scope=EdgarScope(None, _registrants()))[
        mod.Tables.fundamentals_facts
    ]
    assert len(out) == 1  # a row WAS dropped
    print()
    print("=== SANITY: the guard that cannot fire ===")
    print(
        "  2 rows -> 1 after PK dedup, yet distinct accessions stayed 1 -> 1, so the "
        "`nunique()` comparison is unchanged and the ValueError is unreachable."
    )


def test_an_empty_walk_returns_empty_frames_without_touching_the_guard(patched):
    patched([], lambda t, c, f: [])
    out = mod.build_ticker_fundamentals("GOOGL", "0001652044", catalogue=cast(Any, None), gics_by_ticker={}, scope=EdgarScope(None, _registrants()))
    assert out[mod.Tables.fundamentals_facts].empty


def test_resumed_ticker_with_only_legacy_no_xbrl_is_complete_no_new(monkeypatch, caplog):
    """Stored modern facts make a legacy-only remainder an idempotent no-op, not a lost history."""
    seen: dict[str, frozenset[str]] = {}

    def resolve(*args, **kwargs):
        seen["done"] = kwargs["done_accessions"]
        return [_xbrl_filing("0000320193-08-000123")]

    monkeypatch.setattr(mod, "resolve_registrant_filings", resolve)
    caplog.set_level(logging.INFO, logger=mod.__name__)

    out = mod.build_ticker_fundamentals(
        "AAPL",
        "0000320193",
        done_accessions=frozenset({"0000320193-24-000123"}),
        catalogue=cast(Any, None),
        gics_by_ticker={},
        scope=EdgarScope(None, {}),
    )

    message = caplog.text.lower()
    print("\n=== SANITY: resumed legacy-only walk is complete ===")
    print(f"  done={seen['done']}; facts={len(out[mod.Tables.fundamentals_facts])}; log={message.strip()}")
    assert seen["done"] == frozenset({"0000320193-24-000123"})
    assert out[mod.Tables.fundamentals_facts].empty
    assert "no new facts" in message
    assert "1 already stored" in message
    assert "1 no xbrl" in message
    assert "0 unreadable" in message
    assert "whole history is missing" not in message


def test_cold_ticker_with_only_eligible_no_xbrl_filings_is_incomplete(monkeypatch, caplog):
    filings = [_xbrl_filing("0000320193-08-000123"), _xbrl_filing("0000320193-08-000456")]
    monkeypatch.setattr(mod, "resolve_registrant_filings", lambda *args, **kwargs: filings)
    caplog.set_level(logging.INFO, logger=mod.__name__)

    print("\n=== SANITY: a cold all-no-XBRL walk is incomplete ===")
    print("  expected: no persisted coverage + 2 eligible filings without XBRL raises a ticker-level failure")
    with pytest.raises(RuntimeError, match="(?i)no usable xbrl"):
        mod.build_ticker_fundamentals("AAPL", "0000320193", catalogue=cast(Any, None), gics_by_ticker={}, scope=EdgarScope(None, {}))


def test_no_xbrl_and_unreadable_xbrl_are_reported_separately(monkeypatch, caplog):
    filings = [
        _xbrl_filing("0000320193-08-000123"),
        _xbrl_filing("0000320193-08-000456", error=ValueError("bad xml")),
    ]
    monkeypatch.setattr(mod, "resolve_registrant_filings", lambda *args, **kwargs: filings)
    caplog.set_level(logging.INFO, logger=mod.__name__)

    try:
        mod.build_ticker_fundamentals("AAPL", "0000320193", catalogue=cast(Any, None), gics_by_ticker={}, scope=EdgarScope(None, {}))
    except RuntimeError:
        pass

    message = caplog.text.lower()
    print("\n=== SANITY: no-XBRL is distinct from unreadable XBRL ===")
    print(f"  log={message.strip()}")
    assert "1 no xbrl" in message
    assert "1 unreadable" in message


def test_facts_builder_has_no_employee_side_output(patched):
    filing = _filing("0001-a", "0000320193", "2024-02-01", form="10-K")
    patched([filing], lambda t, c, f: [_row(t, c, f)])
    out = mod.build_ticker_fundamentals("AAPL", "0000320193", catalogue=cast(Any, None), gics_by_ticker={}, scope=EdgarScope(None, {}))
    assert set(out) == {mod.Tables.fundamentals_facts}
    print("\nSANITY: fundamentals facts own no employee side output.")
