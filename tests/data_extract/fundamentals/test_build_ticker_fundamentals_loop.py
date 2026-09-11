"""`build_ticker_fundamentals`'s per-filing loop had NO test, and that is how three
`NameError`s shipped.

Phase 9b replaced the two-CIK `cutover` object with the N-segment register but left three call
sites referring to the old name. Two were in this function -- `cutover.cik_for(...)` on every
filing, and `cutover.predecessor_cik` in the dedup guard -- and one was latent in the DEF 14A
lister. `NameError` is in `PROGRAMMING_ERRORS`, which `_worker` re-raises by design, so
`fundamentals-facts` aborted the whole 16-ticker run on its first filing, twice, before anyone
noticed. Nothing caught it because no test ever entered this loop.

These tests are deliberately shallow: they patch the walk and the row builder and assert only
what the loop itself owns -- the CIK stamped on a row, and the dedup guard. That is enough to
turn "this function is never executed in CI" into "it is".
"""
from __future__ import annotations

import pandas as pd
import pytest
from types import SimpleNamespace

from src.data_extract.utils.common.registrant import Registrant, Segment
import src.data_extract.utils.fundamentals.fetch_fundamentals_sec as mod


def _filing(accession: str, cik: str, date: str, form: str = "10-Q"):
    return SimpleNamespace(accession_number=accession, cik=cik, form=form,
                           filing_date=pd.Timestamp(date).date())


def _registrants() -> dict[str, Registrant]:
    return {"GOOGL": Registrant(
        ticker="GOOGL", kind="reorganisation",
        segments=(Segment(cik="0001288776", valid_from=None,
                          valid_to=pd.Timestamp("2015-10-02"), evidence="Google Inc"),
                  Segment(cik="0001652044", valid_from=pd.Timestamp("2015-10-02"),
                          valid_to=None, evidence="Alphabet Inc")))}


@pytest.fixture
def patched(monkeypatch):
    """Patch the walk and the row builder; the loop body is what is under test."""
    def _install(filings, rows_per_filing):
        monkeypatch.setattr(mod, "resolve_registrant_filings",
                            lambda *a, **k: list(filings))
        monkeypatch.setattr(mod, "filing_rows",
                            lambda ticker, cik, filing, cat, gics, failures=None:
                            rows_per_filing(ticker, cik, filing))
        monkeypatch.setattr(mod, "is_headcount_form", lambda form: False)
    return _install


def _row(ticker, cik, filing, field="totalRevenue", period_end="2014-12-31"):
    return {"ticker": ticker, "accession_number": filing.accession_number, "field": field,
            "duration_type": "duration", "period_end": pd.Timestamp(period_end), "cik": cik}


def test_each_row_carries_the_cik_that_FILED_it_not_the_roster(patched):
    """The regression. `filing_cik` was `cutover.cik_for(filing.filing_date)` -> NameError on
    the FIRST filing of the FIRST ticker."""
    pre = _filing("0001-pre", "0001288776", "2014-04-24")      # Google Inc, pre-boundary
    post = _filing("0001-post", "0001652044", "2016-04-21")    # Alphabet, post-boundary
    patched([pre, post], lambda t, c, f: [_row(t, c, f, period_end=str(f.filing_date))])

    out = mod.build_ticker_fundamentals(
        "GOOGL", "0001652044", catalogue=None, gics_by_ticker={},
        registrants=_registrants())[mod.Tables.fundamentals_facts]

    assert list(out["cik"]) == ["0001288776", "0001652044"]
    print("\n=== SANITY: provenance survives the boundary ===")
    print(f"  {len(out)} rows, ciks={list(out['cik'])} -- the pre-boundary row keeps Google "
          f"Inc's CIK even though the roster says Alphabet.")


def test_a_ticker_with_no_register_entry_still_stamps_a_cik(patched):
    f = _filing("0001-a", "0000320193", "2024-02-01")
    patched([f], lambda t, c, fl: [_row(t, c, fl)])
    out = mod.build_ticker_fundamentals(
        "AAPL", "0000320193", catalogue=None, gics_by_ticker={},
        registrants={})[mod.Tables.fundamentals_facts]
    assert list(out["cik"]) == ["0000320193"]


def test_the_dedup_overlap_guard_CANNOT_FIRE(patched):
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
    b = _filing("0001-dup", "0001652044", "2016-04-21")        # same accession, both segments
    patched([a, b], lambda t, c, f: [_row(t, c, f)])           # identical PK -> dedup drops one

    out = mod.build_ticker_fundamentals(
        "GOOGL", "0001652044", catalogue=None, gics_by_ticker={},
        registrants=_registrants())[mod.Tables.fundamentals_facts]
    assert len(out) == 1                                        # a row WAS dropped
    print()
    print("=== SANITY: the guard that cannot fire ===")
    print("  2 rows -> 1 after PK dedup, yet distinct accessions stayed 1 -> 1, so the "
          "`nunique()` comparison is unchanged and the ValueError is unreachable.")


def test_an_empty_walk_returns_empty_frames_without_touching_the_guard(patched):
    patched([], lambda t, c, f: [])
    out = mod.build_ticker_fundamentals("GOOGL", "0001652044", catalogue=None,
                                        gics_by_ticker={}, registrants=_registrants())
    assert out[mod.Tables.fundamentals_facts].empty
