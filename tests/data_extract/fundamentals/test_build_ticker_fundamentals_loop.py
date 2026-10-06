"""`parse_fundamentals`, the per-filing parse the EDGAR driver calls for `fundamentals_facts`.

Shallow on purpose: the row builder is patched and only what the parse itself owns is asserted --
the CIK stamped on a row, the primary-key dedup, the single output table, and which filings become
an empty-filing marker (no XBRL) versus a parse failure (unreadable XBRL) versus a retry
(transient SEC failure).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pandas as pd
import pytest

import src.data_extract.utils.fundamentals.fetch_fundamentals_sec as mod
from src.data_extract.utils.common.edgar_driver import EdgarScope, FilingStamp
from src.data_extract.utils.common.sec_io import ParseFailureError, TransientReadError


def _filing(accession: str, cik: str, date: str, form: str = "10-Q", xbrl: Any = None):
    return SimpleNamespace(accession_number=accession, cik=cik, form=form, filing_date=pd.Timestamp(date).date(), xbrl=xbrl or (lambda: object()))


def _row(ticker, cik, filing, field="totalRevenue", period_end="2014-12-31"):
    return {
        "ticker": ticker,
        "accession_number": filing.accession_number,
        "field": field,
        "duration_type": "duration",
        "period_end": pd.Timestamp(period_end),
        "cik": cik,
    }


@pytest.fixture
def patched(monkeypatch):
    """Patch the XBRL row builder with `rows_for(ticker, cik, filing)`."""

    def _install(rows_for):
        monkeypatch.setattr(mod, "rows_from_xbrl", lambda ticker, stamp, xbrl, cat, gics: rows_for(ticker, stamp.cik, stamp.filing))

    return _install


def _parse(ticker: str, roster_cik: str, filing) -> pd.DataFrame:
    stamp = FilingStamp.of(filing, roster_cik)
    out = mod.parse_fundamentals(ticker, roster_cik, stamp, EdgarScope(), catalogue=cast(Any, None), gics_by_ticker={})
    assert set(out) == {mod.Tables.fundamentals_facts}  # no employee-side output
    return out[mod.Tables.fundamentals_facts]


def test_each_row_carries_the_cik_that_filed_it_not_the_roster(patched):
    patched(lambda t, c, f: [_row(t, c, f, period_end=str(f.filing_date))])
    pre = _parse("GOOGL", "0001652044", _filing("0001-pre", "0001288776", "2014-04-24"))  # Google Inc, pre-boundary
    post = _parse("GOOGL", "0001652044", _filing("0001-post", "0001652044", "2016-04-21"))  # Alphabet

    assert list(pre["cik"]) == ["0001288776"] and list(post["cik"]) == ["0001652044"]
    print("\n=== SANITY: provenance survives the boundary ===")
    print("  the pre-boundary filing keeps Google Inc's CIK even though the roster says Alphabet.")


def test_a_filing_without_a_cik_is_stamped_with_the_roster_cik(patched):
    patched(lambda t, c, f: [_row(t, c, f)])
    out = _parse("AAPL", "0000320193", _filing("0001-a", "", "2024-02-01"))
    assert list(out["cik"]) == ["0000320193"]
    print("\nSANITY: a filing exposing no CIK falls back to the roster CIK.")


def test_a_field_tagged_twice_on_one_window_is_one_row(patched):
    """A Postgres upsert touching one PK twice is an error, so the parse dedups on the PK."""
    patched(lambda t, c, f: [_row(t, c, f), _row(t, c, f)])
    out = _parse("AAPL", "0000320193", _filing("0001-dup", "0000320193", "2024-02-01"))
    assert len(out) == 1
    print("\nSANITY: 2 rows on one PK -> 1.")


def test_no_xbrl_unreadable_and_transient_take_three_paths(patched):
    patched(lambda t, c, f: [_row(t, c, f)])
    no_xbrl = _parse("AAPL", "0000320193", _filing("0001-none", "0000320193", "2008-07-31", xbrl=lambda: None))
    assert no_xbrl.empty  # -> one empty-filing marker in the driver
    with pytest.raises(ParseFailureError, match="0001-bad"):
        _parse("AAPL", "0000320193", _filing("0001-bad", "0000320193", "2008-07-31", xbrl=lambda: (_ for _ in ()).throw(ValueError("bad xml"))))
    with pytest.raises(TransientReadError):
        _parse("AAPL", "0000320193", _filing("0001-503", "0000320193", "2008-07-31", xbrl=lambda: (_ for _ in ()).throw(TransientReadError("503"))))
    print("\n=== SANITY: no XBRL vs unreadable vs transient ===")
    print("  no XBRL -> no rows (a marker); unreadable -> ParseFailureError (a marker, logged); 503 -> TransientReadError (listed again).")
