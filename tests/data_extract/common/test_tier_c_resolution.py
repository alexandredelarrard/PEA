"""Tier C: the bulk data sets, which had NO cutover concept at all until phase 9b.

`insider_transactions`, `notes_num`/`notes_text` and `pension_facts` come from SEC bulk data
sets keyed by CIK, and each mapped a row to a ticker through `cik_to_ticker` -- a dict of
exactly one CIK per ticker. Every row a PREDECESSOR filed therefore resolved to nothing and
was dropped, silently: a row for an unknown CIK is indistinguishable from a row for a company
outside the universe, so there was no error, no warning and no gap signal.

Two different repairs, because the two table families combine differently:

  insider (Forms 3/4/5)   UNION   an event happened whoever indexed it -- no date filter
  notes / pension         SPLIT   consolidating -- a row must also fall in the segment
                                  whose CIK filed it, or Apache Corp's subsidiary notes
                                  blend into APA's

Synthetic fixtures: these are resolution rules, not measurements.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.data_extract.utils.common.registrant import (
    Registrant, Segment, drop_rows_outside_segment)
from src.data_extract.utils.common.sec_utils import cik_to_ticker
from src.data_extract.utils.institutionals.fetch_insider_transactions import _filter_universe

#: GOOGL's real chain, measured: Google Inc -> Alphabet Inc on 2015-10-02.
GOOGL = Registrant(ticker="GOOGL", kind="reorganisation", segments=(
    Segment(cik="0001288776", valid_from=None, valid_to=pd.Timestamp("2015-10-02"),
            evidence="Google Inc, which traded as GOOG"),
    Segment(cik="0001652044", valid_from=pd.Timestamp("2015-10-02"), valid_to=None,
            evidence="Alphabet Inc")))

#: VTRS: Mylan N.V. -> Viatris. The predecessor traded as MYL, so the SYMBOL moved too.
VTRS = Registrant(ticker="VTRS", kind="reorganisation", segments=(
    Segment(cik="0001623613", valid_from=None, valid_to=pd.Timestamp("2020-11-07"),
            evidence="Mylan N.V., which traded as MYL"),
    Segment(cik="0001792044", valid_from=pd.Timestamp("2020-11-07"), valid_to=None,
            evidence="Viatris Inc")))

#: APA: the shape that was SAVED by symbol-first resolution, because APA never moved.
APA = Registrant(ticker="APA", kind="reorganisation", segments=(
    Segment(cik="0000006769", valid_from=None, valid_to=pd.Timestamp("2021-03-01"),
            evidence="Apache Corp, which kept filing as a subsidiary until 2024-11-07"),
    Segment(cik="0001841666", valid_from=pd.Timestamp("2021-03-01"), valid_to=None,
            evidence="APA Corp")))

REGISTRANTS = {"GOOGL": GOOGL, "VTRS": VTRS, "APA": APA}

ROSTER = pd.DataFrame({"cik": ["0001652044", "0001792044", "0001841666", "0000320193"],
                       "ticker": ["GOOGL", "VTRS", "APA", "AAPL"]})


# --------------------------------------------------------------------------- #
# ⚠ The regression fixture the whole of tier C exists for                      #
# --------------------------------------------------------------------------- #
def test_a_predecessor_cik_resolves_to_the_ticker(monkeypatch):
    """⚠ THIS TEST FAILED BEFORE PHASE 9b AND PASSES AFTER, which is the only reason it
    proves anything. A test that was green all along would not have caught this.

    Measured 2026-09-09, the two tickers where the trading SYMBOL moved with the registrant
    so even the insider fetcher's symbol-first path could not save them:

        GOOGL  insider_transactions starts 2015-10-08   (boundary 2015-10-02, was GOOG)
        VTRS   insider_transactions starts 2020-11-16   (was MYL)

    Before the repair, `cik_to_ticker` held only the CURRENT CIK, so Google Inc's and Mylan's
    bulk rows mapped to `NaN` and were filtered out as out-of-universe.
    """
    monkeypatch.setattr("src.data_extract.utils.common.sec_utils.load_registrants",
                        lambda *a, **k: REGISTRANTS)
    mapping = cik_to_ticker(ROSTER)

    assert mapping["0001288776"] == "GOOGL", "Google Inc's rows still resolve to nothing"
    assert mapping["0001623613"] == "VTRS", "Mylan N.V.'s rows still resolve to nothing"
    assert mapping["0000006769"] == "APA"
    assert mapping["0000320193"] == "AAPL", "a ticker with no entry must be untouched"

    print("\n=== SANITY CHECK: predecessor CIKs resolve to their ticker ===")
    print(f"  roster rows in: {len(ROSTER)}   map entries out: {len(mapping)}")
    print("  0001288776 -> GOOGL (Google Inc), 0001623613 -> VTRS (Mylan N.V.)")
    print("  OK: these two mapped to NaN before phase 9b and their rows were dropped.")


def test_a_register_entry_for_a_ticker_outside_this_run_is_ignored(monkeypatch):
    """A `-t AAPL` run must not gain GOOGL's predecessor CIKs. Widening the map beyond the
    universe being walked would route a company's rows to a ticker this run is not writing."""
    monkeypatch.setattr("src.data_extract.utils.common.sec_utils.load_registrants",
                        lambda *a, **k: REGISTRANTS)
    mapping = cik_to_ticker(pd.DataFrame({"cik": ["0000320193"], "ticker": ["AAPL"]}))
    assert mapping == {"0000320193": "AAPL"}
    print("\n=== SANITY CHECK: the map stays scoped to the run's universe ===")
    print("  a one-ticker run gains no register CIKs. Validated.")


# --------------------------------------------------------------------------- #
# UNION vs SPLIT                                                               #
# --------------------------------------------------------------------------- #
def _bulk(rows) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["cik", "ticker", "filed"])


def test_a_split_table_drops_a_predecessor_row_filed_after_the_boundary():
    """`notes` and `pension` are CONSOLIDATING, and this is the Apache case in miniature.
    Apache Corp filed its own reports until 2024-11-07 as a subsidiary; those disclosures
    describe a subsidiary and must not be stored as APA's."""
    df = _bulk([("0000006769", "APA", "2019-05-01"),      # parent-era, keep
                ("0000006769", "APA", "2022-05-01"),      # subsidiary-era, DROP
                ("0001841666", "APA", "2022-05-01")])     # the real parent, keep

    out = drop_rows_outside_segment(df, cik_col="cik", ticker_col="ticker",
                                    filed_col="filed", registrants=REGISTRANTS)

    assert len(out) == 2
    assert list(out["filed"]) == ["2019-05-01", "2022-05-01"]
    assert list(out["cik"]) == ["0000006769", "0001841666"]
    print("\n=== SANITY CHECK: the SPLIT date filter on a bulk table ===")
    print(f"  {len(df)} rows in -> {len(out)} kept; Apache Corp's 2022 subsidiary row dropped.")


def test_a_union_table_keeps_that_same_row():
    """Forms 3/4/5 are EVENTS, so no date filter runs at all: a predecessor's Form 4 filed
    after the boundary is still a real insider transaction in this issuer's security. That is
    the 4,287-vs-4 trade, taken deliberately."""
    df = pd.DataFrame({"ticker": [None, None], "issuer_cik": ["0000006769", "0000006769"],
                       "filed": ["2019-05-01", "2022-05-01"]})
    out = _filter_universe(df, {"APA"}, {"0000006769": "APA"})

    assert len(out) == 2, "the union path must not apply a date filter"
    print("\n=== SANITY CHECK: the UNION path keeps a post-boundary event ===")
    print("  both Apache-CIK Form 4 rows kept, including the 2022 one. Validated.")


def test_symbol_first_still_wins_where_the_symbol_never_moved():
    """The APA shape. Symbol-first resolution is right and recovers rows the CIK map would
    miss; it just could not carry a ticker whose symbol moved too."""
    df = pd.DataFrame({"ticker": ["APA", None], "issuer_cik": ["0000006769", "0000006769"],
                       "filed": ["2019-05-01", "2019-06-01"]})
    out = _filter_universe(df, {"APA"}, {"0000006769": "APA"})

    assert list(out["ticker"]) == ["APA", "APA"]
    print("\n=== SANITY CHECK: symbol-first resolution is preserved ===")
    print("  one row resolved by symbol, one by the CIK fallback. Validated.")


def test_a_ticker_with_no_register_entry_is_untouched():
    """~449 of 491 tickers. The filter must be a no-op for them, not merely harmless."""
    df = _bulk([("0000320193", "AAPL", "2019-05-01"), ("0000320193", "AAPL", "2024-05-01")])
    out = drop_rows_outside_segment(df, cik_col="cik", ticker_col="ticker",
                                    filed_col="filed", registrants=REGISTRANTS)
    pd.testing.assert_frame_equal(out, df)
    print("\n=== SANITY CHECK: an unregistered ticker passes through unchanged ===")
    print("  frame is identical, not merely equal in length. Validated.")


def test_an_unparseable_filed_date_is_dropped_not_silently_kept():
    """A row whose `filed` cannot be parsed belongs to no segment, and for a SPLIT table
    "belongs to no segment" must mean dropped. Keeping it would attribute a subsidiary's
    disclosure to the parent on the strength of a malformed date."""
    df = _bulk([("0000006769", "APA", "not-a-date"), ("0000006769", "APA", "2019-05-01")])
    out = drop_rows_outside_segment(df, cik_col="cik", ticker_col="ticker",
                                    filed_col="filed", registrants=REGISTRANTS)
    assert list(out["filed"]) == ["2019-05-01"]
    print("\n=== SANITY CHECK: an unparseable filed date is dropped ===")
    print("  1 of 2 rows kept. Validated.")


@pytest.mark.parametrize("boundary_offset", [-1, 0, 1])
def test_the_bulk_split_uses_the_same_boundary_convention_as_the_filing_split(boundary_offset):
    """Strictly-before / on-or-after, to the day, exactly as `Segment.covers` does for
    filings. Two different conventions in two layers of one repair would put the seam in two
    places, and only one of them would be documented."""
    boundary = pd.Timestamp("2015-10-02")
    filed = (boundary + pd.Timedelta(days=boundary_offset)).date().isoformat()
    df = _bulk([("0001288776", "GOOGL", filed), ("0001652044", "GOOGL", filed)])

    out = drop_rows_outside_segment(df, cik_col="cik", ticker_col="ticker",
                                    filed_col="filed", registrants=REGISTRANTS)

    expected = "0001288776" if boundary_offset < 0 else "0001652044"
    assert list(out["cik"]) == [expected]
    print(f"\n=== SANITY CHECK: bulk row filed {filed} (boundary {boundary.date()}) ===")
    print(f"  kept the row from {expected}. Validated.")
