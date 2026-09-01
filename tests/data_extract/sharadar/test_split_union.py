"""
test_split_union.py  (tests/data_extract/sharadar/)
-------------------------------------------------------------------------------------------
The corroboration rule that decides which share splits are real.

SYNTHETIC known-truth, no DB and no network: the property under test is a DECISION RULE, and
real data can only show that two vendors disagree -- not which one is right. Each fixture row
is a real, named event whose truth was established independently (SEC cover page or the
filer's own disclosure), so the expected answer is known rather than assumed.

WHY the rule exists: `sharadar_actions` is fresh but HOLED -- it misses nine splits yfinance
has -- and de-adjustment can only divide by events it can see. That made the same column sit
on different bases for different tickers in one cross-section (AAPL de-adjusted, GOOGL not),
which for a cross-sectional long/short corrupts the RANKING, not just a level.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.data_extract.utils.fundamentals_sharadar.field_map import (
    SPLIT_MATCH_DAYS, TranslationReport, _is_split_shaped, split_events,
)


def _actions(rows: list[tuple[str, str, str, float]]) -> pd.DataFrame:
    return pd.DataFrame([{"ticker": t, "date": pd.Timestamp(d), "action": a, "value": v,
                          "contraticker": "N/A"} for t, d, a, v in rows])


def _yf(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    return pd.DataFrame([{"ticker": t, "date": pd.Timestamp(d), "ratio": r}
                         for t, d, r in rows])


# --------------------------------------------------------------------------- #
# the ratio shape test                                                        #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("ratio,expected,why", [
    (4.0, True, "AAPL 2020 4:1 forward"),
    (20.0, True, "GOOGL 2022 20:1 forward"),
    (50.0, True, "CMG 2024 50:1 forward"),
    (0.2, True, "AMCR 2026 1:5 REVERSE -- reciprocal of an integer"),
    (0.125, True, "GE 1:8 reverse"),
    (0.05, True, "AIG 1:20 reverse"),
    (1.5, True, "3:2 forward"),
    (1.25, True, "5:4 forward"),
    (1.05, True, "a 5% stock dividend -- 21/20, a real share-count event"),
    (0.945, False, "SJM 2002 -- a merger share-issuance factor"),
    (1.025, False, "BDX 2022-04-01 -- the Embecta SPINOFF factor, from YFINANCE"),
    (1.272, False, "BDX 2026-02-10 -- a second spinoff factor (14/11 to 7e-4, not a split)"),
    (1.067, False, "CMCSA 2026-01-05 -- a spinoff factor from yfinance (16/15)"),
    (0.3775, False, "WTW 2016 -- an exchange ratio (kept only via corroboration)"),
    (0.0012, False, "CCL -- compounds into a 1000x error"),
    (0.0, False, "a zero is not an event"),
])
def test_split_shape_accepts_integers_and_their_reciprocals(ratio, expected, why):
    """A genuine split is an integer or the RECIPROCAL of one. The reciprocal half is not a
    nicety: without it every genuine REVERSE split is classified as an artefact."""
    assert _is_split_shaped(ratio) is expected, f"{ratio} ({why})"


# --------------------------------------------------------------------------- #
# the four union cases                                                        #
# --------------------------------------------------------------------------- #
def test_the_four_union_cases():
    """One fixture, all four branches, each a real named event with a known answer."""
    actions = _actions([
        ("WTW", "2016-01-05", "split", 0.3775),    # in BOTH -> keep despite the odd ratio
        ("AAPL", "2020-08-31", "split", 4.0),      # in BOTH -> keep
        ("SJM", "2002-05-30", "split", 0.945),     # Sharadar ONLY, not split-shaped -> DROP
        ("TMUS", "2013-04-30", "split", 0.5),      # Sharadar ONLY, split-shaped -> keep+warn
    ])
    yf_splits = _yf([
        ("WTW", "2016-01-05", 0.3775),
        ("AAPL", "2020-08-31", 4.0),
        ("GOOGL", "2022-07-18", 20.0),             # yfinance ONLY -> keep (the nine-hole fix)
        ("AMCR", "2026-01-15", 0.2),               # yfinance ONLY, reverse -> keep
    ])
    report = TranslationReport()
    out = split_events(actions, yf_splits, report=report)
    got = {(r.ticker, pd.Timestamp(r.date).date().isoformat()): r.value
           for r in out.itertuples()}

    assert ("WTW", "2016-01-05") in got, "corroborated -> kept even though 0.3775 is odd"
    assert ("AAPL", "2020-08-31") in got
    assert ("GOOGL", "2022-07-18") in got, "the nine-hole fix: yfinance-only must be kept"
    assert ("AMCR", "2026-01-15") in got, "a genuine 1:5 reverse split, yfinance-only"
    assert ("TMUS", "2013-04-30") in got, "uncorroborated but split-shaped -> kept and warned"
    assert ("SJM", "2002-05-30") not in got, "a merger factor must NOT become a share split"
    assert any("SJM" in r for r in report.splits_rejected), "the drop must be reported"

    print("\n=== SANITY CHECK: the four union cases ===")
    for key, value in sorted(got.items()):
        print(f"  KEPT    {key[0]:6s} {key[1]}  x{value}")
    print(f"  DROPPED SJM    2002-05-30  x0.945  (Sharadar-only, not split-shaped)")
    print(f"  {len(got)} events kept, 1 dropped. Rejections reported: "
          f"{report.splits_rejected}")
    print("  Corroborated, yfinance-only and split-shaped-Sharadar-only all survive; the "
          "merger factor does not. Validated.")


def test_an_uncorroborated_yfinance_spinoff_factor_is_dropped():
    """⚠ yfinance is NOT clean either -- its `Stock Splits` column carries SPINOFF factors.

    Measured: trusting yfinance-only events unconditionally injected BDX 2022-04-01 x1.025
    and 2026-02-10 x1.272 into the split list. They compound to 1.304, so every BDX
    `sharesOutstandingPit` before 2022 came out 23% below the SEC cover page -- 67 bad rows
    on a ticker that had none before. The shape test therefore applies to BOTH vendors, and
    only CORROBORATION (an event both report) overrides it."""
    yf_splits = _yf([("BDX", "1996-08-16", 2.0),        # a real 2:1 -> keep
                     ("BDX", "2022-04-01", 1.025),      # Embecta spinoff -> DROP
                     ("BDX", "2026-02-10", 1.272)])     # a second spinoff -> DROP
    report = TranslationReport()
    out = split_events(pd.DataFrame(), yf_splits, report=report)

    assert len(out) == 1, f"only the real split survives: {out.to_dict('records')}"
    assert pd.Timestamp(out.iloc[0]["date"]).date().isoformat() == "1996-08-16"
    assert out.iloc[0]["value"] == 2.0
    assert len(report.splits_rejected) == 2

    print("\n=== SANITY CHECK: yfinance spinoff factors ===")
    print(f"  3 yfinance events -> 1 kept (the 2:1), 2 dropped: {report.splits_rejected}")
    print("  1.025 x 1.272 = 1.304 never reaches a share count. Validated.")


def test_a_corroborated_odd_ratio_beats_the_shape_test():
    """Corroboration OVERRIDES shape, and it has to: GOOGL's real 2014 split is 1.998 in
    yfinance and 2.0 in Sharadar. 1.998 reproduces no small-integer fraction, so the shape
    test alone would drop a split both vendors independently report."""
    actions = _actions([("GOOGL", "2014-04-03", "split", 2.0)])
    yf_splits = _yf([("GOOGL", "2014-04-03", 1.998)])
    out = split_events(actions, yf_splits)

    assert len(out) == 1 and out.iloc[0]["value"] == 1.998, out.to_dict("records")
    print("\n=== SANITY CHECK: corroboration beats shape ===")
    print("  GOOGL 2014-04-03: sharadar 2.0 + yfinance 1.998 -> kept at 1.998 (yfinance "
          "wins, because close_split steps on ITS date and ratio). Validated.")


def test_the_yfinance_date_wins_when_the_two_vendors_disagree():
    """Where both sources carry an event on NEARBY dates, the yfinance date must win.

    Not cosmetic. `close_split` steps on the day YAHOO adjusted its own prices, and the whole
    correctness argument is that the share factor and the price factor CANCEL -- which they
    only do if they step on the same day. A one-day disagreement otherwise leaves a single
    bar where market cap is wrong by the entire split ratio."""
    actions = _actions([("NVDA", "2024-06-07", "split", 10.0)])   # record date
    yf_splits = _yf([("NVDA", "2024-06-10", 10.0)])               # the ex-date Yahoo used

    out = split_events(actions, yf_splits)
    assert len(out) == 1, f"one event, not two: {out.to_dict('records')}"
    assert pd.Timestamp(out.iloc[0]["date"]).date().isoformat() == "2024-06-10"

    print("\n=== SANITY CHECK: date reconciliation ===")
    print(f"  sharadar 2024-06-07 + yfinance 2024-06-10 (3 days apart, tolerance "
          f"{SPLIT_MATCH_DAYS}d) -> ONE event on 2024-06-10.")
    print("  The share factor now steps on the same day close_split does. Validated.")


def test_a_spinoff_co_dated_row_is_still_rejected():
    """The HON trap survives the union: a `split` row co-dated with a `spinoff` is the
    SPINOFF'S PRICE ADJUSTMENT, not a share-count event. HON's own cover page proves it --
    `sharesbas` reads 316,826,560 on 2026-04-23 and 316,940,010 on 2026-07-23, unchanged
    across the date. Applying it would have DOUBLED every HON share count in the history."""
    actions = _actions([("HON", "2026-06-29", "split", 0.5),
                        ("HON", "2026-06-29", "spinoff", 1.0)])
    out = split_events(actions, pd.DataFrame(columns=["ticker", "date", "ratio"]))
    assert out.empty, f"the HON spinoff price factor must not survive: {out.to_dict('records')}"

    print("\n=== SANITY CHECK: the HON spinoff trap ===")
    print("  split 0.5 co-dated with spinoff 1.0 -> 0 events kept. A 100% error on 19 of 20 "
          "rows, avoided. Validated.")


def test_no_split_source_at_all_is_empty_not_an_error():
    """A cold `prices_splits` and no actions must yield an empty list, not raise. The caller
    (`deadjust_splits`) warns and leaves the share block on the vendor basis -- which is now
    the CORRECT default for the feature columns anyway."""
    out = split_events(None, None)
    assert out.empty and list(out.columns) == ["ticker", "date", "value"]

    print("\n=== SANITY CHECK: no sources ===")
    print("  (None, None) -> empty frame with the right columns, no exception. Validated.")
