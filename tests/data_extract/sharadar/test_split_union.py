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
    SPLIT_INTEGER_TOL, SPLIT_MATCH_DAYS, SPLIT_RATIO_CONFLICT_TOL, TranslationReport,
    _is_split_shaped, split_events,
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


@pytest.mark.parametrize("ratio,expected,gap,why", [
    (0.33333, True, 3.33e-6, "DD / HLT / RRD / SBRA 1:3, as Sharadar rounds it to 5 dp"),
    (0.14286, True, 2.86e-6, "MSI / CCEC / UNTD 1:7"),
    (0.16667, True, 3.33e-6, "WINMQ 1:6"),
    (1.272, False, 7.27e-4, "BDX spinoff -- the NEAREST false positive, 7x the tolerance"),
    (0.945, False, 5.6e-4, "SJM merger factor -- the second nearest"),
    (1.067, False, 3.3e-4, "CMCSA spinoff factor (16/15)"),
    (1.025, False, 2.5e-2, "BDX Embecta -- 41/40 exceeds the denominator cap outright"),
    (0.3775, False, 2.5e-3, "WTW exchange ratio"),
])
def test_the_tolerance_sits_in_a_measured_gap(ratio, expected, gap, why):
    """`SPLIT_INTEGER_TOL` is 1e-4 because that is the middle of a MEASURED gap, and this
    test is the gap.

    Sharadar publishes ratios to 5 decimal places, so a genuine 1:3 arrives as `0.33333` --
    3.33e-6 from the truth. At the old 1e-6 that was REJECTED, and rejecting it is what left
    `sharesOutstandingPit` 3x wrong on DD. The nearest thing on the other side is BDX's
    spinoff factor at 7.27e-4. 1e-4 is 30x above the largest true rounding error and 7x below
    the smallest false positive; there is nothing in between to get wrong.
    """
    assert _is_split_shaped(ratio) is expected, f"{ratio} ({why}, gap {gap:.2e})"
    assert (gap < SPLIT_INTEGER_TOL) is expected, (
        f"the stated gap {gap:.2e} must sit on the same side of the tolerance "
        f"{SPLIT_INTEGER_TOL:.0e} as the verdict -- {why}")


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
    test alone would drop a split both vendors independently report.

    Also the negative case for the conflict rule. 2.0 IS split-shaped and 1.998 is not, so
    without a materiality band the rule would "resolve" this pair in Sharadar's favour and
    silently rewrite a ratio the whole corroboration argument says to trust. 0.1% apart is
    rounding, not a conflict."""
    actions = _actions([("GOOGL", "2014-04-03", "split", 2.0)])
    yf_splits = _yf([("GOOGL", "2014-04-03", 1.998)])
    out = split_events(actions, yf_splits)

    assert len(out) == 1 and out.iloc[0]["value"] == 1.998, out.to_dict("records")
    assert abs(1.998 / 2.0 - 1.0) <= SPLIT_RATIO_CONFLICT_TOL, \
        "a 0.1% disagreement must sit INSIDE the materiality band"
    print("\n=== SANITY CHECK: corroboration beats shape ===")
    print("  GOOGL 2014-04-03: sharadar 2.0 + yfinance 1.998 -> kept at 1.998 (yfinance "
          "wins, because close_split steps on ITS date and ratio).")
    print(f"  0.10% apart, inside the {SPLIT_RATIO_CONFLICT_TOL:.0%} conflict band, so the "
          "ratio rule does not fire on rounding. Validated.")


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


def test_a_spinoff_co_dated_row_is_now_kept():
    """A `split` row CO-DATED WITH A SPINOFF IS STILL A SPLIT, and this test used to assert
    the opposite.

    The veto was justified on HON with "`sharesbas` is unchanged across the date
    (316,826,560 -> 316,940,010)". THAT ARGUMENT IS VOID: Sharadar restates `sharesbas`
    retroactively, so it is continuous across a real split by construction and proves nothing.

    Deep history is what discriminates, and it says the split is real -- HON's 2010
    `sharesbas` reads 390,086,318 against an actual ~780M, its 2010 `price` 94.52 against
    ~47, its 2015 `dps` 1.03 against 0.5175 and its 2015 `epsdil` 3.20 against 1.60. Four
    fields restated 2x, with `marketcap` correct at $36.9bn because the two legs cancel.
    All 27 vetoed rows were split-shaped and 24 were reciprocals of small integers.
    """
    actions = _actions([("HON", "2026-06-29", "split", 0.5),
                        ("HON", "2026-06-29", "spinoff", 1.0)])
    out = split_events(actions, pd.DataFrame(columns=["ticker", "date", "ratio"]))

    assert len(out) == 1, f"the real 1:2 reverse split must survive: {out.to_dict('records')}"
    assert out.iloc[0]["value"] == 0.5

    print("\n=== SANITY CHECK: a split co-dated with a spinoff ===")
    print("  HON split 0.5 + spinoff 1.0 on 2026-06-29 -> 1 event kept at x0.5.")
    print("  sharesOutstandingPit now DOUBLES before 2026-06-29, matching the ~780M HON "
          "actually had in 2010. Validated.")


@pytest.mark.parametrize("ticker,when,yf_ratio,sh_ratio,expected,why", [
    ("DD", "2019-06-03", 0.4725, 0.33333, 0.33333,
     "the Corteva spinoff's PRICE factor vs the real 1:3 reverse split"),
    ("HON", "2026-06-29", 0.9535, 0.5, 0.5,
     "the Solstice spinoff's PRICE factor vs the real 1:2 reverse split"),
])
def test_the_split_shaped_ratio_wins_a_material_conflict(ticker, when, yf_ratio, sh_ratio,
                                                         expected, why):
    """One date, two vendors, two DIFFERENT numbers -- and only one of them is a share factor.

    On a corroborated event the DATE is always yfinance's (`close_split` steps on the day
    Yahoo adjusted its own prices). The RATIO is yfinance's only while the two agree: where
    they differ materially they are describing different things, and the SPLIT-SHAPED one is
    what a share count needs. Taking yfinance's number here is what left `sharesOutstandingPit`
    3x wrong on DD and 2x wrong on HON.
    """
    actions = _actions([(ticker, when, "split", sh_ratio),
                        (ticker, when, "spinoff", 1.0)])
    out = split_events(actions, _yf([(ticker, when, yf_ratio)]))

    assert len(out) == 1, f"one event, not two: {out.to_dict('records')}"
    assert out.iloc[0]["value"] == expected, why
    assert pd.Timestamp(out.iloc[0]["date"]).date().isoformat() == when, \
        "the yfinance DATE must survive even when its RATIO does not"

    print(f"\n=== SANITY CHECK: ratio conflict, {ticker} {when} ===")
    print(f"  yfinance x{yf_ratio} (split-shaped: {_is_split_shaped(yf_ratio)}) vs "
          f"sharadar x{sh_ratio} (split-shaped: {_is_split_shaped(sh_ratio)})")
    print(f"  -> kept x{expected} on {when}. {why}. Validated.")




def test_no_split_source_at_all_is_empty_not_an_error():
    """A cold `prices_splits` and no actions must yield an empty list, not raise. The caller
    (`deadjust_splits`) warns and leaves the share block on the vendor basis -- which is now
    the CORRECT default for the feature columns anyway."""
    out = split_events(None, None)
    assert out.empty and list(out.columns) == ["ticker", "date", "value"]

    print("\n=== SANITY CHECK: no sources ===")
    print("  (None, None) -> empty frame with the right columns, no exception. Validated.")
