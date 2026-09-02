"""
test_level_basis.py  (tests/data_aggregate/)
-------------------------------------------------------------------------------------------
`S(d)` -- the spinoff price factor a share count does not carry.

SYNTHETIC known-truth, no DB and no network. This is parsing math on a closed formula, not an
economic property, so the repo's "feature tests use real data" rule points the other way here:
every fixture below is a real, named event whose factor was measured against the live tables
first (see `reports/planning/active-tasks/2026-09-01-spinoff-level-basis/before.md`), and the
expected value is that measurement.

The property that matters most is the NEGATIVE one. ~89% of cells must come out exactly 1.0,
because a factor that is 1.0000000000000002 on a ticker that never spun anything off would
move every downstream digest and make "this change is targeted" unprovable. So the clean cases
assert `== 1.0` bit-exactly, not `approx`.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.constants.constants import DEFAULT_CONFIG_DIR
from src.data_aggregate.utils.common.level_basis import (
    LEVEL_SNAP_TOL, _vintage_multiplier, apply_level_bugfix, apply_return_seams,
    apply_split_vintage, describe, level_factor, load_bugfix)
from src.data_aggregate.utils.common.pit import daily_market_cap
from src.data_extract.utils.fundamentals_sharadar.field_map import split_events

#: A daily grid that straddles every fixture event below. Business days only, so the frame is
#: shaped like a real `cube_part_prices` slice without being large.
INDEX = pd.DatetimeIndex(pd.bdate_range("1995-01-02", "2026-12-31"), name="date")


def _yf(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    return pd.DataFrame([{"ticker": t, "date": pd.Timestamp(d), "ratio": r}
                         for t, d, r in rows])


def _actions(rows: list[tuple[str, str, str, float]]) -> pd.DataFrame:
    return pd.DataFrame([{"ticker": t, "date": pd.Timestamp(d), "action": a, "value": v,
                          "contraticker": "N/A"} for t, d, a, v in rows])


def _at(factor: pd.DataFrame, ticker: str, when: str) -> float:
    """`S` for one ticker on the last trading day at or before `when`."""
    return float(factor[ticker].loc[:pd.Timestamp(when)].iloc[-1])


# --------------------------------------------------------------------------- #
# the negative case -- the one that protects 89% of the table                 #
# --------------------------------------------------------------------------- #
def test_a_ticker_whose_two_sources_agree_is_exactly_one():
    """AAPL. Every yfinance factor is a genuine split, so it appears in BOTH products and
    cancels term-by-term.

    Asserted as `== 1.0` on the raw floats, not `approx`: 4 x 7 x 2 x 2 x 2 / (4 x 7 x 2 x 2
    x 2) is only exactly 1.0 if both products are accumulated in the same order, which is why
    `_suffix_factor` builds them right-to-left over dates sorted ascending. `approx` would
    pass on a version of this code that silently moves every AAPL market cap by 1 ulp."""
    events = [("AAPL", "1987-06-16", 2.0), ("AAPL", "2000-06-21", 2.0),
              ("AAPL", "2005-02-28", 2.0), ("AAPL", "2014-06-09", 7.0),
              ("AAPL", "2020-08-31", 4.0)]
    yf = _yf(events)
    genuine = split_events(pd.DataFrame(), yf)
    factor = level_factor(INDEX, ["AAPL"], yf, genuine)

    assert (factor["AAPL"].to_numpy() == 1.0).all(), (
        "a ticker with no spinoff must be BIT-identical to 1.0, not merely close: "
        f"{sorted(set(factor['AAPL'])) [:5]}")

    print("\n=== SANITY CHECK: two agreeing sources ===")
    print(f"  AAPL, 5 genuine splits in both sources -> S == 1.0 on all "
          f"{len(factor):,} dates, bit-exactly.")
    print("  The 89% of the table with no spinoff is untouched by this change. Validated.")


def test_no_events_at_all_is_one_everywhere_and_does_not_raise():
    """A cold `prices_splits` must degrade to "no adjustment", not to a crash or a NaN."""
    factor = level_factor(INDEX, ["KO", "JNJ"], pd.DataFrame(), pd.DataFrame())

    assert factor.shape == (len(INDEX), 2)
    assert (factor.to_numpy() == 1.0).all()
    assert list(factor.columns) == ["JNJ", "KO"], "columns are sorted, for determinism"

    print("\n=== SANITY CHECK: empty sources ===")
    print(f"  (empty, empty) -> {factor.shape[0]:,} x {factor.shape[1]} frame of exactly "
          "1.0, no exception. Validated.")


# --------------------------------------------------------------------------- #
# the positive cases -- each a measured factor                                #
# --------------------------------------------------------------------------- #
def test_a_yfinance_only_spinoff_factor_becomes_S():
    """FDX, verbatim from `prices_splits`: two genuine 2:1 splits and x1.241 on 2026-06-01
    (the FedEx Freight separation). `sharadar_actions` has NO row for the last one, and 1.241
    is not split-shaped, so it stays in the numerator alone.

    Measured on the live panel: FDX 2020-12-17 reads `close_split` 235.5036 against Sharadar's
    `price` 292.26, and 292.26 / 235.5036 = 1.2410. Market cap goes $62.425bn -> $77.470bn,
    against Sharadar's $77.470bn."""
    yf = _yf([("FDX", "1996-11-05", 2.0), ("FDX", "1999-05-07", 2.0),
              ("FDX", "2026-06-01", 1.241)])
    genuine = split_events(pd.DataFrame(), yf)
    factor = level_factor(INDEX, ["FDX"], yf, genuine)

    assert sorted(genuine["value"]) == [2.0, 2.0], \
        f"1.241 must NOT reach the denominator: {genuine.to_dict('records')}"
    assert _at(factor, "FDX", "1995-06-30") == pytest.approx(1.241), \
        "the two 2:1 splits cancel; only the spinoff factor is left"
    assert _at(factor, "FDX", "2020-12-17") == pytest.approx(1.241), "the landmark row"
    assert _at(factor, "FDX", "2026-05-29") == pytest.approx(1.241), "the day before"
    assert _at(factor, "FDX", "2026-06-01") == 1.0, "the event date is already restated"

    print("\n=== SANITY CHECK: FDX, a yfinance-only spinoff factor ===")
    print(f"  S = {_at(factor, 'FDX', '2020-12-17'):.4f} before 2026-06-01, "
          f"{_at(factor, 'FDX', '2026-06-01'):.4f} from it onward.")
    print(f"  2020-12-17 market cap 235.5036 x 265,070,592 x S = "
          f"${235.5036 * 265_070_592 * _at(factor, 'FDX', '2020-12-17') / 1e9:.2f}bn "
          "against Sharadar's $77.47bn. Validated.")


def test_stacked_factors_compound_while_a_real_split_cancels():
    """GE, verbatim from `prices_splits` -- the case where BOTH halves of the ratio matter.

    Six events: 2:1 (1997), 3:1 (2000), x1.04 (Wabtec 2019), 1:8 REVERSE (2021), x1.281
    (GE HealthCare 2023) and x1.253 (GE Vernova 2024). Three are genuine share events and
    cancel; the three spinoff factors do not, and a pre-1997 row carries their product:

        1.04 x 1.281 x 1.253 = 1.669297   -- the factor measured on the live panel

    A set SUBTRACTION would leave the 1:8 reverse split in the numerator and put GE's whole
    pre-2021 history out by 8x. Only the ratio of two products is right."""
    yf = _yf([("GE", "1997-05-12", 2.0), ("GE", "2000-05-08", 3.0),
              ("GE", "2019-02-26", 1.04), ("GE", "2021-08-02", 0.125),
              ("GE", "2023-01-04", 1.281), ("GE", "2024-04-02", 1.253)])
    genuine = split_events(pd.DataFrame(), yf)
    factor = level_factor(INDEX, ["GE"], yf, genuine)

    assert sorted(genuine["value"]) == [0.125, 2.0, 3.0], \
        f"the 1:8 reverse split is a REAL share event: {genuine.to_dict('records')}"
    assert _at(factor, "GE", "2005-05-06") == pytest.approx(1.04 * 1.281 * 1.253)
    assert _at(factor, "GE", "2005-05-06") == pytest.approx(1.669297, abs=1e-5), \
        "the measured GE factor, from before.md"
    assert _at(factor, "GE", "2021-04-27") == pytest.approx(1.605093, abs=1e-5), \
        "after 2019 the 1.04 has dropped out -- also measured"
    assert _at(factor, "GE", "2024-06-28") == 1.0

    print("\n=== SANITY CHECK: GE, three spinoff factors over a real reverse split ===")
    print(f"  PROD(yfinance) / PROD(genuine) = "
          f"(2 x 3 x 1.04 x 0.125 x 1.281 x 1.253) / (2 x 3 x 0.125)")
    print(f"  2005-05-06: {_at(factor, 'GE', '2005-05-06'):.6f} (measured 1.669297); "
          f"2021-04-27: {_at(factor, 'GE', '2021-04-27'):.6f} (measured 1.605093); "
          f"2024-06-28: {_at(factor, 'GE', '2024-06-28'):.1f}")
    print("  The 1:8 reverse split cancels instead of leaking an 8x. Validated.")


def test_one_date_two_different_values_is_a_ratio_not_a_set_difference():
    """HON -- the case that decides the whole formula.

    2026-06-29 carries an event in BOTH sources with DIFFERENT values: yfinance 0.9535 (the
    Solstice spinoff's price factor) and Sharadar 0.5 (the co-dated 1:2 reverse split). A set
    SUBTRACTION would cancel the date and lose both; only the ratio of the two products is
    right. Measured on the live panel: the price leg reads 0.4712 = 1/2.1223.

    Note what `split_events` contributes here: it resolves the 91% ratio conflict in favour of
    the split-shaped 0.5, so the DENOMINATOR gets 0.5 while the NUMERATOR keeps 0.9535."""
    yf = _yf([("HON", "1997-09-16", 2.0), ("HON", "2016-10-03", 1.0053282396702523),
              ("HON", "2018-10-01", 1.011), ("HON", "2018-10-29", 1.032),
              ("HON", "2025-10-30", 1.061), ("HON", "2026-06-29", 0.9535)])
    actions = _actions([("HON", "1997-09-16", "split", 2.0),
                        ("HON", "2026-06-29", "split", 0.5),
                        ("HON", "2026-06-29", "spinoff", 1.0)])
    genuine = split_events(actions, yf)
    factor = level_factor(INDEX, ["HON"], yf, genuine)

    assert sorted(genuine["value"]) == [0.5, 2.0], \
        f"the denominator is the two REAL share events: {genuine.to_dict('records')}"
    s = _at(factor, "HON", "1996-12-31")
    assert s == pytest.approx(2.12228, abs=1e-4), "the measured HON factor"
    assert s == pytest.approx((2.0 * 1.0053282396702523 * 1.011 * 1.032 * 1.061 * 0.9535)
                              / (2.0 * 0.5))
    assert 1.0 / s == pytest.approx(0.4712, abs=1e-4), "the measured price leg"

    print("\n=== SANITY CHECK: HON, one date in both sources with different values ===")
    print(f"  PROD(yfinance) = 2 x 1.00533 x 1.011 x 1.032 x 1.061 x 0.9535 = "
          f"{2.0 * 1.00533 * 1.011 * 1.032 * 1.061 * 0.9535:.5f}")
    print(f"  PROD(genuine)  = 2 x 0.5 = {2.0 * 0.5:.5f}")
    print(f"  S = {s:.5f}, so 1/S = {1 / s:.4f} -- the measured price leg is 0.4712. "
          "Validated.")


def test_mnst_cancels_and_is_therefore_not_fixed_here():
    """MNST IS NOT S'S JOB, and this test is where that is written down.

    Yahoo PUBLISHES the 2026-08-11 x2 in its own splits feed. The event is split-shaped, so it
    lands in the genuine list too, appears in both products and cancels to exactly 1.0 -- and
    that cancellation is CORRECT, because a split does move the share count. What Yahoo failed
    to do is apply its own split to most of its own quotes, which leaves the series on two
    bases at once. That is a per-bar defect and `S` is a per-date multiplier, so the repair
    lives in `apply_split_vintage` instead. See the register tests at the foot of this file."""
    yf = _yf([("MNST", "2023-03-28", 2.0), ("MNST", "2026-08-11", 2.0)])
    genuine = split_events(pd.DataFrame(), yf)
    factor = level_factor(INDEX, ["MNST"], yf, genuine)

    assert len(genuine) == 2, "both are split-shaped, so both reach the denominator"
    assert (factor["MNST"].to_numpy() == 1.0).all()

    print("\n=== SANITY CHECK: MNST is NOT fixed by S ===")
    print("  Both x2 events appear in both products -> S == 1.0 everywhere.")
    print("  Its 122 rows are repaired by `apply_split_vintage`, not here. Validated.")


# --------------------------------------------------------------------------- #
# the snap, and the log line                                                  #
# --------------------------------------------------------------------------- #
def test_float_noise_is_snapped_but_a_real_factor_is_not():
    """The snap must catch accumulated float noise and NOTHING else.

    The smallest genuine factor in the universe is HON's 1.00533, five thousand times the
    snap tolerance, so there is no risk of erasing a real adjustment."""
    yf = _yf([("X", "2010-01-04", 1.0 + LEVEL_SNAP_TOL / 10),
              ("Y", "2010-01-04", 1.00533)])
    factor = level_factor(INDEX, ["X", "Y"], yf, pd.DataFrame())

    assert _at(factor, "X", "2005-01-03") == 1.0, "below the tolerance -> snapped"
    assert _at(factor, "Y", "2005-01-03") == pytest.approx(1.00533), "a real factor survives"
    assert _at(factor, "Y", "2005-01-03") != 1.0

    print("\n=== SANITY CHECK: the snap ===")
    print(f"  1 + {LEVEL_SNAP_TOL / 10:.1e} -> exactly 1.0; 1.00533 (HON's smallest real "
          f"factor, {0.00533 / LEVEL_SNAP_TOL:.0e}x the tolerance) -> kept.")
    print("  Noise is erased, signal is not. Validated.")


def test_describe_names_the_biggest_factors_by_abs_log():
    """The diagnostic `StepCubePrices` logs. Ranked by `|log S|`, so a 0.5 and a 2.0 read as
    equally large -- which they are, and a plain `max` would miss every reverse case."""
    yf = _yf([("BIG", "2020-01-02", 4.0), ("SMALL", "2020-01-02", 1.05)])
    line = describe(level_factor(INDEX, ["BIG", "SMALL", "CLEAN"], yf, pd.DataFrame()))

    assert "BIG x4.0000" in line and line.index("BIG") < line.index("SMALL")
    assert "2 of 3 tickers" in line

    print("\n=== SANITY CHECK: the log line ===")
    print(f"  {line}")
    print("  Ranked by |log S|, so the reverse cases are not hidden below the forward ones. "
          "Validated.")


def test_the_index_and_universe_shape_the_output():
    """The frame must align with the other price frames with no reindex at the call site: a
    row per trading date, a column per universe ticker, even for tickers with no events."""
    idx = pd.DatetimeIndex(pd.bdate_range("2020-01-01", "2020-01-10"), name="date")
    factor = level_factor(idx, ["B", "A"], _yf([("A", "2020-01-08", 2.0)]),
                          pd.DataFrame())

    assert factor.index.equals(idx) and list(factor.columns) == ["A", "B"]
    assert factor.index.name == "date" and factor.columns.name == "ticker"
    assert not np.isnan(factor.to_numpy()).any(), "never NaN -- 1.0 is the identity"
    assert factor.loc[pd.Timestamp("2020-01-07"), "A"] == 2.0
    assert factor.loc[pd.Timestamp("2020-01-08"), "A"] == 1.0

    print("\n=== SANITY CHECK: frame shape ===")
    print(f"  {factor.shape[0]} dates x {factor.shape[1]} tickers, columns sorted "
          f"{list(factor.columns)}, no NaN, B (no events) all 1.0. Validated.")


# --------------------------------------------------------------------------- #
# the consumption contract -- `pit.daily_market_cap`                          #
# --------------------------------------------------------------------------- #
def _one_ticker_history(ticker: str, shares: float) -> pd.DataFrame:
    return pd.DataFrame({"ticker": [ticker], "as_of": [pd.Timestamp("2019-01-02")],
                         "sharesOutstanding": [shares]})


def test_market_cap_cannot_be_computed_without_stating_a_basis():
    """`level_factor` is REQUIRED and KEYWORD-ONLY, so a call site added later cannot inherit
    the wrong basis silently.

    This is the mechanism the whole plan leans on for coverage: with a default of `None`, a
    missed consumer returns a number that is quietly 24% low on FDX and 40% low on GE. With no
    default it is a `TypeError` at import-time-adjacent speed."""
    fund = _one_ticker_history("FDX", 265_070_592.0)
    close = pd.DataFrame({"FDX": [235.5036]}, index=pd.DatetimeIndex(["2020-12-17"]))

    with pytest.raises(TypeError, match="level_factor"):
        daily_market_cap(fund, close)              # type: ignore[call-arg]

    print("\n=== SANITY CHECK: the required-kwarg contract ===")
    print("  daily_market_cap(fund, close) -> TypeError naming `level_factor`.")
    print("  A consumer added in six months cannot silently pick the wrong basis. Validated.")


def test_a_unit_factor_is_bit_identical_to_no_factor():
    """The 89% guarantee, at the point of use rather than in the factor.

    `S == 1.0` must give the SAME FLOAT as `level_factor=None`, not a value one ulp away:
    every control-cohort digest in Phase 5 depends on it, and `x * 1.0 == x` only holds
    exactly because 1.0 is the multiplicative identity in IEEE-754 -- which is precisely why
    the snap in `level_factor` exists."""
    idx = pd.DatetimeIndex(pd.bdate_range("2019-01-02", periods=40))
    fund = _one_ticker_history("AAPL", 16_800_000_000.0)
    close = pd.DataFrame({"AAPL": np.linspace(150.0, 190.0, len(idx))}, index=idx)
    ones = pd.DataFrame(1.0, index=idx, columns=["AAPL"])

    without = daily_market_cap(fund, close, level_factor=None)
    with_one = daily_market_cap(fund, close, level_factor=ones)
    pd.testing.assert_frame_equal(with_one, without, check_exact=True, check_dtype=True)

    print("\n=== SANITY CHECK: S == 1 changes nothing ===")
    print(f"  {len(idx)} dates x 1 ticker, bit-identical to the pre-S result "
          f"(check_exact=True).")
    print("  Every ticker without a spinoff is untouched at the point of use too. Validated.")


def test_fdx_market_cap_matches_sharadar_after_the_factor():
    """The landmark, end to end through the function the 7 call sites use.

    Live values: `close_split` 235.5036, `sharesOutstanding` 265,070,592 (both vendors agree),
    `S` 1.241, Sharadar's own `marketcap` $77.470bn."""
    when = pd.DatetimeIndex(["2020-12-17"])
    fund = _one_ticker_history("FDX", 265_070_592.0)
    close = pd.DataFrame({"FDX": [235.5036]}, index=when)
    factor = pd.DataFrame({"FDX": [1.241]}, index=when)
    sharadar_bn = 77.470

    before = float(daily_market_cap(fund, close, level_factor=None).iloc[0, 0]) / 1e9
    after = float(daily_market_cap(fund, close, level_factor=factor).iloc[0, 0]) / 1e9

    assert before == pytest.approx(62.425, abs=0.01), "the defect, reproduced"
    assert after == pytest.approx(sharadar_bn, rel=0.01), "within 1% of Sharadar"
    assert abs(before / sharadar_bn - 1) > 0.19, "and it really was 19% low before"

    print("\n=== SANITY CHECK: FDX 2020-12-17 market cap ===")
    print(f"  without S : ${before:.3f}bn   ({before / sharadar_bn - 1:+.2%} vs Sharadar)")
    print(f"  with S    : ${after:.3f}bn   ({after / sharadar_bn - 1:+.2%} vs Sharadar)")
    print(f"  Sharadar  : ${sharadar_bn:.3f}bn. Validated.")


def test_the_factor_is_aligned_not_assumed():
    """`level_factor` spans the whole universe; the market cap spans the filing history's
    intersection with the price frame. A ticker present in one and not the other must not
    NULL a series or raise -- a missing factor means "no adjustment", never "no market cap"."""
    idx = pd.DatetimeIndex(pd.bdate_range("2019-01-02", periods=10))
    fund = pd.concat([_one_ticker_history("AAA", 1e9), _one_ticker_history("BBB", 2e9)])
    close = pd.DataFrame({"AAA": 10.0, "BBB": 20.0}, index=idx)
    # only AAA has a factor, and it carries a stray extra column
    factor = pd.DataFrame({"AAA": 2.0, "ZZZ": 9.0}, index=idx)

    mcap = daily_market_cap(fund, close, level_factor=factor)
    assert float(mcap["AAA"].iloc[0]) == pytest.approx(10.0 * 1e9 * 2.0)
    assert float(mcap["BBB"].iloc[0]) == pytest.approx(20.0 * 2e9), "no factor -> unchanged"
    assert list(mcap.columns) == ["AAA", "BBB"], "the stray column must not appear"

    print("\n=== SANITY CHECK: alignment ===")
    print(f"  AAA (factor 2.0) -> ${mcap['AAA'].iloc[0]:,.0f}; "
          f"BBB (no factor) -> ${mcap['BBB'].iloc[0]:,.0f}; ZZZ dropped.")
    print("  A missing factor is 1.0, not NaN. Validated.")


# --------------------------------------------------------------------------- #
# the Yahoo bug register                                                       #
# --------------------------------------------------------------------------- #
#: The five bars Yahoo DID back-adjust for MNST's 2026-08-11 two-for-one, leaving 7,793
#: others alone. Real dates, read off the live table.
MNST_ISLANDS = pd.DatetimeIndex(["2026-07-20", "2026-07-21", "2026-07-22",
                                 "2026-07-31", "2026-08-06"])
MNST_SPLIT = pd.Timestamp("2026-08-11")


def _mnst_shaped() -> tuple[pd.Series, pd.Series]:
    """A miniature of Yahoo's live MNST series: `(honest, published)`.

    Everything before the split reads DOUBLE what the stock traded at, except the five island
    bars Yahoo did adjust, which are already right. The honest series drifts smoothly so that
    no genuine step comes anywhere near the flip band -- the real MNST's largest move in
    thirty years is +45.5%, against a band that starts at +82%."""
    idx = pd.DatetimeIndex(pd.bdate_range("2026-06-01", "2026-08-31"), name="date")
    honest = pd.Series(np.linspace(48.0, 45.0, len(idx)), index=idx, name="MNST")
    published = honest.where(idx >= MNST_SPLIT, honest * 2.0)
    published.loc[MNST_ISLANDS] = honest.loc[MNST_ISLANDS]
    return honest, published


def _wide(published: pd.Series) -> dict[str, pd.DataFrame]:
    return {"close_split": published.to_frame(), "close_total": published.to_frame()}


def _vendor(honest: pd.Series, dates: list[str]) -> pd.DataFrame:
    """Sharadar filing rows carrying the price the stock ACTUALLY traded at."""
    when = pd.DatetimeIndex(dates)
    return pd.DataFrame({"ticker": "MNST", "date": when,
                         "price": honest.reindex(when).to_numpy()})


#: Filing rows on stale bars only, so the observed wedge is a clean 0.5 -- which is what the
#: live table gives: 0.50000 on all 122 rows from 1996 to 2026-08-07.
VENDOR_DATES = ["2026-06-05", "2026-06-19", "2026-07-10", "2026-07-24", "2026-08-07"]

MNST_ENTRY = {"split_vintage": {"MNST": [{"before": "2026-08-11", "ratio": 2.0,
                                          "expect_wedge": 0.5}]}}


def test_a_split_vintage_mixture_is_put_back_on_one_basis():
    """THE MNST CASE. Yahoo adjusted five bars for its own published split and left the rest,
    so the series alternates between two bases inside three weeks.

    A `factor` cannot express this: half the affected bars are already correct and multiplying
    them would break them. The backwards walk has to reach the honest series EXACTLY -- both
    the doubled bars and the five it must leave alone."""
    honest, published = _mnst_shaped()
    wide = _wide(published)
    applied = apply_split_vintage(wide, MNST_ENTRY, _vendor(honest, VENDOR_DATES),
                                  lambda *a: None)

    assert applied == 1
    for field in ("close_split", "close_total"):
        assert wide[field]["MNST"].to_numpy() == pytest.approx(honest.to_numpy(), rel=1e-12)

    print("\n=== SANITY CHECK: MNST put back on one basis ===")
    print(f"  published spans {published.min():.2f}..{published.max():.2f} "
          f"-> repaired {wide['close_split']['MNST'].min():.2f}.."
          f"{wide['close_split']['MNST'].max():.2f}")
    print(f"  the {len(MNST_ISLANDS)} bars Yahoo had already adjusted were left alone.")


def test_the_walk_leaves_the_anchor_and_everything_after_it_untouched():
    """Only bars STRICTLY BEFORE the split can move. If the walk touched the anchor side, an
    incremental build whose window starts after the split would correct a second time."""
    honest, published = _mnst_shaped()
    multiplier, flips = _vintage_multiplier(published, MNST_SPLIT, 2.0)

    after = multiplier.loc[MNST_SPLIT:]
    assert (after.to_numpy() == 1.0).all(), "post-anchor bars are the reference, not a target"
    assert flips >= len(MNST_ISLANDS), "each island has an edge going in and one coming out"
    assert set(multiplier.loc[MNST_ISLANDS].round(9)) == {1.0}, "islands were already correct"


def test_the_fabricated_island_returns_disappear():
    """The reason this repair is allowed to move returns at all.

    Each island contributes a -51% and a +96% bar that never happened, and `ret` is what
    momentum, vol, betas and every label are built from. After the walk the largest one-bar
    move must be an ordinary one."""
    honest, published = _mnst_shaped()
    before = published.pct_change().abs().max()
    wide = _wide(published)
    apply_split_vintage(wide, MNST_ENTRY, _vendor(honest, VENDOR_DATES), lambda *a: None)
    after = wide["close_split"]["MNST"].pct_change().abs().max()

    assert before > 0.45, "the fixture must actually contain the defect"
    assert after < 0.01, f"a fabricated move survived the repair: {after:.2%}"


def test_a_vintage_entry_whose_defect_is_gone_is_refused():
    """⚠ THE PROPERTY THE WHOLE REGISTER RESTS ON. If Yahoo fixes its data the observation
    stops matching and the entry must NOT fire -- otherwise the repair lands on top of the
    vendor's own correction and halves a correct series."""
    honest, published = _mnst_shaped()
    wide = _wide(honest)                      # Yahoo has fixed it: published == honest
    logged: list[str] = []
    applied = apply_split_vintage(wide, MNST_ENTRY, _vendor(honest, VENDOR_DATES),
                                  lambda msg, *a: logged.append(msg % a))

    assert applied == 0
    assert wide["close_split"]["MNST"].to_numpy() == pytest.approx(honest.to_numpy(),
                                                                   rel=0, abs=0)
    assert any("GONE or CHANGED" in m for m in logged), logged


def test_a_return_seam_removes_a_move_that_never_happened():
    """The JCI case: one boundary, no islands behind it, so the whole prefix moves together
    and the fabricated -60.52% bar goes with it."""
    idx = pd.DatetimeIndex(pd.bdate_range("2007-06-01", "2007-08-01"), name="date")
    when = pd.Timestamp("2007-07-02")
    series = pd.Series(70.354, index=idx, name="JCI")
    series.loc[when:] = 27.775
    step = 27.775 / 70.354

    wide = {"close_split": series.to_frame(), "close_total": series.to_frame()}
    applied = apply_return_seams(wide, {"return_seams": {"JCI": [
        {"date": "2007-07-02", "step": step}]}}, lambda *a: None)

    assert applied == 1
    repaired = wide["close_split"]["JCI"]
    assert repaired.pct_change().abs().max() == pytest.approx(0.0, abs=1e-12)
    assert repaired.iloc[0] == pytest.approx(27.775, rel=1e-12)


def test_a_stale_return_seam_is_skipped_rather_than_applied():
    """A registered step that no longer matches the frame is a register that has gone out of
    date, not a licence to rescale nine years of history."""
    idx = pd.DatetimeIndex(pd.bdate_range("2007-06-01", "2007-08-01"), name="date")
    series = pd.Series(70.354, index=idx, name="JCI")
    wide = {"close_split": series.to_frame(), "close_total": series.to_frame()}
    logged: list[str] = []
    applied = apply_return_seams(wide, {"return_seams": {"JCI": [
        {"date": "2007-07-02", "step": 0.394789}]}}, lambda m, *a: logged.append(m % a))

    assert applied == 0
    assert (wide["close_split"]["JCI"].to_numpy() == 70.354).all()
    assert any("GONE or CHANGED" in m for m in logged), logged


def test_a_level_wedge_moves_S_and_only_S():
    """IP's shape: Yahoo back-adjusted smoothly for a spinoff and published no feed row, so
    the RETURNS are already right and only the LEVEL is short.

    ⚠ The repair must therefore leave `close_split` alone. It multiplies `S`, which reaches
    market cap, and nothing else."""
    idx = pd.DatetimeIndex(pd.bdate_range("2014-01-01", "2014-12-31"), name="date")
    before = pd.Timestamp("2014-07-01")
    close = pd.DataFrame({"IP": 40.0}, index=idx)
    vendor = pd.DataFrame({"ticker": "IP", "date": pd.DatetimeIndex(
        ["2014-03-31", "2014-05-15"]), "price": 40.0 * 1.07078})
    factor = pd.DataFrame({"IP": 1.0}, index=idx)

    out = apply_level_bugfix(factor, {"level_factor": {"IP": {"segments": [
        {"before": "2014-07-01", "factor": 1.07078}]}}}, vendor, close, lambda *a: None)

    early = out.loc[out.index < before, "IP"].to_numpy()
    assert early == pytest.approx(1.07078)
    assert (out.loc[out.index >= before, "IP"] == 1.0).all()
    assert (close["IP"] == 40.0).all(), "a level wedge must never touch a price"


def test_a_level_wedge_whose_observation_moved_is_refused():
    """Same entry, a vendor that now agrees with Yahoo. The wedge reads 1.0, so the defect is
    gone and the multiply must not happen."""
    idx = pd.DatetimeIndex(pd.bdate_range("2014-01-01", "2014-12-31"), name="date")
    close = pd.DataFrame({"IP": 40.0}, index=idx)
    vendor = pd.DataFrame({"ticker": "IP", "date": pd.DatetimeIndex(
        ["2014-03-31", "2014-05-15"]), "price": 40.0})
    logged: list[str] = []

    out = apply_level_bugfix(pd.DataFrame({"IP": 1.0}, index=idx),
                             {"level_factor": {"IP": {"segments": [
                                 {"before": "2014-07-01", "factor": 1.07078}]}}},
                             vendor, close, lambda m, *a: logged.append(m % a))

    assert (out["IP"] == 1.0).all()
    assert any("GONE or CHANGED" in m for m in logged), logged


def test_a_multi_segment_wedge_is_a_staircase_not_a_product():
    """HBAN's shape, and the bug the build log caught.

    Five annual 10% stock dividends leave a wedge of 1.1^5, 1.1^4, ... 1.1^1 through five
    consecutive eras. Each of those is the WHOLE wedge for its own era, so the segments must
    not compound: the earliest bars want 1.61051, and 1.61051 x 1.46410 x 1.33100 x 1.21 x 1.1
    = 4.177 is not a number anything measured.

    ⚠ The verification has to be windowed too. Measured over an open prefix the median follows
    the older, more numerous rows -- HBAN's second window read 1.46410 where the truth is
    1.33100 -- which SKIPPED three of the five entries as "gone or changed" while they were
    simply being measured in the wrong place."""
    idx = pd.DatetimeIndex(pd.bdate_range("1995-01-02", "2002-12-31"), name="date")
    ladder = [("1996-07-12", 1.61051), ("1997-07-11", 1.46410), ("1998-07-13", 1.33100),
              ("1999-07-12", 1.21000), ("2000-07-12", 1.10000)]
    close = pd.DataFrame({"HBAN": 20.0}, index=idx)

    # One filing row per quarter, priced at the wedge its own era carries.
    filings = pd.DatetimeIndex(pd.date_range("1995-02-15", "2002-11-15", freq="QE"))
    bounds = [pd.Timestamp(d) for d, _ in ladder]
    wedges = [1.61051, 1.46410, 1.33100, 1.21000, 1.10000, 1.0]
    price = [20.0 * wedges[int(np.searchsorted(bounds, d, side="right"))] for d in filings]
    vendor = pd.DataFrame({"ticker": "HBAN", "date": filings, "price": price})

    logged: list[str] = []
    out = apply_level_bugfix(
        pd.DataFrame({"HBAN": 1.0}, index=idx),
        {"level_factor": {"HBAN": {"segments": [
            {"before": d, "factor": f} for d, f in ladder]}}},
        vendor, close, lambda m, *a: logged.append(m % a))

    assert not any("SKIPPED" in m for m in logged), logged
    for era, (when, expected) in enumerate(ladder):
        low = pd.Timestamp(ladder[era - 1][0]) if era else pd.Timestamp.min
        got = out.loc[(out.index >= low) & (out.index < pd.Timestamp(when)), "HBAN"]
        assert got.to_numpy() == pytest.approx(expected), f"era ending {when}"
    assert (out.loc[pd.Timestamp(ladder[-1][0]):, "HBAN"] == 1.0).all(), "the wedge closes"

    print("\n=== SANITY CHECK: HBAN's staircase is absolute, not cumulative ===")
    print(f"  earliest bars x{out['HBAN'].iloc[0]:.5f} (not the 4.17725 a product would give)")

# --------------------------------------------------------------------------- #
# the register file itself                                                    #
# --------------------------------------------------------------------------- #
def test_an_unapproved_register_is_refused(tmp_path):
    """A regenerated proposal is byte-identical to a reviewed decision, so without the
    `_APPROVED` block "human-approved" is only a sentence in a docstring. These entries
    rewrite prices, which is the strongest case in the repo for demanding it."""
    (tmp_path / "prices").mkdir()
    path = tmp_path / "prices" / "yf_price_bugfix.json"
    path.write_text(json.dumps({"level_factor": {}}), encoding="utf-8")

    with pytest.raises(ValueError, match="_APPROVED"):
        load_bugfix(tmp_path)


def test_a_missing_register_is_not_an_error(tmp_path):
    """A checkout without the file must build normally, not fail."""
    assert load_bugfix(tmp_path) == {}


def test_the_shipped_register_is_loadable_and_states_its_evidence():
    """The real `configs/prices/yf_price_bugfix.json`, not a fixture.

    Every entry must carry the value it EXPECTS TO OBSERVE -- that is what lets
    `StepCubePrices` re-measure it and skip it once Yahoo fixes the underlying data. An entry
    with a repair and no expectation can never be re-verified, so it is not allowed."""
    blob = load_bugfix(DEFAULT_CONFIG_DIR)
    assert blob and "_APPROVED" in blob

    for ticker, spec in blob["level_factor"].items():
        assert spec["evidence"], ticker
        for segment in spec["segments"]:
            assert segment["factor"] > 0 and segment["before"], ticker
    for ticker, entries in blob["split_vintage"].items():
        for entry in entries:
            assert entry["ratio"] > 0 and entry["expect_wedge"] > 0, ticker
            assert entry["evidence"] and entry["corroboration"], ticker
    for ticker, entries in blob["return_seams"].items():
        for entry in entries:
            assert entry["step"] > 0 and entry["evidence"], ticker
