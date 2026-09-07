"""
test_momentum_xs_population.py  (tests/data_aggregate/test_momentum_xs_population.py)
------------------------------------------------------------------
Two defects of the momentum part, both about MISSINGNESS carrying information the stored
column cannot express.

D-01 -- a thin cross-section ranked as if it were full. `xs_rank_pct` is
`rank(axis=1, pct=True)`, which ranks over whatever is non-null in the ROW. On the live table
2026-08-28 carried 45 of 491 tickers because one extract run was truncated, and 13 of the 28
features were ranked over those 45: the stored decile boundaries were drawn from 9% of the
universe and the column shows no sign of it. `min(rev_5)` on that date was 1/45 = 0.0222
where every other date reads ~1/491 = 0.0020.

D-04 -- `downside_vol_63` at `min_periods=20`. `neg` keeps only DOWN days, so the period
count is a count of LOSING days, not of available data. Requiring 20 of them nulls exactly
the names that have been going up: 14,376 cells over 410 tickers whose median trailing 63-day
return is +21.36% against +3.87% where present.
"""
import numpy as np
import pandas as pd
import pytest

from src.data_aggregate.utils.common.xs import xs_rank_pct
from src.data_aggregate.utils.momentum.features import (
    MIN_XS_POPULATION_FRAC, XS_POPULATION_WINDOW, _null_thin_cross_sections,
    _thin_cross_sections, build_feature_panel, compute_raw_features)

DATES = pd.bdate_range("2024-01-01", periods=120)
TICKERS = [f"T{i:03d}" for i in range(100)]


def _full(n_names: int = 100) -> pd.DataFrame:
    """A fully-populated (date x ticker) feature frame."""
    rng = np.random.default_rng(7)
    return pd.DataFrame(rng.normal(size=(len(DATES), n_names)),
                        index=DATES, columns=TICKERS[:n_names])


# --------------------------------------------------------------------------- #
# D-01: the detector                                                           #
# --------------------------------------------------------------------------- #
def test_collapse_is_flagged():
    """The live shape: one date at 45 of 491 = 9% of the trailing norm."""
    f = _full()
    bad = DATES[80]
    f.loc[bad, TICKERS[9:100]] = np.nan          # 9 of 100 survive
    thin = _thin_cross_sections({"rev_5": f})
    print(f"  population on {bad.date()} = {int(f.loc[bad].notna().sum())}/100 "
          f"-> flagged={bool(thin.loc[bad, 'rev_5'])}")
    assert thin.loc[bad, "rev_5"]
    assert thin["rev_5"].sum() == 1, "only the collapsed date"


def test_just_above_and_below_the_floor():
    """The threshold binds where the constant says it does, not approximately."""
    f = _full()
    lo, hi = DATES[80], DATES[90]
    keep_below = int(MIN_XS_POPULATION_FRAC * 100) - 1     # 59
    keep_above = int(MIN_XS_POPULATION_FRAC * 100) + 1     # 61
    f.loc[lo, TICKERS[keep_below:]] = np.nan
    f.loc[hi, TICKERS[keep_above:]] = np.nan
    thin = _thin_cross_sections({"x": f})
    print(f"  {keep_below}/100 -> flagged={bool(thin.loc[lo, 'x'])}; "
          f"{keep_above}/100 -> flagged={bool(thin.loc[hi, 'x'])}")
    assert thin.loc[lo, "x"] is np.True_ or thin.loc[lo, "x"]
    assert not thin.loc[hi, "x"]


def test_warmup_ramp_is_never_flagged():
    """A monotonically growing population is a WARM-UP, not a collapse. This is the case the
    day-max reference variant got wrong: a 1,260-day seasonality feature is legitimately
    thinner than a 5-day reversal in early history."""
    f = _full()
    for i, d in enumerate(DATES):
        live = min(100, 3 + i)                  # 3, 4, 5, ... 100
        f.loc[d, TICKERS[live:]] = np.nan
    thin = _thin_cross_sections({"seasonal_h21": f})
    print(f"  ramp 3 -> 100 names over {len(DATES)} dates -> flagged {int(thin.values.sum())}")
    assert not thin.values.any()


def test_structurally_thin_feature_is_not_flagged():
    """A feature that is ALWAYS thin is not defective -- it is compared against its own norm,
    never against the day's best-populated feature."""
    wide, narrow = _full(), _full()
    narrow.loc[:, TICKERS[20:]] = np.nan        # 20 of 100, every single date
    thin = _thin_cross_sections({"wide": wide, "narrow": narrow})
    print(f"  narrow feature at a flat 20/100 alongside a 100/100 one "
          f"-> flagged {int(thin['narrow'].sum())}")
    assert not thin["narrow"].any()


def test_all_empty_date_is_not_flagged():
    """Nothing to rank either way; flagging it would fire through every warm-up."""
    f = _full()
    f.loc[DATES[80], :] = np.nan
    thin = _thin_cross_sections({"x": f})
    assert not thin.loc[DATES[80], "x"]


def test_reference_excludes_the_collapsing_date_itself():
    """`shift(1)` on the reference: a run of bad dates must not drag its own median down to
    the point where the later ones look acceptable."""
    f = _full()
    run = DATES[80:85]
    f.loc[run, TICKERS[9:]] = np.nan
    thin = _thin_cross_sections({"x": f})
    print(f"  5 consecutive collapsed dates -> flagged {int(thin['x'].sum())}")
    assert thin.loc[run, "x"].all(), "every date in the run, not just the first"


def test_reference_window_length_is_honoured():
    """A collapse is still caught once the window has rolled past an earlier one."""
    f = _full()
    late = DATES[40 + XS_POPULATION_WINDOW + 5]
    f.loc[late, TICKERS[9:]] = np.nan
    thin = _thin_cross_sections({"x": f})
    assert thin.loc[late, "x"]


# --------------------------------------------------------------------------- #
# D-01: the masking + logging                                                  #
# --------------------------------------------------------------------------- #
def test_nulling_leaves_every_other_cell_bit_identical(caplog):
    f = _full()
    bad = DATES[80]
    f.loc[bad, TICKERS[9:]] = np.nan
    before = f.copy()

    with caplog.at_level("WARNING"):
        out = _null_thin_cross_sections({"rev_5": f})["rev_5"]

    print(f"  warned: {caplog.text.strip()[:110]}")
    assert out.loc[bad].isna().all(), "the thin date is nulled, not renormalised"
    other = out.index != bad
    pd.testing.assert_frame_equal(out.loc[other], before.loc[other])
    assert "rev_5" in caplog.text and str(bad.date()) in caplog.text
    assert "NULLED" in caplog.text


def test_clean_panel_is_untouched_and_silent(caplog):
    raw = {"a": _full(), "b": _full()}
    before = {k: v.copy() for k, v in raw.items()}
    with caplog.at_level("WARNING"):
        out = _null_thin_cross_sections(raw)
    for k in before:
        pd.testing.assert_frame_equal(out[k], before[k])
    assert caplog.text == "", f"unexpected log: {caplog.text}"


def test_only_the_offending_feature_is_nulled():
    """A thin date affects the features that are thin ON it -- the 15 well-populated features
    of 2026-08-28 stayed correct and usable, and must remain so."""
    thin_f, ok_f = _full(), _full()
    bad = DATES[80]
    thin_f.loc[bad, TICKERS[9:]] = np.nan
    ok_before = ok_f.copy()
    out = _null_thin_cross_sections({"thin": thin_f, "ok": ok_f})
    assert out["thin"].loc[bad].isna().all()
    pd.testing.assert_frame_equal(out["ok"], ok_before)


def test_guard_runs_before_the_rank_not_after():
    """The whole point. Show what the unguarded rank publishes, then that the panel does not
    contain it."""
    prices = _synthetic_prices()
    bad = prices.index[100]
    prices.loc[bad, TICKERS[9:]] = np.nan

    raw = compute_raw_features(prices, prices, _sector_returns(prices))
    unguarded_min = xs_rank_pct(raw["rev_5"]).loc[bad].min()
    n_live = int(raw["rev_5"].loc[bad].notna().sum())
    print(f"  unguarded: {n_live} names -> min(rank)={unguarded_min:.6f} (= 1/{n_live})")
    assert unguarded_min == pytest.approx(1.0 / n_live)

    panel = build_feature_panel(prices, prices, _sector_returns(prices))
    on_bad = panel.loc[panel["date"] == bad, "rev_5"]
    print(f"  guarded  : {len(on_bad)} rows on that date, all-NaN={on_bad.isna().all()}")
    assert on_bad.isna().all(), "a rank drawn from 9% of the universe must not be published"


# --------------------------------------------------------------------------- #
# D-04: downside_vol_63 min_periods                                            #
# --------------------------------------------------------------------------- #
def _synthetic_prices(n_dates: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    idx = pd.bdate_range("2023-01-02", periods=n_dates)
    ret = rng.normal(0.0004, 0.012, size=(n_dates, len(TICKERS)))
    return pd.DataFrame(100.0 * np.exp(np.cumsum(ret, axis=0)), index=idx, columns=TICKERS)


def _sector_returns(prices: pd.DataFrame) -> pd.DataFrame:
    r = prices.pct_change().mean(axis=1)
    return pd.DataFrame({t: r for t in prices.columns})


def test_downside_vol_defined_on_a_strong_uptrend():
    """A name with 5-19 down days in 63 is a name that has been going UP. At min_periods=20
    its feature was NaN, which made the NaN a label."""
    prices = _synthetic_prices()
    idx = prices.index
    # 'UPUP' rises every day except 8 scattered down days in its last 63 sessions.
    up = pd.Series(np.exp(np.arange(len(idx)) * 0.002), index=idx) * 100.0
    down_days = idx[-63:][::8][:8]
    up.loc[down_days] *= 0.99
    prices["T000"] = up.values

    ret_63 = prices["T000"].iloc[-63:].pct_change()
    n_down = int((ret_63 < 0).sum())
    raw = compute_raw_features(prices, prices, _sector_returns(prices))
    got = raw["downside_vol_63"]["T000"].iloc[-1]
    print(f"  T000: {n_down} down days in 63 -> downside_vol_63={got}")
    assert 5 <= n_down < 20, "the test case must sit in the band the old min_periods nulled"
    assert np.isfinite(got), "must be defined; the old min_periods=20 returned NaN here"


def test_downside_vol_still_undefined_below_five_down_days():
    """min_periods=5, not 1: a standard deviation off one or two observations is noise."""
    idx = pd.bdate_range("2023-01-02", periods=200)
    prices = pd.DataFrame(
        {t: 100.0 * np.exp(np.arange(len(idx)) * 0.001) for t in TICKERS[:5]}, index=idx)
    prices.iloc[80, 0] *= 0.99          # exactly ONE down day, far outside the last window
    raw = compute_raw_features(prices, prices, _sector_returns(prices))
    got = raw["downside_vol_63"].iloc[-1, 0]
    print(f"  monotone riser, 0 down days in the last 63 -> downside_vol_63={got}")
    assert pd.isna(got)


def test_exactly_five_down_days_is_enough():
    """Pins min_periods at 5 BEHAVIOURALLY, because 'harmonise the odd min_periods back to the
    20 its neighbours use' is exactly how D-04 would silently reopen."""
    idx = pd.bdate_range("2023-01-02", periods=120)
    prices = pd.DataFrame({t: 100.0 * np.exp(np.arange(len(idx)) * 0.001)
                           for t in TICKERS[:5]}, index=idx)
    prices.iloc[[60, 65, 70, 75, 80], 0] *= 0.98        # exactly 5 down days
    raw = compute_raw_features(prices, prices, _sector_returns(prices))
    got = raw["downside_vol_63"].iloc[85, 0]
    print(f"  exactly 5 down days in the window -> downside_vol_63={got}")
    assert np.isfinite(got), "5 down days must be enough (min_periods=5)"
