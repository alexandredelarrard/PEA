"""`f_ceo_pay_growth` and `f_ceo_pay_vs_revenue_growth`: log ratios with a fixed bound and a CEO-identity guard.

Contract pinned here:

  1. Pay growth is `log(total_t / total_{t-1})`, clipped to `+-PAY_LOG_GROWTH_BOUND`. The bound is the same on
     every date, so the same raw growth gets the same value whatever the other tickers do that day; there is no
     cross-sectional trim.
  2. The CEO-identity guard removes ratios across a change of CEO: two packages of two different people are not a
     growth rate, whatever their size. An unknown identity keeps the value and is counted.
  3. A prior year filed as $0 has no growth rate (NaN, never an infinity).
  3b. A cross-filing pair outside the 250-550 day annual window is NaN and ends the earlier measurement; the filing's
     own SCT pair is kept.
  4. `ceo_pay_vs_revenue_growth` is pay log growth minus revenue log growth, clipped to the same bound.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_aggregate.utils.governance.names import ceo_identity_changed
from src.data_aggregate.utils.governance.panel import (
    PAY_LOG_GROWTH_BOUND,
    _ceo_pay_growth,
    _governance_fields,
)

IDX = pd.bdate_range("2011-01-03", "2012-12-31")
AS_OF_1 = pd.Timestamp("2011-03-15")
AS_OF_2 = pd.Timestamp("2012-03-15")
#: The window the SECOND proxy covers, i.e. where a one-filing growth can exist at all. Before `AS_OF_2` every
#: ticker is NaN and that is correct point-in-time behaviour, not the guard.
COVERED = IDX[IDX >= AS_OF_2]

#: Controls all carry the SAME +15% year, so any change to a control can only come from the other tickers.
N_CONTROLS = 100
ORDINARY_GROWTH = float(np.log(1.15))


def _proxies(
    bad_comp: tuple[float, float] = (1.0, 280_621_552.0),
    bad_names: tuple[str | None, str | None] = ("Larry Page", "Sundar Pichai"),
    extra: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """`N_CONTROLS` well-behaved filers plus `BAD`, two annual proxies each. `BAD` defaults to a real $1 followed
    by a real $280,621,552 across a change of CEO."""
    rows = []
    for i in range(N_CONTROLS):
        t = f"C{i:03d}"
        rows += [
            {"ticker": t, "as_of": AS_OF_1, "ceo_total_comp": 1_000_000.0, "ceo_name_proxy": "Jane Q. Control", "board_size": 9.0},
            {"ticker": t, "as_of": AS_OF_2, "ceo_total_comp": 1_150_000.0, "ceo_name_proxy": "Jane Q. Control", "board_size": 9.0},
        ]
    rows += [
        {"ticker": "BAD", "as_of": AS_OF_1, "ceo_total_comp": bad_comp[0], "ceo_name_proxy": bad_names[0], "board_size": 11.0},
        {"ticker": "BAD", "as_of": AS_OF_2, "ceo_total_comp": bad_comp[1], "ceo_name_proxy": bad_names[1], "board_size": 11.0},
    ]
    df = pd.DataFrame(rows)
    if extra is not None:
        df = pd.concat([df, extra], ignore_index=True)
    return df


def _revenue(tickers: list[str], factors: dict[str, float] | None = None) -> pd.DataFrame:
    """One year-on-year revenue pair per ticker (so the difference column is built); flat unless `factors` says so."""
    factors = factors or {}
    rows = []
    for t in tickers:
        rows += [{"ticker": t, "as_of": AS_OF_1, "totalRevenue": 1.0e9}, {"ticker": t, "as_of": AS_OF_2, "totalRevenue": 1.0e9 * factors.get(t, 1.0)}]
    return pd.DataFrame(rows)


def _shipped(hist: pd.DataFrame, factors: dict[str, float] | None = None) -> dict[str, pd.DataFrame]:
    return _governance_fields(hist, IDX, _revenue(sorted(hist["ticker"].unique()), factors), {})


# --------------------------------------------------------------------------- #
# the CEO-identity guard                                                      #
# --------------------------------------------------------------------------- #


def test_a_ceo_change_is_not_pay_growth():
    """$1 -> $280,621,552 across a change of CEO is NaN."""
    g = _ceo_pay_growth(_proxies(), IDX)
    assert g.loc[COVERED, "BAD"].isna().all(), "the transition was not guarded"
    assert np.allclose(g.loc[COVERED, "C000"], ORDINARY_GROWTH), "a control was disturbed"
    print("SANITY: a $1 -> $280M step across two different CEOs is NaN; the controls keep their log(1.15) growth.")


def test_the_same_ceo_keeps_the_value_up_to_the_bound():
    """The guard is not an outlier detector: a real $28M package following a prior $1 by the SAME person is kept,
    and only the fixed bound limits it."""
    g = _ceo_pay_growth(_proxies(bad_comp=(1.0, 28_095_227.0), bad_names=("Charles Liang", "Charles Liang")), IDX)
    assert np.allclose(g.loc[COVERED, "BAD"], PAY_LOG_GROWTH_BOUND), "the guard over-fired or the bound moved"
    print(f"SANITY: a 28,000,000x same-CEO year is kept and sits exactly at the +{PAY_LOG_GROWTH_BOUND:g} bound.")


def test_a_zero_prior_year_is_nan_not_infinity():
    """A prior year filed as $0 has no growth rate. The ticker may drop out of the frame entirely (its only
    observation is undefined), so both outcomes count; an infinity reaching the grid fails loudly."""
    g = _ceo_pay_growth(_proxies(bad_comp=(0.0, 15_499_993.0), bad_names=("Alan Mulally", "Alan Mulally")), IDX)
    assert "BAD" not in g.columns or g.loc[COVERED, "BAD"].isna().all()
    v = g.loc[COVERED].to_numpy(dtype="float64").ravel()
    assert np.isfinite(v[~np.isnan(v)]).all(), "an infinity reached the daily grid"
    assert np.allclose(g.loc[COVERED, "C000"], ORDINARY_GROWTH), "a control was disturbed"
    print("SANITY: $0 -> $15.5M is NaN, not an infinity; controls untouched.")


def test_an_unknown_identity_keeps_the_value_and_is_counted():
    """An unknown identity does not prove a change, so the value stays and the tally makes the policy visible."""
    tally: dict[str, int] = {}
    g = _ceo_pay_growth(_proxies(bad_names=("Larry Page", None)), IDX, tally)
    assert g.loc[COVERED, "BAD"].notna().all(), "an unknown identity must not blank the value"
    assert tally["ceo_pay_growth: kept on an UNKNOWN CEO identity (filings)"] == 1
    assert tally["ceo_pay_growth: nulled across a CEO change"] == 0
    print("SANITY: an unknown CEO identity keeps the value and is counted once.")


def test_the_flag_cannot_be_borrowed_from_another_filing():
    """Value and flag come out of one subset, so a THIRD filing under the new CEO un-masks the column: the growth
    between the new CEO's first and second year is a real growth rate."""
    as_of_3 = pd.Timestamp("2013-03-15")
    long_idx = pd.bdate_range("2011-01-03", "2013-12-31")
    extra = pd.DataFrame(
        [{"ticker": "BAD", "as_of": as_of_3, "ceo_total_comp": 322_214_785.0, "ceo_name_proxy": "Sundar Pichai", "board_size": 11.0}]
    )
    g = _ceo_pay_growth(_proxies(extra=extra), long_idx)
    masked = long_idx[(long_idx >= AS_OF_2) & (long_idx < as_of_3)]
    assert g.loc[masked, "BAD"].isna().all(), "the transition itself must stay masked"
    live = long_idx[long_idx >= as_of_3]
    assert g.loc[live, "BAD"].notna().all(), "the flag was carried past its own filing"
    assert np.allclose(g.loc[live, "BAD"], np.log(322_214_785.0 / 280_621_552.0))
    print("SANITY: the transition stays masked until the next same-CEO filing, then log(322.2M / 280.6M) is live.")


def test_the_guard_shares_one_definition_with_the_pay_family():
    """`names.ceo_identity_changed` is the only definition of "the CEO changed"; spelling drift must not read as a
    turnover."""
    h = pd.DataFrame({"ticker": ["X", "X", "X", "Y", "Y"], "n": ["Timothy D. Cook", "Timothy Cook", "Tim Cook", "Larry Page", "Sundar Pichai"]})
    changed = ceo_identity_changed(h["n"], h["ticker"])
    assert changed.tolist()[1:3] == [0.0, 0.0], "spelling drift read as a turnover"
    assert changed.iloc[4] == 1.0, "a real transition was missed"
    assert pd.isna(changed.iloc[0]) and pd.isna(changed.iloc[3]), "no prior filing -> unknown"
    print("SANITY: spelling variants of one CEO are not a change; Page -> Pichai is.")


def test_the_tally_counts_cells_and_tickers():
    tally: dict[str, int] = {}
    _ceo_pay_growth(_proxies(), IDX, tally)
    assert tally["ceo_pay_growth: nulled across a CEO change"] == len(COVERED)
    assert tally["ceo_pay_growth: tickers with a nulled transition"] == 1
    print(f"SANITY: {len(COVERED)} nulled cells on 1 ticker are counted.")


# --------------------------------------------------------------------------- #
# the fixed bound                                                             #
# --------------------------------------------------------------------------- #


def test_growth_is_a_log_ratio_with_the_same_bound_on_both_sides():
    """+5x stays log(5); a 50x year and a 98% cut are clipped to +-bound, symmetrically."""
    cases = {"UP5": (1e6, 5e6), "UP50": (1e6, 5e7), "DOWN50": (5e7, 1e6)}
    extra = pd.DataFrame(
        [
            {"ticker": t, "as_of": a, "ceo_total_comp": c, "ceo_name_proxy": "Same Person", "board_size": 9.0}
            for t, pair in cases.items()
            for a, c in zip((AS_OF_1, AS_OF_2), pair, strict=True)
        ]
    )
    g = _ceo_pay_growth(_proxies(extra=extra), IDX)
    assert np.allclose(g.loc[COVERED, "UP5"], np.log(5.0))
    assert np.allclose(g.loc[COVERED, "UP50"], PAY_LOG_GROWTH_BOUND)
    assert np.allclose(g.loc[COVERED, "DOWN50"], -PAY_LOG_GROWTH_BOUND)
    print(f"SANITY: x5 -> {np.log(5.0):.3f}; x50 -> +{PAY_LOG_GROWTH_BOUND:g}; x0.02 -> -{PAY_LOG_GROWTH_BOUND:g}.")


def test_an_ordinary_pay_year_is_log_growth_untouched():
    """The body of the distribution is inert: a +15% year ships exactly as log(1.15)."""
    f = _shipped(_proxies(bad_comp=(1.0, 28_095_227.0), bad_names=("Charles Liang", "Charles Liang")))["ceo_pay_growth"]
    controls = [c for c in f.columns if c.startswith("C")]
    assert len(controls) == N_CONTROLS
    pd.testing.assert_frame_equal(
        f.loc[COVERED, controls], pd.DataFrame(ORDINARY_GROWTH, index=COVERED, columns=controls), check_exact=False, check_names=False, atol=1e-12
    )
    print(f"SANITY: {N_CONTROLS} controls keep log(1.15) = {ORDINARY_GROWTH:.4f} exactly.")


def test_the_same_raw_growth_gets_the_same_value_whatever_the_other_tickers_do():
    """No cross-sectional dependence: ten tickers with a 1000x year on the same date must not move BAD's value
    (a per-date quantile trim would)."""
    alone = _proxies(bad_comp=(1e6, 7.4e6), bad_names=("Same Person", "Same Person"))
    crowd = pd.DataFrame(
        [
            {"ticker": f"E{i:02d}", "as_of": a, "ceo_total_comp": c, "ceo_name_proxy": "Extreme Person", "board_size": 9.0}
            for i in range(10)
            for a, c in zip((AS_OF_1, AS_OF_2), (1e6, 1e9), strict=True)
        ]
    )
    with_crowd = _proxies(bad_comp=(1e6, 7.4e6), bad_names=("Same Person", "Same Person"), extra=crowd)
    first = _shipped(alone)["ceo_pay_growth"].loc[COVERED, "BAD"]
    second = _shipped(with_crowd)["ceo_pay_growth"].loc[COVERED, "BAD"]
    pd.testing.assert_series_equal(first, second, check_names=False)
    assert np.allclose(first, np.log(7.4))
    print(f"SANITY: BAD's {np.log(7.4):.3f} is identical with and without ten 1000x tickers on the same dates.")


@pytest.mark.parametrize("name", ["ceo_pay_growth", "ceo_pay_vs_revenue_growth"])
def test_both_shipped_columns_respect_the_bound(name: str):
    hist = _proxies(bad_comp=(1.0, 28_095_227.0), bad_names=("Charles Liang", "Charles Liang"))
    f = _shipped(hist)[name]
    assert f.loc[COVERED, "BAD"].max() == pytest.approx(PAY_LOG_GROWTH_BOUND)
    assert f.loc[COVERED].abs().max().max() <= PAY_LOG_GROWTH_BOUND + 1e-12
    print(f"SANITY: {name} never leaves [-{PAY_LOG_GROWTH_BOUND:g}, +{PAY_LOG_GROWTH_BOUND:g}].")


def test_the_difference_column_is_pay_minus_revenue_log_growth():
    """Flat revenue: the difference IS the pay leg. Doubling revenue subtracts log(2). A revenue collapse to 0.1%
    would push the gap past the bound and is clipped."""
    fields = _shipped(_proxies())
    diff = fields["ceo_pay_vs_revenue_growth"]
    assert diff.loc[COVERED, "BAD"].isna().all(), "the CEO guard did not reach the difference"
    assert np.allclose(diff.loc[COVERED, "C000"], ORDINARY_GROWTH)

    doubled = _shipped(_proxies(), {"C000": 2.0})["ceo_pay_vs_revenue_growth"]
    assert np.allclose(doubled.loc[COVERED, "C000"], ORDINARY_GROWTH - np.log(2.0))

    collapsed = _shipped(_proxies(), {"C000": 0.001})["ceo_pay_vs_revenue_growth"]
    assert np.allclose(collapsed.loc[COVERED, "C000"], PAY_LOG_GROWTH_BOUND)
    print("SANITY: gap = pay log growth - revenue log growth; x2 revenue shifts it by -log 2; a 99.9% collapse clips at +bound.")


# --------------------------------------------------------------------------- #
# the annual-pair gate (the same 250-550 day rule as `pay_features.prior_annual_leg`)                 #
# --------------------------------------------------------------------------- #

LONG_IDX = pd.bdate_range("2011-01-03", "2015-12-31")


def _gap_proxies(second: str, sct: float | None = None, third: str | None = None) -> pd.DataFrame:
    """One filer, same CEO, 1,000,000 then 1,500,000 (then 3,000,000) at the given dates, 2011-03-15 first."""
    dates = ["2011-03-15", second] + ([third] if third else [])
    comps = [1_000_000.0, 1_500_000.0, 3_000_000.0][: len(dates)]
    rows = [{"ticker": "GAP", "as_of": pd.Timestamp(d), "ceo_total_comp": c, "ceo_name_proxy": "Same Person"} for d, c in zip(dates, comps, strict=True)]
    if sct is not None:
        rows[1]["ceo_comp_growth_sct"] = sct
    return pd.DataFrame(rows)


def test_a_three_year_gap_is_not_yearly_growth_and_a_one_year_gap_is_unchanged():
    """2011 -> 2014 skips two proxies: the cross-filing ratio is NaN. 2011 -> 2012 is the ordinary yearly growth."""
    skipped = _ceo_pay_growth(_gap_proxies("2014-03-15"), LONG_IDX)
    assert "GAP" not in skipped.columns or skipped["GAP"].isna().all(), "a three-year change was labelled as yearly growth"
    yearly = _ceo_pay_growth(_gap_proxies("2012-03-12"), LONG_IDX)
    assert np.allclose(yearly.loc["2012-04-02":"2012-12-31", "GAP"], np.log(1.5))
    print("SANITY: a 3-year gap gives NaN; a 363-day gap keeps log(1.5) = 0.405.")


def test_the_filed_sct_pair_still_wins_over_the_gap_gate():
    """The SCT FY vs FY-1 pair is a same-filing yearly figure, so it is kept even where the cross-filing gap is 3 years."""
    g = _ceo_pay_growth(_gap_proxies("2014-03-15", sct=0.25), LONG_IDX)
    assert np.allclose(g.loc["2014-04-01":"2014-12-31", "GAP"], 0.25)
    print("SANITY: a filed SCT growth of 0.25 ships over a 3-year cross-filing gap.")


def test_a_gated_pair_is_a_carry_barrier_not_a_gap_to_fill():
    """A filing 120 days after the previous one (an amendment) is outside the annual window: its change is NaN and
    the earlier growth does not carry past it, instead of being replaced by log(3.0 / 1.5)."""
    g = _ceo_pay_growth(_gap_proxies("2012-03-12", third="2012-07-10"), LONG_IDX)
    before = g.loc["2012-04-02":"2012-07-06", "GAP"]
    after = g.loc["2012-07-12":"2012-12-31", "GAP"]
    assert np.allclose(before, np.log(1.5)), "the yearly growth is live until the next filing"
    assert after.isna().all(), "the gated change must end the earlier measurement"
    print("SANITY: log(1.5) holds until the 120-day follow-up filing, then NaN (no carry, no 2-filing ratio).")


def test_the_gate_reaches_the_difference_to_revenue_growth():
    """`ceo_pay_vs_revenue_growth` is built from the same leg, so a gated pair is NaN there too while a control keeps its value."""
    hist = pd.concat([_proxies(), _gap_proxies("2014-03-15")], ignore_index=True)
    shipped = _governance_fields(hist, LONG_IDX, _revenue(sorted(hist["ticker"].unique())), {})["ceo_pay_vs_revenue_growth"]
    assert np.allclose(shipped.loc["2012-04-02":"2012-12-31", "C000"], ORDINARY_GROWTH)
    assert "GAP" not in shipped.columns or shipped["GAP"].isna().all()
    print("SANITY: the 3-year pay pair is NaN in ceo_pay_vs_revenue_growth; the control keeps log(1.15).")
