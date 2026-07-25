"""Tests for the peer-relative fundamental features
(src/data_aggregate/utils/fundamental_features.py).

Two layers:
  * Unit (synthetic, known truth) -> exact math of the derived characteristics,
    the fiscal YoY change, and the peer-relative z-score INCLUDING its
    robustness guard (a near-degenerate peer group must not explode to ~1e13).
  * Real data (small sample) -> the panel builds, is well-formed and bounded,
    and (when valuation coverage is available) the marginal effect of the
    headline features has the economically-correct sign.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from src.data_aggregate.utils.fundamental_features import (
    build_peer_relative_panel,
    _FN_PBO_TAG,
    _FN_PLAN_ASSETS_TAG,
    _ratio,
    _fiscal_change_to_daily,
    _peer_relative,
    _self_history_z,
    _derived_fields,
    build_fundamental_feature_panel,
    build_state_panel,
    load_notes_num_scoped,
    load_tagged_facts,
)


# --------------------------------------------------------------------------- #
# Company-regime handling: mask undefined earnings metrics, keep robust ones,  #
# add state flags                                                              #
# --------------------------------------------------------------------------- #
def _synth_mixed_regime():
    """Two firms filed same day: AAA profitable, ZZZ a loss-making hyper-grower
    with negative equity -> exercises every regime branch."""
    rows = [
        dict(ticker="AAA", as_of="2020-02-01", totalRevenue=100.0, netIncome=10.0,
             stockholdersEquity=50.0, freeCashflow=8.0, ebitda=20.0, totalAssets=200.0,
             grossMargins=0.40, revenueGrowth=0.05, sharesOutstanding=1000.0),
        dict(ticker="ZZZ", as_of="2020-02-01", totalRevenue=80.0, netIncome=-30.0,
             stockholdersEquity=-10.0, freeCashflow=-25.0, ebitda=-15.0, totalAssets=120.0,
             grossMargins=0.55, revenueGrowth=0.60, sharesOutstanding=2000.0),
    ]
    return pd.DataFrame(rows)


def test_regime_masks_earnings_metrics_but_keeps_robust_ones():
    fund = _synth_mixed_regime()
    idx = pd.bdate_range("2020-03-02", periods=3)
    close = pd.DataFrame({"AAA": 2.0, "ZZZ": 1.0}, index=idx)
    F = _derived_fields(fund, idx, close)
    d = idx[-1]

    # profitable AAA -> earnings/fcf/ebitda yields defined
    assert np.isfinite(F["earnings_yield"].loc[d, "AAA"])
    assert np.isfinite(F["fcf_yield"].loc[d, "AAA"])
    # loss-making ZZZ -> those are masked to NaN (a negative E/P is not "cheap")
    assert np.isnan(F["earnings_yield"].loc[d, "ZZZ"])
    assert np.isnan(F["fcf_yield"].loc[d, "ZZZ"])
    assert np.isnan(F["ebitda_to_ev"].loc[d, "ZZZ"])      # EBITDA<0
    assert np.isnan(F["book_yield"].loc[d, "ZZZ"])         # equity<0
    # but regime-ROBUST metrics stay valid for BOTH firms
    assert np.isfinite(F["sales_yield"].loc[d, "ZZZ"])     # revenue always +
    # gross_profitability = grossMargins*revenue/assets = 0.55*80/120
    assert abs(F["gross_profitability"].loc[d, "ZZZ"] - 0.55 * 80.0 / 120.0) < 1e-9
    assert abs(F["gross_profitability"].loc[d, "AAA"] - 0.40 * 100.0 / 200.0) < 1e-9

    print("\n=== SANITY CHECK: regime masking ===")
    print("  loss-maker ZZZ: earnings/fcf/ebitda/book yields -> NaN (undefined/non-monotone);")
    print(f"  robust metrics kept: sales_yield={F['sales_yield'].loc[d,'ZZZ']:.4f}, "
          f"gross_profitability={F['gross_profitability'].loc[d,'ZZZ']:.4f}. Validated.")


def test_regime_state_flags_exact_and_raw():
    fund = _synth_mixed_regime()
    idx = pd.bdate_range("2020-03-02", periods=3)
    F = _derived_fields(fund, idx, close=None)
    d = idx[-1]

    assert F["profitable"].loc[d, "AAA"] == 1.0 and F["profitable"].loc[d, "ZZZ"] == 0.0
    assert F["fcf_positive"].loc[d, "AAA"] == 1.0 and F["fcf_positive"].loc[d, "ZZZ"] == 0.0
    assert F["negative_equity"].loc[d, "AAA"] == 0.0 and F["negative_equity"].loc[d, "ZZZ"] == 1.0
    assert F["hyper_growth"].loc[d, "AAA"] == 0.0 and F["hyper_growth"].loc[d, "ZZZ"] == 1.0  # 0.60 > 0.25

    # flags enter the panel RAW as `f_<flag>` (not peer-standardized _vs_peers/_xs)
    state = build_state_panel({k: F[k] for k in
                               ("profitable", "fcf_positive", "negative_equity", "hyper_growth")})
    assert {"f_profitable", "f_fcf_positive", "f_negative_equity", "f_hyper_growth"}.issubset(state.columns)
    assert not any(c.endswith(("_vs_peers", "_xs")) for c in state.columns)

    print("\n=== SANITY CHECK: regime state flags ===")
    print("  AAA=[profitable=1,fcf+=1,negEq=0,hyper=0]  ZZZ=[0,0,1,1]; emitted raw as f_<flag>. Validated.")


# --------------------------------------------------------------------------- #
# Synthetic helpers                                                            #
# --------------------------------------------------------------------------- #
def _synth_fundamentals():
    """Two fiscal years for two tickers, with the fields valuation needs."""
    rows = [
        # ticker, as_of, revenue, netIncome, equity, fcf, ebitda, d2e, gm, rnd, shares
        ("AAA", "2019-02-01", 100.0, 10.0, 50.0, 8.0, 20.0, 0.5, 0.40, 5.0, 1000.0),
        ("AAA", "2020-02-01", 120.0, 15.0, 60.0, 12.0, 26.0, 0.6, 0.45, 7.0, 1100.0),
        ("BBB", "2019-02-01", 200.0, 5.0, 80.0, 4.0, 15.0, 1.0, 0.20, 2.0, 500.0),
        ("BBB", "2020-02-01", 210.0, 6.0, 82.0, 5.0, 16.0, 1.1, 0.21, 2.0, 505.0),
    ]
    cols = ["ticker", "as_of", "totalRevenue", "netIncome", "stockholdersEquity",
            "freeCashflow", "ebitda", "debtToEquity", "grossMargins",
            "researchAndDevelopment", "sharesOutstanding"]
    return pd.DataFrame(rows, columns=cols)


# --------------------------------------------------------------------------- #
# 1. _ratio                                                                    #
# --------------------------------------------------------------------------- #
def test_ratio_aligns_and_guards_denominator():
    idx = pd.bdate_range("2020-01-01", periods=3)
    num = pd.DataFrame({"A": [10.0, 10, 10], "B": [4.0, 4, 4]}, index=idx)
    den = pd.DataFrame({"A": [2.0, 0.0, -1.0], "B": [8.0, 8, 8]}, index=idx)

    out = _ratio(num, den, positive_den=True)
    assert out.loc[idx[0], "A"] == 5.0            # 10/2
    assert np.isnan(out.loc[idx[1], "A"])         # /0 -> NaN
    assert np.isnan(out.loc[idx[2], "A"])         # negative den masked
    assert (out["B"] == 0.5).all()

    print("\n=== SANITY CHECK: _ratio ===")
    print("  10/2=5, 10/0=NaN, 10/(-1)=NaN (positive_den), 4/8=0.5  -> all correct.")


# --------------------------------------------------------------------------- #
# 2. _fiscal_change_to_daily                                                   #
# --------------------------------------------------------------------------- #
def test_fiscal_change_pct_and_diff():
    fund = pd.DataFrame({
        "ticker": ["AAA", "AAA"],
        "as_of": ["2019-02-01", "2020-02-01"],
        "freeCashflow": [100.0, 150.0],
        "grossMargins": [0.40, 0.45],
    })
    idx = pd.bdate_range("2019-01-01", "2020-06-01")

    pct = _fiscal_change_to_daily(fund, "freeCashflow", idx, kind="pct")
    diff = _fiscal_change_to_daily(fund, "grossMargins", idx, kind="diff")

    # before the 2nd filing -> NaN; on/after -> YoY value, held forward.
    assert np.isnan(pct.loc[pd.Timestamp("2019-06-03"), "AAA"])
    after = pd.Timestamp("2020-03-02")
    assert abs(pct.loc[after, "AAA"] - 0.5) < 1e-9       # 150/100 - 1
    assert abs(diff.loc[after, "AAA"] - 0.05) < 1e-9     # 0.45 - 0.40

    print("\n=== SANITY CHECK: fiscal YoY change (PIT ffill) ===")
    print("  FCF 100->150 => +50% growth; gross margin 0.40->0.45 => +0.05 diff.")
    print("  NaN before the second filing is public -> no look-ahead.")


# --------------------------------------------------------------------------- #
# 3. _peer_relative: correct z + explosion guard                               #
# --------------------------------------------------------------------------- #
def test_peer_relative_zscore_correct():
    idx = pd.bdate_range("2020-01-01", periods=1)
    field = pd.DataFrame({"A": [4.0], "B": [1.0], "C": [2.0], "D": [3.0]}, index=idx)
    peers = {"A": {"B": 1.0, "C": 1.0, "D": 1.0}}

    rel = _peer_relative(field, peers)
    # peer mean=2, population std=sqrt(2/3)=0.8165 -> (4-2)/0.8165 = 2.449
    assert abs(rel.loc[idx[0], "A"] - 2.449) < 1e-2

    print("\n=== SANITY CHECK: peer-relative z-score ===")
    print(f"  A=4 vs peers[1,2,3] -> z={rel.loc[idx[0], 'A']:.3f} (expected ~2.449).")


def test_peer_relative_does_not_explode_on_degenerate_peers():
    idx = pd.bdate_range("2020-01-01", periods=2)
    # day 0: peers identical -> std 0 -> must be NaN, never inf/1e13
    # day 1: peers nearly identical -> huge raw z -> must be clipped to +-8
    field = pd.DataFrame({
        "A": [10.0, 10.0],
        "B": [5.0, 5.0],
        "C": [5.0, 5.0],
        "D": [5.0, 5.0 + 1e-9],
    }, index=idx)
    peers = {"A": {"B": 1.0, "C": 1.0, "D": 1.0}}

    rel = _peer_relative(field, peers, clip=8.0)
    assert np.isnan(rel.loc[idx[0], "A"]), "identical peers (std 0) must give NaN"
    assert abs(rel.loc[idx[1], "A"]) <= 8.0 + 1e-9, "near-degenerate std must be clipped"

    print("\n=== SANITY CHECK: peer-std explosion guard ===")
    print(f"  identical peers -> {rel.loc[idx[0], 'A']}; near-degenerate -> "
          f"{rel.loc[idx[1], 'A']:.1f} (clipped to +-8, not 1e13).")


# --------------------------------------------------------------------------- #
# 4. _derived_fields: exact valuation / dilution / R&D math                    #
# --------------------------------------------------------------------------- #
def test_derived_valuation_and_dilution_exact():
    fund = _synth_fundamentals()
    idx = pd.bdate_range("2020-03-01", periods=5)  # after the 2020 filing
    # constant price so market cap = shares * price is easy to reason about
    close = pd.DataFrame({"AAA": 2.0, "BBB": 3.0}, index=idx)

    F = _derived_fields(fund, idx, close)
    d = idx[-1]

    # AAA 2020: revenue120, ni15, equity60, fcf12, shares1100, price2 -> mcap=2200
    assert abs(F["earnings_yield"].loc[d, "AAA"] - 15.0 / 2200) < 1e-9
    assert abs(F["sales_yield"].loc[d, "AAA"] - 120.0 / 2200) < 1e-9
    assert abs(F["book_yield"].loc[d, "AAA"] - 60.0 / 2200) < 1e-9
    assert abs(F["fcf_yield"].loc[d, "AAA"] - 12.0 / 2200) < 1e-9
    assert abs(F["fcf_margin"].loc[d, "AAA"] - 12.0 / 120) < 1e-9
    assert abs(F["rd_intensity"].loc[d, "AAA"] - 7.0 / 120) < 1e-9
    # dilution: shares 1000 -> 1100 = +10%
    assert abs(F["shares_growth"].loc[d, "AAA"] - 0.10) < 1e-9
    # EV/EBITDA yield: EV = mcap + debt(d2e*equity=0.6*60=36) = 2236; ebitda 26
    assert abs(F["ebitda_to_ev"].loc[d, "AAA"] - 26.0 / 2236) < 1e-6
    # FCF/EV yield on the SAME EV (fcf 12) -> the new cross-sector cash-valuation yield
    assert abs(F["fcf_to_ev"].loc[d, "AAA"] - 12.0 / 2236) < 1e-6

    print("\n=== SANITY CHECK: derived valuation / dilution / R&D ===")
    print(f"  AAA E/P={F['earnings_yield'].loc[d,'AAA']:.5f}  S/P={F['sales_yield'].loc[d,'AAA']:.5f}"
          f"  EBITDA/EV={F['ebitda_to_ev'].loc[d,'AAA']:.5f}")
    print(f"  R&D intensity={F['rd_intensity'].loc[d,'AAA']:.4f}  shares growth="
          f"{F['shares_growth'].loc[d,'AAA']:.2%}  -> all match hand calc.")


def test_valuation_skipped_without_close():
    fund = _synth_fundamentals()
    idx = pd.bdate_range("2020-03-01", periods=5)
    F = _derived_fields(fund, idx, close=None)
    assert "earnings_yield" not in F      # needs a price
    assert "grossMargins" in F            # raw ratios still built
    print("\n=== SANITY CHECK: valuation gracefully skipped without prices ===")
    print("  no close -> no valuation yields, but raw ratios still produced.")


# --------------------------------------------------------------------------- #
# 5. Real data: panel is well-formed and bounded                               #
# --------------------------------------------------------------------------- #
def test_real_panel_wellformed_and_bounded(fundamental_panel):
    panel = fundamental_panel["panel"]
    assert {"date", "ticker"}.issubset(panel.columns)

    vp = [c for c in panel.columns if c.endswith("_vs_peers")]
    xs = [c for c in panel.columns if c.endswith("_xs")]
    assert len(vp) >= 8 and len(xs) >= 8

    # vs_peers winsorized to +-8; xs is a percentile in [0,1]
    vp_vals = panel[vp].to_numpy()
    assert np.nanmax(np.abs(vp_vals)) <= 8.0 + 1e-6, "vs_peers not winsorized"
    xs_vals = panel[xs].to_numpy()
    assert np.nanmin(xs_vals) >= 0.0 and np.nanmax(xs_vals) <= 1.0

    # Cross-sectional (_xs) coverage is universe-wide and robust to subsetting;
    # vs_peers is lower in a small test universe because a stock's peer basket
    # (from the full universe) may have <3 members inside the sample.
    covered_xs = [c for c in ["f_profitMargins_xs", "f_revenueGrowth_xs",
                              "f_returnOnEquity_xs"] if c in panel.columns]
    assert covered_xs, "expected core fundamental features to be present"
    cov_xs = panel[covered_xs].notna().mean().max()
    assert cov_xs > 0.3, f"core fundamental xs-coverage too low ({cov_xs:.1%})"
    assert panel[vp].notna().mean().max() > 0.02, "vs_peers has no coverage at all"

    print("\n=== SANITY CHECK: real fundamental panel ===")
    print(f"  {len(vp)} vs_peers + {len(xs)} xs features; "
          f"|vs_peers| max={np.nanmax(np.abs(vp_vals)):.2f} (<=8), xs in [0,1].")
    print(f"  core feature xs-coverage = {cov_xs:.0%}. Panel is well-formed.")


# --------------------------------------------------------------------------- #
# 6. Real data: marginal-effect signs make economic sense (coverage-gated)     #
# --------------------------------------------------------------------------- #
def test_headline_feature_signs_make_sense(fundamental_panel, real_pipeline):
    """Information Coefficient (mean daily Spearman) of the headline features vs
    the 20-day target. Requires valuation coverage (rebuilt fundamentals with
    sharesOutstanding); skips on the shares-sparse canonical file."""
    from scipy.stats import spearmanr

    panel = fundamental_panel["panel"]
    h = 20 if 20 in real_pipeline["labels_rank"] else real_pipeline["horizons"][0]
    tgt = real_pipeline["labels_rank"][h].stack().rename("target")
    tgt.index.set_names(["date", "ticker"], inplace=True)
    df = panel.merge(tgt.reset_index(), on=["date", "ticker"], how="inner")

    checks = {  # feature -> expected IC sign (robust priors only)
        "f_sales_yield_vs_peers": +1,     # cheap on sales outperforms
        "f_ebitda_to_ev_vs_peers": +1,    # cheap on EV/EBITDA outperforms
        "f_shares_growth_vs_peers": -1,   # dilution / heavy SBC underperforms
    }
    available = {f: s for f, s in checks.items()
                 if f in df.columns and df[[f, "target"]].dropna().shape[0] > 5000}
    if not available:
        pytest.skip("valuation/dilution coverage unavailable (regenerate fundamentals)")

    print("\n=== SANITY CHECK: headline feature IC signs ===")
    for f, want in available.items():
        sub = df[["date", f, "target"]].dropna()
        ic = sub.groupby("date").apply(
            lambda g: spearmanr(g[f], g["target"]).statistic if g[f].nunique() > 2 else np.nan,
            include_groups=False)
        mic = float(np.nanmean(ic))
        print(f"  {f:<32} IC={mic:+.4f}  expected sign={want:+d}")
        assert np.sign(mic) == want, f"{f} IC={mic:+.4f} has the wrong sign"
    print("  -> value (cheap) predicts up, dilution predicts down. Economically sound.")


# --------------------------------------------------------------------------- #
# 7. Yearly-TTM momentum features: math + point-in-time                        #
# --------------------------------------------------------------------------- #
def test_yearly_ttm_features_computed_correctly():
    """y_rev_growth, y_rev_growth_accel, y_earnings_growth, y_margin_vs_ttm are
    computed correctly from level columns and are strictly point-in-time."""
    fund = pd.DataFrame({
        "ticker":       ["AAA", "AAA", "AAA"],
        "as_of":        ["2019-02-01", "2020-02-01", "2021-02-01"],
        "totalRevenue": [100.0, 120.0, 132.0],
        "netIncome":    [10.0, 15.0, 18.0],
        "profitMargins":[0.10, 0.125, 0.136],
    })
    idx = pd.bdate_range("2019-01-01", "2021-06-01")

    F = _derived_fields(fund, idx, close=None)

    # ---- y_rev_growth ----
    assert "y_rev_growth" in F, "y_rev_growth missing from _derived_fields"
    before_2020 = pd.Timestamp("2020-01-15")
    assert np.isnan(F["y_rev_growth"].loc[before_2020, "AAA"]), \
        "y_rev_growth must be NaN before the second filing"
    after_2020 = pd.Timestamp("2020-03-02")  # Monday, in bdate_range
    assert abs(F["y_rev_growth"].loc[after_2020, "AAA"] - 0.20) < 1e-9, \
        f"expected +20% revenue growth, got {F['y_rev_growth'].loc[after_2020,'AAA']}"

    # ---- y_rev_growth_accel: change in YoY from period 2->3 ----
    assert "y_rev_growth_accel" in F, "y_rev_growth_accel missing"
    after_2021 = pd.Timestamp("2021-03-01")
    # 2020 YoY = 20%, 2021 YoY = 132/120-1 = 10% -> accel = 10% - 20% = -10%
    assert abs(F["y_rev_growth_accel"].loc[after_2021, "AAA"] - (-0.10)) < 1e-9, \
        f"expected accel=-10%, got {F['y_rev_growth_accel'].loc[after_2021,'AAA']}"

    # ---- y_earnings_growth ----
    assert "y_earnings_growth" in F, "y_earnings_growth missing"
    assert abs(F["y_earnings_growth"].loc[after_2020, "AAA"] - 0.50) < 1e-9, \
        f"expected +50% earnings growth, got {F['y_earnings_growth'].loc[after_2020,'AAA']}"

    # ---- y_margin_vs_ttm: YoY diff in profitMargins ----
    assert "y_margin_vs_ttm" in F, "y_margin_vs_ttm missing"
    expected_margin_chg = round(0.125 - 0.10, 9)
    assert abs(F["y_margin_vs_ttm"].loc[after_2020, "AAA"] - expected_margin_chg) < 1e-9, \
        f"expected margin chg={expected_margin_chg}, got {F['y_margin_vs_ttm'].loc[after_2020,'AAA']}"

    print("\n=== SANITY CHECK: yearly-TTM momentum features ===")
    print(f"  y_rev_growth    2020={F['y_rev_growth'].loc[after_2020,'AAA']:.2%} (expected +20%)")
    print(f"  y_rev_growth_accel 2021={F['y_rev_growth_accel'].loc[after_2021,'AAA']:.2%} (expected -10%)")
    print(f"  y_earnings_growth 2020={F['y_earnings_growth'].loc[after_2020,'AAA']:.2%} (expected +50%)")
    print(f"  y_margin_vs_ttm 2020={F['y_margin_vs_ttm'].loc[after_2020,'AAA']:.4f} (expected +0.025)")
    print("  All NaN before second filing, correct values after -> strictly point-in-time. Validated.")


# --------------------------------------------------------------------------- #
# 8. Valuation MEAN-REVERSION (self-history z-score)                           #
# --------------------------------------------------------------------------- #
def test_self_history_z_mean_reversion():
    """`_self_history_z` z-scores each ticker vs its OWN trailing window: NaN
    until min_periods, negative when the yield sits BELOW its own norm
    (expensive), positive when ABOVE (cheap). Strictly trailing -> no leak."""
    idx = pd.bdate_range("2020-01-01", periods=300)
    # yield: long plateau at 0.05, then a dip to 0.02 (expensive vs own past),
    # then a jump to 0.08 (cheap vs own past).
    vals = np.concatenate([np.full(200, 0.05), np.full(50, 0.02), np.full(50, 0.08)])
    yld = pd.DataFrame({"AAA": vals}, index=idx)

    z = _self_history_z(yld, window=120, min_periods=60)

    # insufficient history -> NaN
    assert z["AAA"].iloc[:59].isna().all(), "z must be NaN before min_periods"
    # into the 0.02 regime: current < own trailing mean -> negative z
    z_dip = z["AAA"].iloc[220]
    assert np.isfinite(z_dip) and z_dip < 0, f"expected negative z in the dip, got {z_dip}"
    # into the 0.08 regime: current > own trailing mean -> positive z
    z_jump = z["AAA"].iloc[-1]
    assert np.isfinite(z_jump) and z_jump > 0, f"expected positive z in the jump, got {z_jump}"
    # winsorized
    assert z["AAA"].abs().max() <= 8.0 + 1e-9

    print("\n=== SANITY CHECK: valuation mean-reversion (self-history z) ===")
    print(f"  yield below own norm -> z={z_dip:+.2f} (<0);  above own norm -> z={z_jump:+.2f} (>0)")
    print("  NaN before min_periods, trailing-only window -> point-in-time. Validated.")


# --------------------------------------------------------------------------- #
# 9. Distress / solvency + S&M + M&A + SBC derived fields                      #
# --------------------------------------------------------------------------- #
def _synth_fundamentals_rich():
    """Two annual filings for AAA carrying every raw level the refined features
    need. Hand-computable so the ratios can be checked exactly."""
    rows = [
        dict(ticker="AAA", as_of="2019-02-01",
             totalRevenue=100.0, netIncome=10.0, ebitda=25.0,
             cash=20.0, longTermDebt=40.0, shortTermDebt=10.0,
             totalLiabilities=80.0, currentAssets=60.0, currentLiabilities=30.0,
             interestExpense=5.0, goodwill=15.0, totalAssets=200.0,
             sellingGeneralAdmin=20.0, stockBasedComp=4.0, acquisitions=6.0,
             operatingCashFlow=18.0, profitMargins=0.10),
        dict(ticker="AAA", as_of="2020-02-03",
             totalRevenue=120.0, netIncome=15.0, ebitda=30.0,
             cash=25.0, longTermDebt=50.0, shortTermDebt=10.0,
             totalLiabilities=90.0, currentAssets=66.0, currentLiabilities=33.0,
             interestExpense=6.0, goodwill=30.0, totalAssets=240.0,
             sellingGeneralAdmin=22.0, stockBasedComp=6.0, acquisitions=12.0,
             operatingCashFlow=24.0, profitMargins=0.125),
    ]
    return pd.DataFrame(rows)


def test_distress_sga_ma_sbc_features_exact():
    fund = _synth_fundamentals_rich()
    idx = pd.bdate_range("2019-01-01", "2020-06-01")
    F = _derived_fields(fund, idx, close=None)   # no close -> valuation skipped, rest built

    d = pd.Timestamp("2020-03-02")   # after the 2020-02-03 filing (uses y2 values)

    # ---- distress / solvency ----
    # net debt = (ltd 50 + std 10) - cash 25 = 35 ; / ebitda 30 = 1.1667
    assert abs(F["net_debt_to_ebitda"].loc[d, "AAA"] - 35.0 / 30.0) < 1e-9
    assert abs(F["interest_coverage"].loc[d, "AAA"] - 30.0 / 6.0) < 1e-9   # ebitda/interest
    assert abs(F["current_ratio"].loc[d, "AAA"] - 66.0 / 33.0) < 1e-9       # 2.0
    assert abs(F["cash_to_debt"].loc[d, "AAA"] - 25.0 / 60.0) < 1e-9

    # ---- S&M efficiency ----
    assert abs(F["sga_intensity"].loc[d, "AAA"] - 22.0 / 120.0) < 1e-9
    assert abs(F["sga_growth"].loc[d, "AAA"] - (22.0 / 20.0 - 1.0)) < 1e-9   # +10%
    # operating leverage = rev growth (20%) - SG&A growth (10%) = +10%
    assert abs(F["operating_leverage"].loc[d, "AAA"] - 0.10) < 1e-9

    # ---- M&A ----
    assert abs(F["acquisition_intensity"].loc[d, "AAA"] - 12.0 / 240.0) < 1e-9   # acq/assets
    assert abs(F["goodwill_growth"].loc[d, "AAA"] - (30.0 / 15.0 - 1.0)) < 1e-9  # +100%

    # ---- SBC ----
    assert abs(F["sbc_intensity"].loc[d, "AAA"] - 6.0 / 120.0) < 1e-9
    assert abs(F["sbc_to_ocf"].loc[d, "AAA"] - 6.0 / 24.0) < 1e-9

    # ---- point-in-time: growth features NaN before the second filing ----
    before = pd.Timestamp("2019-06-03")
    assert np.isnan(F["goodwill_growth"].loc[before, "AAA"]), "goodwill_growth leaked"
    assert np.isnan(F["operating_leverage"].loc[before, "AAA"]), "operating_leverage leaked"
    # level ratios use the y1 filing before y2 is public (still no look-ahead)
    assert abs(F["net_debt_to_ebitda"].loc[before, "AAA"] - (40.0 + 10.0 - 20.0) / 25.0) < 1e-9

    print("\n=== SANITY CHECK: distress / S&M / M&A / SBC ===")
    print(f"  net_debt/EBITDA={F['net_debt_to_ebitda'].loc[d,'AAA']:.3f}  "
          f"interest_cov={F['interest_coverage'].loc[d,'AAA']:.1f}x  "
          f"current={F['current_ratio'].loc[d,'AAA']:.1f}  "
          f"cash/debt={F['cash_to_debt'].loc[d,'AAA']:.3f}")
    print(f"  sga_intensity={F['sga_intensity'].loc[d,'AAA']:.3f}  "
          f"op_leverage={F['operating_leverage'].loc[d,'AAA']:+.2%}  "
          f"acq_intensity={F['acquisition_intensity'].loc[d,'AAA']:.3f}  "
          f"goodwill_growth={F['goodwill_growth'].loc[d,'AAA']:+.0%}")
    print(f"  sbc_intensity={F['sbc_intensity'].loc[d,'AAA']:.3f}  "
          f"sbc/OCF={F['sbc_to_ocf'].loc[d,'AAA']:.2f}")
    print("  All ratios match hand calc; growth NaN before 2nd filing -> point-in-time. Validated.")


def test_panel_emits_vs_hist_columns():
    """End-to-end: the fundamental panel gains `f_<yield>_vs_hist` mean-reversion
    columns (built with a small window so a short synthetic history suffices)."""
    fund = _synth_fundamentals()   # has revenue/netIncome/equity/fcf/ebitda/shares
    idx = pd.bdate_range("2019-01-01", "2020-06-01")
    n = len(idx)
    # varying prices so the valuation yields move day-to-day (else std=0 -> no z)
    close = pd.DataFrame({"AAA": np.linspace(1.5, 3.0, n),
                          "BBB": np.linspace(2.5, 3.5, n)}, index=idx)
    peers = {"AAA": {"BBB": 1.0}, "BBB": {"AAA": 1.0}}

    panel = build_fundamental_feature_panel(
        fund, peers, idx, stock_close=close,
        hist_window=60, hist_min_periods=20,
    )

    hist_cols = [c for c in panel.columns if c.endswith("_vs_hist")]
    assert "f_earnings_yield_vs_hist" in panel.columns, f"no vs_hist columns: {list(panel.columns)}"
    assert panel["f_earnings_yield_vs_hist"].notna().any(), "vs_hist column is entirely NaN"

    print("\n=== SANITY CHECK: panel self-history columns ===")
    print(f"  emitted {len(hist_cols)} f_*_vs_hist columns: {sorted(hist_cols)}")
    print("  f_earnings_yield_vs_hist present with non-null values -> mean-reversion wired in. Validated.")


# --------------------------------------------------------------------------- #
# 10. Accruals + profitability pass-throughs (exact)                           #
# --------------------------------------------------------------------------- #
def test_accruals_and_profitability_passthrough_exact():
    fund = pd.DataFrame([
        dict(ticker="AAA", as_of="2019-02-01", totalRevenue=100.0, netIncome=10.0,
             freeCashflow=8.0, operatingMargins=0.18, profitMargins=0.10,
             returnOnEquity=0.20, debtToEquity=0.5, grossMargins=0.40),
        dict(ticker="AAA", as_of="2020-02-01", totalRevenue=120.0, netIncome=15.0,
             freeCashflow=12.0, operatingMargins=0.20, profitMargins=0.125,
             returnOnEquity=0.25, debtToEquity=0.6, grossMargins=0.45),
    ])
    idx = pd.bdate_range("2020-03-01", periods=3)   # after the 2020 filing
    F = _derived_fields(fund, idx, close=None)
    d = idx[-1]

    # accruals = (netIncome - freeCashflow) / revenue = (15 - 12)/120
    assert abs(F["accruals"].loc[d, "AAA"] - (15.0 - 12.0) / 120.0) < 1e-9
    # raw ratios pass straight through from the fiscal history (latest public value)
    assert abs(F["profitMargins"].loc[d, "AAA"] - 0.125) < 1e-9
    assert abs(F["operatingMargins"].loc[d, "AAA"] - 0.20) < 1e-9
    assert abs(F["returnOnEquity"].loc[d, "AAA"] - 0.25) < 1e-9
    assert abs(F["debtToEquity"].loc[d, "AAA"] - 0.6) < 1e-9

    print("\n=== SANITY CHECK: accruals + profitability pass-through ===")
    print(f"  accruals=(15-12)/120={F['accruals'].loc[d,'AAA']:.4f}; "
          f"profitMargins={F['profitMargins'].loc[d,'AAA']}, ROE={F['returnOnEquity'].loc[d,'AAA']}, "
          f"D/E={F['debtToEquity'].loc[d,'AAA']} -> exact.")


# --------------------------------------------------------------------------- #
# 11. Cross-sectional (_xs) percentile ranks across the universe               #
# --------------------------------------------------------------------------- #
def test_xs_percentile_ranks_across_universe():
    fund = pd.DataFrame([
        dict(ticker=t, as_of="2020-02-01", totalRevenue=100.0, netIncome=10.0,
             stockholdersEquity=50.0, freeCashflow=8.0, profitMargins=pm)
        for t, pm in [("AAA", 0.10), ("BBB", 0.20), ("CCC", 0.30)]
    ])
    idx = pd.bdate_range("2020-03-02", periods=3)
    panel = build_fundamental_feature_panel(fund, peer_dict={}, trading_index=idx,
                                            stock_close=None)
    d = idx[-1]
    xs = panel[panel["date"] == d].set_index("ticker")["f_profitMargins_xs"]

    # rank(pct) over [0.10, 0.20, 0.30] -> [1/3, 2/3, 1.0]
    assert abs(xs["AAA"] - 1 / 3) < 1e-9
    assert abs(xs["BBB"] - 2 / 3) < 1e-9
    assert abs(xs["CCC"] - 1.0) < 1e-9

    print("\n=== SANITY CHECK: cross-sectional _xs percentile ===")
    print(f"  profitMargins [0.10,0.20,0.30] -> xs [{xs['AAA']:.3f},{xs['BBB']:.3f},{xs['CCC']:.3f}]"
          " = [1/3,2/3,1] -> monotone universe percentile. Validated.")


# --------------------------------------------------------------------------- #
# 12. Signed-base %-growth: negative when losing money (INTENDED behavior)     #
# --------------------------------------------------------------------------- #
def test_pct_growth_signed_base_is_negative_when_losing_money():
    """Locks the INTENDED behavior: YoY %-growth on a signed quantity (earnings,
    FCF) goes negative when the base is non-positive. A negative growth is a valid
    'no growth / losing money' signal for the model -- kept on purpose, not a bug.
    Revenue / shares / SG&A / goodwill are non-negative so they are always
    well-defined."""
    fund = pd.DataFrame({
        "ticker": ["AAA", "AAA"],
        "as_of": ["2019-02-01", "2020-02-01"],
        "netIncome": [-50.0, 50.0],
    })
    idx = pd.bdate_range("2019-01-01", "2020-06-01")
    g = _fiscal_change_to_daily(fund, "netIncome", idx, kind="pct")
    val = g.loc[pd.Timestamp("2020-03-02"), "AAA"]

    # (50 - (-50)) / (-50) = -2.0 -> negative growth off a loss base, as intended
    assert abs(val - (-2.0)) < 1e-9

    print("\n=== SANITY CHECK: signed-base %-growth (intended) ===")
    print(f"  netIncome off a loss base -> growth={val:+.0%} (negative = weak/loss signal, kept).")
    print("  Revenue / shares / SG&A / goodwill are non-negative -> always well-defined.")


# --------------------------------------------------------------------------- #
# 13. Enterprise value = mktcap + real debt + SBC - cash (exact)               #
# --------------------------------------------------------------------------- #
def test_valuation_engine_kpis_exact():
    """Altman Z, PEGY, operating-leverage elasticity, and the REIT/energy EV
    multiples, on a 2-year history with a known price."""
    y19, y20 = "2019-12-31", "2020-12-31"
    fund = pd.DataFrame([
        # GEN: 2 years -> growth/elasticity/PEGY; full Altman inputs
        dict(ticker="GEN", as_of=y19, totalRevenue=100.0, netIncome=10.0, operatingIncome=12.0,
             ebitda=15.0, grossProfit=40.0, totalAssets=200.0, currentAssets=80.0,
             currentLiabilities=40.0, retainedEarnings=50.0, totalLiabilities=100.0,
             longTermDebt=50.0, shortTermDebt=10.0, cash=20.0,
             sharesOutstanding=100.0, dilutedShares=100.0, dividendsPaid=4.0),
        dict(ticker="GEN", as_of=y20, totalRevenue=120.0, netIncome=15.0, operatingIncome=18.0,
             ebitda=18.0, grossProfit=50.0, totalAssets=200.0, currentAssets=80.0,
             currentLiabilities=40.0, retainedEarnings=50.0, totalLiabilities=100.0,
             longTermDebt=50.0, shortTermDebt=10.0, cash=20.0,
             sharesOutstanding=100.0, dilutedShares=100.0, dividendsPaid=4.0),
        # REIT: FFO yield + implied cap rate
        dict(ticker="REI", as_of=y20, totalRevenue=300.0, netIncome=50.0, operatingIncome=90.0,
             depAmort=100.0, gainOnDispositions=10.0, realEstateNet=2000.0,
             longTermDebt=800.0, cash=50.0, sharesOutstanding=100.0, dilutedShares=100.0),
        # Energy: EV/EBITDAX
        dict(ticker="OIL", as_of=y20, totalRevenue=1000.0, netIncome=120.0, operatingIncome=180.0,
             depAmort=110.0, explorationExpense=70.0, oilGasPropertyNet=5000.0,
             longTermDebt=1000.0, cash=100.0, sharesOutstanding=100.0, dilutedShares=100.0),
    ])
    idx = pd.bdate_range("2021-03-01", periods=3)
    close = pd.DataFrame({t: 2.0 for t in ("GEN", "REI", "OIL")}, index=idx)
    F = _derived_fields(fund, idx, close)   # annual history -> yoy_periods default 1
    d = idx[-1]

    # GEN: PE = 200 mcap / 15 NI = 13.33; growth 50%, div yield 2% -> PEGY = 13.33/52
    assert F["pegy"].loc[d, "GEN"] == pytest.approx((200 / 15) / (50 + 2), rel=1e-6)
    # operating leverage elasticity = %ΔOI (50%) / %ΔRev (20%) = 2.5
    assert F["operating_leverage_elasticity"].loc[d, "GEN"] == pytest.approx(2.5, rel=1e-6)
    # Altman Z = 1.2*.2 + 1.4*.25 + 3.3*.09 + 0.6*2.0 + 1.0*.6 = 2.687
    assert F["altman_z"].loc[d, "GEN"] == pytest.approx(2.687, abs=1e-3)
    # REIT: FFO = 50+100-10 = 140 -> ffo_yield 140/200 = 0.70; EV=200+800-50=950 -> cap 190/950
    assert F["ffo_yield"].loc[d, "REI"] == pytest.approx(140 / 200)
    assert F["implied_cap_rate"].loc[d, "REI"] == pytest.approx(190 / 950)
    # Energy: EBITDAX=180+110+70=360; EV=200+1000-100=1100 -> 360/1100
    assert F["ebitdax_to_ev"].loc[d, "OIL"] == pytest.approx(360 / 1100)
    # gating: non-REIT has no ffo_yield, non-energy has no ebitdax_to_ev
    assert np.isnan(F["ffo_yield"].loc[d, "OIL"]) and np.isnan(F["ebitdax_to_ev"].loc[d, "REI"])

    print("\n=== SANITY CHECK: valuation-engine KPIs ===")
    print(f"  GEN PEGY={F['pegy'].loc[d,'GEN']:.3f} op_lev_elasticity={F['operating_leverage_elasticity'].loc[d,'GEN']:.2f} "
          f"AltmanZ={F['altman_z'].loc[d,'GEN']:.3f}")
    print(f"  REIT ffo_yield={F['ffo_yield'].loc[d,'REI']:.3f} implied_cap={F['implied_cap_rate'].loc[d,'REI']:.3f}; "
          f"OIL ev/ebitdax_yield={F['ebitdax_to_ev'].loc[d,'OIL']:.3f}; sector-gated. Validated.")


def test_pegy_uses_projected_eps_growth():
    """PEGY's growth term prefers PROJECTED EPS growth (NTM/TTM-1 from the earnings
    archive) when earnings_history is supplied, and falls back to TTM otherwise."""
    fund = pd.DataFrame([dict(ticker="A", as_of="2025-11-05", totalRevenue=100.0,
                              netIncome=50.0, sharesOutstanding=10.0, dividendsPaid=2.0,
                              totalAssets=200.0)])
    earn = pd.DataFrame({
        "ticker": ["A"] * 5,
        "earnings_date": ["2025-02-01", "2025-05-01", "2025-08-01", "2025-11-01", "2026-02-15"],
        "eps_estimate": [1.0, 1.0, 1.0, 1.0, 1.5],
        "eps_actual":   [1.1, 1.2, 1.3, 1.4, np.nan],
    })
    idx = pd.bdate_range("2026-01-05", "2026-01-30")
    close = pd.DataFrame({"A": 100.0}, index=idx)
    d = idx[-1]

    # with projected growth: PE = mcap(1000)/NI(50)=20; growth = 5.4/5.0-1 = 8%; div yield 0.2%
    F_proj = _derived_fields(fund, idx, close, yoy_periods=4, earnings_history=earn)
    assert F_proj["pegy"].loc[d, "A"] == pytest.approx(20 / (8 + 0.2), rel=1e-3)
    # without earnings: single fundamentals row -> TTM growth undefined -> PEGY NaN (fallback path)
    F_ttm = _derived_fields(fund, idx, close, yoy_periods=4)
    assert "pegy" not in F_ttm or np.isnan(F_ttm["pegy"].loc[d, "A"])

    print("\n=== SANITY CHECK: PEGY uses projected EPS growth ===")
    print(f"  projected growth 8% -> PEGY = 20/(8+0.2) = {F_proj['pegy'].loc[d,'A']:.3f}; "
          f"TTM fallback undefined here (single filing). Validated.")


def test_true_enterprise_value_fully_diluted():
    """True EV = diluted-shares*price + total debt + leases + minority interest
    - cash - short-term investments. SBC is NOT part of EV (corrected definition)."""
    fund = pd.DataFrame([
        dict(ticker="AAA", as_of="2020-02-01", totalRevenue=120.0, netIncome=15.0,
             stockholdersEquity=60.0, freeCashflow=12.0, ebitda=26.0,
             sharesOutstanding=1000.0, dilutedShares=1100.0,      # diluted > basic
             longTermDebt=40.0, shortTermDebt=10.0,
             operatingLeaseLiability=8.0, financeLeaseLiability=2.0,
             minorityInterest=5.0, cash=25.0, shortTermInvestments=15.0,
             stockBasedComp=6.0),                                  # present, must NOT affect EV
    ])
    idx = pd.bdate_range("2020-03-02", periods=3)
    close = pd.DataFrame({"AAA": 2.0}, index=idx)
    F = _derived_fields(fund, idx, close)
    d = idx[-1]

    # EV = FD mcap (1100*2=2200) + debt(50) + leases(10) + minority(5) - cash(25) - STI(15)
    ev = 1100 * 2.0 + 50.0 + 10.0 + 5.0 - 25.0 - 15.0     # = 2225
    assert abs(F["ebitda_to_ev"].loc[d, "AAA"] - 26.0 / ev) < 1e-9
    assert abs(F["fcf_to_ev"].loc[d, "AAA"] - 12.0 / ev) < 1e-9
    # uses DILUTED (2200) not basic (2000) for the equity value
    ev_basic = 1000 * 2.0 + 50.0 + 10.0 + 5.0 - 25.0 - 15.0
    assert abs(ev - ev_basic) == 200.0
    # SBC=6 is present but excluded: EV would be 2231 if SBC were added -> assert it is NOT
    assert abs(F["ebitda_to_ev"].loc[d, "AAA"] - 26.0 / (ev + 6.0)) > 1e-9

    print("\n=== SANITY CHECK: True (fully-diluted) enterprise value ===")
    print(f"  EV = 2200 FD-mcap + 50 debt + 10 leases + 5 minority - 25 cash - 15 STI = {ev:.0f}; "
          f"EBITDA/EV = 26/{ev:.0f} = {F['ebitda_to_ev'].loc[d,'AAA']:.5f}; "
          f"FCF/EV = 12/{ev:.0f}. Diluted (not basic) shares used; SBC excluded. Exact.")


# --------------------------------------------------------------------------- #
# Memory-safe scoped facts read: only the pension tags the builder uses         #
# --------------------------------------------------------------------------- #
class _FakeStore:
    """No `.engine` (forces the in-memory fallback) and no `.exists` (guard skipped via hasattr)."""
    def __init__(self, df): self.df = df
    def load(self, table, columns=None): return self.df.copy()


class _FakeCtx:
    def __init__(self, store): self.store = store; self.log = logging.getLogger("test")


def test_load_tagged_facts_reads_only_needed_tags():
    """`load_notes_num_scoped` returns ONLY the 2 footnote pension tags the panel reads — the other
    8 footnote tags are never materialised (the notes_num waste this optimization targets)."""
    rows = []
    for tag in (_FN_PBO_TAG, _FN_PLAN_ASSETS_TAG, "SomeOtherFootnoteTag", "AnotherUnusedTag"):
        rows.append({"ticker": "AAA", "tag": tag, "ddate": "2024-12-31", "qtrs": 0,
                     "value": 100.0, "filed": "2025-02-14"})
    ctx = _FakeCtx(_FakeStore(pd.DataFrame(rows)))
    out = load_notes_num_scoped(ctx)
    assert set(out["tag"]) == {_FN_PBO_TAG, _FN_PLAN_ASSETS_TAG}, "non-pension footnote tags leaked in"
    assert len(out) == 2, f"expected only the 2 pension tags, got {len(out)}"
    # no match / empty -> None (builder then treats pension as unavailable)
    assert load_tagged_facts(ctx, "notes_num", ("NoSuchTag",)) is None
    empty_ctx = _FakeCtx(_FakeStore(pd.DataFrame(columns=["ticker", "tag", "value"])))
    assert load_notes_num_scoped(empty_ctx) is None
    print("\n=== SANITY CHECK: scoped facts read ===")
    print(f"  notes_num (10 tags in prod) -> only {_FN_PBO_TAG} + {_FN_PLAN_ASSETS_TAG} loaded "
          "(2/4 synthetic rows); unused footnote tags dropped; no-match/empty -> None. Validated.")


if __name__ == "__main__":
    test_load_tagged_facts_reads_only_needed_tags()


# --------------------------------------------------------------------------- #
# Regression: a None cell (object-dtype field frame) must NOT crash the peer   #
# panel — it should be treated as a missing value (the earnings_call_embedding  #
# "unsupported operand type(s): NoneType and float" bug at _peer_relative).     #
# --------------------------------------------------------------------------- #
def test_peer_panel_tolerates_none_cells():
    from src.data_aggregate.utils.fundamental_features import _peer_relative

    idx = pd.bdate_range("2024-01-01", periods=4)
    tickers = ["A", "B", "C", "D"]
    peers = {t: {p: 1.0 for p in tickers if p != t} for t in tickers}

    # OBJECT-dtype frame carrying a Python None (what reaches the panel on the full
    # machine when a KPI is genuinely absent for a name) mixed with real floats.
    fdf = pd.DataFrame(
        [[1.0, 2.0, 3.0, 4.0],
         [None, 2.5, 3.5, 4.5],          # <- None cell (ticker A missing on day 2)
         [1.2, None, 3.2, 4.2],          # <- None cell (ticker B missing on day 3)
         [1.3, 2.3, 3.3, 4.3]],
        index=idx, columns=tickers, dtype="object",
    )
    assert fdf["A"].dtype == object and fdf.isin([None]).to_numpy().any()

    # pre-fix, _peer_relative on the raw object frame raises the exact reported error
    with pytest.raises(TypeError):
        _peer_relative(fdf, peers)

    # the shared entry point coerces None -> NaN, so the panel builds cleanly
    panel = build_peer_relative_panel({"kpi": fdf}, peers)
    assert not panel.empty
    assert {"f_kpi_vs_peers", "f_kpi_xs"}.issubset(panel.columns)
    panel = panel.set_index(["date", "ticker"])
    # the None cells surface as NaN features (missing), NOT a crash
    assert np.isnan(panel.loc[(idx[1], "A"), "f_kpi_xs"])
    assert np.isnan(panel.loc[(idx[2], "B"), "f_kpi_xs"])
    # a present cell still produces a finite percentile rank
    assert np.isfinite(panel.loc[(idx[0], "A"), "f_kpi_xs"])

    print("\n=== SANITY CHECK: peer panel tolerates None cells ===")
    print("  raw _peer_relative raises TypeError('NoneType' vs float) on an object frame; "
          "build_peer_relative_panel now coerces None -> NaN, builds f_kpi_{vs_peers,xs}, "
          "and the missing cells are NaN (not a crash). earnings_call_embedding bug fixed.")


if __name__ == "__main__":
    test_peer_panel_tolerates_none_cells()


# --------------------------------------------------------------------------- #
# Memory: the daily wide frames + long panels are float32 (halves the resident #
# footprint that was OOM-killing the fundamental aggregation group, SIGKILL/-9).#
# --------------------------------------------------------------------------- #
def test_panel_features_are_float32_for_memory():
    tickers = ["A", "B", "C", "D", "E", "F"]
    quarters = pd.date_range("2018-02-01", periods=8, freq="2QS")     # 8 filings/ticker
    rows = []
    for i, t in enumerate(tickers):
        for k, q in enumerate(quarters):
            rows.append(dict(ticker=t, as_of=q.strftime("%Y-%m-%d"),
                             totalRevenue=100.0 + 10 * i + 5 * k, netIncome=10.0 + i + k,
                             stockholdersEquity=50.0 + 5 * i, freeCashflow=8.0 + i + k,
                             ebitda=20.0 + 2 * i, debtToEquity=0.5 + 0.1 * i,
                             grossMargins=0.30 + 0.02 * i, researchAndDevelopment=5.0 + i,
                             sharesOutstanding=1000.0 + 100 * i))
    fund = pd.DataFrame(rows)
    idx = pd.bdate_range("2018-01-01", "2022-01-01")
    close = pd.DataFrame({t: 10.0 + i for i, t in enumerate(tickers)}, index=idx)
    peers = {t: {p: 1.0 for p in tickers if p != t} for t in tickers}   # 5 mutual peers >= min_peers

    panel = build_fundamental_feature_panel(fund, peers, idx, stock_close=close)
    feat_cols = [c for c in panel.columns if c not in ("date", "ticker")]
    assert feat_cols, "no fundamental features produced"
    non_f32 = {c: str(panel[c].dtype) for c in feat_cols if panel[c].dtype != np.float32}
    assert not non_f32, f"feature columns must be float32, found: {non_f32}"

    mem32 = panel[feat_cols].memory_usage(deep=True).sum()
    mem64 = panel[feat_cols].astype("float64").memory_usage(deep=True).sum()
    print("\n=== SANITY CHECK: fundamental panel memory (float32) ===")
    print(f"  {len(feat_cols)} feature columns, all float32; panel {panel.shape[0]} rows.")
    print(f"  feature-block memory {mem32/1e6:.2f} MB vs float64 {mem64/1e6:.2f} MB "
          f"(~{mem64/mem32:.1f}x smaller) -> halves the resident footprint that SIGKILL/-9'd "
          "the aggregation on the full-history rebuild.")


if __name__ == "__main__":
    test_panel_features_are_float32_for_memory()
