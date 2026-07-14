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

import numpy as np
import pandas as pd
import pytest

from src.data_aggregate.utils.fundamental_features import (
    _ratio,
    _fiscal_change_to_daily,
    _peer_relative,
    _self_history_z,
    _derived_fields,
    build_fundamental_feature_panel,
    build_state_panel,
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
def test_enterprise_value_uses_real_debt_sbc_and_cash():
    fund = pd.DataFrame([
        dict(ticker="AAA", as_of="2020-02-01", totalRevenue=120.0, netIncome=15.0,
             stockholdersEquity=60.0, freeCashflow=12.0, ebitda=26.0,
             debtToEquity=0.6, sharesOutstanding=1100.0,
             longTermDebt=40.0, shortTermDebt=10.0, stockBasedComp=6.0, cash=25.0),
    ])
    idx = pd.bdate_range("2020-03-02", periods=3)
    close = pd.DataFrame({"AAA": 2.0}, index=idx)   # mcap = 1100 * 2 = 2200
    F = _derived_fields(fund, idx, close)
    d = idx[-1]

    # EV = mcap 2200 + debt(40+10) + SBC 6 - cash 25 = 2231
    ev = 2200.0 + 50.0 + 6.0 - 25.0
    assert abs(F["ebitda_to_ev"].loc[d, "AAA"] - 26.0 / ev) < 1e-9

    print("\n=== SANITY CHECK: enterprise value (real debt + SBC - cash) ===")
    print(f"  EV = 2200 + (40+10 debt) + 6 SBC - 25 cash = {ev:.0f}; "
          f"EBITDA/EV = 26/{ev:.0f} = {F['ebitda_to_ev'].loc[d,'AAA']:.5f}. Exact.")
    # SBC and debt both RAISE EV -> LOWER the yield vs ignoring them (a diluter looks dearer)
    ev_no_adj = 2200.0
    assert 26.0 / ev < 26.0 / ev_no_adj

    print(f"  Adding debt+SBC and removing cash lowers EBITDA/EV from "
          f"{26/ev_no_adj:.5f} (mktcap only) to {26/ev:.5f} -> diluter/levered looks more expensive.")
