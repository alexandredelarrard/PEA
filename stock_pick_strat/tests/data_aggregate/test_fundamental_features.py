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
    _derived_fields,
    build_fundamental_feature_panel,
)


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
