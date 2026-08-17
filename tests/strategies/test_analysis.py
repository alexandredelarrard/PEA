"""
Per-strategy + portfolio analysis modules (src/strategies/analysis, src/portfolio/analysis).
Validates on synthetic data that: L/S IC is high for a signal aligned with forward returns and
the neutrality metrics compute; the long-book / portfolio correlation analyses return a matrix +
average pairwise corr and save their plots.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategies.analysis.common import daily_ic, rolling_beta, rolling_pairwise_corr


def test_rolling_pairwise_corr_survives_misaligned_calendar():
    idx = pd.bdate_range("2023-01-01", periods=600)
    rng = np.random.default_rng(7)
    a = pd.Series(rng.normal(0, 0.01, len(idx)), index=idx)        # full-calendar sleeve
    b = pd.Series(rng.normal(0, 0.01, len(idx)), index=idx)
    c = pd.Series(rng.normal(0, 0.01, len(idx)), index=idx)
    c[rng.random(len(idx)) < 0.1] = np.nan                         # L/S-like: ~10% dates missing
    df = pd.DataFrame({"long_book": a, "trend_cta": b, "ls_equity": c})
    pair_corr, avg = rolling_pairwise_corr(df, window=126)
    # every pair (incl. the misaligned ls_equity ones) must produce real rolling values
    for name, s in pair_corr.items():
        assert s.dropna().shape[0] > 100, f"pair {name} rolling corr is empty (calendar-blanked)"
    assert avg.dropna().shape[0] > 100
    print("\n=== SANITY CHECK: pairwise-corr calendar robustness ===")
    print(f"  all {len(pair_corr)} pairs (incl. 10%-sparse ls_equity) have rolling corr. Validated.")
from src.strategies.analysis.ls_analysis import analyze_ls
from src.strategies.analysis.long_book_analysis import analyze_long_book
from src.portfolio.analysis import analyze_portfolio


def test_daily_ic_detects_a_predictive_signal():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2020-01-01", periods=200)
    tickers = [f"T{i}" for i in range(30)]
    stock_ret = pd.DataFrame(rng.normal(0, 0.02, (len(idx), len(tickers))), index=idx, columns=tickers)
    # signal = a noised version of the NEXT-30d forward return -> IC should be strongly positive
    fwd = stock_ret[::-1].rolling(30, min_periods=18).sum()[::-1].shift(-1)
    signal = fwd + rng.normal(0, 0.01, fwd.shape)
    ic = daily_ic(signal, stock_ret, horizon=30)
    assert len(ic) > 50 and ic.mean() > 0.3, f"IC mean {ic.mean():.3f} should be strongly positive"
    print("\n=== SANITY CHECK: daily IC ===")
    print(f"  aligned signal -> mean IC {ic.mean():.3f}, IC_IR {ic.mean()/ic.std():.2f}. Validated.")


def test_ls_neutrality_metrics_and_plots(tmp_path):
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2020-01-01", periods=400)
    tickers = [f"T{i}" for i in range(25)]
    stock_ret = pd.DataFrame(rng.normal(0, 0.015, (len(idx), len(tickers))), index=idx, columns=tickers)
    signal = pd.DataFrame(rng.normal(0, 1, (len(idx), len(tickers))), index=idx, columns=tickers)
    ls_ret = pd.Series(rng.normal(0.0003, 0.004, len(idx)), index=idx)      # ~market-neutral
    spy = pd.Series(rng.normal(0.0004, 0.011, len(idx)), index=idx)
    energy = pd.Series(rng.normal(0.0002, 0.02, len(idx)), index=idx)
    m = analyze_ls(ls_ret, signal, stock_ret, spy, energy, tmp_path, horizon=30)
    assert (tmp_path / "ls_neutrality.png").exists() and (tmp_path / "ls_ic_sharpe.png").exists()
    assert abs(m["full_beta_sp"]) < 0.3, "synthetic L/S should be ~market-neutral (beta≈0)"
    print("\n=== SANITY CHECK: L/S neutrality analysis ===")
    print(f"  full beta_SP {m['full_beta_sp']:+.2f}, corr_energy {m['full_corr_energy']:+.2f}; plots saved. Validated.")


def test_correlation_analyses(tmp_path):
    rng = np.random.default_rng(2)
    idx = pd.bdate_range("2015-01-01", periods=600)
    assets = pd.DataFrame({"equity": rng.normal(0, 0.011, len(idx)), "gold": rng.normal(0, 0.01, len(idx)),
                           "bond": rng.normal(0, 0.004, len(idx))}, index=idx)
    lb = analyze_long_book(assets, tmp_path / "lb")
    pf = analyze_portfolio(assets.rename(columns={"equity": "ls_equity", "gold": "long_book",
                                                  "bond": "trend_cta"}), tmp_path / "pf")
    assert (tmp_path / "lb" / "long_book_correlations.png").exists()
    assert (tmp_path / "pf" / "sleeve_correlation_evolution.png").exists()
    assert lb["full_corr"].shape == (3, 3) and np.isfinite(pf["avg_pairwise_corr"])
    print("\n=== SANITY CHECK: long-book + portfolio correlation analysis ===")
    print(f"  long-book avg pairwise corr {lb['avg_pairwise_corr']:+.2f}; "
          f"portfolio avg pairwise corr {pf['avg_pairwise_corr']:+.2f}; plots saved. Validated.")


if __name__ == "__main__":
    test_daily_ic_detects_a_predictive_signal()
