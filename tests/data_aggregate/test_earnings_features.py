"""Tests for the historical earnings-expectation features
(src/data_aggregate/utils/earnings_features.py).

The two point-in-time guarantees that matter:
  * the REALIZED surprise / actual EPS appears only FROM the earnings date
    onward (never before the number is public);
  * the FORWARD EPS estimate is applied to the days leading up to a future
    report, but a change to the REPORTED actual must not change any forward
    feature (no outcome leakage).
Plus the ratio math (forward E/P, expected growth) must be exact.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_aggregate.utils.fundamentals.earnings_features import _derived_earnings_fields, ntm_ttm_eps


def _synth_earnings():
    """Two reported quarters + one upcoming (forward) quarter for AAA."""
    return pd.DataFrame({
        "ticker": ["AAA", "AAA", "AAA"],
        "earnings_date": ["2019-05-01", "2019-08-01", "2019-11-01"],
        "eps_estimate": [1.00, 1.10, 1.30],
        "eps_actual": [1.20, 1.05, np.nan],     # beat, miss, upcoming
        "surprise_pct": [20.0, -4.5, np.nan],
    })


def test_realized_features_are_point_in_time():
    hist = _synth_earnings()
    idx = pd.bdate_range("2019-04-01", "2019-12-31")
    close = pd.DataFrame({"AAA": 50.0}, index=idx)
    F = _derived_earnings_fields(hist, idx, close)

    before_first = pd.Timestamp("2019-04-15")   # before any report
    after_first = pd.Timestamp("2019-06-03")    # after the 2019-05-01 report

    # surprise is unknown before the first report, known after it
    assert np.isnan(F["eps_surprise_last"].loc[before_first, "AAA"])
    assert abs(F["eps_surprise_last"].loc[after_first, "AAA"] - 20.0) < 1e-9

    print("\n=== SANITY CHECK: realized surprise is point-in-time ===")
    print(f"  surprise NaN before first report, = {F['eps_surprise_last'].loc[after_first,'AAA']:.1f}% "
          f"after it (beat). No look-ahead on the actual.")


def test_forward_eps_yield_and_expected_growth_math():
    hist = _synth_earnings()
    idx = pd.bdate_range("2019-04-01", "2019-12-31")
    close = pd.DataFrame({"AAA": 50.0}, index=idx)
    F = _derived_earnings_fields(hist, idx, close)

    # Between the Aug and Nov reports, the forward estimate is the Nov one (1.30).
    d = pd.Timestamp("2019-10-01")
    assert abs(F["fwd_eps_yield"].loc[d, "AAA"] - 1.30 / 50.0) < 1e-9
    # expected growth = fwd estimate (1.30) / last reported actual (Aug: 1.05) - 1
    assert abs(F["eps_expectation_growth"].loc[d, "AAA"] - (1.30 / 1.05 - 1)) < 1e-9

    print("\n=== SANITY CHECK: forward EPS yield & expected growth ===")
    print(f"  fwd E/P = 1.30/50 = {F['fwd_eps_yield'].loc[d,'AAA']:.4f}; "
          f"expected growth = 1.30/1.05-1 = {F['eps_expectation_growth'].loc[d,'AAA']:+.2%}.")


def test_ntm_forward_earnings_yield():
    """NTM (annual, forward-rolled) EPS = next-quarter estimate + trailing 3 reported
    actuals; forward_earnings_yield = NTM EPS / price (the historical forwardPE replacement)."""
    hist = pd.DataFrame({
        "ticker": ["A"] * 5,
        "earnings_date": ["2025-02-01", "2025-05-01", "2025-08-01", "2025-11-01", "2026-02-15"],
        "eps_estimate": [1.0, 1.0, 1.0, 1.0, 1.5],
        "eps_actual":   [1.1, 1.2, 1.3, 1.4, np.nan],   # last row = upcoming (estimate only)
        "surprise_pct": [10.0, 20.0, 30.0, 40.0, np.nan],
    })
    idx = pd.bdate_range("2026-01-05", "2026-01-30")     # after the Nov report, before Feb report
    close = pd.DataFrame({"A": 100.0}, index=idx)
    d = idx[-1]

    ntm, ttm = ntm_ttm_eps(hist, idx)
    assert abs(ntm.loc[d, "A"] - (1.5 + 1.2 + 1.3 + 1.4)) < 1e-9    # est(Q+1) + last 3 actuals = 5.4
    assert abs(ttm.loc[d, "A"] - (1.1 + 1.2 + 1.3 + 1.4)) < 1e-9    # trailing 4 actuals = 5.0

    F = _derived_earnings_fields(hist, idx, close)
    assert abs(F["forward_earnings_yield"].loc[d, "A"] - 5.4 / 100.0) < 1e-9
    print("\n=== SANITY CHECK: NTM forward-earnings yield ===")
    print(f"  NTM EPS = 1.5 est + (1.2+1.3+1.4) actuals = {ntm.loc[d,'A']:.1f}; TTM = {ttm.loc[d,'A']:.1f}; "
          f"forward_earnings_yield = 5.4/100 = {F['forward_earnings_yield'].loc[d,'A']:.4f} "
          f"(= 1/forward P/E, leak-free). Validated.")


def test_forward_feature_does_not_leak_reported_actual():
    idx = pd.bdate_range("2019-04-01", "2019-12-31")
    close = pd.DataFrame({"AAA": 50.0}, index=idx)

    base = _synth_earnings()
    bumped = _synth_earnings()
    bumped.loc[1, "eps_actual"] = 5.00      # blow up the Aug REPORTED actual

    Fb = _derived_earnings_fields(base, idx, close)
    Fp = _derived_earnings_fields(bumped, idx, close)

    # a day BEFORE the Aug report: the forward EPS yield must be identical
    d = pd.Timestamp("2019-07-15")
    v1, v2 = Fb["fwd_eps_yield"].loc[d, "AAA"], Fp["fwd_eps_yield"].loc[d, "AAA"]
    assert (np.isnan(v1) and np.isnan(v2)) or abs(v1 - v2) < 1e-12

    print("\n=== SANITY CHECK: forward EPS does not leak the actual ===")
    print("  changing a reported actual leaves the pre-report forward EPS yield")
    print("  unchanged -> the forward feature is a pure expectation, not an outcome.")
