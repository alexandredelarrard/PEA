"""Regression: a missing SHARED factor (oil / gold / USD-EUR) must not wipe out
the target for the whole cross-section.

compute_epsilon subtracts shared-factor forward returns from every stock. A gap
in a shared factor used to NaN the residual for all names on the affected dates
(and forward_compound's min_periods=h spreads one gap over ~h dates), dropping
the entire date from the cube. The fix fills only the FACTOR forward with 0 there
(skip that factor's neutralization) while keeping the stock's own forward return
required, so the genuine tail stays undefined.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd

from src.data_aggregate.utils.target.targets import compute_epsilon, forward_compound


def _number(value: object) -> float:
    assert not isinstance(value, pd.Series | pd.DataFrame)
    return float(cast(float, value))


def _setup(horizon=20, gap_at=100, t=150):
    dates = pd.bdate_range("2021-01-04", periods=t)
    tickers = ["A", "B", "C"]
    rng = np.random.default_rng(0)

    close = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0, 0.01, (t, 3)), axis=0), index=dates, columns=tickers)

    # shared factor panel: market + oil; oil has an INTERIOR one-day gap
    factor = pd.DataFrame(
        {
            "market": rng.normal(0, 0.01, t),
            "oil": rng.normal(0, 0.02, t),
        },
        index=dates,
    )
    oil_column = factor.columns.get_loc("oil")
    assert isinstance(oil_column, int)
    factor.iloc[gap_at, oil_column] = np.nan  # the data gap
    macro_cols = []  # oil = return factor

    betas = {}
    for tk in tickers:
        betas[tk] = pd.DataFrame(
            {
                "beta_market": 1.0,
                "beta_oil": 0.3,
            },
            index=dates,
        )
    return close, close.pct_change(), betas, factor, macro_cols, horizon, dates, gap_at


def test_shared_factor_gap_does_not_drop_the_cross_section():
    (close, ret, betas, factor, macro_cols, horizon, dates, gap_at) = _setup()

    eps = compute_epsilon(ret, betas, factor, macro_cols, horizon)
    assert isinstance(eps, pd.DataFrame)

    # dates whose forward window [t+1, t+h] contains the oil gap -> oil forward NaN
    oil_fwd = forward_compound(factor["oil"], horizon)
    assert isinstance(oil_fwd, pd.Series)
    affected = dates[(dates >= dates[gap_at - horizon]) & (dates < dates[gap_at])]
    t0 = pd.Timestamp(affected[len(affected) // 2])
    assert not np.isfinite(oil_fwd.loc[t0]), "test setup: oil forward should be NaN at t0"

    # WITH the fix: target is still defined at t0 for every stock whose OWN
    # forward return is defined (oil neutralization simply skipped there)
    stock_fwd_defined = close["A"].shift(-horizon).loc[t0]
    assert np.isfinite(stock_fwd_defined), "test setup: stock forward defined at t0"
    eps_at_t0 = eps.loc[t0]
    assert isinstance(eps_at_t0, pd.Series)
    assert eps_at_t0.notna().all(), f"factor gap NaN'd the cross-section at {t0.date()} (eps={eps_at_t0.to_dict()})"

    # the genuine TAIL (no future price) is still correctly undefined
    tail = pd.Timestamp(dates[-1])
    eps_at_tail = eps.loc[tail]
    assert isinstance(eps_at_tail, pd.Series)
    assert eps_at_tail.isna().all(), "tail target should be NaN (no forward price)"

    # coverage sanity: the affected window is now populated, not blank
    cov_affected = eps.loc[affected].notna().mean().mean()
    print("\n=== SANITY CHECK: shared-factor gap no longer drops the target ===")
    print(
        f"  oil gap on {dates[gap_at].date()} makes oil-forward NaN for "
        f"{len(affected)} dates; target coverage on those dates = "
        f"{cov_affected * 100:.0f}% (was ~0% before the fix). Tail stays NaN. "
        f"A missing oil/gold/FX factor now skips its neutralization instead of "
        f"blanking the whole cross-section. Validated."
    )


def test_missing_beta_still_propagates_nan():
    """Filling the FACTOR (not the product) means a missing BETA still NaNs the
    residual -- early-history behaviour is preserved, only factor DATA gaps heal."""
    (close, ret, betas, factor, macro_cols, horizon, dates, gap_at) = _setup()
    # blank ticker A's market beta on an interior date with a defined forward
    t = dates[50]
    betas["A"].loc[t, "beta_market"] = np.nan

    eps = compute_epsilon(ret, betas, factor, macro_cols, horizon)
    assert isinstance(eps, pd.DataFrame)
    assert np.isnan(_number(eps.loc[t, "A"])), "missing beta must still yield NaN (unchanged)"
    assert np.isfinite(_number(eps.loc[t, "B"])), "other stocks unaffected"
    print("\n=== SANITY CHECK: missing beta still propagates NaN ===")
    print(
        f"  A's blanked market beta on {t.date()} -> A target NaN (preserved); "
        f"B unaffected. Only factor DATA gaps are healed, not missing betas. "
        f"Validated."
    )


if __name__ == "__main__":
    test_shared_factor_gap_does_not_drop_the_cross_section()
    test_missing_beta_still_propagates_nan()
