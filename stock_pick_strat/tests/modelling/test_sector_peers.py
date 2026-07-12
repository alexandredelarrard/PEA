"""Tests for compute_sector_returns (src/modelling/utils_model/sector_peers.py).

The sector return is a weighted average of a stock's peer returns. It must be
NaN-TOLERANT: a single missing peer on a date (e.g. a peer that only listed
recently) must NOT wipe out the whole date. The original implementation used a
raw matrix product `returns[cols] @ w`, which propagates a single NaN to the
entire date -- this silently truncated stocks' beta/target history (see
test_targets.py) to the short window where every peer happened to be listed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.modelling.utils_model.sector_peers import compute_sector_returns


def _returns_with_late_peer():
    dates = pd.bdate_range("2020-01-01", periods=250)
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        rng.normal(0, 0.01, size=(len(dates), 3)),
        index=dates, columns=["AAA", "BBB", "LATE"],
    )
    # LATE only starts trading 80% of the way through.
    df.loc[df.index[: int(0.8 * len(df))], "LATE"] = np.nan
    return df


def test_sector_return_is_nan_tolerant():
    returns = _returns_with_late_peer()
    peers = {"AAA": {"BBB": 0.5, "LATE": 0.5}}

    sector = compute_sector_returns(returns, peers)["AAA"]

    # Before LATE lists, the sector return must fall back to the available
    # peer (BBB) rather than being NaN for the whole early period.
    early = returns.index[: int(0.8 * len(returns))]
    assert sector.loc[early].notna().mean() > 0.99, (
        "sector return collapsed to NaN where a single peer was missing"
    )

    # When only BBB is available, sector return == BBB return (weights renormalize).
    np.testing.assert_allclose(
        sector.loc[early].to_numpy(),
        returns.loc[early, "BBB"].to_numpy(),
        rtol=1e-9, atol=1e-12,
    )

    print("\n=== SANITY CHECK: sector return NaN-tolerance ===")
    print(f"  peer LATE missing for first 80% of dates.")
    print(f"  sector non-null over that window = {sector.loc[early].notna().mean():.2%}")
    print("  -> falls back to available peers instead of nuking the date. Correct.")


def test_sector_return_equals_weighted_mean_when_all_present():
    returns = _returns_with_late_peer()
    peers = {"AAA": {"BBB": 0.3, "LATE": 0.7}}
    sector = compute_sector_returns(returns, peers)["AAA"]

    both = returns.dropna(subset=["BBB", "LATE"]).index
    expected = 0.3 * returns.loc[both, "BBB"] + 0.7 * returns.loc[both, "LATE"]
    np.testing.assert_allclose(sector.loc[both].to_numpy(), expected.to_numpy(),
                               rtol=1e-9, atol=1e-12)

    print("\n=== SANITY CHECK: sector return == weighted peer mean when all present ===")
    print("  -> matches the plain weighted average exactly. Correct.")
