"""Tests for compute_sector_returns (src/data_peers/utils/sector_peers.py).

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

from src.data_peers.utils.sector_peers import (
    compute_sector_returns, build_peer_dict, dedupe_share_classes,
)


def test_dual_class_excluded_from_peer_candidates():
    """A secondary share class (GOOG) must NEVER be anyone's peer (it's the same
    company as GOOGL, correlation ~1.0), and it inherits GOOGL's basket so it still
    has valid, non-self peers."""
    dates = pd.bdate_range("2020-01-01", periods=300)
    rng = np.random.default_rng(1)
    df = pd.DataFrame({t: rng.normal(0, 0.01, len(dates))
                       for t in ["AAA", "BBB", "CCC", "DDD", "GOOGL"]}, index=dates)
    df["GOOG"] = df["GOOGL"] + rng.normal(0, 1e-6, len(dates))     # ~identical twin

    peers = build_peer_dict(df, top_k=3, weighting="corr", min_obs=50,
                            redundant_map={"GOOG": "GOOGL"})

    # GOOG is never a peer CANDIDATE for anyone (incl. GOOGL)
    assert all("GOOG" not in basket for basket in peers.values()), "GOOG leaked as a peer"
    assert "GOOG" not in peers["GOOGL"] and "GOOGL" not in peers["GOOGL"]
    # GOOG inherits GOOGL's basket and never contains itself or its twin
    assert peers["GOOG"] == peers["GOOGL"]
    assert "GOOG" not in peers["GOOG"] and "GOOGL" not in peers["GOOG"]

    print("\n=== SANITY CHECK: dual-class peer dedup (build) ===")
    print(f"  GOOG (~identical to GOOGL) is NOT a peer of any stock; "
          f"GOOG inherits GOOGL's basket = {sorted(peers['GOOG'])}. No self/twin peers. Validated.")


def test_dedupe_share_classes_fixes_cached_dict():
    """The load-path post-processor strips secondaries from a cached dict, renormalizes
    the survivors, and gives each secondary its primary's basket (no re-embedding)."""
    cached = {
        "AAA": {"GOOG": 0.5, "BBB": 0.5},        # GOOG must be stripped + renormalized
        "GOOGL": {"AAA": 0.6, "GOOG": 0.4},      # its own twin must be stripped
        "GOOG": {"AAA": 1.0},                     # will inherit GOOGL's cleaned basket
    }
    out = dedupe_share_classes(cached, {"GOOG": "GOOGL"})
    assert out["AAA"] == {"BBB": 1.0}             # 0.5 -> renorm 1.0
    assert out["GOOGL"] == {"AAA": 1.0}           # 0.6 -> renorm 1.0
    assert out["GOOG"] == out["GOOGL"]            # secondary inherits primary
    assert dedupe_share_classes(out, {"GOOG": "GOOGL"}) == out   # idempotent

    print("\n=== SANITY CHECK: dual-class dedup (cached load) ===")
    print(f"  GOOG stripped from AAA -> {out['AAA']}; GOOGL twin stripped -> {out['GOOGL']}; "
          f"GOOG inherits GOOGL. Idempotent. Fixes existing cache without re-embedding. Validated.")


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
