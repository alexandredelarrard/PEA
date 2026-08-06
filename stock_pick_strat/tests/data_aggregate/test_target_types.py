"""Tests for the multi-version target (rank + z-score) stored in the cube and
selected at model time.

  * _apply_label            -> exact rank / z-score transforms of the residual
  * build_targets_multi     -> computes epsilon once, emits BOTH versions
  * _labels_to_long         -> nested targets become target_rank / target_zscore
  * panel_from_cube         -> `target_type` picks the column (legacy fallback)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_aggregate.utils.target.targets import (
    _apply_label, build_targets_multi, cross_sectional_zscore,
)
from src.data_aggregate.utils.assemble.cube import _labels_to_long, panel_from_cube


def test_zscore_target_is_winsorized():
    dates = pd.bdate_range("2020-01-01", periods=1)
    cols = [f"T{i}" for i in range(21)]            # 20 zeros + 1 outlier -> raw z ~4.36
    eps = pd.DataFrame([[0.0] * 20 + [5.0]], index=dates, columns=cols)

    z = cross_sectional_zscore(eps, min_names=3, clip=3.0)
    z_noclip = cross_sectional_zscore(eps, min_names=3, clip=None)

    assert z.abs().max().max() <= 3.0 + 1e-9, "z-score target not winsorized to +-3"
    assert z_noclip.abs().max().max() > 4.0, "sanity: the outlier's raw z should exceed 3"

    print("\n=== SANITY CHECK: z-score target winsorized ===")
    print(f"  outlier raw z={z_noclip.abs().max().max():.2f} -> clipped to "
          f"{z.abs().max().max():.2f} (+-3). Fat tails no longer dominate. Validated.")


# --------------------------------------------------------------------------- #
# 1. _apply_label: rank vs zscore math                                         #
# --------------------------------------------------------------------------- #
def test_apply_label_rank_and_zscore_exact():
    dates = pd.bdate_range("2020-01-01", periods=2)
    eps = pd.DataFrame([[1.0, 2.0, 3.0, 4.0, 5.0],
                        [5.0, 4.0, 3.0, 2.0, 1.0]], index=dates, columns=list("ABCDE"))

    r = _apply_label(eps, "rank", min_names=3)
    z = _apply_label(eps, "zscore", min_names=3)

    # rank: percentile in [0,1]; pct-rank mean is (n+1)/(2n)=0.6 for n=5 (-> 0.5 as n grows)
    assert r.values.min() >= 0.0 and r.values.max() <= 1.0
    assert abs(r.mean(axis=1).mean() - 0.6) < 1e-9
    assert r.loc[dates[0], "E"] == 1.0 and r.loc[dates[0], "A"] == 0.2

    # zscore: mean 0, std 1 (sample) per day; keeps magnitude/ordering
    assert z.mean(axis=1).abs().max() < 1e-9
    assert abs(z.std(axis=1).mean() - 1.0) < 1e-9
    assert z.loc[dates[0], "C"] == 0.0                      # middle value -> 0

    print("\n=== SANITY CHECK: rank vs z-score target ===")
    print(f"  rank row0 = {list(r.loc[dates[0]].round(2))} (in [0,1], (n+1)/2n=0.6 for n=5)")
    print(f"  zscore row0 = {list(z.loc[dates[0]].round(3))} (mean 0, std 1). Both exact.")


# --------------------------------------------------------------------------- #
# 2. build_targets_multi emits BOTH versions from one epsilon                   #
# --------------------------------------------------------------------------- #
def _mini_inputs(T=40, N=6, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=T)
    tickers = [f"T{i}" for i in range(N)]
    stock_ret = pd.DataFrame(rng.normal(0, 0.02, (T, N)), index=dates, columns=tickers)
    close = (1 + stock_ret).cumprod() * 100.0
    factor_panel = pd.DataFrame({"market": rng.normal(0.0004, 0.01, T)}, index=dates)
    betas = {t: pd.DataFrame({"beta_market": 1.0}, index=dates) for t in tickers}
    return close, betas, factor_panel


def test_build_targets_multi_returns_rank_and_zscore():
    close, betas, factor_panel = _mini_inputs()
    out = build_targets_multi(close, betas, factor_panel, macro_cols=[],
                              horizons=(5,), labels=("rank", "zscore"), min_names=3)

    assert set(out.keys()) == {5}
    assert set(out[5].keys()) == {"rank", "zscore"}
    rank, z = out[5]["rank"], out[5]["zscore"]
    # both cover the same (date,ticker) grid; rank in [0,1], zscore centered per day
    valid = z.notna().sum(axis=1) >= 3
    rv = rank[valid].to_numpy()
    assert np.nanmin(rv) >= 0.0 and np.nanmax(rv) <= 1.0
    assert z[valid].mean(axis=1).abs().max() < 1e-9
    # rank and zscore rank-order names identically (same eps, monotone transforms)
    d0 = valid[valid].index[0]
    assert (rank.loc[d0].rank().dropna() == z.loc[d0].rank().dropna()).all()

    print("\n=== SANITY CHECK: build_targets_multi ===")
    print(f"  horizon 5 -> versions {sorted(out[5])}; epsilon computed once; rank in [0,1], "
          "zscore centered at 0; both rank-order names identically. Validated.")


# --------------------------------------------------------------------------- #
# 3. cube plumbing: nested labels -> columns, panel_from_cube selects type      #
# --------------------------------------------------------------------------- #
def test_labels_to_long_creates_per_version_columns():
    dates = pd.bdate_range("2020-01-01", periods=2)
    tkrs = ["AAA", "BBB", "CCC"]
    mk = lambda v: pd.DataFrame(v, index=dates, columns=tkrs)
    labels = {5: {"rank": mk([[0.2, 0.5, 0.8]] * 2),
                  "zscore": mk([[-1.0, 0.0, 1.0]] * 2)}}
    long = _labels_to_long(labels)
    assert {"target_rank", "target_zscore", "target_horizon", "date", "ticker"}.issubset(long.columns)
    print("\n=== SANITY CHECK: _labels_to_long nested ===")
    print(f"  columns -> {sorted(long.columns)} (one target_<version> each). Validated.")


def _mini_cube():
    dates = pd.bdate_range("2020-01-01", periods=2)
    rows = []
    for d in dates:
        for tk, r, z, f in [("AAA", 0.2, -1.0, 1.1), ("BBB", 0.5, 0.0, 2.2), ("CCC", 0.8, 1.0, 3.3)]:
            rows.append(dict(date=d, ticker=tk, target_horizon=5,
                             target_rank=r, target_zscore=z, f_feat=f))
    return pd.DataFrame(rows)


def test_panel_from_cube_selects_target_type():
    cube = _mini_cube()
    p_rank = panel_from_cube(cube, 5, "y", feature_cols=["f_feat"], target_type="rank")
    p_z = panel_from_cube(cube, 5, "y", feature_cols=["f_feat"], target_type="zscore")

    assert list(p_rank["y"]) == [0.2, 0.5, 0.8, 0.2, 0.5, 0.8]     # target_rank
    assert list(p_z["y"]) == [-1.0, 0.0, 1.0, -1.0, 0.0, 1.0]      # target_zscore
    # the OTHER target column must not leak in as a feature
    assert "target_zscore" not in p_rank.columns and "target_rank" not in p_z.columns
    assert "f_feat" in p_rank.columns

    print("\n=== SANITY CHECK: panel_from_cube target_type selection ===")
    print("  target_type='rank' -> y is target_rank; 'zscore' -> y is target_zscore; "
          "other target column dropped. Validated.")


def test_panel_from_cube_legacy_single_target_fallback():
    # old cube with a single 'target' column -> still works
    dates = pd.bdate_range("2020-01-01", periods=1)
    cube = pd.DataFrame([dict(date=dates[0], ticker="AAA", target_horizon=5,
                              target=0.7, f_feat=1.0)])
    p = panel_from_cube(cube, 5, "y", feature_cols=["f_feat"], target_type="rank")
    assert list(p["y"]) == [0.7]

    print("\n=== SANITY CHECK: legacy single-target cube fallback ===")
    print("  a pre-existing 'target'-only cube still loads (falls back). Validated.")
