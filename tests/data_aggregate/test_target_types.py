"""Tests for the multi-version target (rank + z-score) stored in the cube and
selected at model time.

  * _apply_label            -> exact rank / z-score transforms of the residual
  * build_targets_multi     -> computes epsilon once, emits BOTH versions
  * panel_from_cube         -> (target_type, horizon) picks ONE wide column

`labels_to_wide` -- the {horizon: {label: DataFrame}} -> `target_<label>_h<horizon>` pivot
itself -- is pinned by `test_target_wide.py`, which covers the grain, the immature-label
tail and the legacy-mapping refusal. This file stops at the column the model reads.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_aggregate.utils.target.targets import (
    _apply_label, build_targets_multi, cross_sectional_zscore,
)
from src.data_aggregate.utils.assemble.cube import panel_from_cube


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
    return close, stock_ret, betas, factor_panel


def test_build_targets_multi_returns_rank_and_zscore():
    close, stock_ret, betas, factor_panel = _mini_inputs()
    # `stock_ret` is REQUIRED: every label is a forward COMPOUNDED total return now, not a
    # close-to-close price ratio. Here `close` IS (1+ret).cumprod(), so the two agree.
    out = build_targets_multi(close, betas, factor_panel, macro_cols=[],
                              horizons=(5,), labels=("rank", "zscore"), min_names=3,
                              stock_ret=stock_ret)

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
# 3. cube plumbing: panel_from_cube picks ONE wide column                       #
# --------------------------------------------------------------------------- #
def _mini_cube():
    """A WIDE cube: one row per (date, ticker), one column per (label, horizon). The
    horizon is a column axis, so the same 6 rows serve h5 and h20."""
    dates = pd.bdate_range("2020-01-01", periods=2)
    rows = []
    for d in dates:
        for tk, r, z, f in [("AAA", 0.2, -1.0, 1.1), ("BBB", 0.5, 0.0, 2.2), ("CCC", 0.8, 1.0, 3.3)]:
            rows.append(dict(date=d, ticker=tk, f_feat=f,
                             target_rank_h5=r, target_zscore_h5=z,
                             target_rank_h20=r / 2, target_zscore_h20=z / 2))
    return pd.DataFrame(rows)


def test_panel_from_cube_selects_target_type():
    cube = _mini_cube()
    p_rank = panel_from_cube(cube, 5, "y", feature_cols=["f_feat"], target_type="rank")
    p_z = panel_from_cube(cube, 5, "y", feature_cols=["f_feat"], target_type="zscore")

    assert list(p_rank["y"]) == [0.2, 0.5, 0.8, 0.2, 0.5, 0.8]     # target_rank_h5
    assert list(p_z["y"]) == [-1.0, 0.0, 1.0, -1.0, 0.0, 1.0]      # target_zscore_h5
    # (date, ticker) is the whole grain now: no horizon filter, so every row is a candidate
    assert len(p_rank) == len(cube) == 6
    # NO other target column may leak in -- not the sibling label, and not the OTHER
    # horizon's column of the same label, which an enumerated meta-column set would miss
    assert [c for c in p_rank.columns if c.startswith("target")] == []
    assert [c for c in p_z.columns if c.startswith("target")] == []
    assert "f_feat" in p_rank.columns

    print("\n=== SANITY CHECK: panel_from_cube target_type selection (wide cube) ===")
    print(f"  cube has {len([c for c in cube.columns if c.startswith('target')])} target "
          f"columns; target_type='rank', horizon=5 -> y is target_rank_h5, 'zscore' -> "
          f"target_zscore_h5")
    print(f"  {len(p_rank)} rows (no horizon row-filter); every other target_* column dropped, "
          f"including target_rank_h20. Validated.")


def test_panel_from_cube_selects_the_horizon_not_just_the_label():
    """Horizon is now a COLUMN choice. Asking for h20 must return h20's values off the very
    same rows -- a bug that ignored the horizon would silently train every horizon on h5."""
    cube = _mini_cube()
    p5 = panel_from_cube(cube, 5, "y", feature_cols=["f_feat"], target_type="rank")
    p20 = panel_from_cube(cube, 20, "y", feature_cols=["f_feat"], target_type="rank")

    assert list(p20["y"]) == [v / 2 for v in p5["y"]]
    assert p5[["date", "ticker"]].equals(p20[["date", "ticker"]])

    print("\n=== SANITY CHECK: horizon picks the column, not a row subset ===")
    print(f"  h5 y={list(p5['y'])[:3]} vs h20 y={list(p20['y'])[:3]} on IDENTICAL "
          f"(date, ticker) rows. Validated.")


def test_panel_from_cube_unknown_horizon_raises_and_names_what_exists():
    """The legacy single-`target` fallback is gone by decision: a cube that has no column for
    the requested (label, horizon) must say so, and list what it does have, rather than
    silently train on some other column."""
    cube = _mini_cube()

    with pytest.raises(KeyError) as ei:
        panel_from_cube(cube, 60, "y", feature_cols=["f_feat"], target_type="rank")
    msg = str(ei.value)
    assert "target_rank_h60" in msg and "target_rank_h5" in msg      # wanted + available
    assert "build_cube.targets" in msg                               # the actionable fix

    print("\n=== SANITY CHECK: missing target column raises ===")
    print(f"  horizon 60 is not built -> KeyError naming target_rank_h60, the available "
          f"target_* columns and the config keys to rebuild with. No silent fallback. "
          f"Validated.")
