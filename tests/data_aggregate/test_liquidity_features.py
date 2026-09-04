"""Step 1 — Liquidity features (dollar volume, Amihud illiquidity, relative
volume). Verify they are produced only when volume is supplied, are point-in-time
(no look-ahead), rank the universe sensibly, and are cross-sectionally
standardized in the built panel.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_aggregate.utils.momentum.features import compute_raw_features, build_feature_panel

_LIQ = ["dollar_volume_63", "amihud_63", "rel_volume_5_63"]


def _synth(T=260, N=25, seed=0):
    dates = pd.bdate_range("2022-01-03", periods=T)
    tickers = [f"S{i:02d}" for i in range(N)]
    rng = np.random.default_rng(seed)
    close = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0, 0.015, (T, N)), axis=0),
                         index=dates, columns=tickers)
    open_ = close.shift(1).fillna(close.iloc[0])
    # volume: each stock a different base level -> a clear liquidity ranking
    base = np.linspace(1e5, 1e7, N)
    volume = pd.DataFrame(base[None, :] * rng.lognormal(0, 0.3, (T, N)),
                          index=dates, columns=tickers)
    sector = pd.DataFrame(rng.normal(0, 0.01, (T, N)), index=dates, columns=tickers)
    return dates, tickers, close, open_, volume, sector


def test_liquidity_skipped_without_volume_present_with():
    dates, tickers, close, open_, volume, sector = _synth()
    no_vol = compute_raw_features(close, open_, sector)
    assert not any(k in no_vol for k in _LIQ), "liquidity must be skipped without volume"

    with_vol = compute_raw_features(close, open_, sector, volume=volume)
    for k in _LIQ:
        assert k in with_vol, f"{k} missing when volume supplied"
        # finite after warmup (need >=20 obs for the 63d windows)
        assert np.isfinite(with_vol[k].iloc[80]).sum() >= 20, f"{k} mostly NaN post-warmup"

    print("\n=== SANITY CHECK: liquidity present iff volume supplied ===")
    print(f"  no volume -> none of {_LIQ}; with volume -> all present & finite post-warmup. Validated.")


def test_no_lookahead():
    """A feature at date t must not change if FUTURE rows are altered."""
    dates, tickers, close, open_, volume, sector = _synth()
    t_idx = 150
    base = compute_raw_features(close, open_, sector, volume=volume)

    close2, volume2 = close.copy(), volume.copy()
    close2.iloc[t_idx + 1:] *= 1.5          # perturb everything AFTER t
    volume2.iloc[t_idx + 1:] *= 3.0
    pert = compute_raw_features(close2, open_, sector, volume=volume2)

    for k in _LIQ:
        a = base[k].iloc[t_idx].to_numpy()
        b = pert[k].iloc[t_idx].to_numpy()
        assert np.allclose(a, b, equal_nan=True), f"{k} leaks future data at t"
    print("\n=== SANITY CHECK: liquidity features are point-in-time ===")
    print("  perturbing all rows AFTER t left every liquidity value at t unchanged. Validated.")


def test_dollar_volume_ranks_liquidity():
    dates, tickers, close, open_, volume, sector = _synth()
    raw = compute_raw_features(close, open_, sector, volume=volume)
    dv = raw["dollar_volume_63"].iloc[-1]      # last date
    # S24 has ~100x the base volume of S00 -> must rank far higher on $ volume
    assert dv[tickers[-1]] > dv[tickers[0]], (dv[tickers[0]], dv[tickers[-1]])

    panel = build_feature_panel(close, open_, sector, method="rank", volume=volume)
    for k in _LIQ:
        assert k in panel.columns, f"{k} not in built panel"
        col = panel[k].dropna()
        assert col.between(0, 1).all(), f"{k} not rank-standardized to [0,1]"
    print("\n=== SANITY CHECK: dollar_volume ranks liquidity; panel standardized ===")
    print(f"  high-volume name outranks low-volume name on dollar_volume_63; "
          f"all liquidity cols rank-standardized to [0,1] in the panel. Validated.")


def test_amihud_is_monotone_in_illiquidity_and_rank_invariant_under_the_log():
    """`amihud_63` is `np.log` of the averaged statistic. Two properties, and the change is
    only free because BOTH hold:

      * MONOTONE -- the log preserves the ordering, so an illiquid name still reads higher than
        a liquid one and the economic meaning is untouched.
      * RANK-INVARIANT -- the STORED panel under the live `standardize_method: rank` is
        identical to what the un-logged statistic produced, so the transform costs nothing
        today. It exists so a flip to `zscore` is not crowded: un-logged, the statistic spans
        eight orders of magnitude and 76.8% of the cross-section sits inside +/-0.25sd.
    """
    dates, tickers, close, open_, volume, sector = _synth()
    raw = compute_raw_features(close, open_, sector, volume=volume)
    logged = raw["amihud_63"]

    # the un-logged statistic, rebuilt here as the reference
    ret = close.pct_change(fill_method=None)
    dollar_vol = close * volume
    plain = ((ret.abs() / dollar_vol.where(dollar_vol > 0))
             .rolling(63, min_periods=20).mean())

    row_l, row_p = logged.iloc[-1].dropna(), plain.iloc[-1].dropna()
    shared = row_l.index.intersection(row_p.index)
    # monotone: the rank orderings agree exactly
    assert (row_l[shared].rank().to_numpy() == row_p[shared].rank().to_numpy()).all()
    # and the low-volume name is still the illiquid one
    assert row_l[tickers[0]] > row_l[tickers[-1]]

    # rank-invariant across the WHOLE panel, not just one row
    from src.data_aggregate.utils.common.xs import xs_rank_pct
    a, b = xs_rank_pct(logged), xs_rank_pct(plain.reindex_like(logged))
    both = a.notna() & b.notna()
    assert int(both.to_numpy().sum()) > 0
    np.testing.assert_allclose(a.to_numpy()[both.to_numpy()], b.to_numpy()[both.to_numpy()],
                               rtol=0, atol=0)

    # the zscore crowding this transform exists to fix
    def crowding(frame):
        z = frame.sub(frame.mean(axis=1), axis=0).div(frame.std(axis=1), axis=0)
        return float((z.abs() < 0.25).to_numpy().sum() / z.notna().to_numpy().sum())

    tight_plain, tight_log = crowding(plain), crowding(logged)
    assert tight_log < tight_plain

    print("\n=== SANITY CHECK: amihud_63 log transform ===")
    print(f"  rank ordering identical on {len(shared)} names; illiquid {tickers[0]} "
          f"{row_l[tickers[0]]:.3f} > liquid {tickers[-1]} {row_l[tickers[-1]]:.3f}")
    print(f"  stored RANK panel bit-identical over {int(both.to_numpy().sum())} cells "
          "(atol=0) -> the change is free under standardize_method: rank")
    print(f"  +/-0.25sd crowding under zscore: {tight_plain:.1%} un-logged -> "
          f"{tight_log:.1%} logged. Validated.")


def test_an_all_flat_window_yields_nan_not_minus_inf():
    """`|ret|` is exactly 0 on a flat day, so an all-flat 63-day window averages to 0 and
    `log(0)` is -inf. The `.where(> 0)` guard has to run BEFORE the log, or every build emits a
    NumPy divide warning and the cell reaches `sanitize` as an infinity."""
    dates = pd.bdate_range("2022-01-03", periods=120)
    tickers = ["FLAT", "MOVES"]
    close = pd.DataFrame({"FLAT": 100.0,
                          "MOVES": 100 * np.cumprod(1 + np.full(len(dates), 0.001))},
                         index=dates)
    volume = pd.DataFrame(1e6, index=dates, columns=tickers)
    sector = pd.DataFrame(0.0, index=dates, columns=tickers)

    with np.errstate(divide="raise", invalid="raise"):
        raw = compute_raw_features(close, close.shift(1).bfill(), sector, volume=volume)

    flat = raw["amihud_63"]["FLAT"]
    assert flat.isna().all()
    assert not np.isinf(flat.to_numpy()).any()
    assert raw["amihud_63"]["MOVES"].iloc[-1] < 0        # a tiny illiquidity -> log is negative

    print("\n=== SANITY CHECK: log(0) guard on a flat window ===")
    print(f"  FLAT (|ret| == 0 throughout): {int(flat.notna().sum())} non-null, 0 infinities, "
          "and no divide/invalid FloatingPointError raised under errstate. Validated.")


if __name__ == "__main__":
    test_liquidity_skipped_without_volume_present_with()
    test_no_lookahead()
    test_dollar_volume_ranks_liquidity()
    test_amihud_is_monotone_in_illiquidity_and_rank_invariant_under_the_log()
    test_an_all_flat_window_yields_nan_not_minus_inf()
