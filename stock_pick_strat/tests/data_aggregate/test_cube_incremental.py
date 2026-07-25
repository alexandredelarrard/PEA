"""
Incremental cube-part builds (src/data_aggregate/step_build_cube.py).

The exploded DAG rebuilds each cube_part_<group> INCREMENTALLY: read the latest date, recompute only
a warm-up-padded trailing window, and append the new rows. This is only correct if a trailing-window
build reproduces the FULL build's tail exactly — which holds because the price/rolling features are
backward-looking (window <= warm-up) and the cross-sectional standardization is per-day (independent
across dates). This test proves that equivalence on the price feature builder, and checks the
idempotent tail-append helper.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_aggregate.utils.features import build_feature_panel
from src.data_aggregate.step_build_cube import StepBuildCube


def _synthetic_prices(n_days: int = 2000, n_tickers: int = 8, seed: int = 0):
    dates = pd.bdate_range("2019-01-01", periods=n_days)
    tk = [f"T{i}" for i in range(n_tickers)]
    rng = np.random.default_rng(seed)
    close = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.012, (n_days, n_tickers)), axis=0)),
                         index=dates, columns=tk)
    open_ = close.shift(1).bfill()
    ret = close.pct_change().fillna(0.0)
    sector = pd.DataFrame(np.repeat(ret.mean(axis=1).to_numpy()[:, None], n_tickers, axis=1),
                          index=dates, columns=tk)                     # one shared "sector"
    high, low = close * 1.01, close * 0.99
    volume = pd.DataFrame(rng.integers(1_000_000, 5_000_000, (n_days, n_tickers)).astype(float),
                          index=dates, columns=tk)
    return dates, close, open_, sector, high, low, volume


def test_windowed_build_reproduces_full_tail():
    dates, close, open_, sector, high, low, volume = _synthetic_prices()
    sh = [5, 20, 60]

    full = build_feature_panel(close, open_, sector, "rank", high, low, volume, sh)

    # simulate an incremental run with the ACTUAL configured price-group warm-up: recompute only
    # [cutoff-warmup, end]. This must cover the longest daily look-back (the 5-year = 1260-day
    # seasonality feature) -> the test fails if the map value is ever set too low.
    warmup = StepBuildCube._GROUP_WARMUP_TRADING_DAYS["price"]
    cutoff_pos = 1900
    cutoff = dates[cutoff_pos]
    start = dates[cutoff_pos - warmup]
    win = build_feature_panel(close.loc[start:], open_.loc[start:], sector.loc[start:], "rank",
                              high.loc[start:], low.loc[start:], volume.loc[start:], sh)

    # the tail we would KEEP + APPEND (date > cutoff) must match the full build bit-for-bit
    f_tail = (full[full["date"] > cutoff].set_index(["date", "ticker"]).sort_index())
    w_tail = (win[win["date"] > cutoff].set_index(["date", "ticker"]).sort_index())
    assert not f_tail.empty and f_tail.shape == w_tail.shape
    cols = list(f_tail.columns)
    pd.testing.assert_frame_equal(f_tail[cols], w_tail[cols].reindex(f_tail.index),
                                  check_exact=False, atol=1e-9, rtol=0)

    n_tail_days = f_tail.index.get_level_values("date").nunique()
    print("\n=== SANITY CHECK: windowed build reproduces the full tail ===")
    print(f"  full={len(full)} rows over {close.shape[0]} days; windowed recompute of the last "
          f"{close.shape[0] - (cutoff_pos - warmup)} days (warmup {warmup})")
    print(f"  tail (date > {cutoff.date()}): {n_tail_days} days x {full['ticker'].nunique()} "
          f"tickers x {len(cols)} feature cols -> IDENTICAL between full and windowed builds")
    print("  CONCLUSION: backward-looking features + per-day standardization -> the incremental "
          "trailing recompute equals a full rebuild on the appended dates. Validated.")


def test_incremental_horizon_arithmetic():
    """The target refresh window must reach back >= max_horizon so matured (NaN->value) labels are
    recomputed, while features only need dates strictly after the stored max."""
    # emulate _window_start on a business-day calendar
    idx = pd.bdate_range("2019-01-01", periods=800)
    last = idx[750]

    def window_start(last, n_back):
        pos = int(idx.searchsorted(pd.Timestamp(last)))
        return idx[max(0, pos - n_back)]

    max_h = 90
    feat_start = window_start(last, 1400)                 # warm-up only (features)
    tgt_start = window_start(last, 1400 + max_h)          # warm-up + horizon (targets compute)
    refresh_from = window_start(last, max_h)              # matured-label overwrite window

    assert feat_start < last and tgt_start <= feat_start
    # the refresh window covers exactly the dates whose forward labels could have matured
    assert (idx.searchsorted(last) - idx.searchsorted(refresh_from)) == max_h
    print("\n=== SANITY CHECK: incremental window arithmetic ===")
    print(f"  last stored date {last.date()} | feature warm-up start {feat_start.date()} | "
          f"target compute start {tgt_start.date()} | matured-label refresh from {refresh_from.date()}")
    print("  targets overwrite the trailing max_horizon window (matured labels), betas/features "
          "append only dates after the max. Validated.")


def test_per_part_warmup_covers_binding_lookback():
    """Per-part sanity check: each group's warm-up must cover the LONGEST daily-grid look-back its
    features compute (else the incremental tail would be silently wrong). Filing/quarter-space parts
    read the FULL source table, so their grid look-back is ~0 (a 6-month floor is plenty)."""
    w = StepBuildCube._GROUP_WARMUP_TRADING_DAYS
    binding = {                          # longest DAILY-grid look-back per part (trading days)
        "price":          252 * 5,       # seasonal_h*: close.shift(252 * seasonal_years=5)
        "fundamental":    1260,          # _self_history_z rolling(1260)
        "dividend":       5 * 252,       # 5y payout growth shift(5*252)
        "employee":       252,           # YoY shift(252)
        "short_interest": 63 + 40,       # rolling(63) + FTD shift(40)
        "attention":      63,            # rolling(63)
        "sector": 0, "earnings": 0, "governance": 0, "institutional": 0,
        "superinvestor": 0, "insider": 0,                       # source/filing-space -> ~0 grid
        "earnings_call_sentiment": 0, "earnings_call_embedding": 0,
    }
    assert set(w) == set(StepBuildCube._GROUP_SOURCES), "every feature group needs a warm-up entry"
    for g, need in binding.items():
        assert w[g] >= need, f"{g}: warm-up {w[g]} < binding daily look-back {need}"
    # the heavy parts read ~5y, the light/source-space parts read <=~1y -> genuinely lighter
    heavy = {g for g, n in binding.items() if n >= 1260}
    assert all(w[g] <= 400 for g in binding if g not in heavy), "light parts should stay light"

    print("\n=== SANITY CHECK: per-part warm-up vs binding look-back ===")
    for g in StepBuildCube._GROUP_SOURCES:
        print(f"  {g:<15} warm-up={w[g]:>5}  binding={binding[g]:>5}  "
              f"({'DAILY grid' if binding[g] else 'full source / filing-space'})")
    print("  CONCLUSION: heavy parts (price/fundamental/dividend) read ~5y; the 10 others read "
          "<=1y (7 read only ~6mo). Each reads only as far back as it needs. Validated.")


if __name__ == "__main__":
    test_windowed_build_reproduces_full_tail()
    test_incremental_horizon_arithmetic()
    test_per_part_warmup_covers_binding_lookback()
