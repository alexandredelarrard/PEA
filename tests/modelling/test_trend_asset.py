"""
Multi-asset time-series-momentum sleeve (src/modelling/utils_trend_asset/trend_signal.py).
Synthetic: an UP-trending asset, a DOWN-trending asset, and a mean-reverting CHOP asset. The trend
forecast must be long the uptrend, short the downtrend, ~flat on chop; the sleeve must PROFIT on the
trending pair (long the winner + short the loser); vol targeting must hit the requested vol.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.modelling.trend.utils import (
    apply_class_budget,
    carry_forecast,
    combine_signals,
    combined_forecast,
    sleeve_returns,
    value_forecast,
    vol_scaled_positions,
    realized_ann_vol,
    vol_target_scalar,
)


def _synthetic_close(n: int = 700, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-01", periods=n)
    up = 100 * np.cumprod(1 + rng.normal(0.0006, 0.010, n))       # steady uptrend
    dn = 100 * np.cumprod(1 + rng.normal(-0.0006, 0.010, n))      # steady downtrend
    chop = np.empty(n)                                            # mean-reverting LEVEL -> no trend
    x = 0.0                                                        # OU log-deviation around 100
    for t in range(n):
        x = 0.85 * x + rng.normal(0, 0.02)                        # stationary -> price reverts to 100
        chop[t] = 100.0 * np.exp(x)
    return pd.DataFrame({"UP": up, "DOWN": dn, "CHOP": chop}, index=idx)


def test_trend_sleeve_directions_and_pnl():
    close = _synthetic_close()
    f = combined_forecast(close, lookbacks=[63, 126, 252], vol_window=63, cap=2.0)
    fu, fd, fc = f["UP"].dropna().mean(), f["DOWN"].dropna().mean(), f["CHOP"].dropna().mean()

    assert fu > 0.15, f"uptrend should forecast LONG (got {fu:.2f})"
    assert fd < -0.15, f"downtrend should forecast SHORT (got {fd:.2f})"
    assert abs(fc) < 0.6 * fu, f"chop should be ~flat (got {fc:.2f} vs up {fu:.2f})"
    assert np.nanmax(np.abs(f.to_numpy())) <= 2.0 + 1e-9, "forecast must respect the cap"

    w = vol_scaled_positions(f, close, vol_window=63, per_asset_vol_target=0.15)
    sl = sleeve_returns(w, close, fee_bps=1.0, spread_bps=5.0, rebalance_freq=5)
    ann = realized_ann_vol(sl["ret"])
    sharpe = sl["ret"].mean() / sl["ret"].std() * np.sqrt(252)

    assert sl["ret"].mean() > 0, "trend sleeve should profit on a persistent trending pair"
    # vol-target calibration: scaling the sleeve to 0.10 should realize ~0.10
    k = vol_target_scalar(sl["ret"], 0.10)
    ann_scaled = realized_ann_vol(sl["ret"] * k)
    assert abs(ann_scaled - 0.10) < 0.005, f"vol target off: {ann_scaled:.3f}"

    print("\n=== SANITY CHECK: multi-asset trend sleeve (synthetic) ===")
    print(f"  forecast mean: UP={fu:+.2f} (long)  DOWN={fd:+.2f} (short)  CHOP={fc:+.2f} (~flat)")
    print(f"  sleeve: mean daily {sl['ret'].mean():.5f}, ann vol {ann:.3f}, Sharpe {sharpe:.2f}, "
          f"avg gross {sl['gross'].mean():.2f}")
    print(f"  vol-target scalar to 0.10 -> realized {ann_scaled:.3f} (calibration works)")
    print("  CONCLUSION: trend = time-series momentum (NOT mean reversion): long the uptrend, short "
          "the downtrend, profits on persistent moves; vol targeting hits the requested vol.")


def test_asset_class_budget_equalizes_classes():
    # 1 equity instrument vs 3 commodity instruments, all equal raw weight. Class budgeting must
    # give the equity CLASS the same total risk as the commodity CLASS (not 1/4 vs 3/4).
    w = pd.DataFrame({"SPY": [1.0, 1.0], "OIL": [1.0, 1.0], "GOLD": [1.0, 1.0], "DBC": [1.0, 1.0]})
    amap = {"SPY": "equity", "OIL": "commodity", "GOLD": "commodity", "DBC": "commodity"}
    wb = apply_class_budget(w, amap)                       # 2 classes -> 0.5 budget each
    eq = float(wb["SPY"].iloc[0])
    commod = float(wb[["OIL", "GOLD", "DBC"]].iloc[0].sum())
    assert abs(eq - 0.5) < 1e-9, f"equity class budget wrong: {eq}"
    assert abs(commod - 0.5) < 1e-9, f"commodity class budget wrong: {commod}"
    assert abs(float(wb["OIL"].iloc[0]) - 0.5 / 3) < 1e-9, "within-class split should be equal"

    print("\n=== SANITY CHECK: asset-class risk budgeting ===")
    print(f"  equity class total {eq:.3f} == commodity class total {commod:.3f} "
          f"(3 commodity instruments split {0.5/3:.3f} each)")
    print("  CONCLUSION: the trend sleeve times equity / commodities / FX on an EQUAL-risk footing; "
          "a crowded class (3 commodities) no longer dominates a thin one (1 equity index).")


def test_value_is_reversal_and_signal_blend():
    close = _synthetic_close()
    v = value_forecast(close, lookback=252, vol_window=63, cap=2.0)
    t = combined_forecast(close, lookbacks=[63, 126, 252], vol_window=63, cap=2.0)
    vu, vd = float(v["UP"].dropna().mean()), float(v["DOWN"].dropna().mean())
    # VALUE is long-horizon reversal: SHORT the expensive uptrend, LONG the cheap downtrend
    assert vu < 0, f"value should short the risen asset (got {vu:.2f})"
    assert vd > 0, f"value should long the fallen asset (got {vd:.2f})"

    # equal-weight blend of trend+value = their mean where both present (last row, within cap)
    comb = combine_signals({"trend": t, "value": v}, {"trend": 1.0, "value": 1.0})
    last = comb.index[-1]
    assert np.allclose(comb.loc[last].to_numpy(),
                       ((t.loc[last] + v.loc[last]) / 2).to_numpy(), atol=1e-9), "blend != mean"
    # NaN fallback: if value is missing for a column, the blend uses trend alone there
    v2 = v.copy(); v2["CHOP"] = np.nan
    comb2 = combine_signals({"trend": t, "value": v2}, {"trend": 1.0, "value": 1.0})
    ok = t["CHOP"].notna()
    assert np.allclose(comb2["CHOP"][ok].to_numpy(), t["CHOP"][ok].to_numpy(), atol=1e-9), \
        "missing-signal fallback failed"

    print("\n=== SANITY CHECK: value signal + multi-signal blend ===")
    print(f"  value forecast: UP={vu:+.2f} (short the winner)  DOWN={vd:+.2f} (long the loser) "
          f"-> OPPOSITE of trend")
    print("  combine_signals: equal-weight blend = mean(trend, value); a missing signal falls back "
          "to the others.")
    print("  CONCLUSION: per-class DIRECTION is now trend + value (+ carry) -- value is the "
          "negatively-correlated diversifier that steadies the trend whipsaw.")


if __name__ == "__main__":
    test_trend_sleeve_directions_and_pnl()
    test_asset_class_budget_equalizes_classes()
    test_value_is_reversal_and_signal_blend()
