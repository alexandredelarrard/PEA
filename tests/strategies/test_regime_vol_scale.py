"""
Vol-regime de-risking overlay (src/post_processing/utils/strategies_opt.py::regime_vol_scale).

The backtest scales BOTH sleeves by clip(target_vol / trailing-SPY-vol, floor, cap) so the
book de-risks through high-vol periods. Properties under test:
  * calm markets -> exposure saturates at `cap` (lever up modestly);
  * a sustained vol spike -> exposure collapses toward `floor` (de-risk);
  * point-in-time / leak-free: the multiplier at t uses only returns up to t, so a spike
    does not shrink exposure on days BEFORE it starts;
  * warmup (no vol yet) -> neutral 1.0.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategies.utils.strategies_opt import regime_vol_scale

_ANN = 252.0


def test_regime_vol_scale_derisks_in_high_vol_and_is_pointintime():
    rng = np.random.default_rng(0)
    win = 20
    calm = 0.003 * rng.standard_normal(120)          # ~4.8% ann vol  -> ratio >> 1 -> cap
    storm = 0.05 * rng.standard_normal(80)            # ~79% ann vol   -> ratio << 1 -> floor
    idx = pd.bdate_range("2020-01-01", periods=len(calm) + len(storm))
    spy = pd.Series(np.concatenate([calm, storm]), index=idx)

    floor, cap, target = 0.3, 1.5, 0.15
    scale = regime_vol_scale(spy, window=win, target_vol=target, floor=floor, cap=cap)

    assert scale.between(floor, cap).all(), "scale escaped [floor, cap]"
    # warmup (first value, no rolling vol yet) -> neutral
    assert scale.iloc[0] == 1.0

    # settled calm (well past warmup, before the storm) -> at the cap (lever up)
    calm_settled = scale.iloc[win + 5:110]
    assert (calm_settled > 1.2).mean() > 0.9, f"calm regime not levered: {calm_settled.mean():.2f}"

    # settled storm (well after the vol regime fully enters the trailing window) -> at floor
    storm_settled = scale.iloc[len(calm) + win + 5:]
    assert np.isclose(storm_settled, floor).mean() > 0.9, f"storm not de-risked: {storm_settled.mean():.2f}"

    # leak-free: on the LAST calm day the storm has not entered the window yet -> still high
    assert scale.iloc[len(calm) - 1] > 1.2, "future volatility leaked into an earlier day"

    # the transition only de-risks AFTER the storm starts (monotone-ish drop across it)
    assert scale.iloc[len(calm) - 1] > scale.iloc[len(calm) + win + 5]

    print("\n=== SANITY CHECK: vol-regime de-risking overlay ===")
    print(f"  calm (~5% ann vol)  -> exposure x{calm_settled.mean():.2f} (saturates at cap {cap})")
    print(f"  storm (~79% ann vol)-> exposure x{storm_settled.mean():.2f} (collapses to floor {floor})")
    print(f"  last calm day still x{scale.iloc[len(calm)-1]:.2f} (no look-ahead: the spike had "
          "not entered the trailing window yet)")
    print("  CONCLUSION: exposure ~ 1/vol, clipped to [floor,cap], point-in-time -> the book "
          "de-risks through high-vol periods without leaking future vol. Validated.")


if __name__ == "__main__":
    test_regime_vol_scale_derisks_in_high_vol_and_is_pointintime()
