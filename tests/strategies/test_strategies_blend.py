"""
Day-by-day vol-based strategy blender (src/post_processing/utils/strategies_blend.py).
Three uncorrelated synthetic sleeves at 5% / 20% / 10% annual vol. Inverse-vol weighting must give
the CALM sleeve more weight than the WILD one (weights sum to 1); the blended book must realize the
requested PORTFOLIO vol target (the separate leverage step), and diversify below the widest sleeve.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.portfolio.utils.blend import (
    blend_strategies,
    inverse_vol_weights,
    trailing_vol,
)

_ANN = np.sqrt(252.0)


def _sleeves(n: int = 800, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-01", periods=n)
    return pd.DataFrame({
        "calm": rng.normal(0.0002, 0.05 / _ANN, n),    # 5% ann vol
        "wild": rng.normal(0.0002, 0.20 / _ANN, n),    # 20% ann vol
        "mid":  rng.normal(0.0002, 0.10 / _ANN, n),    # 10% ann vol
    }, index=idx)


def test_inverse_vol_and_portfolio_vol_target():
    rets = _sleeves()
    w = inverse_vol_weights(rets, window=63, scheme="inverse_vol", max_weight=0.9)
    wsum = w.sum(axis=1)
    valid = wsum[wsum > 0]
    assert np.allclose(valid, 1.0, atol=1e-6), "mix weights must sum to 1 each active day"
    mw = w.replace(0.0, np.nan).mean()
    assert mw["calm"] > mw["mid"] > mw["wild"], f"inverse-vol order wrong: {mw.round(3).to_dict()}"

    blended, weights = blend_strategies(rets, portfolio_vol_target=0.10, vol_window=63,
                                        scheme="inverse_vol", max_weight=0.9, max_leverage=3.0)
    real = blended["ret"].dropna()
    blended_vol = float(real.std() * _ANN)
    sleeve_vols = (rets.std() * _ANN)
    assert 0.08 <= blended_vol <= 0.12, f"portfolio vol target missed: {blended_vol:.3f}"
    assert blended_vol < sleeve_vols.max(), "blend should diversify below the widest sleeve"

    print("\n=== SANITY CHECK: vol-based day-by-day strategy blender ===")
    print(f"  sleeve ann vols: {sleeve_vols.round(3).to_dict()}")
    print(f"  avg inverse-vol weights: {mw.round(3).to_dict()} (calm > mid > wild)")
    print(f"  blended realized ann vol {blended_vol:.3f} (target 0.10), "
          f"avg leverage {blended['leverage'].mean():.2f}")
    print("  CONCLUSION: risk-parity mix favors the calm sleeve; a separate leverage hits the "
          "global vol target without collapsing exposure. Day-by-day, point-in-time.")


if __name__ == "__main__":
    test_inverse_vol_and_portfolio_vol_target()
