"""
Target neutralization to the ACTUAL GICS sector + industry_group (per-day within-
group demeaning), replacing the return-correlation peer-basket ("neighbor sector").

The point: if the target keeps a sector/industry tilt, sector & industry_group
become top model drivers (a tell that the target is NOT sector-neutral). These tests
prove the group-demean zeroes every group's per-day mean and that, post-neutralization,
sector membership can no longer predict the rank target.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_aggregate.utils.target.targets import (
    cross_sectional_group_neutralize, cross_sectional_rank, _neutralize_sector_industry,
)


def _panel_with_sector_tilt(seed: int = 0):
    dates = pd.bdate_range("2024-01-01", periods=25)
    tickers = [f"T{i}" for i in range(20)]
    sector = {t: ("HOT" if i < 10 else "COLD") for i, t in enumerate(tickers)}
    rng = np.random.default_rng(seed)
    eps = pd.DataFrame(rng.standard_normal((len(dates), len(tickers))), index=dates, columns=tickers)
    for t, s in sector.items():
        if s == "HOT":
            eps[t] += 5.0                                    # strong per-day sector tilt
    hot = [t for t in tickers if sector[t] == "HOT"]
    cold = [t for t in tickers if sector[t] == "COLD"]
    return eps, sector, hot, cold


def test_group_neutralize_zeros_group_means_preserves_spread():
    eps, sector, hot, cold = _panel_with_sector_tilt()
    out = cross_sectional_group_neutralize(eps, sector)
    assert np.allclose(out[hot].mean(axis=1), 0.0, atol=1e-9)     # each group's per-day mean = 0
    assert np.allclose(out[cold].mean(axis=1), 0.0, atol=1e-9)
    # within-group RELATIVE spacing preserved (demean only removes a per-day constant)
    assert np.allclose(out[hot[0]] - out[hot[1]], eps[hot[0]] - eps[hot[1]])
    # unmapped (unknown-group) tickers are left untouched
    partial = {t: "HOT" for t in hot}                            # cold unmapped
    out2 = cross_sectional_group_neutralize(eps, partial)
    assert (out2[cold] == eps[cold]).all().all()
    print("\n=== SANITY CHECK: group demean ===")
    print("  each group's per-day mean -> 0; within-group spread preserved; unknown untouched. Validated.")


def test_sector_neutralization_kills_sector_prediction():
    eps, sector, hot, cold = _panel_with_sector_tilt()
    # BEFORE: the sector tilt makes sector membership PREDICT the rank target
    raw = cross_sectional_rank(eps, min_names=2)
    hot_raw, cold_raw = raw[hot].mean().mean(), raw[cold].mean().mean()
    assert hot_raw > 0.7 and cold_raw < 0.3                       # sector predicts -> NOT neutral
    # AFTER GICS neutralization: each sector's mean rank ~0.5 -> sector no longer predicts
    neu = cross_sectional_rank(cross_sectional_group_neutralize(eps, sector), min_names=2)
    hot_neu, cold_neu = neu[hot].mean().mean(), neu[cold].mean().mean()
    assert abs(hot_neu - 0.5) < 0.08 and abs(cold_neu - 0.5) < 0.08
    print("\n=== SANITY CHECK: target sector-neutrality ===")
    print(f"  raw mean rank HOT={hot_raw:.2f} COLD={cold_raw:.2f} (sector PREDICTS) -> "
          f"neutralized HOT={hot_neu:.2f} COLD={cold_neu:.2f} (~0.5, sector can't predict). Validated.")


def test_neutralize_both_sector_and_industry():
    dates = pd.bdate_range("2024-01-01", periods=8)
    cols = ["A", "B", "C", "D"]
    # sector X = {A,B,C,D}; industries X1={A,B} (high), X2={C,D} (low)
    eps = pd.DataFrame(np.tile([10.0, 12.0, -1.0, 1.0], (len(dates), 1)), index=dates, columns=cols)
    groups = {"sector": {c: "X" for c in cols},
              "industry_group": {"A": "X1", "B": "X1", "C": "X2", "D": "X2"}}
    out = _neutralize_sector_industry(eps.copy(), groups)
    assert np.allclose(out[cols].mean(axis=1), 0.0, atol=1e-9)        # sector mean 0
    assert np.allclose(out[["A", "B"]].mean(axis=1), 0.0, atol=1e-9)  # AND each industry mean 0
    assert np.allclose(out[["C", "D"]].mean(axis=1), 0.0, atol=1e-9)
    print("\n=== SANITY CHECK: sector + industry both neutral ===")
    print("  sequential demean (sector then nested industry) -> both levels' per-day means 0. Validated.")


if __name__ == "__main__":
    test_group_neutralize_zeros_group_means_preserves_spread()
    test_sector_neutralization_kills_sector_prediction()
    test_neutralize_both_sector_and_industry()
