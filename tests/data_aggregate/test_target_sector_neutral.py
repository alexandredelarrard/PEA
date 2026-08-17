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

from src.data_aggregate.utils.common.prices import forward_compound
from src.data_aggregate.utils.common.xs import xs_group_dummies, xs_project_out
from src.data_aggregate.utils.target.targets import (
    build_targets_multi, compute_epsilon, cross_sectional_rank,
)


def _group_neutralize(values: pd.DataFrame, group_map: dict) -> pd.DataFrame:
    """Per-day within-group demean, expressed the way the target now does it: an indicator
    block in `xs_project_out`. An OLS residual is exactly orthogonal to every indicator
    column, so each group's per-day mean is exactly zero -- the property the old dedicated
    `cross_sectional_group_neutralize` provided by construction."""
    return xs_project_out(values, [], xs_group_dummies(group_map, values.columns))


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
    out = _group_neutralize(eps, sector)
    assert np.allclose(out[hot].mean(axis=1), 0.0, atol=1e-9)     # each group's per-day mean = 0
    assert np.allclose(out[cold].mean(axis=1), 0.0, atol=1e-9)
    # within-group RELATIVE spacing preserved (demean only removes a per-day constant)
    assert np.allclose(out[hot[0]] - out[hot[1]], eps[hot[0]] - eps[hot[1]])
    # An unmapped ticker now joins a shared `__UNK__` group and is demeaned WITHIN it, rather
    # than being left untouched as the old dedicated demeaner did. That is the point of the
    # indicator block: every name sits in exactly one group, so the block spans the constant.
    partial = {t: "HOT" for t in hot}                            # cold unmapped
    out2 = _group_neutralize(eps, partial)
    assert np.allclose(out2[cold].mean(axis=1), 0.0, atol=1e-9)
    assert np.allclose(out2[cold[0]] - out2[cold[1]], eps[cold[0]] - eps[cold[1]])
    print("\n=== SANITY CHECK: group demean via the indicator block ===")
    print("  each group's per-day mean -> 0; within-group spread preserved; unmapped names "
          "demeaned within a shared __UNK__ group. Validated.")


def test_sector_neutralization_kills_sector_prediction():
    eps, sector, hot, cold = _panel_with_sector_tilt()
    # BEFORE: the sector tilt makes sector membership PREDICT the rank target
    raw = cross_sectional_rank(eps, min_names=2)
    hot_raw, cold_raw = raw[hot].mean().mean(), raw[cold].mean().mean()
    assert hot_raw > 0.7 and cold_raw < 0.3                       # sector predicts -> NOT neutral
    # AFTER GICS neutralization: each sector's mean rank ~0.5 -> sector no longer predicts
    neu = cross_sectional_rank(_group_neutralize(eps, sector), min_names=2)
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
    industry = {"A": "X1", "B": "X1", "C": "X2", "D": "X2"}
    # ONLY the industry indicators are used: industry is NESTED in sector, so a sector mean is
    # a weighted average of its industries' (zero) means and comes out zero for free.
    out = _group_neutralize(eps, industry)
    assert np.allclose(out[cols].mean(axis=1), 0.0, atol=1e-9)        # sector mean 0
    assert np.allclose(out[["A", "B"]].mean(axis=1), 0.0, atol=1e-9)  # AND each industry mean 0
    assert np.allclose(out[["C", "D"]].mean(axis=1), 0.0, atol=1e-9)
    print("\n=== SANITY CHECK: sector + industry both neutral ===")
    print("  industry indicators alone (industry nested in sector) -> both levels' per-day "
          "means 0. Validated.")


# --------------------------------------------------------------------------- #
# The per-stock `beta_sector` path (NOT the group demean above)                 #
#                                                                              #
# `sector_excess` is the ONE regressor that differs per stock, so it travels as #
# a date x ticker frame -- the same shape `estimate_all_betas` takes. It used   #
# to be typed as a {name: frame} dict here while the caller passed the frame,   #
# which raised on `(per_stock_factors or {})` and, but for that raise, would    #
# have looked up `beta_<ticker>` and silently skipped the term. Both modes are  #
# pinned below.                                                                 #
# --------------------------------------------------------------------------- #
def _panel_with_sector_excess(seed: int = 3):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-01", periods=260)
    tickers = [f"T{i}" for i in range(25)]

    sector_excess = pd.DataFrame(rng.normal(0, 0.008, (len(dates), len(tickers))),
                                 index=dates, columns=tickers)
    # every stock loads +0.9 on its OWN sector basket, so the term is never a no-op
    ret = 0.9 * sector_excess + rng.normal(0, 0.004, (len(dates), len(tickers)))
    close = pd.DataFrame(100 * np.cumprod(1 + ret.to_numpy(), axis=0),
                         index=dates, columns=tickers)
    factor_panel = pd.DataFrame({"market": rng.normal(0, 0.01, len(dates))}, index=dates)
    betas = {t: pd.DataFrame({"beta_market": 1.0, "beta_sector": 0.9}, index=dates)
             for t in tickers}
    return close, betas, factor_panel, sector_excess, tickers


def test_sector_excess_frame_flows_through_build_targets_multi():
    """The caller hands a date x ticker FRAME. This raised ValueError
    ('truth value of a DataFrame is ambiguous') before the contract was unified."""
    close, betas, factor_panel, sector_excess, tickers = _panel_with_sector_excess()

    out = build_targets_multi(close, betas, factor_panel, macro_cols=[],
                              horizons=(20,), labels=("rank",), min_names=5,
                              sector_groups={"sector": {t: "S" for t in tickers}},
                              sector_excess=sector_excess)

    non_null = int(out[20]["rank"].notna().sum().sum())
    assert non_null > 0, "targets are empty -> the sector frame did not flow through"
    print("\n=== SANITY CHECK: sector_excess frame flows through build_targets_multi ===")
    print(f"  passed a {sector_excess.shape[0]}x{sector_excess.shape[1]} date x ticker frame "
          f"-> {non_null} non-null labels (previously ValueError). Validated.")


def test_sector_beta_actually_changes_epsilon():
    """A signature-only fix would still let the term be silently skipped, so assert
    the sector loading genuinely moves epsilon rather than merely not crashing."""
    close, betas, factor_panel, sector_excess, _ = _panel_with_sector_excess()

    with_sector = compute_epsilon(close, betas, factor_panel, [], 20,
                                  sector_excess=sector_excess)
    without = compute_epsilon(close, betas, factor_panel, [], 20, sector_excess=None)

    both = with_sector.notna() & without.notna()
    assert both.to_numpy().any(), "no overlapping non-null cells to compare"
    max_diff = (with_sector - without)[both].abs().max().max()
    assert max_diff > 1e-6, "beta_sector was NOT subtracted -- the term is a silent no-op"

    # and it REMOVES exposure rather than merely perturbing it. Compare against the
    # FORWARD-compounded basket -- the object actually subtracted -- not the daily one.
    fwd_sector = forward_compound(sector_excess, 20)
    flat = lambda d: d.where(both).to_numpy().ravel()
    keep = ~(np.isnan(flat(with_sector)) | np.isnan(flat(fwd_sector)))
    corr_with = abs(np.corrcoef(flat(with_sector)[keep], flat(fwd_sector)[keep])[0, 1])
    corr_without = abs(np.corrcoef(flat(without)[keep], flat(fwd_sector)[keep])[0, 1])
    assert corr_without > 0.5, "the synthetic panel should start with heavy sector exposure"
    assert corr_with < 0.1, f"sector exposure survived in epsilon (corr={corr_with:.3f})"

    print("\n=== SANITY CHECK: beta_sector is genuinely subtracted ===")
    print(f"  max |eps(with) - eps(without)| = {max_diff:.5f} (a no-op would be 0)")
    print(f"  |corr(eps, fwd sector basket)|  without={corr_without:.3f} -> with={corr_with:.3f}")
    print("  -> the per-stock sector loading is stripped, not just accepted. Validated.")


if __name__ == "__main__":
    test_group_neutralize_zeros_group_means_preserves_spread()
    test_sector_neutralization_kills_sector_prediction()
    test_neutralize_both_sector_and_industry()
    test_sector_excess_frame_flows_through_build_targets_multi()
    test_sector_beta_actually_changes_epsilon()
