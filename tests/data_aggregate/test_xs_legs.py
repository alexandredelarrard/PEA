"""Known-truth tests of every leg preset: the shared core, peer, own-history, universe z and universe rank."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_aggregate.utils.common.panel import peer_relative
from src.data_aggregate.utils.common.xs import HIST_MIN_PERIODS, self_history_z, xs_rank_pct, xs_z, zscore

DATES = pd.bdate_range("2021-01-04", periods=6)


def test_core_is_value_minus_centre_over_scale_with_optional_clip() -> None:
    x = pd.DataFrame({"A": [10.0, 0.0], "B": [2.0, 4.0]})
    centre = pd.Series([4.0, 4.0])
    scale = pd.Series([2.0, 2.0])
    out = zscore(x, centre, scale)
    assert out["A"].tolist() == [3.0, -2.0] and out["B"].tolist() == [-1.0, 0.0]
    clipped = zscore(x, centre, scale, clip=2.5)
    assert clipped["A"].tolist() == [2.5, -2.0], "the clip bounds the output symmetrically"
    series = zscore(x["A"], centre, scale)
    assert series.tolist() == [3.0, -2.0]
    print("SANITY: (10 - 4) / 2 = 3, (0 - 4) / 2 = -2; a clip of 2.5 caps 3 at 2.5; a Series input gives the same numbers.")


def test_peer_value_is_the_weighted_peer_mean_and_std_distance() -> None:
    peers = ["P0", "P1", "P2", "P3"]
    frame = pd.DataFrame({"S": 10.0, **dict(zip(peers, [2.0, 4.0, 6.0, 8.0], strict=True))}, index=DATES)
    out = peer_relative(frame, {"S": dict.fromkeys(peers, 0.25)})["S"]
    # peers 2,4,6,8: mean 5, population std sqrt(5) -> (10 - 5) / sqrt(5)
    assert np.allclose(out, 5.0 / np.sqrt(5.0))
    print(f"SANITY: (10 - peer mean 5) / peer std {np.sqrt(5.0):.3f} = {out.iloc[0]:.3f}, no clip applied.")


def test_peer_floor_binds_only_when_the_basket_is_tighter_than_a_fraction_of_the_universe() -> None:
    peers = ["P0", "P1", "P2", "P3"]
    universe = pd.DataFrame({f"U{i}": float(i) for i in range(40)}, index=DATES)
    frame = pd.concat([pd.DataFrame({"S": 7.0, **dict(zip(peers, [5.0, 5.0, 5.0, 5.001], strict=True))}, index=DATES), universe], axis=1)
    out = peer_relative(frame, {"S": dict.fromkeys(peers, 0.25)})["S"]
    floor = 0.10 * frame.std(axis=1)
    mean = frame[peers].mean(axis=1)
    assert (frame[peers].std(axis=1, ddof=0) < floor).all()
    assert np.allclose(out, (7.0 - mean) / floor)
    flat = frame.copy()
    flat[peers] = 5.0
    assert peer_relative(flat, {"S": dict.fromkeys(peers, 1.0)})["S"].isna().all(), "an exactly-zero spread stays NaN, never floored"
    print(f"SANITY: a near-constant basket is divided by the floor {floor.iloc[0]:.2f}, an exactly constant one gives NaN.")


def test_own_history_is_trailing_with_the_current_day_and_nan_before_min_periods() -> None:
    values = pd.Series(np.arange(1.0, 41.0), index=pd.bdate_range("2021-01-04", periods=40))
    z = self_history_z(values.to_frame("A"), window=10, min_periods=5)["A"]
    assert z.iloc[:4].isna().all(), "no value before min_periods"
    window = values.iloc[:5]
    assert np.isclose(z.iloc[4], (window.iloc[-1] - window.mean()) / window.std())
    window = values.iloc[20:30]
    assert np.isclose(z.iloc[29], (window.iloc[-1] - window.mean()) / window.std()), "trailing 10 incl. today"
    print(f"SANITY: first value on day 5 (z={z.iloc[4]:.3f}), then the z of today against the last 10 days (z={z.iloc[29]:.3f}).")


def test_own_history_has_no_look_ahead() -> None:
    rng = np.random.default_rng(7)
    base = pd.DataFrame({"A": rng.normal(size=60)}, index=pd.bdate_range("2021-01-04", periods=60))
    changed = base.copy()
    changed.iloc[40:, 0] += 100.0
    before = self_history_z(base, window=20, min_periods=5)
    after = self_history_z(changed, window=20, min_periods=5)
    assert before.iloc[:40].equals(after.iloc[:40]), "a future change moved a past z"
    print("SANITY: shifting values from day 40 on leaves days 0..39 untouched, so the window never reads the future.")


def test_own_history_constant_history_is_nan_and_default_warmup_is_the_declared_minimum() -> None:
    flat = pd.DataFrame({"A": 3.0}, index=pd.bdate_range("2021-01-04", periods=HIST_MIN_PERIODS + 5))
    assert self_history_z(flat).isna().all().all(), "zero trailing std -> NaN, never 0 or inf"
    ramp = pd.DataFrame({"A": np.arange(float(HIST_MIN_PERIODS + 5))}, index=flat.index)
    z = self_history_z(ramp)["A"]
    assert z.iloc[: HIST_MIN_PERIODS - 1].isna().all() and z.iloc[HIST_MIN_PERIODS - 1 :].notna().all()
    print(f"SANITY: a constant history is NaN; the first value appears after {HIST_MIN_PERIODS} days.")


def test_universe_z_is_the_day_cross_section_with_a_zero_sd_policy() -> None:
    day = pd.DataFrame({"A": [1.0, 5.0], "B": [2.0, 5.0], "C": [3.0, 5.0]}, index=DATES[:2])
    z = xs_z(day, clip=None)
    assert np.allclose(z.iloc[0], [-1.0, 0.0, 1.0]), "mean 2, sample std 1"
    guarded = xs_z(day, clip=3.0)
    assert guarded.iloc[1].isna().all(), "a constant day divides 0 by NaN under the eps guard"
    assert xs_z(day, clip=0.5).iloc[0].tolist() == [-0.5, 0.0, 0.5], "clip bounds symmetrically"
    print("SANITY: values 1,2,3 -> -1,0,1 (mean 2, sample std 1); a constant day gives NaN; clip 0.5 caps at +-0.5.")


def test_universe_rank_is_the_average_percentile_per_day() -> None:
    day = pd.DataFrame({"A": [10.0], "B": [20.0], "C": [20.0], "D": [40.0]})
    rank = xs_rank_pct(day).iloc[0]
    assert rank.tolist() == [0.25, 0.625, 0.625, 1.0], "ties share the average rank"
    print("SANITY: ranks 1, 2.5, 2.5, 4 of 4 -> 0.25, 0.625, 0.625, 1.0.")
