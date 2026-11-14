"""Known-truth tests of every leg preset: the shared core, peer, own-history, universe z and universe rank."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_aggregate.utils.common.panel import peer_relative
from src.data_aggregate.utils.common.xs import HIST_MIN_PERIODS, Z_CLIP, self_history_z, xs_rank_pct, xs_z, zscore

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


def _spike_frame(n_days: int) -> pd.DataFrame:
    """A noisy universe plus one name that is flat then jumps on the last day: its z is exactly (n - 1) / sqrt(n)."""
    rng = np.random.default_rng(11)
    index = pd.bdate_range("2021-01-04", periods=n_days)
    frame = pd.DataFrame(rng.normal(size=(n_days, 99)), index=index, columns=[f"T{i}" for i in range(99)])
    frame["SPIKE"] = 0.0
    frame.iloc[-1, frame.columns.get_loc("SPIKE")] = 7.0
    return frame


def test_own_history_value_below_the_clip_is_not_winsorized() -> None:
    frame = _spike_frame(26)
    z = self_history_z(frame, window=26, min_periods=26).iloc[-1]
    expected = 25 / np.sqrt(26)
    assert np.isclose(z["SPIKE"], expected), f"a value under the clip must pass through untouched: {z['SPIKE']}"
    assert z["SPIKE"] > z.drop("SPIKE").quantile(0.99), "the spike is the top of the day, a 99% trim would have cut it"
    print(f"SANITY: the spike keeps its exact z {z['SPIKE']:.4f} = 25/sqrt(26) although it is the day's maximum.")


def test_own_history_is_clipped_at_the_shared_constant() -> None:
    frame = _spike_frame(30)
    z = self_history_z(frame, window=30, min_periods=30).iloc[-1]
    assert 29 / np.sqrt(30) > Z_CLIP, "fixture must exceed the bound"
    assert z["SPIKE"] == Z_CLIP == 5.0, f"the clip is the shared +-5: {z['SPIKE']}"
    print(f"SANITY: raw z {29 / np.sqrt(30):.3f} is clipped to exactly {z['SPIKE']}; 8 no longer binds.")


def _universe_with_basket(subject: float, peer_values: list[float]) -> tuple[pd.DataFrame, dict]:
    """Forty universe names spread evenly over [-1.7, 1.7] (std about 1), a subject `S` and a four-name basket."""
    peers = [f"P{i}" for i in range(len(peer_values))]
    universe = pd.DataFrame({f"U{i}": float(v) for i, v in enumerate(np.linspace(-1.7, 1.7, 40))}, index=DATES)
    named = pd.DataFrame({"S": subject, **dict(zip(peers, peer_values, strict=True))}, index=DATES)
    return pd.concat([named, universe], axis=1), {"S": dict.fromkeys(peers, 0.25)}


def test_peer_score_is_clipped_at_the_shared_bound_and_below_it_untouched() -> None:
    frame, basket = _universe_with_basket(10.0, [0.0, 0.0, 0.0, 0.001])
    peers = ["P0", "P1", "P2", "P3"]
    raw = (10.0 - frame[peers].mean(axis=1)) / (0.5 * frame.std(axis=1))
    assert (raw > Z_CLIP).all(), "fixture must exceed the bound"
    assert (peer_relative(frame, basket)["S"] == Z_CLIP).all()
    inside, basket_inside = _universe_with_basket(1.0, [0.0, 0.5, -0.5, 0.2])
    expected = (1.0 - inside[peers].mean(axis=1)) / np.maximum(0.5 * inside.std(axis=1), inside[peers].std(axis=1, ddof=0))
    assert np.allclose(peer_relative(inside, basket_inside)["S"], expected) and (expected.abs() < Z_CLIP).all()
    print(f"SANITY: raw z {raw.iloc[0]:.2f} is clipped to exactly {Z_CLIP}; a z of {expected.iloc[0]:.2f} passes through unchanged.")


def test_peer_inputs_are_not_trimmed() -> None:
    peers = [f"P{i}" for i in range(5)]
    frame = pd.DataFrame({"S": 1.0, **dict(zip(peers, [1.0, 1.0, 1.0, 1.0, 1000.0], strict=True))}, index=DATES)
    frame = pd.concat([frame, pd.DataFrame({f"U{i}": 1.0 + 0.01 * i for i in range(60)}, index=DATES)], axis=1)
    out = peer_relative(frame, {"S": dict.fromkeys(peers, 0.2)})["S"]
    mean = frame[peers].mean(axis=1)
    std = frame[peers].std(axis=1, ddof=0)
    assert np.allclose(out, ((1.0 - mean) / np.maximum(0.5 * frame.std(axis=1), std)).clip(-Z_CLIP, Z_CLIP))
    assert (mean > 200).all(), "the 1000 stays in the basket mean: no 1%/99% trim"
    print(f"SANITY: the outlier peer value 1000 moves the basket mean to {mean.iloc[0]:.1f}; nothing was trimmed.")


def test_peer_non_finite_subject_is_nan_never_a_fabricated_clip() -> None:
    frame, basket = _universe_with_basket(3.0, [1.0, 2.0, 3.0, 4.0])
    frame.loc[DATES[0], "S"] = np.inf
    out = peer_relative(frame, basket)["S"]
    assert np.isnan(out.iloc[0]) and out.iloc[1:].notna().all(), "an inf subject is missing, not clipped to +5"
    print("SANITY: an inf subject value gives NaN rather than the clip bound.")


def test_peer_basket_never_contains_the_subject() -> None:
    frame, basket = _universe_with_basket(10.0, [1.0, 2.0, 3.0, 4.0])
    other = frame.copy()
    other["S"] = 20.0
    first = peer_relative(frame, basket)["S"]
    second = peer_relative(other, basket)["S"]
    mean, std = 2.5, np.sqrt(1.25)
    floor = 0.5 * frame.std(axis=1)
    assert np.allclose(first, np.minimum((10.0 - mean) / np.maximum(floor, std), Z_CLIP)), "the score uses the peers' moments only"
    assert (second >= first).all()
    print("SANITY: the subject's value enters only as the numerator; the basket mean 2.5 and std 1.118 come from the four peers.")
