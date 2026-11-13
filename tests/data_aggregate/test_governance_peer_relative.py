"""Governance peer legs use the shared `peer_relative`: (X - weighted peer mean) / max(weighted peer std, 0.10 x universe std), no trim, no clip."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_aggregate.utils.common.panel import peer_relative
from src.data_aggregate.utils.governance.panel import _peer_only

DATES = pd.bdate_range("2020-01-01", periods=6)
PEERS = [f"P{i}" for i in range(7)]
BASKET = {"S": dict.fromkeys(PEERS, 1.0 / 7.0)}


def _frame(subject: float, peer_values: list[float], present: int | None = None) -> pd.DataFrame:
    """One subject `S` and seven peers over `DATES`; only the first `present` peers report (all when None)."""
    data = {"S": [subject] * len(DATES)}
    for i, value in enumerate(peer_values):
        data[PEERS[i]] = [value if present is None or i < present else np.nan] * len(DATES)
    return pd.DataFrame(data, index=DATES)


def _expected(frame: pd.DataFrame) -> pd.Series:
    """Hand calculation of the definition for equal weights and a full basket."""
    mean = frame[PEERS].mean(axis=1)
    std = frame[PEERS].std(axis=1, ddof=0)
    floor = 0.10 * frame.std(axis=1)
    return (frame["S"] - mean) / std.clip(lower=floor)


def test_value_is_centred_on_the_peer_mean_and_scaled_by_the_peer_std() -> None:
    frame = _frame(subject=10.0, peer_values=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    out = peer_relative(frame, BASKET)
    # peers 1..7: mean 4, population std 2 -> (10 - 4) / 2 = 3
    assert np.allclose(out["S"], 3.0)
    pd.testing.assert_series_equal(out["S"], _expected(frame), check_names=False)
    print(f"SANITY: (10 - peer mean 4) / peer std 2 = {out['S'].iloc[0]:.3f}.")


def test_the_peer_std_is_floored_at_a_tenth_of_the_universe_std() -> None:
    frame = _frame(subject=6.0, peer_values=[5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.001])
    universe = pd.DataFrame({f"U{i}": [float(i)] * len(DATES) for i in range(50)}, index=DATES)
    full = pd.concat([frame, universe], axis=1)
    out = peer_relative(full, BASKET)["S"]
    floor = 0.10 * full.std(axis=1)
    assert (full[PEERS].std(axis=1, ddof=0) < floor).all(), "the basket must be tighter than the floor"
    pd.testing.assert_series_equal(out, (6.0 - full[PEERS].mean(axis=1)) / floor, check_names=False)
    print(f"SANITY: peers 5.0..5.001, subject 6.0 -> z={out.iloc[0]:.2f}, divided by the floor {floor.iloc[0]:.2f}, not by ~4e-4.")


def test_peers_that_all_report_the_same_value_give_nan() -> None:
    frame = _frame(subject=6.0, peer_values=[5.0] * 7)
    unit_weights = {"S": dict.fromkeys(PEERS, 1.0)}  # exact arithmetic: a mean of seven 5.0 is exactly 5.0
    assert peer_relative(frame, unit_weights)["S"].isna().all()
    print("SANITY: a zero-spread basket cannot rank anything, so the score stays NaN instead of dividing by the floor.")


def test_fewer_than_three_peers_with_data_is_nan_and_three_is_enough() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    two = peer_relative(_frame(10.0, values, present=2), BASKET)
    three = peer_relative(_frame(10.0, values, present=3), BASKET)
    assert two["S"].isna().all(), "two peers must not be enough"
    assert three["S"].notna().all(), "three peers must be enough"
    print("SANITY: 2 reporting peers -> NaN; 3 -> a value, so early history with few peers stays blank.")


def test_the_peer_count_is_evaluated_per_date() -> None:
    frame = _frame(10.0, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    frame.loc[DATES[:3], ["P2", "P3", "P4", "P5", "P6"]] = np.nan
    out = peer_relative(frame, BASKET)["S"]
    assert out.iloc[:3].isna().all() and out.iloc[3:].notna().all()
    print("SANITY: the same basket is blank on the first three dates (2 peers) and live afterwards (7 peers).")


def test_an_extreme_value_is_not_trimmed_or_clipped() -> None:
    frame = _frame(subject=1_000.0, peer_values=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    universe = pd.DataFrame({f"U{i}": [1.0 + i % 10] * len(DATES) for i in range(100)}, index=DATES)
    value = float(peer_relative(pd.concat([frame, universe], axis=1), BASKET)["S"].iloc[0])
    assert value > 8.0, f"a clip or a winsorize reached the value: {value}"
    print(f"SANITY: a 1000 against peers around 4 stays {value:.1f}, far beyond the former +-8 clip.")


def test_each_governance_field_is_transformed_before_scoring() -> None:
    for name, transform in (("ceo_pay_ratio", np.log1p), ("insider_ownership_pct", np.sqrt), ("board_size", lambda x: x)):
        frame = _frame(subject=40.0, peer_values=[1.0, 2.0, 3.0, 4.0, 9.0, 16.0, 25.0])
        long = _peer_only({name: frame}, BASKET)
        got = long.set_index(["date", "ticker"])[f"f_{name}_vs_peers"].xs("S", level="ticker")
        expected = _expected(transform(frame))
        pd.testing.assert_series_equal(got, expected.astype("float32"), check_names=False, check_freq=False, rtol=1e-5)
    print("SANITY: log1p for ceo_pay_ratio, sqrt for insider_ownership_pct, none for board_size, each scored by the shared peer_relative.")
