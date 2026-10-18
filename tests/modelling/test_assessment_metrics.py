"""Known-truth temporal and ex-ante cohort assessment contracts."""

import numpy as np
import pandas as pd
import pytest

from src.modelling.utils.assessment_metrics import formation_cohorts, forward_returns, matched_cohort_returns, temporal_diagnostics, trend_summary


def test_daily_diagnostics_preserve_dates_and_model_row_alignment() -> None:
    dates = pd.to_datetime(["2000-01-03 09:30", "2000-01-04 10:45", "2000-01-05 15:00"])
    predictions = pd.DataFrame(
        {
            "date": np.repeat(dates, [5, 4, 6]),
            "ticker": list("ABCDEABCDABCDEF"),
            "y": [0, 1, 2, 3, np.nan, 0, 1, 2, 3, 0, 1, 2, 3, np.nan, np.nan],
            "member": [0, 1, 2, 3, 4, 3, 2, 1, 0, 0, 2, 1, 3, 4, 5],
            "fold": np.repeat([1, 1, 2], [5, 4, 6]),
        }
    ).iloc[::-1]
    predictions["inverse"] = -predictions.member
    stats, daily = temporal_diagnostics(predictions, ["member", "inverse"], 60, 2, 0, 0.95, 4325)
    assert "date" in daily.columns
    assert "index" not in daily.columns
    expected = pd.DataFrame(
        {
            "date": np.tile(dates, 2),
            "model": ["member"] * 3 + ["inverse"] * 3,
            "fold": [1, 1, 2] * 2,
            "universe": [5, 4, 6] * 2,
            "paired": [4] * 6,
        }
    )
    pd.testing.assert_frame_equal(daily[expected.columns], expected)
    assert daily.ic.to_numpy() == pytest.approx([1.0, -1.0, 0.8, -1.0, 1.0, -0.8])
    assert stats["member"]["mean_ic"] == pytest.approx(0.8 / 3)
    assert stats["inverse"]["mean_ic"] == pytest.approx(-0.8 / 3)
    print("SANITY: both models retain exact formation timestamps, fold/count alignment and known ICs; no anonymous index column is exported.")


def test_weak_window_minimizes_mean_ic_instead_of_ir() -> None:
    dates = pd.bdate_range("2000-01-03", periods=253)
    predictions = pd.DataFrame(
        {
            "date": np.repeat(dates, 4),
            "ticker": list("ABCD") * len(dates),
            "y": [0.0, 1.0, 2.0, 3.0] * len(dates),
            "member": [3.0, 2.0, 1.0, 0.0] + [1.0, 2.0, 3.0, 0.0, 2.0, 1.0, 3.0, 0.0] * 126,
            "fold": 1,
        }
    )
    expected_ic = np.asarray([-1.0] + [-0.2, -0.4] * 126)
    stats, daily = temporal_diagnostics(predictions, ["member"], 60, 252, 2, 0.95, 4325)
    assert daily.ic.to_numpy() == pytest.approx(expected_ic)
    assert daily.rolling_ic.idxmin() == 251
    assert daily.rolling_ir.idxmin() == 252
    weakest = stats["member"]
    assert weakest["worst_rolling_end"] == dates[251]
    assert weakest["worst_rolling_start"] == dates[0]
    assert weakest["worst_rolling_mean_ic"] == pytest.approx(expected_ic[:252].mean())
    assert weakest["worst_rolling_ic_ir"] == pytest.approx(expected_ic[:252].mean() / expected_ic[:252].std() * np.sqrt(252 / 60))
    assert weakest["worst_rolling_n_days"] == 252
    assert weakest["mean_ic"] == pytest.approx(expected_ic.mean())
    print("SANITY: the weakest full 252-session window minimizes mean IC, despite a different IR minimum; its dates, mean, IR and count reconcile.")


@pytest.mark.parametrize("sessions", [251, 252])
def test_weak_window_requires_a_full_window_but_not_finite_ir(sessions: int) -> None:
    dates = pd.bdate_range("2000-01-03", periods=sessions)
    predictions = pd.DataFrame(
        {
            "date": np.repeat(dates, 4),
            "ticker": list("ABCD") * sessions,
            "y": [0.0, 1.0, 2.0, 3.0] * sessions,
            "member": [3.0, 2.0, 1.0, 0.0] * sessions,
            "fold": 1,
        }
    )
    stats, daily = temporal_diagnostics(predictions, ["member"], 60, 252, 0, 0.95, 4325)
    weakest = stats["member"]
    if sessions < 252:
        assert daily.rolling_ic.isna().all()
        assert weakest["worst_rolling_end"] is None
        assert weakest["worst_rolling_start"] is None
        assert weakest["worst_rolling_mean_ic"] is None
        assert weakest["worst_rolling_ic_ir"] is None
        assert weakest["worst_rolling_n_days"] == 0
    else:
        assert weakest["worst_rolling_end"] == dates[-1]
        assert weakest["worst_rolling_start"] == dates[0]
        assert weakest["worst_rolling_mean_ic"] == pytest.approx(-1.0)
        assert weakest["worst_rolling_n_days"] == 252
        assert np.isneginf(weakest["worst_rolling_ic_ir"])
    print(
        "SANITY: fewer than 252 valid daily ICs leave the weak period unsupported; a full constant-IC window remains identifiable with undefined IR."
    )


def test_cohorts_are_fixed_before_future_outcomes() -> None:
    scores = pd.DataFrame({"date": [pd.Timestamp("2022-01-03")] * 11, "ticker": [f"T{i:02}" for i in range(11)], "lgbm": np.arange(11.0)})
    chosen = formation_cohorts(scores, ["lgbm"])
    assert set(chosen.ticker) == {"T09", "T10"}
    assert chosen.weight.tolist() == [0.5, 0.5]
    outcomes = scores[["date", "ticker"]].assign(realized_return=0.1, valid=True)
    outcomes.loc[outcomes.ticker == "T10", ["realized_return", "valid"]] = [np.nan, False]
    market = pd.DataFrame({"date": [scores.date.iloc[0]], "market_return": [0.02], "market_valid": [True]})
    result = matched_cohort_returns(chosen, outcomes, market)
    assert pd.isna(result.cohort_return.iloc[0])
    assert result.selected.iloc[0] == 2 and result.supported.iloc[0] == 1
    assert result.known_weight.iloc[0] == 0.5 and result.known_contribution.iloc[0] == pytest.approx(0.05)
    assert set(chosen.ticker) == {"T09", "T10"}
    print("SANITY: ceil(N/10) predictions define fixed equal weights; a missing selected outcome cannot replace or reweight a stock.")


def test_positive_endpoints_cannot_hide_an_interior_masked_return() -> None:
    calendar = pd.bdate_range("2022-01-03", periods=6)
    prices = pd.DataFrame({"date": calendar, "ticker": "A", "close_total": 100 * 1.01 ** np.arange(6), "ret": [0.01] * 6})
    valid = forward_returns(prices, calendar, 3)
    first = valid.iloc[0]
    assert first.valid and first.realized_return == pytest.approx(1.01**3 - 1)
    prices.loc[2, "ret"] = np.nan
    rejected = forward_returns(prices, calendar, 3).iloc[0]
    assert not rejected.valid and rejected.reason == "incomplete_path"
    assert pd.isna(rejected.realized_return)
    prices.loc[2, "ret"] = 0.20
    seam = forward_returns(prices, calendar, 3).iloc[0]
    assert not seam.valid and seam.reason == "endpoint_path_mismatch"
    print("SANITY: every interior bar is required; masked seams and price/return disagreement fail despite positive endpoints.")


def test_trend_assessment_distinguishes_flat_and_reversing_curves() -> None:
    assert trend_summary(np.arange(5), np.zeros(5))["shape"] == "flat"
    assert trend_summary(np.arange(5), np.arange(5))["shape"] == "increasing"
    assert trend_summary(np.arange(5), -np.arange(5))["shape"] == "decreasing"
    assert trend_summary(np.arange(5), [0, 2, 3, 2, 0])["shape"] == "nonmonotonic"
    print("SANITY: measured flat, directional and reversing PDP shapes have separate interpretations; direction alone is not causal evidence.")


@pytest.mark.parametrize(
    ("levels", "shape", "increasing", "decreasing"),
    [
        ([0, 1], "increasing", 1.0, 0.0),
        ([1, 0], "decreasing", 0.0, 1.0),
        ([0, 1, 0], "nonmonotonic", 0.5, 0.5),
        ([1, 0, 1], "nonmonotonic", 0.5, 0.5),
    ],
)
@pytest.mark.parametrize("jitter", [0.0, 1e-8])
def test_trend_plateaus_cannot_outvote_directional_moves(levels: list[int], shape: str, increasing: float, decreasing: float, jitter: float) -> None:
    values = np.repeat(levels, 15).astype(float)
    values += np.resize([0.0, jitter, -jitter], len(values))
    result = trend_summary(np.arange(len(values)), values)
    assert result["shape"] == shape
    assert result["increasing_fraction"] == increasing
    assert result["decreasing_fraction"] == decreasing
    assert result["points"] == len(values)
    assert result["span"] == pytest.approx(np.ptp(values))
    if shape != "nonmonotonic":
        assert np.sign(result["rho"]) == (1 if shape == "increasing" else -1)
    print(f"SANITY: {levels} plateaus with jitter {jitter:g} remain {shape}; only material directional moves vote.")


@pytest.mark.parametrize(("direction", "shape"), [(1, "increasing"), (-1, "decreasing")])
def test_trend_preserves_gentle_cumulative_direction(direction: int, shape: str) -> None:
    values = 1 + direction * np.linspace(0, 4e-6, 30)
    tolerance = np.max(np.abs(values)) * 1e-6
    assert np.max(np.abs(np.diff(values))) < tolerance < np.ptp(values)
    result = trend_summary(np.arange(len(values)), values)
    assert result["shape"] == shape
    assert result["increasing_fraction"] == (1 if direction > 0 else 0)
    assert result["decreasing_fraction"] == (1 if direction < 0 else 0)
    assert result["rho"] == pytest.approx(direction)
    print(f"SANITY: individually sub-tolerance steps retain their meaningful cumulative {shape} direction.")


def test_trend_retains_gentle_moves_alongside_a_material_reversal() -> None:
    values = np.r_[1 + np.linspace(0, 4e-6, 30), 1 + 2e-6]
    result = trend_summary(np.arange(len(values)), values)
    assert result["shape"] == "nonmonotonic"
    assert 0 < result["increasing_fraction"] < 0.9
    assert 0 < result["decreasing_fraction"] < 0.9
    print("SANITY: cumulative gentle increases and a material reversal both count; neither can disappear behind the tolerance.")


def test_trend_retains_directional_threshold_and_flat_tolerance() -> None:
    directional = trend_summary(np.arange(11), [0, 1, 2, 3, 2, 3, 4, 5, 6, 7, 8])
    assert directional["shape"] == "increasing"
    assert directional["increasing_fraction"] == pytest.approx(0.9)
    assert directional["decreasing_fraction"] == pytest.approx(0.1)
    for values in [np.asarray([0, 1e-9, -1e-9]), 1 + np.asarray([0, 1e-7, -1e-7])]:
        flat = trend_summary(np.arange(len(values)), values)
        assert flat == {"shape": "flat", "points": len(values), "rho": None, "span": float(np.ptp(values))}
    print("SANITY: the existing 90% directional threshold and absolute/relative flat tolerances remain intact.")


def test_trend_without_a_material_anchor_move_has_no_direction_votes() -> None:
    values = 1 + np.asarray([0, 0.75e-6, -0.75e-6])
    assert np.ptp(values) > np.max(np.abs(values)) * 1e-6
    result = trend_summary(np.arange(len(values)), values)
    assert result["shape"] == "nonmonotonic"
    assert result["increasing_fraction"] == result["decreasing_fraction"] == 0.0
    print(
        "SANITY: a centered reversing curve can span the tolerance without leaving its first anchor; zero material moves cast zero direction votes."
    )


@pytest.mark.parametrize(
    ("x", "y", "points"),
    [([], [], 0), ([0], [1], 1), ([1, 1], [0, 1], 2), ([0, 1], [np.nan, np.inf], 0), ([0, np.inf], [1, 2], 1)],
)
def test_trend_requires_two_finite_pairs_with_distinct_x(x: list[float], y: list[float], points: int) -> None:
    assert trend_summary(x, y) == {"shape": "unsupported", "points": points, "rho": None, "span": None}
    print(f"SANITY: {points} usable pairs without distinct supported x values retain the unsupported artifact shape.")


def test_trend_sorts_and_filters_only_finite_pairs() -> None:
    result = trend_summary([3, np.nan, 1, 2, 4, 5, 6], [0, 8, 1, 1, np.nan, np.inf, 0])
    assert result == {
        "shape": "decreasing",
        "points": 4,
        "rho": pytest.approx(-2 / np.sqrt(5)),
        "span": 1.0,
        "increasing_fraction": 0.0,
        "decreasing_fraction": 1.0,
    }
    print("SANITY: unsorted/nonfinite observations leave four sorted finite pairs and the same decreasing plateau direction.")
