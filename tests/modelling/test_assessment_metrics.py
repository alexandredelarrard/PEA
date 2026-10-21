"""Known-truth temporal and ex-ante cohort assessment contracts."""

import numpy as np
import pandas as pd
import pytest

from src.modelling.utils.assessment_metrics import native_error_diagnostics, tail_universes, temporal_diagnostics, universe_diagnostics


def test_tail_universe_is_selected_before_outcomes() -> None:
    frame = pd.DataFrame(
        {"date": pd.Timestamp("2021-01-04"), "ticker": list("ABCDEF"), "member": range(6), "y": [0.0, 1.0, 2.0, 3.0, 4.0, np.nan], "fold": 1}
    )
    chosen = tail_universes(frame.iloc[::-1], "member", (2,))[2]
    assert set(chosen.ticker) == set("ABEF") and "D" not in set(chosen.ticker)
    stats, daily = universe_diagnostics(frame, ["member"], 60, 2, (2,))
    assert daily.universe.tolist() == [4] and daily.paired.tolist() == [3]
    assert stats["2"]["member"]["mean_ic"] == pytest.approx(1.0)
    print("SANITY: missing output on the top prediction does not promote a middle ticker; union4, comparable3, IC1.")


def test_tail_universes_handle_ties_overlap_and_distinct_model_rankings() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.Timestamp("2021-01-04"),
            "ticker": list("ABCDEFG"),
            "member": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, np.inf],
            "other": [0.0, 2.0, 3.0, 4.0, 1.0, 5.0, np.nan],
            "y": np.arange(7.0),
            "fold": 1,
        }
    )
    sizes = tail_universes(frame.iloc[::-1], "member", (2, 10))
    assert set(sizes[2].ticker) == set("ABEF")
    assert len(sizes[10]) == 6 and sizes[10].ticker.is_unique
    assert set(tail_universes(frame, "other", (2,))[2].ticker) == set("ADEF")
    stats, daily = universe_diagnostics(frame, ["member", "other"], 60, 2, (2,))
    assert np.isnan(stats["2"]["member"]["mean_ic"]) and daily.universe.tolist() == [4, 4]
    print("SANITY: ticker ties are deterministic, nonfinite scores excluded, overlapping tails deduplicated, and each model uses its own ranking.")


def test_selected_universe_summary_uses_all_valid_days_and_preserves_rolling_support() -> None:
    dates = pd.bdate_range("2021-01-04", periods=4)
    frame = pd.DataFrame(
        {
            "date": np.repeat(dates, 4),
            "ticker": list("ABCD") * 4,
            "member": [0.0, 1.0, 2.0, 3.0] * 4,
            "y": [0.0, 1.0, 2.0, 3.0, 3.0, 2.0, 1.0, 0.0, 0.0, 1.0, 3.0, 2.0, np.nan, np.nan, np.nan, np.nan],
            "fold": [1] * 4 + [2] * 12,
        }
    )
    stats, daily = universe_diagnostics(frame, ["member"], 60, 2, (10,))
    values = np.array([1.0, -1.0, 0.8])
    result = stats["10"]["member"]
    assert result["n_days"] == 3 and result["mean_ic"] == pytest.approx(values.mean())
    assert result["ic_std"] == pytest.approx(values.std())
    assert result["ic_ir"] == pytest.approx(values.mean() / values.std() * np.sqrt(252 / 60))
    assert daily.date.tolist() == dates.tolist() and daily.rolling_ic.iloc[[0, 3]].isna().all()
    assert result["mean_ic"] != pytest.approx(np.mean([1.0, (-1.0 + 0.8) / 2]))
    print(
        "SANITY: summary weights valid dates equally rather than folds; missing daily IC keeps its calendar slot and invalidates the rolling window."
    )


def test_native_errors_use_raw_pairs_independent_of_normalized_scores() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.bdate_range("2022-01-03", periods=5),
            "ticker": ["A"] * 4 + ["B"],
            "raw_member": [2.0, 2.0, 2.0, np.inf, 1.0],
            "member": np.nan,
            "y": [0.0, 4.0, np.nan, 0.0, np.nan],
            "fold": 0,
        }
    )
    errors = native_error_diagnostics(frame, ["member"]).set_index("ticker")
    assert errors.at["A", "n"] == 2 and errors.at["A", "mae"] == 2.0
    assert errors.at["A", "bias"] == 0.0 and errors.at["A", "rmse"] == 2.0
    assert errors.at["B", "n"] == 0 and np.isnan(errors.at["B", "rmse"])
    print(
        "SANITY: constant native forecasts remain comparable despite undefined normalized scores; missing/nonfinite outcomes never become zero errors."
    )


def test_ticker_rmse_uses_only_cutoff_rows_and_reports_support() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.bdate_range("2021-12-29", periods=5),
            "ticker": ["A", "A", "A", "A", "B"],
            "raw_member": [100.0, 100.0, 0.0, 4.0, 0.0],
            "y": [0.0, 0.0, 0.0, 0.0, 0.0],
            "fold": [1, 2, 0, 0, 0],
        }
    )
    errors = native_error_diagnostics(frame, ["member"])
    cutoff = errors.loc[errors.population == "backtest"].set_index("ticker")
    assert cutoff.at["A", "rmse"] == pytest.approx(np.sqrt(8.0)) and cutoff.at["A", "n"] == 2
    assert cutoff.at["B", "rmse"] == 0.0 and cutoff.at["B", "n"] == 1
    assert errors.loc[errors.population == "validation", "rmse"].tolist() == [100.0]
    print("SANITY: cutoff RMSE excludes fold errors and averages squared residuals over each ticker's own comparable dates, with visible support.")


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
    stats, daily = temporal_diagnostics(predictions, ["member", "inverse"], 60, 2)
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


def test_nonfinite_member_scores_use_their_own_finite_pair_count() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2022-01-03"] * 6),
            "ticker": list("ABCDEF"),
            "y": [0.0, 1.0, 2.0, 3.0, 4.0, np.nan],
            "full": np.arange(6.0),
            "partial": [0.0, 1.0, 2.0, np.nan, np.inf, 5.0],
            "fold": 0,
        }
    )
    _, daily = temporal_diagnostics(frame, ["full", "partial"], 60, 2)
    assert daily.paired.tolist() == [5, 3] and daily.universe.tolist() == [6, 6]
    np.testing.assert_allclose(daily.ic, [1.0, 1.0])
    print("SANITY: finite prediction/label pairing is model-specific; one member's missing scores cannot alter another member's support or IC.")


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
    stats, daily = temporal_diagnostics(predictions, ["member"], 60, 252)
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
    stats, daily = temporal_diagnostics(predictions, ["member"], 60, 252)
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
        assert np.isnan(weakest["worst_rolling_ic_ir"])
    print(
        "SANITY: fewer than 252 valid daily ICs leave the weak period unsupported; a full constant-IC window remains identifiable with undefined IR."
    )


@pytest.mark.parametrize(("gap", "missing_labels"), [(0, True), (126, True), (252, True), (126, False)])
def test_rolling_windows_preserve_undefined_observed_sessions(gap: int, missing_labels: bool) -> None:
    dates = pd.bdate_range("2000-01-03", periods=253)
    predictions = pd.DataFrame(
        {
            "date": np.repeat(dates, 4),
            "ticker": list("ABCD") * len(dates),
            "y": [0.0, 1.0, 2.0, 3.0] * len(dates),
            "member": np.tile([0.0, 1.0, 2.0, 3.0, 3.0, 2.0, 1.0, 0.0], 127)[: 4 * len(dates)],
            "fold": 1,
        }
    )
    predictions.loc[predictions.date == dates[gap], "y" if missing_labels else "member"] = np.nan if missing_labels else 0.0
    stats, daily = temporal_diagnostics(predictions.iloc[::-1], ["member"], 60, 252)
    assert daily.date.tolist() == dates.tolist()
    assert daily.loc[gap, "paired"] == (0 if missing_labels else 4)
    expected_ic = (-1.0) ** np.arange(len(dates))
    expected_ic[gap] = np.nan
    np.testing.assert_allclose(daily.ic, expected_ic, equal_nan=True)
    supported = [end for end in (251, 252) if not end - 251 <= gap <= end]
    assert daily.loc[daily.rolling_ic.notna(), "date"].tolist() == [dates[end] for end in supported]
    assert daily.loc[daily.rolling_ir.notna(), "date"].tolist() == [dates[end] for end in supported]
    np.testing.assert_allclose(daily.rolling_ic.dropna(), 0.0)
    np.testing.assert_allclose(daily.rolling_ir.dropna(), 0.0)
    summary = stats["member"]
    assert summary["n_days"] == summary["years"]["2000"]["n_days"] == summary["folds"]["1"]["n_days"] == 252
    assert summary["worst_rolling_n_days"] == (252 if supported else 0)
    assert summary["worst_rolling_start"] == (dates[supported[0] - 251] if supported else None)
    assert summary["worst_rolling_end"] == (dates[supported[0]] if supported else None)
    assert summary["worst_rolling_mean_ic"] == (0.0 if supported else None)
    assert summary["worst_rolling_ic_ir"] == (0.0 if supported else None)
    print(f"SANITY: undefined session {gap} stays in all 253 observed slots; only {len(supported)} genuinely consecutive full windows are supported.")
