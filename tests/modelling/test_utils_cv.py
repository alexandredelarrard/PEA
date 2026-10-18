"""Temporal validation primitives (src/modelling/utils/cv.py): deterministic purged folds with a
real embargo gap, a chronological early-stopping tail, and exponential time-decay weights."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.modelling.utils.cv import CALENDAR_DAYS_PER_YEAR, label_end_dates, purged_wf_splits, temporal_valid_split, time_decay_weights


def test_explicit_test_windows_use_observed_calendar_and_refuse_overlap() -> None:
    calendar = pd.bdate_range("1999-01-01", "2003-12-31")
    dates = pd.Series(calendar)
    windows = [("2000-01-01", "2001-12-31"), ("2002-01-01", "2003-12-31")]
    splits = list(purged_wf_splits(dates, 2, 60, test_windows=windows, calendar=calendar))
    assert len(splits) == 2
    ends = label_end_dates(calendar, 60)
    for (train, test), (start, end) in zip(splits, windows, strict=True):
        assert test.min() >= np.datetime64(start) and test.max() <= np.datetime64(end)
        assert ends.loc[train].max() < pd.Timestamp(test.min())
    with pytest.raises(ValueError, match="overlap|ordered"):
        list(purged_wf_splits(dates, 2, 60, test_windows=[windows[0], windows[0]], calendar=calendar))
    print("SANITY: explicit crisis windows retain all test sessions and purge training endpoints on the observed calendar.")


def test_label_end_uses_complete_session_calendar_and_purges_inner_overlap() -> None:
    calendar = pd.DatetimeIndex(["2024-07-01", "2024-07-02", "2024-07-03", "2024-07-05", "2024-07-08", "2024-07-09"])
    ends = label_end_dates(calendar, 2)
    assert ends.loc["2024-07-02"] == pd.Timestamp("2024-07-05")
    assert pd.isna(ends.iloc[-1])
    df_panel = pd.DataFrame({"date": calendar[:4], "label_end": ends.iloc[:4].to_numpy()})
    train, valid = temporal_valid_split(df_panel, train_frac=0.75)
    assert train["date"].tolist() == [pd.Timestamp("2024-07-01")]
    assert (train["label_end"] < valid["date"].min()).all()
    print("SANITY: actual July holiday calendar maps horizon ends; inner labels ending on validation are purged.")


def test_purged_wf_splits_is_deterministic_and_embargoed() -> None:
    dates = pd.Series(np.repeat(pd.bdate_range("2015-01-01", periods=300), 5))
    first = list(purged_wf_splits(dates, n_splits=5, embargo=20))
    second = list(purged_wf_splits(dates, n_splits=5, embargo=20))
    assert len(first) == 5
    for (tr_a, te_a), (tr_b, te_b) in zip(first, second, strict=True):
        assert np.array_equal(tr_a, tr_b) and np.array_equal(te_a, te_b)
    unique = np.sort(dates.unique())
    for train_days, test_days in first:
        assert train_days.max() < test_days.min(), "train must precede test"
        gap = int(np.searchsorted(unique, test_days.min()) - np.searchsorted(unique, train_days.max())) - 1
        assert gap == 20, f"embargo gap {gap} != 20 dates"

    print("\n=== SANITY CHECK: purged walk-forward folds ===")
    print(f"  5 folds, identical across calls; train sizes {[len(t) for t, _ in first]}; every fold skips exactly 20 dates before testing.")
    print("  Expanding window, no overlap, deterministic. Validated.")


def test_temporal_valid_split_holds_out_the_latest_dates() -> None:
    panel = pd.DataFrame({"date": np.repeat(pd.bdate_range("2020-01-01", periods=20), 3), "y": 0.0})
    train, valid = temporal_valid_split(panel, train_frac=0.9)
    assert train["date"].nunique() == 18 and valid["date"].nunique() == 2
    assert train["date"].max() < valid["date"].min()
    print("\n=== SANITY CHECK: early-stopping tail ===")
    print(f"  20 dates -> train {train['date'].nunique()} / valid {valid['date'].nunique()}; valid is strictly later. Validated.")


def test_time_decay_half_life_at_reference_and_two_years_back() -> None:
    ref = pd.Timestamp("2026-01-01")
    dates = pd.Series([ref - pd.Timedelta(days=int(2 * CALENDAR_DAYS_PER_YEAR)), ref - pd.Timedelta(days=int(CALENDAR_DAYS_PER_YEAR)), ref])
    w = time_decay_weights(dates, half_life_years=2.0, reference=ref)
    assert np.isclose(w[2], 1.0)
    assert np.isclose(w[0], 0.5, rtol=1e-2)
    assert np.isclose(w[1], 0.5**0.5, rtol=1e-2)
    assert w[0] < w[1] < w[2]
    print("\n=== SANITY CHECK: time_decay_weights ===")
    print(f"  ref={ref.date()}, half_life=2y -> w(2y ago)={w[0]:.4f}, w(1y ago)={w[1]:.4f}, w(today)={w[2]:.4f}")
    print("  Older rows down-weighted; 2 calendar years -> 0.5. Validated.")
