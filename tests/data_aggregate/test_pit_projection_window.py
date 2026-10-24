"""Known-truth projection seeds, publication times, null barriers and bounded pivots."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd

from src.data_aggregate.utils.common.pit import fundamentals_to_daily
from src.data_aggregate.utils.sec_thirteen.value_basis import nullable_value_to_daily
from src.utils.price_adjustments import ShareBasis


def test_window_projection_keeps_original_age_and_bounds_dense_pivots() -> None:
    calendar = pd.DatetimeIndex(["2025-01-06", "2025-01-08", "2025-01-10"])
    old_dates = pd.date_range("1990-01-01", periods=1000)
    history = pd.DataFrame({"ticker": "FRESH", "as_of": old_dates, "value": 1.0})
    history = pd.concat(
        [
            history,
            pd.DataFrame(
                {
                    "ticker": ["FRESH", "FRESH", "FRESH", "OLD", "FUTURE", "NULL", "ZERO"],
                    "as_of": ["2025-01-05", "2025-01-07 12:00", "2025-01-08 12:00", "2024-12-01", "2025-02-01", "2025-01-05", "2025-01-05"],
                    "value": [10.0, 20.0, 30.0, 99.0, 77.0, np.nan, 0.0],
                }
            ),
        ],
        ignore_index=True,
    )
    history["as_of"] = pd.to_datetime(history["as_of"], format="mixed")

    with patch.object(pd.DataFrame, "pivot", autospec=True, side_effect=pd.DataFrame.pivot) as pivot:
        daily = fundamentals_to_daily(history, "value", calendar, max_age_days=2)

    expected = pd.DataFrame({"FRESH": [10.0, 20.0, 30.0], "FUTURE": np.nan, "OLD": np.nan, "ZERO": [0.0, np.nan, np.nan]}, index=calendar)
    expected.columns.name = "ticker"
    pd.testing.assert_frame_equal(daily, expected, check_exact=True)
    assert pivot.call_count == 2
    assert all(call.args[0]["as_of"].nunique() <= len(calendar) + 1 for call in pivot.call_args_list)
    assert all(call.args[0]["as_of"].max() <= calendar[-1] for call in pivot.call_args_list)
    print("Sanity: bounded pivots preserved publication times, original seed age, zero values and future-only ticker columns.")


def test_nullable_projection_keeps_pre_window_and_mid_window_unknown_barriers() -> None:
    calendar = pd.DatetimeIndex(["2025-01-06", "2025-01-08", "2025-01-10"])
    history = pd.DataFrame({"ticker": "A", "as_of": pd.date_range("1990-01-01", periods=1000), "value": 100.0})
    history = pd.concat(
        [
            history,
            pd.DataFrame(
                {
                    "ticker": ["A", "A", "B", "B", "B", "FUTURE", "NULL"],
                    "as_of": ["2025-01-04", "2025-01-09", "2025-01-04", "2025-01-07", "2025-01-09", "2025-02-01", "2025-01-04"],
                    "value": [np.nan, 120.0, 7.0, np.nan, 8.0, 77.0, np.nan],
                }
            ),
        ],
        ignore_index=True,
    )
    history["as_of"] = pd.to_datetime(history["as_of"], format="mixed")

    with patch.object(pd.DataFrame, "pivot", autospec=True, side_effect=pd.DataFrame.pivot) as pivot:
        daily = nullable_value_to_daily(history, "value", calendar)

    expected = pd.DataFrame({"A": [np.nan, np.nan, 120.0], "B": [7.0, np.nan, 8.0], "FUTURE": np.nan}, index=calendar)
    expected.columns.name = "ticker"
    pd.testing.assert_frame_equal(daily, expected, check_exact=True)
    assert all(call.args[0]["as_of"].nunique() <= len(calendar) + 1 for call in pivot.call_args_list)
    print("Sanity: a pre-window unknown replaced the old known seed; later unknowns stayed unknown until a new known observation.")


def test_window_projection_preserves_numeric_dtype_and_empty_calendar_columns() -> None:
    calendar = pd.DatetimeIndex(["2025-01-06"])
    history = pd.DataFrame(
        {"ticker": ["A", "B", "A", "B"], "as_of": pd.to_datetime(["1990-01-01", "1991-01-01", "2025-01-06", "2025-01-06"]), "value": [1, 2, 3, 4]}
    )
    daily = fundamentals_to_daily(history, "value", calendar)
    expected = pd.DataFrame({"A": [3.0], "B": [4.0]}, index=calendar)
    expected.columns.name = "ticker"
    pd.testing.assert_frame_equal(daily, expected, check_exact=True)

    future = pd.DataFrame({"ticker": ["FUTURE"], "as_of": pd.to_datetime(["2025-02-01"]), "value": pd.Series([7.0], dtype="float32")})
    future_daily = fundamentals_to_daily(future, "value", calendar)
    expected_future = pd.DataFrame({"FUTURE": pd.Series(np.nan, index=calendar, dtype="float32")}, index=calendar)
    expected_future.columns.name = "ticker"
    pd.testing.assert_frame_equal(future_daily, expected_future, check_exact=True)
    empty = fundamentals_to_daily(future, "value", pd.DatetimeIndex([]))
    pd.testing.assert_frame_equal(empty, expected_future.reindex(pd.DatetimeIndex([])), check_exact=True)
    print("Sanity: historical sparse integer pivots kept float promotion and future-only float32 columns survived empty windows.")


def test_window_seed_keeps_production_date_for_split_basis_and_expiry() -> None:
    calendar = pd.DatetimeIndex(["2025-01-06", "2025-01-08", "2025-01-10"])
    history = pd.DataFrame({"ticker": ["AAPL"], "as_of": pd.to_datetime(["2025-01-03"]), "sharesOutstandingPit": [100.0]})
    splits = pd.DataFrame({"ticker": ["AAPL"], "date": pd.to_datetime(["2025-01-07"]), "value": [2.0]})
    basis = ShareBasis(actual=splits, vendor=splits.iloc[:0], price=splits.iloc[:0], policies={"AAPL": {}})

    daily = fundamentals_to_daily(history, "sharesOutstandingPit", calendar, max_age_days=5, share_basis=basis)

    expected = pd.DataFrame({"AAPL": [100.0, 200.0, np.nan]}, index=calendar)
    expected.columns.name = "ticker"
    pd.testing.assert_frame_equal(daily, expected, check_exact=True)
    print("Sanity: the pre-window share seed retained its original split basis and expired from its actual production date.")


def test_projection_preserves_produced_at_payload_and_its_actual_age() -> None:
    calendar = pd.DatetimeIndex(["2025-01-06", "2025-01-07"])
    history = pd.DataFrame(
        {"ticker": ["A", "A"], "as_of": pd.to_datetime(["2025-01-04", "2025-01-05"]), "_produced_at": [739249.0, 739250.0]}, index=[0, 0]
    )
    expected = pd.DataFrame({"A": [739250.0, 739250.0]}, index=calendar)
    expected.columns.name = "ticker"

    pd.testing.assert_frame_equal(fundamentals_to_daily(history, "_produced_at", calendar), expected, check_exact=True)
    expected.iloc[-1, 0] = np.nan
    pd.testing.assert_frame_equal(fundamentals_to_daily(history, "_produced_at", calendar, max_age_days=1), expected, check_exact=True)
    print("Sanity: numeric governance source ages retained the latest weekend publication and actual expiry with duplicate input indices.")
