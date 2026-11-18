"""Independent known-truth source clocks and actual-market-session publication."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data_store.schema import Tables
from src.utils.config import read_config
from src.utils.institutional_vintages import filing_public_dates
from src.utils.publication import point_in_time


@pytest.mark.parametrize(
    ("lag", "expected"), [(0, "2026-07-02"), (1, "2026-07-06"), (2, "2026-07-06"), (3, "2026-07-06"), (12, "2026-07-14"), (21, "2026-07-23")]
)
def test_calendar_lag_then_actual_holiday_roll(lag: int, expected: str) -> None:
    config = read_config("configs")
    config.data.tables.sec_8k.pit.lag_days = lag
    sessions = pd.bdate_range("2026-07-01", "2026-07-31").difference(pd.DatetimeIndex(["2026-07-03"]))
    rows = pd.DataFrame({"filing_date": ["2026-07-02"], "value": [7.0]})
    ready = point_in_time(rows, config, Tables.sec_8k, sessions)
    assert ready.as_of.tolist() == [pd.Timestamp(expected)]
    assert ready.filing_date.tolist() == rows.filing_date.tolist() and ready.value.tolist() == [7.0]
    print(f"SANITY: explicit lag{lag} maps July2 to{expected}, respecting the July3 market holiday.")


def test_notes_all_period_types_use_twelve_days_and_keep_original_metadata() -> None:
    config = read_config("configs")
    sessions = pd.bdate_range("2026-04-01", "2026-04-30")
    rows = pd.DataFrame({"period": ["2026q1", "2026_03"], "filed": ["2026-04-20"] * 2, "available_at": ["2026-04-25"] * 2})
    ready = point_in_time(rows, config, Tables.notes_num, sessions)
    assert ready.as_of.tolist() == [pd.Timestamp("2026-04-13")] * 2
    pd.testing.assert_frame_equal(ready[rows.columns], rows)
    print("SANITY: monthly and quarterly period ends both add12CD; filed/observed evidence remains unchanged.")


def test_invalid_dates_and_unavailable_future_sessions_are_excluded(caplog: pytest.LogCaptureFixture) -> None:
    config = read_config("configs")
    sessions = pd.DatetimeIndex(["2026-11-06", "2026-11-09", "2026-11-10"])
    rows = pd.DataFrame({"filing_date": ["invalid", None, 1, "2099-01-01", "2026-11-06"], "key": [1, 2, 3, 4, 5]}, index=[0, 0, 1, 1, 2])
    original = rows.copy(deep=True)
    ready = point_in_time(rows, config, Tables.sec_8k, sessions)
    assert ready.key.tolist() == [5] and ready.as_of.tolist() == [pd.Timestamp("2026-11-09")]
    assert "3 invalid/missing" in caplog.text
    pd.testing.assert_frame_equal(rows, original)
    with pytest.raises(ValueError, match="missing"):
        point_in_time(rows.drop(columns="filing_date"), config, Tables.sec_8k, sessions)
    with pytest.raises(ValueError, match="already"):
        point_in_time(ready, config, Tables.sec_8k, sessions)
    print("SANITY: invalid rows are reported/excluded; future dates never roll backward; missing clocks/double lags refuse.")


def test_ftd_half_month_end_and_13f_exact_declared_clock() -> None:
    config = read_config("configs")
    sessions = pd.bdate_range("2026-04-01", "2026-05-01").difference(pd.DatetimeIndex(["2026-04-03"]))
    ftd = point_in_time(pd.DataFrame({"period": ["202603a", "202603b", "2026q9"]}), config, Tables.sec_fails_to_deliver_security, sessions)
    assert ftd.as_of.tolist() == list(pd.to_datetime(["2026-04-06", "2026-04-21"]))
    manifest = pd.DataFrame({"period": ["2025-12-31", "3006-09-30"], "filing_date": ["2026-04-01"] * 2, "accepted_at": ["2026-04-10"] * 2})
    actual = filing_public_dates(manifest, sessions, config)
    assert actual.iloc[0] == pd.Timestamp("2026-04-06") and pd.isna(actual.iloc[1])
    print("SANITY: FTD uses persisted half-month end+21;13F uses declared filing+2 and rejects future reporting quarters.")
