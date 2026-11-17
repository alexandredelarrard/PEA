"""Known-truth publication clocks, session rolls and explicit source buffers."""

from __future__ import annotations

import pandas as pd

from src.data_store.schema import Tables
from src.utils.publication import publication_dates
from tests.data_extract.fake_context import extract_config


def test_filing_publication_uses_calendar_buffer_and_the_market_calendar() -> None:
    config = extract_config()
    # Friday, then Tuesday: Monday is deliberately absent (a closed market).
    sessions = pd.DatetimeIndex(["2026-11-06", "2026-11-10", "2026-11-11", "2026-11-12"])
    rows = pd.DataFrame({"filing_date": pd.to_datetime(["2026-11-06", "2026-11-10"])})
    actual = publication_dates(config, Tables.sec_8k, rows, sessions)
    assert actual.tolist() == [pd.Timestamp("2026-11-10"), pd.Timestamp("2026-11-11")]
    config.data.tables.sec_8k.pit.lag.value = 2
    assert publication_dates(config, Tables.sec_8k, rows, sessions).iloc[1] == pd.Timestamp("2026-11-12")
    print("SANITY: configured calendar-day buffers apply before actual-session roll; no weekday or numeric fallback.")


def test_13f_uses_public_filing_and_later_accepted_date_without_quarter_deadline() -> None:
    config = extract_config()
    sessions = pd.DatetimeIndex(["2026-11-06", "2026-11-09", "2026-11-10", "2026-11-11", "2026-11-12"])
    rows = pd.DataFrame(
        {
            "period": pd.to_datetime(["2026-09-30"] * 3),
            "filing_date": pd.to_datetime(["2026-11-06"] * 3),
            "accepted_at": [None, "2026-11-10T01:00:00Z", "2026-11-10T05:30:00Z"],
        }
    )
    actual = publication_dates(config, Tables.sec13f_hr, rows, sessions)
    assert actual.tolist() == [pd.Timestamp("2026-11-09"), pd.Timestamp("2026-11-11"), pd.Timestamp("2026-11-12")]
    print("SANITY: filing+2CD has no quarter+45 floor; later acceptance uses its actual New York public date.")


def test_call_child_inherits_the_source_buffer_once() -> None:
    config = extract_config()
    sessions = pd.bdate_range("2026-11-02", "2026-11-13")
    rows = pd.DataFrame({"as_of": pd.to_datetime(["2026-11-06"])})
    parent = publication_dates(config, Tables.earnings_call_sections, rows, sessions)
    child = publication_dates(config, Tables.earnings_call_sentiment, rows, sessions)
    assert parent.equals(child) and parent.iloc[0] == pd.Timestamp("2026-11-09")
    print("SANITY: inherited call timing remains call+3CD, without a second processing-time buffer.")


def test_archive_release_estimates_preserve_later_observed_availability() -> None:
    config = extract_config()
    sessions = pd.bdate_range("2026-04-01", "2026-11-16").difference(pd.DatetimeIndex(["2026-05-25", "2026-06-19", "2026-07-03", "2026-09-07"]))
    notes = pd.DataFrame(
        {
            "period": ["2026q1", "2026_09", "2026_09"],
            "filed": pd.to_datetime(["2026-04-01", "2026-10-01", "2026-10-01"]),
            "available_at": pd.to_datetime([None, None, "2026-10-15"]),
        }
    )
    dates = publication_dates(config, Tables.notes_num, notes, sessions)
    assert dates.tolist() == list(pd.to_datetime(["2026-04-21", "2026-10-12", "2026-10-15"]))
    ftd = pd.DataFrame({"period": ["202610a", "202610b"]})
    assert publication_dates(config, Tables.sec_fails_to_deliver_security, ftd, sessions).tolist() == [pd.Timestamp("2026-11-05"), pd.NaT]
    print("SANITY: notes use quarter+21/monthly day12 and preserve later availability; FTD uses half-month end+21, never an early last session.")
