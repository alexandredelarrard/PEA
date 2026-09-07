"""
test_sessions.py  (tests/data_extract/common/test_sessions.py)
------------------------------------------------------------------
`last_completed_session` is the clamp that stops a fetcher writing a bar for a session that
is still trading. The live table had one: its final bar carried 0.535x the ticker's own
trailing median volume and was indistinguishable downstream from a settled close.

The property under test is CONSERVATISM -- the answer must never be a session whose close
has not printed. Being a day early is harmless (the rolling re-pull floor revisits it); being
a day late is the defect.
"""
import pandas as pd
import pytest

from src.data_extract.utils.common.sessions import last_completed_session

# 2026-09-03 is a Thursday, 09-04 a Friday, 09-05 a Saturday, 09-06 a Sunday.
CASES = [
    ("2026-09-04 10:00", "2026-09-03", "Friday pre-close -> Thursday"),
    ("2026-09-04 15:59", "2026-09-03", "Friday one minute pre-close -> Thursday"),
    ("2026-09-04 16:00", "2026-09-04", "Friday at the close -> Friday"),
    ("2026-09-04 21:00", "2026-09-04", "Friday post-close -> Friday"),
    ("2026-09-05 12:00", "2026-09-04", "Saturday -> Friday"),
    ("2026-09-06 21:25", "2026-09-04", "Sunday -> Friday"),
    ("2026-09-07 09:00", "2026-09-04", "Monday pre-close -> Friday"),
]


@pytest.mark.parametrize("now,expected,label", CASES)
def test_clock_positions(now, expected, label):
    got = last_completed_session(pd.Timestamp(now, tz="America/New_York"))
    print(f"  {label:42s} {now} ET -> {got.date()}")
    assert got == pd.Timestamp(expected)
    assert got.tzinfo is None, "fetchers compare against tz-naive DB dates"
    assert got == got.normalize(), "must be a date, not a datetime"


def test_naive_now_is_read_as_exchange_local():
    """A naive `now` is exchange-local: 'before the close' is meaningless in any other frame."""
    assert last_completed_session(pd.Timestamp("2026-09-04 10:00")) == pd.Timestamp("2026-09-03")


def test_utc_now_is_converted_not_truncated():
    """14:00 UTC on Friday is 10:00 ET -- still pre-close, so Thursday. A naive truncation of
    the UTC wall clock would wrongly answer Friday."""
    got = last_completed_session(pd.Timestamp("2026-09-04 14:00", tz="UTC"))
    print(f"  2026-09-04 14:00 UTC (= 10:00 ET) -> {got.date()}")
    assert got == pd.Timestamp("2026-09-03")


def test_utc_midnight_saturday_is_friday_et():
    """00:30 UTC Saturday is 20:30 ET FRIDAY, after the close -> Friday, not Thursday."""
    got = last_completed_session(pd.Timestamp("2026-09-05 00:30", tz="UTC"))
    print(f"  2026-09-05 00:30 UTC (= Fri 20:30 ET) -> {got.date()}")
    assert got == pd.Timestamp("2026-09-04")


def test_never_returns_a_weekend():
    """Swept across a fortnight of hourly clock positions."""
    for ts in pd.date_range("2026-08-24", "2026-09-07", freq="h", tz="America/New_York"):
        day = last_completed_session(ts)
        assert day.weekday() < 5, f"{ts} -> {day} is a weekend"
        assert day <= ts.tz_localize(None).normalize(), f"{ts} -> {day} is in the future"


def test_live_clock_is_a_settled_past_session():
    """No frozen `now`: the real call the fetchers make must still satisfy the invariant."""
    got = last_completed_session()
    print(f"  live clock -> {got.date()}")
    assert got.weekday() < 5
    assert got <= pd.Timestamp.now(tz="America/New_York").tz_localize(None).normalize()
