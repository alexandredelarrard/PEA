"""Independent date/interval identities for the shared production Window contract."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import cast

import pandas as pd
import pytest

from src.utils.window import Offset, TimeUnit, Window, WindowPolicy, WindowRun, resolve_window, shift_date


def test_normalized_inclusive_window_is_immutable() -> None:
    window = Window(pd.Timestamp("2026-11-09 12:30"), pd.Timestamp("2026-11-13 18:00"))
    values = pd.Series(pd.to_datetime(["2026-11-08", "2026-11-09", "2026-11-11", "2026-11-13", "2026-11-14", None]))
    assert window.start == pd.Timestamp("2026-11-09")
    assert window.end == pd.Timestamp("2026-11-13")
    assert window.mask(values).tolist() == [False, True, True, True, False, False]
    with pytest.raises(FrozenInstanceError):
        window.start = pd.Timestamp("2000-01-01")
    print("SANITY: inclusive normalized bounds retain both endpoints, exclude missing/outside dates and cannot mutate.")


@pytest.mark.parametrize(
    ("start", "end"),
    [("2026-11-13", "2026-11-09"), ("NaT", "2026-11-09"), ("2026-11-09T00:00:00Z", "2026-11-13T00:00:00Z")],
)
def test_invalid_or_zoned_date_bounds_fail_before_work(start: str, end: str) -> None:
    with pytest.raises(ValueError):
        Window(pd.Timestamp(start), pd.Timestamp(end))
    print("SANITY: reversed, missing or timezone-bearing interval bounds cannot reach a producer or writer.")


@pytest.mark.parametrize("days,expected", [(7, "2026-11-09"), (30, "2026-10-17"), (100, "2026-08-08"), (380, "2025-11-01")])
def test_calendar_overlap_boundaries_match_independent_oracle(days: int, expected: str) -> None:
    actual = shift_date(pd.Timestamp("2026-11-16"), Offset(days, "calendar_days"))
    assert actual == pd.Timestamp(expected)
    print(f"SANITY: {days} calendar days before2026-11-16 is exactly{expected}, including weekend boundaries.")


@pytest.mark.parametrize(
    ("anchor", "amount", "unit", "expected"),
    [("2026-08-17", 1, "quarters", "2026-05-17"), ("2024-05-31", 3, "months", "2024-02-29")],
)
def test_calendar_month_and_quarter_subtraction_clamps_real_dates(anchor: str, amount: int, unit: str, expected: str) -> None:
    actual = shift_date(pd.Timestamp(anchor), Offset(amount, cast(TimeUnit, unit)))
    assert actual == pd.Timestamp(expected)
    print(f"SANITY: {anchor} minus{amount} {unit} is{expected}; calendar units preserve actual month ends.")


def test_target_ninety_session_boundary_keeps_ninety_one_inclusive_dates() -> None:
    # The independent P0 oracle excludes July3 and LaborDay, and includes VeteransDay.
    calendar = pd.bdate_range("2026-07-01", "2026-11-16").difference(pd.to_datetime(["2026-07-03", "2026-09-07"]))
    start = shift_date(pd.Timestamp("2026-11-16"), Offset(90, "trading_sessions"), calendar)
    assert start == pd.Timestamp("2026-07-10")
    assert len(calendar[calendar >= start]) == 91
    assert pd.Timestamp("2026-11-11") in calendar
    assert shift_date(start, Offset(90, "trading_sessions"), calendar, direction=1) == pd.Timestamp("2026-11-16")
    print("SANITY: the90-session rewrite starts July10, spans91 dates inclusively, and keeps the actual market holiday distinction.")


def test_session_shift_requires_a_calendar_and_preserves_naturally_short_history() -> None:
    with pytest.raises(ValueError, match="calendar"):
        shift_date(pd.Timestamp("2026-11-16"), Offset(90, "trading_sessions"))
    calendar = pd.bdate_range("2026-11-09", "2026-11-16")
    assert shift_date(calendar[-1], Offset(90, "trading_sessions"), calendar) == calendar[0]
    print("SANITY: session arithmetic refuses an absent calendar and clamps naturally short retained history to its first date.")


def test_intersection_returns_only_authorized_inclusive_dates() -> None:
    requested = Window(pd.Timestamp("2026-11-09"), pd.Timestamp("2026-11-13"))
    intersecting = Window(pd.Timestamp("2026-11-12"), pd.Timestamp("2026-11-16"))
    disjoint = Window(pd.Timestamp("2026-11-16"), pd.Timestamp("2026-11-20"))
    assert requested.intersection(intersecting) == Window(pd.Timestamp("2026-11-12"), pd.Timestamp("2026-11-13"))
    assert requested.intersection(disjoint) is None
    print("SANITY: interval intersection cannot widen hard bounds, and disjoint work remains absent.")


@pytest.mark.parametrize("full", [False, True])
def test_resolver_separates_requested_reads_and_bounded_writes(full: bool) -> None:
    calendar = pd.bdate_range("2026-01-01", "2026-11-16")
    requested = Window(pd.Timestamp("2026-09-01"), pd.Timestamp("2026-11-13"))
    policy = WindowPolicy(requested, Offset(7, "calendar_days"), Offset(30, "calendar_days"))
    run = WindowRun("full" if full else "incremental", "independent finite scope")
    plan = resolve_window(policy, run, pd.Timestamp("2026-11-16"), calendar, (("ticker", ("AAPL",)),))
    assert plan.requested == requested and plan.write.end == requested.end
    assert plan.write.start == pd.Timestamp("2026-09-01" if full else "2026-11-06")
    assert plan.read.start == pd.Timestamp("2026-08-02" if full else "2026-10-07")
    assert plan.scope == (("ticker", ("AAPL",)),)
    print("SANITY: warm-up expands reads only; writes retain both finite hard bounds and explicit ticker scope.")


def test_initial_mode_is_explicit_and_cannot_widen_requested_dates() -> None:
    calendar = pd.bdate_range("2026-01-01", "2026-11-16")
    requested = Window(pd.Timestamp("2026-11-09"), pd.Timestamp("2026-11-13"))
    policy = WindowPolicy(requested, Offset(30, "calendar_days"), None)
    with pytest.raises(ValueError, match="explicit initial mode"):
        resolve_window(policy, WindowRun("incremental", "missing frontier"), None, calendar, (("ticker", ("AAPL",)),))
    plan = resolve_window(policy, WindowRun("initial", "first selected output"), None, calendar, (("ticker", ("AAPL",)),))
    assert plan.write == requested and plan.read == Window(calendar[0], requested.end)
    print("SANITY: missing state requires initial mode; full-history reads never widen finite write bounds.")
