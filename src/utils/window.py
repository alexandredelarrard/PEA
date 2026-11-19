"""Immutable inclusive date intervals and explicit calendar/session arithmetic."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

from src.utils.string import run_date

type TimeUnit = Literal["calendar_days", "trading_sessions", "months", "quarters"]


def _date(value: pd.Timestamp) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tz is not None:
        raise ValueError("Window dates must be present, timezone-naive market dates")
    return stamp.normalize()


@dataclass(frozen=True, slots=True, order=True)
class Window:
    """Normalized inclusive dates; computation context and write scope use separate instances."""

    start: pd.Timestamp
    end: pd.Timestamp

    def __post_init__(self) -> None:
        start, end = _date(self.start), _date(self.end)
        if start > end:
            raise ValueError(f"Window start {start.date()} exceeds end {end.date()}")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)

    def __iter__(self) -> Iterator[pd.Timestamp]:
        yield self.start
        yield self.end

    def mask(self, values: pd.Series) -> pd.Series:
        dates = pd.to_datetime(values, errors="raise")
        if dates.dt.tz is not None:
            raise ValueError("Window rows must use timezone-naive market dates")
        return dates.dt.normalize().between(self.start, self.end, inclusive="both")

    def intersection(self, other: Window) -> Window | None:
        start, end = max(self.start, other.start), min(self.end, other.end)
        return Window(start, end) if start <= end else None


@dataclass(frozen=True, slots=True)
class Offset:
    """A nonnegative configured amount whose unit is never inferred from its magnitude."""

    value: int
    unit: TimeUnit

    def __post_init__(self) -> None:
        if isinstance(self.value, bool) or not isinstance(self.value, int) or self.value < 0:
            raise ValueError("Window offsets require a nonnegative integer")
        if self.unit not in {"calendar_days", "trading_sessions", "months", "quarters"}:
            raise ValueError(f"Unsupported Window offset unit: {self.unit}")


@dataclass(frozen=True, slots=True)
class WindowPolicy:
    """Hard requested bounds, rewrite overlap and additional computation context."""

    requested: Window
    overlap: Offset | None
    warmup: Offset | None
    lookahead: Offset | None = None
    lookback: Offset | None = None
    inherited: bool = False

    def __post_init__(self) -> None:
        if sum((self.overlap is not None, self.lookback is not None, self.inherited)) > 1:
            raise ValueError("A Window policy requires one reprocessing anchor")


@dataclass(frozen=True, slots=True)
class WindowRun:
    mode: Literal["incremental", "initial", "full", "recovery"]
    reason: str
    as_of: pd.Timestamp
    instant: pd.Timestamp = field(init=False)

    def __post_init__(self) -> None:
        if self.mode not in {"incremental", "initial", "full", "recovery"} or not self.reason.strip():
            raise ValueError("Window runs require an explicit supported mode and reason")
        instant = pd.Timestamp(self.as_of)
        object.__setattr__(self, "instant", instant)
        object.__setattr__(self, "as_of", run_date(instant))


@dataclass(frozen=True, slots=True)
class WindowPlan:
    requested: Window
    read: Window
    write: Window
    run: WindowRun
    frontier: pd.Timestamp | None
    scope: tuple[tuple[str, tuple[str, ...]], ...]

    def __post_init__(self) -> None:
        if self.write.intersection(self.requested) != self.write or self.read.intersection(self.write) != self.write:
            raise ValueError("Write Window must stay inside requested bounds and computation context")
        if not self.scope or len(dict(self.scope)) != len(self.scope) or any(not keys for _, keys in self.scope):
            raise ValueError("Window plans require explicit nonempty unique key scopes")

    @property
    def is_full(self) -> bool:
        return self.run.mode in {"initial", "full", "recovery"}


def resolve_window(
    policy: WindowPolicy,
    run: WindowRun,
    frontier: pd.Timestamp | None,
    calendar: pd.DatetimeIndex,
    scope: tuple[tuple[str, tuple[str, ...]], ...],
    *,
    inherited_start: pd.Timestamp | None = None,
) -> WindowPlan:
    """Resolve overlap once; warm-up expands reads and never authorizes extra writes."""
    requested = Window(policy.requested.start, min(policy.requested.end, run.as_of))
    if policy.inherited != (inherited_start is not None):
        raise ValueError("Inherited Window policies require actual selected component boundaries")
    if run.mode == "incremental" and policy.overlap is not None and frontier is None:
        raise ValueError("Incremental Window runs require a scoped frontier; use explicit initial mode")
    start = requested.start
    if run.mode == "incremental" and policy.overlap is not None:
        assert frontier is not None
        start = max(start, shift_date(min(_date(frontier), requested.end), policy.overlap, calendar))
    if run.mode == "incremental" and policy.lookback is not None:
        start = max(start, shift_date(run.as_of, policy.lookback, calendar))
    if run.mode == "incremental" and inherited_start is not None:
        catchup = requested.start if frontier is None else min(_date(frontier), requested.end)
        start = max(start, min(_date(inherited_start), catchup))
    write = Window(start, requested.end)
    read_start = calendar[0] if policy.warmup is None and len(calendar) else requested.start
    if policy.warmup is not None:
        read_start = shift_date(write.start, policy.warmup, calendar)
    read_end = requested.end
    if policy.lookahead is not None:
        read_end = min(run.as_of, max(read_end, shift_date(requested.end, policy.lookahead, calendar, direction=1)))
    return WindowPlan(requested, Window(min(read_start, write.start), read_end), write, run, frontier, scope)


def shift_date(anchor: pd.Timestamp, offset: Offset, calendar: pd.DatetimeIndex | None = None, *, direction: Literal[-1, 1] = -1) -> pd.Timestamp:
    """Move by explicit units, clamping session offsets to genuinely retained history."""
    stamp = _date(anchor)
    if direction not in {-1, 1}:
        raise ValueError("Window direction must be -1 or 1")
    amount = direction * offset.value
    if offset.unit == "calendar_days":
        return stamp + pd.Timedelta(days=amount)
    if offset.unit in {"months", "quarters"}:
        return stamp + pd.DateOffset(months=amount * (3 if offset.unit == "quarters" else 1))

    # Session moves use actual market dates, including holidays, with no weekday fallback.
    if calendar is None or calendar.empty:
        raise ValueError("Trading-session offsets require a retained market calendar")
    if calendar.tz is not None or calendar.hasnans or calendar.has_duplicates or not calendar.is_monotonic_increasing:
        raise ValueError("The market calendar must contain unique ordered timezone-naive dates")
    if not offset.value:
        return stamp
    position = int(calendar.searchsorted(stamp, side="right" if direction == 1 else "left")) - int(direction == 1)
    return _date(calendar[max(0, min(len(calendar) - 1, position + amount))])
