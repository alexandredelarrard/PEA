"""Work-list planners: what each nightly run fetches, derived from the table's own rows, its
`schema.Resume` contract and the run date. Nothing else is read or written.

`series_windows` (dated per-key series): per key, its own last date minus the overlap through
`until`; the full history for an absent table, `full`, or a new key whose stored history does not
yet reach the floor, or its own first date in an optional `listing` table when later (a new key
whose full pull is stored resumes like any other); the table-wide frontier minus
the overlap for a rowless key; plus one window per run of calendar sessions missing inside the
key's stored span. Every window is clamped to the key's planner bounds (`planner_bounds`: `[first_price_date, ended_on]`
with the U-65 grace after the end; a ticker-keyed series has no row before its listing's first price); an ended key with
stored rows gets none once the run date is past its grace end plus the table's overlap (`ended_keys`).

`archive_worklist` (bulk archives): only published units missing from a required table on normal runs.
Completed units replay only under the Context's explicit full/recovery mode, inside configured unit bounds.
A ticker subset holds missing global units for the full-universe run, so partial work cannot hide remaining keys.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Collection, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import cast

import pandas as pd

from src.context import Context
from src.data_extract.utils.common.incremental import stored_values
from src.data_store.schema import Table, Tables
from src.utils.data_policy import window_policy
from src.utils.publication import period_end
from src.utils.tickers_universe.universe import Bounds
from src.utils.window import Offset, Window, WindowPlan, WindowRun, resolve_window, shift_date

logger = logging.getLogger(__name__)

#: Key classes reported by `resume-plan`.
KEY_NEW = "new"
KEY_ROWLESS = "rowless"
KEY_ESTABLISHED = "established"
#: Done-set scopes: accessions stored under the key itself, or anywhere in the table.
DONE_PER_KEY = "key"
DONE_TABLE = "table"


def recently_changed(context: Context, stamps: Mapping[str, pd.Timestamp | None]) -> list[str]:
    """Changed lineage keys inside the configured recheck offset, on the shared run anchor."""
    section = context.config.data.planning.lineage_recheck
    offset = Offset(section.value, section.unit)
    calendar = trading_calendar(context) if offset.unit == "trading_sessions" else None
    since = shift_date(context.run.as_of, offset, calendar)
    return sorted(str(key) for key, stamp in stamps.items() if stamp is not None and not pd.isna(stamp) and pd.Timestamp(stamp) >= since)


def planner_bounds(context: Context) -> Bounds:
    """Listing bounds plus configured end grace; unavailable future sessions never become weekdays."""
    section = context.config.data.planning.ended_grace
    offset = Offset(section.value, section.unit)
    universe = context.universe
    cached = getattr(context, "_planner_bounds", None)
    if cached is not None and cached[0] is universe and cached[1] == offset and cached[2] is context.run:
        return cached[3]
    bounds = universe.bounds
    if any(end is not None for _, end in bounds.values()):
        calendar = trading_calendar(context) if offset.unit == "trading_sessions" else pd.DatetimeIndex([])
        bounds = {key: (first, None if end is None else _grace_end(calendar, end, offset)) for key, (first, end) in bounds.items()}
    context._planner_bounds = (universe, offset, context.run, bounds)
    return bounds


def _grace_end(calendar: pd.DatetimeIndex, end: pd.Timestamp, offset: Offset) -> pd.Timestamp | None:
    """Hold an end open within the requested Window until its full session grace is observable."""
    if not offset.value:
        return end
    if offset.unit == "trading_sessions" and len(calendar[calendar > end]) < offset.value:
        return None
    return shift_date(end, offset, calendar, direction=1)


def bounded_span(bounds: Bounds, key: str, since: pd.Timestamp, until: pd.Timestamp) -> Window | None:
    """`(since, until)` clamped to the key's `[first_price_date, ended_on]`; None when nothing is left. A key
    without bounds (a secondary class, a key outside the universe) keeps its span."""
    first, ended = bounds.get(key, (None, None))
    start = max(since, first) if first is not None else since
    end = min(until, ended) if ended is not None else until
    return Window(start, end) if start <= end else None


def source_plan(context: Context, table: Table, frontier: pd.Timestamp | None, scope: tuple[tuple[str, tuple[str, ...]], ...]) -> WindowPlan | None:
    """Resolve a source-family plan from config and observed resume state."""
    policy = window_policy(context.config, table, context.run.as_of)
    if policy is None:
        return None
    run = context.run
    if frontier is None and run.mode == "incremental":
        run = WindowRun("initial", "missing configured source history", run.instant)
    offsets = (policy.overlap, policy.warmup, policy.lookback, policy.lookahead)
    calendar = (
        trading_calendar(context) if any(offset is not None and offset.unit == "trading_sessions" for offset in offsets) else pd.DatetimeIndex([])
    )
    if run.mode == "incremental" and policy.lookback is not None and shift_date(run.as_of, policy.lookback, calendar) > policy.requested.end:
        return None
    return resolve_window(policy, run, frontier, calendar, scope)


def stored_accessions(context: Context, table: Table, keys: Sequence[str], scope: str, where: dict[str, object] | None = None) -> dict[str, set[str]]:
    """Accessions already stored on the rows matching `where`, markers included: per key (`DONE_PER_KEY`,
    one two-column read) or one table-wide set shared by every key (`DONE_TABLE`, one `SELECT DISTINCT`)."""
    if table == Tables.sec13f_hr and not {"accession_number", "parse_status"} <= set(context.store.columns(table)):
        return {key: set() for key in keys}
    key_col = table.resume.key if table.resume is not None and table.resume.key else "ticker"
    if scope == DONE_TABLE:
        everywhere = set(stored_values(context, table, "accession_number", where=where))
        return {key: everywhere for key in keys}
    df = context.store.load(table, columns=[key_col, "accession_number"], where={**(where or {}), key_col: list(keys)}, markers=True, optional=True)
    done: dict[str, set[str]] = {key: set() for key in keys}
    if df is not None:
        for key, group in df.groupby(key_col, sort=False):
            done.setdefault(str(key), set()).update(group["accession_number"].astype(str))
    return done


@dataclass
class SeriesWork:
    """A series work list: per key, disjoint `(since, until)` windows; plus each key's class and stored last date."""

    windows: dict[str, list[Window]]
    key_class: dict[str, str]
    last: dict[str, pd.Timestamp] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)

    def groups(self) -> list[tuple[Window, list[str]]]:
        """Keys sharing an identical window, one group per window, oldest `since` first."""
        by_window: dict[Window, list[str]] = {}
        for key, spans in self.windows.items():
            for span in spans:
                by_window.setdefault(span, []).append(key)
        return [(span, sorted(keys)) for span, keys in sorted(by_window.items())]

    def merge(self, other: SeriesWork) -> SeriesWork:
        """The union of two work lists (each key's windows unioned); key classes and `last` stay this work's."""
        keys = set(self.windows) | set(other.windows)
        windows = {key: _union_spans(self.windows.get(key, []) + other.windows.get(key, [])) for key in keys}
        return SeriesWork(windows=windows, key_class=other.key_class | self.key_class, last=self.last, timings=self.timings | other.timings)


def _union_spans(spans: list[Window]) -> list[Window]:
    """Overlapping or touching `(since, until)` spans merged, sorted."""
    merged: list[Window] = []
    for since, until in sorted(spans):
        if merged and since <= merged[-1].end + pd.Timedelta(days=1):
            merged[-1] = Window(merged[-1].start, max(merged[-1].end, until))
        else:
            merged.append(Window(since, until))
    return merged


def trading_calendar(context: Context) -> pd.DatetimeIndex:
    """The distinct dates of `prices`, sorted; empty when the table is absent. Dates only: a secondary class
    in `prices` is fetched inside its company's own windows, so it adds no session."""
    return pd.DatetimeIndex(pd.to_datetime(context.store.distinct(Tables.prices, "date"))).normalize().unique().sort_values()


def _hole_spans(calendar: pd.DatetimeIndex, stored: pd.Series, first: pd.Timestamp, edge: pd.Timestamp) -> list[Window]:
    """Runs of calendar sessions in `[first, edge)` with no stored row, as `(first missing, last missing)` spans."""
    span = calendar[(calendar >= first) & (calendar < edge)]
    missing = ~span.isin(pd.DatetimeIndex(pd.to_datetime(stored)).normalize())
    if not missing.any():
        return []
    positions = pd.Series(range(len(span)))[missing]
    runs = (positions.diff() != 1).cumsum()
    return [Window(span[int(run.iloc[0])], span[int(run.iloc[-1])]) for _, run in positions.groupby(runs)]


def _has_holes(calendar: pd.DatetimeIndex | None, first: pd.Timestamp, last: pd.Timestamp, n: int) -> bool:
    """True when the daily `calendar` holds more sessions in `[first, last]` than the key's `n` stored rows."""
    return calendar is not None and bool(n < calendar.searchsorted(last, side="right") - calendar.searchsorted(first, side="left"))


def _listing_floors(context: Context, listing: Table | None, key_col: str, keys: Sequence[str], floor: pd.Timestamp) -> dict[str, pd.Timestamp]:
    """Per key, the later of `floor` and its first date in `listing`; a key with no `listing` row is absent."""
    if listing is None or not keys:
        return {}
    df = context.store.key_stats(listing, key_col, where={key_col: list(keys)})
    return {str(key): max(floor, pd.Timestamp(first)) for key, first in zip(df["key"], df["first"], strict=True) if pd.notna(first)}


@dataclass(frozen=True)
class SeriesRequest:
    """Per-key source scope and optional shared frontier/calendar observations."""

    keys: Sequence[str]
    calendar: pd.DatetimeIndex | None
    until: pd.Timestamp | None = None
    listing: Table | None = None
    stored: tuple[pd.DataFrame, pd.Timestamp | None] | None = None


def series_windows(context: Context, table: Table, request: SeriesRequest) -> SeriesWork:
    """Resolve each key's configured tail, preserving initial history and interior gaps."""
    started = time.perf_counter()
    keys, calendar = request.keys, request.calendar
    policy = window_policy(context.config, table, context.run.as_of)
    if policy is None or (request.until is not None and request.until < policy.requested.start):
        return SeriesWork({key: [] for key in keys}, {})
    if table.resume is None or table.resume.key is None or table.resume.frontier_col is None or policy.overlap is None:
        raise ValueError(f"{table.name} requires a per-key resume contract and configured tail")
    if request.until is not None:
        policy = replace(policy, requested=Window(policy.requested.start, min(request.until, policy.requested.end)))

    # Load only resume observations and the listing floors needed by new keys.
    key_col, date_col, floor = table.resume.key, table.resume.frontier_col, policy.requested.start
    df_stats = context.store.key_stats(table, key_col, date_col, where={key_col: list(keys)}) if request.stored is None else request.stored[0]
    stats = dict(zip(map(str, df_stats["key"]), zip(df_stats["first"], df_stats["last"], map(int, df_stats["n"]), strict=True), strict=True))
    table_max = context.store.max_date(table, date_col) if request.stored is None else request.stored[1]
    cutoff = shift_date(context.run.as_of, policy.overlap, calendar)
    new = {key for key, added in context.universe.added_on_dates(keys, context.run.as_of).items() if added >= cutoff}
    reach = _listing_floors(context, request.listing, key_col, [key for key in keys if key in new], floor)

    # Compute the same shared Window plan independently at each stored frontier.
    windows: dict[str, list[Window]] = {}
    key_class: dict[str, str] = {}
    holed: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {}
    universe_keys = set(context.universe.tickers)
    for key in keys:
        complete = key in stats and pd.Timestamp(stats[key][0]) <= shift_date(reach.get(key, floor), policy.overlap, calendar, direction=1)
        initial = table_max is None or (key not in stats and key not in universe_keys) or (key in new and not complete)
        key_class[key] = KEY_NEW if initial else KEY_ESTABLISHED if key in stats else KEY_ROWLESS
        run = (
            WindowRun("initial", "missing configured source history", context.run.instant)
            if initial and context.run.mode == "incremental"
            else context.run
        )
        frontier = stats[key][1] if key in stats else table_max
        plan = resolve_window(policy, run, frontier, pd.DatetimeIndex([]) if calendar is None else calendar, ((key_col, (key,)),))
        windows[key] = [plan.write]
        if key_class[key] == KEY_ESTABLISHED and _has_holes(calendar, *stats[key]):
            holed[key] = (max(stats[key][0], floor), plan.write.start)

    # Read interior dates only for keys whose stored counts demonstrate a gap.
    read_stats = time.perf_counter()
    if holed:
        df_dates = cast(
            pd.DataFrame,
            context.store.load(
                table,
                columns=[key_col, date_col],
                where={key_col: list(holed)},
                since=floor,
                until=policy.requested.end,
                date_col=date_col,
                markers=True,
            ),
        )
        for key, group in df_dates.groupby(key_col, sort=False):
            windows[str(key)] = _union_spans(windows[str(key)] + _hole_spans(cast(pd.DatetimeIndex, calendar), group[date_col], *holed[str(key)]))

    # Intersect source work with listing/end eligibility without widening config.
    last = {key: edges[1] for key, edges in stats.items()}
    bounds = planner_bounds(context)
    done = {key for key in last if (end := bounds.get(key, (None, None))[1]) is not None and end < cutoff}
    windows = {key: [] if key in done else [b for span in spans if (b := bounded_span(bounds, key, *span))] for key, spans in windows.items()}
    timings = {f"{table.name}.key_stats": read_stats - started, f"{table.name}.holes": time.perf_counter() - read_stats}
    return SeriesWork(windows=windows, key_class=key_class, last=last, timings=timings)


@dataclass
class ArchiveWork:
    """Missing global units and explicitly requested completed-unit replays for an authorized key scope."""

    listed: list[str]
    pending: list[str]
    rescan: list[str]
    keys: list[str]
    screen: list[str]
    scoped: bool
    held: list[str] = field(default_factory=list)

    def units(self) -> Iterator[tuple[str, list[str]]]:
        """`(period, keys to keep)` in period order; a period is never in both lists."""
        yield from ((period, self.keys) for period in sorted(set(self.pending) | set(self.rescan)))


def _period_column(table: Table) -> str:
    if table.resume is None or table.resume.period_col is None:
        raise ValueError(f"{table.name} declares no archive period column")
    return table.resume.period_col


def archive_scopes(context: Context, tables: Sequence[Table], periods: Collection[str]) -> dict[str, tuple[Table, ...]]:
    """Map each configured source-unit end to the tables that require it, without reading completion state."""
    windows = {table: policy.requested for table in tables if (policy := window_policy(context.config, table, context.run.as_of)) is not None}
    scopes = {}
    for period in sorted(set(periods)):
        end = pd.Timestamp(period_end(period))
        required = tuple(table for table, window in windows.items() if window.start <= end <= window.end)
        if required:
            scopes[period] = required
    return scopes


def archive_periods(context: Context, tables: Sequence[Table], published: Sequence[str]) -> tuple[list[str], frozenset[str]]:
    """Published unit-end dates inside each table's bounds, and units complete in all required tables."""
    policies = {table: policy.requested for table in tables if (policy := window_policy(context.config, table, context.run.as_of)) is not None}
    if not policies:
        return [], frozenset()
    values = {table: stored_values(context, table, _period_column(table)) for table in policies}
    required = set(published)
    if tuple(tables) == (Tables.insider_transactions,):
        required.update(values.get(Tables.insider_transactions, ()))
    scopes = archive_scopes(context, tuple(policies), required)
    complete = frozenset(period for period, needed in scopes.items() if all(period in values[table] for table in needed))
    return list(scopes), complete


def archive_worklist(context: Context, tables: Sequence[Table], published: Sequence[str], keys: Sequence[str]) -> ArchiveWork:
    """Plan missing units once; only explicit bounded recovery replays completed archives."""
    keys = sorted(set(keys))
    listed, stored = archive_periods(context, tables, published)
    missing = [period for period in listed if period not in stored]
    scoped = not context.universe.is_full(keys)
    pending, held = ([], missing) if scoped else (missing, [])
    rescan = [period for period in listed if period in stored] if context.run.mode in {"full", "recovery"} else []
    if held:
        logger.info("%s: scoped run leaves %d unparsed unit(s) to the full-universe run: %s", tables[0].name, len(held), held)
    return ArchiveWork(
        listed=listed,
        pending=pending,
        rescan=rescan,
        keys=keys,
        screen=sorted(set(context.universe.tickers) | set(keys)),
        scoped=scoped,
        held=held,
    )
