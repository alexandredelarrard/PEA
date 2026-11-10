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

`archive_worklist` (bulk archives): the published periods missing from any of the fetcher's tables,
parsed for every key; plus, for a new key with no archive row yet, every cached period re-parsed for it alone.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Collection, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import cast

import pandas as pd

from src.context import Context
from src.data_extract.utils.common.incremental import stored_values
from src.data_store.schema import Resume, Table, Tables
from src.utils.string import run_date
from src.utils.tickers_universe.universe import Bounds

logger = logging.getLogger(__name__)

#: Key classes reported by `resume-plan`.
KEY_NEW = "new"
KEY_ROWLESS = "rowless"
KEY_ESTABLISHED = "established"
#: Done-set scopes: accessions stored under the key itself, or anywhere in the table.
DONE_PER_KEY = "key"
DONE_TABLE = "table"
#: Days a new ticker with no row re-parses the cached archives, for every archive (as the 13F backfill).
ARCHIVE_NEW_KEY_DAYS = 7
#: Days after a lineage scope change during which the identity propagation re-checks the key (purge,
#: re-parse, re-stamp, history), the window a new key gets for its archive history.
LINEAGE_RECHECK_DAYS = ARCHIVE_NEW_KEY_DAYS


def recently_changed(stamps: Mapping[str, pd.Timestamp | None], as_of: pd.Timestamp | None, days: int = LINEAGE_RECHECK_DAYS) -> list[str]:
    """Keys whose change stamp (a lineage or master `scope_changed_at`) is on or after `as_of - days` (`as_of` None: today), sorted.

    Every re-check it drives is idempotent, so a key seen on several runs inside the window costs reads only,
    and a failed run is healed by the next one."""
    since = run_date(as_of) - pd.Timedelta(days=days)
    return sorted(str(key) for key, stamp in stamps.items() if stamp is not None and not pd.isna(stamp) and pd.Timestamp(stamp) >= since)


#: U-65: every planner keeps fetching for this many trading sessions after a company's `ended_on` (its completion 8-K,
#: its last calls and surprises are dated just after the last trade).
END_GRACE_SESSIONS = 10


def planner_bounds(context: Context) -> Bounds:
    """`Universe.bounds` with each `ended_on` moved `END_GRACE_SESSIONS` sessions later on the `prices` calendar (U-65);
    weekdays where the calendar does not reach that far yet. Open keys and first price dates are unchanged. Built once per
    context and universe (a per-ticker planner calls it for every ticker)."""
    cached = getattr(context, "_planner_bounds", None)
    if cached is not None and cached[0] is context.universe:
        return cached[1]
    bounds = context.universe.bounds
    if any(end is not None for _, end in bounds.values()):
        calendar = trading_calendar(context)
        bounds = {key: (first, None if end is None else _grace_end(calendar, end)) for key, (first, end) in bounds.items()}
    context._planner_bounds = (context.universe, bounds)
    return bounds


def _grace_end(calendar: pd.DatetimeIndex, end: pd.Timestamp) -> pd.Timestamp:
    """The `END_GRACE_SESSIONS`-th session after `end` in `calendar`, else that many weekdays after it."""
    position = int(calendar.searchsorted(end, side="right")) + END_GRACE_SESSIONS - 1
    return calendar[position] if position < len(calendar) else end + pd.offsets.BDay(END_GRACE_SESSIONS)


def bounded_span(bounds: Bounds, key: str, since: pd.Timestamp, until: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """`(since, until)` clamped to the key's `[first_price_date, ended_on]`; None when nothing is left. A key
    without bounds (a secondary class, a key outside the universe) keeps its span."""
    first, ended = bounds.get(key, (None, None))
    start = max(since, first) if first is not None else since
    end = min(until, ended) if ended is not None else until
    return (start, end) if start <= end else None


def ended_keys(bounds: Bounds, stored: Collection[str], as_of: pd.Timestamp, overlap_days: int) -> list[str]:
    """Keys of `stored` (keys with stored rows) whose planner end (`planner_bounds`: `ended_on` plus the U-65 grace) plus
    the table's `overlap_days` is before the run date `as_of`: nothing is left to fetch, sorted. Until then a late final
    row is still fetched inside the grace; a key with no stored row is never in it."""
    cutoff = run_date(as_of) - pd.Timedelta(days=overlap_days)
    return sorted(key for key in stored if (end := bounds.get(key, (None, None))[1]) is not None and end < cutoff)


def document_floor(table: Table, as_of: pd.Timestamp, years_history: int, runtime_floor: pd.Timestamp | None = None) -> pd.Timestamp:
    """The earliest filing date listed: `as_of - years_history`, the source start or the runtime floor, whichever is latest."""
    floors = [run_date(as_of) - pd.DateOffset(years=years_history)]
    if table.resume is not None and table.resume.source_start is not None:
        floors.append(pd.Timestamp(table.resume.source_start))
    if runtime_floor is not None:
        floors.append(pd.Timestamp(runtime_floor).normalize())
    return max(floors)


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

    windows: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]]
    key_class: dict[str, str]
    last: dict[str, pd.Timestamp] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)

    def groups(self) -> list[tuple[pd.Timestamp, pd.Timestamp, list[str]]]:
        """Keys sharing an identical window, one group per window, oldest `since` first."""
        by_window: dict[tuple[pd.Timestamp, pd.Timestamp], list[str]] = {}
        for key, spans in self.windows.items():
            for span in spans:
                by_window.setdefault(span, []).append(key)
        return [(since, until, sorted(keys)) for (since, until), keys in sorted(by_window.items())]

    def merge(self, other: SeriesWork) -> SeriesWork:
        """The union of two work lists (each key's windows unioned); key classes and `last` stay this work's."""
        keys = set(self.windows) | set(other.windows)
        windows = {key: _union_spans(self.windows.get(key, []) + other.windows.get(key, [])) for key in keys}
        return SeriesWork(windows=windows, key_class=other.key_class | self.key_class, last=self.last, timings=self.timings | other.timings)


def _union_spans(spans: list[tuple[pd.Timestamp, pd.Timestamp]]) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Overlapping or touching `(since, until)` spans merged, sorted."""
    merged: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for since, until in sorted(spans):
        if merged and since <= merged[-1][1] + pd.Timedelta(days=1):
            merged[-1] = (merged[-1][0], max(merged[-1][1], until))
        else:
            merged.append((since, until))
    return merged


def trading_calendar(context: Context) -> pd.DatetimeIndex:
    """The distinct dates of `prices`, sorted; empty when the table is absent. Dates only: a secondary class
    in `prices` is fetched inside its company's own windows, so it adds no session."""
    return pd.DatetimeIndex(pd.to_datetime(context.store.distinct(Tables.prices, "date"))).normalize().unique().sort_values()


def _hole_spans(calendar: pd.DatetimeIndex, stored: pd.Series, first: pd.Timestamp, edge: pd.Timestamp) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Runs of calendar sessions in `[first, edge)` with no stored row, as `(first missing, last missing)` spans."""
    span = calendar[(calendar >= first) & (calendar < edge)]
    missing = ~span.isin(pd.DatetimeIndex(pd.to_datetime(stored)).normalize())
    if not missing.any():
        return []
    positions = pd.Series(range(len(span)))[missing]
    runs = (positions.diff() != 1).cumsum()
    return [(span[int(run.iloc[0])], span[int(run.iloc[-1])]) for _, run in positions.groupby(runs)]


def _has_holes(calendar: pd.DatetimeIndex | None, first: pd.Timestamp, last: pd.Timestamp, n: int) -> bool:
    """True when the daily `calendar` holds more sessions in `[first, last]` than the key's `n` stored rows."""
    return calendar is not None and bool(n < calendar.searchsorted(last, side="right") - calendar.searchsorted(first, side="left"))


def _reaches_floor(edges: tuple[pd.Timestamp, pd.Timestamp, int] | None, floor: pd.Timestamp, tolerance: pd.Timedelta) -> bool:
    """True when a key's stored history starts within `tolerance` of the floor: its full pull is done.

    The tolerance (the table's overlap) absorbs a floor on a non-session day and a quarterly series'
    first filing after it."""
    return edges is not None and pd.Timestamp(edges[0]) <= floor + tolerance


def _listing_floors(context: Context, listing: Table | None, key_col: str, keys: Sequence[str], floor: pd.Timestamp) -> dict[str, pd.Timestamp]:
    """Per key, the later of `floor` and its first date in `listing`; a key with no `listing` row is absent."""
    if listing is None or not keys:
        return {}
    df = context.store.key_stats(listing, key_col, where={key_col: list(keys)})
    return {str(key): max(floor, pd.Timestamp(first)) for key, first in zip(df["key"], df["first"], strict=True) if pd.notna(first)}


def series_windows(
    context: Context,
    table: Table,
    keys: Sequence[str],
    as_of: pd.Timestamp,
    *,
    until: pd.Timestamp,
    years_history: int,
    full: bool = False,
    calendar: pd.DatetimeIndex | None,
    listing: Table | None = None,
    stored: tuple[pd.DataFrame, pd.Timestamp | None] | None = None,
) -> SeriesWork:
    """Per-key fetch windows for a dated series `table` from one `key_stats` read (plus one date read
    for the keys with interior holes against the daily `calendar`; None for a non-daily series, which
    gets no hole windows). Never earlier than the history floor.

    `listing`: a table keyed like `table` whose first date per new key is that key's listing floor
    when later than the floor; a key listed late has its full pull done once its history reaches it."""
    started = time.perf_counter()
    if table.resume is None or table.resume.key is None:
        raise ValueError(f"{table.name} declares no per-key resume contract")

    key_col, date_col = table.resume.key, cast(str, table.resume.frontier_col)
    overlap = pd.Timedelta(days=table.resume.overlap_days)

    # get first date to extract
    floor = document_floor(table, as_of, years_history)

    # get per ticker, first, last date and n count
    df_stats = context.store.key_stats(table, key_col, date_col, where={key_col: list(keys)}) if stored is None else stored[0]
    stats = dict(zip(map(str, df_stats["key"]), zip(df_stats["first"], df_stats["last"], map(int, df_stats["n"]), strict=True), strict=True))
    new = set() if full else context.universe.new_since(table.resume.overlap_days, as_of)
    table_max = context.store.max_date(table, date_col) if stored is None else stored[1]
    reach = _listing_floors(context, listing, key_col, [key for key in keys if key in new], floor)

    windows: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]] = {}
    key_class: dict[str, str] = {}
    holed: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {}
    for key in keys:
        if full or table_max is None or (key in new and not _reaches_floor(stats.get(key), reach.get(key, floor), overlap)):
            key_class[key], since = KEY_NEW, floor
        elif key not in stats:
            key_class[key], since = KEY_ROWLESS, max(table_max - overlap, floor)
        else:
            key_class[key], since = KEY_ESTABLISHED, max(stats[key][1] - overlap, floor)
        if key_class[key] == KEY_ESTABLISHED and _has_holes(calendar, *stats[key]):
            holed[key] = (max(stats[key][0], floor), since)

        # start date to redo is latest day - overlap days
        windows[key] = [(since, until)] if since <= until else []

    read_stats = time.perf_counter()
    if holed:
        df_dates = cast(pd.DataFrame, context.store.load(table, columns=[key_col, date_col], where={key_col: list(holed)}, markers=True))
        for key, group in df_dates.groupby(key_col, sort=False):
            windows[str(key)] = _union_spans(windows[str(key)] + _hole_spans(cast(pd.DatetimeIndex, calendar), group[date_col], *holed[str(key)]))
    last = {key: edges[1] for key, edges in stats.items()}

    # clamp every window to the key's planner bounds (end + grace); an ended key with rows is done past grace end + overlap
    bounds = planner_bounds(context)
    done = set(ended_keys(bounds, [key for key in last if key in windows], as_of, table.resume.overlap_days))
    windows = {key: [] if key in done else [b for span in spans if (b := bounded_span(bounds, key, *span))] for key, spans in windows.items()}
    if done:
        logger.info(
            "%s: %d ended key(s) with stored rows past their grace end plus overlap, no window: %s", table.name, len(done), ", ".join(sorted(done))
        )
    timings = {f"{table.name}.key_stats": read_stats - started, f"{table.name}.holes": time.perf_counter() - read_stats}
    return SeriesWork(windows=windows, key_class=key_class, last=last, timings=timings)


@dataclass
class ArchiveWork:
    """An archive fetcher's work list.

    `pending` periods are parsed for every key of the run (`screen` is the universe a parse screens
    against); `rescan` periods, already stored in every table, are re-parsed for `rescan_keys` only.
    A scoped (`-t`) run parses no pending period: parsing it for a few keys would mark it stored for all.
    """

    listed: list[str]
    pending: list[str]
    rescan: list[str]
    keys: list[str]
    rescan_keys: list[str]
    screen: list[str]
    scoped: bool
    held: list[str] = field(default_factory=list)

    def units(self) -> Iterator[tuple[str, list[str]]]:
        """`(period, keys to keep)` in period order; a period is never in both lists."""
        by_period = {period: self.keys for period in self.pending} | {period: self.rescan_keys for period in self.rescan if self.rescan_keys}
        yield from sorted(by_period.items())


def _period_column(table: Table) -> str:
    if table.resume is None or table.resume.period_col is None:
        raise ValueError(f"{table.name} declares no archive period column")
    return table.resume.period_col


def period_end(tag: str) -> date:
    """Last calendar day covered by a `YYYYqN`, `YYYY_MM` or semi-monthly `YYYYMMa`/`YYYYMMb` archive tag."""
    if tag[-1] in "ab":
        month_end = pd.Period(year=int(tag[:4]), month=int(tag[4:6]), freq="M").end_time.date()
        return month_end.replace(day=15) if tag[-1] == "a" else month_end
    month = int(tag[-2:]) if "_" in tag else int(tag[-1]) * 3
    return pd.Period(year=int(tag[:4]), month=month, freq="M").end_time.date()


def archive_periods(context: Context, tables: Sequence[Table], published: Sequence[str]) -> tuple[list[str], frozenset[str]]:
    """`(listed, stored)`: the `published` periods from the latest source start of `tables` on, and the
    periods stored in every one of `tables` (an absent table stores none)."""
    starts = [pd.Timestamp(t.resume.source_start).date() for t in tables if t.resume is not None and t.resume.source_start]
    floor = max(starts) if starts else None
    stored: frozenset[str] | None = None
    for table in tables:
        values = stored_values(context, table, _period_column(table))
        stored = values if stored is None else stored & values
    required = list(published)
    if tuple(tables) == (Tables.insider_transactions,):
        required.extend(sorted(stored or ()))
    listed = [p for p in dict.fromkeys(required) if floor is None or period_end(p) >= floor]
    return listed, stored or frozenset()


def _new_rowless_keys(context: Context, tables: Sequence[Table], keys: Collection[str], as_of: pd.Timestamp) -> set[str]:
    """Keys added in the last `ARCHIVE_NEW_KEY_DAYS` days that have no archive row in any of `tables`.

    An archive row carries its period; a row without one (an EDGAR row or marker of a table both
    legs write) does not make the key's archive history stored."""
    new = context.universe.new_since(ARCHIVE_NEW_KEY_DAYS, as_of) & set(keys)
    for table in tables:
        if not new:
            break
        key_col = cast(str, cast(Resume, table.resume).key)
        if {key_col, _period_column(table)} <= set(context.store.columns(table)):
            where = {key_col: sorted(new), _period_column(table): context.store.NOT_NULL}
            new -= set(context.store.distinct(table, key_col, where=where))
    return new


def archive_worklist(
    context: Context,
    tables: Sequence[Table],
    published: Sequence[str],
    cached: Collection[str],
    keys: Sequence[str],
    as_of: pd.Timestamp,
    *,
    full: bool = False,
) -> ArchiveWork:
    """Which archive periods to parse, and for which keys, from the stored periods of `tables` alone.

    Pending = listed periods missing from any table. Rescan = listed periods stored in every table and
    cached on disk (any stored period under `full`), for the new rowless keys, or every key under `full`.
    """
    keys = sorted(set(keys))
    listed, stored = archive_periods(context, tables, published)
    missing = [p for p in listed if p not in stored]
    scoped = not context.universe.is_full(keys)
    rescan_keys = keys if full else sorted(_new_rowless_keys(context, tables, keys, as_of))
    on_disk = set(cached)
    rescan = [p for p in listed if p in stored and (full or p in on_disk)] if rescan_keys else []
    if rescan_keys and not full:
        absent = [p for p in listed if p in stored and p not in on_disk]
        if absent:
            logger.warning(
                "%s: %d stored period(s) have no cached archive and are not re-parsed for new keys: %s", tables[0].name, len(absent), absent
            )
    pending, held = ([], missing) if scoped else (missing, [])
    if held:
        logger.info("%s: scoped run leaves %d unparsed period(s) to the full run: %s", tables[0].name, len(held), held)
    return ArchiveWork(
        listed=listed,
        pending=pending,
        rescan=rescan,
        keys=keys,
        rescan_keys=rescan_keys,
        screen=sorted(set(context.universe.tickers) | set(keys)),
        scoped=scoped,
        held=held,
    )
