"""Work-list planners: what each nightly run fetches, derived from the table's own rows, its
`schema.Resume` contract and the run date. Nothing else is read or written.

`document_worklist` (EDGAR document tables): the local EDGAR index rows of the table's forms for
each key's filing scope (event CIKs and CIK windows), from the floor on, minus the accessions already stored (empty-filing
markers included; optionally only on the rows of one source); newest first and capped per run.

`series_windows` (dated per-key series): per key, its own last date minus the overlap through
`until`; the full history for a new key, an absent table or `full`; the table-wide frontier minus
the overlap for a rowless key; plus one window per run of calendar sessions missing inside the
key's stored span.

`archive_worklist` (bulk archives): the published periods missing from any of the fetcher's tables,
parsed for every key; plus, for a new key with no archive row yet, every cached period re-parsed for it alone.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Collection, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import cast

import pandas as pd

from src.context import Context
from src.data_extract.utils.common import edgar_index
from src.data_extract.utils.common.bulk_cache import period_end
from src.data_extract.utils.common.identity import FilingScope, Identity
from src.data_extract.utils.common.incremental import stored_values
from src.data_extract.utils.common.registrant import listing_ciks, resolve_registrant_entries
from src.data_store.schema import Resume, Table, Tables
from src.utils.universe import load_universe_tickers, new_tickers

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
    since = pd.Timestamp(as_of if as_of is not None else pd.Timestamp.today()).normalize() - pd.Timedelta(days=days)
    return sorted(str(key) for key, stamp in stamps.items() if stamp is not None and not pd.isna(stamp) and pd.Timestamp(stamp) >= since)


@dataclass
class DocumentWork:
    """One table's document work list: index rows to read per key (oldest first), plus what `resume-plan` reports."""

    units: dict[str, pd.DataFrame]
    key_class: dict[str, str]
    counts: dict[str, int] = field(default_factory=dict)
    uncapped: int = 0
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def size(self) -> int:
        return sum(len(df) for df in self.units.values())


def document_floor(table: Table, as_of: pd.Timestamp, years_history: int, runtime_floor: pd.Timestamp | None = None) -> pd.Timestamp:
    """The earliest filing date listed: `as_of - years_history`, the source start or the runtime floor, whichever is latest."""
    floors = [pd.Timestamp(as_of).normalize() - pd.DateOffset(years=years_history)]
    if table.resume is not None and table.resume.source_start is not None:
        floors.append(pd.Timestamp(table.resume.source_start))
    if runtime_floor is not None:
        floors.append(pd.Timestamp(runtime_floor).normalize())
    return max(floors)


def stored_accessions(context: Context, table: Table, keys: Sequence[str], scope: str, where: dict[str, object] | None = None) -> dict[str, set[str]]:
    """Accessions already stored on the rows matching `where`, markers included: per key (`DONE_PER_KEY`,
    one two-column read) or one table-wide set shared by every key (`DONE_TABLE`, one `SELECT DISTINCT`)."""
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


def _key_class(ticker: str, new: set[str], df_listed: pd.DataFrame, df_units: pd.DataFrame) -> str:
    """`KEY_NEW` inside the table's overlap since `added_on`, `KEY_ROWLESS` when none of its listed documents is stored, else `KEY_ESTABLISHED`."""
    if ticker in new:
        return KEY_NEW
    return KEY_ROWLESS if len(df_units) == len(df_listed) else KEY_ESTABLISHED


def _classify_units(df_listed: pd.DataFrame, df_units: pd.DataFrame, key_class: str, overlap_days: int) -> dict[str, int]:
    """Unit counts by class: a new or rowless key's units, else forward (inside the overlap of the key's
    last stored filing) and gap (older)."""
    if df_units.empty:
        return {}
    if key_class != KEY_ESTABLISHED:
        return {key_class: len(df_units)}
    stored = df_listed.loc[~df_listed["accession"].isin(df_units["accession"]), "filed"]
    edge = (stored.max() if not stored.empty else df_units["filed"].min()) - pd.Timedelta(days=overlap_days)
    n_forward = int((df_units["filed"] >= edge).sum())
    return {"forward": n_forward, "gap": len(df_units) - n_forward}


def _cap(units: dict[str, pd.DataFrame], cap: int | None, desc: str) -> tuple[dict[str, pd.DataFrame], int]:
    """At most `cap` units in total, newest first, each key's list kept oldest first; an ERROR names the uncapped size."""
    uncapped = sum(len(df) for df in units.values())
    if cap is None or uncapped <= cap:
        return units, uncapped
    logger.error(
        "%s: work list of %d document(s) exceeds the per-run cap of %d; the newest %d run tonight, the rest on later runs", desc, uncapped, cap, cap
    )
    df_all = pd.concat([df.assign(_key=key) for key, df in units.items()], ignore_index=True)
    df_kept = df_all.sort_values(["filed", "accession"], ascending=False, kind="mergesort").head(cap)
    capped = {str(key): group.drop(columns="_key").sort_values(["filed", "accession"], ignore_index=True) for key, group in df_kept.groupby("_key")}
    return capped, uncapped


def document_worklist(
    context: Context,
    table: Table,
    keys: pd.DataFrame,
    as_of: pd.Timestamp,
    *,
    forms: Sequence[str],
    identity: Identity | None,
    years_history: int,
    runtime_floor: pd.Timestamp | None = None,
    done_scope: str = DONE_TABLE,
    done_where: dict[str, object] | None = None,
    full: bool = False,
    cap: int | None = None,
) -> DocumentWork:
    """The documents `table` still lacks for `keys` (`ticker`, `cik`): index rows from the floor on,
    resolved through each key's filing scope (its roster CIK alone without `identity`), minus the
    accessions stored on the rows matching `done_where` (ignored under `full`)."""
    started = time.perf_counter()
    floor = document_floor(table, as_of, years_history, runtime_floor)
    roster = dict(zip(keys["ticker"].astype(str), keys["cik"].astype(str), strict=True))
    scopes = {
        ticker: identity.filing_scope(ticker) if identity is not None else FilingScope.roster_only(ticker, cik) for ticker, cik in roster.items()
    }
    candidates = {ticker: listing_ciks(scope) for ticker, scope in scopes.items()}
    df_index = edgar_index.entries(context, {c for ciks in candidates.values() for c in ciks}, forms, since=floor)
    by_cik = dict(tuple(df_index.groupby("cik", sort=False))) if not df_index.empty else {}
    read_index = time.perf_counter()
    done = {key: set() for key in roster} if full else stored_accessions(context, table, list(roster), done_scope, done_where)
    read_done = time.perf_counter()
    overlap = table.resume.overlap_days if table.resume is not None else 0
    new = new_tickers(context.store, overlap, as_of) if overlap else set()
    units: dict[str, pd.DataFrame] = {}
    key_class: dict[str, str] = {}
    counts: dict[str, int] = {}
    for ticker in roster:
        frames = [by_cik[c] for c in candidates[ticker] if c in by_cik]
        df_listed = resolve_registrant_entries(scopes[ticker], pd.concat(frames, ignore_index=True), forms) if frames else df_index.iloc[:0]
        done_here = done.get(ticker, set()) & set(df_listed["accession"].astype(str))
        df_units = df_listed[~df_listed["accession"].isin(done_here)].reset_index(drop=True)
        key_class[ticker] = _key_class(ticker, new, df_listed, df_units)
        for name, n in _classify_units(df_listed, df_units, key_class[ticker], overlap).items():
            counts[name] = counts.get(name, 0) + n
        if not df_units.empty:
            units[ticker] = df_units
    units, uncapped = _cap(units, cap, table.name)
    timings = {"index_read": read_index - started, "done_read": read_done - read_index, "total": time.perf_counter() - started}
    return DocumentWork(units=units, key_class=key_class, counts=counts, uncapped=uncapped, timings=timings)


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


def session_dates(calendar: pd.DatetimeIndex, since: pd.Timestamp, until: pd.Timestamp) -> pd.DatetimeIndex:
    """Calendar sessions in `[since, until]`, plus business days after the calendar's last session."""
    inside = calendar[(calendar >= since) & (calendar <= until)]
    after = calendar.max() + pd.Timedelta(days=1) if len(calendar) else since
    return inside.union(pd.bdate_range(max(after, since), until)) if after <= until else inside


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
) -> SeriesWork:
    """Per-key fetch windows for a dated series `table` from one `key_stats` read (plus one date read
    for the keys with interior holes against the daily `calendar`; None for a non-daily series, which
    gets no hole windows). Never earlier than the history floor."""
    started = time.perf_counter()
    if table.resume is None or table.resume.key is None:
        raise ValueError(f"{table.name} declares no per-key resume contract")
    key_col, date_col = table.resume.key, cast(str, table.resume.frontier_col)
    overlap = pd.Timedelta(days=table.resume.overlap_days)
    floor = document_floor(table, as_of, years_history)
    df_stats = context.store.key_stats(table, key_col, date_col, where={key_col: list(keys)})
    stats = dict(zip(map(str, df_stats["key"]), zip(df_stats["first"], df_stats["last"], map(int, df_stats["n"]), strict=True), strict=True))
    new = set() if full else new_tickers(context.store, table.resume.overlap_days, as_of)
    table_max = context.store.max_date(table, date_col)
    windows: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]] = {}
    key_class: dict[str, str] = {}
    holed: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {}
    for key in keys:
        if full or table_max is None or key in new:
            key_class[key], since = KEY_NEW, floor
        elif key not in stats:
            key_class[key], since = KEY_ROWLESS, max(table_max - overlap, floor)
        else:
            key_class[key], since = KEY_ESTABLISHED, max(stats[key][1] - overlap, floor)
        if key_class[key] == KEY_ESTABLISHED and _has_holes(calendar, *stats[key]):
            holed[key] = (max(stats[key][0], floor), since)
        windows[key] = [(since, until)] if since <= until else []
    read_stats = time.perf_counter()
    if holed:
        df_dates = cast(pd.DataFrame, context.store.load(table, columns=[key_col, date_col], where={key_col: list(holed)}, markers=True))
        for key, group in df_dates.groupby(key_col, sort=False):
            windows[str(key)] = _union_spans(windows[str(key)] + _hole_spans(cast(pd.DatetimeIndex, calendar), group[date_col], *holed[str(key)]))
    last = {key: edges[1] for key, edges in stats.items()}
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


def archive_periods(context: Context, tables: Sequence[Table], published: Sequence[str]) -> tuple[list[str], frozenset[str]]:
    """`(listed, stored)`: the `published` periods from the latest source start of `tables` on, and the
    periods stored in every one of `tables` (an absent table stores none)."""
    starts = [pd.Timestamp(t.resume.source_start).date() for t in tables if t.resume is not None and t.resume.source_start]
    floor = max(starts) if starts else None
    listed = [p for p in dict.fromkeys(published) if floor is None or period_end(p) >= floor]
    stored: frozenset[str] | None = None
    for table in tables:
        values = stored_values(context, table, _period_column(table))
        stored = values if stored is None else stored & values
    return listed, stored or frozenset()


def _new_rowless_keys(context: Context, tables: Sequence[Table], keys: Collection[str], as_of: pd.Timestamp) -> set[str]:
    """Keys added in the last `ARCHIVE_NEW_KEY_DAYS` days that have no archive row in any of `tables`.

    An archive row carries its period; a row without one (an EDGAR row or marker of a table both
    legs write) does not make the key's archive history stored."""
    new = new_tickers(context.store, ARCHIVE_NEW_KEY_DAYS, as_of) & set(keys)
    for table in tables:
        if not new:
            break
        key_col = cast(str, cast(Resume, table.resume).key)
        if {key_col, _period_column(table)} <= set(context.store.columns(table)):
            where = {key_col: sorted(new), _period_column(table): context.store.NOT_NULL}
            new -= {str(k).upper() for k in context.store.distinct(table, key_col, where=where)}
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
    keys = sorted({str(k).strip().upper() for k in keys})
    listed, stored = archive_periods(context, tables, published)
    missing = [p for p in listed if p not in stored]
    universe = set(load_universe_tickers(context))
    scoped = bool(universe - set(keys))
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
        screen=sorted(universe | set(keys)),
        scoped=scoped,
        held=held,
    )
