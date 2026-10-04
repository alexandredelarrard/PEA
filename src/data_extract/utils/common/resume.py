"""Work-list planners: what each nightly run fetches, derived from the table's own rows, its
`schema.Resume` contract and the run date. Nothing else is read or written.

`document_worklist` (EDGAR document tables): the local EDGAR index rows of the table's forms for
each key's registrant lineage, from the floor on, minus the accessions already stored (empty-filing
markers included); newest first and capped per run.

`series_windows` (dated per-key series): per key, its own last date minus the overlap through
`until`; the full history for a new key, an absent table or `full`; the table-wide frontier minus
the overlap for a rowless key; plus one window per run of calendar sessions missing inside the
key's stored span.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import cast

import pandas as pd

from src.context import Context
from src.data_extract.utils.common import edgar_index
from src.data_extract.utils.common.identity import Identity
from src.data_extract.utils.common.registrant import Registrant, listing_ciks, resolve_registrant_entries
from src.data_store.schema import Table, Tables
from src.utils.universe import new_tickers

logger = logging.getLogger(__name__)

#: Key classes reported by `resume-plan`.
KEY_NEW = "new"
KEY_ROWLESS = "rowless"
KEY_ESTABLISHED = "established"
#: Done-set scopes: accessions stored under the key itself, or anywhere in the table.
DONE_PER_KEY = "key"
DONE_TABLE = "table"


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


def stored_accessions(context: Context, table: Table, keys: Sequence[str], scope: str) -> dict[str, set[str]]:
    """Accessions already stored, markers included: per key (`DONE_PER_KEY`, one two-column read) or one
    table-wide set shared by every key (`DONE_TABLE`, one `SELECT DISTINCT`)."""
    key_col = table.resume.key if table.resume is not None and table.resume.key else "ticker"
    if scope == DONE_TABLE:
        everywhere = {str(a) for a in context.store.distinct(table, "accession_number")}
        return {key: everywhere for key in keys}
    df = context.store.load(table, columns=[key_col, "accession_number"], where={key_col: list(keys)}, markers=True, optional=True)
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
    registrants: dict[str, Registrant],
    identity: Identity | None,
    years_history: int,
    runtime_floor: pd.Timestamp | None = None,
    done_scope: str = DONE_TABLE,
    full: bool = False,
    cap: int | None = None,
) -> DocumentWork:
    """The documents `table` still lacks for `keys` (`ticker`, `cik`): index rows from the floor on,
    resolved through each key's registrant lineage, minus stored accessions (ignored under `full`)."""
    started = time.perf_counter()
    floor = document_floor(table, as_of, years_history, runtime_floor)
    roster = dict(zip(keys["ticker"].astype(str), keys["cik"].astype(str), strict=True))
    candidates = {ticker: listing_ciks(ticker, cik, registrants, identity) for ticker, cik in roster.items()}
    df_index = edgar_index.entries(context, {c for ciks in candidates.values() for c in ciks}, forms, since=floor)
    by_cik = dict(tuple(df_index.groupby("cik", sort=False))) if not df_index.empty else {}
    read_index = time.perf_counter()
    done = {key: set() for key in roster} if full else stored_accessions(context, table, list(roster), done_scope)
    read_done = time.perf_counter()
    overlap = table.resume.overlap_days if table.resume is not None else 0
    new = new_tickers(context.store, overlap, as_of) if overlap else set()
    units: dict[str, pd.DataFrame] = {}
    key_class: dict[str, str] = {}
    counts: dict[str, int] = {}
    for ticker, cik in roster.items():
        frames = [by_cik[c] for c in candidates[ticker] if c in by_cik]
        df_listed = (
            resolve_registrant_entries(ticker, cik, pd.concat(frames, ignore_index=True), forms, registrants=registrants, identity=identity)
            if frames
            else df_index.iloc[:0]
        )
        df_units = df_listed[~df_listed["accession"].isin(done.get(ticker, set()))].reset_index(drop=True)
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
    """The distinct dates of `prices`, sorted; empty when the table is absent."""
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
    stats = {str(k): (pd.Timestamp(f), pd.Timestamp(lst), int(n)) for k, f, lst, n in df_stats[["key", "first", "last", "n"]].itertuples(index=False)}
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
            first, last, n = stats[key]
            key_class[key], since = KEY_ESTABLISHED, max(last - overlap, floor)
            if calendar is not None and n < calendar.searchsorted(last, side="right") - calendar.searchsorted(first, side="left"):
                holed[key] = (max(first, floor), since)
        windows[key] = [(since, until)] if since <= until else []
    read_stats = time.perf_counter()
    if holed:
        df_dates = cast(pd.DataFrame, context.store.load(table, columns=[key_col, date_col], where={key_col: list(holed)}, markers=True))
        for key, group in df_dates.groupby(key_col, sort=False):
            windows[str(key)] = _union_spans(windows[str(key)] + _hole_spans(cast(pd.DatetimeIndex, calendar), group[date_col], *holed[str(key)]))
    last = {key: edges[1] for key, edges in stats.items()}
    timings = {f"{table.name}.key_stats": read_stats - started, f"{table.name}.holes": time.perf_counter() - read_stats}
    return SeriesWork(windows=windows, key_class=key_class, last=last, timings=timings)
