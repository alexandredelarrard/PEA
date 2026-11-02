"""EDGAR document work list: what each nightly run fetches for an EDGAR document table.

`document_worklist`: the local EDGAR index rows of the table's forms for
each key's filing scope (event CIKs and CIK windows), from the floor on, minus the accessions already stored (empty-filing
markers included; optionally only on the rows of one source); newest first and capped per run.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import pandas as pd

from src.context import Context
from src.data_extract.utils.common.identity import FilingScope, Identity
from src.data_extract.utils.common.registrant import listing_ciks, resolve_registrant_entries
from src.data_extract.utils.common.resume import DONE_TABLE, KEY_ESTABLISHED, KEY_NEW, KEY_ROWLESS, document_floor, stored_accessions
from src.data_extract.utils.edgar_extract import edgar_index
from src.data_store.schema import Table
from src.utils.universe import new_in_window

logger = logging.getLogger(__name__)

#: `(context, keys, runtime_floor) -> {key: done accessions}`: a fetch's own stored-set rule.
DoneFn = Callable[[Context, Sequence[str], pd.Timestamp | None], dict[str, set[str]]]


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


@dataclass(frozen=True)
class WorklistRequest:
    """One walk's planning inputs: the `keys` (`ticker`, `cik`) on the `as_of` run date, the table's `forms`, the
    `identity` (None: roster CIKs only), the `years_history` window, the run's `added_on` map (read once per run),
    the runtime floor, the done-accession scope and filter, `full` re-read and the per-run `cap` (None: uncapped).
    `listing_floor` replaces the window floor (a source-history fetch lists from its source start); `done_accessions`
    replaces the scope/filter read with the fetch's own stored-set rule."""

    keys: pd.DataFrame
    as_of: pd.Timestamp
    forms: Sequence[str]
    identity: Identity | None
    years_history: int
    added_on: Mapping[str, pd.Timestamp]
    runtime_floor: pd.Timestamp | None = None
    done_scope: str = DONE_TABLE
    done_where: dict[str, object] | None = None
    full: bool = False
    cap: int | None = None
    listing_floor: pd.Timestamp | None = None
    done_accessions: DoneFn | None = None


def document_worklist(context: Context, table: Table, request: WorklistRequest) -> DocumentWork:
    """The documents `table` still lacks for the request's keys: index rows from the floor on, resolved through
    each key's filing scope (its roster CIK alone without `identity`), minus the accessions stored on the rows
    matching `done_where`, or by the request's own `done_accessions` (ignored under `full`)."""
    keys, as_of, forms, identity, runtime_floor, full = (
        request.keys,
        request.as_of,
        request.forms,
        request.identity,
        request.runtime_floor,
        request.full,
    )
    started = time.perf_counter()
    floor = request.listing_floor if request.listing_floor is not None else document_floor(table, as_of, request.years_history, runtime_floor)
    roster = dict(zip(keys["ticker"].astype(str), keys["cik"].astype(str), strict=True))
    scopes = {
        ticker: identity.filing_scope(ticker) if identity is not None else FilingScope.roster_only(ticker, cik) for ticker, cik in roster.items()
    }
    candidates = {ticker: listing_ciks(scope) for ticker, scope in scopes.items()}
    df_index = edgar_index.entries(context, {c for ciks in candidates.values() for c in ciks}, forms, since=floor)
    by_cik = dict(tuple(df_index.groupby("cik", sort=False))) if not df_index.empty else {}
    read_index = time.perf_counter()
    if full:
        done: dict[str, set[str]] = {key: set() for key in roster}
    elif request.done_accessions is not None:
        done = request.done_accessions(context, list(roster), runtime_floor)
    else:
        done = stored_accessions(context, table, list(roster), request.done_scope, request.done_where)
    read_done = time.perf_counter()
    overlap = table.resume.overlap_days if table.resume is not None else 0
    new = new_in_window(request.added_on, overlap, as_of) if overlap else set()
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
    units, uncapped = _cap(units, request.cap, table.name)
    timings = {"index_read": read_index - started, "done_read": read_done - read_index, "total": time.perf_counter() - started}
    return DocumentWork(units=units, key_class=key_class, counts=counts, uncapped=uncapped, timings=timings)
