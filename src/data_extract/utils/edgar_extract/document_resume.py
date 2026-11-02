"""EDGAR document work list: what each nightly run fetches for an EDGAR document table.

`document_worklist`: the local EDGAR index rows of the table's forms for
each key's filing scope (event CIKs and CIK windows), from the floor on, minus the accessions already stored (empty-filing
markers included; optionally only on the rows of one source); newest first and capped per run.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import cast

import pandas as pd

from src.context import Context
from src.data_extract.utils.common.identity import FilingScope, Identity
from src.data_extract.utils.common.registrant import listing_ciks, resolve_registrant_entries
from src.data_extract.utils.common.resume import DONE_TABLE, KEY_ESTABLISHED, KEY_NEW, KEY_ROWLESS, document_floor, stored_accessions
from src.data_extract.utils.edgar_extract import edgar_index
from src.data_store.schema import Table, Tables
from src.utils.universe import new_tickers

logger = logging.getLogger(__name__)


@dataclass
class DocumentWork:
    """One table's document work list: index rows to read per key (oldest first), plus what `resume-plan` reports."""

    units: dict[str, pd.DataFrame]
    key_class: dict[str, str]
    counts: dict[str, int] = field(default_factory=dict)
    uncapped: int = 0
    timings: dict[str, float] = field(default_factory=dict)
    listed: dict[str, pd.DataFrame] = field(default_factory=dict)

    @property
    def size(self) -> int:
        return sum(len(df) for df in self.units.values())


def _insider_done(context: Context, keys: Sequence[str], recent: pd.Timestamp | None) -> dict[str, set[str]]:
    """Historical ZIP units count as present; only the recent overlap requires EDGAR authority."""
    rows = context.store.load(
        Tables.insider_transactions,
        columns=["ticker", "accession_number", "source", "filing_date"],
        where={"ticker": list(keys)},
        markers=True,
        optional=True,
    )
    done: dict[str, set[str]] = {key: set() for key in keys}
    if rows is None:
        return done
    accepted = rows["source"].eq("edgar")
    if recent is not None:
        accepted |= pd.to_datetime(rows["filing_date"]).lt(recent)
    for ticker, group in rows.loc[accepted].groupby("ticker", sort=False):
        done[str(ticker)].update(group["accession_number"].astype(str))
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
    floor = (
        pd.Timestamp(cast(str, table.resume.source_start))
        if table == Tables.insider_transactions and table.resume is not None
        else document_floor(table, as_of, years_history, runtime_floor)
    )
    roster = dict(zip(keys["ticker"].astype(str), keys["cik"].astype(str), strict=True))
    scopes = {
        ticker: identity.filing_scope(ticker) if identity is not None else FilingScope.roster_only(ticker, cik) for ticker, cik in roster.items()
    }
    candidates = {ticker: listing_ciks(scope) for ticker, scope in scopes.items()}
    df_index = edgar_index.entries(context, {c for ciks in candidates.values() for c in ciks}, forms, since=floor)
    by_cik = dict(tuple(df_index.groupby("cik", sort=False))) if not df_index.empty else {}
    read_index = time.perf_counter()
    done = (
        ({key: set() for key in roster} if full else _insider_done(context, list(roster), runtime_floor))
        if table == Tables.insider_transactions
        else ({key: set() for key in roster} if full else stored_accessions(context, table, list(roster), done_scope, done_where))
    )
    read_done = time.perf_counter()
    overlap = table.resume.overlap_days if table.resume is not None else 0
    new = new_tickers(context.store, overlap, as_of) if overlap else set()
    units: dict[str, pd.DataFrame] = {}
    key_class: dict[str, str] = {}
    counts: dict[str, int] = {}
    listed: dict[str, pd.DataFrame] = {}
    for ticker in roster:
        frames = [by_cik[c] for c in candidates[ticker] if c in by_cik]
        df_listed = resolve_registrant_entries(scopes[ticker], pd.concat(frames, ignore_index=True), forms) if frames else df_index.iloc[:0]
        listed[ticker] = df_listed
        done_here = done.get(ticker, set()) & set(df_listed["accession"].astype(str))
        df_units = df_listed[~df_listed["accession"].isin(done_here)].reset_index(drop=True)
        key_class[ticker] = _key_class(ticker, new, df_listed, df_units)
        for name, n in _classify_units(df_listed, df_units, key_class[ticker], overlap).items():
            counts[name] = counts.get(name, 0) + n
        if not df_units.empty:
            units[ticker] = df_units
    units, uncapped = _cap(units, cap, table.name)
    timings = {"index_read": read_index - started, "done_read": read_done - read_index, "total": time.perf_counter() - started}
    return DocumentWork(units=units, key_class=key_class, counts=counts, uncapped=uncapped, timings=timings, listed=listed)
