"""Persist fully parsed insider accessions as one existing source pair."""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator, Sequence
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from typing import Protocol, cast

import pandas as pd

from src.context import Context
from src.data_store.schema import Table, Tables
from src.data_store.store import DataStore


class _InsiderRuntime(Protocol):
    _insider_source_store: DataStore
    _insider_source_lock: AbstractContextManager[object]
    _insider_source_captured: set[str]
    _insider_source_max_cohort: int
    _insider_source_identity: dict[Table, pd.DataFrame] | None


def _runtime(context: Context) -> _InsiderRuntime:
    """Type the existing owner-created fields; their lifetime remains inside insider_source_run."""
    return cast(_InsiderRuntime, context)


@contextmanager
def insider_source_run(context: Context) -> Iterator[None]:
    """Bound actual reconciled payloads to one explicitly isolated, disposable source pair."""
    runtime = _runtime(context)
    if getattr(context, "_insider_source_store", None) is not None:
        yield
        return
    started = time.perf_counter()
    with DataStore.temporary_insider_source() as staged:
        runtime._insider_source_store = staged
        runtime._insider_source_lock = threading.RLock()
        runtime._insider_source_captured = set()
        runtime._insider_source_max_cohort = 0
        runtime._insider_source_identity = None
        try:
            yield
        finally:
            database = Path(str(staged.engine.url.database))
            context.log.info(
                "insider reconciliation staging: %d rows, %d bytes, largest issuer %d rows, %.2fs",
                staged.row_count(Tables.insider_transactions),
                database.stat().st_size if database.exists() else 0,
                runtime._insider_source_max_cohort,
                time.perf_counter() - started,
            )
            del runtime._insider_source_store
            del runtime._insider_source_lock
            del runtime._insider_source_captured
            del runtime._insider_source_max_cohort
            del runtime._insider_source_identity


def _source_pair(
    store: DataStore, transactions: pd.DataFrame, notes: pd.DataFrame, identity: dict[Table, pd.DataFrame] | None = None
) -> dict[Table, int]:
    accessions = sorted(set(transactions["accession_number"].astype(str)))
    return store.replace_institutional_source(
        {Tables.insider_transactions: transactions, Tables.insider_footnotes: notes},
        {Tables.insider_transactions: {"accession_number": accessions}, Tables.insider_footnotes: {"accession_number": accessions}},
        identity,
    )


def bind_insider_identity(context: Context, identity: dict[Table, pd.DataFrame]) -> None:
    """Freeze the actual parser resolver for this source lifetime; another resolver requires a retry."""
    runtime = _runtime(context)
    if getattr(context, "_insider_source_store", None) is None:
        return
    with runtime._insider_source_lock:
        previous = runtime._insider_source_identity
        if previous is not None and not _same_insider_identity(previous, identity):
            raise ValueError("Insider parser identity cohort changed during the source run")
        if previous is None:
            runtime._insider_source_identity = {table: frame.copy(deep=True) for table, frame in identity.items()}


def _same_insider_identity(left: dict[Table, pd.DataFrame], right: dict[Table, pd.DataFrame]) -> bool:
    if set(left) != set(right):
        return False
    for table, frame in left.items():
        other = right[table]
        if set(frame) != set(other):
            return False
        keys = [key for key in table.pk if key in frame]
        first = frame.sort_values(keys) if keys else frame
        second = other.sort_values(keys) if keys else other
        if not first.reset_index(drop=True).equals(second.reindex(columns=first.columns).reset_index(drop=True)):
            return False
    return True


def capture_insider_cohort(context: Context, ticker: str, expected_accessions: Sequence[str] | None) -> None:
    """Reuse only a consistently captured prior joint certificate, never legacy stored payload."""
    runtime = _runtime(context)
    staged = getattr(context, "_insider_source_store", None)
    if staged is None:
        return
    with runtime._insider_source_lock:
        if ticker in runtime._insider_source_captured:
            if expected_accessions is not None:
                context.store.insider_inventory_snapshot(ticker, expected_accessions)
            return
        transactions, notes = context.store.insider_inventory_snapshot(ticker, expected_accessions)
        runtime._insider_source_captured.add(ticker)
        proof = pd.to_datetime(transactions.get("inventory_checked_through", pd.Series(dtype="datetime64[ns]")), errors="coerce")
        if transactions.empty or len(proof) != len(transactions) or proof.isna().any() or proof.nunique() != 1:
            return
        present = set(staged.distinct(Tables.insider_transactions, "accession_number"))
        missing = set(transactions["accession_number"].astype(str)) - present
        payload = transactions.loc[transactions["accession_number"].isin(missing)]
        if not payload.empty:
            _source_pair(staged, payload, notes.loc[notes["accession_number"].isin(missing)])


def certify_insider_source(context: Context, ticker: str, through: pd.Timestamp, accessions: Sequence[str]) -> bool:
    """Compare the actual source/prior-joint cohort against the current payload in one guarded stamp."""
    runtime = _runtime(context)
    staged = getattr(context, "_insider_source_store", None)
    if staged is None:
        return False
    capture_insider_cohort(context, ticker, accessions)
    with runtime._insider_source_lock:
        transactions = staged.load(
            Tables.insider_transactions, columns=staged.columns(Tables.insider_transactions), where={"ticker": ticker}, markers=True, optional=True
        )
        if transactions is None or set(transactions["accession_number"].astype(str)) != set(accessions):
            return False
        notes = staged.load(
            Tables.insider_footnotes, columns=staged.columns(Tables.insider_footnotes), where={"accession_number": list(accessions)}, optional=True
        )
        runtime._insider_source_max_cohort = max(runtime._insider_source_max_cohort, len(transactions))
        expected_identity = runtime._insider_source_identity
        if expected_identity is None:
            return False
        try:
            expected_notes = notes if notes is not None else pd.DataFrame(columns=staged.columns(Tables.insider_footnotes))
            context.store.certify_insider_inventory(ticker, through, transactions, expected_notes, expected_identity)
            return True
        except ValueError as exc:
            context.log.warning("insider inventory: %s not certified (%s)", ticker, exc)
            return False


def save_insider_source(context: Context, transactions: pd.DataFrame, notes: pd.DataFrame, identity: dict[Table, pd.DataFrame]) -> dict[Table, int]:
    """EDGAR authority, payload/notes and retained proof share the store's guarded transaction."""
    runtime = _runtime(context)
    staged = getattr(context, "_insider_source_store", None)
    bind_insider_identity(context, identity)
    if staged is not None:
        for ticker in transactions["ticker"].dropna().astype(str).unique():
            capture_insider_cohort(context, ticker, None)
    result = _source_pair(context.store, transactions, notes, identity)
    if staged is not None:
        with runtime._insider_source_lock:
            _source_pair(staged, transactions, notes)
    return result
