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
from src.data_extract.utils.common.identity_store import SOURCE_LOCK, assert_source_identity, invalidate_source_proof, load_source_identity
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
    with DataStore.temporary([Tables.insider_transactions, Tables.insider_footnotes]) as staged:
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
    return replace_insider_source(
        store,
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
                insider_inventory_snapshot(context.store, ticker, expected_accessions)
            return
        transactions, notes = insider_inventory_snapshot(context.store, ticker, expected_accessions)
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
            certify_insider_inventory(context.store, ticker, through, transactions, expected_notes, expected_identity)
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


def _insider_payload(
    old_transactions: pd.DataFrame, old_notes: pd.DataFrame, transactions: pd.DataFrame, notes: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Preserve current EDGAR authority and another issuer's actual owner-only filing."""
    if old_transactions.empty or transactions.empty:
        return transactions, notes
    if "quarter" in old_transactions:
        quarter_of = old_transactions.groupby("accession_number")["quarter"].first()
        quarter = transactions.get("quarter", pd.Series(index=transactions.index, dtype=str))
        transactions = transactions.assign(quarter=quarter.fillna(transactions["accession_number"].map(quarter_of)))
    markers = transactions["security_type"].eq("_empty")
    marker_only = set(transactions.loc[markers, "accession_number"]) - set(transactions.loc[~markers, "accession_number"])
    incoming_keys = pd.MultiIndex.from_frame(transactions.loc[markers, ["accession_number", "ticker"]])
    stored_keys = pd.MultiIndex.from_frame(old_transactions[["accession_number", "ticker"]])
    retained = old_transactions.loc[old_transactions["accession_number"].isin(marker_only) & ~stored_keys.isin(incoming_keys)]
    source = transactions.get("source", pd.Series(index=transactions.index, dtype=str))
    zipped = transactions.loc[source.eq("zip")]
    if not zipped.empty and "source" in old_transactions:
        edgar = old_transactions.loc[old_transactions["source"].eq("edgar") & old_transactions["accession_number"].isin(zipped["accession_number"])]
        if not edgar.empty:
            quarter_of = zipped.groupby("accession_number")["quarter"].first()
            edgar = edgar.assign(quarter=edgar["accession_number"].map(quarter_of))
            transactions = transactions.loc[~transactions["accession_number"].isin(edgar["accession_number"])]
            retained = pd.concat([retained, edgar], ignore_index=True)
    if retained.empty:
        return transactions, notes
    preserved = set(retained["accession_number"])
    payload = pd.concat([transactions, retained], ignore_index=True).drop_duplicates(list(Tables.insider_transactions.pk), keep="last")
    notes = pd.concat(
        [notes.loc[~notes["accession_number"].isin(preserved)], old_notes.loc[old_notes["accession_number"].isin(preserved)]], ignore_index=True
    )
    return payload, notes


def _insider_removal_mask(payload: pd.DataFrame, removed: Sequence[str] | pd.DataFrame) -> pd.Series:
    """Issuer-scoped keyed deletions cannot erase another retained owner of the accession."""
    if not isinstance(removed, pd.DataFrame):
        return payload["accession_number"].isin(removed)
    if removed.empty:
        return pd.Series(False, index=payload.index)
    allowed = {*Tables.insider_transactions.pk, "ticker", "issuer_cik"}
    if not set(Tables.insider_transactions.pk).issubset(removed) or set(removed) - allowed:
        raise ValueError("Insider deletion requires declared row keys and optional actual issuer identity")
    columns = list(removed.columns)
    return pd.Series(pd.MultiIndex.from_frame(payload[columns]).isin(pd.MultiIndex.from_frame(removed[columns])), index=payload.index)


def _insider_scope_changes(ticker: str, transactions: pd.DataFrame, identity: dict[Table, pd.DataFrame]) -> pd.Series:
    changes = []
    ciks = set(transactions.get("issuer_cik", pd.Series(dtype=str)).dropna().astype(str).str.zfill(10))
    for spec, company in ((Tables.entity_lineage, "canonical_ticker"), (Tables.security_master, "canonical_company")):
        frame = identity[spec]
        if "scope_changed_at" not in frame:
            continue
        applicable = frame[company].eq(ticker) if company in frame else pd.Series(False, index=frame.index)
        cik_column = "cik" if spec == Tables.entity_lineage else "issuer_cik"
        if cik_column in frame:
            applicable |= frame[cik_column].astype("string").str.zfill(10).isin(ciks)
        changes.append(pd.to_datetime(frame.loc[applicable, "scope_changed_at"], errors="coerce"))
    return pd.concat(changes, ignore_index=True) if changes else pd.Series(dtype="datetime64[ns]")


def _load_pair(store: DataStore, scopes: dict[Table, dict[str, object] | None]) -> dict[Table, pd.DataFrame]:
    """Load the actual source pair once, including markers, before reconciliation."""
    transactions, notes = Tables.insider_transactions, Tables.insider_footnotes
    transaction_columns, note_columns = store.columns(transactions), store.columns(notes)
    df_transactions = store.load(transactions, columns=transaction_columns, where=scopes.get(transactions), markers=True, optional=True)
    if df_transactions is None:
        df_transactions = pd.DataFrame(columns=transaction_columns)
    accessions = df_transactions.get("accession_number", pd.Series(dtype=str)).dropna().unique().tolist()
    df_notes = store.load(notes, columns=note_columns, where=scopes.get(notes, {"accession_number": accessions}), optional=True)
    return {transactions: df_transactions, notes: df_notes if df_notes is not None else pd.DataFrame(columns=note_columns)}


def _same_source_rows(store: DataStore, table: Table, actual: pd.DataFrame, expected: pd.DataFrame) -> bool:
    """Compare the actual source fields while excluding the proof being established."""
    excluded = ["inventory_checked_through"] if table == Tables.insider_transactions else []
    return store.same_rows(table, actual.drop(columns=excluded, errors="ignore"), expected.drop(columns=excluded, errors="ignore"))


def replace_insider_source(
    store: DataStore,
    frames: dict[Table, pd.DataFrame],
    scopes: dict[Table, dict[str, object] | None],
    expected_identity: dict[Table, pd.DataFrame] | None = None,
    expected_source: dict[Table, pd.DataFrame] | None = None,
) -> dict[Table, int]:
    """Commit one insider payload/notes cohort, preserving EDGAR authority and invalidating proof."""
    transactions, notes = Tables.insider_transactions, Tables.insider_footnotes
    if set(frames) != {transactions, notes} or set(scopes) != set(frames) or any(scope == {} for scope in scopes.values()):
        raise ValueError("An insider source write needs its exact source pair and explicit scopes")
    for table, df_rows in frames.items():
        if not set(table.pk).issubset(df_rows) or df_rows.duplicated(list(table.pk)).any():
            raise ValueError(f"{table}: missing or repeated source keys")

    # Load source and identity before comparing expected snapshots or reconciling authority.
    with store.transaction(SOURCE_LOCK) as bound:
        old = _load_pair(bound, scopes)
        identity = load_source_identity(bound) if expected_identity is not None else None
        if expected_identity is not None and identity is not None:
            assert_source_identity(bound, identity, expected_identity)
        for table, df_expected in (expected_source or {}).items():
            if table not in old or not _same_source_rows(bound, table, old[table], df_expected):
                raise ValueError("Institutional source cohort changed before atomic restamping")
        df_transactions, df_notes = _insider_payload(old[transactions], old[notes], frames[transactions], frames[notes])
        df_transactions = df_transactions.assign(inventory_checked_through=pd.Series(pd.NaT, index=df_transactions.index, dtype="datetime64[ns]"))
        tickers = set(old[transactions].get("ticker", pd.Series(dtype=str)).dropna().astype(str))
        tickers.update(df_transactions.get("ticker", pd.Series(dtype=str)).dropna().astype(str))

        # Publish the reconciled pair and clear every retained issuer certificate together.
        invalidate_source_proof(bound, tickers)
        for table in scopes:
            if scopes[table] is None:
                continue
            bound.delete(table, cast(dict[str, object], scopes[table]))
        prepared = {transactions: df_transactions, notes: df_notes}
        counts = {}
        for table, df_rows in prepared.items():
            counts[table] = bound.replace(table, df_rows) if scopes[table] is None else bound.save(table, df_rows, ignore_all_conflicts=True)
        supplied = frames[transactions]
        if "source" in supplied and supplied["source"].eq("zip").all():
            keys = pd.MultiIndex.from_frame(supplied[list(transactions.pk)])
            incoming = pd.MultiIndex.from_frame(df_transactions[list(transactions.pk)]).isin(keys)
            counts[transactions] = int((incoming & df_transactions["source"].eq("zip")).sum())
        return counts


def patch_insider_source(
    store: DataStore,
    patch: pd.DataFrame,
    removed: Sequence[str] | pd.DataFrame,
    expected_identity: dict[Table, pd.DataFrame] | None = None,
) -> int:
    """Apply actual quarter/lineage patches and keyed removals under the source guard."""
    table = Tables.insider_transactions
    if set(patch.columns) - {*table.pk, "quarter", "economic_date", "lineage_role"}:
        raise ValueError("An insider patch carries only existing quarter/lineage fields")
    dropped = removed.get("accession_number", pd.Series(dtype=str)).astype(str) if isinstance(removed, pd.DataFrame) else removed
    accessions = sorted(set(patch.get("accession_number", pd.Series(dtype=str)).astype(str)) | set(dropped))
    if not accessions or not store.exists(table):
        return 0

    # Capture the actual pair and identity before applying any patch or removal.
    with store.transaction(SOURCE_LOCK) as bound:
        old = _load_pair(bound, {table: {"accession_number": accessions}})
        identity = load_source_identity(bound) if expected_identity is not None else None
        if expected_identity is not None and identity is not None:
            assert_source_identity(bound, identity, expected_identity)
        df_stored, df_notes = old[table], old[Tables.insider_footnotes]
        df_payload = df_stored.set_index(list(table.pk))
        if not patch.empty:
            df_changes = patch.set_index(list(table.pk))
            if not df_changes.index.isin(df_payload.index).all():
                raise ValueError("An insider patch cannot invent an accession row")
            df_payload.loc[df_changes.index, df_changes.columns] = df_changes
        df_payload = df_payload.reset_index()
        df_payload = df_payload.loc[~_insider_removal_mask(df_payload, removed)]
        tickers = set(df_stored["ticker"].dropna().astype(str)) | set(df_payload["ticker"].dropna().astype(str))
        orphaned = set(accessions) - set(df_payload["accession_number"])
        if "inventory_checked_through" in df_payload:
            df_payload["inventory_checked_through"] = pd.NaT

        # Preserve unrelated issuer rows and remove only genuinely orphaned notes.
        invalidate_source_proof(bound, tickers)
        bound.delete(table, {"accession_number": accessions})
        if not df_notes.empty and orphaned:
            bound.delete(Tables.insider_footnotes, {"accession_number": sorted(orphaned)})
        bound.save(table, df_payload, ignore_all_conflicts=True)
        return len(df_stored) - len(df_payload) if len(removed) else len(patch)


def insider_inventory_snapshot(store: DataStore, ticker: str, expected_accessions: Sequence[str] | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Capture the guarded joint cohort, clearing retained proof when inventory changed."""
    with store.transaction(SOURCE_LOCK) as bound:
        pair = _load_pair(bound, {Tables.insider_transactions: {"ticker": ticker}})
        df_transactions = pair[Tables.insider_transactions]
        accessions = set(df_transactions.get("accession_number", pd.Series(dtype=str)).astype(str))
        if expected_accessions is not None and accessions != set(expected_accessions):
            invalidate_source_proof(bound, {ticker})
        return df_transactions, pair[Tables.insider_footnotes]


def certify_insider_inventory(
    store: DataStore,
    ticker: str,
    through: pd.Timestamp,
    expected_transactions: pd.DataFrame,
    expected_footnotes: pd.DataFrame,
    expected_identity: dict[Table, pd.DataFrame],
) -> int:
    """Certify only the exact freshly reconciled source/notes/identity cohort."""
    identities = {Tables.sp500_tickers, Tables.entity_lineage, Tables.symbol_tenure, Tables.security_master}
    if set(expected_identity) != identities or expected_transactions.empty:
        raise ValueError("Certification requires an actual accession cohort and the closed identity cohort")

    # Read each current cohort before payload, footnote and freshness checks.
    with store.transaction(SOURCE_LOCK) as bound:
        pair = _load_pair(bound, {Tables.insider_transactions: {"ticker": ticker}})
        identity = load_source_identity(bound)
        df_actual, df_notes = pair[Tables.insider_transactions], pair[Tables.insider_footnotes]
        if "inventory_checked_through" not in df_actual or not _same_source_rows(
            bound, Tables.insider_transactions, df_actual, expected_transactions
        ):
            raise ValueError("Insider source cohort changed before certification")
        if not _same_source_rows(bound, Tables.insider_footnotes, df_notes, expected_footnotes):
            raise ValueError("Insider footnote cohort changed before certification")
        assert_source_identity(bound, identity, expected_identity)
        fetched = pd.to_datetime(df_actual.get("fetched_at", pd.Series(pd.NaT, index=df_actual.index)), errors="coerce")
        changes = _insider_scope_changes(ticker, df_actual, expected_identity)
        if fetched.isna().any() or changes.notna().any() and fetched.lt(changes.max()).any():
            raise ValueError("Every insider accession must be freshly reconciled for its dated identity scope")
        return bound.update(Tables.insider_transactions, {"inventory_checked_through": through}, {"ticker": ticker})
