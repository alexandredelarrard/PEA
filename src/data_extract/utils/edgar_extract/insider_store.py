"""Persist fully parsed insider accessions as one existing source pair."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import pandas as pd

from src.context import Context
from src.data_store.schema import Table, Tables
from src.data_store.store import DataStore


def _source_pair(store: DataStore, transactions: pd.DataFrame, notes: pd.DataFrame) -> dict[Table, int]:
    accessions = sorted(set(transactions["accession_number"].astype(str)))
    return replace_insider_source(
        store,
        {Tables.insider_transactions: transactions, Tables.insider_footnotes: notes},
        {Tables.insider_transactions: {"accession_number": accessions}, Tables.insider_footnotes: {"accession_number": accessions}},
    )


def save_insider_source(context: Context, transactions: pd.DataFrame, notes: pd.DataFrame) -> dict[Table, int]:
    """EDGAR authority and the payload/notes pair share the store's guarded transaction."""
    return _source_pair(context.store, transactions, notes)


def _reconciled_payload(
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


def _insider_payload(
    old_transactions: pd.DataFrame, old_notes: pd.DataFrame, transactions: pd.DataFrame, notes: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The reconciled payload plus the empty-filing markers other keys hold on the same accessions: a key that
    was only an owner stays done, whatever the issuer's rows replace."""
    payload, notes = _reconciled_payload(old_transactions, old_notes, transactions, notes)
    if old_transactions.empty or transactions.empty:
        return payload, notes
    other_keys = old_transactions["security_type"].eq("_empty") & old_transactions["accession_number"].isin(transactions["accession_number"])
    markers = old_transactions.loc[other_keys & ~old_transactions["ticker"].isin(set(transactions["ticker"]))]
    return pd.concat([payload, markers], ignore_index=True).drop_duplicates(list(Tables.insider_transactions.pk), keep="first"), notes


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


def replace_insider_source(
    store: DataStore,
    frames: dict[Table, pd.DataFrame],
    scopes: dict[Table, dict[str, object] | None],
    expected_source: dict[Table, pd.DataFrame] | None = None,
) -> dict[Table, int]:
    """Commit one insider payload/notes cohort, preserving EDGAR authority."""
    transactions, notes = Tables.insider_transactions, Tables.insider_footnotes
    if set(frames) != {transactions, notes} or set(scopes) != set(frames) or any(scope == {} for scope in scopes.values()):
        raise ValueError("An insider source write needs its exact source pair and explicit scopes")
    for table, df_rows in frames.items():
        if not set(table.pk).issubset(df_rows) or df_rows.duplicated(list(table.pk)).any():
            raise ValueError(f"{table}: missing or repeated source keys")

    # Load the source before comparing expected snapshots or reconciling authority.
    with store.transaction() as bound:
        old = _load_pair(bound, scopes)
        for table, df_expected in (expected_source or {}).items():
            if table not in old or not bound.same_rows(table, old[table], df_expected):
                raise ValueError("Institutional source cohort changed before atomic restamping")
        df_transactions, df_notes = _insider_payload(old[transactions], old[notes], frames[transactions], frames[notes])

        # Publish the reconciled pair.
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


def patch_insider_source(store: DataStore, patch: pd.DataFrame, removed: Sequence[str] | pd.DataFrame) -> int:
    """Apply actual quarter/lineage patches and keyed removals in one transaction."""
    table = Tables.insider_transactions
    if set(patch.columns) - {*table.pk, "quarter", "economic_date", "lineage_role"}:
        raise ValueError("An insider patch carries only existing quarter/lineage fields")
    dropped = removed.get("accession_number", pd.Series(dtype=str)).astype(str) if isinstance(removed, pd.DataFrame) else removed
    accessions = sorted(set(patch.get("accession_number", pd.Series(dtype=str)).astype(str)) | set(dropped))
    if not accessions or not store.exists(table):
        return 0

    # Capture the actual pair before applying any patch or removal.
    with store.transaction() as bound:
        old = _load_pair(bound, {table: {"accession_number": accessions}})
        df_stored, df_notes = old[table], old[Tables.insider_footnotes]
        df_payload = df_stored.set_index(list(table.pk))
        if not patch.empty:
            df_changes = patch.set_index(list(table.pk))
            if not df_changes.index.isin(df_payload.index).all():
                raise ValueError("An insider patch cannot invent an accession row")
            df_payload.loc[df_changes.index, df_changes.columns] = df_changes
        df_payload = df_payload.reset_index()
        df_payload = df_payload.loc[~_insider_removal_mask(df_payload, removed)]
        orphaned = set(accessions) - set(df_payload["accession_number"])

        # Preserve unrelated issuer rows and remove only genuinely orphaned notes.
        bound.delete(table, {"accession_number": accessions})
        if not df_notes.empty and orphaned:
            bound.delete(Tables.insider_footnotes, {"accession_number": sorted(orphaned)})
        bound.save(table, df_payload, ignore_all_conflicts=True)
        return len(df_stored) - len(df_payload) if len(removed) else len(patch)
