"""Disposable contracts for exact source pairs (the identity commit invalidates no source proof since identity-security-grain P4)."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest
from sqlalchemy import event

from src.data_extract.utils.common.identity import load_source_identity
from src.data_extract.utils.edgar_extract.insider_store import patch_insider_source, replace_insider_source
from src.data_store.schema import Table, Tables
from src.data_store.store import DataStore
from tests.canonical_schema import create_canonical


def _inventory(store: DataStore) -> tuple[pd.DataFrame, pd.DataFrame, dict[Table, pd.DataFrame]]:
    create_canonical(
        store.engine,
        [
            Tables.universe_cik_symbol,
            Tables.symbol_tenure,
            Tables.universe_cusip,
            Tables.insider_transactions,
            Tables.insider_footnotes,
            Tables.sec_short_volume_security,
            Tables.sec_fails_to_deliver_security,
        ],
    )
    transactions = pd.DataFrame(
        {
            "accession_number": ["a", "b"],
            "security_type": ["nonderiv"] * 2,
            "row_sequence": [1, 1],
            "ticker": ["AAA"] * 2,
            "issuer_cik": ["0000000001"] * 2,
            "shares": [2.0, 3.0],
            "transaction_date": pd.to_datetime(["2026-07-01"] * 2),
            "fetched_at": pd.to_datetime(["2026-07-02"] * 2),
        }
    )
    notes = pd.DataFrame({"accession_number": ["a"], "footnote_id": ["F1"], "footnote_text": ["known original"]})
    identities = {
        Tables.universe_cik_symbol: pd.DataFrame(
            {
                "company_id": ["E1"],
                "canonical_ticker": ["AAA"],
                "cik": ["0000000001"],
                "role": ["canonical_current"],
                "symbol": ["AAA"],
                "valid_from": pd.to_datetime(["2020-01-01"]),
                "scope_changed_at": pd.to_datetime(["2026-06-01"]),
            }
        ),
        Tables.symbol_tenure: pd.DataFrame(
            {
                "symbol": ["AAA"],
                "issuer_cik": ["0000000001"],
                "valid_from": pd.to_datetime(["2020-01-01"]),
                "source": ["form345"],
                "evidence_period": ["2026q2"],
            }
        ),
        Tables.universe_cusip: pd.DataFrame(
            {
                "security_id": ["S1"],
                "source": ["ftd"],
                "source_symbol": ["AAA"],
                "valid_from": pd.to_datetime(["2020-01-01"]),
                "canonical_company": ["AAA"],
                "issuer_cik": ["0000000001"],
            }
        ),
    }
    for spec, frame in identities.items():
        store.save(spec, frame)
    store.save(Tables.insider_transactions, transactions)
    store.save(Tables.insider_footnotes, notes)
    transactions = store.load(Tables.insider_transactions, columns=store.columns(Tables.insider_transactions), markers=True)
    notes = store.load(Tables.insider_footnotes, columns=store.columns(Tables.insider_footnotes))
    assert transactions is not None and notes is not None
    return transactions, notes, load_source_identity(store)


def test_source_pair_rolls_back_payload_and_retained_proof(sqlite_store: DataStore) -> None:
    transactions, notes, _ = _inventory(sqlite_store)
    before = sqlite_store.load(Tables.insider_transactions, markers=True)
    assert before is not None
    old_notes = sqlite_store.load(Tables.insider_footnotes)
    assert old_notes is not None

    def refuse_notes(conn: Any, cursor: Any, statement: str, parameters: Any, context: Any, executemany: bool) -> None:
        if statement.startswith(f"INSERT INTO {Tables.insider_footnotes.name}"):
            raise RuntimeError("controlled notes insert failure")

    event.listen(sqlite_store.engine, "before_cursor_execute", refuse_notes)
    try:
        with pytest.raises(RuntimeError, match="controlled"):
            replace_insider_source(
                sqlite_store,
                {
                    Tables.insider_transactions: transactions.iloc[:1].assign(shares=99.0),
                    Tables.insider_footnotes: notes.assign(footnote_text="changed"),
                },
                {Tables.insider_transactions: {"accession_number": ["a"]}, Tables.insider_footnotes: {"accession_number": ["a"]}},
            )
    finally:
        event.remove(sqlite_store.engine, "before_cursor_execute", refuse_notes)
    stored_frame = sqlite_store.load(Tables.insider_transactions, markers=True)
    assert stored_frame is not None
    pd.testing.assert_frame_equal(stored_frame, before)
    stored_frame = sqlite_store.load(Tables.insider_footnotes)
    assert stored_frame is not None
    pd.testing.assert_frame_equal(stored_frame, old_notes)
    print("SANITY: a notes failure rolls back transaction rows and notes together.")


def test_legacy_projection_does_not_add_proof_columns(sqlite_store: DataStore) -> None:
    line = pd.DataFrame({"date": pd.to_datetime(["2026-07-01"]), "source_symbol": ["AAA"], "short_volume": [1.0], "total_volume": [2.0]})
    sqlite_store.save(Tables.sec_short_volume_security, line)
    rows = sqlite_store.load(Tables.sec_short_volume_security, project=True)
    assert rows is not None and rows["short_volume"].tolist() == [1.0]
    assert "source_checked_at" not in rows and "source_checked_at" not in sqlite_store.columns(Tables.sec_short_volume_security)
    print("SANITY: the projection of an as-filed source table preserves explicit observations and performs no DDL.")


def test_partial_patch_and_deleted_accession_keep_full_payload_atomic(sqlite_store: DataStore) -> None:
    transactions, notes, _ = _inventory(sqlite_store)
    patch = transactions.iloc[:1][list(Tables.insider_transactions.pk)].assign(quarter="2026q2")
    assert patch_insider_source(sqlite_store, patch, []) == 1
    actual = sqlite_store.load(Tables.insider_transactions, markers=True)
    assert actual is not None and actual["shares"].tolist() == [3.0, 2.0]
    stored_frame = sqlite_store.load(Tables.insider_footnotes)
    assert stored_frame is not None
    pd.testing.assert_frame_equal(stored_frame, notes)
    assert patch_insider_source(sqlite_store, pd.DataFrame(), ["a"]) == 1
    assert sqlite_store.load(Tables.insider_footnotes, optional=True) is None
    assert patch_insider_source(sqlite_store, pd.DataFrame(), ["b"]) == 1
    assert sqlite_store.load(Tables.insider_transactions, markers=True, optional=True) is None
    print(
        "SANITY: a quarter-only patch preserves complete economics/notes; deletion removes notes and the last actual accession leaves the issuer unknown."
    )


def test_zip_preserves_existing_edgar_payload_notes_and_freshness(sqlite_store: DataStore) -> None:
    transactions, notes, _ = _inventory(sqlite_store)
    sqlite_store.save(Tables.insider_transactions, transactions.assign(source="edgar"))
    counts = replace_insider_source(
        sqlite_store,
        {
            Tables.insider_transactions: transactions.assign(source="zip", quarter="2026q2", shares=99.0, fetched_at=pd.Timestamp("2026-07-03")),
            Tables.insider_footnotes: notes.assign(footnote_text="ZIP must not replace EDGAR note"),
        },
        {Tables.insider_transactions: {"accession_number": ["a", "b"]}, Tables.insider_footnotes: {"accession_number": ["a", "b"]}},
    )
    actual = sqlite_store.load(Tables.insider_transactions, markers=True)
    assert counts[Tables.insider_transactions] == 0 and actual is not None, "Retained EDGAR authority contributes no newly committed ZIP rows"
    assert actual["source"].eq("edgar").all() and actual["quarter"].eq("2026q2").all()
    assert actual["shares"].tolist() == [2.0, 3.0] and pd.to_datetime(actual["fetched_at"]).eq(pd.Timestamp("2026-07-02")).all()
    stored_frame = sqlite_store.load(Tables.insider_footnotes)
    assert stored_frame is not None
    pd.testing.assert_frame_equal(stored_frame, notes)
    print("SANITY: a later ZIP keeps existing EDGAR payload, footnotes and actual acquisition times while recording its quarter atomically.")


def test_owner_only_marker_preserves_other_issuer_payload_and_notes(sqlite_store: DataStore) -> None:
    transactions, notes, _ = _inventory(sqlite_store)
    marker = transactions.iloc[:1].assign(ticker="BBB", security_type="_empty", row_sequence=0, shares=float("nan"))
    replace_insider_source(
        sqlite_store,
        {Tables.insider_transactions: marker, Tables.insider_footnotes: notes.iloc[:0]},
        {Tables.insider_transactions: {"accession_number": ["a"]}, Tables.insider_footnotes: {"accession_number": ["a"]}},
    )
    actual = sqlite_store.load(Tables.insider_transactions, markers=True)
    assert actual is not None and len(actual) == 3
    assert actual.loc[actual["ticker"].eq("AAA"), "shares"].sum() == 5.0
    stored_frame = sqlite_store.load(Tables.insider_footnotes)
    assert stored_frame is not None
    pd.testing.assert_frame_equal(stored_frame, notes)
    print("SANITY: a reporting-owner empty marker preserves the actual other issuer's complete payload and notes.")


def test_owner_only_retention_is_per_accession_and_issuer(sqlite_store: DataStore) -> None:
    transactions, notes, _ = _inventory(sqlite_store)
    old = transactions.copy()
    old.loc[old["accession_number"].eq("b"), "ticker"] = "CCC"
    retained = old.iloc[1:].assign(accession_number="c")
    sqlite_store.save(Tables.insider_transactions, pd.concat([old, retained], ignore_index=True))
    incoming = pd.concat(
        [transactions.iloc[:1].assign(ticker="BBB", security_type="_empty", row_sequence=0), transactions.iloc[1:]], ignore_index=True
    )
    replace_insider_source(
        sqlite_store,
        {Tables.insider_transactions: incoming, Tables.insider_footnotes: notes.iloc[:0]},
        {Tables.insider_transactions: {"accession_number": ["a", "b"]}, Tables.insider_footnotes: {"accession_number": ["a", "b"]}},
    )
    actual = sqlite_store.load(Tables.insider_transactions, markers=True)
    assert actual is not None and len(actual) == 4
    assert len(actual.loc[actual["accession_number"].eq("a") & actual["ticker"].eq("AAA")]) == 1
    stored_frame = sqlite_store.load(Tables.insider_footnotes)
    assert stored_frame is not None
    pd.testing.assert_frame_equal(stored_frame, notes)
    print("SANITY: another accession's same ticker cannot erase an owner-only accession's actual issuer.")
