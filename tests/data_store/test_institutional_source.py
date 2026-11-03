"""Disposable contracts for exact source pairs and identity-driven proof invalidation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import event

from src.data_aggregate.utils.institutionals import inputs
from src.data_aggregate.utils.institutionals import short_flow_features as short
from src.data_extract.utils.common.identity import load_source_identity
from src.data_extract.utils.edgar_extract.insider_store import patch_insider_source, replace_insider_source
from src.data_extract.utils.identity.identity_store import save_identity
from src.data_store.schema import Table, Tables
from src.data_store.store import DataStore
from src.utils.price_adjustments import ShareBasis
from tests.canonical_schema import create_canonical


def _inventory(store: DataStore) -> tuple[pd.DataFrame, pd.DataFrame, dict[Table, pd.DataFrame]]:
    create_canonical(
        store.engine,
        [
            Tables.sp500_tickers,
            Tables.entity_lineage,
            Tables.symbol_tenure,
            Tables.security_master,
            Tables.insider_transactions,
            Tables.insider_footnotes,
            Tables.short_interest,
            Tables.sec_fails_to_deliver,
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
        Tables.sp500_tickers: pd.DataFrame({"ticker": ["AAA"], "cik": ["0000000001"]}),
        Tables.entity_lineage: pd.DataFrame(
            {
                "entity_id": ["E1"],
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
        Tables.security_master: pd.DataFrame(
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


def _tape_observations(store: DataStore) -> dict[Table, pd.DataFrame]:
    dates = pd.bdate_range("2026-06-01", periods=4)
    checked = pd.to_datetime(pd.Series(["2026-06-05", "2026-06-05", None, "2026-06-05"]))
    frames = {
        Tables.short_interest: pd.DataFrame(
            {
                "ticker": "AAA",
                "date": dates,
                "short_volume": [0.0, 2.0, 0.0, 0.0],
                "total_volume": [np.nan, 4.0, 6.0, 7.0],
                "source_checked_at": checked,
            }
        ),
        Tables.sec_fails_to_deliver: pd.DataFrame(
            {
                "ticker": "AAA",
                "date": dates,
                "fails_quantity": [0.0, 3.0, 0.0, 0.0],
                "fails_value": [0.0, 30.0, 0.0, 0.0],
                "period": "202606a",
                "source_checked_at": checked,
            }
        ),
        Tables.sec_short_volume_security: pd.DataFrame(
            {"source_symbol": "AAA", "ticker": "AAA", "date": dates[[1, 3]], "short_volume": [2.0, 0.0], "total_volume": [4.0, 7.0]}
        ),
        Tables.sec_fails_to_deliver_security: pd.DataFrame(
            {"cusip": "000000001", "ticker": "AAA", "date": dates[[1, 3]], "fails_quantity": [3.0, 0.0], "price": 10.0, "period": "202606a"}
        ),
    }
    for table, frame in frames.items():
        store.save(table, frame)
    return frames


def test_identity_replacement_clears_proof_in_same_commit(sqlite_store: DataStore) -> None:
    transactions, _, identities = _inventory(sqlite_store)
    originals = _tape_observations(sqlite_store)
    raw_before = {
        table: sqlite_store.load(table, columns=[str(column) for column in originals[table]])
        for table in (Tables.sec_short_volume_security, Tables.sec_fails_to_deliver_security)
    }
    changed = identities[Tables.entity_lineage].assign(valid_from=pd.Timestamp("2019-01-01"))
    save_identity(sqlite_store, Tables.entity_lineage, changed)
    actual = sqlite_store.load(Tables.insider_transactions, markers=True)
    assert actual is not None and len(actual) == len(transactions)
    for table, quantity in ((Tables.short_interest, "short_volume"), (Tables.sec_fails_to_deliver, "fails_quantity")):
        rows = sqlite_store.load(table)
        assert rows is not None
        rows = rows.sort_values("date").reset_index(drop=True)
        assert pd.isna(rows.loc[0, quantity]) and rows.loc[1, quantity] == originals[table].loc[1, quantity] and rows.loc[2, quantity] == 0.0
        assert rows["source_checked_at"].isna().all()
        if table == Tables.sec_fails_to_deliver:
            assert pd.isna(rows.loc[0, "fails_value"]) and pd.isna(rows.loc[3, quantity]) and pd.isna(rows.loc[3, "fails_value"])
        else:
            assert rows.loc[3, quantity] == 0.0, "a real FINRA zero with a reported total remains an observation"
    for table in (Tables.sec_short_volume_security, Tables.sec_fails_to_deliver_security):
        rows = sqlite_store.load(table, columns=[str(column) for column in originals[table]])
        expected = raw_before[table]
        assert rows is not None and expected is not None
        pd.testing.assert_frame_equal(rows, expected)
    print(
        "SANITY: semantic identity changes clear proof and checked quiet operands atomically; positives, legacy zeros, real FINRA zero/total pairs and raw payload survive."
    )


@pytest.mark.parametrize("metric", ["finra", "ftd"])
def test_invalidated_quiet_rows_cannot_become_legacy_cube_observations(sqlite_store: DataStore, metric: str) -> None:
    _, _, identities = _inventory(sqlite_store)
    _tape_observations(sqlite_store)
    save_identity(sqlite_store, Tables.entity_lineage, identities[Tables.entity_lineage].assign(valid_from=pd.Timestamp("2019-01-01")))
    rows = sqlite_store.load(Tables.short_interest, project=True)
    assert rows is not None
    dates = pd.bdate_range("2026-06-01", periods=4)
    empty = pd.DataFrame(columns=["ticker", "date", "value"])
    basis = inputs.QuantityBasis(ShareBasis(empty, empty, empty, {}), dates[-1])
    prepared = inputs.short_quantity_history(rows.sort_values("date").reset_index(drop=True), basis)
    assert prepared is not None
    assert prepared.loc[1, "short_volume"] == 2.0 and prepared.loc[2, "short_volume"] == 0.0
    fails, coverage = inputs.load_fails(sqlite_store, ["AAA"])
    assert fails is not None and coverage is not None
    operands, _ = short._settlement_fails(fails, coverage, dates, pd.Index(["AAA"]), None)
    assert operands.loc[dates[1], "AAA"] == 3.0 and operands.loc[dates[2], "AAA"] == 0.0
    if metric == "finra":
        assert pd.isna(prepared.loc[0, "short_volume"]), "invalidated inferred FINRA zero reached the cube as an explicit operand"
    else:
        assert pd.isna(operands.loc[dates[0], "AAA"]) and pd.isna(operands.loc[dates[3], "AAA"]), (
            "invalidated checked FTD zero reached the cube as an explicit operand"
        )
    print(
        "SANITY: actual cube preparation consumes invalidated checked quiet rows as unknown, while explicit positive and genuine legacy-zero operands remain usable."
    )


@pytest.mark.parametrize("failed_table", [Tables.short_interest, Tables.sec_fails_to_deliver], ids=["finra", "ftd"])
def test_identity_invalidation_rolls_back_quiet_payloads_and_proof(sqlite_store: DataStore, failed_table: Table) -> None:
    _, _, identities = _inventory(sqlite_store)
    originals = _tape_observations(sqlite_store)
    originals[Tables.entity_lineage] = identities[Tables.entity_lineage]
    before = {table: sqlite_store.load(table, columns=[str(column) for column in frame]) for table, frame in originals.items()}

    def refuse_stamp(_conn: Any, _cursor: Any, statement: str, _parameters: Any, _context: Any, _executemany: bool) -> None:
        if statement.startswith(f"UPDATE {failed_table.name} SET source_checked_at"):
            raise RuntimeError("controlled failure after quiet invalidation")

    event.listen(sqlite_store.engine, "before_cursor_execute", refuse_stamp)
    try:
        with pytest.raises(RuntimeError, match="controlled failure"):
            save_identity(sqlite_store, Tables.entity_lineage, identities[Tables.entity_lineage].assign(valid_from=pd.Timestamp("2019-01-01")))
    finally:
        event.remove(sqlite_store.engine, "before_cursor_execute", refuse_stamp)
    for table, expected in before.items():
        actual = sqlite_store.load(table, columns=[str(column) for column in originals[table]])
        assert expected is not None and actual is not None
        pd.testing.assert_frame_equal(actual, expected)
    print("SANITY: a failure after quiet-operand invalidation rolls back identity, source values, raw payload and proof in the same transaction.")


def test_legacy_projection_does_not_add_proof_columns(sqlite_store: DataStore) -> None:
    sqlite_store.save(
        Tables.short_interest, pd.DataFrame({"ticker": ["AAA"], "date": pd.to_datetime(["2026-07-01"]), "short_volume": [1.0], "total_volume": [2.0]})
    )
    rows = sqlite_store.load(Tables.short_interest, project=True)
    assert rows is not None and rows["short_volume"].tolist() == [1.0]
    assert "source_checked_at" not in rows and "source_checked_at" not in sqlite_store.columns(Tables.short_interest)
    print("SANITY: physically absent legacy proof columns preserve explicit observations and projection performs no DDL.")


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
