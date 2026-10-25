"""Disposable contracts for exact source pairs and joint insider inventory certification."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from time import perf_counter
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine, event, inspect

from src.context import Context
from src.data_aggregate.utils.institutionals import inputs
from src.data_aggregate.utils.institutionals import short_flow_features as short
from src.data_extract.utils.common.identity import load_source_identity
from src.data_extract.utils.institutionals.insider_store import certify_insider_source, insider_source_run, save_insider_source
from src.data_store.schema import Table, Tables
from src.data_store.store import DataStore
from src.utils.price_adjustments import ShareBasis


def _inventory(store: DataStore) -> tuple[pd.DataFrame, pd.DataFrame, dict[Table, pd.DataFrame]]:
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
            "inventory_checked_through": pd.to_datetime(["2026-07-01"] * 2),
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
    return transactions, notes, identities


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
            sqlite_store.replace_institutional_source(
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
    print("SANITY: a notes failure rolls back transaction rows, notes and retained issuer inventory proof together.")


@pytest.mark.parametrize("changed", ["shares", "footnote"])
def test_same_accession_mutation_refuses_old_certification(sqlite_store: DataStore, changed: str) -> None:
    transactions, notes, identities = _inventory(sqlite_store)
    replacement = transactions.iloc[:1].copy()
    replacement_notes = notes.copy()
    if changed == "shares":
        replacement["shares"] = 99.0
    else:
        replacement_notes["footnote_text"] = "changed same accession"
    sqlite_store.replace_institutional_source(
        {Tables.insider_transactions: replacement, Tables.insider_footnotes: replacement_notes},
        {Tables.insider_transactions: {"accession_number": ["a"]}, Tables.insider_footnotes: {"accession_number": ["a"]}},
    )
    with pytest.raises(ValueError, match="cohort changed"):
        sqlite_store.certify_insider_inventory("AAA", pd.Timestamp("2026-07-03"), transactions, notes, identities)
    actual = sqlite_store.load(Tables.insider_transactions, markers=True)
    assert actual is not None and actual["inventory_checked_through"].isna().all()
    print("SANITY: same-accession payload/footnote changes invalidate every retained issuer row and refuse stale certification.")


def test_every_accession_must_be_fresh_before_certification(sqlite_store: DataStore) -> None:
    transactions, notes, identities = _inventory(sqlite_store)
    transactions.loc[0, "fetched_at"] = pd.Timestamp("2026-05-01")
    sqlite_store.save(Tables.insider_transactions, transactions)
    with pytest.raises(ValueError, match="Every insider accession"):
        sqlite_store.certify_insider_inventory("AAA", pd.Timestamp("2026-07-03"), transactions, notes, identities)
    print("SANITY: a recent maximum fetched_at cannot hide an older accession behind the identity scope change.")


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
    sqlite_store.replace_institutional_identity(Tables.entity_lineage, changed)
    actual = sqlite_store.load(Tables.insider_transactions, markers=True)
    assert actual is not None and len(actual) == len(transactions) and actual["inventory_checked_through"].isna().all()
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
    sqlite_store.replace_institutional_identity(
        Tables.entity_lineage, identities[Tables.entity_lineage].assign(valid_from=pd.Timestamp("2019-01-01"))
    )
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
            sqlite_store.replace_institutional_identity(
                Tables.entity_lineage, identities[Tables.entity_lineage].assign(valid_from=pd.Timestamp("2019-01-01"))
            )
    finally:
        event.remove(sqlite_store.engine, "before_cursor_execute", refuse_stamp)
    for table, expected in before.items():
        actual = sqlite_store.load(table, columns=[str(column) for column in originals[table]])
        assert expected is not None and actual is not None
        pd.testing.assert_frame_equal(actual, expected)
    print("SANITY: a failure after quiet-operand invalidation rolls back identity, source values, raw payload and proof in the same transaction.")


def test_exact_reconciliation_can_certify_all_actual_rows(sqlite_store: DataStore) -> None:
    transactions, notes, identities = _inventory(sqlite_store)
    assert sqlite_store.certify_insider_inventory("AAA", pd.Timestamp("2026-07-03"), transactions, notes, identities) == 2
    rows = sqlite_store.load(Tables.insider_transactions, markers=True)
    assert rows is not None and pd.to_datetime(rows["inventory_checked_through"]).eq(pd.Timestamp("2026-07-03")).all()
    print("SANITY: exact payload, notes, identity and every-accession freshness jointly stamp the real issuer cohort.")


def test_concurrent_same_key_mutation_cannot_leave_certification(tmp_path: Path) -> None:
    store = DataStore(create_engine(f"sqlite:///{(tmp_path / 'race.sqlite').as_posix()}"))
    transactions, notes, identities = _inventory(store)
    compared, mutate_started, release = Event(), Event(), Event()

    def pause_stamp(conn: Any, cursor: Any, statement: str, parameters: Any, context: Any, executemany: bool) -> None:
        if statement.startswith(f"UPDATE {Tables.insider_transactions.name} SET inventory_checked_through") and parameters[0] is not None:
            compared.set()
            assert release.wait(10), "Controlled certification interleaving timed out"

    def mutate() -> None:
        mutate_started.set()
        store.replace_institutional_source(
            {Tables.insider_transactions: transactions.iloc[:1].assign(shares=77.0), Tables.insider_footnotes: notes.assign(footnote_text="raced")},
            {Tables.insider_transactions: {"accession_number": ["a"]}, Tables.insider_footnotes: {"accession_number": ["a"]}},
        )

    event.listen(store.engine, "before_cursor_execute", pause_stamp)
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            certification = executor.submit(store.certify_insider_inventory, "AAA", pd.Timestamp("2026-07-03"), transactions, notes, identities)
            assert compared.wait(10)
            mutation = executor.submit(mutate)
            assert mutate_started.wait(10)
            assert not mutation.done(), "Mutation must wait while compare-and-stamp holds the write guard"
            release.set()
            assert certification.result(timeout=10) == 2
            mutation.result(timeout=10)
    finally:
        release.set()
        event.remove(store.engine, "before_cursor_execute", pause_stamp)
    rows = store.load(Tables.insider_transactions, markers=True)
    assert rows is not None and rows["inventory_checked_through"].isna().all()
    assert rows.loc[rows["accession_number"].eq("a"), "shares"].iloc[0] == 77.0
    store.engine.dispose()
    print("SANITY: a mutation between comparison and stamp waits, then atomically invalidates the stamp after changing the same key/notes.")


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
    assert sqlite_store.patch_insider_source(patch, []) == 1
    actual = sqlite_store.load(Tables.insider_transactions, markers=True)
    assert actual is not None and actual["shares"].tolist() == [3.0, 2.0]
    assert actual["inventory_checked_through"].isna().all()
    stored_frame = sqlite_store.load(Tables.insider_footnotes)
    assert stored_frame is not None
    pd.testing.assert_frame_equal(stored_frame, notes)
    assert sqlite_store.patch_insider_source(pd.DataFrame(), ["a"]) == 1
    assert sqlite_store.load(Tables.insider_footnotes, optional=True) is None
    assert sqlite_store.patch_insider_source(pd.DataFrame(), ["b"]) == 1
    assert sqlite_store.load(Tables.insider_transactions, markers=True, optional=True) is None
    print(
        "SANITY: a quarter-only patch preserves complete economics/notes; deletion removes notes and the last actual accession leaves the issuer unknown."
    )


def test_zip_preserves_existing_edgar_payload_notes_and_freshness(sqlite_store: DataStore) -> None:
    transactions, notes, _ = _inventory(sqlite_store)
    sqlite_store.save(Tables.insider_transactions, transactions.assign(source="edgar"))
    counts = sqlite_store.replace_institutional_source(
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
    sqlite_store.replace_institutional_source(
        {Tables.insider_transactions: marker, Tables.insider_footnotes: notes.iloc[:0]},
        {Tables.insider_transactions: {"accession_number": ["a"]}, Tables.insider_footnotes: {"accession_number": ["a"]}},
    )
    actual = sqlite_store.load(Tables.insider_transactions, markers=True)
    assert actual is not None and len(actual) == 3
    assert actual.loc[actual["ticker"].eq("AAA"), "shares"].sum() == 5.0
    stored_frame = sqlite_store.load(Tables.insider_footnotes)
    assert stored_frame is not None
    pd.testing.assert_frame_equal(stored_frame, notes)
    assert actual["inventory_checked_through"].isna().all()
    print(
        "SANITY: a reporting-owner empty marker preserves the actual other issuer's complete payload and notes, with both issuer proofs invalidated."
    )


def test_owner_only_retention_is_per_accession_and_issuer(sqlite_store: DataStore) -> None:
    transactions, notes, _ = _inventory(sqlite_store)
    old = transactions.copy()
    old.loc[old["accession_number"].eq("b"), "ticker"] = "CCC"
    retained = old.iloc[1:].assign(accession_number="c")
    sqlite_store.save(Tables.insider_transactions, pd.concat([old, retained], ignore_index=True))
    incoming = pd.concat(
        [transactions.iloc[:1].assign(ticker="BBB", security_type="_empty", row_sequence=0), transactions.iloc[1:]], ignore_index=True
    )
    sqlite_store.replace_institutional_source(
        {Tables.insider_transactions: incoming, Tables.insider_footnotes: notes.iloc[:0]},
        {Tables.insider_transactions: {"accession_number": ["a", "b"]}, Tables.insider_footnotes: {"accession_number": ["a", "b"]}},
    )
    actual = sqlite_store.load(Tables.insider_transactions, markers=True)
    assert actual is not None and len(actual) == 4
    assert len(actual.loc[actual["accession_number"].eq("a") & actual["ticker"].eq("AAA")]) == 1
    assert actual["inventory_checked_through"].isna().all()
    stored_frame = sqlite_store.load(Tables.insider_footnotes)
    assert stored_frame is not None
    pd.testing.assert_frame_equal(stored_frame, notes)
    print("SANITY: another accession's same ticker cannot erase an owner-only accession's actual issuer; old/new retained issuers both invalidate.")


def test_master_scope_change_requires_each_accession_freshness(sqlite_store: DataStore) -> None:
    transactions, notes, identities = _inventory(sqlite_store)
    master = identities[Tables.security_master].assign(scope_changed_at=pd.Timestamp("2026-07-03"))
    sqlite_store.save(Tables.security_master, master)
    identities[Tables.security_master] = master
    with pytest.raises(ValueError, match="Every insider accession"):
        sqlite_store.certify_insider_inventory("AAA", pd.Timestamp("2026-07-04"), transactions, notes, identities)
    print("SANITY: unchanged accession sets and lineage stamps cannot bypass a newer applicable security-master scope change.")


def test_temporary_source_roundtrip_and_cleanup_preserve_exact_cohort() -> None:
    staged_path: Path | None = None
    started = perf_counter()
    staging_bytes = 0
    with pytest.raises(RuntimeError, match="controlled context exit"):
        with DataStore.temporary_insider_source() as stage:
            staged_path = Path(str(stage.engine.url.database))
            transactions = pd.DataFrame(
                {
                    "accession_number": ["a", "b"],
                    "security_type": ["nonderiv", "deriv_holding"],
                    "row_sequence": [1, 1],
                    "ticker": ["AAA", "AAA"],
                    "issuer_cik": ["0000000001", "0000000001"],
                    "shares": [9007199254740994.0, np.nan],
                    "shares_owned_after": [np.nan, 1.0000000000000002],
                    "filing_date": pd.to_datetime(["2026-07-01", "2026-07-02"]),
                    "fetched_at": pd.to_datetime(["2026-07-03 01:02:03.123456"] * 2),
                }
            )
            notes = pd.DataFrame({"accession_number": ["a", "b"], "footnote_id": ["F1", "F2"], "footnote_text": ['Exact "quoted"\nnotes é', None]})
            stage.replace_institutional_source(
                {Tables.insider_transactions: transactions, Tables.insider_footnotes: notes},
                {Tables.insider_transactions: {"accession_number": ["a", "b"]}, Tables.insider_footnotes: {"accession_number": ["a", "b"]}},
            )
            actual, actual_notes = stage.insider_inventory_snapshot("AAA", ["a", "b"])
            expected = transactions.assign(filing_date=transactions["filing_date"].dt.date)
            pd.testing.assert_frame_equal(actual.drop(columns="inventory_checked_through"), expected, check_dtype=False, check_exact=True)
            pd.testing.assert_frame_equal(actual_notes, notes, check_dtype=False)
            assert set(inspect(stage.engine).get_table_names()) == {Tables.insider_transactions.name, Tables.insider_footnotes.name}
            assert staged_path.exists()
            staging_bytes = staged_path.stat().st_size
            raise RuntimeError("controlled context exit")
    assert staged_path is not None and not staged_path.exists() and not staged_path.parent.exists()
    print(
        f"SANITY: transient existing-table SQLite preserves exact payloads/notes; 2 transaction rows, 2 note rows, {staging_bytes} bytes, "
        f"largest issuer2 rows, {perf_counter() - started:.3f}s; rollback exit removes database and directory."
    )


@pytest.mark.parametrize(
    "table,count", [(Tables.entity_lineage, "n_observations"), (Tables.security_master, "n_observations"), (Tables.symbol_tenure, "n_filings")]
)
def test_count_and_evidence_churn_preserves_identity_proof(sqlite_store: DataStore, table: Table, count: str) -> None:
    _, _, identities = _inventory(sqlite_store)
    frames = _tape_observations(sqlite_store)
    before = {spec: sqlite_store.load(spec, columns=[str(column) for column in frame]) for spec, frame in frames.items()}
    rows = identities[table].assign(**{count: 99, "evidence": "additional observations, unchanged semantic scope"})
    sqlite_store.replace_institutional_identity(table, rows)
    actual = sqlite_store.load(Tables.insider_transactions, markers=True)
    assert actual is not None and actual["inventory_checked_through"].notna().all()
    for spec, expected in before.items():
        actual = sqlite_store.load(spec, columns=[str(column) for column in frames[spec]])
        assert expected is not None and actual is not None
        pd.testing.assert_frame_equal(actual, expected)
    print("SANITY: count/evidence-only refresh preserves unchanged insider and tape proof, quiet payload and raw observations.")


@pytest.mark.parametrize("prior_verified", [True, False])
def test_producer_certifies_actual_payload_and_valid_prior_cohort_only(sqlite_store: DataStore, prior_verified: bool) -> None:
    transactions, notes, identities = _inventory(sqlite_store)
    if not prior_verified:
        sqlite_store.save(
            Tables.insider_transactions,
            transactions.assign(inventory_checked_through=pd.Series(pd.NaT, index=transactions.index, dtype="datetime64[ns]")),
        )
    context_stub = SimpleNamespace(
        store=sqlite_store, _institutional_identity_cohort=identities, log=SimpleNamespace(info=lambda *args: None, warning=lambda *args: None)
    )
    context = cast(Context, context_stub)
    with insider_source_run(context):
        save_insider_source(context, transactions.iloc[:1].assign(shares=9.0), notes, identities)
        assert certify_insider_source(context, "AAA", pd.Timestamp("2026-07-04"), ["a", "b"]) is prior_verified
        if not prior_verified:
            save_insider_source(context, transactions.iloc[1:], notes.iloc[:0], identities)
            assert certify_insider_source(context, "AAA", pd.Timestamp("2026-07-04"), ["a", "b"])
        assert not certify_insider_source(context, "AAA", pd.Timestamp("2026-07-05"), ["a"])
    assert not hasattr(context, "_insider_source_store")
    rows = sqlite_store.load(Tables.insider_transactions, markers=True)
    assert rows is not None and rows["inventory_checked_through"].isna().all()
    print(
        "SANITY: actual producer stage can reuse a captured joint cohort; legacy NULL requires every original payload, and changed inventory clears proof."
    )


def test_actual_empty_footnote_cohort_can_be_jointly_certified(sqlite_store: DataStore) -> None:
    transactions, _, identities = _inventory(sqlite_store)
    notes = pd.DataFrame(columns=list(Tables.insider_footnotes.retrieval_columns))
    context_stub = SimpleNamespace(
        store=sqlite_store, _institutional_identity_cohort=identities, log=SimpleNamespace(info=lambda *args: None, warning=lambda *args: None)
    )
    context = cast(Context, context_stub)
    with insider_source_run(context):
        save_insider_source(context, transactions, notes, identities)
        assert certify_insider_source(context, "AAA", pd.Timestamp("2026-07-04"), ["a", "b"])
    assert sqlite_store.row_count(Tables.insider_footnotes) == 0
    print("SANITY: an actually reconciled empty note set is complete without fabricating a footnote or losing its empty schema during staging.")


def test_one_source_lifetime_refuses_a_different_parser_resolver(sqlite_store: DataStore) -> None:
    transactions, notes, identities = _inventory(sqlite_store)
    context_stub = SimpleNamespace(
        store=sqlite_store, _institutional_identity_cohort=identities, log=SimpleNamespace(info=lambda *args: None, warning=lambda *args: None)
    )
    context = cast(Context, context_stub)
    with insider_source_run(context):
        save_insider_source(context, transactions.iloc[:1].assign(shares=9.0), notes, identities)
        master = identities[Tables.security_master].assign(canonical_company="BBB")
        sqlite_store.replace_institutional_identity(Tables.security_master, master)
        later = load_source_identity(context)
        context_stub._institutional_identity_cohort = later
        with pytest.raises(ValueError, match="parser identity cohort changed"):
            save_insider_source(context, transactions.iloc[1:].assign(ticker="BBB", shares=99.0), notes.iloc[:0], later)
        assert not certify_insider_source(context, "AAA", pd.Timestamp("2026-07-04"), ["a", "b"])
    rows = sqlite_store.load(Tables.insider_transactions)
    assert rows is not None
    rows = rows.set_index("accession_number")
    assert rows.loc["a", "shares"] == 9.0 and rows.loc["b", "shares"] == 3.0
    assert rows["inventory_checked_through"].isna().all()
    print(
        "SANITY: a second resolver cannot rebind a live staging cohort; prior good payload remains and the older resolver cannot certify after the identity change."
    )


def test_stale_classification_identity_refuses_source_pair_commit(sqlite_store: DataStore) -> None:
    _, _, identities = _inventory(sqlite_store)
    raw = pd.DataFrame({"source_symbol": ["AAA"], "date": pd.to_datetime(["2026-07-01"]), "ticker": ["AAA"], "short_volume": [1.0]})
    grain = pd.DataFrame(
        {
            "ticker": ["AAA"],
            "date": pd.to_datetime(["2026-07-01"]),
            "short_volume": [1.0],
            "total_volume": [2.0],
            "source_checked_at": pd.to_datetime(["2026-07-04"]),
        }
    )
    sqlite_store.save(Tables.sec_short_volume_security, raw)
    sqlite_store.save(Tables.short_interest, grain)
    changed = identities[Tables.entity_lineage].assign(valid_from=pd.Timestamp("2019-01-01"))
    sqlite_store.replace_institutional_identity(Tables.entity_lineage, changed)
    before = sqlite_store.load(Tables.short_interest)
    assert before is not None
    with pytest.raises(ValueError, match="identity cohort changed"):
        sqlite_store.replace_institutional_source(
            {Tables.sec_short_volume_security: raw.assign(short_volume=99.0), Tables.short_interest: grain.assign(short_volume=99.0)},
            {Tables.sec_short_volume_security: {"ticker": "AAA"}, Tables.short_interest: {"ticker": "AAA"}},
            identities,
        )
    stored_frame = sqlite_store.load(Tables.short_interest)
    assert stored_frame is not None
    pd.testing.assert_frame_equal(stored_frame, before)
    actual_raw = sqlite_store.load(Tables.sec_short_volume_security)
    assert actual_raw is not None
    assert actual_raw["short_volume"].tolist() == [1.0]
    print(
        "SANITY: I1 classification cannot commit checked source output after I2 identity invalidation; raw, derived and retained proof remain atomic."
    )


@pytest.mark.parametrize("case", ["disappeared", "ambiguous", "rollback"])
def test_ftd_null_owner_reconciliation_is_exact_and_atomic(sqlite_store: DataStore, case: str) -> None:
    _, _, identities = _inventory(sqlite_store)
    master = pd.DataFrame(
        {
            "security_id": ["C111", "C222"],
            "source": ["ftd"] * 2,
            "source_symbol": ["AAA", "BBB"],
            "valid_from": pd.to_datetime(["2020-01-01"] * 2),
            "valid_to": [pd.NaT] * 2,
            "canonical_company": ["AAA", "BBB"],
            "issuer_cik": ["0000000001", "0000000002"],
            "cusip": ["111", "222"],
        }
    )
    if case == "ambiguous":
        master = pd.concat([master, master.iloc[:1].assign(security_id="C333", canonical_company="CCC")], ignore_index=True)
    sqlite_store.replace(Tables.security_master, master)
    identities[Tables.security_master] = master
    raw = pd.DataFrame(
        {
            "cusip": ["111", "222"],
            "date": pd.to_datetime(["2026-07-03"] * 2),
            "trade_date": pd.to_datetime(["2026-07-01"] * 2),
            "source_symbol": ["AAA", "BBB"],
            "ticker": [None, "BBB"],
            "period": ["202607a"] * 2,
            "fails_quantity": [10.0, 20.0],
        }
    )
    grain = pd.DataFrame(
        {
            "ticker": ["AAA"],
            "date": pd.to_datetime(["2026-07-03"]),
            "period": ["202607a"],
            "fails_quantity": [5.0],
            "fails_value": [50.0],
            "source_checked_at": pd.to_datetime(["2026-07-04"]),
        }
    )
    sqlite_store.save(Tables.sec_fails_to_deliver_security, raw)
    sqlite_store.save(Tables.sec_fails_to_deliver, grain)
    before_raw = sqlite_store.load(Tables.sec_fails_to_deliver_security)
    assert before_raw is not None
    before_grain = sqlite_store.load(Tables.sec_fails_to_deliver)
    assert before_grain is not None

    def refuse(conn: Any, cursor: Any, statement: str, parameters: Any, context: Any, executemany: bool) -> None:
        if statement.startswith(f"INSERT INTO {Tables.sec_fails_to_deliver.name} "):
            raise RuntimeError("controlled derived failure")

    if case == "rollback":
        event.listen(sqlite_store.engine, "before_cursor_execute", refuse)
    try:
        if case in ("rollback", "ambiguous"):
            with pytest.raises((RuntimeError, ValueError), match="controlled|Ambiguous"):
                sqlite_store.replace_institutional_source(
                    {
                        Tables.sec_fails_to_deliver_security: raw.iloc[:0],
                        Tables.sec_fails_to_deliver: grain.assign(fails_quantity=0.0, fails_value=0.0),
                    },
                    {
                        Tables.sec_fails_to_deliver_security: {"ticker": ["AAA"], "period": "202607a"},
                        Tables.sec_fails_to_deliver: {"ticker": ["AAA"], "period": "202607a"},
                    },
                    identities,
                )
            stored_frame = sqlite_store.load(Tables.sec_fails_to_deliver_security)
            assert stored_frame is not None
            pd.testing.assert_frame_equal(stored_frame, before_raw)
            stored_frame = sqlite_store.load(Tables.sec_fails_to_deliver)
            assert stored_frame is not None
            pd.testing.assert_frame_equal(stored_frame, before_grain)
        else:
            sqlite_store.replace_institutional_source(
                {Tables.sec_fails_to_deliver_security: raw.iloc[:0], Tables.sec_fails_to_deliver: grain.assign(fails_quantity=0.0, fails_value=0.0)},
                {
                    Tables.sec_fails_to_deliver_security: {"ticker": ["AAA"], "period": "202607a"},
                    Tables.sec_fails_to_deliver: {"ticker": ["AAA"], "period": "202607a"},
                },
                identities,
            )
            remaining = sqlite_store.load(Tables.sec_fails_to_deliver_security)
            assert remaining is not None and remaining["cusip"].tolist() == ["222"] and remaining["fails_quantity"].tolist() == [20.0]
    finally:
        if case == "rollback":
            event.remove(sqlite_store.engine, "before_cursor_execute", refuse)
    print(
        "SANITY: disappeared dated target NULL-owner keys reconcile atomically; retained other issuers survive, ambiguity refuses proof and failed derived inserts roll back raw deletion."
    )


@pytest.mark.parametrize("case", ["complete", "partial", "mixed", "unfetched", "marker", "absent"])
def test_inventory_boundary_requires_complete_consistent_rows(sqlite_store: DataStore, case: str) -> None:
    rows, _, _ = _inventory(sqlite_store)
    if case == "partial":
        rows.loc[0, "inventory_checked_through"] = pd.NaT
    elif case == "mixed":
        rows.loc[0, "inventory_checked_through"] = pd.Timestamp("2026-07-02")
    elif case == "unfetched":
        rows.loc[0, "fetched_at"] = pd.NaT
    elif case == "marker":
        rows = rows.iloc[:1].assign(security_type="_empty")
    sqlite_store.delete(Tables.insider_transactions, where={"ticker": "AAA"})
    if case != "absent":
        sqlite_store.save(Tables.insider_transactions, rows)
    expected = pd.Series(pd.NaT, index=pd.Index(["AAA", "MISSING"], name="ticker"), dtype="datetime64[ns]")
    if case in {"complete", "marker"}:
        expected.loc["AAA"] = pd.Timestamp("2026-07-01")
    pd.testing.assert_series_equal(sqlite_store.insider_inventory_boundaries(["MISSING", "AAA", "AAA"]), expected)
    empty = sqlite_store.insider_inventory_boundaries([])
    assert empty.empty and str(empty.dtype) == "datetime64[ns]" and empty.index.name == "ticker"
    print(f"SANITY: {case} proof includes markers, requires every row and preserves sorted unique requests/NaT/datetime dtype.")
