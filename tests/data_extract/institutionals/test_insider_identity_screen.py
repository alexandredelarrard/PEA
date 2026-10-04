"""The identity screen inside `screen_insider_rows`: CIK-first resolution, the kept / rejected
partition, the exclusion warning, and the stored-row sweep that the upsert cannot do.

The fixture is the four real cases the screen was designed against, shrunk to the two
columns the screen reads (`ticker`, `issuer_cik`) plus the ones the warning counts. Real
CIKs throughout, because a synthetic id would let a wrong entity mapping pass unnoticed:

    IR     Ingersoll-Rand plc 0001466258 is TT's registrant today, and filed under `IR` for
           eleven years. The single-axis "same people" oracle says KEEP and is wrong.
    DD     DuPont E I de Nemours 0000030554 IS DD's entity -- genuine predecessor history a
           fail-closed rule would delete.
    COR    AmerisourceBergen 0001140859 filed as `ABC`; today's roster says `COR`. The ADMIT
           case: the symbol path drops it, the CIK path keeps it.
    AVGO   Avicena Group 0001317092 typed `AVGO` before Broadcom existed under it.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest
from sqlalchemy import create_engine

from src.data_extract.utils.common.identity import build_identity
from src.data_extract.utils.institutionals import fetch_insider_transactions as ins
from src.data_extract.utils.institutionals.fetch_insider_transactions import _footnote_strings
from src.data_extract.utils.institutionals.insider_common import (
    BULK_DATE_FORMATS,
    INSIDER_COLUMNS,
    build_insider_frame,
    exclusion_message,
    exclusion_rows,
    filter_footnotes,
    insider_verdicts,
    log_exclusions,
    screen_insider_rows,
    screened_accessions,
)
from src.data_store.schema import Tables
from src.data_store.store import DataStore

# --------------------------------------------------------------------------- #
# fixtures                                                                      #
# --------------------------------------------------------------------------- #

BROADCOM = "0001441634"  # Broadcom Inc, AVGO's roster CIK
BROADCOM_LTD = "0001649338"  # Broadcom Ltd, the Singapore segment
AVAGO = "0001441634"  # (same CIK family; the register chains them)
AVICENA = "0001317092"  # Avicena Group -- typed AVGO, unrelated company
TRANE = "0001466258"  # Ingersoll-Rand plc, TT's registrant today
TRANE_NEW = "0001699150"  # Trane Technologies plc, TT's roster CIK
INGERSOLL_OLD = "0000836102"  # Ingersoll-Rand Company -- the entity IR's ROSTER points at
DUPONT_EI = "0000030554"  # DuPont E I de Nemours -- DD's own predecessor
ABC = "0001140859"  # AmerisourceBergen / Cencora, COR's roster CIK
CORESITE = "0001490892"  # CoreSite Realty -- held COR until 2021

UNIVERSE = ("AVGO", "TT", "DD", "COR", "IR")
#: The 3 golden quarters; the cache path is relative to the repo root the suite runs from.
INSIDER_CACHE = Path("data/sec_insider_transactions")
GOLDEN_QUARTERS = ("2024q4", "2025q3", "2026q1")


@pytest.fixture(scope="module")
def identity():
    """Axis A only -- the screen never reads tenure (D1), so tenure carries just enough to
    satisfy the D19 cross-check that every roster ticker's filings name its roster entity."""
    lineage = pd.DataFrame(
        [
            # Broadcom: a register chain, two CIKs one entity
            {"cik": BROADCOM, "entity_id": "E0001441634", "source": "register"},
            {"cik": BROADCOM_LTD, "entity_id": "E0001441634", "source": "register"},
            # Trane: the old Ingersoll-Rand plc belongs with Trane Technologies, NOT with `IR`
            {"cik": TRANE, "entity_id": "E0001699150", "source": "register"},
            {"cik": TRANE_NEW, "entity_id": "E0001699150", "source": "register"},
            # `IR` (Ingersoll Rand Inc) is a DIFFERENT entity that happens to hold the symbol
            {"cik": INGERSOLL_OLD, "entity_id": "E0000836102", "source": "roster"},
            {"cik": DUPONT_EI, "entity_id": "E0000030554", "source": "register"},
            {"cik": ABC, "entity_id": "E0001140859", "source": "roster"},
            {"cik": CORESITE, "entity_id": "E0001490892", "source": "owner_overlap"},
        ]
    ).assign(confidence=None, evidence="test")
    tenure = pd.DataFrame(
        [
            {"symbol": s, "issuer_cik": c, "valid_from": pd.Timestamp(f), "valid_to": None if t is None else pd.Timestamp(t), "n_filings": n}
            for s, c, f, t, n in [
                ("AVGO", BROADCOM, "2018-04-04", None, 900),
                ("AVGO", AVICENA, "2006-03-01", "2008-01-01", 12),
                ("TT", TRANE_NEW, "2020-03-02", None, 400),
                ("IR", TRANE, "2009-07-01", "2020-02-28", 2075),
                ("IR", INGERSOLL_OLD, "2020-03-01", None, 500),
                ("DD", DUPONT_EI, "2006-01-03", "2017-09-01", 3422),
                ("COR", ABC, "2023-09-01", None, 700),
                ("COR", CORESITE, "2010-06-01", "2021-12-28", 1207),
            ]
        ]
    ).assign(source="form345", evidence="")
    roster = pd.DataFrame(
        [{"ticker": t, "cik": c} for t, c in [("AVGO", BROADCOM), ("TT", TRANE_NEW), ("IR", INGERSOLL_OLD), ("DD", DUPONT_EI), ("COR", ABC)]]
    )
    return build_identity(lineage=lineage, tenure=tenure, roster=roster)


def _rows(pairs) -> pd.DataFrame:
    """One transaction row per (claimed symbol, issuer CIK)."""
    return pd.DataFrame(
        [
            {
                "accession_number": f"a{i}",
                "security_type": "nonderiv",
                "row_sequence": i + 1,
                "ticker": sym,
                "issuer_cik": cik,
                "issuer_name": f"CO {i}",
                "filing_date": pd.Timestamp("2015-06-01"),
                "transaction_code": "P",
                "value_usd": 1000.0 * (i + 1),
            }
            for i, (sym, cik) in enumerate(pairs)
        ]
    )


# --------------------------------------------------------------------------- #
# the four named cases                                                          #
# --------------------------------------------------------------------------- #


def test_a_trane_row_filed_under_ir_is_relabelled_to_tt_not_rejected(identity):
    """The headline defect -- Trane Technologies insider trading sitting in `IR`'s panel -- and
    THE OUTCOME IS A RELABEL, NOT A REJECTION: Trane's entity holds a universe ticker, so
    CIK-first moves the rows to `TT`. Rejection is for a symbol whose earlier holder is not in the
    universe at all -- `COR`/CoreSite, next test."""
    kept, rejected = screen_insider_rows(_rows([("IR", TRANE)]), UNIVERSE, identity)

    assert rejected.empty, "Trane's entity holds TT, so the row moves rather than dying"
    assert list(kept["ticker"]) == ["TT"]
    assert list(kept["claimed_ticker"]) == ["IR"]
    print("\n=== the IR case: a THIRD outcome ===")
    print(f"  Trane CIK {TRANE} claimed IR -> relabelled to TT. Not kept, not lost: moved.")


def test_a_coresite_row_filed_under_cor_is_rejected_as_entity_mismatch(identity):
    """The rejection shape: CoreSite Realty held `COR` until 2021 and its entity holds NO
    universe ticker, so its rows are not stored; the rejected frame keeps the claim for the warning."""
    kept, rejected = screen_insider_rows(_rows([("COR", CORESITE)]), UNIVERSE, identity)

    assert kept.empty, "a CoreSite filing must not be kept under COR"
    assert len(rejected) == 1
    row = rejected.iloc[0]
    assert row["reject_reason"] == "entity_mismatch"
    assert row["claimed_ticker"] == "COR"
    assert pd.isna(row["ticker"]), "the CIK resolves to no universe ticker"
    assert not {"resolved_entity_id", "universe_entity_id", "screened_on"} & set(rejected.columns), "no quarantine-only verdict columns"
    print("\n=== SANITY: the COR reuse case ===")
    print(f"  CoreSite {CORESITE} claimed COR -> rejected as entity_mismatch, claim kept for the warning, nothing stored.")


def test_a_dupont_e_i_row_filed_under_dd_is_kept(identity):
    """The case a fail-closed rule would destroy: genuine predecessor history. Same entity,
    different CIK -- which is exactly what `entity_lineage` exists to say."""
    kept, rejected = screen_insider_rows(_rows([("DD", DUPONT_EI)]), UNIVERSE, identity)

    assert rejected.empty
    assert list(kept["ticker"]) == ["DD"]
    print("\n=== the DD case ===")
    print(f"  DuPont E I {DUPONT_EI} kept under DD: predecessor history is not symbol reuse.")


def test_an_amerisourcebergen_row_filed_as_abc_is_admitted_as_cor(identity):
    """THE ADMIT CASE: `ABC` is not a universe ticker, so the symbol path dropped the row
    outright; the CIK path resolves it to the ticker its own entity holds today."""
    kept, rejected = screen_insider_rows(_rows([("ABC", ABC)]), UNIVERSE, identity)

    assert rejected.empty, "an admitted row is not a rejected row"
    assert list(kept["ticker"]) == ["COR"]
    assert list(kept["claimed_ticker"]) == ["ABC"]
    print("\n=== the COR admit case ===")
    print("  a filing typed ABC resolves to COR. Symbol-first dropped it; CIK-first keeps it.")


def test_avicena_is_rejected_while_every_broadcom_segment_is_kept(identity):
    """One symbol, a register CHAIN and a reuse case at once -- the shape that cannot be
    expressed by one CIK per ticker, and the reason axis A is a table rather than a dict."""
    kept, rejected = screen_insider_rows(_rows([("AVGO", AVICENA), ("AVGO", BROADCOM), ("AVGO", BROADCOM_LTD)]), UNIVERSE, identity)

    assert list(rejected["issuer_cik"]) == [AVICENA]
    assert rejected.iloc[0]["reject_reason"] == "entity_mismatch"
    assert set(kept["issuer_cik"]) == {BROADCOM, BROADCOM_LTD}
    assert set(kept["ticker"]) == {"AVGO"}
    print("\n=== the AVGO case ===")
    print(f"  Avicena {AVICENA} rejected; both Broadcom segments kept under AVGO.")


# --------------------------------------------------------------------------- #
# the reasons and the partition                                                 #
# --------------------------------------------------------------------------- #


def test_a_row_with_no_issuer_cik_gets_its_own_reason(identity):
    """The branch exists so that a CIK-less source row is a LABELLED reject in the warning and
    not a silently dropped one."""
    _, rejected = screen_insider_rows(_rows([("DD", None)]), UNIVERSE, identity)

    assert list(rejected["reject_reason"]) == ["no_issuer_cik"]
    print("\n=== no_issuer_cik ===")
    print("  a CIK-less row is rejected with its own reason, never silently dropped.")


def test_a_departed_universe_ticker_is_reason_entity_not_in_universe(identity):
    """The `EA` / `AVB` shape: 8,099 stored rows whose ticker left the universe. The parse
    cannot see them at all (no zip row claims a ticker the universe lost AND resolves), so
    this reason is mostly produced by the stored-row sweep -- but the label is the same one."""
    frame = _rows([("COR", CORESITE)])
    scored = insider_verdicts(frame, UNIVERSE, identity)

    assert list(scored["reject_reason"]) == ["entity_mismatch"]  # COR IS in the universe
    gone = insider_verdicts(frame, tuple(symbol for symbol in UNIVERSE if symbol != "COR"), identity)
    assert list(gone["reject_reason"]) == ["entity_not_in_universe"]
    print("\n=== the two rejection reasons are distinguishable ===")
    print("  the same CoreSite row reads entity_mismatch while COR is in the universe and entity_not_in_universe once it leaves.")


def test_the_partition_is_disjoint_and_loses_no_in_scope_row(identity):
    """The one invariant a partition must have. Scope is checked separately: `df` is every filer in
    the quarter, so an exhaustive rejected set would count rows about companies nothing here reads."""
    pairs = [("IR", TRANE), ("DD", DUPONT_EI), ("ABC", ABC), ("AVGO", AVICENA), ("AVGO", BROADCOM), ("COR", CORESITE), ("ZZZZ", "0009999999")]
    frame = _rows(pairs)
    kept, rejected = screen_insider_rows(frame, UNIVERSE, identity)

    keys = ["accession_number", "security_type", "row_sequence"]
    kept_keys = set(map(tuple, kept[keys].to_numpy()))
    rej_keys = set(map(tuple, rejected[keys].to_numpy()))
    assert not (kept_keys & rej_keys), "a row cannot be both kept and rejected"
    assert len(kept) + len(rejected) == len(frame) - 1, "only the off-universe row is dropped silently"
    assert "ZZZZ" not in set(rejected["claimed_ticker"])
    print("\n=== SANITY: partition ===")
    print(f"  {len(frame)} rows in -> {len(kept)} kept + {len(rejected)} rejected in scope, disjoint; 1 unrelated filer dropped without counting.")


def test_the_exclusion_message_counts_filings_rows_reasons_and_claims(identity):
    """REQ-008: filings, rows and P/S rows; filings per reason in a fixed order; top claimed tickers
    by filings (ties in symbol order)."""
    frame = _rows([("COR", CORESITE), ("COR", CORESITE), ("AVGO", AVICENA), ("DD", None)])
    frame.loc[1, "accession_number"] = "a0"  # a second row of the first filing
    frame.loc[1, "transaction_code"] = "A"
    frame.loc[2, "transaction_code"] = "S"
    _, rejected = screen_insider_rows(frame, UNIVERSE, identity)

    message = exclusion_message("zip run", exclusion_rows(rejected))

    assert message.startswith("insider zip run: excluded 3 filing(s), 4 row(s), 3 P/S row(s)")
    assert "by reason: entity_mismatch 2, no_issuer_cik 1;" in message
    assert message.endswith("top claimed: AVGO 1, COR 1, DD 1")
    print(f"\n=== SANITY: exclusion message ===\n  {message}")


def test_log_exclusions_warns_once_and_is_info_when_nothing_was_excluded(identity, caplog):
    """One WARNING for all frames of a run; no exclusion is an INFO line, not a warning."""
    caplog.set_level(logging.INFO)
    log = logging.getLogger("tests.identity_screen")
    _, first = screen_insider_rows(_rows([("COR", CORESITE)]), UNIVERSE, identity)
    _, second = screen_insider_rows(_rows([("AVGO", AVICENA)]).assign(accession_number="b0"), UNIVERSE, identity)

    log_exclusions(log, "EDGAR run", [exclusion_rows(first), exclusion_rows(second), exclusion_rows(first.iloc[0:0])])
    log_exclusions(log, "zip run", [])

    warnings = [record.getMessage() for record in caplog.records if record.levelno == logging.WARNING]
    infos = [record.getMessage() for record in caplog.records if record.levelno == logging.INFO]
    assert len(warnings) == 1 and "excluded 2 filing(s), 2 row(s)" in warnings[0]
    assert "insider zip run: no filing excluded by the identity screen" in infos
    print(f"\n=== SANITY: one exclusion warning per run ===\n  {warnings[0]}")


def test_an_empty_frame_returns_two_frames_not_one(identity):
    """The caller unpacks unconditionally, so the empty path must keep the same arity."""
    kept, rejected = screen_insider_rows(pd.DataFrame(), UNIVERSE, identity)
    assert kept.empty and rejected.empty
    print("\n=== empty input ===")
    print("  (kept, rejected) arity preserved on an empty quarter.")


def test_no_date_filter_runs_on_a_post_boundary_predecessor_row(identity):
    """Forms 3/4/5 are UNION events (`registrant.FORM_POLICY`); a date cut here is the named
    XOM-`SCHEDULE 13G` regression."""
    frame = _rows([("DD", DUPONT_EI), ("DD", DUPONT_EI)])
    frame.loc[0, "filing_date"] = pd.Timestamp("2006-02-01")  # inside the tenure
    frame.loc[1, "filing_date"] = pd.Timestamp("2025-02-01")  # long past its close
    kept, rejected = screen_insider_rows(frame, UNIVERSE, identity)

    assert len(kept) == 2 and rejected.empty
    print("\n=== no date filter ===")
    print("  a DuPont E I Form 4 filed in 2025 is kept: the union policy is unchanged.")


def test_footnotes_follow_the_kept_accessions_only(identity):
    """`filter_footnotes` rides `set(kept['accession_number'])`, which is why the screen creates no
    NEW orphans."""
    kept, rejected = screen_insider_rows(_rows([("DD", DUPONT_EI), ("COR", CORESITE)]), UNIVERSE, identity)
    notes = pd.DataFrame({"ACCESSION_NUMBER": ["a0", "a1"], "FOOTNOTE_ID": ["F1", "F1"], "FOOTNOTE_TXT": ["kept", "rejected"]})
    out = filter_footnotes(_footnote_strings(notes), set(kept["accession_number"]))

    assert list(out["accession_number"]) == ["a0"]
    assert "a1" in set(rejected["accession_number"])
    print("\n=== footnotes ===")
    print("  the rejected accession's footnote is not stored, so no new orphan appears.")


# --------------------------------------------------------------------------- #
# the stored-row sweep -- the only code that DELETEs                            #
# --------------------------------------------------------------------------- #


def _sweep_context(store: Any) -> Any:
    ctx = cast(Any, type("_Ctx", (), {})())
    ctx.store = store
    return ctx


def test_the_sweep_deletes_the_stored_rejects_and_warns(tmp_path, identity, caplog):
    """THE PARSE SCREEN CANNOT DO THIS: `store.save` upserts, so a `--reparse` never removes a
    rejected row, and a SHRINKING universe is reconciled only here. Delete-only: nothing is kept
    elsewhere, the WARNING is the record."""
    caplog.set_level(logging.INFO)
    store = DataStore(create_engine(f"sqlite:///{tmp_path / 'sweep.db'}"))
    stored = _rows(
        [
            ("DD", DUPONT_EI),  # keep
            ("COR", CORESITE),  # reject: entity_mismatch
            ("IR", TRANE),  # keep, relabelled to TT
            ("ZZZZ", CORESITE),  # reject: the claimed ticker is not in the universe
        ]
    )
    store.save(Tables.insider_transactions, stored)

    deleted = ins._screen_stored_rows(_sweep_context(store), UNIVERSE, identity)

    left = store.load(Tables.insider_transactions)
    assert left is not None
    assert deleted == 2
    assert sorted(left["accession_number"]) == ["a0", "a2"], "only the two keepers remain"
    warnings = [record.getMessage() for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "insider stored-row sweep: excluded 2 filing(s), 2 row(s), 2 P/S row(s)" in warnings[0]
    assert "entity_mismatch 1, entity_not_in_universe 1" in warnings[0] and "top claimed: COR 1, ZZZZ 1" in warnings[0]
    print("\n=== SANITY: the stored-row sweep ===")
    print(f"  {len(stored)} stored -> {len(left)} kept, {deleted} deleted; one WARNING: {warnings[0]}")


def test_the_sweep_is_idempotent_and_a_clean_table_is_a_no_op(tmp_path, identity, caplog):
    """A second run must not re-delete or raise, and logs only the INFO no-exclusion line."""
    caplog.set_level(logging.INFO)
    store = DataStore(create_engine(f"sqlite:///{tmp_path / 'idem.db'}"))
    store.save(Tables.insider_transactions, _rows([("DD", DUPONT_EI), ("COR", CORESITE)]))
    ctx = _sweep_context(store)

    first = ins._screen_stored_rows(ctx, UNIVERSE, identity)
    caplog.clear()
    second = ins._screen_stored_rows(ctx, UNIVERSE, identity)

    assert first == 1
    assert second == 0, "a clean table must be a no-op, not a second delete"
    assert [record.levelno for record in caplog.records if record.levelno >= logging.WARNING] == []
    remaining = store.load(Tables.insider_transactions)
    assert remaining is not None and len(remaining) == 1
    print("\n=== SANITY: sweep idempotence ===")
    print(f"  first pass deleted {first}, second pass {second} with no WARNING. Re-running is safe.")


class _SpyStore:
    """A store proxy counting the rows every `load` returns and recording its `where`."""

    def __init__(self, store: Any) -> None:
        self.store = store
        self.loaded_rows = 0
        self.load_wheres: list[Any] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self.store, name)

    def load(self, table: Any, *args: Any, **kwargs: Any) -> Any:
        df = self.store.load(table, *args, **kwargs)
        self.load_wheres.append(kwargs.get("where"))
        self.loaded_rows += 0 if df is None else len(df)
        return df


def test_the_sweep_scores_distinct_ciks_and_loads_only_rejected_rows(tmp_path, identity, caplog):
    """The sweep scores each distinct `issuer_cik` once and loads only the accessions of rejected
    CIKs: a kept CIK stays, a rejected CIK and a NULL CIK are deleted, and a mixed accession is
    deleted whole while only its rejected row is counted in the warning (the delete keys on
    `accession_number`)."""
    caplog.set_level(logging.INFO)
    store = DataStore(create_engine(f"sqlite:///{tmp_path / 'distinct.db'}"))
    single = _rows([("DD", DUPONT_EI), ("COR", CORESITE), ("IR", TRANE), ("DD", None)])
    mixed = _rows([("DD", DUPONT_EI), ("COR", CORESITE)]).assign(accession_number="m0")
    stored = pd.concat([single, mixed], ignore_index=True)
    store.save(Tables.insider_transactions, stored)
    ctx = _sweep_context(_SpyStore(store))

    deleted = ins._screen_stored_rows(ctx, UNIVERSE, identity)

    left = store.load(Tables.insider_transactions)
    assert left is not None
    assert deleted == 4
    assert sorted(left["accession_number"]) == ["a0", "a2"]
    warnings = [record.getMessage() for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warnings) == 1 and "excluded 3 filing(s), 3 row(s)" in warnings[0] and "entity_mismatch 2, no_issuer_cik 1" in warnings[0]
    assert all(where and set(where) == {"accession_number"} for where in ctx.store.load_wheres), "no whole-table load"
    assert ctx.store.loaded_rows == 4, "only the rows of a1, a3 and m0 are read"
    print("\n=== SANITY: distinct-CIK sweep ===")
    print(
        f"  {len(stored)} stored rows over 4 distinct CIKs (one NULL); read {ctx.store.loaded_rows} rows of the 3 rejected accessions; "
        f"deleted {deleted}, kept a0/a2; warning: {warnings[0]}"
    )


@pytest.mark.skipif(
    not all((INSIDER_CACHE / f"{quarter}.zip").exists() for quarter in GOLDEN_QUARTERS), reason="cached insider zips absent (run from the repo root)"
)
def test_the_accession_prefilter_drops_only_accessions_the_row_screen_drops(identity):
    """On the 3 golden quarters, building every filer's rows then screening gives the same kept and
    rejected values as `_parse_quarter`, which first cuts the members to `screened_accessions`;
    no accession outside that set has a kept or rejected row. A dtype may differ only on a column
    that is entirely null (its inferred resolution follows the rows it sees), which stores the same."""
    summary = []
    null_only: set[str] = set()
    for quarter in GOLDEN_QUARTERS:
        tables = ins._read_tables(INSIDER_CACHE / f"{quarter}.zip")
        assert tables is not None
        sub, own, nonderiv, deriv, _ = tables
        fetched_at = pd.Timestamp("2026-10-03 12:00")
        df_built = build_insider_frame(*ins.extract_bulk_strings(sub, own, nonderiv, deriv), date_formats=BULK_DATE_FORMATS)
        full_kept, full_rejected = screen_insider_rows(df_built.assign(source="zip", quarter=quarter, fetched_at=fetched_at), UNIVERSE, identity)
        kept, rejected, _ = ins._parse_quarter(tables, quarter, UNIVERSE, identity, fetched_at)
        accessions = screened_accessions(ins._member_strings(sub, "filing"), UNIVERSE, identity)

        assert set(full_kept["accession_number"]) | set(full_rejected["accession_number"]) <= accessions
        full_kept = full_kept[[column for column in INSIDER_COLUMNS if column in full_kept.columns]]
        for new, full in ((kept, full_kept), (rejected, full_rejected)):
            pd.testing.assert_frame_equal(new.reset_index(drop=True), full.reset_index(drop=True), check_dtype=False)
            retyped = [column for column in new.columns if new[column].dtype != full[column].dtype]
            assert all(new[column].isna().all() for column in retyped), retyped
            null_only |= set(retyped)
        summary.append(
            f"{quarter}: {len(df_built)} rows built -> {len(kept)} kept + {len(rejected)} rejected from {len(accessions)}/{len(sub)} accessions"
        )
    print("\n=== SANITY: accession prefilter == row screen ===")
    print("  " + "; ".join(summary) + f". Identical values on all 3 golden quarters; dtype differs only on all-null columns {sorted(null_only)}.")
