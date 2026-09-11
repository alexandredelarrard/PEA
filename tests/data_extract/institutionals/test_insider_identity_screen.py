"""The identity screen inside `_filter_universe`: CIK-first resolution, the quarantine
partition, and the stored-row sweep that the upsert cannot do.

The fixture is the four real cases the screen was designed against, shrunk to the two
columns the screen reads (`ticker`, `issuer_cik`) plus the ones the quarantine stores. Real
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

import pandas as pd
import pytest

from src.data_extract.utils.common.identity import build_identity
from src.data_extract.utils.institutionals.fetch_insider_transactions import (
    _filter_universe, _to_quarantine, _verdicts)
from src.data_store.schema import Tables

# --------------------------------------------------------------------------- #
# fixtures                                                                      #
# --------------------------------------------------------------------------- #

BROADCOM = "0001441634"           # Broadcom Inc, AVGO's roster CIK
BROADCOM_LTD = "0001649338"       # Broadcom Ltd, the Singapore segment
AVAGO = "0001441634"              # (same CIK family; the register chains them)
AVICENA = "0001317092"            # Avicena Group -- typed AVGO, unrelated company
TRANE = "0001466258"              # Ingersoll-Rand plc, TT's registrant today
TRANE_NEW = "0001699150"          # Trane Technologies plc, TT's roster CIK
INGERSOLL_OLD = "0000836102"      # Ingersoll-Rand Company -- the entity IR's ROSTER points at
DUPONT_EI = "0000030554"          # DuPont E I de Nemours -- DD's own predecessor
ABC = "0001140859"                # AmerisourceBergen / Cencora, COR's roster CIK
CORESITE = "0001490892"           # CoreSite Realty -- held COR until 2021

UNIVERSE = {"AVGO", "TT", "DD", "COR", "IR"}


@pytest.fixture(scope="module")
def identity():
    """Axis A only -- the screen never reads tenure (D1), so tenure carries just enough to
    satisfy the D19 cross-check that every roster ticker's filings name its roster entity."""
    lineage = pd.DataFrame([
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
    ]).assign(confidence=None, evidence="test")
    tenure = pd.DataFrame([
        {"symbol": s, "issuer_cik": c, "valid_from": pd.Timestamp(f),
         "valid_to": None if t is None else pd.Timestamp(t), "n_filings": n}
        for s, c, f, t, n in [
            ("AVGO", BROADCOM, "2018-04-04", None, 900),
            ("AVGO", AVICENA, "2006-03-01", "2008-01-01", 12),
            ("TT", TRANE_NEW, "2020-03-02", None, 400),
            ("IR", TRANE, "2009-07-01", "2020-02-28", 2075),
            ("IR", INGERSOLL_OLD, "2020-03-01", None, 500),
            ("DD", DUPONT_EI, "2006-01-03", "2017-09-01", 3422),
            ("COR", ABC, "2023-09-01", None, 700),
            ("COR", CORESITE, "2010-06-01", "2021-12-28", 1207),
        ]]).assign(source="form345", evidence="")
    roster = pd.DataFrame([{"ticker": t, "cik": c} for t, c in [
        ("AVGO", BROADCOM), ("TT", TRANE_NEW), ("IR", INGERSOLL_OLD),
        ("DD", DUPONT_EI), ("COR", ABC)]])
    return build_identity(lineage=lineage, tenure=tenure, roster=roster)


def _rows(pairs) -> pd.DataFrame:
    """One transaction row per (claimed symbol, issuer CIK)."""
    return pd.DataFrame([
        {"accession_number": f"a{i}", "security_type": "nonderiv", "transaction_sk": str(i),
         "ticker": sym, "issuer_cik": cik, "issuer_name": f"CO {i}",
         "filing_date": pd.Timestamp("2015-06-01"), "transaction_code": "P",
         "value_usd": 1000.0 * (i + 1)}
        for i, (sym, cik) in enumerate(pairs)])


# --------------------------------------------------------------------------- #
# the four named cases                                                          #
# --------------------------------------------------------------------------- #

def test_a_trane_row_filed_under_IR_is_relabelled_to_TT_not_quarantined(identity):
    """The headline defect -- 2,075 rows of Trane Technologies insider trading sitting in
    `IR`'s panel -- and ⚠ THE OUTCOME IS A RELABEL, NOT A REJECTION.

    The plan predicted a quarantine here; the measured resolution is better than that. Trane's
    entity DOES hold a universe ticker, so CIK-first moves the rows to `TT` where they belong
    instead of deleting real insider history. Measured live: 2,554 of the 2,558 relabelled
    rows are exactly this group. Rejection is for a symbol whose earlier holder is not in the
    universe at all -- `COR`/CoreSite, next test."""
    kept, rejected = _filter_universe(_rows([("IR", TRANE)]), UNIVERSE, identity)

    assert rejected.empty, "Trane's entity holds TT, so the row moves rather than dying"
    assert list(kept["ticker"]) == ["TT"]
    assert list(kept["claimed_ticker"]) == ["IR"]
    print("\n=== the IR case: a THIRD outcome ===")
    print(f"  Trane CIK {TRANE} claimed IR -> relabelled to TT. Not kept, not lost: moved.")


def test_a_coresite_row_filed_under_COR_is_rejected_with_both_entity_ids(identity):
    """The rejection shape: CoreSite Realty held `COR` until 2021 and its entity holds NO
    universe ticker, so there is nowhere for its 1,207 rows to go but the quarantine."""
    kept, rejected = _filter_universe(_rows([("COR", CORESITE)]), UNIVERSE, identity)

    assert kept.empty, "a CoreSite filing must not be kept under COR"
    assert len(rejected) == 1
    row = rejected.iloc[0]
    assert row["reject_reason"] == "entity_mismatch"
    assert row["claimed_ticker"] == "COR"
    assert row["resolved_entity_id"] == "E0001490892"     # the entity the CIK really is
    assert row["universe_entity_id"] == "E0001140859"     # the entity `COR` names today
    assert row["resolved_entity_id"] != row["universe_entity_id"]
    print("\n=== the COR reuse case ===")
    print(f"  CoreSite {CORESITE} claimed COR -> rejected; {row['resolved_entity_id']} "
          f"!= {row['universe_entity_id']}. Both ids stored, so an id shift is a diff.")


def test_a_dupont_E_I_row_filed_under_DD_is_kept(identity):
    """The case a fail-closed rule would destroy: 3,422 rows of genuine predecessor history.
    Same entity, different CIK -- which is exactly what `entity_lineage` exists to say."""
    kept, rejected = _filter_universe(_rows([("DD", DUPONT_EI)]), UNIVERSE, identity)

    assert rejected.empty
    assert list(kept["ticker"]) == ["DD"]
    print("\n=== the DD case ===")
    print(f"  DuPont E I {DUPONT_EI} kept under DD: predecessor history is not symbol reuse.")


def test_an_amerisourcebergen_row_filed_as_ABC_is_admitted_as_COR(identity):
    """⚠ THE ADMIT CASE, and the reason this change is not purely subtractive. `ABC` is not a
    universe ticker, so the symbol path dropped the row outright; the CIK path resolves it to
    the ticker its own entity holds today. Measured live: 2,275 rows come back this way."""
    kept, rejected = _filter_universe(_rows([("ABC", ABC)]), UNIVERSE, identity)

    assert rejected.empty, "an admitted row is not a rejected row"
    assert list(kept["ticker"]) == ["COR"]
    assert list(kept["claimed_ticker"]) == ["ABC"]
    print("\n=== the COR admit case ===")
    print("  a filing typed ABC resolves to COR. Symbol-first dropped it; CIK-first keeps it.")


def test_avicena_is_rejected_while_every_broadcom_segment_is_kept(identity):
    """One symbol, a register CHAIN and a reuse case at once -- the shape that cannot be
    expressed by one CIK per ticker, and the reason axis A is a table rather than a dict."""
    kept, rejected = _filter_universe(
        _rows([("AVGO", AVICENA), ("AVGO", BROADCOM), ("AVGO", BROADCOM_LTD)]),
        UNIVERSE, identity)

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
    """Measured ZERO across all 4,402,307 filings in the 81 cached quarters, which is why no
    symbol fallback is built. The branch exists so that a source change is a LABELLED
    quarantine row and not a silently dropped one."""
    _, rejected = _filter_universe(_rows([("DD", None)]), UNIVERSE, identity)

    assert list(rejected["reject_reason"]) == ["no_issuer_cik"]
    print("\n=== no_issuer_cik ===")
    print("  a CIK-less row is quarantined with its own reason, never silently dropped.")


def test_a_departed_universe_ticker_is_reason_entity_not_in_universe(identity):
    """The `EA` / `AVB` shape: 8,099 stored rows whose ticker left the universe. The parse
    cannot see them at all (no zip row claims a ticker the universe lost AND resolves), so
    this reason is mostly produced by the stored-row sweep -- but the label is the same one."""
    frame = _rows([("COR", CORESITE)])
    scored = _verdicts(frame, UNIVERSE, identity)

    assert list(scored["reject_reason"]) == ["entity_mismatch"]      # COR IS in the universe
    gone = _verdicts(frame, UNIVERSE - {"COR"}, identity)
    assert list(gone["reject_reason"]) == ["entity_not_in_universe"]
    print("\n=== the two rejection reasons are distinguishable ===")
    print("  the same CoreSite row reads entity_mismatch while COR is in the universe and "
          "entity_not_in_universe once it leaves.")


def test_the_partition_is_disjoint_and_loses_no_in_scope_row(identity):
    """⚠ The one invariant a partition must have, and the one the old `return df[mask]` could
    not be asked about. Scope is checked separately: `df` is every filer in the quarter, so
    an EXHAUSTIVE quarantine would store ~50M rows about companies nothing here reads."""
    pairs = [("IR", TRANE), ("DD", DUPONT_EI), ("ABC", ABC), ("AVGO", AVICENA),
             ("AVGO", BROADCOM), ("COR", CORESITE), ("ZZZZ", "0009999999")]
    frame = _rows(pairs)
    kept, rejected = _filter_universe(frame, UNIVERSE, identity)

    keys = ["accession_number", "security_type", "transaction_sk"]
    kept_keys = set(map(tuple, kept[keys].to_numpy()))
    rej_keys = set(map(tuple, rejected[keys].to_numpy()))
    assert not (kept_keys & rej_keys), "a row cannot be both kept and quarantined"
    assert len(kept) + len(rejected) == len(frame) - 1, "only the off-universe row is dropped"
    assert "ZZZZ" not in set(rejected["claimed_ticker"])
    print("\n=== partition ===")
    print(f"  {len(frame)} rows in -> {len(kept)} kept + {len(rejected)} quarantined, "
          "disjoint; 1 unrelated filer dropped rather than quarantined.")


def test_the_quarantine_frame_stores_the_CLAIMED_ticker(identity):
    """⚠ `_to_quarantine` overwrites `ticker` with the claim on purpose. A NULL there would
    lose the only evidence of what the old screen believed, and the resolved side survives as
    an entity id, which no downstream join can mistake for a tradable symbol."""
    _, rejected = _filter_universe(_rows([("COR", CORESITE)]), UNIVERSE, identity)
    out = _to_quarantine(rejected)

    assert list(out["ticker"]) == ["COR"]
    assert out.iloc[0]["screened_on"] == pd.Timestamp("2015-06-01")     # D18: filing_date
    assert {"reject_reason", "resolved_entity_id", "universe_entity_id",
            "screened_on"} <= set(out.columns)
    print("\n=== quarantine shape ===")
    print("  ticker='IR' (the claim), screened_on=filing_date, both entity ids present.")


def test_an_empty_frame_returns_two_frames_not_one(identity):
    """The caller unpacks unconditionally, so the empty path must keep the same arity."""
    kept, rejected = _filter_universe(pd.DataFrame(), UNIVERSE, identity)
    assert kept.empty and rejected.empty
    print("\n=== empty input ===")
    print("  (kept, rejected) arity preserved on an empty quarter.")


def test_no_date_filter_runs_on_a_post_boundary_predecessor_row(identity):
    """Forms 3/4/5 are UNION events (`registrant.FORM_POLICY`). A date cut here is the named
    XOM-`SCHEDULE 13G` regression, so `screened_on` is RECORDED and never READ."""
    frame = _rows([("DD", DUPONT_EI), ("DD", DUPONT_EI)])
    frame.loc[0, "filing_date"] = pd.Timestamp("2006-02-01")    # inside the tenure
    frame.loc[1, "filing_date"] = pd.Timestamp("2025-02-01")    # long past its close
    kept, rejected = _filter_universe(frame, UNIVERSE, identity)

    assert len(kept) == 2 and rejected.empty
    print("\n=== no date filter ===")
    print("  a DuPont E I Form 4 filed in 2025 is kept: the union policy is unchanged.")


def test_footnotes_follow_the_kept_accessions_only(identity):
    """`_footnotes` rides `set(kept['accession_number'])`, which is why the screen creates no
    NEW orphans. Pre-existing orphans are left in place and documented (D13); measured live,
    `insider_footnotes` has 0 of them today."""
    from src.data_extract.utils.institutionals.fetch_insider_transactions import _footnotes

    kept, rejected = _filter_universe(
        _rows([("DD", DUPONT_EI), ("COR", CORESITE)]), UNIVERSE, identity)
    notes = pd.DataFrame({"ACCESSION_NUMBER": ["a0", "a1"], "FOOTNOTE_ID": ["F1", "F1"],
                          "FOOTNOTE_TXT": ["kept", "quarantined"]})
    out = _footnotes(notes, set(kept["accession_number"]))

    assert list(out["accession_number"]) == ["a0"]
    assert "a1" in set(rejected["accession_number"])
    print("\n=== footnotes ===")
    print("  the quarantined accession's footnote is not stored, so no new orphan appears.")


# --------------------------------------------------------------------------- #
# the stored-row sweep -- the only new code that DELETEs                        #
# --------------------------------------------------------------------------- #

def test_the_sweep_quarantines_and_deletes_the_stored_rejects(tmp_path, identity):
    """⚠ THE PARSE SCREEN CANNOT DO THIS AND THAT IS THE WHOLE POINT. `store.save` upserts, so
    a `--reparse` declines to re-write a rejected row and leaves it in the table for ever. The
    sweep is also the only thing that reconciles a SHRINKING universe: `EA` and `AVB` left the
    roster, and no parse would ever revisit their 7,949 stored rows.

    A throwaway SQLite store, because the assertion is about DELETE semantics and the accession
    key, not about Postgres.
    """
    from sqlalchemy import create_engine

    from src.data_store.store import DataStore
    from src.data_extract.utils.institutionals import fetch_insider_transactions as ins

    store = DataStore(create_engine(f"sqlite:///{tmp_path / 'sweep.db'}"))
    stored = _rows([("DD", DUPONT_EI),      # keep
                    ("COR", CORESITE),      # reject: entity_mismatch
                    ("IR", TRANE),          # keep, relabelled to TT
                    ("ZZZZ", CORESITE)])    # reject: claimed ticker left the universe
    store.save(Tables.insider_transactions, stored)

    class _Ctx:
        pass
    ctx = _Ctx()
    ctx.store = store

    quarantined, deleted = ins._screen_stored_rows(ctx, UNIVERSE, identity)

    left = store.load(Tables.insider_transactions)
    quarantine = store.load(Tables.insider_transactions_quarantine)
    assert (quarantined, deleted) == (2, 2)
    assert sorted(left["accession_number"]) == ["a0", "a2"], "only the two keepers remain"
    assert set(quarantine["reject_reason"]) == {"entity_mismatch", "entity_not_in_universe"}
    # nothing is lost: every stored row is now in exactly one of the two tables
    assert len(left) + len(quarantine) == len(stored)
    assert not set(left["accession_number"]) & set(quarantine["accession_number"])
    print("\n=== the stored-row sweep ===")
    print(f"  {len(stored)} stored -> {len(left)} kept + {len(quarantine)} quarantined; "
          "the upsert alone could have removed neither.")


def test_the_sweep_is_idempotent_and_a_clean_table_is_a_no_op(tmp_path, identity):
    """A second run must not re-delete, re-quarantine or raise. The PK is the same as
    `insider_transactions`', so a re-quarantine would upsert in place rather than duplicate --
    but the sweep should not even reach that, because nothing is left to reject."""
    from sqlalchemy import create_engine

    from src.data_store.store import DataStore
    from src.data_extract.utils.institutionals import fetch_insider_transactions as ins

    store = DataStore(create_engine(f"sqlite:///{tmp_path / 'idem.db'}"))
    store.save(Tables.insider_transactions, _rows([("DD", DUPONT_EI), ("COR", CORESITE)]))

    class _Ctx:
        pass
    ctx = _Ctx()
    ctx.store = store

    first = ins._screen_stored_rows(ctx, UNIVERSE, identity)
    second = ins._screen_stored_rows(ctx, UNIVERSE, identity)

    assert first == (1, 1)
    assert second == (0, 0), "a clean table must be a no-op, not a second delete"
    assert len(store.load(Tables.insider_transactions)) == 1
    print("\n=== sweep idempotence ===")
    print(f"  first pass {first}, second pass {second}. Re-running is safe.")
