"""P35: a CIK's 8-K and 13D/13G filings belong to a ticker only on the dates the CIK is the company.

Listing keeps them inside the CIK's seam-widened window, an acquired target's event-only CIK contributes
none, the propagation purge and the validator remove stored rows outside the windows, and the gate names
the removal. Insider forms keep listing over every CIK of the entity with their stored role (D-Q2-1).
Offline: a real `Identity` over hand-written lineage rows (CB, MRK, TMUS) and a real SQLite `DataStore`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from scripts import identity_regression_gate as gate
from src.constants.constants import CANONICAL_CURRENT, CANONICAL_PREDECESSOR, SEC_8K_FORMS, SEC_13D_FORMS, SEC_13G_FORMS, SEC_INSIDER_FORMS
from src.data_extract import identity_propagate as prop
from src.data_extract.utils.common.edgar_driver import EdgarScope, FilingStamp
from src.data_extract.utils.common.identity import Identity, build_identity
from src.data_extract.utils.common.registrant import Combine, combine_for, resolve_registrant_entries
from src.data_extract.utils.common.security_master import ACQUIRED_CONSTITUENT
from src.data_extract.utils.institutionals import schedule_rows
from src.data_extract.utils.institutionals.fetch_13g_edgar import SCHEDULE_13G
from src.data_store.schema import Tables
from src.validate.checks.identity import pending_removals

SENTINEL = "1900-01-01"
ACE, OLD_CHUBB = "0000896159", "0000020171"  # CB: Chubb Ltd (ACE) bought The Chubb Corporation on 2016-01-14
MRK, OLD_MERCK = "0000310158", "0000064978"  # MRK: Schering-Plough's CIK is Merck & Co. from 2009-11-03
TMUS, TMO_USA = "0001283699", "0001097609"  # TMUS: T-Mobile USA, a co-registrant (event only)
MRK_SEAM = "2009-11-03"
ROSTER = {"CB": ACE, "MRK": MRK, "TMUS": TMUS}
CHANGED = pd.Timestamp("2026-01-12 09:00")
RUN_DATE = pd.Timestamp("2026-01-13")


def _lineage() -> pd.DataFrame:
    rows = [
        ("CB", ACE, "cik_window", SENTINEL, None),
        ("CB", OLD_CHUBB, "cik_event", SENTINEL, None),
        ("MRK", OLD_MERCK, "cik_window", SENTINEL, MRK_SEAM),
        ("MRK", MRK, "cik_window", MRK_SEAM, None),
        ("TMUS", TMUS, "cik_window", SENTINEL, None),
        ("TMUS", TMO_USA, "cik_event", SENTINEL, None),
    ]
    return pd.DataFrame(
        [
            {
                "entity_id": f"E{ROSTER[ticker]}",
                "canonical_ticker": ticker,
                "cik": cik,
                "role": role,
                "symbol": "",
                "valid_from": pd.Timestamp(start),
                "valid_to": pd.Timestamp(end) if end else pd.NaT,
                "status": "curated",
                "sources": "register",
                "oracle": "register",
                "confidence": None,
                "n_observations": 1,
                "evidence": "fixture",
                "scope_changed_at": CHANGED,
            }
            for ticker, cik, role, start, end in rows
        ]
    )


def _identity() -> Identity:
    tenure = pd.DataFrame(
        [
            {
                "symbol": t,
                "issuer_cik": c,
                "valid_from": pd.Timestamp("2000-01-01"),
                "valid_to": None,
                "n_filings": 5,
                "source": "form345",
                "evidence": "",
            }
            for t, c in ROSTER.items()
        ]
    )
    roster = pd.DataFrame([{"ticker": t, "cik": c} for t, c in ROSTER.items()])
    return build_identity(lineage=_lineage(), tenure=tenure, roster=roster, co_registrant_ciks=(TMO_USA,))


def _index(rows: list[tuple[str, str, str, str]]) -> pd.DataFrame:
    """Local EDGAR index rows: (cik, form, filed, accession)."""
    return pd.DataFrame([{"cik": c, "company": "x", "form": f, "filed": d, "accession": a} for c, f, d, a in rows])


def _listed(identity: Identity, ticker: str, df_index: pd.DataFrame, forms: list[str]) -> list[str]:
    return resolve_registrant_entries(identity.filing_scope(ticker), df_index, forms)["accession"].tolist()


# --------------------------------------------------------------------------- listing


def test_event_forms_are_date_limited_and_insider_forms_still_union() -> None:
    for forms in (SEC_8K_FORMS, SEC_13D_FORMS, SEC_13G_FORMS, ["8-K12B"]):
        assert combine_for(forms) is Combine.SPLIT, forms
    assert combine_for(SEC_INSIDER_FORMS) is Combine.UNION
    print("\nsanity: 8-K, 13D and 13G list by dated CIK window (SPLIT); Forms 3/4/5 still union over every CIK of the entity")


def test_old_chubbs_8ks_are_not_cbs_on_any_date() -> None:
    """The Chubb Corporation's CIK is event-only under CB: neither its pre-merger nor its post-close 8-Ks are CB's."""
    df = _index(
        [
            (OLD_CHUBB, "8-K", "2015-07-01", "chubb-2015"),
            (OLD_CHUBB, "8-K", "2016-01-15", "chubb-close"),
            (ACE, "8-K", "2015-07-01", "ace-2015"),
            (ACE, "8-K/A", "2016-02-01", "ace-2016"),
        ]
    )
    assert _listed(_identity(), "CB", df, SEC_8K_FORMS) == ["ace-2015", "ace-2016"]
    print("\nsanity: CB lists ACE's 8-Ks only; old Chubb's 2015 and post-close 8-Ks are not CB's")


def test_mrks_8ks_follow_schering_plough_and_old_merck_windows() -> None:
    """SGP's CIK (MRK's roster CIK) is Merck only from 2009-11-03 (widened 31 days); old Merck's is Merck before it."""
    df = _index(
        [
            (MRK, "8-K", "2009-06-01", "sgp-2009"),
            (MRK, "8-K", "2008-02-01", "sgp-2008"),
            (MRK, "8-K", "2009-10-20", "sgp-margin"),
            (MRK, "8-K", "2010-02-01", "mrk-2010"),
            (OLD_MERCK, "8-K", "2008-05-01", "merck-2008"),
            (OLD_MERCK, "8-K", "2009-11-20", "merck-margin"),
            (OLD_MERCK, "8-K", "2010-06-01", "merck-late"),
        ]
    )
    got = _listed(_identity(), "MRK", df, SEC_8K_FORMS)
    assert got == ["merck-2008", "sgp-margin", "merck-margin", "mrk-2010"], got
    print("\nsanity: SGP's 8-Ks before the seam margin and old Merck's after it are not MRK's; each CIK's 8-Ks inside its widened window are")


def test_an_acquired_targets_schedule_filings_are_not_the_acquirers(monkeypatch: pytest.MonkeyPatch) -> None:
    """13D/13G indexed under old Chubb are not listed for CB; one listed under ACE but about old Chubb is rejected."""
    identity = _identity()
    df = _index(
        [
            (OLD_CHUBB, "SC 13G", "2014-02-10", "g-chubb"),
            (OLD_CHUBB, "SC 13D/A", "2015-08-01", "d-chubb"),
            (ACE, "SC 13G/A", "2015-02-10", "g-ace"),
            (MRK, "SC 13G", "2008-02-10", "g-sgp"),
            (OLD_MERCK, "SC 13G", "2008-02-11", "g-merck"),
        ]
    )
    assert _listed(identity, "CB", df, SEC_13G_FORMS) == ["g-ace"]
    assert _listed(identity, "CB", df, SEC_13D_FORMS) == []
    assert _listed(identity, "MRK", df, SEC_13G_FORMS) == ["g-merck"]
    scope = EdgarScope(identity)
    stamp = FilingStamp("g-joint", "SC 13G", ACE, pd.Timestamp("2015-02-10"), False, None, filing=SimpleNamespace())
    monkeypatch.setattr(schedule_rows, "header_subject_ciks", lambda filing: frozenset({OLD_CHUBB}))
    monkeypatch.setattr(schedule_rows, "schedule_filing_rows", lambda stamp, spec: [{"cik": OLD_CHUBB, "rp_seq": 0}])
    assert not schedule_rows.schedule_is_subject("CB", ACE, stamp, scope)
    assert schedule_rows.schedule_ticker_rows("CB", ACE, stamp, scope, SCHEDULE_13G) == []
    monkeypatch.setattr(schedule_rows, "header_subject_ciks", lambda filing: frozenset({ACE}))
    monkeypatch.setattr(schedule_rows, "schedule_filing_rows", lambda stamp, spec: [{"cik": ACE, "rp_seq": 0}])
    assert schedule_rows.schedule_is_subject("CB", ACE, stamp, scope)
    assert [row["ticker"] for row in schedule_rows.schedule_ticker_rows("CB", ACE, stamp, scope, SCHEDULE_13G)] == ["CB"]
    print("\nsanity: schedules whose subject is old Chubb never reach CB (not listed, subject and issuer guards refuse); ACE's own 13G is kept")


def test_insider_forms_of_an_event_only_cik_are_still_listed_with_the_acquired_role() -> None:
    identity = _identity()
    df = _index([(OLD_CHUBB, "4", "2015-07-01", "f4-chubb"), (ACE, "4", "2015-07-02", "f4-ace")])
    assert _listed(identity, "CB", df, SEC_INSIDER_FORMS) == ["f4-chubb", "f4-ace"]
    assert identity.lineage_role(OLD_CHUBB, pd.Timestamp("2015-07-01")) == ACQUIRED_CONSTITUENT
    assert identity.lineage_role(OLD_MERCK, pd.Timestamp("2008-05-01")) == CANONICAL_PREDECESSOR
    assert identity.lineage_role(MRK, pd.Timestamp("2010-02-01")) == CANONICAL_CURRENT
    assert identity.lineage_role(TMO_USA, pd.Timestamp("2015-07-01")) is None
    print("\nsanity: old Chubb's Form 4s are still listed for CB as acquired_constituent; the co-registrant has no role (not stored)")


# --------------------------------------------------------------------------- stored rows


def _8k(ticker: str, rows: list[tuple[str, str, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"ticker": ticker, "accession_number": a, "item": item, "cik": c, "filing_date": pd.Timestamp(f)}
            for a, c, f in rows
            for item in ("2.02", "9.01")
        ]
    )


def _13g(ticker: str, rows: list[tuple[str, str, str]]) -> pd.DataFrame:
    return pd.DataFrame([{"ticker": ticker, "accession_number": a, "rp_seq": 0, "cik": c, "filing_date": pd.Timestamp(f)} for a, c, f in rows])


def _insider(ticker: str, rows: list[tuple[str, str, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"ticker": ticker, "accession_number": a, "security_type": "nonderiv", "row_sequence": 1, "issuer_cik": c, "filing_date": pd.Timestamp(f)}
            for a, c, f in rows
        ]
    )


CB_8K = [("ace-1", ACE, "2015-07-01"), ("chubb-1", OLD_CHUBB, "2015-07-01"), ("chubb-2", OLD_CHUBB, "2016-01-15")]
MRK_8K = [("sgp-1", MRK, "2009-06-01"), ("merck-1", OLD_MERCK, "2008-05-01"), ("merck-late", OLD_MERCK, "2010-06-01"), ("mrk-1", MRK, "2010-02-01")]
TMUS_8K = [("t-1", TMUS, "2020-02-01"), ("t-usa", TMO_USA, "2020-02-01")]


def _seed(store) -> None:
    store.save(Tables.sec_8k, pd.concat([_8k("CB", CB_8K), _8k("MRK", MRK_8K), _8k("TMUS", TMUS_8K)], ignore_index=True))
    store.save(Tables.sec_13g, pd.concat([_13g("CB", [("g-ace", ACE, "2015-02-10"), ("g-chubb", OLD_CHUBB, "2014-02-10")])], ignore_index=True))
    store.save(Tables.insider_transactions, _insider("CB", [("f4-chubb", OLD_CHUBB, "2015-07-01"), ("f4-ace", ACE, "2015-07-02")]))


def _context(store, tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(store=store, paths={"DATA_STORE": tmp_path}, log=logging.getLogger("test.event_windows"), config_dir="./configs")


def _keys(store, table) -> set[str]:
    return set(store.load(table, columns=["accession_number"])["accession_number"])


def test_the_purge_removes_event_rows_outside_their_ciks_windows_and_keeps_insider_rows(sqlite_store, tmp_path, monkeypatch) -> None:
    context = _context(sqlite_store, tmp_path)
    _seed(sqlite_store)
    identity = _identity()
    monkeypatch.setattr(prop, "_reparse_bulk", lambda context, changed, scopes, co_registrants: {})
    monkeypatch.setattr(prop, "_refresh_insider", lambda context, identity, changed, dry_run: [])
    lineage = _lineage()
    pending = pending_removals(context, lineage, list(ROSTER))

    result = prop.propagate_identity(context, list(ROSTER), identity=identity, as_of=RUN_DATE)

    assert _keys(sqlite_store, Tables.sec_8k) == {"ace-1", "merck-1", "mrk-1", "t-1"}
    assert _keys(sqlite_store, Tables.sec_13g) == {"g-ace"}
    assert _keys(sqlite_store, Tables.insider_transactions) == {"f4-chubb", "f4-ace"}
    removed = {(r.table, r.ticker, r.cik, r.rows) for r in result.removals.itertuples(index=False)}
    assert removed == {
        ("sec_8k", "CB", OLD_CHUBB, 4),
        ("sec_8k", "MRK", MRK, 2),
        ("sec_8k", "MRK", OLD_MERCK, 2),
        ("sec_8k", "TMUS", TMO_USA, 2),
        ("sec_13g", "CB", OLD_CHUBB, 1),
    }, removed
    assert {(r.table, r.ticker, r.cik, r.rows) for r in pending.itertuples(index=False)} == removed
    print(
        "\nsanity: the purge (and the validator's pending list) removes old Chubb's, the co-registrant's, SGP's pre-seam and old Merck's late 8-K/13G rows; insider rows stay"
    )


def test_the_gate_names_every_removed_event_row() -> None:
    identity = _identity()
    scopes = {t: identity.filing_scope(t) for t in ROSTER}
    old = {t: frozenset(identity.filing_scope(t).event_ciks) for t in ROSTER}

    def frame(rows: list[tuple[str, str, str, str, str]]) -> pd.DataFrame:
        out = pd.DataFrame(rows, columns=["ticker", "accession", "cik", "form", "filed"]).assign(period_end=None)
        out["filed"] = pd.to_datetime(out["filed"])
        return out

    before = frame(
        [
            ("CB", "chubb-1", OLD_CHUBB, "8-K", "2015-07-01"),
            ("MRK", "sgp-1", MRK, "8-K", "2009-06-01"),
            ("MRK", "merck-late", OLD_MERCK, "8-K", "2010-06-01"),
            ("TMUS", "t-usa", TMO_USA, "8-K", "2020-02-01"),
            ("MRK", "merck-1", OLD_MERCK, "8-K", "2008-05-01"),
        ]
    )
    after = before[before["accession"].eq("merck-1")]
    diff = gate.filing_diff("sec_8k", before, after, scopes, old, identity.co_registrant_ciks)
    assert {r.accession: r.reason for r in diff.itertuples(index=False)} == {
        "chubb-1": "acquired_constituent_purge",
        "sgp-1": "outside_cik_window",
        "merck-late": "outside_cik_window",
        "t-usa": "co_registrant_purge",
    }
    g13 = gate.filing_diff("sec_13g", frame([("CB", "g-chubb", OLD_CHUBB, "SC 13G", "2014-02-10")]), frame([]), scopes, old)
    assert g13["reason"].tolist() == ["acquired_constituent_purge"]
    assert {"acquired_constituent_purge", "outside_cik_window"} <= gate.REASONS
    print("\nsanity: every removed 8-K/13G row carries a reason: acquired_constituent_purge, outside_cik_window or co_registrant_purge")
