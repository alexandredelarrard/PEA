"""Insider lineage (P25): each stored Form 3/4/5 row carries `source_symbol`, `economic_date` and
`lineage_role`; co-registrant rows are dropped at ingest; the cube reads canonical roles only; a
lineage change re-stamps the stored rows in place.

Known-truth fixtures on the real CIKs and seams: Merck / Schering-Plough (2009-11-03, closing
after the 16:00 close), Prologis = AMB with old ProLogis an acquired target (2011-06-03, the traded-security
view), old Chubb (event-only acquired target), Digital Realty LP and the PG&E utility (co-registrants).
Merger metadata comes from the shipped `configs/sec/security_master_manual.json` (MRK only).
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from src.data_aggregate.utils.institutionals import inputs as institutional_inputs
from src.data_extract import identity_propagate as prop
from src.data_extract.utils.common import security_master as sm
from src.data_extract.utils.common.identity import build_identity
from src.data_extract.utils.institutionals import fetch_insider_transactions as ins
from src.data_extract.utils.institutionals import insider_common as ic
from src.data_store import ddl
from src.data_store.schema import Tables
from tests.data_extract.fake_context import extract_config

OLD_MERCK, SGP = "0000064978", "0000310158"
OLD_PROLOGIS, AMB = "0000899881", "0001045609"
ACE, OLD_CHUBB = "0000896159", "0000020171"
DLR, DLR_LP = "0001297996", "0001494877"
PCG, PCG_UTILITY = "0001004980", "0000075488"
ACN_BERMUDA, ACN_IRELAND = "0001134538", "0001467373"
ROSTER = {"MRK": SGP, "PLD": AMB, "CB": ACE, "DLR": DLR, "PCG": PCG, "ACN": ACN_IRELAND}
UNIVERSE = tuple(ROSTER)
SENTINEL = "1900-01-01"
STAMP = pd.Timestamp("2026-01-05 09:00")
CO_REGISTRANTS = (DLR_LP, PCG_UTILITY)


def _row(ticker: str, cik: str, role: str, start: str = SENTINEL, end: str | None = None, stamp: pd.Timestamp = STAMP) -> dict[str, Any]:
    return {
        "entity_id": f"E{ROSTER[ticker]}",
        "canonical_ticker": ticker,
        "cik": cik,
        "role": role,
        "symbol": "",
        "valid_from": start,
        "valid_to": end,
        "status": "curated",
        "sources": "register" if role == "cik_window" and ticker in ("MRK", "PLD") else "roster",
        "oracle": "fixture",
        "confidence": None,
        "n_observations": 1,
        "evidence": "fixture",
        "scope_changed_at": stamp,
    }


def _lineage(*, mrk_windows: bool = True, pld_windows: bool = False, stamp: pd.Timestamp = STAMP) -> pd.DataFrame:
    """MRK's register seam (or, unset, the legal acquirer as one open roster window). PLD follows the traded security by
    default (AMB's CIK one open window, old ProLogis event-only); `pld_windows` restores the former accounting register."""
    rows = []
    if mrk_windows:
        rows += [_row("MRK", OLD_MERCK, "cik_window", end="2009-11-03"), _row("MRK", SGP, "cik_window", start="2009-11-03")]
    else:
        rows += [_row("MRK", SGP, "cik_window"), _row("MRK", OLD_MERCK, "cik_event")]
    if pld_windows:
        rows += [
            _row("PLD", OLD_PROLOGIS, "cik_window", end="2011-06-03", stamp=stamp),
            _row("PLD", AMB, "cik_window", start="2011-06-03", stamp=stamp),
        ]
    else:
        rows += [_row("PLD", AMB, "cik_window", stamp=stamp), _row("PLD", OLD_PROLOGIS, "cik_event", stamp=stamp)]
    rows += [
        _row("CB", ACE, "cik_window"),
        _row("CB", OLD_CHUBB, "cik_event"),
        _row("DLR", DLR, "cik_window"),
        _row("DLR", DLR_LP, "cik_event"),
        _row("PCG", PCG, "cik_window"),
        _row("PCG", PCG_UTILITY, "cik_event"),
        _row("ACN", ACN_BERMUDA, "cik_window", end="2009-09-01"),
        _row("ACN", ACN_IRELAND, "cik_window", start="2009-09-01"),
    ]
    frame = pd.DataFrame(rows)
    # MRK and PLD each form one entity whatever the oldest CIK is.
    frame.loc[frame["canonical_ticker"].eq("MRK"), "entity_id"] = f"E{OLD_MERCK}"
    frame.loc[frame["canonical_ticker"].eq("PLD"), "entity_id"] = f"E{OLD_PROLOGIS}"
    return frame


#: The former accounting view's PLD boundary day (removed from the shipped config by the traded-security realignment).
ACCOUNTING_PLD = sm.MergerBoundary("PLD", pd.Timestamp("2011-06-03"), None, AMB, OLD_PROLOGIS, "AMB")


def _identity(*, mergers: tuple[sm.MergerBoundary, ...] = (), **kwargs: Any) -> Any:
    tenure = pd.DataFrame(
        [
            {"symbol": t, "issuer_cik": c, "valid_from": pd.Timestamp("2000-01-01"), "valid_to": None, "n_filings": 5, "source": "form345"}
            for t, c in ROSTER.items()
        ]
    )
    roster = pd.DataFrame([{"ticker": t, "cik": c} for t, c in ROSTER.items()])
    shipped = sm.merger_boundaries(sm.load_security_manual(None))
    return build_identity(_lineage(**kwargs), tenure, roster, co_registrant_ciks=CO_REGISTRANTS, mergers=(*shipped, *mergers))


def _tx(
    accession: str,
    cik: str,
    symbol: str,
    *,
    form: str = "4",
    code: str | None = "P",
    txn: str | None = None,
    por: str | None = None,
    filed: str,
    seq: int = 1,
    owner: str = "0000000001",
    title: str = "Common Stock",
    orig: str | None = None,
) -> dict[str, Any]:
    return {
        "accession_number": accession,
        "security_type": "nonderiv",
        "row_sequence": seq,
        "ticker": symbol,
        "issuer_cik": cik,
        "issuer_name": "fixture",
        "owner_cik": owner,
        "document_type": form,
        "transaction_code": code,
        "transaction_date": pd.Timestamp(txn) if txn else pd.NaT,
        "period_of_report": pd.Timestamp(por) if por else pd.NaT,
        "filing_date": pd.Timestamp(filed),
        "original_submission_date": pd.Timestamp(orig) if orig else pd.NaT,
        "security_title": title,
        "value_usd": 1000.0,
    }


def _screen(rows: list[dict[str, Any]], identity: Any | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    return ic.screen_insider_rows(pd.DataFrame(rows), UNIVERSE, identity or _identity())


def _role(kept: pd.DataFrame, accession: str, seq: int = 1) -> str:
    return kept.set_index(["accession_number", "row_sequence"]).loc[(accession, seq), "lineage_role"]


# --------------------------------------------------------------------------- #
# economic date                                                                 #
# --------------------------------------------------------------------------- #


def test_economic_date_rules_form3_form45_amendment_and_holdings() -> None:
    """E21 and the §3.6 table: Form 3 -> period of report; Form 4/5 -> each row's transaction date
    (a late Form 5 keeps its old date); an amendment without a date inherits the original row's;
    a row without a transaction date (holdings) -> period of report."""
    frame = pd.DataFrame(
        [
            _tx("f3", SGP, "MRK", form="3", code=None, txn=None, por="2010-02-01", filed="2010-02-04"),
            _tx("f4", SGP, "MRK", txn="2010-03-01", por="2010-03-01", filed="2010-03-03"),
            _tx("f5", AMB, "AMB", form="5", code="G", txn="2011-01-15", por="2011-12-31", filed="2012-02-10"),
            _tx("orig", SGP, "SGP", txn="2009-10-01", por="2009-10-01", filed="2009-10-05", seq=2, title="Stock Option"),
            _tx("amend", SGP, "SGP", form="4/A", txn=None, por="2009-10-20", filed="2009-12-01", seq=2, title="Stock Option", orig="2009-10-05"),
            _tx("lone", SGP, "SGP", form="4/A", txn=None, por="2009-10-20", filed="2009-12-01", seq=3, title="Warrant", orig="2009-10-05"),
            _tx("hold", SGP, "MRK", form="4", code=None, txn=None, por="2010-05-01", filed="2010-05-03"),
        ]
    )
    got = dict(zip(frame["accession_number"], ic.economic_dates(frame), strict=True))
    expected = {
        "f3": "2010-02-01",
        "f4": "2010-03-01",
        "f5": "2011-01-15",
        "orig": "2009-10-01",
        "amend": "2009-10-01",
        "lone": "2009-10-20",
        "hold": "2010-05-01",
    }
    assert {k: str(pd.Timestamp(v).date()) for k, v in got.items()} == expected
    print(
        "\nSANITY: Form 3 dates on its event, Form 4/5 on each transaction (late Form 5 kept), the 4/A inherits its original row, holdings take the report date"
    )


# --------------------------------------------------------------------------- #
# roles                                                                         #
# --------------------------------------------------------------------------- #


def test_window_roles_predecessor_acquired_and_current() -> None:
    """MRK's reverse merger off the boundary day, PLD's traded view (AMB canonical throughout, old ProLogis an acquired
    target), and E17 (old Chubb, an event-only acquired target)."""
    kept, rejected = _screen(
        [
            _tx("om", OLD_MERCK, "MRK", txn="2009-06-01", filed="2009-06-03"),
            _tx("sgp", SGP, "SGP", txn="2009-06-01", filed="2009-06-03"),
            _tx("new", SGP, "MRK", txn="2010-01-05", filed="2010-01-07"),
            _tx("amb", AMB, "AMB", txn="2011-05-01", filed="2011-05-03"),
            _tx("opld", OLD_PROLOGIS, "PLD", txn="2011-05-01", filed="2011-05-03"),
            _tx("pld", AMB, "PLD", txn="2012-01-05", filed="2012-01-07"),
            _tx("chubb", OLD_CHUBB, "CB", txn="2015-06-01", filed="2015-06-03"),
            _tx("ace", ACE, "ACE", txn="2015-06-01", filed="2015-06-03"),
        ]
    )
    roles = dict(zip(kept["accession_number"], kept["lineage_role"], strict=True))
    assert rejected.empty
    assert roles == {
        "om": "canonical_predecessor",
        "sgp": "acquired_constituent",
        "new": "canonical_current",
        "amb": "canonical_current",
        "opld": "acquired_constituent",
        "pld": "canonical_current",
        "chubb": "acquired_constituent",
        "ace": "canonical_current",
    }
    assert set(kept["ticker"]) == {"MRK", "PLD", "CB"}  # roles never relabel: every row keeps its company
    print(
        "\nSANITY: a window CIK is canonical inside its declared window and acquired outside it; AMB is PLD's own history; old ProLogis and old Chubb are acquired"
    )


def test_boundary_day_rules_mrk_after_close_and_pld() -> None:
    """E22: on 2009-11-03 (closing after 16:00) a row reporting SGP is acquired; under the legal acquirer
    code A is current, D/J acquired, open-market P/S acquired, other codes follow the window; rows under the
    accounting predecessor are predecessor. AC-015 (E3): PLD 2011-06-03 has no merger metadata, so the issuer CIK
    decides whatever the symbol or code: AMB's CIK is canonical, old ProLogis (which also reported PLD) acquired."""
    day = "2009-11-03"
    kept, _ = _screen(
        [
            _tx("a", SGP, "MRK", code="A", txn=day, filed="2009-11-05"),
            _tx("a_sgp", SGP, "SGP", code="A", txn=day, filed="2009-11-05"),
            _tx("d", SGP, "MRK", code="D", txn=day, filed="2009-11-05"),
            _tx("j", SGP, "MRK", code="J", txn=day, filed="2009-11-05"),
            _tx("s", SGP, "MRK", code="S", txn=day, filed="2009-11-05"),
            _tx("m", SGP, "MRK", code="M", txn=day, filed="2009-11-05"),
            _tx("old_d", OLD_MERCK, "MRK", code="D", txn=day, filed="2009-11-05"),
            _tx("amb_day", AMB, "AMB", code="D", txn="2011-06-03", filed="2011-06-07"),
            _tx("pld_day", AMB, "PLD", code="A", txn="2011-06-03", filed="2011-06-07"),
            _tx("pld_day_d", AMB, "PLD", code="D", txn="2011-06-03", filed="2011-06-07"),
            _tx("opld_day", OLD_PROLOGIS, "PLD", code="D", txn="2011-06-03", filed="2011-06-07"),
            _tx("opld_day_a", OLD_PROLOGIS, "PLD", code="A", txn="2011-06-03", filed="2011-06-07"),
        ]
    )
    roles = dict(zip(kept["accession_number"], kept["lineage_role"], strict=True))
    assert roles == {
        "a": "canonical_current",
        "a_sgp": "acquired_constituent",
        "d": "acquired_constituent",
        "j": "acquired_constituent",
        "s": "acquired_constituent",
        "m": "canonical_current",
        "old_d": "canonical_predecessor",
        "amb_day": "canonical_current",
        "pld_day": "canonical_current",
        "pld_day_d": "canonical_current",
        "opld_day": "acquired_constituent",
        "opld_day_a": "acquired_constituent",
    }
    print(
        "\nSANITY: MRK's boundary-day rows follow its merger metadata; PLD's follow the issuer CIK alone (AMB canonical, old ProLogis acquired, even reporting PLD)"
    )


def test_amendment_and_late_form5_after_the_seam_keep_the_pre_seam_role() -> None:
    """E19: an SGP 4/A filed after the seam, dated or inheriting its original's date, stays acquired.
    E20: AMB's Form 5 filed after the seam for a pre-seam transaction keeps AMB's role, canonical (PLD's own history)."""
    kept, _ = _screen(
        [
            _tx("orig", SGP, "SGP", txn="2009-10-01", filed="2009-10-05", seq=1),
            _tx("amend_dated", SGP, "SGP", form="4/A", txn="2009-10-01", filed="2009-12-01", orig="2009-10-05", seq=1),
            _tx("amend_blank", SGP, "MRK", form="4/A", txn=None, por=None, filed="2009-12-01", orig="2009-10-05", seq=1),
            _tx("f5", AMB, "PLD", form="5", code="G", txn="2011-01-15", por="2011-12-31", filed="2012-02-10"),
        ]
    )
    roles = dict(zip(kept["accession_number"], kept["lineage_role"], strict=True))
    assert roles == {
        "orig": "acquired_constituent",
        "amend_dated": "acquired_constituent",
        "amend_blank": "acquired_constituent",
        "f5": "canonical_current",
    }
    dates = dict(zip(kept["accession_number"], kept["economic_date"], strict=True))
    assert dates["amend_blank"] == pd.Timestamp("2009-10-01")
    print("\nSANITY: amendments and late reports filed after the seam keep the role of the transaction they report")


def test_multi_cik_same_day_rows_keep_their_own_roles() -> None:
    """E23: Form 4s on 2009-11-03 under both Merck CIKs are distinct rows, each with its own CIK's role; no row is merged or dropped."""
    day = "2009-11-03"
    kept, _ = _screen(
        [_tx("x1", OLD_MERCK, "MRK", code="D", txn=day, filed="2009-11-05"), _tx("x2", SGP, "MRK", code="A", txn=day, filed="2009-11-05")]
    )
    assert len(kept) == 2
    assert dict(zip(kept["issuer_cik"], kept["lineage_role"], strict=True)) == {OLD_MERCK: "canonical_predecessor", SGP: "canonical_current"}
    print("\nSANITY: two CIKs on one day give two rows with two roles, both canonical, counted once each")


def test_co_registrant_rows_are_rejected_at_ingest() -> None:
    """E18 under D-Q2-1: Digital Realty LP and the PG&E utility rows are not stored; they are counted as
    in-scope rejects (reason `co_registrant`) so the run's exclusion warning shows them."""
    kept, rejected = _screen(
        [
            _tx("lp", DLR_LP, "DLR", txn="2015-06-01", filed="2015-06-03"),
            _tx("util", PCG_UTILITY, "PCG", txn="2022-06-01", filed="2022-06-03"),
            _tx("dlr", DLR, "DLR", txn="2015-06-01", filed="2015-06-03"),
        ]
    )
    assert list(kept["accession_number"]) == ["dlr"]
    assert set(rejected["reject_reason"]) == {"co_registrant"} and set(rejected["accession_number"]) == {"lp", "util"}
    message = ic.exclusion_message("zip 2015q2", ic.exclusion_rows(rejected))
    assert "co_registrant 2" in message
    print("\nSANITY: co-registrant rows never reach the table and are named in the exclusion warning")


def test_screen_stamps_the_three_lineage_columns_and_the_contract_carries_them() -> None:
    """`source_symbol` is the filing's own trading symbol (the resolved `ticker` overwrites only `ticker`)."""
    kept, _ = _screen([_tx("sgp", SGP, " sgp ", txn="2009-06-01", filed="2009-06-03")])
    assert kept.loc[kept.index[0], "source_symbol"] == "SGP" and kept.loc[kept.index[0], "ticker"] == "MRK"
    assert kept.loc[kept.index[0], "economic_date"] == pd.Timestamp("2009-06-01")
    assert {"source_symbol", "economic_date", "lineage_role"} <= set(ic.INSIDER_COLUMNS)
    assert "economic_date" in Tables.insider_transactions.date_type_cols
    print("\nSANITY: kept rows carry source_symbol, economic_date and lineage_role, and the saved column contract includes them")


def test_insider_ddl_block_carries_the_lineage_columns() -> None:
    schema_sql = (Path(__file__).resolve().parents[3] / "sql/schema.sql").read_text(encoding="utf-8")
    block = ddl.existing_blocks(schema_sql)["insider_transactions"]
    for column in ('"source_symbol" TEXT', '"economic_date" DATE', '"lineage_role" TEXT'):
        assert column in block, column
    assert 'PRIMARY KEY ("accession_number", "security_type", "row_sequence")' in block
    print("\nSANITY: sql/schema.sql's insider_transactions block gained the three lineage columns; the PK is unchanged")


def test_co_registrant_ciks_union_the_declared_list() -> None:
    """The flags rule finds co-registrants beside a roster window; curated subsidiaries (ACN SCA, T-Mobile USA,
    Iron Mountain Global, WFC Holdings) are declared, each with a source."""
    manual = sm.load_security_manual(None)
    declared = {entry for entry in manual.co_registrants}
    assert {"0001143908", "0001097609", "0001132694", "0000105598"} <= declared
    got = sm.co_registrant_ciks(_lineage(), None, declared=declared)
    assert declared <= got
    print("\nSANITY: the shipped config declares the four curated co-registrants and co_registrant_ciks returns them")


# --------------------------------------------------------------------------- #
# reader filter                                                                 #
# --------------------------------------------------------------------------- #


def test_the_cube_reads_canonical_insider_rows_only(sqlite_store: Any) -> None:
    rows = pd.DataFrame(
        [
            {
                "accession_number": f"a{i}",
                "security_type": "nonderiv",
                "row_sequence": 1,
                "ticker": "MRK",
                "filing_date": pd.Timestamp("2010-01-05"),
                "lineage_role": role,
            }
            for i, role in enumerate(["canonical_current", "canonical_predecessor", "acquired_constituent", None])
        ]
    )
    sqlite_store.save(Tables.insider_transactions, rows)
    got = institutional_inputs.load_insider_transactions(sqlite_store, logging.getLogger("test"), ["MRK"])
    assert got is not None and sorted(got["accession_number"]) == ["a0", "a1"]
    print("\nSANITY: the insider reader keeps canonical_predecessor and canonical_current rows; acquired and unstamped rows are not read")


# --------------------------------------------------------------------------- #
# re-stamp after a lineage change                                               #
# --------------------------------------------------------------------------- #


def _context(store: Any, tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        store=store,
        paths={"DATA_STORE": tmp_path},
        log=logging.getLogger("test.insider_lineage"),
        config=extract_config(data_extract={"years_history": 15, "manifest_full_rescan_days": 30}),
        config_dir="./configs",
    )


def _stored(identity: Any) -> pd.DataFrame:
    """Rows as an ingest under `identity` stores them (zip and EDGAR alike)."""
    kept, _ = ic.screen_insider_rows(
        pd.DataFrame(
            [
                _tx("amb", AMB, "AMB", txn="2011-05-01", filed="2011-05-03"),
                _tx("pld", AMB, "PLD", txn="2012-01-05", filed="2012-01-07"),
                _tx("opld", OLD_PROLOGIS, "PLD", txn="2011-05-01", filed="2011-05-03"),
                _tx("dlr", DLR, "DLR", txn="2015-06-01", filed="2015-06-03"),
            ]
        ),
        UNIVERSE,
        identity,
    )
    sources = {"amb": "edgar", "pld": "zip", "opld": "zip", "dlr": "edgar"}
    sourced = kept.assign(source=kept["accession_number"].map(sources))
    return sourced[[c for c in ic.INSIDER_COLUMNS if c in sourced.columns]]


def test_a_lineage_change_restamps_stored_rows_in_place(sqlite_store: Any, tmp_path: Path) -> None:
    """E31-style: removing PLD's accounting register window moves AMB's pre-merger rows into canonical history and old
    ProLogis's out of it. The re-stamp reads stored rows only (no ZIP, no EDGAR), so EDGAR-sourced rows change too; an
    unchanged lineage writes nothing (E32); a co-registrant row stored before D-Q2-1 is purged."""
    before = _identity(pld_windows=True, mergers=(ACCOUNTING_PLD,))
    stored = _stored(before)
    assert dict(zip(stored["accession_number"], stored["lineage_role"], strict=True)) == {
        "amb": "acquired_constituent",
        "pld": "canonical_current",
        "opld": "canonical_predecessor",
        "dlr": "canonical_current",
    }
    lp_row = pd.DataFrame([_tx("lp", DLR_LP, "DLR", txn="2015-06-01", filed="2015-06-03")]).assign(
        ticker="DLR", source="edgar", source_symbol="DLR", economic_date=pd.Timestamp("2015-06-01"), lineage_role="acquired_constituent"
    )
    sqlite_store.save(Tables.insider_transactions, pd.concat([stored, lp_row[[c for c in stored.columns if c in lp_row.columns]]], ignore_index=True))
    context = _context(sqlite_store, tmp_path)

    after = _identity()
    records = ins.restamp_insider_lineage(context, list(UNIVERSE), identity=after)
    got = sqlite_store.load(Tables.insider_transactions, columns=["accession_number", "lineage_role", "source"])
    roles = dict(zip(got["accession_number"], got["lineage_role"], strict=True))
    assert roles == {"amb": "canonical_current", "pld": "canonical_current", "opld": "acquired_constituent", "dlr": "canonical_current"}
    assert [(r["ticker"], r["cik"], r["rows"]) for r in records] == [("DLR", DLR_LP, 1)]

    saves: list[int] = []
    original_save = sqlite_store.save
    sqlite_store.save = lambda table, frame, *a, **k: saves.append(len(frame)) or original_save(table, frame, *a, **k)
    again = ins.restamp_insider_lineage(context, list(UNIVERSE), identity=after)
    sqlite_store.save = original_save
    assert again == [] and saves == []
    print(
        "\nSANITY: the traded view re-stamps AMB's stored EDGAR row to canonical and old ProLogis's to acquired in place, purges the co-registrant row, and a second pass writes nothing"
    )


def test_identity_propagate_restamps_insider_rows_of_changed_tickers(sqlite_store: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The re-stamp is wired into `identity-propagate`: driven by the lineage `scope_changed_at` inside the re-check window."""
    sqlite_store.save(Tables.insider_transactions, _stored(_identity(pld_windows=True, mergers=(ACCOUNTING_PLD,))))
    context = _context(sqlite_store, tmp_path)
    monkeypatch.setattr(prop, "_reparse_bulk", lambda *a, **k: {})
    after = _identity(stamp=pd.Timestamp("2026-01-12 09:00"))
    prop.propagate_identity(context, list(UNIVERSE), identity=after, as_of=pd.Timestamp("2026-01-13"))
    got = sqlite_store.load(Tables.insider_transactions, columns=["accession_number", "lineage_role"])
    roles = dict(zip(got["accession_number"], got["lineage_role"], strict=True))
    assert roles["amb"] == "canonical_current" and roles["opld"] == "acquired_constituent"
    print("\nSANITY: identity-propagate re-stamps the stored insider rows of the company whose lineage changed")


def test_a_config_edit_restamps_and_purges_on_the_next_propagation_without_a_lineage_change(
    sqlite_store: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """P32: declaring a co-registrant or a merger in the manual config moves no lineage stamp, yet the next
    propagation re-checks the tickers the config names: the PG&E utility row is purged and the boundary-day row
    reporting SGP, canonical without the MRK merger entry, becomes acquired."""
    tenure = pd.DataFrame(
        [
            {"symbol": t, "issuer_cik": c, "valid_from": pd.Timestamp("2000-01-01"), "valid_to": None, "n_filings": 5, "source": "form345"}
            for t, c in ROSTER.items()
        ]
    )
    roster = pd.DataFrame([{"ticker": t, "cik": c} for t, c in ROSTER.items()])
    before = build_identity(_lineage(), tenure, roster, co_registrant_ciks=(DLR_LP,), mergers=())
    rows = [
        _tx("sgp_day", SGP, "SGP", code="A", txn="2009-11-03", filed="2009-11-05"),
        _tx("util", PCG_UTILITY, "PCG", txn="2022-06-01", filed="2022-06-03"),
        _tx("pcg", PCG, "PCG", txn="2022-06-01", filed="2022-06-03"),
    ]
    kept, _ = ic.screen_insider_rows(pd.DataFrame(rows), UNIVERSE, before)
    assert dict(zip(kept["accession_number"], kept["lineage_role"], strict=True)) == {
        "sgp_day": "canonical_current",
        "util": "acquired_constituent",
        "pcg": "canonical_current",
    }
    sourced = kept.assign(source="zip")
    sqlite_store.save(Tables.insider_transactions, sourced[[c for c in ic.INSIDER_COLUMNS if c in sourced.columns]])
    context = _context(sqlite_store, tmp_path)
    monkeypatch.setattr(prop, "_reparse_bulk", lambda *a, **k: {})

    after = _identity()  # same lineage and stamps; the shipped merger metadata and both fixture co-registrants
    result = prop.propagate_identity(context, list(UNIVERSE), identity=after, as_of=STAMP + pd.Timedelta(days=10))
    got = sqlite_store.load(Tables.insider_transactions, columns=["accession_number", "lineage_role"])
    roles = dict(zip(got["accession_number"], got["lineage_role"], strict=True))
    print("\n=== SANITY CHECK: config-driven re-stamp (P32) ===")
    print(f"  roles after: {roles}")
    print(result.removals[["table", "ticker", "cik", "rows"]].to_string(index=False))
    assert roles == {"sgp_day": "acquired_constituent", "pcg": "canonical_current"}
    assert [(r.ticker, r.cik, r.rows) for r in result.removals.itertuples()] == [("PCG", PCG_UTILITY, 1)]
    print(
        "  OK: with no lineage change the propagation re-checks the config-named tickers: the new co-registrant's row is purged, the MRK merger entry re-stamps the SGP row."
    )


def test_a_re_registered_company_keeps_its_late_filings_canonical() -> None:
    """A domestication is not a merger: Accenture plc's Form 4 filed 2009-11-18 for a 2004 gift (before its window)
    is the company's own history, dated into the Bermuda registrant's era; only a merger's legal acquirer is acquired
    before its seam."""
    kept, _ = _screen(
        [
            _tx("late", ACN_IRELAND, "ACN", code="G", txn="2004-03-05", filed="2009-11-18"),
            _tx("old", ACN_BERMUDA, "ACN", txn="2008-03-05", filed="2008-03-07"),
            _tx("new", ACN_IRELAND, "ACN", txn="2010-03-05", filed="2010-03-07"),
            _tx("sgp", SGP, "SGP", txn="2009-06-01", filed="2009-06-03"),
        ]
    )
    roles = dict(zip(kept["accession_number"], kept["lineage_role"], strict=True))
    assert roles == {"late": "canonical_predecessor", "old": "canonical_predecessor", "new": "canonical_current", "sgp": "acquired_constituent"}
    print(
        "\nSANITY: a re-registered company's late filing stays canonical (predecessor era); the merger's legal acquirer before its seam stays acquired"
    )
