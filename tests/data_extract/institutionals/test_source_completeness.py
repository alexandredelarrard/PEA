"""Known-truth source-unit failures cannot erase or certify institutional observations."""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from zipfile import ZipFile

import pandas as pd
import pytest
from sqlalchemy import event

from src.data_extract import identity_propagate
from src.data_extract.utils.common.edgar_driver import EdgarFetch, EdgarScope, _read_unit, _walk_key
from src.data_extract.utils.common.identity import Identity, load_source_identity
from src.data_extract.utils.common.sec_io import ParseFailureError
from src.data_extract.utils.institutionals import fetch_fails_to_deliver as ftd
from src.data_extract.utils.institutionals import fetch_insider_edgar as insider_edgar
from src.data_extract.utils.institutionals import fetch_insider_transactions as insider_zip
from src.data_extract.utils.institutionals import fetch_short_interest as finra
from src.data_extract.utils.institutionals.insider_edgar_parser import extract_xml_strings
from src.data_extract.utils.institutionals.insider_store import certify_insider_source, insider_source_run
from src.data_store.schema import Tables
from src.data_store.store import DataStore
from src.utils.price_adjustments import ShareBasis
from tests.data_extract.fake_context import extract_config
from tests.data_extract.institutionals.test_short_volume_security import _identity, _master


def _archive(path: Path, members: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(path, "w") as archive:
        for name, content in members.items():
            archive.writestr(name, content)


def _bulk_members() -> dict[str, str]:
    return {
        "SUBMISSION.TSV": "ACCESSION_NUMBER\tISSUERCIK\tISSUERTRADINGSYMBOL\tDOCUMENT_TYPE\tFILING_DATE\tPERIOD_OF_REPORT\n"
        "0000920760-26-000001\t920760\tLEN\t4\t01-JUL-2026\t01-JUL-2026\n",
        "REPORTINGOWNER.TSV": "ACCESSION_NUMBER\tRPTOWNERCIK\tRPTOWNERNAME\tRPTOWNER_RELATIONSHIP\n0000920760-26-000001\t1\tKnown Owner\tOfficer\n",
        "NONDERIV_TRANS.TSV": "ACCESSION_NUMBER\tNONDERIV_TRANS_SK\tSECURITY_TITLE\tTRANS_DATE\tTRANS_CODE\tTRANS_SHARES\tTRANS_PRICEPERSHARE\n"
        "0000920760-26-000001\t1\tCommon Stock\t01-JUL-2026\tP\t12\t10\n",
        "DERIV_TRANS.TSV": "ACCESSION_NUMBER\tDERIV_TRANS_SK\tSECURITY_TITLE\n",
        "NONDERIV_HOLDING.TSV": "ACCESSION_NUMBER\tNONDERIV_HOLDING_SK\tSECURITY_TITLE\n",
        "DERIV_HOLDING.TSV": "ACCESSION_NUMBER\tDERIV_HOLDING_SK\tSECURITY_TITLE\n",
        "FOOTNOTES.TSV": "ACCESSION_NUMBER\tFOOTNOTE_ID\tFOOTNOTE_TXT\n",
    }


def _file(short: str = "1", total: str = "2", count: int = 1) -> str:
    return "Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market\n" + f"20260701|LEN|{short}|0|{total}|Q,N\n{count}\n"


@pytest.mark.parametrize("payload", ["not|a source", _file("9", "10", 2), _file("1e999", "10")], ids=["header", "count", "overflow"])
def test_failed_finra_source_preserves_last_good_rows(sqlite_store: DataStore, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, payload: str) -> None:
    day = pd.Timestamp("2026-07-01")
    identity = _identity()
    raw = finra.stamp_short_volume(finra._parse_regsho(_file()), identity, ["LEN"])
    sqlite_store.save(Tables.sec_short_volume_security, raw[list(finra.SECURITY_COLUMNS)])
    expected = pd.DataFrame({"ticker": ["LEN"], "date": [day], "short_volume": [1.0], "total_volume": [2.0]})
    sqlite_store.save(Tables.short_interest, expected)
    before_raw = sqlite_store.load(Tables.sec_short_volume_security)
    assert before_raw is not None
    before = sqlite_store.load(Tables.short_interest)
    assert before is not None
    context = cast(Any, SimpleNamespace(store=sqlite_store, paths={"DATA_STORE": tmp_path}, config=extract_config(), log=logging.getLogger(__name__)))
    monkeypatch.setattr(finra, "_plan_days", lambda *args, **kwargs: pd.DatetimeIndex([day]))
    monkeypatch.setattr(finra, "_fetch_day", lambda *args, **kwargs: payload)
    monkeypatch.setattr(finra, "restamp_short_volume", lambda *args, **kwargs: [])

    finra.fetch_short_interest(context, ["LEN"], pause=0.0, identity=identity, all_tickers=["LEN"])

    stored_frame = sqlite_store.load(Tables.short_interest, optional=True)
    assert stored_frame is not None
    pd.testing.assert_frame_equal(stored_frame, before)
    stored_frame = sqlite_store.load(Tables.sec_short_volume_security, optional=True)
    assert stored_frame is not None
    pd.testing.assert_frame_equal(stored_frame, before_raw)
    print("SANITY: malformed/truncated/nonfinite FINRA source preserves both last good tables and cannot certify an absence.")


def test_ftd_same_key_conflicting_source_periods_refuse() -> None:
    lines = pd.DataFrame(
        {
            "cusip": ["037833100", "037833100"],
            "date": pd.to_datetime(["2024-01-15"] * 2),
            "source_symbol": ["AAPL", "AAPL"],
            "fails_quantity": [11.0, 12.0],
            "period": ["202401a", "202401b"],
        }
    )
    with pytest.raises(ValueError, match="conflict"):
        ftd._unique_keys(lines, "known conflicting ZIPs")
    print("SANITY: incompatible same-key FTD ZIP values refuse before any latest/first-wins persistence.")


def test_insider_parse_failure_remains_retryable() -> None:
    def failed(ticker: str, cik: str, stamp: Any, scope: Any) -> dict[Any, pd.DataFrame]:
        raise ParseFailureError("known unreadable ownership XML")

    fetch = EdgarFetch(tables=(Tables.insider_transactions,), parse=failed)
    stamp = cast(Any, SimpleNamespace(accession_number="0000000001-26-000001", filed=pd.Timestamp("2026-07-01"), form="4", cik="0000000001"))
    frames, marker = _read_unit(fetch, cast(Any, SimpleNamespace()), "AAA", "0000000001", stamp)
    assert frames is None and not marker, "Failed parsing must remain eligible for retry, never become a valid empty filing."
    print("SANITY: failed insider XML creates no completed marker or success proof.")


def test_current_finra_fractional_quantities_remain_valid() -> None:
    rows = finra._parse_regsho(_file("0.123456", "1.234567"))
    assert rows["short_volume"].iloc[0] == 0.123456
    assert rows["total_volume"].iloc[0] == 1.234567
    print("SANITY: valid finite nonnegative current FINRA quantities retain all six fractional decimal places.")


def test_finra_legacy_layout_and_true_empty_file_remain_valid() -> None:
    legacy = "Date|Symbol|ShortVolume|TotalVolume|Market\n20090102|LEN|1|2|Q,N\n1\n"
    rows = finra._parse_regsho(legacy)
    assert rows["short_volume"].tolist() == [1.0] and rows["short_exempt_volume"].isna().all()
    assert finra._parse_regsho(_file().splitlines()[0] + "\n0\n").empty
    print("SANITY: legitimate five-column historical and header-plus-zero-trailer files pass full line accounting.")


@pytest.mark.parametrize("row", ["20260701|037833100|AAPL|bad|APPLE|1", "bad line", "20260701|037833100|AAPL|1e999|APPLE|1"])
def test_malformed_ftd_lines_cannot_be_a_complete_source(row: str) -> None:
    with pytest.raises(ValueError):
        ftd._parse_ftd_lines("SETTLEMENT DATE|CUSIP|SYMBOL|QUANTITY (FAILS)|DESCRIPTION|PRICE\n" + row + "\n")
    print("SANITY: an FTD malformed or nonfinite source row refuses the complete period before absence certification.")


def test_checked_finra_absence_has_no_fabricated_total(sqlite_store: DataStore, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    day = pd.Timestamp("2026-07-01")
    context = cast(Any, SimpleNamespace(store=sqlite_store, paths={"DATA_STORE": tmp_path}, config=extract_config(), log=logging.getLogger(__name__)))
    monkeypatch.setattr(finra, "_plan_days", lambda *args, **kwargs: pd.DatetimeIndex([day]))
    monkeypatch.setattr(finra, "_fetch_day", lambda *args, **kwargs: _file().splitlines()[0] + "\n0\n")
    monkeypatch.setattr(finra, "restamp_short_volume", lambda *args, **kwargs: [])
    identity = _identity()
    for table, frame in identity.source_cohort.items():
        if not frame.empty:
            sqlite_store.save(table, frame)
    finra.fetch_short_interest(context, ["LEN"], pause=0.0, identity=identity, all_tickers=["LEN"])
    rows = sqlite_store.load(Tables.short_interest)
    assert rows is not None and rows["short_volume"].tolist() == [0.0]
    assert rows["total_volume"].isna().all() and rows["source_checked_at"].notna().all()
    assert sqlite_store.row_count(Tables.sec_short_volume_security) == 0
    print("SANITY: fully parsed mapped FINRA absence is short flow zero, denominator NULL and no fabricated raw row.")


def test_normal_finra_retry_recovers_old_null_quantity_once(sqlite_store: DataStore, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    context = cast(Any, SimpleNamespace(store=sqlite_store, paths={"DATA_STORE": tmp_path}, config=extract_config(), log=logging.getLogger(__name__)))
    identity = _identity()
    for table, frame in identity.source_cohort.items():
        if not frame.empty:
            sqlite_store.save(table, frame)
    old, latest, as_of = pd.Timestamp("2026-07-01"), pd.Timestamp("2027-01-29"), pd.Timestamp("2027-02-01")
    sqlite_store.save(Tables.prices, pd.DataFrame({"ticker": "LEN", "date": [old, latest], "close": 100.0, "volume": 1000.0}))
    sqlite_store.save(
        Tables.short_interest,
        pd.DataFrame(
            {
                "ticker": "LEN",
                "date": [old, latest],
                "short_volume": [float("nan"), 2.0],
                "total_volume": [float("nan"), 4.0],
                "source_checked_at": pd.to_datetime(pd.Series([None, "2027-01-30"])),
            }
        ),
    )
    requested: list[pd.Timestamp] = []

    def served(day: pd.Timestamp, session: Any) -> str:
        assert session is not None
        requested.append(day)
        return _file().splitlines()[0] + "\n0\n"

    monkeypatch.setattr(finra, "_fetch_day", served)
    planned = finra._plan_days(context, ["LEN"], 2, False, as_of)
    assert old in planned, "an old NULL source operand was hidden by the stored date/frontier"
    finra.fetch_short_interest(context, ["LEN"], years_history=2, pause=0.0, identity=identity, as_of=as_of, all_tickers=["LEN"])
    rows = sqlite_store.load(Tables.short_interest, where={"ticker": "LEN", "date": old})
    assert rows is not None and rows["short_volume"].eq(0).all() and rows["source_checked_at"].notna().all()
    finra.fetch_short_interest(context, ["LEN"], years_history=2, pause=0.0, identity=identity, as_of=as_of, all_tickers=["LEN"])
    assert requested.count(old) == 1
    print(
        "SANITY: normal FINRA planning requeues an old NULL quantity outside the 7-day refresh window, and successful source refill removes that historical retry key."
    )


def test_finra_null_retries_respect_document_floor_completed_session_and_calendar(sqlite_store: DataStore, tmp_path: Path) -> None:
    context = cast(Any, SimpleNamespace(store=sqlite_store, paths={"DATA_STORE": tmp_path}, config=extract_config(), log=logging.getLogger(__name__)))
    old, latest, future = pd.Timestamp("2026-07-01"), pd.Timestamp("2027-01-29"), pd.Timestamp("2027-02-02")
    before_floor, weekend, as_of = pd.Timestamp("2010-01-04"), pd.Timestamp("2026-07-04"), pd.Timestamp("2027-02-01")
    calendar = pd.DatetimeIndex([old, latest, future])
    sqlite_store.save(Tables.prices, pd.DataFrame({"ticker": "LEN", "date": calendar, "close": 100.0, "volume": 1000.0}))
    sqlite_store.save(
        Tables.short_interest,
        pd.DataFrame(
            {
                "ticker": "LEN",
                "date": [before_floor, old, weekend, latest, future],
                "short_volume": [float("nan"), float("nan"), float("nan"), 2.0, float("nan")],
                "total_volume": 4.0,
            }
        ),
    )
    planned = finra._plan_days(context, ["LEN"], 2, False, as_of)
    assert old in planned and not any(day in planned for day in (before_floor, weekend, future))
    assert planned.isin(calendar).all() and (planned <= finra.last_completed_session(as_of)).all()
    print("SANITY: historical NULL retry keys stay inside the declared source/document floor, completed-session boundary and expected calendar.")


def test_normal_ftd_retry_recovers_old_null_quantity_once(sqlite_store: DataStore, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    context = cast(Any, SimpleNamespace(store=sqlite_store, paths={"DATA_STORE": tmp_path}, config=extract_config(), log=logging.getLogger(__name__)))
    identity = _identity()
    for table, frame in identity.source_cohort.items():
        if not frame.empty:
            sqlite_store.save(table, frame)
    good = "SETTLEMENT DATE|CUSIP|SYMBOL|QUANTITY (FAILS)|DESCRIPTION|PRICE\n20260701|526057104|LEN|12|LEN|10\n"
    raw = ftd.stamp_lines(ftd._parse_ftd_lines(good).assign(period="202607a"), ftd.prepare_master(_master()))
    sqlite_store.save(Tables.sec_fails_to_deliver_security, raw[list(ftd.SECURITY_COLUMNS)])
    sqlite_store.save(
        Tables.sec_fails_to_deliver,
        pd.DataFrame(
            {
                "ticker": ["LEN"],
                "date": pd.to_datetime(["2026-07-01"]),
                "fails_quantity": [float("nan")],
                "fails_value": [float("nan")],
                "period": ["202607a"],
                "source_checked_at": pd.Series(pd.NaT, index=[0], dtype="datetime64[ns]"),
            }
        ),
    )
    path = tmp_path / context.config.local.paths.fails_deliver / "cnsfails202607a.zip"
    _archive(path, {"fails.txt": good})
    requested: list[str] = []
    reader = ftd._read_period

    def observed(ctx: Any, cache: Path, period: str, *, download: bool) -> pd.DataFrame | None:
        requested.append(period)
        return reader(ctx, cache, period, download=download)

    monkeypatch.setattr(ftd, "_read_period", observed)
    ftd.fetch_fails_to_deliver(context, ["LEN"], as_of=pd.Timestamp("2027-02-01"), all_tickers=["LEN"], identity=identity)
    assert requested == ["202607a"], "normal FTD planning skipped a NULL derived period because its raw rows and ZIP already existed"
    rows = sqlite_store.load(Tables.sec_fails_to_deliver, where={"ticker": "LEN"})
    assert rows is not None and rows["fails_quantity"].tolist() == [12.0] and rows["source_checked_at"].notna().all()
    ftd.fetch_fails_to_deliver(context, ["LEN"], as_of=pd.Timestamp("2027-02-01"), all_tickers=["LEN"], identity=identity)
    assert requested == ["202607a"]
    print(
        "SANITY: normal FTD planning rereads a complete original for an old NULL quantity outside the 7-day scope window; successful paired refill is idempotent."
    )


@pytest.mark.parametrize("holding_value", [None, "1234.5"])
def test_xml_holding_payload_and_notes_are_not_discarded(holding_value: str | None) -> None:
    xml = """<ownershipDocument><documentType>3</documentType><issuer><issuerCik>1</issuerCik></issuer>
    <periodOfReport>2026-07-01</periodOfReport><nonDerivativeTable><nonDerivativeHolding>
    <securityTitle><value>Common Stock</value></securityTitle><postTransactionAmounts>
    <sharesOwnedFollowingTransaction><value>123</value><footnoteId id="F1"/></sharesOwnedFollowingTransaction>
    </postTransactionAmounts></nonDerivativeHolding></nonDerivativeTable><footnotes><footnote id="F1">Known holding note</footnote></footnotes>
    </ownershipDocument>"""
    if holding_value is not None:
        xml = xml.replace(
            "</postTransactionAmounts>",
            f"<valueOwnedFollowingTransaction><value>{holding_value}</value></valueOwnedFollowingTransaction></postTransactionAmounts>",
        )
    rows, _, notes = extract_xml_strings(xml, "known-holding")
    assert rows["security_type"].tolist() == ["nonderiv_holding"]
    assert rows["shares_owned_after"].tolist() == ["123"] and rows["footnote_ids"].tolist() == ["F1"]
    assert rows["transaction_date"].isna().all() and rows["transaction_code"].isna().all() and rows["shares"].isna().all()
    assert notes["footnote_text"].tolist() == ["Known holding note"]
    assert rows["total_value"].isna().all() if holding_value is None else rows["total_value"].tolist() == [holding_value]
    print("SANITY: actual XML holdings/notes survive inventory parsing without inventing a trade date, code or flow quantity.")


def test_bulk_holding_payload_uses_its_actual_key_and_fields() -> None:
    submission = pd.DataFrame({"ACCESSION_NUMBER": ["holding"], "ISSUERCIK": ["1"], "DOCUMENT_TYPE": ["3"]})
    holding = pd.DataFrame(
        {
            "ACCESSION_NUMBER": ["holding"],
            "DERIV_HOLDING_SK": ["11"],
            "SECURITY_TITLE": ["Option"],
            "SHRS_OWND_FOLWNG_TRANS": ["123"],
            "EXERCISE_DATE": ["01-JUL-2026"],
            "SHRS_OWND_FOLWNG_TRANS_FN": ["F1"],
        }
    )
    rows, _ = insider_zip.extract_bulk_strings(submission, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), holding)
    assert rows["security_type"].tolist() == ["deriv_holding"] and rows["row_sequence"].tolist() == [1]
    assert rows["shares_owned_after"].tolist() == ["123"] and rows["exercise_date"].tolist() == ["01-JUL-2026"]
    assert rows["transaction_date"].isna().all() and rows["transaction_code"].isna().all() and rows["shares"].isna().all()
    print("SANITY: actual bulk holding key, holdings and holding-specific exercise spelling survive without fabricated transactions.")


@pytest.mark.parametrize("payload", [_file("3", "2"), _file("1e308", "1e308").rsplit("\n1\n", 1)[0] + "\n20260701|LEN|1e308|0|1e308|Q,N\n2\n"])
def test_finra_staged_quantities_must_remain_finite_and_within_total(payload: str) -> None:
    with pytest.raises(ValueError):
        finra._parse_regsho(payload)
    print("SANITY: a short quantity above total or a finite-input duplicate-symbol overflow cannot receive source proof.")


def test_finra_class_conversion_overflow_refuses_before_source_commit() -> None:
    master = _master()
    master.loc[master["source_symbol"].eq("LEN"), "conversion_ratio"] = 10.0
    identity = _identity(master)
    stamped = finra.stamp_short_volume(finra._parse_regsho(_file("1e308", "1e308")), identity, ["LEN"])
    with pytest.raises(ValueError):
        finra.ticker_rows(stamped, ["LEN"])
    print("SANITY: finite source quantities cannot become certified infinite class-converted aggregate quantities.")


def test_unresolved_relevant_class_cannot_create_checked_company_absence() -> None:
    master = _master()
    clash = master.loc[master["source_symbol"].eq("LENB")].assign(security_id="C999999999", cusip="999999999")
    identity = _identity(pd.concat([master, clash], ignore_index=True))
    rows = finra._parse_regsho(_file().replace("|LEN|", "|LENB|"))
    stamped = finra.stamp_short_volume(rows, identity, ["LEN"])
    grain = finra._checked_regsho_grain(stamped, [pd.Timestamp("2026-07-01")], identity, ["LEN"])
    assert grain.empty or grain.loc[grain["ticker"].eq("LEN"), "source_checked_at"].isna().all()
    print("SANITY: a mapped absent canonical class plus an ambiguous present secondary class cannot certify zero or a partial company total.")


def test_ftd_verified_absence_uses_only_actual_served_dates() -> None:
    source = ftd._parse_ftd_lines(
        "SETTLEMENT DATE|CUSIP|SYMBOL|QUANTITY (FAILS)|DESCRIPTION|PRICE\n20260701|526057104|LEN|5|LEN|10\n20260703|000000000|OTHER|1|OTHER|1\n"
    ).assign(period="202607a")
    basis = ShareBasis(pd.DataFrame(columns=["ticker", "date", "value"]), pd.DataFrame(), pd.DataFrame(), {})
    grain = ftd._checked_ftd_grain(source, ftd.prepare_master(_master()), frozenset(["LEN"]), basis)
    assert grain["date"].tolist() == [pd.Timestamp("2026-07-01"), pd.Timestamp("2026-07-03")]
    assert grain["fails_quantity"].tolist() == [5.0, 0.0] and grain["source_checked_at"].notna().all()
    print("SANITY: mapped FTD absence is zero only on dates actually served in the fully parsed ZIP, with no synthetic missing weekday.")


def test_ftd_class_ratio_and_raw_quantity_use_settlement_basis() -> None:
    source = ftd._parse_ftd_lines(
        "SETTLEMENT DATE|CUSIP|SYMBOL|QUANTITY (FAILS)|DESCRIPTION|PRICE\n"
        "20100122|084670108|BRKA|10|BERKSHIRE CLASS A|100000\n"
        "20100122|084670702|BRKB|100|BERKSHIRE CLASS B|50\n"
    ).assign(period="201001b")
    master = ftd.prepare_master(_master())
    events = pd.DataFrame({"ticker": ["BRK-B"], "date": pd.to_datetime(["2010-01-21"]), "value": [50.0]})
    basis = ShareBasis(events, events, events, {})
    stamped = ftd.stamp_lines(source, master)
    stamped["market_symbol"] = stamped["source_symbol"].map({"BRKA": "BRK-A", "BRKB": "BRK-B"})
    grain = ftd.ticker_rows(stamped, ["BRK-B"], basis)
    assert grain["fails_quantity"].tolist() == [15100.0]
    assert grain["fails_value"].tolist() == [1005000.0]
    print(
        "SANITY: actual class units are rebased settlement→identity date→canonical settlement once; the Class B split is not doubled and raw USD stays unchanged."
    )


@pytest.mark.parametrize("ratio", [float("nan"), 0.0, float("inf")])
def test_missing_or_invalid_secondary_factor_cannot_create_checked_absence(ratio: float) -> None:
    master = _master()
    master.loc[master["source_symbol"].eq("LENB"), "conversion_ratio"] = ratio
    identity = _identity(master)
    empty = pd.DataFrame(columns=[*finra.SECURITY_COLUMNS, "conversion_ratio"])
    assert finra._checked_regsho_grain(empty, [pd.Timestamp("2026-07-01")], identity, ["LEN"]).empty
    source = ftd._parse_ftd_lines("SETTLEMENT DATE|CUSIP|SYMBOL|QUANTITY (FAILS)|DESCRIPTION|PRICE\n20260701|000000000|OTHER|1|OTHER|1\n").assign(
        period="202607a"
    )
    assert ftd._checked_ftd_grain(source, ftd.prepare_master(master), frozenset(["LEN"]), None).empty
    print(
        "SANITY: missing, zero or infinite secondary-class conversion remains unknown in FINRA and FTD; it cannot become class factor1 or checked absence0."
    )


@pytest.mark.parametrize("bad", ["malformed", "wrong-period"])
def test_ftd_normal_planner_retries_bad_cached_payload(sqlite_store: DataStore, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, bad: str) -> None:
    context = cast(Any, SimpleNamespace(store=sqlite_store, paths={"DATA_STORE": tmp_path}, config=extract_config(), log=logging.getLogger(__name__)))
    identity = _identity()
    for table, frame in identity.source_cohort.items():
        if not frame.empty:
            sqlite_store.save(table, frame)
    good = "SETTLEMENT DATE|CUSIP|SYMBOL|QUANTITY (FAILS)|DESCRIPTION|PRICE\n20260701|526057104|LEN|12|LEN|10\n"
    raw = ftd.stamp_lines(ftd._parse_ftd_lines(good).assign(period="202607a"), ftd.prepare_master(_master()))
    sqlite_store.save(Tables.sec_fails_to_deliver_security, raw[list(ftd.SECURITY_COLUMNS)])
    sqlite_store.save(
        Tables.sec_fails_to_deliver,
        pd.DataFrame(
            {
                "ticker": ["LEN"],
                "date": pd.to_datetime(["2026-07-01"]),
                "fails_quantity": [12.0],
                "fails_value": [120.0],
                "period": ["202607a"],
                "source_checked_at": pd.to_datetime(["2026-07-03"]),
            }
        ),
    )
    before_raw = sqlite_store.load(Tables.sec_fails_to_deliver_security)
    assert before_raw is not None
    before = sqlite_store.load(Tables.sec_fails_to_deliver)
    assert before is not None
    path = tmp_path / context.config.local.paths.fails_deliver / "cnsfails202607a.zip"
    payload = good.replace("|12|", "|bad|") if bad == "malformed" else good.replace("20260701", "20260720")
    _archive(path, {"fails.txt": payload})
    requests: list[str] = []

    def download(ctx: Any, url: str, target: Path, timeout: int) -> int:
        requests.append(url)
        _archive(target, {"fails.txt": good})
        return 200

    monkeypatch.setattr("src.data_extract.utils.common.bulk_cache.download", download)
    monkeypatch.setattr(ftd, "_periods", lambda *args: ["202607a"])
    monkeypatch.setattr(ftd, "_changed_scope", lambda *args: (frozenset(), frozenset()))
    assert ftd.download_fails_to_deliver(context, full=True, as_of=pd.Timestamp("2026-07-30")) == 0
    stored_frame = sqlite_store.load(Tables.sec_fails_to_deliver_security)
    assert stored_frame is not None
    pd.testing.assert_frame_equal(stored_frame, before_raw)
    stored_frame = sqlite_store.load(Tables.sec_fails_to_deliver)
    assert stored_frame is not None
    pd.testing.assert_frame_equal(stored_frame, before)
    assert not path.exists() and not requests
    assert ftd.download_fails_to_deliver(context, as_of=pd.Timestamp("2026-07-30")) == 1
    assert len(requests) == 1
    recovered_raw = sqlite_store.load(Tables.sec_fails_to_deliver_security)
    assert recovered_raw is not None
    assert recovered_raw["fails_quantity"].tolist() == [12.0]
    print(
        "SANITY: a valid ZIP with malformed/wrong-period FTD content preserves both good tables and proof; the next normal planner downloads the required original again."
    )


@pytest.mark.parametrize("concurrent", [False, True], ids=["coherent", "stale-copy"])
@pytest.mark.parametrize("proof_column", [False, True], ids=["physical-legacy", "proof-column"])
def test_pending_ftd_download_cannot_restore_a_stale_derived_cohort(
    sqlite_store: DataStore,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    concurrent: bool,
    proof_column: bool,
) -> None:
    context = cast(Any, SimpleNamespace(store=sqlite_store, paths={"DATA_STORE": tmp_path}, config=extract_config(), log=logging.getLogger(__name__)))
    for table, frame in _identity().source_cohort.items():
        if not frame.empty:
            sqlite_store.save(table, frame)
    text = "SETTLEMENT DATE|CUSIP|SYMBOL|QUANTITY (FAILS)|DESCRIPTION|PRICE\n20260701|526057104|LEN|12|LEN|10\n"
    raw = ftd.stamp_lines(ftd._parse_ftd_lines(text).assign(period="202607a"), ftd.prepare_master(_master()))[list(ftd.SECURITY_COLUMNS)]
    grain = pd.DataFrame(
        {
            "ticker": ["LEN"],
            "date": pd.to_datetime(["2026-07-01"]),
            "fails_quantity": [12.0],
            "fails_value": [120.0],
            "period": ["202607a"],
            "source_checked_at": pd.to_datetime(["2026-07-03"]),
        }
    )
    sqlite_store.save(Tables.sec_fails_to_deliver_security, raw)
    sqlite_store.save(Tables.sec_fails_to_deliver, grain if proof_column else grain.drop(columns="source_checked_at"))
    path = tmp_path / context.config.local.paths.fails_deliver / "cnsfails202607a.zip"
    _archive(path, {"fails.txt": text})
    newer_raw = pd.concat([raw.assign(fails_quantity=99.0), raw.assign(date=pd.Timestamp("2026-07-02"), fails_quantity=7.0)], ignore_index=True)
    newer_grain = pd.concat(
        [grain.assign(fails_quantity=99.0, fails_value=990.0), grain.assign(date=pd.Timestamp("2026-07-02"), fails_quantity=7.0, fails_value=70.0)],
        ignore_index=True,
    )
    load = sqlite_store.load
    changed = False
    recovered: dict[Any, pd.DataFrame] = {}

    def interleaved(table: Any, **kwargs: Any) -> pd.DataFrame | None:
        nonlocal changed
        observed = load(table, **kwargs)
        if concurrent and table == Tables.sec_fails_to_deliver and not changed:
            changed = True
            sqlite_store.replace_institutional_source(
                {Tables.sec_fails_to_deliver_security: newer_raw, Tables.sec_fails_to_deliver: newer_grain},
                {Tables.sec_fails_to_deliver_security: {"period": "202607a"}, Tables.sec_fails_to_deliver: {"period": "202607a"}},
            )
            for member in (Tables.sec_fails_to_deliver_security, Tables.sec_fails_to_deliver):
                recovered_rows = load(member)
                assert recovered_rows is not None
                recovered[member] = recovered_rows
        return observed

    monkeypatch.setattr(sqlite_store, "load", interleaved)
    monkeypatch.setattr(ftd, "_periods", lambda *args: ["202607a"])
    monkeypatch.setattr(ftd, "_changed_scope", lambda *args: (frozenset(), frozenset()))
    if concurrent:
        with pytest.raises(ValueError, match="source cohort changed"):
            ftd.download_fails_to_deliver(context, full=True, as_of=pd.Timestamp("2026-07-30"))
        for member, expected in recovered.items():
            stored_frame = load(member)
            assert stored_frame is not None
            pd.testing.assert_frame_equal(stored_frame, expected)
            assert expected["fails_quantity"].tolist() == [99.0, 7.0]
    else:
        assert ftd.download_fails_to_deliver(context, full=True, as_of=pd.Timestamp("2026-07-30")) == 1
        pending_rows = load(Tables.sec_fails_to_deliver)
        assert pending_rows is not None
        assert pending_rows["source_checked_at"].isna().all()
    print(
        "SANITY: the normal pending download commits a coherent raw/unknown pair, but refuses a stale derived copy and preserves every newer recovered row and proof."
    )


@pytest.mark.parametrize("bad", ["missing-member", "invalid-row-key"])
def test_insider_normal_planner_retries_invalid_cached_quarter(
    sqlite_store: DataStore, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, bad: str
) -> None:
    context, identity = _parser_identity_context(sqlite_store, tmp_path)
    monkeypatch.setattr(insider_zip, "load_identity", lambda ctx: identity)
    monkeypatch.setattr(insider_zip, "quarter_periods", lambda *args: ["2026q3"])
    monkeypatch.setattr(insider_zip, "_screen_stored_rows", lambda *args: 0)
    members = _bulk_members()
    path = tmp_path / context.config.local.paths.insider_transactions / "2026q3.zip"
    _archive(path, members)
    assert insider_zip.fetch_insider_transactions(context, ["LEN"], 1, all_tickers=["LEN"], as_of=pd.Timestamp("2026-10-01")) == 1
    sqlite_store.save(
        Tables.insider_transactions,
        pd.DataFrame(
            {
                "ticker": ["LEN"],
                "accession_number": ["0000920760-26-000001"],
                "security_type": ["nonderiv"],
                "row_sequence": [1],
                "inventory_checked_through": pd.to_datetime(["2026-09-30"]),
            }
        ),
    )
    before = sqlite_store.load(Tables.insider_transactions)
    assert before is not None
    damaged = dict(members)
    if bad == "missing-member":
        damaged.pop("DERIV_TRANS.TSV")
    else:
        damaged["NONDERIV_TRANS.TSV"] = damaged["NONDERIV_TRANS.TSV"].replace("\t1\tCommon Stock", "\tbad\tCommon Stock")
    _archive(path, damaged)
    requests: list[str] = []

    def download(ctx: Any, url: str, target: Path, timeout: int) -> int:
        requests.append(url)
        _archive(target, members)
        return 200

    monkeypatch.setattr("src.data_extract.utils.common.bulk_cache.download", download)
    assert insider_zip.fetch_insider_transactions(context, ["LEN"], 1, reparse=True, all_tickers=["LEN"], as_of=pd.Timestamp("2026-10-01")) == 0
    stored_frame = sqlite_store.load(Tables.insider_transactions)
    assert stored_frame is not None
    pd.testing.assert_frame_equal(stored_frame, before)
    assert not path.exists() and not requests
    assert insider_zip.fetch_insider_transactions(context, ["LEN"], 1, all_tickers=["LEN"], as_of=pd.Timestamp("2026-10-01")) == 1
    recovered_rows = sqlite_store.load(Tables.insider_transactions)
    assert recovered_rows is not None
    assert len(requests) == 1 and recovered_rows["shares"].tolist() == [12.0]
    print(
        "SANITY: incomplete or semantically invalid insider ZIP preserves good payload/proof and is downloaded again through the next normal production worklist."
    )


@pytest.mark.parametrize(
    "body",
    [
        "<nonDerivativeTable><nonDerivativeTransactoin/></nonDerivativeTable>",
        '<footnotes><footnote id="F1">one</footnote><footnote id="F1">two</footnote></footnotes>',
        '<nonDerivativeTable><nonDerivativeHolding><securityTitle><footnoteId id="F1"/></securityTitle></nonDerivativeHolding></nonDerivativeTable>',
        "<nonDerivativeTable><nonDerivativeTransaction><transactionAmounts><transactionShares><value>1e999</value></transactionShares></transactionAmounts></nonDerivativeTransaction></nonDerivativeTable>",
        "<nonDerivativeTable><nonDerivativeHolding><postTransactionAmounts><valueOwnedFollowingTransaction><value>NaN</value></valueOwnedFollowingTransaction></postTransactionAmounts></nonDerivativeHolding></nonDerivativeTable>",
        "<nonDerivativeTable><nonDerivativeHolding><postTransactionAmounts><valueOwnedFollowingTransaction><value>Inf</value></valueOwnedFollowingTransaction></postTransactionAmounts></nonDerivativeHolding></nonDerivativeTable>",
        "<nonDerivativeTable><nonDerivativeHolding><postTransactionAmounts><valueOwnedFollowingTransaction><value>bad</value></valueOwnedFollowingTransaction></postTransactionAmounts></nonDerivativeHolding></nonDerivativeTable>",
    ],
)
def test_incomplete_xml_payload_cannot_be_valid_empty_or_complete(body: str) -> None:
    xml = "<ownershipDocument><issuer><issuerCik>920760</issuerCik></issuer>" + body + "</ownershipDocument>"
    with pytest.raises(ValueError):
        extract_xml_strings(xml, "0000920760-26-000001")
    print(
        "SANITY: unknown ownership row kinds, ambiguous/missing notes and nonfinite quantities refuse source completeness instead of becoming empty or reduced payloads."
    )


def test_bulk_invalid_economic_number_is_retryable_source_failure(tmp_path: Path) -> None:
    members = _bulk_members()
    members["NONDERIV_TRANS.TSV"] = members["NONDERIV_TRANS.TSV"].replace("\t12\t10", "\t1e999\t10")
    path = tmp_path / "2026q3.zip"
    _archive(path, members)
    assert insider_zip._read_tables(path) is None and not path.exists()
    print("SANITY: a nonfinite bulk economic operand cannot be silently coerced to missing and certified; its source unit remains retryable.")


@pytest.mark.parametrize("fail", [False, True], ids=["unchanged", "rollback"])
def test_finra_restamp_preserves_or_atomically_rolls_back_source_pair(sqlite_store: DataStore, tmp_path: Path, fail: bool) -> None:
    context = cast(Any, SimpleNamespace(store=sqlite_store, paths={"DATA_STORE": tmp_path}, config=extract_config(), log=logging.getLogger(__name__)))
    identity = _identity()
    for table, frame in identity.source_cohort.items():
        if not frame.empty:
            sqlite_store.save(table, frame)
    raw = finra.stamp_short_volume(finra._parse_regsho(_file()), identity, ["LEN"])[list(finra.SECURITY_COLUMNS)]
    if fail:
        raw["security_id"] = "known-stale-stamp"
    sqlite_store.save(Tables.sec_short_volume_security, raw)
    sqlite_store.save(
        Tables.short_interest,
        pd.DataFrame(
            {
                "ticker": ["LEN"],
                "date": pd.to_datetime(["2026-07-01"]),
                "short_volume": [1.0],
                "total_volume": [2.0],
                "source_checked_at": pd.to_datetime(["2026-07-03"]),
            }
        ),
    )
    before_raw = sqlite_store.load(Tables.sec_short_volume_security)
    assert before_raw is not None
    before = sqlite_store.load(Tables.short_interest)
    assert before is not None
    writes: list[str] = []

    def reject(conn: Any, cursor: Any, statement: str, parameters: Any, execution: Any, many: bool) -> None:
        if statement.startswith("INSERT INTO"):
            writes.append(statement)
        if fail and statement.startswith(f"INSERT INTO {Tables.short_interest.name}"):
            raise RuntimeError("controlled derived insertion failure")

    event.listen(sqlite_store.engine, "before_cursor_execute", reject)
    try:
        if fail:
            with pytest.raises(RuntimeError, match="controlled"):
                finra.restamp_short_volume(context, ["LEN"], ["LEN"], identity=identity, others=frozenset())
        else:
            assert finra.restamp_short_volume(context, ["LEN"], ["LEN"], identity=identity, others=frozenset()) == []
            assert not writes, "Unchanged normal restamp must preserve its already joint-certified source cohort without rewriting."
    finally:
        event.remove(sqlite_store.engine, "before_cursor_execute", reject)
    stored_frame = sqlite_store.load(Tables.sec_short_volume_security)
    assert stored_frame is not None
    pd.testing.assert_frame_equal(stored_frame, before_raw)
    stored_frame = sqlite_store.load(Tables.short_interest)
    assert stored_frame is not None
    pd.testing.assert_frame_equal(stored_frame, before)
    print(
        "SANITY: the actual FINRA restamp skips an unchanged source cohort and a derived-write failure rolls back raw stamps and existing proof together."
    )


def test_captured_brkb_settlement_units_and_previous_close_notional_are_conserved() -> None:
    # Captured original rows qualified by f005-brk2010-qualified-source-control.json; no new source read.
    raw = "SETTLEMENT DATE|CUSIP|SYMBOL|QUANTITY (FAILS)|DESCRIPTION|PRICE\n" + (
        "20100122|084670702|BRKB|41800|BERKSHIRE CLASS B|72.72\n20100125|084670702|BRKB|1870995|BERKSHIRE CLASS B|70.10\n"
    )
    source = ftd._parse_ftd_lines(raw).assign(period="201001b")
    events = pd.DataFrame({"ticker": ["BRK-B"], "date": pd.to_datetime(["2010-01-21"]), "value": [50.0]})
    basis = ShareBasis(events, events, events, {})
    master = ftd.prepare_master(_master())
    master.loc[master["canonical_company"].eq("BRK-B") & master["source_symbol"].eq("BRKB"), "market_symbol"] = "BRK-B"
    grain = ftd.ticker_rows(ftd.stamp_lines(source, master), ["BRK-B"], basis)
    assert grain["fails_quantity"].tolist() == [41800.0, 1870995.0]
    assert grain["fails_value"].tolist() == pytest.approx([3039696.0, 131156749.5])
    print(
        "SANITY: captured BRKB source quantities cross the 50:1 split without a doubled class conversion; previous-close USD is conserved. Actual secondary-A firing remains unobserved."
    )


def test_ftd_b_zip_retains_documented_day15_overlap(tmp_path: Path) -> None:
    context = cast(Any, SimpleNamespace(paths={"DATA_STORE": tmp_path}, config=extract_config()))
    cache = tmp_path / context.config.local.paths.fails_deliver
    path = cache / "cnsfails202607b.zip"
    _archive(path, {"fails.txt": "SETTLEMENT DATE|CUSIP|SYMBOL|QUANTITY (FAILS)|DESCRIPTION|PRICE\n20260715|526057104|LEN|12|LEN|10\n"})
    source = ftd._read_period(context, cache, "202607b", download=False)
    assert source is not None and source["period"].tolist() == ["202607b"] and source["date"].tolist() == [pd.Timestamp("2026-07-15")]
    assert path.exists()
    print("SANITY: the documented day15 b-ZIP overlap retains its actual assigned source tag and publication clock.")


@pytest.mark.parametrize("race", [False, True], ids=["coherent", "identity-interleaving"])
def test_identity_propagation_ftd_restamp_binds_classification_cohort(
    sqlite_store: DataStore, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, race: bool
) -> None:
    context = cast(Any, SimpleNamespace(store=sqlite_store, paths={"DATA_STORE": tmp_path}, config=extract_config(), log=logging.getLogger(__name__)))
    identity = _identity()
    for table, frame in identity.source_cohort.items():
        if not frame.empty:
            sqlite_store.save(table, frame)
    text = "SETTLEMENT DATE|CUSIP|SYMBOL|QUANTITY (FAILS)|DESCRIPTION|PRICE\n20260701|526057104|LEN|12|LEN|10\n"
    _archive(tmp_path / context.config.local.paths.fails_deliver / "cnsfails202607a.zip", {"fails.txt": text})
    raw = ftd.stamp_lines(ftd._parse_ftd_lines(text).assign(period="202607a"), ftd.prepare_master(_master()))
    sqlite_store.save(Tables.sec_fails_to_deliver_security, raw[list(ftd.SECURITY_COLUMNS)])
    sqlite_store.save(
        Tables.sec_fails_to_deliver,
        pd.DataFrame(
            {
                "ticker": ["LEN"],
                "date": pd.to_datetime(["2026-07-01"]),
                "fails_quantity": [12.0],
                "fails_value": [120.0],
                "period": ["202607a"],
                "source_checked_at": pd.to_datetime(["2026-07-03"]),
            }
        ),
    )
    before = sqlite_store.load(Tables.sec_fails_to_deliver_security)
    assert before is not None
    original = ftd.restamp_fails

    def restamp(ctx: Any, companies: Any, tickers: Any, **kwargs: Any) -> list[dict]:
        if race:
            changed = _master()
            changed.loc[changed["source_symbol"].eq("LEN"), "conversion_ratio"] = 2.0
            sqlite_store.replace_institutional_identity(Tables.security_master, changed)
        return original(ctx, companies, tickers, **kwargs)

    monkeypatch.setattr(identity_propagate, "restamp_fails", restamp)
    if race:
        with pytest.raises(ValueError, match="identity cohort changed"):
            identity_propagate._refresh_fails(context, ["LEN"], pd.Timestamp("2026-10-06"), dry_run=False, every=True, others=frozenset())
        stored_frame = sqlite_store.load(Tables.sec_fails_to_deliver_security)
        assert stored_frame is not None
        pd.testing.assert_frame_equal(stored_frame, before)
        invalidated_rows = sqlite_store.load(Tables.sec_fails_to_deliver)
        assert invalidated_rows is not None
        assert invalidated_rows["source_checked_at"].isna().all()
    else:
        assert identity_propagate._refresh_fails(context, ["LEN"], pd.Timestamp("2026-10-06"), dry_run=False, every=True, others=frozenset()) == []
        rows = sqlite_store.load(Tables.sec_fails_to_deliver)
        assert rows is not None
        assert rows["fails_quantity"].tolist() == [12.0] and rows["source_checked_at"].notna().all()
    print(
        "SANITY: the actual identity propagation route certifies only its original classification cohort; a later identity commit refuses and preserves raw economics with invalidated proof."
    )


def test_zip_reconciliation_refuses_concurrently_changed_edgar_payload(sqlite_store: DataStore, monkeypatch: pytest.MonkeyPatch) -> None:
    context = cast(Any, SimpleNamespace(store=sqlite_store, log=logging.getLogger(__name__)))
    rows = pd.DataFrame(
        {
            "accession_number": ["a", "a"],
            "ticker": ["LEN", "LEN"],
            "issuer_cik": ["0000920760"] * 2,
            "security_type": ["nonderiv"] * 2,
            "row_sequence": [1, 2],
            "source": ["edgar", "zip"],
            "shares": [10.0, 20.0],
            "quarter": [None, "2026q3"],
            "filing_date": pd.to_datetime(["2026-07-01"] * 2),
        }
    )
    notes = pd.DataFrame({"accession_number": ["a"], "footnote_id": ["F1"], "footnote_text": ["old"]})
    sqlite_store.save(Tables.insider_transactions, rows)
    sqlite_store.save(Tables.insider_footnotes, notes)
    original = sqlite_store.replace_institutional_source
    concurrent = rows.assign(source="edgar", shares=99.0)
    entered = False

    def pair(frames: Any, scopes: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal entered
        if not entered:
            entered = True
            original(
                {Tables.insider_transactions: concurrent, Tables.insider_footnotes: notes.assign(footnote_text="new")},
                {Tables.insider_transactions: {"accession_number": ["a"]}, Tables.insider_footnotes: {"accession_number": ["a"]}},
            )
        return original(frames, scopes, *args, **kwargs)

    monkeypatch.setattr(sqlite_store, "replace_institutional_source", pair)
    with pytest.raises(ValueError, match="source cohort changed"):
        insider_edgar.replace_zip_accessions(context, pd.Timestamp("2026-07-01"))
    actual_rows = sqlite_store.load(Tables.insider_transactions)
    actual_notes = sqlite_store.load(Tables.insider_footnotes)
    assert actual_rows is not None and actual_notes is not None
    assert actual_rows["shares"].tolist() == [99.0, 99.0]
    assert actual_notes["footnote_text"].tolist() == ["new"]
    print("SANITY: ZIP reconciliation cannot overwrite a concurrent EDGAR payload or note update with a stale outside-transaction copy.")


@pytest.mark.parametrize("race", [False, True], ids=["issuer-scoped-delete", "stale-resolver"])
def test_generic_identity_purge_uses_guarded_insider_issuer_deletion(
    sqlite_store: DataStore, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, race: bool
) -> None:
    context = cast(Any, SimpleNamespace(store=sqlite_store, paths={"DATA_STORE": tmp_path}, config=extract_config(), log=logging.getLogger(__name__)))
    identity = _identity()
    for table, frame in identity.source_cohort.items():
        if not frame.empty:
            sqlite_store.save(table, frame)
    rows = pd.DataFrame(
        {
            "accession_number": ["a", "a", "b"],
            "ticker": ["LEN", "AAA", "LEN"],
            "issuer_cik": ["0000000099", "0000000100", "0000920760"],
            "security_type": ["nonderiv"] * 3,
            "row_sequence": [1, 2, 1],
            "filing_date": pd.to_datetime(["2026-07-01"] * 3),
            "shares": [1.0, 2.0, 3.0],
            "fetched_at": pd.to_datetime(["2026-10-07"] * 3),
            "inventory_checked_through": pd.to_datetime(["2026-10-06"] * 3),
        }
    )
    notes = pd.DataFrame({"accession_number": ["a"], "footnote_id": ["F1"], "footnote_text": ["shared actual note"]})
    sqlite_store.save(Tables.insider_transactions, rows)
    sqlite_store.save(Tables.insider_footnotes, notes)
    spec = next(spec for spec in identity_propagate.PURGE_TABLES if spec.table == Tables.insider_transactions)
    monkeypatch.setattr(identity_propagate, "PURGE_TABLES", (spec,))
    monkeypatch.setattr(identity_propagate, "_reparse_bulk", lambda *args: {})
    for name in ("_refresh_insider", "_refresh_fails", "_refresh_short_volume"):
        monkeypatch.setattr(identity_propagate, name, lambda *args, **kwargs: [])
    foreign_rows = identity_propagate._foreign_rows

    def classify(*args: Any) -> pd.DataFrame:
        foreign = foreign_rows(*args)
        if race:
            changed = identity.source_cohort[Tables.entity_lineage].copy()
            extra = changed.loc[changed["canonical_ticker"].eq("LEN")].iloc[:1].assign(cik="0000000099", valid_from=pd.Timestamp("2000-01-01"))
            sqlite_store.replace_institutional_identity(Tables.entity_lineage, pd.concat([changed, extra], ignore_index=True))
            actual, actual_notes = sqlite_store.insider_inventory_snapshot("LEN", None)
            sqlite_store.certify_insider_inventory("LEN", pd.Timestamp("2026-10-06"), actual, actual_notes, load_source_identity(context))
        return foreign

    monkeypatch.setattr(identity_propagate, "_foreign_rows", classify)
    if race:
        with pytest.raises(ValueError, match="identity cohort changed"):
            identity_propagate.propagate_identity(
                context, ["LEN"], every_ticker=True, identity=identity, as_of=pd.Timestamp("2026-10-08"), all_tickers=["LEN", "AAA"]
            )
        assert sqlite_store.row_count(Tables.insider_transactions) == 3
    else:
        identity_propagate.propagate_identity(
            context, ["LEN"], every_ticker=True, identity=identity, as_of=pd.Timestamp("2026-10-08"), all_tickers=["LEN", "AAA"]
        )
        kept = sqlite_store.load(Tables.insider_transactions)
        assert kept is not None
        assert set(zip(kept["accession_number"], kept["ticker"], strict=True)) == {("a", "AAA"), ("b", "LEN")}
        assert kept.loc[kept["ticker"].eq("LEN"), "inventory_checked_through"].isna().all()
        kept_notes = sqlite_store.load(Tables.insider_footnotes)
        assert kept_notes is not None
        assert kept_notes["footnote_text"].tolist() == ["shared actual note"]
    print(
        "SANITY: actual generic identity propagation preserves another issuer's shared-accession payload/notes, atomically clears retained proof, and refuses a stale resolver after newer joint certification."
    )


def _parser_identity_context(store: DataStore, directory: Path) -> tuple[Any, Identity]:
    context = SimpleNamespace(store=store, paths={"DATA_STORE": directory}, config=extract_config(), log=logging.getLogger(__name__))
    identity = _identity()
    for table, frame in identity.source_cohort.items():
        if not frame.empty:
            store.save(table, frame)
    context._institutional_identity_cohort = identity.source_cohort
    return context, identity


def _advance_parser_identity(context: Any, identity: Identity) -> None:
    changed = identity.source_cohort[Tables.entity_lineage].copy()
    selected = changed["canonical_ticker"].eq("LEN")
    changed.loc[selected, "valid_to"] = pd.Timestamp("2020-01-01")
    changed.loc[selected, "scope_changed_at"] = pd.Timestamp("2026-10-05")
    context.store.replace_institutional_identity(Tables.entity_lineage, changed)
    context._institutional_identity_cohort = load_source_identity(context)


@pytest.mark.parametrize("race", [False, True], ids=["coherent", "same-context-new-resolver"])
def test_actual_xml_parser_identity_is_bound_through_write_and_certification(
    sqlite_store: DataStore, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, race: bool
) -> None:
    context, identity = _parser_identity_context(sqlite_store, tmp_path)
    accession = "0000920760-26-000001"
    xml = """<ownershipDocument><documentType>4</documentType><periodOfReport>2026-07-01</periodOfReport>
    <issuer><issuerCik>920760</issuerCik><issuerTradingSymbol>LEN</issuerTradingSymbol></issuer>
    <nonDerivativeTable><nonDerivativeTransaction><securityTitle><value>Common Stock</value></securityTitle>
    <transactionDate><value>2026-07-01</value></transactionDate><transactionCoding><transactionCode>P</transactionCode></transactionCoding>
    <transactionAmounts><transactionShares><value>12</value></transactionShares><transactionPricePerShare><value>10</value></transactionPricePerShare></transactionAmounts>
    </nonDerivativeTransaction></nonDerivativeTable><footnotes><footnote id="F1">Actual source note</footnote></footnotes></ownershipDocument>"""

    def source() -> str:
        if race:
            _advance_parser_identity(context, identity)
        return xml

    filing = SimpleNamespace(
        accession_number=accession,
        cik=920760,
        company="LEN",
        form="4",
        filing_date=pd.Timestamp("2026-07-01"),
        header=SimpleNamespace(acceptance_datetime="2026-07-01 16:05:00"),
        xml=source,
    )
    monkeypatch.setattr("src.data_extract.utils.common.edgar_index.index_filing", lambda *args: filing)
    units = pd.DataFrame(
        {"cik": ["0000920760"], "company": ["LEN"], "form": ["4"], "filed": pd.to_datetime(["2026-07-01"]), "accession": [accession]}
    )
    with insider_edgar._insider_run_scope(context, ["LEN"], pd.Timestamp("2026-10-06")) as prepared:
        outcome = _walk_key(context, prepared, EdgarScope(identity), {"LEN": units}, "LEN", "0000920760")
        certified = certify_insider_source(context, "LEN", pd.Timestamp("2026-07-03"), [accession])
        assert certified is not race and bool(len(outcome.failed)) is race
    if not race:
        rows = sqlite_store.load(Tables.insider_transactions)
        assert rows is not None
        assert pd.to_datetime(rows["fetched_at"]).min() >= pd.Timestamp("2026-10-05")
        assert rows["inventory_checked_through"].notna().all()
    else:
        assert sqlite_store.row_count(Tables.insider_transactions) == 0
    print(
        "SANITY: actual XML parsing under I1 cannot use same-context I2 or a later fetched timestamp to save or certify stale classification; the coherent source run certifies."
    )


@pytest.mark.parametrize("race", [False, True], ids=["coherent", "same-context-new-resolver"])
def test_actual_bulk_parser_identity_is_bound_through_write_and_certification(
    sqlite_store: DataStore, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, race: bool
) -> None:
    context, identity = _parser_identity_context(sqlite_store, tmp_path)
    monkeypatch.setattr(insider_zip, "load_identity", lambda ctx: identity)
    monkeypatch.setattr(insider_zip, "quarter_periods", lambda *args: ["2026q3"])
    monkeypatch.setattr(insider_zip, "_screen_stored_rows", lambda *args: 0)
    members = _bulk_members()
    members["FOOTNOTES.TSV"] += "0000920760-26-000001\tF1\tActual source note\n"
    _archive(tmp_path / context.config.local.paths.insider_transactions / "2026q3.zip", members)
    parse = insider_zip._parse_quarter

    def classify(
        tables: tuple[pd.DataFrame, ...], quarter: str, universe: Any, resolver: Identity, fetched_at: pd.Timestamp
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        result = parse(tables, quarter, universe, resolver, fetched_at)
        assert fetched_at >= pd.Timestamp("2026-10-05")
        if race:
            _advance_parser_identity(context, identity)
        return result

    monkeypatch.setattr(insider_zip, "_parse_quarter", classify)
    refused = False
    with insider_source_run(context):
        try:
            insider_zip.fetch_insider_transactions(context, ["LEN"], 1, all_tickers=["LEN"], as_of=pd.Timestamp("2026-10-06"))
        except ValueError as exc:
            assert "identity cohort changed" in str(exc)
            refused = True
        certified = certify_insider_source(context, "LEN", pd.Timestamp("2026-07-03"), ["0000920760-26-000001"])
        assert certified is not race and refused is race
    if race:
        assert sqlite_store.row_count(Tables.insider_transactions) == 0
    print(
        "SANITY: actual bulk parsing retains its original resolver snapshot through atomic write and certification; latest mutable context and fetched_after_I2 cannot reclassify the staged cohort."
    )
