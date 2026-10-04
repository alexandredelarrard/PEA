"""The single-table insider load flow on a real SQLite `DataStore`: the zip ingest stamps `quarter`
on EDGAR-stored filings and inserts only the filings EDGAR lacks, EDGAR replaces a zip-sourced
filing whole, both reruns are no-ops, and the REQ-007 / REQ-008 log lines are emitted.

The SEC is faked at the fetchers' IO seams (zip members per quarter, the EDGAR filing lister);
parsing, screening, the shared driver and every store read and write are the real code.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from src.data_extract.utils.common.identity import Identity, build_identity
from src.data_extract.utils.common.run_manifest import get_entry
from src.data_extract.utils.common.sec_utils import CIK_MAPPING_COLS
from src.data_extract.utils.institutionals import fetch_insider_edgar as edgar
from src.data_extract.utils.institutionals import fetch_insider_transactions as ins
from src.data_store.schema import Tables
from tests.data_extract.fake_context import extract_config

AAA_CIK = "0000000001"
OTHER_CIK = "0000000099"  # an unrelated issuer whose filings claim `AAA`
OWNER_CIK = "0000000777"
UNIVERSE = ["AAA"]
QUARTERS = ("2026q1", "2026q2")
PK = list(Tables.insider_transactions.pk)


@dataclass(frozen=True)
class Trade:
    date: str
    code: str
    shares: float
    price: float
    after: float


@dataclass(frozen=True)
class Spec:
    """One Form 4 as both sources publish it."""

    accession: str
    filed: str
    trades: tuple[Trade, ...]
    issuer_cik: str = AAA_CIK


def _bulk_date(iso: str) -> str:
    return pd.Timestamp(iso).strftime("%d-%b-%Y").upper()


def _zip_tables(specs: list[Spec]) -> tuple[pd.DataFrame, ...]:
    """(SUBMISSION, REPORTINGOWNER, NONDERIV_TRANS, DERIV_TRANS, FOOTNOTES) string members."""
    sub = pd.DataFrame(
        [
            {
                "ACCESSION_NUMBER": spec.accession,
                "ISSUERCIK": spec.issuer_cik,
                "ISSUERNAME": "AAA CORP",
                "ISSUERTRADINGSYMBOL": "AAA",
                "DOCUMENT_TYPE": "4",
                "FILING_DATE": _bulk_date(spec.filed),
                "PERIOD_OF_REPORT": _bulk_date(spec.trades[0].date),
            }
            for spec in specs
        ]
    )
    own = pd.DataFrame(
        [
            {"ACCESSION_NUMBER": spec.accession, "RPTOWNERCIK": OWNER_CIK, "RPTOWNERNAME": "DOE JANE", "RPTOWNER_RELATIONSHIP": "Officer"}
            for spec in specs
        ]
    )
    nonderiv = pd.DataFrame(
        [
            {
                "ACCESSION_NUMBER": spec.accession,
                "NONDERIV_TRANS_SK": str(1000 * k + i),
                "SECURITY_TITLE": "Common Stock",
                "TRANS_DATE": _bulk_date(trade.date),
                "TRANS_CODE": trade.code,
                "TRANS_SHARES": str(trade.shares),
                "TRANS_PRICEPERSHARE": str(trade.price),
                "TRANS_ACQUIRED_DISP_CD": "A" if trade.code == "P" else "D",
                "SHRS_OWND_FOLWNG_TRANS": str(trade.after),
                "DIRECT_INDIRECT_OWNERSHIP": "D",
            }
            for k, spec in enumerate(specs)
            for i, trade in enumerate(spec.trades)
        ]
    )
    notes = pd.DataFrame(columns=["ACCESSION_NUMBER", "FOOTNOTE_ID", "FOOTNOTE_TXT"])
    return sub, own, nonderiv, pd.DataFrame(), notes


def _xml(spec: Spec) -> str:
    rows = "".join(
        "<nonDerivativeTransaction><securityTitle><value>Common Stock</value></securityTitle>"
        f"<transactionDate><value>{trade.date}</value></transactionDate>"
        f"<transactionCoding><transactionFormType>4</transactionFormType><transactionCode>{trade.code}</transactionCode></transactionCoding>"
        f"<transactionAmounts><transactionShares><value>{trade.shares}</value></transactionShares>"
        f"<transactionPricePerShare><value>{trade.price}</value></transactionPricePerShare>"
        f"<transactionAcquiredDisposedCode><value>{'A' if trade.code == 'P' else 'D'}</value></transactionAcquiredDisposedCode></transactionAmounts>"
        f"<postTransactionAmounts><sharesOwnedFollowingTransaction><value>{trade.after}</value></sharesOwnedFollowingTransaction></postTransactionAmounts>"
        "<ownershipNature><directOrIndirectOwnership><value>D</value></directOrIndirectOwnership></ownershipNature>"
        "</nonDerivativeTransaction>"
        for trade in spec.trades
    )
    return (
        f"<ownershipDocument><documentType>4</documentType><periodOfReport>{spec.trades[0].date}</periodOfReport>"
        f"<issuer><issuerCik>{spec.issuer_cik}</issuerCik><issuerName>AAA CORP</issuerName><issuerTradingSymbol>AAA</issuerTradingSymbol></issuer>"
        f"<reportingOwner><reportingOwnerId><rptOwnerCik>{OWNER_CIK}</rptOwnerCik><rptOwnerName>DOE JANE</rptOwnerName></reportingOwnerId>"
        "<reportingOwnerRelationship><isOfficer>1</isOfficer></reportingOwnerRelationship></reportingOwner>"
        f"<nonDerivativeTable>{rows}</nonDerivativeTable></ownershipDocument>"
    )


class _Filing:
    """The attributes `fetch_insider_edgar` reads from an edgartools filing."""

    def __init__(self, spec: Spec) -> None:
        self.accession_number = spec.accession
        self.filing_date = pd.Timestamp(spec.filed)
        self.header = SimpleNamespace(acceptance_datetime=f"{spec.filed} 16:05:00")
        self._xml = _xml(spec)

    def xml(self) -> str:
        return self._xml


class _FakeSec:
    """Zip members per quarter and EDGAR filings, patched over the fetchers' IO seams."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, identity: Identity, cache: Path) -> None:
        self.zips: dict[str, tuple[pd.DataFrame, ...]] = {}
        self.filings: list[Spec] = []
        self.listing_since: list[pd.Timestamp | None] = []
        monkeypatch.setattr(ins, "load_identity", lambda context: identity)
        monkeypatch.setattr(ins, "cache_dir", lambda context, key: cache)
        monkeypatch.setattr(ins, "quarter_periods", lambda span, first_year: list(QUARTERS))
        monkeypatch.setattr(ins, "ensure_zip", lambda context, path, url, **kwargs: path)
        monkeypatch.setattr(ins, "_read_tables", lambda path: self.zips.get(path.stem))
        monkeypatch.setattr(edgar, "load_identity", lambda context: identity)
        monkeypatch.setattr(edgar, "insider_filings", self._list)

    def _list(self, ticker: str, cik: str, *, since: pd.Timestamp | None, through: pd.Timestamp, done_accessions: frozenset[str], scope: Any) -> list:
        self.listing_since.append(since)
        return [_Filing(spec) for spec in self.filings if spec.accession not in done_accessions]


@pytest.fixture(scope="module")
def identity() -> Identity:
    return build_identity(
        lineage=pd.DataFrame([{"cik": AAA_CIK, "entity_id": "E0000000001", "source": "roster", "confidence": None, "evidence": "test"}]),
        tenure=pd.DataFrame(
            [
                {
                    "symbol": "AAA",
                    "issuer_cik": AAA_CIK,
                    "valid_from": pd.Timestamp("2006-01-03"),
                    "valid_to": None,
                    "n_filings": 100,
                    "source": "form345",
                    "evidence": "",
                }
            ]
        ),
        roster=pd.DataFrame([{"ticker": "AAA", "cik": AAA_CIK}]),
    )


def _context(tmp_path: Path, store: Any) -> Any:
    store.save(Tables.sp500_tickers, pd.DataFrame({col: ["1" if col == "cik" else f"{col}-AAA"] for col in CIK_MAPPING_COLS} | {"ticker": ["AAA"]}))
    return SimpleNamespace(
        store=store,
        paths={"DATA_STORE": tmp_path},
        log=logging.getLogger("tests.insider_load_flow"),
        config=extract_config(data_extract={"manifest_full_rescan_days": 30}),
        ensure_edgar_identity=lambda: None,
        config_dir=tmp_path,
    )


def _table(store: Any) -> pd.DataFrame:
    df = store.load(Tables.insider_transactions)
    return df.sort_values(PK).reset_index(drop=True)


def _rows_of(df: pd.DataFrame, accession: str) -> pd.DataFrame:
    return df[df["accession_number"] == accession].reset_index(drop=True)


def _nulls_as_none(df: pd.DataFrame) -> pd.DataFrame:
    return df.astype(object).where(df.notna(), None)


def _messages(caplog: pytest.LogCaptureFixture, level: int) -> list[str]:
    return [record.getMessage() for record in caplog.records if record.levelno == level]


def _mixed_accessions(df: pd.DataFrame) -> int:
    return int((df.groupby("accession_number")["source"].nunique() > 1).sum())


def _run_zip(context: Any, *, reparse: bool = False) -> int:
    return ins.fetch_insider_transactions(context, tickers=UNIVERSE, years_history=1, reparse=reparse)


def _run_edgar(context: Any, *, full: bool = False) -> None:
    edgar.fetch_insider_edgar(context, tickers=UNIVERSE, years_history=15, full=full)


Q0 = Spec("0000000001-26-000010", "2026-02-02", (Trade("2026-01-30", "P", 10, 5, 110),))
E1 = Spec("0000000001-26-000100", "2026-05-04", (Trade("2026-05-01", "P", 100, 10, 1000), Trade("2026-05-01", "S", 50, 11, 950)))
E1_ZIP = Spec(E1.accession, E1.filed, (Trade("2026-05-01", "P", 100.004, 10, 1000), Trade("2026-05-01", "S", 50, 11.5, 950)))
E2 = Spec("0000000001-26-000101", "2026-05-05", (Trade("2026-05-04", "S", 20, 12, 930),))
Z0 = Spec("0000000001-26-000090", "2026-04-01", (Trade("2026-03-30", "S", 5, 9, 105),))
Z1 = Spec("0000000001-26-000102", "2026-05-06", (Trade("2026-05-05", "P", 7, 12, 937),))
R1 = Spec("0000000099-26-000001", "2026-05-07", (Trade("2026-05-06", "P", 1, 1, 1), Trade("2026-05-06", "A", 2, 0, 3)), issuer_cik=OTHER_CIK)


def test_zip_over_edgar_stamps_quarter_inserts_only_missing_filings_and_reports(tmp_path, sqlite_store, monkeypatch, identity, caplog):
    """AC-006 (a), (b), (d), (e); AC-007 WARNING + INFO; AC-008 exclusion WARNING."""
    caplog.set_level(logging.INFO)
    sec = _FakeSec(monkeypatch, identity, tmp_path)
    context = _context(tmp_path, sqlite_store)

    sec.zips["2026q1"] = _zip_tables([Q0])
    _run_zip(context)
    sec.filings = [E1, E2]
    _run_edgar(context)
    df_edgar_only = _table(sqlite_store)
    assert sec.listing_since[-1] == pd.Timestamp("2026-01-26"), "EDGAR lists from the max stored filing date - 7 days"

    caplog.clear()
    sec.zips["2026q2"] = _zip_tables([E1_ZIP, Z0, Z1, R1])
    inserted = _run_zip(context)
    df_after = _table(sqlite_store)

    # (a) EDGAR-stored filings: only `quarter` changes, no row is added
    for spec in (E1,):
        before, after = _rows_of(df_edgar_only, spec.accession), _rows_of(df_after, spec.accession)
        assert list(after["quarter"]) == ["2026q2", "2026q2"]
        pd.testing.assert_frame_equal(after.drop(columns="quarter"), before.drop(columns="quarter"))
    pd.testing.assert_frame_equal(_rows_of(df_after, E2.accession), _rows_of(df_edgar_only, E2.accession))
    assert _rows_of(df_after, E2.accession)["quarter"].isna().all(), "a filing the zip lacks keeps a NULL quarter"
    # (b) zip-only filings are inserted with source='zip'; the rejected filing is not stored
    for spec in (Z0, Z1):
        rows = _rows_of(df_after, spec.accession)
        assert list(rows["source"]) == ["zip"] and list(rows["quarter"]) == ["2026q2"]
    assert R1.accession not in set(df_after["accession_number"])
    assert inserted == 2
    assert len(df_after) == len(df_edgar_only) + 2
    # (e) one source per accession
    assert _mixed_accessions(df_after) == 0
    # D-06: the zip ingest never writes the insider_transactions manifest entry; EDGAR's run does
    entry = get_entry(context, Tables.insider_transactions)
    assert entry is not None and entry["coverage_complete"] is True

    warnings, infos = _messages(caplog, logging.WARNING), _messages(caplog, logging.INFO)
    report = "insider 2026q2: 1 / 2 filings missing from EDGAR (50.0%), added from zip; top: AAA 1"
    assert report in warnings, warnings
    mismatch = next(message for message in infos if message.startswith("insider 2026q2: 1 EDGAR-only filing"))
    assert "1 / 2 shared row(s) mismatched (50.0%)" in mismatch and "0 key row(s) on one side only" in mismatch
    exclusion = next(message for message in warnings if "excluded" in message)
    assert "1 filing(s), 2 row(s), 1 P/S row(s)" in exclusion and "entity_mismatch 1" in exclusion and "top claimed: AAA 1" in exclusion
    assert "insider 2026q1: loaded from zip (no EDGAR coverage)" not in infos, "2026q1 is stored, so it is not re-parsed"

    # (d) reruns with no new source data change no row (a re-parse refreshes only zip `fetched_at`)
    _run_zip(context, reparse=True)
    _run_edgar(context)
    df_rerun = _table(sqlite_store)
    pd.testing.assert_frame_equal(df_rerun.drop(columns="fetched_at"), df_after.drop(columns="fetched_at"))
    edgar_rows = df_after["source"].eq("edgar")
    pd.testing.assert_frame_equal(df_rerun[edgar_rows], df_after[edgar_rows])
    print(
        f"\nSANITY: zip over EDGAR -> {inserted} zip-only rows inserted, E1's 2 EDGAR rows got only quarter=2026q2, E2 kept NULL quarter, "
        f"R1 excluded; report '{report}'; reruns left all {len(df_rerun)} rows unchanged and no accession holds two sources."
    )


def test_a_quarter_without_edgar_rows_logs_info_only(tmp_path, sqlite_store, monkeypatch, identity, caplog):
    """AC-007: a rebuild quarter with no EDGAR rows gives one INFO line and no false 100%-missing WARNING."""
    caplog.set_level(logging.INFO)
    sec = _FakeSec(monkeypatch, identity, tmp_path)
    context = _context(tmp_path, sqlite_store)
    sec.zips["2026q2"] = _zip_tables([Z0, Z1])

    inserted = _run_zip(context)

    df = _table(sqlite_store)
    assert inserted == 2 and set(df["source"]) == {"zip"}
    assert "insider 2026q2: loaded from zip (no EDGAR coverage)" in _messages(caplog, logging.INFO)
    assert _messages(caplog, logging.WARNING) == []
    assert get_entry(context, Tables.insider_transactions) is None, "the zip ingest records no manifest entry (D-06)"
    print("\nSANITY: a zip quarter with no EDGAR rows inserted 2 zip rows, logged 'loaded from zip (no EDGAR coverage)' and no WARNING.")


A = Spec("0000000001-26-000200", "2026-05-04", tuple(Trade("2026-05-01", "S", 10 + i, 20, 500 - i) for i in range(3)))
A_EDGAR = Spec(A.accession, A.filed, (Trade("2026-05-01", "S", 30, 20, 470), Trade("2026-05-01", "S", 31, 20, 439)))
B = Spec("0000000001-26-000201", "2026-05-05", (Trade("2026-05-04", "P", 1, 20, 440),))
B_EDGAR = Spec(B.accession, B.filed, (Trade("2026-05-04", "P", 1, 20, 440), Trade("2026-05-04", "P", 2, 20, 442)))
C = Spec("0000000001-26-000202", "2026-05-06", (Trade("2026-05-05", "S", 3, 20, 439),))
N = Spec("0000000001-26-000300", "2026-07-01", (Trade("2026-06-30", "P", 4, 21, 443),))
X = Spec("0000000099-26-000002", "2026-07-01", (Trade("2026-06-30", "S", 9, 9, 9),), issuer_cik=OTHER_CIK)


def test_edgar_replaces_a_zip_filing_wholesale_and_keeps_its_quarter(tmp_path, sqlite_store, monkeypatch, identity, caplog):
    """AC-006 (c), (d), (e); AC-008 for the EDGAR run (exclusions collected across the worker pool)."""
    caplog.set_level(logging.INFO)
    sec = _FakeSec(monkeypatch, identity, tmp_path)
    context = _context(tmp_path, sqlite_store)
    sec.zips["2026q2"] = _zip_tables([A, B, C])
    _run_zip(context)
    df_zip = _table(sqlite_store)

    caplog.clear()
    sec.filings = [A_EDGAR, B_EDGAR, N, X]
    _run_edgar(context)
    df_after = _table(sqlite_store)

    assert sec.listing_since[-1] == pd.Timestamp("2026-04-29"), "max stored filing_date 2026-05-06 - 7 days"
    rows_a = _rows_of(df_after, A.accession)
    assert len(rows_a) == 2 and set(rows_a["source"]) == {"edgar"} and set(rows_a["quarter"]) == {"2026q2"}
    assert list(rows_a["shares"]) == [30.0, 31.0], "EDGAR's values replace the zip's"
    rows_b = _rows_of(df_after, B.accession)
    assert len(rows_b) == 2 and set(rows_b["source"]) == {"edgar"} and list(rows_b["quarter"]) == ["2026q2", "2026q2"]
    # NULL-normalised: a column all-NULL on zip rows (e.g. `acceptance_datetime`) reads back typed once EDGAR rows fill it
    pd.testing.assert_frame_equal(_nulls_as_none(_rows_of(df_after, C.accession)), _nulls_as_none(_rows_of(df_zip, C.accession)))
    rows_n = _rows_of(df_after, N.accession)
    assert list(rows_n["source"]) == ["edgar"] and rows_n["quarter"].isna().all()
    assert X.accession not in set(df_after["accession_number"])
    assert _mixed_accessions(df_after) == 0
    exclusion = next(message for message in _messages(caplog, logging.WARNING) if "excluded" in message)
    assert "1 filing(s), 1 row(s), 1 P/S row(s)" in exclusion and "entity_mismatch 1" in exclusion

    _run_edgar(context)
    pd.testing.assert_frame_equal(_table(sqlite_store), df_after)
    print(
        f"\nSANITY: EDGAR re-read zip filing A (3 zip rows -> {len(rows_a)} EDGAR rows) and B (1 -> 2), keeping quarter 2026q2 on every row; "
        f"C stays zip, N is new with NULL quarter, X excluded with a WARNING; an EDGAR rerun changed none of the {len(df_after)} rows."
    )
