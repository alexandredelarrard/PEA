"""The single-table insider load flow on a real SQLite `DataStore`: the zip ingest stamps `quarter`
on EDGAR-stored filings and inserts only the filings EDGAR lacks, EDGAR reads only after the last
zip quarter and replaces a zip-sourced filing there whole, both reruns are no-ops, and the
REQ-007 / REQ-008 log lines are emitted.

The SEC is faked at the fetchers' IO seams (zip members per quarter, the local EDGAR index and the
filing objects it yields); parsing, screening, the planner, the shared driver and every store read
and write are the real code.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest
from sqlalchemy import text

from src.data_aggregate.utils.institutionals.frontiers import schedule_complete_through
from src.data_extract.utils.common import edgar_driver, edgar_index
from src.data_extract.utils.common.identity import Identity, build_identity
from src.data_extract.utils.common.sec_utils import CIK_MAPPING_COLS
from src.data_extract.utils.institutionals import fetch_insider_edgar as edgar
from src.data_extract.utils.institutionals import fetch_insider_transactions as ins
from src.data_store import ddl
from src.data_store import store as store_module
from src.data_store.schema import Tables
from tests.data_extract.edgar_fixtures import seed_index
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
    symbol: str = "AAA"


def _bulk_date(iso: str) -> str:
    return pd.Timestamp(iso).strftime("%d-%b-%Y").upper()


def _zip_tables(specs: list[Spec]) -> tuple[pd.DataFrame, ...]:
    """(SUBMISSION, REPORTINGOWNER, NONDERIV_TRANS, DERIV_TRANS, FOOTNOTES) string members."""
    sub = pd.DataFrame(
        [
            {
                "ACCESSION_NUMBER": spec.accession,
                "ISSUERCIK": spec.issuer_cik,
                "ISSUERNAME": f"{spec.symbol} CORP",
                "ISSUERTRADINGSYMBOL": spec.symbol,
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
        f"<issuer><issuerCik>{spec.issuer_cik}</issuerCik><issuerName>{spec.symbol} CORP</issuerName>"
        f"<issuerTradingSymbol>{spec.symbol}</issuerTradingSymbol></issuer>"
        f"<reportingOwner><reportingOwnerId><rptOwnerCik>{OWNER_CIK}</rptOwnerCik><rptOwnerName>DOE JANE</rptOwnerName></reportingOwnerId>"
        "<reportingOwnerRelationship><isOfficer>1</isOfficer></reportingOwnerRelationship></reportingOwner>"
        f"<nonDerivativeTable>{rows}</nonDerivativeTable></ownershipDocument>"
    )


class _Filing:
    """The attributes `fetch_insider_edgar` and the driver read from an edgartools filing."""

    def __init__(self, spec: Spec, cik: str) -> None:
        self.accession_number = spec.accession
        self.cik = int(cik)
        self.form = "4"
        self.company = f"{spec.symbol} CORP"
        self.filing_date = pd.Timestamp(spec.filed)
        self.header = SimpleNamespace(acceptance_datetime=f"{spec.filed} 16:05:00")
        self._xml = _xml(spec)

    def xml(self) -> str:
        return self._xml


class _FakeSec:
    """Zip members per quarter, plus EDGAR filings published in the local index (under the issuer's
    CIK, or under a listed owner's CIK) and served by accession; `read` records every filing read."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, identity: Identity, cache: Path) -> None:
        self.zips: dict[str, tuple[pd.DataFrame, ...]] = {}
        self.specs: dict[str, Spec] = {}
        self.read: list[str] = []
        monkeypatch.setattr(ins, "load_identity", lambda context: identity)
        monkeypatch.setattr(ins, "cache_dir", lambda context, key: cache)
        monkeypatch.setattr(ins, "quarter_periods", lambda *args: list(QUARTERS))
        monkeypatch.setattr(ins, "ensure_zip", lambda context, path, url, **kwargs: path)
        monkeypatch.setattr(ins, "_read_tables", lambda path: self.zips.get(path.stem))
        monkeypatch.setattr(edgar_driver, "load_identity", lambda context: identity)
        # One worker (the in-memory SQLite store shares one connection across threads) and no index download.
        monkeypatch.setattr(edgar, "run_edgar_fetch", partial(edgar_driver.run_edgar_fetch, max_workers=1, refresh_index=False))
        monkeypatch.setattr(edgar_index, "index_filing", self._filing)

    def publish(self, context: Any, specs: list[Spec], *, owner_listed: tuple[tuple[Spec, str], ...] = ()) -> None:
        """EDGAR's index lists `specs` under their issuer CIK and each `(spec, cik)` of `owner_listed` under that CIK."""
        listed = [(spec, spec.issuer_cik) for spec in specs] + list(owner_listed)
        self.specs = {spec.accession: spec for spec, _ in listed}
        seed_index(context, [(int(cik), f"{spec.symbol} CORP", "4", spec.filed, spec.accession) for spec, cik in listed])

    def _filing(self, cik: object, company: object, form: object, filed: object, accession: object) -> _Filing:
        self.read.append(str(accession))
        return _Filing(self.specs[str(accession)], str(cik))


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


def _context(tmp_path: Path, store: Any, universe: tuple[str, ...] = ("AAA",)) -> Any:
    """A fake context whose `sp500_tickers` (the analysis universe) holds `universe`, CIKs 1, 2, ..."""
    store.save(
        Tables.sp500_tickers,
        pd.DataFrame(
            {col: [str(i + 1) if col == "cik" else f"{col}-{t}" for i, t in enumerate(universe)] for col in CIK_MAPPING_COLS}
            | {"ticker": list(universe)}
        ),
    )
    return SimpleNamespace(
        store=store,
        paths={"DATA_STORE": tmp_path},
        log=logging.getLogger("tests.insider_load_flow"),
        config=extract_config(data_extract={"redundant_ticks": []}),
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
    sec.publish(context, [Q0, E1, E2])
    _run_edgar(context)
    df_edgar_only = _table(sqlite_store)
    assert sec.read == [E1.accession, E2.accession], "EDGAR reads only after the stored 2026q1 zip quarter"

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


LAST_SESSION = pd.Timestamp("2026-05-08")


def zip_after_edgar_keeps_the_frontier(context: Any, sec: _FakeSec, run_zip: Callable[[Any], int]) -> pd.Timestamp | None:
    """EDGAR run, then a zip quarter through `run_zip`; asserts the table's DB frontier is unchanged. Returns it."""
    sec.zips["2026q1"] = _zip_tables([Q0])
    run_zip(context)
    sec.publish(context, [E1, E2])
    _run_edgar(context)
    frontier_before = schedule_complete_through(context.store, Tables.insider_transactions, LAST_SESSION)
    assert frontier_before == LAST_SESSION, "EDGAR's latest filing (2026-05-05) is inside the 7-day overlap"

    sec.zips["2026q2"] = _zip_tables([E1_ZIP, Z0, Z1])
    run_zip(context)
    frontier_after = schedule_complete_through(context.store, Tables.insider_transactions, LAST_SESSION)
    assert frontier_after == frontier_before, f"frontier moved: {frontier_before} -> {frontier_after}"
    return frontier_after


def test_a_zip_ingest_after_an_edgar_run_keeps_the_db_frontier(tmp_path, sqlite_store, monkeypatch, identity):
    """AC-009 (O12): a zip ingest neither erases nor moves the frontier the EDGAR rows give."""
    sec = _FakeSec(monkeypatch, identity, tmp_path)
    context = _context(tmp_path, sqlite_store)

    frontier = zip_after_edgar_keeps_the_frontier(context, sec, _run_zip)

    assert frontier is not None
    print(f"\nSANITY: after a zip quarter following the EDGAR run, complete_through stays {frontier.date()} (no manifest is read).")


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
    print("\nSANITY: a zip quarter with no EDGAR rows inserted 2 zip rows, logged 'loaded from zip (no EDGAR coverage)' and no WARNING.")


# A and B sit in the 2026q2 zip but carry a 2026-07-01 filing date (accepted after the quarter's last
# dissemination), so they fall after the zip floor where EDGAR reads.
A = Spec("0000000001-26-000200", "2026-07-01", tuple(Trade("2026-06-29", "S", 10 + i, 20, 500 - i) for i in range(3)))
A_EDGAR = Spec(A.accession, A.filed, (Trade("2026-06-29", "S", 30, 20, 470), Trade("2026-06-29", "S", 31, 20, 439)))
B = Spec("0000000001-26-000201", "2026-07-01", (Trade("2026-06-30", "P", 1, 20, 440),))
B_EDGAR = Spec(B.accession, B.filed, (Trade("2026-06-30", "P", 1, 20, 440), Trade("2026-06-30", "P", 2, 20, 442)))
C = Spec("0000000001-26-000202", "2026-05-06", (Trade("2026-05-05", "S", 3, 20, 439),))
N = Spec("0000000001-26-000300", "2026-07-01", (Trade("2026-06-30", "P", 4, 21, 443),))
X = Spec("0000000099-26-000002", "2026-07-01", (Trade("2026-06-30", "S", 9, 9, 9),), issuer_cik=OTHER_CIK)


def test_edgar_replaces_a_zip_filing_wholesale_and_keeps_its_quarter(tmp_path, sqlite_store, monkeypatch, identity, caplog):
    """AC-006 (c), (d), (e): a zip row never makes a filing look read by EDGAR; a filing listed under
    AAA only as an owner (X) becomes AAA's marker and is excluded nowhere."""
    caplog.set_level(logging.INFO)
    sec = _FakeSec(monkeypatch, identity, tmp_path)
    context = _context(tmp_path, sqlite_store)
    sec.zips["2026q2"] = _zip_tables([A, B, C])
    _run_zip(context)
    df_zip = _table(sqlite_store)

    caplog.clear()
    sec.publish(context, [A_EDGAR, B_EDGAR, C, N], owner_listed=((X, AAA_CIK),))
    _run_edgar(context)
    df_after = _table(sqlite_store)

    assert sorted(sec.read) == sorted([A.accession, B.accession, N.accession, X.accession]), "C (May) is behind the 2026q2 floor"
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
    marker = sqlite_store.load(Tables.insider_transactions, markers=True, where={"accession_number": X.accession})
    assert marker[["ticker", "security_type", "source"]].to_dict("records") == [{"ticker": "AAA", "security_type": "_empty", "source": "edgar"}]
    assert _mixed_accessions(df_after) == 0
    assert "insider EDGAR run: no filing excluded by the identity screen" in _messages(caplog, logging.INFO)

    sec.read.clear()
    _run_edgar(context)
    pd.testing.assert_frame_equal(_table(sqlite_store), df_after)
    assert sec.read == [], "every filing after the floor is now stored from EDGAR (X as AAA's marker)"
    print(
        f"\nSANITY: EDGAR re-read zip filings A (3 zip rows -> {len(rows_a)} EDGAR rows) and B (1 -> 2) filed after the 2026q2 floor, "
        f"keeping quarter 2026q2; C stays zip; N is new with NULL quarter; X (AAA only an owner) is AAA's marker; "
        f"an EDGAR rerun read nothing and changed none of the {len(df_after)} rows."
    )


BBB_CIK = "0000000002"
B0 = Spec("0000000002-26-000010", "2026-02-03", (Trade("2026-01-30", "P", 8, 6, 80),), issuer_cik=BBB_CIK, symbol="BBB")
B1 = Spec("0000000002-26-000100", "2026-05-04", (Trade("2026-05-01", "S", 3, 7, 77),), issuer_cik=BBB_CIK, symbol="BBB")


@pytest.fixture(scope="module")
def identity_two() -> Identity:
    """AAA (CIK 1) and BBB (CIK 2), both universe companies."""
    pairs = (("AAA", AAA_CIK), ("BBB", BBB_CIK))
    return build_identity(
        lineage=pd.DataFrame([{"cik": cik, "entity_id": f"E{cik}", "source": "roster", "confidence": None, "evidence": "test"} for _, cik in pairs]),
        tenure=pd.DataFrame(
            [
                {
                    "symbol": s,
                    "issuer_cik": cik,
                    "valid_from": pd.Timestamp("2006-01-03"),
                    "valid_to": None,
                    "n_filings": 100,
                    "source": "form345",
                    "evidence": "",
                }
                for s, cik in pairs
            ]
        ),
        roster=pd.DataFrame([{"ticker": s, "cik": cik} for s, cik in pairs]),
    )


def test_a_ticker_subset_run_never_deletes_another_universe_company(tmp_path, sqlite_store, monkeypatch, identity_two, caplog):
    """R-01: `insider-transactions -t AAA -F` re-reads AAA only; the stored-row sweep, which adjudicates
    against the whole universe, is skipped, so BBB's zip and EDGAR rows survive unchanged."""
    caplog.set_level(logging.INFO)
    sec = _FakeSec(monkeypatch, identity_two, tmp_path)
    context = _context(tmp_path, sqlite_store, universe=("AAA", "BBB"))
    sweep = "insider: stored-row sweep -- "

    sec.zips["2026q1"] = _zip_tables([Q0, B0])
    ins.fetch_insider_transactions(context, tickers=["AAA", "BBB"], years_history=1)
    sec.publish(context, [E1, B1])
    edgar.fetch_insider_edgar(context, tickers=["AAA", "BBB"], years_history=15)
    df_before = _table(sqlite_store)
    bbb = df_before["ticker"].eq("BBB")
    assert set(df_before.loc[bbb, "source"]) == {"zip", "edgar"}, "BBB holds a zip and an EDGAR filing"
    assert any(message.startswith(sweep) for message in _messages(caplog, logging.INFO)), "the full-universe run sweeps"

    caplog.clear()
    ins.fetch_insider_transactions(context, tickers=["AAA"], years_history=1, reparse=True)
    edgar.fetch_insider_edgar(context, tickers=["AAA"], years_history=15, full=True)
    df_after = _table(sqlite_store)

    infos = _messages(caplog, logging.INFO)
    assert not any(message.startswith(sweep) for message in infos), "a subset run never sweeps"
    skipped = next(message for message in infos if "sweep skipped" in message)
    pd.testing.assert_frame_equal(df_after[df_after["ticker"].eq("BBB")].reset_index(drop=True), df_before[bbb].reset_index(drop=True))
    assert set(df_after["accession_number"]) == set(df_before["accession_number"])
    print(
        f"\nSANITY: after `-t AAA -F` the table still holds all {len(df_after)} rows; BBB's {int(bbb.sum())} rows "
        f"(zip and EDGAR) are unchanged; log: '{skipped}'."
    )


def test_un_normalised_universe_tickers_sweep_against_the_loaded_universe(tmp_path, sqlite_store, monkeypatch, identity_two, caplog):
    """F-002: `[" bbb ", "aaa"]` equals the universe once normalised, so the sweep runs, and it must
    adjudicate against the loaded universe, not the raw spelling, so every universe company's rows survive."""
    caplog.set_level(logging.INFO)
    sec = _FakeSec(monkeypatch, identity_two, tmp_path)
    context = _context(tmp_path, sqlite_store, universe=("AAA", "BBB"))
    sec.zips["2026q1"] = _zip_tables([Q0, B0])
    ins.fetch_insider_transactions(context, tickers=["AAA", "BBB"], years_history=1)
    sec.publish(context, [E1, B1])
    edgar.fetch_insider_edgar(context, tickers=["AAA", "BBB"], years_history=15)
    df_before = _table(sqlite_store)

    caplog.clear()
    ins.fetch_insider_transactions(context, tickers=[" bbb ", "aaa"], years_history=1)
    df_after = _table(sqlite_store)

    sweep = next(message for message in _messages(caplog, logging.INFO) if message.startswith("insider: stored-row sweep -- "))
    assert "0 of" in sweep, sweep
    pd.testing.assert_frame_equal(df_after, df_before)
    print(f"\nSANITY: tickers [' bbb ', 'aaa'] ran the sweep against the loaded universe and kept all {len(df_after)} rows; log: '{sweep}'.")


def test_a_ticker_subset_edgar_run_never_hides_another_tickers_filing(tmp_path, sqlite_store, monkeypatch, identity_two):
    """R-07: the work list is each ticker's indexed filings minus its own stored EDGAR accessions, so a
    `-t AAA` EDGAR run cannot hide a BBB filing filed in between; the next full run reads and stores it."""
    sec = _FakeSec(monkeypatch, identity_two, tmp_path)
    context = _context(tmp_path, sqlite_store, universe=("AAA", "BBB"))
    sec.zips["2026q1"] = _zip_tables([Q0, B0])
    ins.fetch_insider_transactions(context, tickers=["AAA", "BBB"], years_history=1)
    sec.publish(context, [E1, B1])
    edgar.fetch_insider_edgar(context, tickers=["AAA", "BBB"], years_history=15)

    a_late = Spec("0000000001-26-000900", "2026-07-01", (Trade("2026-06-30", "P", 5, 10, 50),))
    b_gap = Spec("0000000002-26-000700", "2026-06-01", (Trade("2026-05-29", "P", 5, 10, 50),), issuer_cik=BBB_CIK, symbol="BBB")
    sec.publish(context, [E1, B1, a_late, b_gap])
    sec.read.clear()
    edgar.fetch_insider_edgar(context, tickers=["AAA"], years_history=15)
    subset_read = list(sec.read)

    sec.read.clear()
    edgar.fetch_insider_edgar(context, tickers=["AAA", "BBB"], years_history=15)
    full_read = list(sec.read)
    df = _table(sqlite_store)

    assert subset_read == [a_late.accession], "the -t AAA run reads AAA's new filing only"
    assert full_read == [b_gap.accession], "the full run reads BBB's filing from the gap, and nothing already stored"
    assert b_gap.accession in set(df["accession_number"]), "the BBB filing in the old gap is stored"
    assert _mixed_accessions(df) == 0
    print(
        f"\nSANITY: a -t AAA EDGAR run read only AAA's {a_late.filed} filing; the next full run read BBB's {b_gap.filed} filing "
        f"(and nothing already stored), so it is stored ({len(df)} rows, one source per accession)."
    )


def _postgres_add_column(engine: Any, name: str, df: pd.DataFrame) -> list[str]:
    """The store's Postgres schema evolution (`ADD COLUMN` for frame columns the table lacks), run on
    SQLite too, where `store.ensure_columns` is a no-op."""
    if not store_module.table_exists(engine, name):
        return []
    missing = [column for column in df.columns if column not in store_module._reflect(engine, name).c]
    with engine.begin() as conn:
        for column in missing:
            sql_type = ddl.sql_type(column, df[column].dtype, spec=Tables.insider_transactions)
            conn.execute(text(f'ALTER TABLE "{name}" ADD COLUMN "{column}" {sql_type}'))
    return missing


def test_an_edgar_first_run_then_a_zip_ingest_stamps_and_reconciles(tmp_path, sqlite_store, monkeypatch, identity, caplog):
    """R-03: EDGAR creates the table (no `quarter` column, no zip row); its reconcile is a no-op. A
    zip ingest afterwards adds `quarter` (Postgres schema evolution, emulated here on SQLite), stamps
    E1 and inserts Z1; EDGAR then re-reads Z1 whole and keeps its quarter."""
    caplog.set_level(logging.INFO)
    monkeypatch.setattr(store_module, "ensure_columns", _postgres_add_column)
    sec = _FakeSec(monkeypatch, identity, tmp_path)
    context = _context(tmp_path, sqlite_store)

    sec.publish(context, [E1, E2])
    _run_edgar(context)
    df_edgar_only = _table(sqlite_store)
    assert "quarter" not in df_edgar_only.columns and len(df_edgar_only) == 3
    assert not any("zip-sourced filing" in message for message in _messages(caplog, logging.INFO)), "no zip quarter -> no reconcile"

    sec.zips["2026q2"] = _zip_tables([E1_ZIP, Z1])
    assert _run_zip(context) == 1
    df_zip = _table(sqlite_store)
    assert list(_rows_of(df_zip, E1.accession)["quarter"]) == ["2026q2", "2026q2"]
    pd.testing.assert_frame_equal(_rows_of(df_zip, E1.accession).drop(columns="quarter"), _rows_of(df_edgar_only, E1.accession))
    assert _rows_of(df_zip, E2.accession)["quarter"].isna().all()
    assert list(_rows_of(df_zip, Z1.accession)["source"]) == ["zip"]

    z1_edgar = Spec(Z1.accession, Z1.filed, (*Z1.trades, Trade("2026-05-05", "P", 8, 12, 945)))
    sec.publish(context, [E1, E2, z1_edgar])
    sec.read.clear()
    _run_edgar(context)
    df_final = _table(sqlite_store)
    rows_z1 = _rows_of(df_final, Z1.accession)
    assert sec.read == [], "Z1 is behind the 2026q2 zip floor, so EDGAR never re-reads it"
    assert list(rows_z1["source"]) == ["zip"] and list(rows_z1["quarter"]) == ["2026q2"]
    assert _mixed_accessions(df_final) == 0
    print(
        f"\nSANITY: EDGAR-first run saved {len(df_edgar_only)} rows with no reconcile (no zip quarter); the zip then stamped "
        f"quarter 2026q2 on E1 and inserted Z1, which EDGAR never re-reads (filed before the floor); no accession holds two sources."
    )


def test_a_reconcile_failure_never_masks_the_run_error(tmp_path, sqlite_store, monkeypatch, identity, caplog):
    """R-03: when the EDGAR run fails, an error in the follow-up reconcile is logged and the run's own error propagates."""
    sec = _FakeSec(monkeypatch, identity, tmp_path)
    context = _context(tmp_path, sqlite_store)
    sec.zips["2026q1"] = _zip_tables([Q0])
    _run_zip(context)  # a stored zip quarter, so the run reconciles after it

    def run_fails(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("EDGAR listing failed")

    def reconcile_fails(context: Any, since: pd.Timestamp) -> int:
        raise KeyError("quarter")

    monkeypatch.setattr(edgar, "run_edgar_fetch", run_fails)
    monkeypatch.setattr(edgar, "replace_zip_accessions", reconcile_fails)
    with pytest.raises(RuntimeError, match="EDGAR listing failed"):
        _run_edgar(context)
    logged = next(record for record in caplog.records if record.levelno == logging.ERROR)
    print(f"\nSANITY: the run's RuntimeError propagated; the reconcile's KeyError was logged instead: '{logged.getMessage()}'.")
