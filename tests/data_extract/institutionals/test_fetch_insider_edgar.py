"""Daily insider EDGAR build: the per-ticker frames target `insider_transactions`, the listing window
and the owner-inclusive discovery."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pandas as pd

from src.data_extract.utils.institutionals import fetch_insider_edgar as module
from src.data_store.ddl import columns_from_frame
from src.data_store.schema import Tables

_FORM4_XML = """<ownershipDocument><documentType>4</documentType>
  <issuer><issuerCik>0000000001</issuerCik><issuerTradingSymbol>AAA</issuerTradingSymbol></issuer>
  <nonDerivativeTable><nonDerivativeTransaction>
    <transactionDate><value>2026-07-01</value></transactionDate><transactionCoding><transactionCode>P</transactionCode></transactionCoding>
    <transactionAmounts><transactionShares><value>10</value></transactionShares><transactionPricePerShare><value>20</value></transactionPricePerShare></transactionAmounts>
  </nonDerivativeTransaction></nonDerivativeTable>
</ownershipDocument>"""


class _Filing:
    accession_number = "0000000001-26-000001"
    filing_date = pd.Timestamp("2026-07-02")
    header = SimpleNamespace(acceptance_datetime="2026-07-02 16:05:00")

    @staticmethod
    def xml() -> str:
        return _FORM4_XML


def _build(excluded: list[pd.DataFrame]) -> dict:
    return module.build_ticker_insider_edgar(
        "AAA",
        "1",
        universe=["AAA"],
        identity=cast(Any, object()),
        scan_through=pd.Timestamp("2026-09-22"),
        scope=module.EdgarScope(None, {}),
        excluded=excluded,
    )


def test_a_zero_filing_scan_builds_empty_frames_for_the_two_declared_tables(monkeypatch):
    monkeypatch.setattr(module, "insider_filings", lambda *args, **kwargs: [])
    excluded: list[pd.DataFrame] = []
    out = _build(excluded)
    assert set(out) == {Tables.insider_transactions, Tables.insider_footnotes}
    assert out[Tables.insider_transactions].empty and excluded == []
    print("SANITY: a scan with zero new filings builds only empty insider_transactions/insider_footnotes frames; no coverage row exists any more.")


def test_duplicate_listing_is_idempotent_and_keeps_acceptance_time(monkeypatch):
    monkeypatch.setattr(module, "insider_filings", lambda *args, **kwargs: [_Filing(), _Filing()])
    monkeypatch.setattr(module, "screen_insider_rows", lambda frame, universe, identity: (frame, pd.DataFrame()))
    out = _build([])
    rows = out[Tables.insider_transactions]
    assert len(rows) == 1
    assert rows.iloc[0]["acceptance_datetime"] == pd.Timestamp("2026-07-02 16:05:00")
    assert rows.iloc[0]["accession_number"] == _Filing.accession_number
    assert rows.iloc[0]["value_usd"] == 200.0
    assert rows.iloc[0]["source"] == "edgar" and "quarter" not in rows.columns, "the EDGAR frame never carries the zip quarter"
    print(
        "SANITY: listing the same accession twice produced one insider_transactions PK row (source=edgar, no quarter) with the 16:05 acceptance time."
    )


def test_rejected_rows_go_to_the_run_collector_not_to_a_table(monkeypatch):
    monkeypatch.setattr(module, "insider_filings", lambda *args, **kwargs: [_Filing()])
    rejected = pd.DataFrame(
        {"accession_number": ["X"], "transaction_code": ["P"], "claimed_ticker": ["AAA"], "reject_reason": ["entity_mismatch"], "extra": [1]}
    )
    monkeypatch.setattr(module, "screen_insider_rows", lambda frame, universe, identity: (frame.iloc[0:0], rejected))
    excluded: list[pd.DataFrame] = []
    out = _build(excluded)
    assert out[Tables.insider_transactions].empty
    assert len(excluded) == 1 and list(excluded[0].columns) == ["accession_number", "transaction_code", "claimed_ticker", "reject_reason"]
    print("SANITY: a rejected filing is stored nowhere; only its four warning columns reach the run's exclusion collector.")


def test_each_ticker_lists_from_seven_days_before_its_own_latest_stored_filing(sqlite_store):
    ctx = cast(Any, SimpleNamespace(store=sqlite_store))
    no_rows = pd.Timestamp.today().normalize() - pd.DateOffset(years=15)
    empty = module.listing_since_by_ticker(ctx, ["AAA"], 15)
    sqlite_store.save(
        Tables.insider_transactions,
        pd.DataFrame(
            {
                "accession_number": ["a", "b", "c"],
                "security_type": "nonderiv",
                "row_sequence": 1,
                "ticker": ["AAA", "AAA", "BBB"],
                "filing_date": pd.to_datetime(["2026-05-01", "2026-06-10", "2026-03-02"]),
            }
        ),
    )
    since = module.listing_since_by_ticker(ctx, ["AAA", "BBB", "CCC"], 15)
    assert empty == {"AAA": no_rows}
    assert since == {"AAA": pd.Timestamp("2026-06-03"), "BBB": pd.Timestamp("2026-02-23"), "CCC": no_rows}
    print(
        "SANITY: every ticker lists from its own max(filing_date) - 7 days (AAA 2026-06-10 -> 2026-06-03, BBB 2026-03-02 -> 2026-02-23); "
        "a ticker with no stored row (and every ticker on an empty table) lists from today - years_history."
    )


def test_edgar_audit_clocks_are_timestamps_not_dates():
    frame = pd.DataFrame(
        {
            "accession_number": ["A"],
            "security_type": ["nonderiv"],
            "row_sequence": [1],
            "acceptance_datetime": [pd.Timestamp("2026-07-02 16:05:00")],
            "fetched_at": [pd.Timestamp("2026-07-02 16:06:00")],
        }
    )
    types = dict(columns_from_frame(Tables.insider_transactions, frame))
    assert types["acceptance_datetime"] == "TIMESTAMP"
    assert types["fetched_at"] == "TIMESTAMP"
    print("SANITY: filing acceptance and fetch clocks keep intraday TIMESTAMP precision in insider_transactions.")


def test_owner_inclusive_atom_finds_reporting_owner_accessions(monkeypatch):
    feeds = {
        "3": "",
        "4": """
          <entry><content><accession-number>0001182379-26-000004</accession-number>
          <filing-date>2026-06-02</filing-date><filing-type>4</filing-type></content></entry>
          <entry><content><accession-number>0000000000-26-000001</accession-number>
          <filing-date>2026-06-02</filing-date><filing-type>424B2</filing-type></content></entry>
        """,
        "5": "",
    }

    def download(url: str) -> str:
        family = next(form for form in feeds if f"type={form}&" in url)
        return f'<feed xmlns="http://www.w3.org/2005/Atom">{feeds[family]}</feed>'

    monkeypatch.setattr("edgar.httprequests.download_text", download)
    filings = module.ownership_filings(
        "DLR",
        "0001494877",
        since=pd.Timestamp("2026-04-01"),
        through=pd.Timestamp("2026-06-30"),
        done_accessions=frozenset(),
    )
    assert [filing.accession_number for filing in filings] == ["0001182379-26-000004"]
    assert filings[0].cik == 1494877
    print(
        "SANITY: issuer ownership discovery retains a Form 4 submitted under its reporting owner's accession CIK and filters a prefix-matched 424B2."
    )
