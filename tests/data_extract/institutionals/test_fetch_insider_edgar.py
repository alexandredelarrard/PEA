"""Daily insider EDGAR orchestration and coverage semantics."""

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


def test_successful_zero_filing_scan_still_advances_ticker_coverage(monkeypatch):
    monkeypatch.setattr(module, "insider_filings", lambda *args, **kwargs: [])
    out = module.build_ticker_insider_edgar(
        "AAA",
        "1",
        universe=["AAA"],
        identity=cast(Any, object()),
        scan_through=pd.Timestamp("2026-09-22"),
        scope=module.EdgarScope(),
    )
    assert out[Tables.insider_transactions_live].empty
    assert out[Tables.insider_transactions_live_coverage].to_dict("records") == [
        {
            "ticker": "AAA",
            "complete_through": pd.Timestamp("2026-09-22"),
            "updated_at": out[Tables.insider_transactions_live_coverage].iloc[0]["updated_at"],
        }
    ]
    print(
        "SANITY: a successful AAA scan with zero new filings wrote no transaction sentinel but advanced AAA's explicit coverage through 2026-09-22."
    )


def test_duplicate_listing_is_idempotent_and_keeps_acceptance_time(monkeypatch):
    monkeypatch.setattr(module, "insider_filings", lambda *args, **kwargs: [_Filing(), _Filing()])
    monkeypatch.setattr(module, "screen_insider_rows", lambda frame, universe, identity: (frame, pd.DataFrame()))
    out = module.build_ticker_insider_edgar(
        "AAA",
        "1",
        universe=["AAA"],
        identity=cast(Any, object()),
        scan_through=pd.Timestamp("2026-09-22"),
        scope=module.EdgarScope(),
    )
    live = out[Tables.insider_transactions_live]
    assert len(live) == 1
    assert live.iloc[0]["acceptance_datetime"] == pd.Timestamp("2026-07-02 16:05:00")
    assert live.iloc[0]["accession_number"] == _Filing.accession_number
    assert live.iloc[0]["value_usd"] == 200.0
    print("SANITY: listing the same accession twice produced one live PK row and retained the 16:05 EDGAR acceptance timestamp.")


def test_live_audit_clocks_are_timestamps_not_dates():
    live = pd.DataFrame(
        {
            "accession_number": ["A"],
            "security_type": ["nonderiv"],
            "source_row_sequence": [1],
            "acceptance_datetime": [pd.Timestamp("2026-07-02 16:05:00")],
            "fetched_at": [pd.Timestamp("2026-07-02 16:06:00")],
        }
    )
    coverage = pd.DataFrame(
        {
            "ticker": ["AAA"],
            "complete_through": [pd.Timestamp("2026-07-02")],
            "updated_at": [pd.Timestamp("2026-07-02 16:06:00")],
        }
    )
    live_types = dict(columns_from_frame(Tables.insider_transactions_live, live))
    coverage_types = dict(columns_from_frame(Tables.insider_transactions_live_coverage, coverage))
    assert live_types["acceptance_datetime"] == "TIMESTAMP"
    assert live_types["fetched_at"] == "TIMESTAMP"
    assert coverage_types["complete_through"] == "DATE"
    assert coverage_types["updated_at"] == "TIMESTAMP"
    print("SANITY: filing acceptance/fetch/update clocks retain intraday TIMESTAMP precision; only the inclusive coverage frontier is a DATE.")


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


def test_live_listing_walks_every_event_cik_of_the_scope(monkeypatch):
    """AC-029: a predecessor's (or co-registrant's) Forms 3/4/5 are listed from its own CIK, both from
    its submissions and its owner-inclusive search; a filing from another filer is skipped and counted."""
    from src.data_extract.utils.common.edgar_driver import EdgarScope
    from tests.data_extract.common.scope_fixtures import SENTINEL, dated_identity, filing, patch_company

    identity = dated_identity(
        [("XOM", "0000034088", "cik_window", SENTINEL, "2026-07-01"), ("XOM", "0002115436", "cik_window", "2026-07-01", None)],
        {"XOM": "0002115436"},
    )
    built = patch_company(
        monkeypatch,
        {
            34088: [filing("pred-4", "2026-08-03", 34088), filing("foreign-4", "2026-08-04", 825313)],
            2115436: [filing("succ-4", "2026-08-05", 2115436)],
        },
    )
    searched: list[str] = []

    def owner_search(ticker, cik, **kwargs):
        searched.append(cik)
        return [filing(f"owner-{cik[-4:]}", "2026-08-06", int(cik))]

    monkeypatch.setattr(module, "ownership_filings", owner_search)
    scope = EdgarScope(identity)
    out = module.insider_filings("XOM", "0002115436", since=None, through=pd.Timestamp("2026-09-01"), done_accessions=frozenset(), scope=scope)

    assert [f.accession_number for f in out] == ["pred-4", "succ-4", "owner-4088", "owner-5436"]
    assert sorted(built) == [34088, 2115436]
    assert searched == ["0000034088", "0002115436"]
    assert scope.guard.skipped == 1
    print("SANITY: insider live listed both XOM CIKs (submissions and owner search); the foreign Form 4 was skipped and counted.")
