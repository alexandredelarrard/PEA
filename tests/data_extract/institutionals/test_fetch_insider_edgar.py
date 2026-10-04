"""Daily insider EDGAR orchestration and coverage semantics."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from src.data_extract.utils.institutionals import fetch_insider_edgar as module
from src.data_store.ddl import columns_from_frame
from src.data_store.schema import Tables
from tests.data_extract.edgar_fixtures import identity_for

_FORM4_XML = """<ownershipDocument><documentType>4</documentType>
  <issuer><issuerCik>0000000001</issuerCik><issuerTradingSymbol>AAA</issuerTradingSymbol></issuer>
  <nonDerivativeTable><nonDerivativeTransaction>
    <transactionDate><value>2026-07-01</value></transactionDate><transactionCoding><transactionCode>P</transactionCode></transactionCoding>
    <transactionAmounts><transactionShares><value>10</value></transactionShares><transactionPricePerShare><value>20</value></transactionPricePerShare></transactionAmounts>
  </nonDerivativeTransaction></nonDerivativeTable>
</ownershipDocument>"""


class _Filing:
    accession_number = "0000000001-26-000001"
    cik = 1
    form = "4"
    filing_date = pd.Timestamp("2026-07-02")
    header = SimpleNamespace(acceptance_datetime="2026-07-02 16:05:00")

    @staticmethod
    def xml() -> str:
        return _FORM4_XML


def test_coverage_runs_through_the_run_date_or_the_day_before_the_oldest_unread_filing():
    clean = module.insider_coverage("AAA", pd.Timestamp("2026-09-22"), [])
    behind = module.insider_coverage("AAA", pd.Timestamp("2026-09-22"), [pd.Timestamp("2026-09-10"), pd.Timestamp("2026-09-05")])
    assert clean["complete_through"].tolist() == [pd.Timestamp("2026-09-22")]
    assert behind["complete_through"].tolist() == [pd.Timestamp("2026-09-04")]
    assert list(clean.columns) == ["ticker", "complete_through", "updated_at"]
    print("SANITY: a clean AAA run covers through the run date; one with unread filings stops the day before the oldest (2026-09-04).")


def test_one_filing_parses_to_one_live_row_with_its_acceptance_time(monkeypatch):
    monkeypatch.setattr(module, "screen_insider_rows", lambda frame, universe, identity: (frame, pd.DataFrame()))
    scope = module.EdgarScope(identity_for({"AAA": "1"}), {})
    out = module.parse_insider("AAA", "0000000001", module.FilingStamp.of(_Filing(), "0000000001"), scope, universe=["AAA"])
    live = out[Tables.insider_transactions_live]
    assert len(live) == 1
    assert live.iloc[0]["acceptance_datetime"] == pd.Timestamp("2026-07-02 16:05:00")
    assert live.iloc[0]["accession_number"] == _Filing.accession_number
    assert live.iloc[0]["value_usd"] == 200.0
    print("SANITY: one Form 4 parsed to one live PK row and retained the 16:05 EDGAR acceptance timestamp.")


def test_a_filing_on_another_issuer_parses_to_nothing():
    """Listed under AAA because AAA reported it as an owner: no rows, so the driver stores AAA's marker."""
    scope = module.EdgarScope(identity_for({"AAA": "1", "BBB": "2"}), {})
    assert module.parse_insider("BBB", "0000000002", module.FilingStamp.of(_Filing(), "0000000002"), scope, universe=["AAA", "BBB"]) == {}
    print("SANITY: a Form 4 whose XML issuer is AAA parses to nothing for BBB (BBB only an owner -> one marker).")


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
