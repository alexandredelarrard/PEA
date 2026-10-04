"""Daily insider EDGAR parse: one filing -> `insider_transactions` rows (source='edgar'), the
owner-role rule and the run's exclusion collector."""

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


def _parse(ticker: str, cik: str, universe: list[str], excluded: list[pd.DataFrame]) -> dict:
    scope = module.EdgarScope(identity_for({"AAA": "1", "BBB": "2"}), {})
    return module.parse_insider(ticker, cik, module.FilingStamp.of(_Filing(), cik), scope, universe=universe, excluded=excluded)


def test_one_filing_parses_to_one_edgar_row_with_its_acceptance_time(monkeypatch):
    monkeypatch.setattr(module, "screen_insider_rows", lambda frame, universe, identity: (frame, pd.DataFrame()))
    excluded: list[pd.DataFrame] = []
    out = _parse("AAA", "0000000001", ["AAA"], excluded)
    rows = out[Tables.insider_transactions]
    assert set(out) == {Tables.insider_transactions, Tables.insider_footnotes}
    assert len(rows) == 1 and excluded == []
    assert rows.iloc[0]["acceptance_datetime"] == pd.Timestamp("2026-07-02 16:05:00")
    assert rows.iloc[0]["accession_number"] == _Filing.accession_number
    assert rows.iloc[0]["value_usd"] == 200.0
    assert rows.iloc[0]["source"] == "edgar" and "quarter" not in rows.columns, "the EDGAR frame never carries the zip quarter"
    print("SANITY: one Form 4 parsed to one insider_transactions PK row (source=edgar, no quarter) with the 16:05 acceptance time.")


def test_a_filing_on_another_issuer_parses_to_nothing():
    """Listed under BBB because BBB reported it as an owner: no rows, so the driver stores BBB's marker."""
    assert _parse("BBB", "0000000002", ["AAA", "BBB"], []) == {}
    print("SANITY: a Form 4 whose XML issuer is AAA parses to nothing for BBB (BBB only an owner -> one marker).")


def test_rejected_rows_go_to_the_run_collector_not_to_a_table(monkeypatch):
    rejected = pd.DataFrame(
        {"accession_number": ["X"], "transaction_code": ["P"], "claimed_ticker": ["AAA"], "reject_reason": ["entity_mismatch"], "extra": [1]}
    )
    monkeypatch.setattr(module, "screen_insider_rows", lambda frame, universe, identity: (frame.iloc[0:0], rejected))
    excluded: list[pd.DataFrame] = []
    out = _parse("AAA", "0000000001", ["AAA"], excluded)
    assert out[Tables.insider_transactions].empty
    assert len(excluded) == 1 and list(excluded[0].columns) == ["accession_number", "transaction_code", "claimed_ticker", "reject_reason"]
    print("SANITY: a rejected filing is stored nowhere; only its four warning columns reach the run's exclusion collector.")


def test_the_fetch_counts_only_edgar_rows_as_done_per_key():
    fetch = module.insider_fetch(["AAA"])
    assert fetch.done == Tables.insider_transactions and fetch.done_where == {"source": "edgar"} and fetch.done_scope == "key"
    assert set(fetch.tables) == {Tables.insider_transactions, Tables.insider_footnotes}
    print("SANITY: the EDGAR leg writes the single table and its done set is per key on source='edgar' rows (markers included).")


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
