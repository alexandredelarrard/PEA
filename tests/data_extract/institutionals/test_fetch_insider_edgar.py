"""Daily insider EDGAR orchestration and coverage semantics."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from src.data_extract.utils.institutionals import fetch_insider_edgar as module
from src.data_store.ddl import columns_from_frame
from src.data_store.schema import Tables


class _Filing:
    accession_number = "0000000001-26-000001"
    filing_date = pd.Timestamp("2026-07-02")
    header = SimpleNamespace(acceptance_datetime="2026-07-02 16:05:00")

    @staticmethod
    def xml() -> str:
        return "<ownershipDocument/>"


def _parsed_transaction() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source_row_sequence": 1,
                "security_type": "nonderiv",
                "issuer_cik": "0000000001",
                "ticker": "AAA",
                "transaction_date": pd.Timestamp("2026-07-01"),
                "transaction_code": "P",
                "shares": 10.0,
                "price_per_share": 20.0,
                "value_usd": 200.0,
            }
        ]
    )


def test_successful_zero_filing_scan_still_advances_ticker_coverage(monkeypatch):
    monkeypatch.setattr(module, "insider_filings", lambda *args, **kwargs: [])
    out = module.build_ticker_insider_edgar(
        "AAA",
        "1",
        universe=["AAA"],
        identity=object(),
        scan_through=pd.Timestamp("2026-09-22"),
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
        "SANITY: a successful AAA scan with zero new filings wrote no transaction sentinel "
        "but advanced AAA's explicit coverage through 2026-09-22."
    )


def test_duplicate_listing_is_idempotent_and_keeps_acceptance_time(monkeypatch):
    monkeypatch.setattr(module, "insider_filings", lambda *args, **kwargs: [_Filing(), _Filing()])
    monkeypatch.setattr(
        module,
        "parse_ownership_xml",
        lambda xml: (_parsed_transaction(), pd.DataFrame(columns=["footnote_id", "footnote_text"])),
    )
    monkeypatch.setattr(
        module,
        "_filter_universe",
        lambda frame, universe, identity: (frame, pd.DataFrame()),
    )
    out = module.build_ticker_insider_edgar(
        "AAA",
        "1",
        universe=["AAA"],
        identity=object(),
        scan_through=pd.Timestamp("2026-09-22"),
    )
    live = out[Tables.insider_transactions_live]
    assert len(live) == 1
    assert live.iloc[0]["acceptance_datetime"] == pd.Timestamp("2026-07-02 16:05:00")
    assert live.iloc[0]["accession_number"] == _Filing.accession_number
    print("SANITY: listing the same accession twice produced one live PK row and retained the " "16:05 EDGAR acceptance timestamp.")


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
    print("SANITY: filing acceptance/fetch/update clocks retain intraday TIMESTAMP precision; " "only the inclusive coverage frontier is a DATE.")


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
        return '<feed xmlns="http://www.w3.org/2005/Atom">' f"{feeds[family]}</feed>"

    monkeypatch.setattr(module, "download_text", download)
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
        "SANITY: issuer ownership discovery retains a Form 4 submitted under its reporting "
        "owner's accession CIK and filters a prefix-matched 424B2."
    )
