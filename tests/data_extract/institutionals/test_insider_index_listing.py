"""AC-013: the live insider listing is the local EDGAR index, so no SEC page can shorten it.

The defect it replaces: JPM's owner-inclusive Atom search 503'd at offset 5,100 and the family
returned a truncated list. Here JPM's 5,200 indexed Form 4s are all in the work list, with no
network listing call, and each stays listed until it is stored. A Form 4 the key only reported as
an owner becomes one marker under that key and never hides the filing from its issuer."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from src.data_extract.utils.common import edgar_driver
from src.data_extract.utils.common.edgar_driver import EdgarScope, plan_fetch, run_edgar_fetch
from src.data_extract.utils.institutionals import fetch_insider_edgar as module
from src.data_store.schema import Tables
from tests.data_extract.edgar_fixtures import fake_context, fake_filing, identity_for, patch_index_filings, seed_index

_AS_OF = pd.Timestamp("2026-09-30")
_ROSTER = {"JPM": "19617", "OXY": "797468", "BRK-B": "1067983"}


def _form4_xml(issuer_cik: str, symbol: str) -> str:
    return f"""<ownershipDocument><documentType>4</documentType>
  <issuer><issuerCik>{issuer_cik}</issuerCik><issuerTradingSymbol>{symbol}</issuerTradingSymbol></issuer>
  <nonDerivativeTable><nonDerivativeTransaction>
    <transactionDate><value>2026-09-01</value></transactionDate><transactionCoding><transactionCode>P</transactionCode></transactionCoding>
    <transactionAmounts><transactionShares><value>10</value></transactionShares><transactionPricePerShare><value>20</value></transactionPricePerShare></transactionAmounts>
  </nonDerivativeTransaction></nonDerivativeTable>
</ownershipDocument>"""


def _form4(accession: str, cik: str, issuer_cik: str, symbol: str) -> SimpleNamespace:
    xml = _form4_xml(issuer_cik, symbol)
    return fake_filing(accession, cik, "2026-09-02", form="4", xml=lambda: xml, header=SimpleNamespace(acceptance_datetime="2026-09-02 16:05:00"))


@pytest.fixture
def ctx(tmp_path, sqlite_store, monkeypatch) -> Any:
    context = fake_context(tmp_path, sqlite_store, list(_ROSTER), ciks=list(_ROSTER.values()))
    identity = identity_for(_ROSTER)
    monkeypatch.setattr(edgar_driver, "load_identity", lambda _context: identity)
    monkeypatch.setattr(edgar_driver, "load_registrants", lambda _config_dir: {})
    monkeypatch.setattr(module, "screen_insider_rows", lambda frame, universe, identity: (frame, pd.DataFrame()))

    def no_network(*args: object, **kwargs: object) -> None:
        raise AssertionError("the insider listing must not call SEC")

    monkeypatch.setattr("edgar.httprequests.download_text", no_network)
    monkeypatch.setattr("edgar.Company", no_network)
    return context


def _jpm_book(n: int) -> list[tuple]:
    days = pd.date_range("2026-04-01", "2026-09-29", periods=n).normalize()
    return [(19617, "JPMORGAN CHASE & CO", "4", day.date().isoformat(), f"0000019617-26-{i:06d}") for i, day in enumerate(days)]


def test_every_indexed_form_4_is_listed_until_stored(ctx, sqlite_store):
    rows = _jpm_book(5_200)
    seed_index(ctx, rows)
    cik_map = pd.DataFrame({"ticker": ["JPM"], "cik": ["0000019617"]})
    scope = EdgarScope(edgar_driver.load_identity(ctx), {})
    fetch = module.insider_fetch(["JPM"])

    night1 = plan_fetch(ctx, fetch, cik_map, scope, _AS_OF, 15)
    stored = [r[4] for r in rows[:5_100]]
    sqlite_store.save(
        Tables.insider_transactions_live,
        pd.DataFrame({"accession_number": stored, "security_type": "nonderiv", "source_row_sequence": 1, "ticker": "JPM", "filing_date": _AS_OF}),
    )
    night2 = plan_fetch(ctx, fetch, cik_map, scope, _AS_OF + pd.Timedelta(days=1), 15)

    assert night1.size == 5_200
    assert night2.size == 100 and night2.units["JPM"]["accession"].tolist() == [r[4] for r in rows[5_100:]]
    print("\n=== SANITY CHECK: AC-013 ===")
    print(f"  5,200 JPM Form 4s indexed -> work list {night1.size} with no SEC listing call; after 5,100 stored -> {night2.size} still listed.")


def test_an_owner_role_form_4_is_a_marker_and_never_hides_the_issuers_filing(ctx, sqlite_store, monkeypatch):
    """BRK reports a Form 4 on OXY as a 10% owner: the index lists it under both CIKs."""
    accession = "0001067983-26-000042"
    seed_index(ctx, [(797468, "OCCIDENTAL PETROLEUM", "4", "2026-09-02", accession), (1067983, "BERKSHIRE HATHAWAY", "4", "2026-09-02", accession)])
    patch_index_filings(monkeypatch, {accession: _form4(accession, "1067983", "0000797468", "OXY")})
    # the live table exists with its full column set, as its DDL makes it on Postgres
    stamps = dict.fromkeys(("transaction_date", "filing_date", "acceptance_datetime", "fetched_at"), [pd.Timestamp("2026-01-02")])
    numbers = dict.fromkeys(("shares", "price_per_share", "value_usd"), [1.0])
    seed = {c: [None] for c in module._LIVE_COLUMNS} | stamps | numbers
    seed |= {"accession_number": ["seed"], "security_type": ["nonderiv"], "source_row_sequence": [1], "ticker": ["ZZZ"], "issuer_cik": ["0"]}
    sqlite_store.save(Tables.insider_transactions_live, pd.DataFrame(seed))

    first = run_edgar_fetch(ctx, list(_ROSTER), 15, module.insider_fetch(list(_ROSTER)), as_of=_AS_OF, refresh_index=False, max_workers=1)

    shown = sqlite_store.load(Tables.insider_transactions_live, markers=True, where={"ticker": list(_ROSTER)}).sort_values("ticker")
    assert shown[["ticker", "security_type", "source_row_sequence"]].to_dict("records") == [
        {"ticker": "BRK-B", "security_type": "_empty", "source_row_sequence": 0},
        {"ticker": "OXY", "security_type": "nonderiv", "source_row_sequence": 1},
    ]
    assert pd.isna(shown.loc[shown["ticker"] == "BRK-B", "issuer_cik"]).all()  # a marker never claims an issuer
    assert sqlite_store.load(Tables.insider_transactions_live, where={"ticker": list(_ROSTER)})["ticker"].tolist() == ["OXY"]
    assert first.outcomes["BRK-B"].markers == 1 and first.outcomes["OXY"].markers == 0

    sqlite_store.delete(Tables.insider_transactions_live, {"ticker": "OXY"})
    again = run_edgar_fetch(
        ctx, list(_ROSTER), 15, module.insider_fetch(list(_ROSTER)), as_of=_AS_OF + pd.Timedelta(days=1), refresh_index=False, max_workers=1
    )
    assert list(again.work.units) == ["OXY"]
    coverage = sqlite_store.load(Tables.insider_transactions_live_coverage)
    assert set(coverage["ticker"]) == set(_ROSTER) and (pd.to_datetime(coverage["complete_through"]) == _AS_OF + pd.Timedelta(days=1)).all()
    print("\n=== SANITY CHECK: owner-role Form 4 ===")
    print("  BRK-B (owner) -> one '_empty' marker; OXY (issuer) -> its transaction row. Deleting OXY's row lists the filing")
    print("  again for OXY only: BRK-B's marker never hides it. Every key's coverage advanced to the run date.")


def test_coverage_stops_the_day_before_the_oldest_unread_filing(ctx, sqlite_store, monkeypatch):
    accession = "0000797468-26-000007"
    seed_index(ctx, [(797468, "OCCIDENTAL PETROLEUM", "4", "2026-09-02", accession)])
    filing = _form4(accession, "797468", "0000797468", "OXY")
    filing.xml = lambda: (_ for _ in ()).throw(module.TransientReadError("HTTP 503"))
    patch_index_filings(monkeypatch, {accession: filing})

    run_edgar_fetch(ctx, ["OXY"], 15, module.insider_fetch(["OXY"]), as_of=_AS_OF, refresh_index=False, max_workers=1)

    coverage = sqlite_store.load(Tables.insider_transactions_live_coverage)
    assert pd.Timestamp(coverage["complete_through"].iloc[0]) == pd.Timestamp("2026-09-01")
    print("\nSANITY: a Form 4 filed 2026-09-02 that 503'd holds OXY's coverage at 2026-09-01, so the aggregation frontier stays honest.")
