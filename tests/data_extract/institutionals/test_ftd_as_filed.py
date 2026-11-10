"""FTD ingest as filed (P4): the scope comes from the lineage, Sharadar and curated CUSIP-6, never from `universe_cusip`;
a scope key that grows re-reads the cached periods for that key only."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from src.data_extract.utils.zips import fetch_fails_to_deliver as ftd
from src.data_store.schema import Tables
from tests.canonical_schema import create_canonical
from tests.support.cusip_build import write_ftd_zips
from tests.support.universe import seed_universe

_HEADER = "SETTLEMENT DATE|CUSIP|SYMBOL|QUANTITY (FAILS)|DESCRIPTION|PRICE\n"
RUN_DATE = pd.Timestamp("2026-11-10")
#: The stored source columns of `sec_fails_to_deliver_security`.
SOURCE = ["date", "trade_date", "cusip", "source_symbol", "description", "price", "fails_quantity", "fails_value", "period"]


@pytest.fixture
def sqlite_store(sqlite_store):
    create_canonical(
        sqlite_store.engine, [Tables.universe_cik_symbol, Tables.symbol_tenure, Tables.universe_cusip, Tables.sec_fails_to_deliver_security]
    )
    return sqlite_store


def _context(store, tmp_path, tickers: tuple[str, ...]) -> Any:
    return SimpleNamespace(
        store=store,
        log=logging.getLogger("test.ftd_as_filed"),
        paths={"DATA_STORE": tmp_path},
        config_dir=str(tmp_path / "configs"),
        config=SimpleNamespace(local=SimpleNamespace(paths=SimpleNamespace(fails_deliver="ftd"))),
        universe=seed_universe(store, tickers),
    )


def _lineage(store, rows: list[tuple[str, str, str, pd.Timestamp]]) -> None:
    """One cik_window and one symbol row per (ticker, cik, symbol, scope_changed_at)."""
    frame = pd.DataFrame(
        [
            {
                "company_id": f"E{cik}",
                "canonical_ticker": ticker,
                "cik": cik,
                "role": role,
                "symbol": symbol if role == "symbol" else "",
                "valid_from": pd.Timestamp("2000-01-01"),
                "valid_to": None,
                "status": "corroborated",
                "sources": "form345,roster",
                "scope_changed_at": stamp,
            }
            for ticker, cik, symbol, stamp in rows
            for role in ("cik_window", "symbol")
        ]
    ).drop_duplicates(["cik", "role", "symbol"])
    store.save(Tables.universe_cik_symbol, frame)


def _stored(store) -> pd.DataFrame:
    rows = store.load(Tables.sec_fails_to_deliver_security, columns=SOURCE, optional=True)
    return pd.DataFrame(columns=SOURCE) if rows is None else rows.sort_values(["period", "cusip"], ignore_index=True)


def test_ingest_keeps_the_class_c_line_under_the_alphabet_prefix_with_no_universe_cusip(sqlite_store, tmp_path, caplog):
    """GOOGL is the lineage symbol; the GOOG line shares its CUSIP-6, so it is stored as filed. No `universe_cusip` exists."""
    context = _context(sqlite_store, tmp_path, ("GOOGL",))
    _lineage(sqlite_store, [("GOOGL", "0001652044", "GOOGL", pd.Timestamp("2020-01-01"))])
    write_ftd_zips(
        tmp_path / "ftd",
        {
            "202601a": _HEADER
            + "20260105|02079K305|GOOGL|100|ALPHABET INC CL A|300.0\n"
            + "20260105|02079K107|GOOG|40|ALPHABET INC CL C|301.0\n"
            + "20260105|999999ZZZ|ZZZ|9|OTHER CO|3.0\n"
        },
    )
    assert sqlite_store.row_count(Tables.universe_cusip) == 0

    with caplog.at_level(logging.INFO):
        ftd.fetch_fails_to_deliver(context, as_of=RUN_DATE)

    stored = _stored(sqlite_store)
    assert stored[["cusip", "source_symbol", "fails_quantity"]].values.tolist() == [["02079K107", "GOOG", 40.0], ["02079K305", "GOOGL", 100.0]]
    print("\nSANITY (no universe_cusip): the GOOGL lineage symbol keeps its line and puts CUSIP-6 02079K in scope, so the GOOG")
    print(f"  class C line is stored as filed: {stored[['cusip', 'source_symbol', 'fails_quantity']].values.tolist()}; ZZZ is not.")


def test_a_new_key_rereads_the_cached_periods_for_that_key_only(sqlite_store, tmp_path, caplog, monkeypatch):
    """XYZ joins company ABC's lineage today and has no stored line: both cached periods are re-read for XYZ alone (a keyed
    upsert); the stored ABC lines survive and nothing else is added. A stale key with no line is never re-read."""
    context = _context(sqlite_store, tmp_path, ("ABC",))
    _lineage(
        sqlite_store,
        [
            ("ABC", "0000000777", "ABC", pd.Timestamp("2020-01-01")),
            ("ABC", "0000000778", "XYZ", RUN_DATE),
            ("ABC", "0000000779", "OLD", pd.Timestamp("2020-01-01")),
        ],
    )
    write_ftd_zips(
        tmp_path / "ftd",
        {
            "202601a": _HEADER
            + "20260105|000111AAA|ABC|100|ABC CORP|10.0\n20260105|555555XXX|XYZ|7|XYZ CORP|2.0\n20260105|999999ZZZ|ZZZ|9|OTHER|3.0\n",
            "202601b": _HEADER + "20260120|000111AAA|ABC|50|ABC CORP|10.0\n20260120|555555XXX|XYZ|8|XYZ CORP|2.0\n",
        },
    )
    stored_abc = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-05", "2026-01-20"]),
            "trade_date": pd.to_datetime(["2026-01-02", "2026-01-16"]),
            "cusip": ["000111AAA", "000111AAA"],
            "source_symbol": ["ABC", "ABC"],
            "description": ["ABC CORP", "ABC CORP"],
            "price": [10.0, 10.0],
            "fails_quantity": [100.0, 50.0],
            "fails_value": [1000.0, 500.0],
            "period": ["202601a", "202601b"],
        }
    )
    sqlite_store.save(Tables.sec_fails_to_deliver_security, stored_abc)
    reads: list[str] = []
    reader = ftd.read_zip_text
    monkeypatch.setattr(ftd, "read_zip_text", lambda path, log=None: reads.append(path.stem.removeprefix("cnsfails")) or reader(path, log=log))
    with caplog.at_level(logging.INFO):
        ftd.fetch_fails_to_deliver(context, as_of=RUN_DATE)

    stored = _stored(sqlite_store)
    assert stored[["period", "cusip", "fails_quantity"]].values.tolist() == [
        ["202601a", "000111AAA", 100.0],
        ["202601a", "555555XXX", 7.0],
        ["202601b", "000111AAA", 50.0],
        ["202601b", "555555XXX", 8.0],
    ]
    assert sorted(reads) == ["202601a", "202601b"], "each cached period is read once, for the new key only"
    assert any("re-read" in record.getMessage() for record in caplog.records), "one log line names the scope-growth re-read"
    print(f"\nSANITY: new key XYZ (scope changed {RUN_DATE.date()}) re-read from {sorted(reads)}; 2 XYZ lines upserted, the 2 ABC lines")
    print("  survive, ZZZ never stored; OLD (stale, no line) triggers nothing.")
