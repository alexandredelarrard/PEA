"""FTD ingest as filed (P4): the scope comes from the lineage, Sharadar and curated CUSIP-6, never from `universe_cusip`;
a scope key that grows re-reads the cached periods for that key only."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from src.data_extract.utils.common.security_lines import SecurityManual, ftd_scope
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


def test_the_ftd_scope_keeps_equity_symbols_not_notes_debt_or_noise_symbols() -> None:
    """ABBV is a filed equity symbol; ABBV28B is a Notes cover-page (`dei`-only) bond symbol and TRUE a `dei` noise row:
    only ABBV is an FTD scope symbol."""
    rows = [("ABBV", "form345,roster", "corroborated"), ("ABBV28B", "dei", "single_source"), ("TRUE", "dei", "noise")]
    lineage = pd.DataFrame(
        [
            {
                "canonical_ticker": "ABBV",
                "cik": "0001551152",
                "role": "symbol",
                "symbol": symbol,
                "sources": sources,
                "status": status,
                "valid_from": pd.Timestamp("2013-01-02"),
                "valid_to": None,
            }
            for symbol, sources, status in rows
        ]
    )

    scope = ftd_scope(lineage, {}, SecurityManual.empty())

    print(f"\nSANITY (FTD scope symbols): {sorted(scope.symbols)} from lineage symbols {[symbol for symbol, _, _ in rows]}")
    assert scope.symbols == frozenset({"ABBV"}), "a Notes debt symbol and a noise symbol are not FTD scope keys"


def test_a_key_changed_after_the_previous_period_is_reread_once_when_a_new_period_is_stored(sqlite_store, tmp_path, caplog, monkeypatch):
    """202601a/b are stored; 202602a is new. XYZ joined ABC's lineage on 2026-02-03 (after 202601b ended, long before the
    run) with no stored line and is not in 202602a, so the run that stores 202602a re-reads 202601a/b for XYZ alone. LATE
    changed before 202601b ended: never re-read."""
    context = _context(sqlite_store, tmp_path, ("ABC",))
    _lineage(
        sqlite_store,
        [
            ("ABC", "0000000777", "ABC", pd.Timestamp("2020-01-01")),
            ("ABC", "0000000778", "XYZ", pd.Timestamp("2026-02-03")),
            ("ABC", "0000000779", "LATE", pd.Timestamp("2026-01-25")),
        ],
    )
    write_ftd_zips(
        tmp_path / "ftd",
        {
            "202601a": _HEADER
            + "20260105|000111AAA|ABC|100|ABC CORP|10.0\n20260105|555555XXX|XYZ|7|XYZ CORP|2.0\n20260105|777777LLL|LATE|5|LATE CO|1.0\n",
            "202601b": _HEADER + "20260120|000111AAA|ABC|50|ABC CORP|10.0\n20260120|555555XXX|XYZ|8|XYZ CORP|2.0\n",
            "202602a": _HEADER + "20260205|000111AAA|ABC|60|ABC CORP|10.0\n",
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
    first, reads[:] = list(reads), []
    ftd.fetch_fails_to_deliver(context, as_of=RUN_DATE)

    stored = _stored(sqlite_store)
    print(f"\nSANITY (growth on a new-period run): run 1 read {first}, run 2 read {reads} (expected none);")
    print(f"  stored {stored[['period', 'cusip', 'fails_quantity']].values.tolist()}")
    assert first == ["202602a", "202601a", "202601b"], "the new period, then each stored period once for XYZ"
    assert reads == [], "a run storing no new period does no growth pass"
    assert stored[["period", "cusip", "fails_quantity"]].values.tolist() == [
        ["202601a", "000111AAA", 100.0],
        ["202601a", "555555XXX", 7.0],
        ["202601b", "000111AAA", 50.0],
        ["202601b", "555555XXX", 8.0],
        ["202602a", "000111AAA", 60.0],
    ], "XYZ's history is upserted, LATE (changed before 202601b ended) is not re-read"
    assert any("re-read" in record.getMessage() for record in caplog.records), "one log line names the scope-growth re-read"


def test_a_new_key_no_file_lists_is_reread_once_not_on_every_run(sqlite_store, tmp_path, monkeypatch):
    """NEW joins ABC's lineage today and appears in no FTD file: the cold run reads both cached periods with the full scope
    and no separate growth pass; a second run the same day stores no new period, so it reads nothing."""
    context = _context(sqlite_store, tmp_path, ("ABC",))
    _lineage(sqlite_store, [("ABC", "0000000777", "ABC", pd.Timestamp("2020-01-01")), ("ABC", "0000000778", "NEW", RUN_DATE)])
    write_ftd_zips(
        tmp_path / "ftd",
        {"202601a": _HEADER + "20260105|000111AAA|ABC|100|ABC CORP|10.0\n", "202601b": _HEADER + "20260120|000111AAA|ABC|50|ABC CORP|10.0\n"},
    )
    reads: list[str] = []
    reader = ftd.read_zip_text
    monkeypatch.setattr(ftd, "read_zip_text", lambda path, log=None: reads.append(path.stem.removeprefix("cnsfails")) or reader(path, log=log))
    ftd.fetch_fails_to_deliver(context, as_of=RUN_DATE)
    first, reads[:] = list(reads), []

    ftd.fetch_fails_to_deliver(context, as_of=RUN_DATE)

    print(f"\nSANITY (rowless growth key): run 1 read {sorted(first)}, run 2 read {sorted(reads)} (expected none)")
    assert first == ["202601a", "202601b"], "the cold run reads each period once, with no separate growth pass"
    assert reads == [], "a never-matching growth key already re-read is not re-read on the next run"
