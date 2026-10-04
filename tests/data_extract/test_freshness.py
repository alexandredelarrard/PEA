"""Schema-driven extraction freshness gate and recent-tail replay."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any, cast

import pandas as pd

from src.data_extract import cli as extraction_cli
from src.data_extract.utils.prices import fetch_prices as prices_fetcher
from src.data_store.schema import Resume, Tables


def _context(store) -> Any:
    return SimpleNamespace(
        store=store,
        log=logging.getLogger("test.extraction_freshness"),
        config=SimpleNamespace(data_extract=SimpleNamespace(redundant_ticks=[])),
    )


def test_gate_reads_table_cadence_and_date_column_from_schema(sqlite_store, monkeypatch):
    sqlite_store.replace(
        Tables.prices,
        pd.DataFrame({"ticker": ["AAA"], "date": pd.to_datetime(["2026-09-25"]), "close_split": [10.0]}),
    )
    monkeypatch.setattr(extraction_cli, "freshness_tables", lambda: (Tables.prices,))

    green = extraction_cli._extraction_status_report(cast(Any, _context(sqlite_store)), as_of=pd.Timestamp("2026-09-27"))
    status = cast(dict[str, Any], cast(dict[str, object], green["tables"])[Tables.prices.name])
    assert green["ok"]
    assert status["date_column"] == Tables.prices.freshness_col
    assert status["cadence"] == Tables.prices.freshness

    sqlite_store.replace(
        Tables.prices,
        pd.DataFrame({"ticker": ["AAA"], "date": pd.to_datetime(["2026-09-20"]), "close_split": [10.0]}),
    )
    red = extraction_cli._extraction_status_report(cast(Any, _context(sqlite_store)), as_of=pd.Timestamp("2026-09-27"))
    assert not red["ok"] and red["behind"] == [Tables.prices.name]

    print("\n=== SANITY CHECK: schema-driven extraction gate ===")
    print("  prices at 2026-09-25 -> GREEN; prices at 2026-09-20 -> RED")
    print("  OK: the gate reads its table, date column and cadence directly from schema.py")


def test_notes_tables_declare_monthly_archive_availability():
    for table in (Tables.notes_num, Tables.notes_text):
        assert table.date_col == "ddate"
        assert table.freshness == "monthly"
        assert table.freshness_col == "available_at"
        assert "available_at" in table.date_type_cols

    print("\n=== SANITY CHECK: financial-notes registry contract ===")
    print("  notes_num and notes_text retain ddate as data time and use date-typed available_at for monthly freshness. Validated.")


def test_dividends_replay_the_same_recent_tail_as_prices(sqlite_store, monkeypatch):
    as_of = pd.Timestamp("2026-09-30")
    last = pd.Timestamp("2026-09-28")
    sqlite_store.replace(Tables.prices, pd.DataFrame({"ticker": ["AAA"], "date": [last], "close_split": [1.0]}))
    sqlite_store.replace(Tables.dividends, pd.DataFrame({"ticker": ["AAA"], "date": [last], "dividends": [0.0]}))
    captured: list[pd.Timestamp] = []

    def _download(tickers, since, until, *args, **kwargs):
        del tickers, until, args, kwargs
        captured.append(pd.Timestamp(since))
        return pd.DataFrame()

    monkeypatch.setattr(prices_fetcher, "download_ohlcv", _download)
    prices_fetcher.fetch_prices_and_actions(_context(sqlite_store), ["AAA"], years_history=15, as_of=as_of, pause=0.0)

    overlap = pd.Timedelta(days=cast(Resume, Tables.dividends.resume).overlap_days)
    assert captured == [last - overlap]

    print("\n=== SANITY CHECK: dividend repair window ===")
    print(f"  prices and dividends at {last.date()} -> one download from {(last - overlap).date()} (the {overlap.days}-day overlap)")
    print("  OK: a recent interior dividend-grid hole can self-heal on the next run")
