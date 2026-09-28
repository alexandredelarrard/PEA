"""Schema-driven extraction freshness gate and recent-tail replay."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any, cast

import pandas as pd

from src.data_extract import cli as extraction_cli
from src.data_extract.utils.prices import fetch_dividends as dividends_fetcher
from src.data_extract.utils.prices.fetch_prices import PRICE_REFRESH_TRADING_DAYS
from src.data_store.schema import Tables, freshness_tables


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


def test_retired_attention_tables_do_not_declare_freshness():
    checked = set(freshness_tables())
    assert Tables.wiki_pageviews not in checked
    assert Tables.google_trends not in checked

    print("\n=== SANITY CHECK: retired extraction tables ===")
    print("  Wikipedia pageviews and Google Trends declare no freshness in schema.py")
    print("  OK: retired attention tables cannot block aggregation")


def test_dividends_replay_the_same_recent_tail_as_prices(sqlite_store, monkeypatch):
    today = pd.Timestamp.today().normalize()
    sqlite_store.replace(
        Tables.dividends,
        pd.DataFrame({"ticker": ["AAA"], "date": [today], "dividends": [0.0]}),
    )
    captured: list[pd.Timestamp] = []

    def _download(tickers, since, until, *args, **kwargs):
        del tickers, until, args, kwargs
        captured.append(pd.Timestamp(since))
        return pd.DataFrame()

    monkeypatch.setattr(dividends_fetcher, "download_ohlcv", _download)
    monkeypatch.setattr(dividends_fetcher, "record_run", lambda *args, **kwargs: None)
    dividends_fetcher.fetch_dividends(_context(sqlite_store), ["AAA"], years_history=15, pause=0.0)

    expected = today - pd.tseries.offsets.BDay(PRICE_REFRESH_TRADING_DAYS)
    assert captured == [expected]

    print("\n=== SANITY CHECK: dividend repair window ===")
    print(f"  current frontier still replays from {expected.date()} ({PRICE_REFRESH_TRADING_DAYS} business days)")
    print("  OK: a recent interior dividend-grid hole can self-heal on the next run")
