"""Price and dividend work share the configured correction tail."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from src.data_extract.utils.prices import fetch_prices as prices_fetcher
from src.data_store.schema import Tables
from src.utils.window import Window, WindowRun
from tests.data_extract.edgar_fixtures import fake_context
from tests.support.store_identity import store_identity


def test_dividends_replay_the_same_recent_tail_as_prices(tmp_path: Any, sqlite_store: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    last = pd.Timestamp("2026-09-28")
    sqlite_store.replace(Tables.prices, pd.DataFrame({"ticker": ["AAA"], "date": [last], "close_split": [1.0]}))
    sqlite_store.replace(Tables.dividends, pd.DataFrame({"ticker": ["AAA"], "date": [last], "dividends": [0.0]}))
    context = fake_context(tmp_path, sqlite_store, ["AAA"])
    context._identity_resolver = store_identity(sqlite_store)
    context.run = WindowRun("incremental", "dividend overlap", pd.Timestamp("2026-09-30"))
    captured: list[pd.Timestamp] = []

    def download(tickers: Any, window: Window, *args: Any, **kwargs: Any) -> pd.DataFrame:
        assert kwargs["actions"] is True
        captured.append(window.start)
        return pd.DataFrame()

    monkeypatch.setattr(prices_fetcher, "download_ohlcv", download)
    monkeypatch.setattr(prices_fetcher, "anchor_restated_keys", lambda *args, **kwargs: [])
    prices_fetcher.fetch_prices(context, prices_fetcher.YahooSession(pause=0.0), ["AAA"])
    overlap = pd.Timedelta(days=context.config.data.tables.prices_dividends.overlap.days)
    assert captured == [last - overlap]
    print(f"SANITY: prices/dividends share one download beginning at their saved reference minus {overlap.days} calendar days.")
