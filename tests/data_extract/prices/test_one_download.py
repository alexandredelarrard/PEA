"""Prices, dividends and splits from ONE yfinance `actions=True` download per window group (AC-009).

The yfinance response is a recorded one (AAPL, MO, NVDA, 2024-05-01..2024-06-28: an AAPL and an MO
dividend, the NVDA 10:1 split and its first post-split dividend), served by a fake `yf.download` that
honours the tickers, the date range and the `actions` flag. No network call is made.
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pandas as pd
import pytest
from omegaconf import OmegaConf
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from src.data_extract.transformers import step_extract_prices as step_module
from src.data_extract.transformers.step_extract_prices import StepExtractPrices
from src.data_extract.utils.prices import fetch_dividends as fd
from src.data_extract.utils.prices import fetch_prices as fp
from src.data_extract.utils.prices import fetch_splits as fs
from src.data_store.schema import Tables
from src.data_store.store import DataStore

_FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "yfinance" / "aapl_mo_nvda_2024-05_06_actions.csv"
_TICKERS = ["AAPL", "MO", "NVDA"]
_AS_OF = pd.Timestamp("2024-06-29")  # the last completed session is Friday 2024-06-28
_ACTIONS = ["Dividends", "Stock Splits"]


def _recorded() -> pd.DataFrame:
    """The recorded response in yfinance's own shape: Date index, (ticker, field) columns."""
    df_long = pd.read_csv(_FIXTURE, parse_dates=["Date"])
    wide = cast(pd.DataFrame, df_long.set_index(["Date", "Ticker"]).unstack("Ticker"))
    wide.columns = cast(pd.MultiIndex, wide.columns).swaplevel(0, 1)
    wide = wide.sort_index(axis=1)
    wide.index.name = "Date"
    return wide


class FakeYahoo:
    """`yf.download` over the recorded response; counts calls and keeps their arguments."""

    def __init__(self, response: pd.DataFrame) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def __call__(self, tickers: list[str], start: str, end: str, **kwargs: Any) -> pd.DataFrame:
        self.calls.append({"tickers": list(tickers), "start": start, "end": end} | kwargs)
        rows = (self.response.index >= pd.Timestamp(start)) & (self.response.index < pd.Timestamp(end))
        fields = [f for f in self.response.columns.get_level_values(1).unique() if kwargs.get("actions") or f not in _ACTIONS]
        columns = pd.MultiIndex.from_tuples([(t, f) for t in tickers for f in fields if (t, f) in self.response.columns])
        return self.response.loc[rows, columns]


def _context(store: DataStore) -> Any:
    config = OmegaConf.create({"data_extract": {"years_history": 3, "macro_years_history": 1}})
    return SimpleNamespace(store=store, log=logging.getLogger("test.one_download"), config=config, config_dir=".")


def _second_store() -> DataStore:
    return DataStore(create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False}))


def _sorted(store: DataStore, table: Any) -> pd.DataFrame:
    return cast(pd.DataFrame, store.load(table)).sort_values(["ticker", "date"], ignore_index=True)


@pytest.fixture
def yahoo(monkeypatch: pytest.MonkeyPatch) -> FakeYahoo:
    fake = FakeYahoo(_recorded())
    monkeypatch.setattr(fp.yf, "download", fake)
    monkeypatch.setattr(fp.time, "sleep", lambda seconds: None)
    return fake


def test_one_call_per_window(sqlite_store: DataStore, yahoo: FakeYahoo, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(step_module, "fetch_macro", lambda *a, **k: None)
    ctx = _context(sqlite_store)

    StepExtractPrices(context=ctx, config=ctx.config).run(tickers=_TICKERS)

    assert len(yahoo.calls) == 1, f"{len(yahoo.calls)} yfinance downloads for one window group"
    assert yahoo.calls[0]["actions"] is True and yahoo.calls[0]["auto_adjust"] is False
    assert all(sqlite_store.exists(t) for t in (Tables.prices, Tables.dividends, Tables.prices_splits))
    print("\n=== SANITY CHECK: one download ===")
    print(f"  StepExtractPrices on a cold table -> {len(yahoo.calls)} yfinance call (actions=True, auto_adjust=False) wrote all three tables.")


def test_the_three_tables_equal_the_three_call_path(sqlite_store: DataStore, yahoo: FakeYahoo) -> None:
    fp.fetch_prices_and_actions(_context(sqlite_store), _TICKERS, years_history=1, as_of=_AS_OF, pause=0.0)
    (call,) = yahoo.calls
    since, until = pd.Timestamp(call["start"]), pd.Timestamp(call["end"]) - pd.Timedelta(days=1)

    reference = _second_store()
    df_bars = fp.download_ohlcv(_TICKERS, since, until, pause=0.0, auto_adjust=False, actions=False)
    df_actions = fp.download_ohlcv(_TICKERS, since, until, pause=0.0, auto_adjust=False, actions=True)
    reference.save(Tables.prices, fp.trim_prelisting_bars(df_bars))
    reference.save(Tables.dividends, fd._extract_dividends(df_actions))
    reference.save(Tables.prices_splits, fs._extract_splits(df_actions))

    for table in (Tables.prices, Tables.dividends, Tables.prices_splits):
        pd.testing.assert_frame_equal(_sorted(sqlite_store, table), _sorted(reference, table), check_exact=True)
    df_prices = _sorted(sqlite_store, Tables.prices)
    splits = _sorted(sqlite_store, Tables.prices_splits)
    assert {"close_split", "close_total"} <= set(df_prices.columns) and not {"dividends", "stock splits"} & set(df_prices.columns)
    assert splits[["ticker", "ratio"]].values.tolist() == [["NVDA", 10.0]]
    print("\n=== SANITY CHECK: AC-009 ===")
    print(f"  one call over {since.date()}..{until.date()}: prices {len(df_prices)} rows, dividends {len(_sorted(sqlite_store, Tables.dividends))},")
    print("  splits [NVDA x10] -- each frame equal, cell for cell (close_split/close_total exact), to the three-call path.")


def test_a_split_after_the_last_bar_re_pulls_that_ticker_only(sqlite_store: DataStore, yahoo: FakeYahoo) -> None:
    recorded = fp.download_ohlcv(_TICKERS, pd.Timestamp("2024-05-01"), pd.Timestamp("2024-06-07"), pause=0.0, auto_adjust=False, actions=False)
    stale = recorded.assign(close_split=lambda d: d["close_split"].where(d["ticker"] != "NVDA", d["close_split"] * 10))
    sqlite_store.save(Tables.prices, stale)
    sqlite_store.save(Tables.dividends, fd._extract_dividends(recorded.assign(dividends=0.0)))
    yahoo.calls.clear()

    fp.fetch_prices_and_actions(_context(sqlite_store), _TICKERS, years_history=1, as_of=pd.Timestamp("2024-06-13"), pause=0.0)

    assert [c["tickers"] for c in yahoo.calls] == [_TICKERS, ["NVDA"]]
    assert pd.Timestamp(yahoo.calls[1]["start"]) == pd.Timestamp("2023-06-13")
    stored = cast(pd.DataFrame, sqlite_store.load(Tables.prices, where={"date": [pd.Timestamp("2024-05-01")]})).set_index("ticker")["close_split"]
    truth = recorded[recorded["date"] == pd.Timestamp("2024-05-01")].set_index("ticker")["close_split"]
    assert stored["NVDA"] == truth["NVDA"], "NVDA's pre-split bars must be restated"
    print("\n=== SANITY CHECK: post-split re-pull ===")
    print("  NVDA split 2024-06-10 after its last bar 2024-06-07 -> one extra full-window call for NVDA only;")
    print(f"  its 2024-05-01 close_split is restated from {truth['NVDA'] * 10:.3f} to {stored['NVDA']:.3f}.")


def test_a_failed_download_saves_nothing_and_logs_the_tickers(sqlite_store: DataStore, monkeypatch: pytest.MonkeyPatch, caplog) -> None:
    monkeypatch.setattr(fp, "download_ohlcv", lambda *a, **k: pd.DataFrame())
    with caplog.at_level("WARNING"):
        fp.fetch_prices_and_actions(_context(sqlite_store), _TICKERS, years_history=1, as_of=_AS_OF, pause=0.0)
    assert not sqlite_store.exists(Tables.prices)
    assert "No bars returned for 3 ticker(s)" in caplog.text
    print("\n=== SANITY CHECK: failed download ===")
    print("  every chunk failed -> nothing written, the 3 tickers are named and re-listed next run (frontier unchanged).")


def test_the_recorded_fixture_is_the_yfinance_shape() -> None:
    wide = _recorded()
    fields = set(wide.columns.get_level_values(1))
    assert {"Open", "High", "Low", "Close", "Adj Close", "Volume", "Dividends", "Stock Splits"} == fields
    assert cast(pd.DatetimeIndex, wide.index).is_monotonic_increasing and len(wide) == 41
    print("\n=== SANITY CHECK: fixture ===")
    print(f"  {len(wide)} sessions x {sorted(set(wide.columns.get_level_values(0)))}, yfinance (ticker, field) columns.")
