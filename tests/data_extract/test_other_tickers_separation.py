"""
other_tickers are MARKET/MACRO price series, not part of the equity universe.

Guarantees:
  (1) the equity universe handed to every sub-step (StepExtractAllData._resolve_tickers)
      excludes other_tickers entirely — so no fundamentals/behavioral/feature work runs
      on SPY/^VIX/oil/gold/FX;
  (2) prices and dividends are DECOUPLED: `fetch_price_history` writes clean OHLCV and
      never touches the `dividends` table, so running it over other_tickers (the market
      /macro path) cannot invent dividend rows for SPY/FX. `fetch_dividends` is the only
      writer of ex-dates, and is handed the equity universe only.
Peers/cube exclusion of these names is covered by tests/utils/test_universe.py (the
universe = sp500_tickers, which never contains other_tickers).
"""
from __future__ import annotations

import logging
from types import SimpleNamespace

import pandas as pd
from omegaconf import OmegaConf

from src.data_extract.step_extract_all_data import StepExtractAllData
from src.data_extract.utils.prices import fetch_prices as fp
from src.data_extract.utils.prices import fetch_dividends as fd


def test_equity_universe_excludes_other_tickers(sqlite_store):
    sqlite_store.replace("sp500_tickers", pd.DataFrame({"ticker": ["AAPL", "MSFT", "XOM"]}))
    fake = SimpleNamespace(
        _context=SimpleNamespace(store=sqlite_store),
        _config=OmegaConf.create({"data_extract": {
            "refresh_universe": False,
            "other_tickers": ["SPY", "^VIX", "CL=F", "GC=F", "USDEUR=X"]}}),
        _log=logging.getLogger("test"),
    )
    # bind the method to a fake self (avoids constructing the sub-steps / hitting the DB)
    tickers = StepExtractAllData._resolve_tickers(fake)

    assert tickers == ["AAPL", "MSFT", "XOM"]                    # universe only, sorted
    for o in ["SPY", "^VIX", "CL=F", "GC=F", "USDEUR=X"]:
        assert o not in tickers, f"{o} leaked into the equity universe"
    print("\n=== SANITY CHECK: equity universe excludes market/macro tickers ===")
    print(f"  _resolve_tickers -> {tickers}; none of SPY/^VIX/CL=F/GC=F/USDEUR=X present "
          "(no sub-step builds features on them). Validated.")


def _yf_frame(ticker: str, dividend: float) -> pd.DataFrame:
    """A normalized yfinance `actions=True` response: OHLCV + the action columns."""
    return pd.DataFrame({
        "date": pd.to_datetime(["2024-03-01"]), "ticker": [ticker],
        "open": [10.0], "high": [11.0], "low": [9.0], "close": [10.5],
        "volume": [1_000_000.0],
        "dividends": [dividend], "stock splits": [0.0], "capital gains": [0.0],
    })


def test_price_fetch_writes_ohlcv_only_and_never_the_dividends_table(sqlite_store, monkeypatch):
    """The market/macro path runs fetch_price_history over other_tickers. Because
    prices and dividends are decoupled, that can only ever write clean OHLCV — a
    dividend column coming back for SPY/FX must not create `dividends` rows."""
    monkeypatch.setattr(fp, "download_ohlcv",
                        lambda *a, **k: _yf_frame("SPY", dividend=1.23))
    monkeypatch.setattr(fp, "record_run", lambda *a, **k: None)
    ctx = SimpleNamespace(store=sqlite_store)

    out = fp.fetch_price_history(ctx, tickers=["SPY"], years_history=15)

    saved = sqlite_store.load("prices")
    assert set(saved.columns) == {"date", "ticker", "open", "high", "low", "close", "volume"}
    assert not sqlite_store.exists("dividends"), "price fetch wrote the dividends table"
    assert list(out["ticker"]) == ["SPY"]

    print("\n=== SANITY CHECK: prices/dividends decoupled ===")
    print(f"  fetch_price_history(['SPY']) -> prices cols {sorted(saved.columns)}; "
          "dividends table NOT created even though the download carried a 1.23 ex-date. Validated.")


def test_dividend_fetch_is_the_only_ex_date_writer(sqlite_store, monkeypatch):
    monkeypatch.setattr(fd, "download_ohlcv",
                        lambda *a, **k: _yf_frame("AAPL", dividend=0.24))
    monkeypatch.setattr(fd, "record_run", lambda *a, **k: None)
    ctx = SimpleNamespace(store=sqlite_store)

    out = fd.fetch_dividends(ctx, tickers=["AAPL"], years_history=15)

    saved = sqlite_store.load("dividends")
    assert list(saved.columns) == ["date", "ticker", "dividend"]
    assert saved["dividend"].tolist() == [0.24]
    assert len(out) == 1

    # no tickers -> no download, no write (the market/macro set is never passed here)
    assert fd.fetch_dividends(ctx, tickers=[], years_history=15).empty

    print("\n=== SANITY CHECK: dividend fetcher ===")
    print(f"  fetch_dividends(['AAPL']) -> {len(saved)} ex-date row {saved['dividend'].tolist()} "
          "in `dividends`, schema [date,ticker,dividend]; empty ticker list -> no-op. Validated.")


if __name__ == "__main__":
    test_equity_universe_excludes_other_tickers()
    import pytest
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
