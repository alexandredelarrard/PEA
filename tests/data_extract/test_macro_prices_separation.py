"""
`prices` is the EQUITY universe and nothing else; the market/macro series live in
`prices_macro`.

This inverts what this file used to assert. The market/commodity/FX series (SPY, ^VIX, oil,
gold, energy, FX) used to be fetched by `fetch_price_history` over `other_tickers` and stored
as extra OHLCV rows INSIDE `prices` -- and the old test pinned exactly that. Every consumer of
`prices` then needed a firewall against them: a separate `cube_part_market` part table, three
`drop(columns=[market])` guards, a `^`-prefix filter, a zero-volume trim exemption. Enforcing
the separation at the SOURCE is what let all of those go, so it is worth a test.

Guarantees:
  (1) the equity universe handed to every sub-step (`StepExtractAllData._resolve_tickers`) is
      `sp500_tickers` and nothing else;
  (2) `fetch_macro` writes ONLY `prices_macro`, with the (date, ticker, close) schema, and
      never touches `prices`;
  (3) prices and dividends stay DECOUPLED: `fetch_price_history` writes clean OHLCV and never
      the `dividends` table.
"""
from __future__ import annotations

import logging
from types import SimpleNamespace

import pandas as pd
from omegaconf import OmegaConf

from src.constants.constants import MACRO_PRICE_SERIES
from src.data_extract.step_extract_all_data import StepExtractAllData
from src.data_extract.utils.prices import fetch_macro as fm
from src.data_extract.utils.prices import fetch_prices as fp
from src.data_extract.utils.prices import fetch_dividends as fd
from src.data_store.schema import Tables


def test_equity_universe_is_sp500_tickers_only(sqlite_store):
    sqlite_store.replace("sp500_tickers", pd.DataFrame({"ticker": ["AAPL", "MSFT", "XOM"]}))
    fake = SimpleNamespace(
        _context=SimpleNamespace(store=sqlite_store),
        _config=OmegaConf.create({"data_extract": {"refresh_universe": False}}),
        _log=logging.getLogger("test"),
    )
    # bind the method to a fake self (avoids constructing the sub-steps / hitting the DB)
    tickers = StepExtractAllData._resolve_tickers(fake)

    assert tickers == ["AAPL", "MSFT", "XOM"]                    # universe only, sorted
    for symbol in MACRO_PRICE_SERIES:
        assert symbol not in tickers, f"{symbol} leaked into the equity universe"

    print("\n=== SANITY CHECK: equity universe is sp500_tickers only ===")
    print(f"  _resolve_tickers -> {tickers}")
    print(f"  none of {list(MACRO_PRICE_SERIES)} present -- and there is no longer an "
          f"`other_tickers` config key that could put them there. Validated.")


def test_fetch_macro_writes_only_prices_macro(sqlite_store, monkeypatch):
    """The core separation: the macro fetcher must not create or touch `prices`."""
    dates = pd.bdate_range("2024-01-01", periods=4)

    def _fake_price_leg(context, since, until):
        return pd.DataFrame({"equity_tr": [400.0, 401.0, 402.0, 403.0],
                             "vix": [14.0, 15.0, 13.0, 16.0]}, index=dates)

    def _fake_fred_leg(since):
        return pd.DataFrame({"yield_10y": [4.0, 4.1, 4.2, 4.3],
                             "cash_rate": [5.0, 5.0, 5.0, 4.9],
                             "yield_2y": [3.5, 3.6, 3.7, 3.8]}, index=dates)

    monkeypatch.setattr(fm, "_fetch_price_leg", _fake_price_leg)
    monkeypatch.setattr(fm, "_fetch_fred_leg", _fake_fred_leg)
    monkeypatch.setattr(fm, "record_run", lambda *a, **k: None)
    monkeypatch.setenv("FRED_API_KEY", "test-key")

    ctx = SimpleNamespace(store=sqlite_store,
                          log=SimpleNamespace(info=lambda *a, **k: None,
                                              warning=lambda *a, **k: None))
    fm.fetch_macro(ctx, years_history=31)

    saved = sqlite_store.load(Tables.prices_macro)
    assert set(saved.columns) == {"date", "ticker", "close"}
    assert not sqlite_store.exists(Tables.prices), "the macro fetcher wrote `prices`"
    # no OHLV/volume leaked through -- close only ("trim the volume")
    assert not {"open", "high", "low", "volume"} & set(saved.columns)
    series = sorted(saved["ticker"].unique())
    assert "equity_tr" in series and "vix" in series
    assert "yield_curve_10y2y" in series and "bond_10y_tr" in series   # derived came through

    print("\n=== SANITY CHECK: fetch_macro writes only prices_macro ===")
    print(f"  {len(saved)} rows, schema {sorted(saved.columns)}, series {series}")
    print(f"  `prices` table exists: {sqlite_store.exists(Tables.prices)} (must be False). "
          f"Validated.")


def _yf_frame(ticker: str, dividend: float | None = None) -> pd.DataFrame:
    """A normalized yfinance response. `actions=False` (the price path) returns OHLCV only;
    pass `dividend` to get the `actions=True` shape the dividend path asks for."""
    df = pd.DataFrame({
        "date": pd.to_datetime(["2024-03-01"]), "ticker": [ticker],
        "open": [10.0], "high": [11.0], "low": [9.0], "close": [10.5],
        "volume": [1_000_000.0],
    })
    if dividend is not None:
        df["dividends"] = [dividend]
        df["stock splits"] = [0.0]
        df["capital gains"] = [0.0]
    return df


def test_price_fetch_writes_clean_ohlcv_and_never_the_dividends_table(sqlite_store, monkeypatch):
    """`fetch_price_history` writes clean OHLCV and never the `dividends` table."""
    monkeypatch.setattr(fp, "download_ohlcv", lambda *a, **k: _yf_frame("AAPL"))
    monkeypatch.setattr(fp, "record_run", lambda *a, **k: None)
    ctx = SimpleNamespace(store=sqlite_store)

    fp.fetch_price_history(ctx, tickers=["AAPL"], years_history=15)

    # assert on what LANDED, not on the return value: these fetchers write to the store and
    # their return is incidental (fetch_macro returns None outright)
    saved = sqlite_store.load(Tables.prices)
    assert set(saved.columns) == {"date", "ticker", "open", "high", "low", "close", "volume"}
    assert not sqlite_store.exists(Tables.dividends), "price fetch wrote the dividends table"
    assert list(saved["ticker"]) == ["AAPL"]

    print("\n=== SANITY CHECK: prices holds clean OHLCV ===")
    print(f"  fetch_price_history(['AAPL']) -> prices cols {sorted(saved.columns)}; "
          f"dividends table created: {sqlite_store.exists(Tables.dividends)}. Validated.")


def test_actions_are_requested_only_on_the_dividend_path():
    """WHY `prices` stays clean OHLCV: `download_ohlcv` only asks yfinance for the action
    columns when the CALLER's `desc` says it wants dividends. The price path therefore never
    receives a `dividends` column to leak, which is what replaced the old explicit
    drop(_ACTION_COLS) -- so this is the invariant that now carries that guarantee."""
    seen: list[bool] = []

    def _spy_chunk(chunk, start, end, pause, actions):
        seen.append(actions)
        return []

    import src.data_extract.utils.prices.fetch_prices as mod
    original = mod._download_price_chunk
    try:
        mod._download_price_chunk = _spy_chunk
        since, until = pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-05")
        mod.download_ohlcv(["AAPL"], since, until, pause=0.0)                    # default desc
        mod.download_ohlcv(["AAPL"], since, until, pause=0.0, desc="Downloading dividends")
        mod.download_ohlcv(list(MACRO_PRICE_SERIES), since, until, pause=0.0,
                           desc="Downloading macro/market prices")
    finally:
        mod._download_price_chunk = original

    assert seen == [False, True, False], f"actions flags {seen}"

    print("\n=== SANITY CHECK: actions flag per caller ===")
    print(f"  desc 'Downloading prices' -> actions={seen[0]}; "
          f"'Downloading dividends' -> actions={seen[1]}; "
          f"'Downloading macro/market prices' -> actions={seen[2]}.")
    print("  Only the dividend fetcher pulls ex-dates, so no other path can leak them. "
          "Validated.")


def test_dividend_fetch_is_the_only_ex_date_writer(sqlite_store, monkeypatch):
    """Carried over unchanged from the old file. Asserts the `dividends` table schema that
    `sql/schema.sql` / `Tables.dividends` declare: [date, ticker, dividends]."""
    monkeypatch.setattr(fd, "download_ohlcv",
                        lambda *a, **k: _yf_frame("AAPL", dividend=0.24))
    monkeypatch.setattr(fd, "record_run", lambda *a, **k: None)
    ctx = SimpleNamespace(store=sqlite_store)

    fd.fetch_dividends(ctx, tickers=["AAPL"], years_history=15)

    saved = sqlite_store.load(Tables.dividends)
    assert list(saved.columns) == ["date", "ticker", "dividends"]
    assert len(saved) == 1

    print("\n=== SANITY CHECK: dividend fetcher ===")
    print(f"  fetch_dividends(['AAPL']) -> {len(saved)} row in `dividends`, "
          f"schema {list(saved.columns)}. Validated.")


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
