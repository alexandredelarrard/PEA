"""
other_tickers are MARKET/MACRO price series, not part of the equity universe.

Guarantees:
  (1) the equity universe handed to every sub-step (StepExtractAllData._resolve_tickers)
      excludes other_tickers entirely — so no fundamentals/behavioral/feature work runs
      on SPY/^VIX/oil/gold/FX;
  (2) fetch_market_prices fetches ONLY other_tickers, OHLCV-only (no dividends), into
      `prices` so the cube can still read the market-beta benchmark + factor series.
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


class _FakeStore:
    def __init__(self, universe: list[str]):
        self._u = universe

    def row_count(self, name):
        return len(self._u) if name == "sp500_tickers" else 0

    def load(self, name, columns=None, limit=None):
        if name == "sp500_tickers":
            df = pd.DataFrame({"ticker": self._u})
            return df[columns] if columns else df
        return pd.DataFrame(columns=columns or [])


def test_equity_universe_excludes_other_tickers():
    fake = SimpleNamespace(
        _context=SimpleNamespace(store=_FakeStore(["AAPL", "MSFT", "XOM"])),
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


def test_fetch_market_prices_ohlcv_only_for_other_tickers(monkeypatch):
    captured: dict = {}

    def _fake_fph(context, tickers, chunk_size=50, pause=2.0, download_dividends=True):
        captured["tickers"] = list(tickers)
        captured["download_dividends"] = download_dividends
        return "prices-frame"

    monkeypatch.setattr(fp, "fetch_price_history", _fake_fph)

    ctx = SimpleNamespace(config=OmegaConf.create(
        {"data_extract": {"other_tickers": ["SPY", "CL=F", "USDEUR=X"]}}))
    out = fp.fetch_market_prices(ctx)

    assert captured["tickers"] == ["SPY", "CL=F", "USDEUR=X"]     # ONLY the market/macro set
    assert captured["download_dividends"] is False               # OHLCV only, no dividends
    assert out == "prices-frame"

    # no market tickers configured -> nothing fetched (no crash)
    captured.clear()
    empty_ctx = SimpleNamespace(config=OmegaConf.create({"data_extract": {"other_tickers": []}}))
    assert fp.fetch_market_prices(empty_ctx) is None and not captured
    print("\n=== SANITY CHECK: market/macro price pull ===")
    print("  fetch_market_prices -> fetch_price_history(['SPY','CL=F','USDEUR=X'], "
          "download_dividends=False); empty list -> no-op. Validated.")


if __name__ == "__main__":
    test_equity_universe_excludes_other_tickers()
    import pytest
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
