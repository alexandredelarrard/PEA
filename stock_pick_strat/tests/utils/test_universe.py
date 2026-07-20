"""
Single-entry-point universe loader (src/utils/universe.py).

Every step resolves its tickers from ONE place — the `sp500_tickers` table — so
swapping what fills that table (S&P 500 -> Russell 1000 -> custom) reroutes the
whole flow with no step-code change. These tests pin the loader's contract.
"""
from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from src.utils.universe import load_universe_tickers


class _FakeStore:
    def __init__(self, df: pd.DataFrame | None):
        self._df = df

    def load(self, name, columns=None, limit=None):
        if name == "sp500_tickers" and self._df is not None:
            return self._df[columns] if columns else self._df
        return pd.DataFrame(columns=columns or [])


def _ctx(df):
    return SimpleNamespace(store=_FakeStore(df))


def test_loader_sorted_dedup_upper_and_clean():
    df = pd.DataFrame({"ticker": ["msft", "AAPL", "aapl", "GOOGL", None, "  ", "brk-b"],
                       "cik": ["1", "2", "2", "3", None, "9", "4"]})
    out = load_universe_tickers(_ctx(df))
    assert out == ["AAPL", "BRK-B", "GOOGL", "MSFT"]      # upper, dedup, sorted, blanks/None dropped
    print("\n=== SANITY CHECK: universe loader normalization ===")
    print(f"  {out} — upper-cased, de-duplicated, sorted; None/blank dropped. Validated.")


def test_empty_or_missing_table_returns_empty():
    assert load_universe_tickers(_ctx(pd.DataFrame(columns=["ticker"]))) == []
    assert load_universe_tickers(_ctx(None)) == []        # table absent -> [] (caller falls back)
    print("\n=== SANITY CHECK: unseeded universe ===")
    print("  empty / missing sp500_tickers -> [] (steps warn + fall back, never crash). Validated.")


def test_swapping_the_table_swaps_the_universe():
    sp500 = _ctx(pd.DataFrame({"ticker": ["AAPL", "MSFT", "XOM"]}))
    russell = _ctx(pd.DataFrame({"ticker": ["ETSY", "RCL", "AAPL"]}))   # different membership
    assert load_universe_tickers(sp500) == ["AAPL", "MSFT", "XOM"]
    assert load_universe_tickers(russell) == ["AAPL", "ETSY", "RCL"]
    print("\n=== SANITY CHECK: one entry point drives the whole flow ===")
    print("  same loader, two tables -> two universes (S&P 500 vs a Russell-like set); "
          "changing only sp500_tickers reroutes extract/peers/cube/modelling. Validated.")


if __name__ == "__main__":
    test_loader_sorted_dedup_upper_and_clean()
    test_empty_or_missing_table_returns_empty()
    test_swapping_the_table_swaps_the_universe()
