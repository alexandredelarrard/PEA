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


def _ctx(store, df: pd.DataFrame | None, redundant: list[str] | None = None):
    """A context on the REAL store. `df=None` leaves `sp500_tickers` uncreated, which is the
    unseeded-DB case the loader must answer with [] rather than a raise."""
    if df is not None:
        store.replace("sp500_tickers", df)
    return SimpleNamespace(
        store=store,
        config=SimpleNamespace(
            data_extract=SimpleNamespace(redundant_ticks=redundant or [])))


def test_loader_sorted_dedup_upper_and_clean(sqlite_store):
    # No None ticker: `ticker` is the PK, so the DB rejects NULL outright (the old in-memory fake
    # accepted it and the test asserted on a row the schema cannot hold). A BLANK ticker is
    # storable, so that is the case worth pinning.
    df = pd.DataFrame({"ticker": ["msft", "AAPL", "aapl", "GOOGL", "  ", "brk-b"],
                       "cik": ["1", "2", "2", "3", "9", "4"]})
    out = load_universe_tickers(_ctx(sqlite_store, df))
    assert out == ["AAPL", "BRK-B", "GOOGL", "MSFT"]      # upper, dedup, sorted, blanks dropped
    print("\n=== SANITY CHECK: universe loader normalization ===")
    print(f"  {out} — upper-cased, de-duplicated, sorted; blank dropped. Validated.")


def test_empty_or_missing_table_returns_empty(sqlite_store):
    # table absent -> [] (the caller falls back and seeds). This is the cold-start path: with a
    # plain `load` the raise-on-empty contract would abort the very run meant to seed the table.
    assert load_universe_tickers(_ctx(sqlite_store, None)) == []
    sqlite_store.save("sp500_tickers", pd.DataFrame({"ticker": ["AAPL"]}))
    sqlite_store.delete("sp500_tickers", {"ticker": "AAPL"})              # created but empty
    assert load_universe_tickers(_ctx(sqlite_store, None)) == []
    print("\n=== SANITY CHECK: unseeded universe ===")
    print("  missing AND created-but-empty sp500_tickers -> [] (steps warn + fall back, never "
          "crash). Validated on the real store.")


def test_swapping_the_table_swaps_the_universe(sqlite_store):
    ctx = _ctx(sqlite_store, pd.DataFrame({"ticker": ["AAPL", "MSFT", "XOM"]}))
    assert load_universe_tickers(ctx) == ["AAPL", "MSFT", "XOM"]
    sqlite_store.replace("sp500_tickers",
                         pd.DataFrame({"ticker": ["ETSY", "RCL", "AAPL"]}))  # different membership
    assert load_universe_tickers(ctx) == ["AAPL", "ETSY", "RCL"]
    print("\n=== SANITY CHECK: one entry point drives the whole flow ===")
    print("  same loader, two tables -> two universes (S&P 500 vs a Russell-like set); "
          "changing only sp500_tickers reroutes extract/peers/cube/modelling. Validated.")


def test_insufficient_history_tickers_excluded(sqlite_store):
    """Recent IPOs / spin-offs with < 4y history (INSUFFICIENT_HISTORY_TICKERS) are dropped from
    the universe even when present in the table, so the cube's multi-year look-backs / backtest
    only see names with enough history."""
    from src.constants.constants import INSUFFICIENT_HISTORY_TICKERS
    excl = sorted(INSUFFICIENT_HISTORY_TICKERS)
    df = pd.DataFrame({"ticker": ["AAPL", "MSFT"] + excl + ["gev"]})   # incl. a lower-case dupe
    out = load_universe_tickers(_ctx(sqlite_store, df))
    assert not (set(out) & INSUFFICIENT_HISTORY_TICKERS), f"excluded names leaked: {set(out) & INSUFFICIENT_HISTORY_TICKERS}"
    assert out == ["AAPL", "MSFT"], out
    print("\n=== SANITY CHECK: insufficient-history exclusion ===")
    print(f"  dropped {len(excl)} <4y names {excl} (incl. lower-case) -> universe = {out}. Validated.")


def test_redundant_share_classes_excluded(sqlite_store):
    """`data_extract.redundant_ticks` lists the second share class of a dual-listed name
    (GOOG next to GOOGL, FOX next to FOXA): counted twice they double that issuer's weight in
    every cross-sectional feature. Matched after upper-casing, like INSUFFICIENT_HISTORY_TICKERS."""
    df = pd.DataFrame({"ticker": ["AAPL", "GOOGL", "GOOG", "goog", "FOX", "MSFT"]})
    out = load_universe_tickers(_ctx(sqlite_store, df, redundant=["GOOG", "FOX"]))
    assert out == ["AAPL", "GOOGL", "MSFT"], out
    print("\n=== SANITY CHECK: redundant share-class exclusion ===")
    print(f"  redundant_ticks=['GOOG','FOX'] dropped GOOG (both cases) and FOX, kept GOOGL "
          f"-> {out}. Validated.")


# no __main__ block: every test now takes the `sqlite_store` fixture, so run them with pytest.
