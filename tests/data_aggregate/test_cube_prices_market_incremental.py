"""
`StepCubePrices._persist` (src/data_aggregate/transformers/step_cube_prices.py).

Bug: on an incremental run (`full=False`), `cube_part_market` was always persisted via a
plain `replace()` using `market_long` -- which only covers the warm-up-padded trailing
window that was READ for that run, not the part's full stored history. So every
incremental build silently truncated `cube_part_market` down to ~262 trading days instead
of appending the new tail, unlike `cube_part_prices` (which already went through
`write_part` and appended correctly).

Fix: `cube_part_market` now goes through the same `write_part` helper, on the SAME
`window` as `cube_part_prices` (both parts are built from the one trimmed read), so a full
run replaces and an incremental run appends only the tail after `window.last`.

This test drives `_persist` directly against a fake `PartStore` that reproduces the real
`replace` / `append_tail` semantics, first as a FULL build, then as an INCREMENTAL build,
and checks that `cube_part_market`'s pre-existing history survives the incremental step.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.data_store.schema import Tables
from src.data_aggregate.transformers.step_cube_prices import StepCubePrices
from src.data_aggregate.utils.common.incremental import PartWindow



def _wide(dates: pd.DatetimeIndex, tickers: list[str], seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.01, (len(dates), len(tickers))),
                                               axis=0)),
                        index=dates, columns=pd.Index(tickers, name="ticker"))


def test_incremental_market_run_appends_instead_of_truncating(sqlite_store):
    tickers = ["AAA", "BBB"]
    full_dates = pd.bdate_range("2024-01-01", periods=50, name="date")
    close_full = _wide(full_dates, tickers, seed=1)
    mkt_close_full = _wide(full_dates, ["SPY"], seed=2)
    mkt_ret_full = mkt_close_full.pct_change().fillna(0.0)

    step = object.__new__(StepCubePrices)
    step._store = sqlite_store
    step._log = logging.getLogger("test")

    # --- FULL build: both parts fully populated ---
    step._persist({"close": close_full}, (mkt_close_full, mkt_ret_full),
                  PartWindow(last=None, since=None))

    prices_after_full = sqlite_store.load(Tables.cube_part_prices)
    market_after_full = sqlite_store.load(Tables.cube_part_market)
    assert len(market_after_full) == 50
    assert market_after_full["date"].min() == full_dates[0]
    assert market_after_full["date"].max() == full_dates[-1]

    # --- INCREMENTAL build: re-reads a trailing warm-up window (last 20 stored days +
    # 10 new days), and must APPEND the 10 new days rather than replacing history ---
    last_stored = full_dates[-1]
    warmup_dates = full_dates[-20:]
    new_dates = pd.bdate_range(last_stored + pd.Timedelta(days=1), periods=10)
    window_dates = warmup_dates.append(new_dates)

    close_window = _wide(window_dates, tickers, seed=3)
    mkt_close_window = _wide(window_dates, ["SPY"], seed=4)
    mkt_ret_window = mkt_close_window.pct_change().fillna(0.0)

    step._persist({"close": close_window}, (mkt_close_window, mkt_ret_window),
                  PartWindow(last=last_stored, since=warmup_dates[0]))

    prices_after_incr = sqlite_store.load(Tables.cube_part_prices)
    market_after_incr = sqlite_store.load(Tables.cube_part_market)

    # cube_part_prices already worked: full 60-day history preserved
    assert prices_after_incr["date"].min() == full_dates[0]
    assert prices_after_incr["date"].max() == new_dates[-1]
    assert len(prices_after_incr) == 60 * len(tickers)

    # cube_part_market must now match: the pre-existing 50 days must SURVIVE the
    # incremental run, with the 10 new days appended -- this is the bug being fixed
    # (the old code would have left only the 10-day tail here)
    assert market_after_incr["date"].min() == full_dates[0], (
        "cube_part_market lost its pre-existing history on an incremental run")
    assert market_after_incr["date"].max() == new_dates[-1]
    assert len(market_after_incr) == 60, (
        f"expected 60 rows (50 old + 10 new), got {len(market_after_incr)} -> "
        "cube_part_market was truncated instead of appended")

    print("\n=== SANITY CHECK: cube_part_market incremental append (not truncate) ===")
    print(f"  FULL build:        {len(market_after_full)} rows "
          f"({market_after_full['date'].min().date()} .. {market_after_full['date'].max().date()})")
    print(f"  INCREMENTAL build: {len(market_after_incr)} rows "
          f"({market_after_incr['date'].min().date()} .. {market_after_incr['date'].max().date()})")
    print("  CONCLUSION: cube_part_market now follows write_part's full-vs-incremental rule "
          "identically to cube_part_prices -- an incremental run appends the new tail and "
          "keeps prior history, instead of replacing the table with just the warm-up window. "
          "Validated.")


if __name__ == "__main__":
    test_incremental_market_run_appends_instead_of_truncating()
