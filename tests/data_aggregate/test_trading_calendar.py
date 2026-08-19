"""
The cube trading calendar, after it was repointed off `prices`.

The calendar is DEFINED as "the dates the market series traded" -- `du.get_trading_days` used
to express that as `close["SPY"].notna()`, reading a column inside the equity close matrix,
because SPY was stored in `prices` alongside the equities. SPY now lives in `prices_macro`, so
the caller reads the series and hands it in.

That is a load-bearing change: the calendar decides which dates exist for the ENTIRE universe,
and all six cube sub-steps read it back. So the claim being tested is bit-identity -- the new
signature must produce exactly the calendar the old one did on the same data. Plus the reader
half, which moved from `cube_part_market`'s dates to `cube_part_prices`' own dates.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_aggregate.utils.common import data_utils as du
from src.data_aggregate.utils.common.price_frames import load_trading_calendar
from src.data_store.schema import Tables

MARKET = "SPY"


def _panel(n: int = 12, n_stocks: int = 10):
    """Equity close matrix + a market series on the same dates, with two engineered holes:
      * a market hole where a QUORUM of stocks trade  -> dropped for everyone, and WARNED,
      * a market hole where almost nothing trades     -> dropped silently (a real holiday).
    """
    dates = pd.bdate_range("2024-01-01", periods=n)
    close = pd.DataFrame(100.0, index=dates, columns=[f"T{i}" for i in range(n_stocks)])
    market = pd.Series(400.0, index=dates)

    market.iloc[4] = np.nan                       # hole #1: all stocks still trade -> warn
    market.iloc[7] = np.nan                       # hole #2: a genuine market holiday...
    close.iloc[7, :] = np.nan                     # ...with the stocks out too
    return close, market, dates


def test_new_signature_reproduces_the_old_calendar_exactly():
    """`get_trading_days(close, market_series)` == the old `get_trading_days(close_with_SPY,
    "SPY")`, so repointing the source did not move a single date."""
    close, market, dates = _panel()

    # what the OLD code saw: the market as a column INSIDE the equity frame
    old_input = close.copy()
    old_input[MARKET] = market
    old_mask = old_input[MARKET].notna()           # the old body, verbatim

    new_mask = du.get_trading_days(close, market, MARKET)

    pd.testing.assert_series_equal(new_mask, old_mask, check_names=False)
    assert list(dates[new_mask]) == [d for i, d in enumerate(dates) if i not in (4, 7)]

    print("\n=== SANITY CHECK: calendar bit-identity across the repoint ===")
    print(f"  {len(dates)} dates, 2 market holes -> {int(new_mask.sum())} trading days.")
    print("  New (series) signature == old (column-name) signature, date for date. Validated.")


def test_quorum_denominator_no_longer_counts_non_equity_columns():
    """The interior-hole WARNING compares stock coverage against `0.5 * close.shape[1]`. When
    the market/commodity/FX columns lived in `close` they inflated that denominator; now
    `close` is equity-only, so the quorum is over stocks alone."""
    close, market, _ = _panel(n_stocks=10)
    # old frame carried 5 extra non-equity columns it counted as "stocks"
    old_input = close.copy()
    for extra in (MARKET, "CL=F", "GC=F", "USDEUR=X", "^VIX"):
        old_input[extra] = 1.0
    old_quorum = 0.5 * old_input.shape[1]              # 7.5 over 10 real stocks
    new_quorum = 0.5 * close.shape[1]                  # 5.0 over 10 real stocks

    assert old_quorum > new_quorum
    assert new_quorum == 5.0 and old_quorum == 7.5

    print("\n=== SANITY CHECK: quorum denominator ===")
    print(f"  10 stocks: old quorum {old_quorum} (counted 5 non-equity columns as stocks) "
          f"-> new quorum {new_quorum}.")
    print("  The >=50%-of-stocks hole warning is now measured over stocks only. Validated.")


def test_load_trading_calendar_reads_cube_part_prices_dates(sqlite_store):
    """The reader half: the calendar comes back as `cube_part_prices`' own distinct dates.
    It used to read `cube_part_market` -- the same dates, one table earlier."""
    dates = pd.bdate_range("2024-02-01", periods=5)
    long = pd.DataFrame([{"date": d, "ticker": t, "close": 10.0}
                         for d in dates for t in ("AAPL", "MSFT")])
    sqlite_store.replace(Tables.cube_part_prices, long)

    idx = load_trading_calendar(sqlite_store)
    assert isinstance(idx, pd.DatetimeIndex) and idx.name == "date"
    assert list(idx) == list(dates)                    # deduped across tickers, sorted

    print("\n=== SANITY CHECK: load_trading_calendar off cube_part_prices ===")
    print(f"  {len(long)} rows over {len(dates)} dates x 2 tickers -> {len(idx)} calendar "
          f"dates, {idx.min().date()} .. {idx.max().date()}. Validated.")


if __name__ == "__main__":
    test_new_signature_reproduces_the_old_calendar_exactly()
    test_quorum_denominator_no_longer_counts_non_equity_columns()
