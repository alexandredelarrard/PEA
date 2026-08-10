"""
`cube_part_prices` is the contract every other cube sub-step depends on, so persisting it and
reading it back must reproduce the in-memory frames EXACTLY.

If it does not, the failure is silent and diffuse: every downstream feature would be computed
on subtly different prices than the old monolithic path used, and the fingerprint (which calls
the builders directly, not through the parts) would not notice.

Checks here:
  * wide -> long -> wide is bit-identical for all seven fields, with float64 preserved and the
    index/column names intact;
  * the market table round-trips and is the definition of the trading calendar;
  * a projected read (`fields=("close",)`) leaves the other six frames None rather than
    silently materialising them;
  * an interior market-ticker hole is dropped from the calendar on BOTH sides;
  * the universe column order is deterministic and sorted (it used to come from a `set`, so it
    varied with PYTHONHASHSEED across processes -- and each CLI sub-step is its own process).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from src.data_store.schema import Tables
from src.data_aggregate.utils.common.price_frames import (
    ALL_FIELDS, frames_to_long, load_price_frames, load_trading_calendar,
    universe_columns,
)

MARKET = "SPY"
OTHER = "CL=F"
TICKERS = ["AAA", "BBB", "CCC", "DDD"]
# the VALUE fields only -- `date`/`ticker` are the join keys `load_price_frames` adds itself,
# so listing them here made the projection ask for them twice
ALL_PRICE_FIELDS = list(ALL_FIELDS)


@pytest.fixture
def frames() -> dict:
    """A synthetic price panel with the awkward cases: an interior market-ticker hole, a late
    IPO, a delisting, and a non-universe `other_ticker`."""
    idx = pd.bdate_range("2024-01-01", periods=60, name="date")
    rng = np.random.default_rng(11)
    cols = TICKERS + [MARKET, OTHER]
    close = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0, 0.01, (len(idx), len(cols))), axis=0)),
        index=idx, columns=pd.Index(cols, name="ticker"))
    close.iloc[:10, close.columns.get_loc("CCC")] = np.nan          # late IPO
    close.iloc[-8:, close.columns.get_loc("DDD")] = np.nan          # delisting
    close.iloc[25, close.columns.get_loc(MARKET)] = np.nan          # interior calendar hole
    volume = pd.DataFrame(rng.lognormal(14, 0.4, close.shape), index=idx, columns=close.columns)
    return {"close": close,
            # "open", not "open_": the canonical field name in price_frames.ALL_FIELDS
            "open": close.shift(1).bfill() * 1.001,
            "high": close * 1.01,
            "low": close * 0.99,
            "volume": volume}


def _normalize(frames: dict) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DatetimeIndex]:
    """Mirror StepCubePrices: calendar -> universe restriction -> returns -> sector returns."""
    from src.data_aggregate.utils.common import data_utils as du

    close = frames["close"]
    mask = du.get_trading_days(close, MARKET)
    idx = pd.DatetimeIndex(close.index[mask.to_numpy()], name="date")
    on_cal = {k: (v.loc[idx] if k != "volume" else v.reindex(idx)) for k, v in frames.items()}
    returns = du.daily_returns(on_cal["close"])

    universe = universe_columns(TICKERS, on_cal["close"])
    uni = {k: v.reindex(columns=universe) for k, v in on_cal.items()}
    uni["ret"] = returns.reindex(columns=universe)
    # a deterministic stand-in for compute_sector_returns (equal-weight mean of the others)
    uni["sector_ret"] = pd.DataFrame(
        {t: uni["ret"].drop(columns=[t]).mean(axis=1) for t in universe},
        index=idx, columns=pd.Index(universe, name="ticker"))
    market_cols = [MARKET, OTHER]
    return uni, on_cal["close"][market_cols], returns[market_cols], idx


def test_price_part_round_trip_is_bit_identical(frames, sqlite_store):
    uni, mkt_close, mkt_ret, idx = _normalize(frames)
    parts = sqlite_store

    prices_long, market_long = frames_to_long(uni, mkt_close, mkt_ret)
    parts.replace(Tables.cube_part_prices, prices_long)
    parts.replace(Tables.cube_part_market, market_long)

    back = load_price_frames(parts, peers={}, market_ticker=MARKET,
                             fields=ALL_PRICE_FIELDS, with_market=True,
                             other_tickers=[OTHER])

    for field in ALL_PRICE_FIELDS:
        expected = uni[field]
        got = getattr(back, field)
        # the long form drops all-NaN rows, so reindex onto the original grid before comparing
        got = got.reindex(index=expected.index, columns=expected.columns)
        pd.testing.assert_frame_equal(got, expected, check_exact=True, check_dtype=True,
                                      check_names=True)

    pd.testing.assert_series_equal(
        back.market_close.reindex(idx), mkt_close[MARKET].reindex(idx),
        check_exact=True, check_dtype=True, check_names=False)
    pd.testing.assert_series_equal(
        back.mkt_ret.reindex(idx), mkt_ret[MARKET].reindex(idx),
        check_exact=True, check_dtype=True, check_names=False)

    assert back.trading_index.equals(idx)
    assert load_trading_calendar(parts).equals(idx)
    assert OTHER in back.other_close.columns and MARKET in back.other_close.columns

    print("\n=== SANITY CHECK: cube_part_prices round-trip ===")
    print(f"  {len(ALL_PRICE_FIELDS)} wide fields + 2 market series over {len(idx)} dates x "
          f"{len(back.universe)} tickers")
    print(f"  float64 preserved: {all(getattr(back, f).dtypes.eq('float64').all() for f in ALL_PRICE_FIELDS)}"
          f" | index/column names intact | calendar from the market part == the in-memory one")
    print("  CONCLUSION: persisting and reloading the price grid reproduces the in-memory frames "
          "bit-for-bit (check_exact=True), so every downstream step sees identical prices. "
          "Validated.")


def test_projected_read_leaves_other_fields_none(frames, sqlite_store):
    uni, mkt_close, mkt_ret, _ = _normalize(frames)
    parts = sqlite_store
    prices_long, market_long = frames_to_long(uni, mkt_close, mkt_ret)
    parts.replace(Tables.cube_part_prices, prices_long)
    parts.replace(Tables.cube_part_market, market_long)

    back = load_price_frames(parts, peers={}, market_ticker=MARKET, fields=("close",))
    assert back.close is not None
    for field in ALL_PRICE_FIELDS:
        if field != "close":
            assert getattr(back, field) is None, f"{field} was materialised but not requested"
    assert back.market_close is None and back.other_close is None    # with_market=False

    # and `require` fails loudly rather than letting a None reach the arithmetic
    with pytest.raises(ValueError, match="loaded without"):
        back.require("close", "volume")

    print("\n=== SANITY CHECK: projected price read ===")
    print("  fields=('close',) -> 1 frame materialised, 6 left None, market series absent")
    print("  require('volume') raises instead of a None-arithmetic TypeError deep in a builder")
    print("  CONCLUSION: a step only pays for the price fields it declares. Validated.")


def test_interior_calendar_hole_is_dropped_and_universe_is_sorted(frames):
    uni, _, _, idx = _normalize(frames)
    hole = frames["close"].index[25]
    assert hole not in idx, "the market-ticker hole must be dropped from the calendar"
    assert len(idx) == len(frames["close"].index) - 1

    # deterministic, sorted universe -- from a SHUFFLED input list, twice
    shuffled = list(reversed(TICKERS))
    a = universe_columns(shuffled, frames["close"])
    b = universe_columns(TICKERS, frames["close"])
    assert a == b == sorted(a), f"universe order is not deterministic/sorted: {a} vs {b}"
    assert list(uni["ret"].columns) == list(uni["close"].columns)

    with pytest.raises(RuntimeError, match="cube universe is empty"):
        universe_columns(["NOT_LISTED"], frames["close"])

    print("\n=== SANITY CHECK: calendar hole + deterministic universe ===")
    print(f"  interior {hole.date()} hole (market ticker missing) dropped: "
          f"{len(frames['close'].index)} -> {len(idx)} dates")
    print(f"  universe from a shuffled ticker list is stable and sorted: {a}")
    print("  CONCLUSION: the universe no longer comes from a set, so column order (and therefore "
          "float summation order) is identical across processes. Validated.")
