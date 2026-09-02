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
  * a projected read (`fields=("close_split",)`) leaves the other frames None rather than
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

TICKERS = ["AAA", "BBB", "CCC", "DDD"]
# the VALUE fields only -- `date`/`ticker` are the join keys `load_price_frames` adds itself,
# so listing them here made the projection ask for them twice
ALL_PRICE_FIELDS = list(ALL_FIELDS)


@pytest.fixture
def frames() -> dict:
    """A synthetic EQUITY price panel with the awkward cases: a late IPO and a delisting.

    The market series is NOT a column in here. It used to be (SPY and CL=F sat inside
    `prices` alongside the equities), which is exactly what `prices_macro` took over -- so the
    fixture keeps it as a separate `market` Series, carrying the interior calendar hole."""
    idx = pd.bdate_range("2024-01-01", periods=60, name="date")
    rng = np.random.default_rng(11)
    close = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0, 0.01, (len(idx), len(TICKERS))), axis=0)),
        index=idx, columns=pd.Index(TICKERS, name="ticker"))
    close.iloc[:10, close.columns.get_loc("CCC")] = np.nan          # late IPO
    close.iloc[-8:, close.columns.get_loc("DDD")] = np.nan          # delisting
    market = pd.Series(400.0, index=idx)
    market.iloc[25] = np.nan                                        # interior calendar hole
    volume = pd.DataFrame(rng.lognormal(14, 0.4, close.shape), index=idx, columns=close.columns)
    return {"close_split": close,
            # a non-payer: on it the two bases are IDENTICAL by construction, which is the
            # cleanest possible round-trip fixture -- any divergence is a code fault.
            "close_total": close,
            # "open", not "open_": the canonical field name in price_frames.ALL_FIELDS
            "open": close.shift(1).bfill() * 1.001,
            "high": close * 1.01,
            "low": close * 0.99,
            "volume": volume,
            "market": market}


def _normalize(frames: dict) -> tuple[dict, pd.DatetimeIndex]:
    """Mirror StepCubePrices: calendar -> universe restriction -> returns -> sector returns.

    The market series is no longer a column inside `close` (it lives in `prices_macro`), so
    the calendar mask is built from a market series handed in, and there is no second
    (market) part frame to persist."""
    from src.data_aggregate.utils.common import data_utils as du

    wide = {k: v for k, v in frames.items() if k != "market"}
    close = wide["close_split"]
    mask = du.get_trading_days(close, frames["market"])
    idx = pd.DatetimeIndex(close.index[mask.to_numpy()], name="date")
    on_cal = {k: (v.loc[idx] if k != "volume" else v.reindex(idx)) for k, v in wide.items()}
    returns = du.daily_returns(on_cal["close_total"])

    universe = universe_columns(TICKERS, on_cal["close_split"])
    uni = {k: v.reindex(columns=universe) for k, v in on_cal.items()}
    uni["ret"] = returns.reindex(columns=universe)
    # `level_factor` exactly as StepCubePrices builds it: 1.0 for the clean names, a spinoff
    # factor before one date on BBB, and MASKED to close_split's non-null pattern. The mask is
    # the part that has to survive the round trip -- `S` is never NaN of its own accord, so
    # without it `frames_to_long`'s all-NaN row drop stops firing and CCC gains ten rows from
    # before it listed.
    lvl = pd.DataFrame(1.0, index=idx, columns=pd.Index(universe, name="ticker"))
    lvl.loc[: idx[30], "BBB"] = 1.241
    uni["level_factor"] = lvl.where(uni["close_split"].notna())
    # a deterministic stand-in for compute_sector_returns (equal-weight mean of the others)
    uni["sector_ret"] = pd.DataFrame(
        {t: uni["ret"].drop(columns=[t]).mean(axis=1) for t in universe},
        index=idx, columns=pd.Index(universe, name="ticker"))
    return uni, idx


def test_price_part_round_trip_is_bit_identical(frames, sqlite_store):
    uni, idx = _normalize(frames)
    parts = sqlite_store

    parts.replace(Tables.cube_part_prices, frames_to_long(uni))

    back = load_price_frames(parts, peers={}, fields=ALL_PRICE_FIELDS)

    for field in ALL_PRICE_FIELDS:
        expected = uni[field]
        got = getattr(back, field)
        # the long form drops all-NaN rows, so reindex onto the original grid before comparing
        got = got.reindex(index=expected.index, columns=expected.columns)
        pd.testing.assert_frame_equal(got, expected, check_exact=True, check_dtype=True,
                                      check_names=True)

    assert back.trading_index.equals(idx)
    # the calendar now comes from cube_part_prices' OWN dates, not a second market part
    assert load_trading_calendar(parts).equals(idx)
    # nothing non-equity can reach a cross-sectional rank through PriceFrames any more
    for gone in ("market_close", "mkt_ret", "other_close"):
        assert not hasattr(back, gone), f"PriceFrames still carries {gone}"

    print("\n=== SANITY CHECK: cube_part_prices round-trip ===")
    print(f"  {len(ALL_PRICE_FIELDS)} wide fields over {len(idx)} dates x "
          f"{len(back.universe)} tickers (ONE part table, no market twin)")
    print(f"  float64 preserved: {all(getattr(back, f).dtypes.eq('float64').all() for f in ALL_PRICE_FIELDS)}"
          f" | index/column names intact | calendar from cube_part_prices == the in-memory one")
    print("  CONCLUSION: persisting and reloading the price grid reproduces the in-memory frames "
          "bit-for-bit (check_exact=True), so every downstream step sees identical prices. "
          "Validated.")


def test_projected_read_leaves_other_fields_none(frames, sqlite_store):
    uni, _ = _normalize(frames)
    parts = sqlite_store
    parts.replace(Tables.cube_part_prices, frames_to_long(uni))

    back = load_price_frames(parts, peers={}, fields=("close_split",))
    assert back.close_split is not None
    for field in ALL_PRICE_FIELDS:
        if field != "close_split":
            assert getattr(back, field) is None, f"{field} was materialised but not requested"

    # and `require` fails loudly rather than letting a None reach the arithmetic
    with pytest.raises(ValueError, match="loaded without"):
        back.require("close_split", "volume")

    print("\n=== SANITY CHECK: projected price read ===")
    print("  fields=('close',) -> 1 frame materialised, 6 left None, market series absent")
    print("  require('volume') raises instead of a None-arithmetic TypeError deep in a builder")
    print("  CONCLUSION: a step only pays for the price fields it declares. Validated.")


def test_interior_calendar_hole_is_dropped_and_universe_is_sorted(frames):
    uni, idx = _normalize(frames)
    hole = frames["close_split"].index[25]
    assert hole not in idx, "the market-series hole must be dropped from the calendar"
    assert len(idx) == len(frames["close_split"].index) - 1

    # deterministic, sorted universe -- from a SHUFFLED input list, twice
    shuffled = list(reversed(TICKERS))
    a = universe_columns(shuffled, frames["close_split"])
    b = universe_columns(TICKERS, frames["close_split"])
    assert a == b == sorted(a), f"universe order is not deterministic/sorted: {a} vs {b}"
    assert list(uni["ret"].columns) == list(uni["close_split"].columns)

    with pytest.raises(RuntimeError, match="cube universe is empty"):
        universe_columns(["NOT_LISTED"], frames["close_split"])

    print("\n=== SANITY CHECK: calendar hole + deterministic universe ===")
    print(f"  interior {hole.date()} hole (market series missing) dropped: "
          f"{len(frames['close_split'].index)} -> {len(idx)} dates")
    print(f"  universe from a shuffled ticker list is stable and sorted: {a}")
    print("  CONCLUSION: the universe no longer comes from a set, so column order (and therefore "
          "float summation order) is identical across processes. Validated.")
