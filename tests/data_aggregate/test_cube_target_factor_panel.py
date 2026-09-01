"""
`StepCubeTarget._factor_panel` (src/data_aggregate/transformers/step_cube_target.py).

`_factor_panel` calls `build_characteristics` and `characteristic_to_factor_return` directly,
alongside `_macro_changes` / `_asset_factors` / `_market_return`, so every factor family that
goes into the panel is visible as a flat list of calls rather than behind another wrapper.

This test proves it reproduces the SAME factor panel as composing those functions by hand --
guarding the aggregate-fingerprint invariant.

It also pins the read contract after the macro consolidation: `_factor_panel` makes exactly
ONE store read, of `prices_macro`. It used to make three across two tables (`cube_part_market`
for the market return and the commodity/FX closes, `macro` for the change factors), so the spy
store below asserts both the table identity AND the call count.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.constants.constants_price import MACRO_CUBE_FACTORS, MACRO_MARKET_SERIES
from src.data_aggregate.transformers.step_cube_target import StepCubeTarget
from src.data_store.schema import name_of
from src.data_aggregate.utils.common.price_frames import PriceFrames
from src.data_aggregate.utils.target.factors import (
    assemble_factor_panel, build_characteristics, characteristic_to_factor_return,
)

_MACRO_SERIES = [MACRO_MARKET_SERIES, *MACRO_CUBE_FACTORS.values()]


class _SpyStore:
    """A read spy, not a store: it records WHICH tables `_factor_panel` reads and how often.

    `name_of` because call sites pass the `Table` object; the assertion is about identity.
    Returns the long `prices_macro` shape, so the adapter's pivot is exercised for real."""

    def __init__(self, long: pd.DataFrame):
        self._long = long
        self.reads: list[str] = []

    def load(self, table, columns=None, where=None, **kw):
        name = name_of(table)
        self.reads.append(name)
        assert name == "prices_macro", f"unexpected store.load({table!r})"
        df = self._long
        if where and "ticker" in where:
            df = df[df["ticker"].isin(list(where["ticker"]))]
        return df.copy() if not df.empty else None


def _synthetic_inputs():
    tickers = ["AAA", "BBB", "CCC"]
    dates = pd.bdate_range("2022-01-01", periods=300, name="date")
    rng = np.random.default_rng(7)

    close = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0, 0.01, (len(dates), len(tickers))), axis=0)),
        index=dates, columns=pd.Index(tickers, name="ticker"))
    ret = close.pct_change().fillna(0.0)
    sector_ret = pd.DataFrame(np.repeat(ret.mean(axis=1).to_numpy()[:, None], len(tickers), axis=1),
                              index=dates, columns=tickers)

    # the macro series the cube consumes, as the LONG frame `prices_macro` really stores.
    # No FRED level series -> `_macro_changes` short-circuits to an empty frame, the same
    # branch the old empty-`macro` fixture exercised.
    macro_wide = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0, 0.008, (len(dates), len(_MACRO_SERIES))), axis=0)),
        index=dates, columns=_MACRO_SERIES)
    macro_long = (macro_wide.rename_axis("date").reset_index()
                  .melt(id_vars="date", var_name="ticker", value_name="close"))

    # quarterly fundamentals history -> exercises the size/value/quality characteristics too,
    # not just the price-only momentum/resvol ones
    as_of_dates = [dates[0], dates[60], dates[120], dates[180], dates[240]]
    rows = []
    for t in tickers:
        base = rng.uniform(0.8, 1.2)
        for i, d in enumerate(as_of_dates):
            rows.append({"ticker": t, "as_of": d,
                        "sharesOutstanding": 1_000_000 * base,
                        "netIncome": 50_000 * base * (1 + 0.05 * i),
                        "freeCashflow": 40_000 * base * (1 + 0.04 * i),
                        "stockholdersEquity": 500_000 * base,
                        "returnOnEquity": 0.10 * base,
                        "grossMargins": 0.40 * base,
                        "profitMargins": 0.15 * base,
                        "debtToEquity": 0.50 * base})
    fundamentals = pd.DataFrame(rows)

    frames = PriceFrames(trading_index=dates, universe=tuple(tickers), peers={},
                         close=close, ret=ret, sector_ret=sector_ret)
    return frames, fundamentals, close, ret, macro_wide, macro_long, dates


def _step(macro_long: pd.DataFrame) -> StepCubeTarget:
    step = object.__new__(StepCubeTarget)
    step._log = logging.getLogger("test")
    store = _SpyStore(macro_long)
    step._store = store
    step._context = type("Ctx", (), {"store": store})()
    return step


def _expected_returns(macro_wide: pd.DataFrame, calendar: pd.DatetimeIndex,
                      series: str) -> pd.Series:
    s = macro_wide[series].astype(float)
    s = s.reindex(s.index.union(calendar)).ffill()
    return s.pct_change(fill_method=None).reindex(calendar)


def test_factor_panel_matches_hand_composed_reference():
    frames, fundamentals, close, ret, macro_wide, macro_long, dates = _synthetic_inputs()
    step = _step(macro_long)

    panel, macro_cols = step._factor_panel(frames, fundamentals)

    # hand-composed reference: the same building blocks `_factor_panel` calls, composed
    # independently here rather than by importing its loop
    chars_expected = build_characteristics(close, ret, fundamentals, resvol_window=63)
    style_cols_expected = {}
    for name, char in chars_expected.items():
        char.name = name
        style_cols_expected[name] = characteristic_to_factor_return(char, ret)
    style_expected = pd.DataFrame(style_cols_expected)

    asset = pd.DataFrame({col: _expected_returns(macro_wide, dates, series)
                          for col, series in MACRO_CUBE_FACTORS.items()}, index=dates)
    fx_cols = [c for c in asset.columns if MACRO_CUBE_FACTORS[c].startswith("fx_")]
    commodity_expected = asset.drop(columns=fx_cols)
    currency_expected = asset[fx_cols]
    mkt_expected = _expected_returns(macro_wide, dates, MACRO_MARKET_SERIES)
    macro_chg_expected = pd.DataFrame(index=dates)      # no FRED levels -> short-circuits

    panel_expected, macro_cols_expected = assemble_factor_panel(
        mkt_expected, style_expected, commodity_expected, currency_expected, macro_chg_expected)

    pd.testing.assert_frame_equal(panel, panel_expected, check_exact=True)
    assert macro_cols == macro_cols_expected
    # fundamentals-gated characteristics actually got built (not silently skipped)
    assert {"size", "value"} <= set(style_expected.columns)
    # the panel column NAMES survived the move off `prices` unchanged
    assert set(MACRO_CUBE_FACTORS) <= set(panel.columns)

    print("\n=== SANITY CHECK: _factor_panel matches the hand-composed reference ===")
    print(f"  panel: {panel.shape[1]} factors over {len(panel)} dates; "
          f"style factors present: {sorted(set(style_expected.columns))}")
    print(f"  macro/market factor columns preserved: {sorted(set(MACRO_CUBE_FACTORS))} + market")
    print("  CONCLUSION: bit-identical to composing the same blocks by hand. Validated.")


def test_factor_panel_makes_exactly_one_macro_read():
    """The consolidation claim, pinned: ONE read of ONE table for all macro information.
    Three reads across two tables before (cube_part_market twice, macro once)."""
    frames, fundamentals, *_rest, macro_long, _dates = _synthetic_inputs()
    step = _step(macro_long)

    step._factor_panel(frames, fundamentals)

    assert step._store.reads == ["prices_macro"], f"reads were {step._store.reads}"

    print("\n=== SANITY CHECK: one macro read, one pivot ===")
    print(f"  _factor_panel store reads: {step._store.reads}")
    print("  market return + commodity/FX factors + macro changes all come off that single "
          "wide pivot. Validated.")


def test_panel_market_column_is_the_market_series_return():
    """`_sector_factor` de-markets the sector basket with the panel's OWN `market` column, so
    the sector regressor and the betas are neutralised against the identical series. Pin that
    the column really is the market series' return."""
    frames, fundamentals, close, ret, macro_wide, macro_long, dates = _synthetic_inputs()
    step = _step(macro_long)

    panel, _ = step._factor_panel(frames, fundamentals)

    assert "market" in panel.columns, "the panel lost its market column"
    pd.testing.assert_series_equal(
        panel["market"].reindex(ret.index),
        _expected_returns(macro_wide, dates, MACRO_MARKET_SERIES).reindex(ret.index),
        check_names=False)

    print("\n=== SANITY CHECK: panel['market'] is the market series return ===")
    print(f"  identical over {len(ret.index)} dates -> `{MACRO_MARKET_SERIES}` from "
          f"prices_macro is the single source for market beta and epsilon. Validated.")


if __name__ == "__main__":
    test_factor_panel_matches_hand_composed_reference()
    test_factor_panel_makes_exactly_one_macro_read()
    test_panel_market_column_is_the_market_series_return()
