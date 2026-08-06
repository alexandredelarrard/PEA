"""
`StepCubeTarget._factor_panel` (src/data_aggregate/transformers/step_cube_target.py).

`_factor_panel` calls `build_characteristics` and `characteristic_to_factor_return` directly,
alongside `_macro_changes` / `_asset_factors`, so every factor family that goes into the panel
is visible as a flat list of calls rather than behind another wrapper.

This test proves it reproduces the SAME factor panel as composing those two functions and
`assemble_factor_panel` by hand -- guarding the aggregate-fingerprint invariant.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.data_aggregate.transformers.step_cube_target import (
    _COMMODITY_TICKERS, _CURRENCY_TICKERS, StepCubeTarget,
)
from src.data_aggregate.utils.common.price_frames import PriceFrames
from src.data_aggregate.utils.common.prices import price_column_returns
from src.data_aggregate.utils.target.factors import (
    assemble_factor_panel, build_characteristics, characteristic_to_factor_return,
)


class _FakeStore:
    """`_macro_changes` is the only store read `_factor_panel` triggers; empty macro exercises
    its early-return branch (mirrored by hand in the expected panel below)."""

    def load(self, name, columns=None):
        assert name == "macro", f"unexpected store.load({name!r})"
        return pd.DataFrame()


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

    other_cols = ["CL=F", "GC=F", "USDEUR=X"]
    other_close = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0, 0.008, (len(dates), len(other_cols))), axis=0)),
        index=dates, columns=other_cols)
    mkt_ret = pd.Series(rng.normal(0, 0.01, len(dates)), index=dates, name="SPY")

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
                         close=close, ret=ret, sector_ret=sector_ret,
                         mkt_ret=mkt_ret, other_close=other_close)
    return frames, fundamentals, close, ret, other_close, mkt_ret, dates


def test_factor_panel_matches_hand_composed_reference():
    frames, fundamentals, close, ret, other_close, mkt_ret, dates = _synthetic_inputs()

    step = object.__new__(StepCubeTarget)
    step._log = logging.getLogger("test")
    step._context = type("Ctx", (), {"store": _FakeStore()})()

    panel, macro_cols = step._factor_panel(frames, fundamentals)

    # hand-composed reference: the same two building blocks `_factor_panel` calls, composed
    # independently here rather than by importing its loop
    chars_expected = build_characteristics(close, ret, fundamentals, resvol_window=63)
    style_cols_expected = {}
    for name, char in chars_expected.items():
        char.name = name
        style_cols_expected[name] = characteristic_to_factor_return(char, ret)
    style_expected = pd.DataFrame(style_cols_expected)
    commodity_expected = price_column_returns(other_close, dict(_COMMODITY_TICKERS))
    currency_expected = price_column_returns(other_close, dict(_CURRENCY_TICKERS))
    macro_chg_expected = pd.DataFrame(index=dates)          # empty macro -> _macro_changes short-circuits
    panel_expected, macro_cols_expected = assemble_factor_panel(
        mkt_ret, style_expected, commodity_expected, currency_expected, macro_chg_expected)

    pd.testing.assert_frame_equal(panel, panel_expected, check_exact=True)
    assert macro_cols == macro_cols_expected
    # fundamentals-gated characteristics actually got built (not silently skipped)
    assert {"size", "value"} <= set(style_expected.columns)

    print("\n=== SANITY CHECK: _factor_panel matches the hand-composed reference ===")
    print(f"  panel: {panel.shape[1]} factors over {len(panel)} dates; "
          f"style factors present: {sorted(set(style_expected.columns))}")
    print("  CONCLUSION: _factor_panel's direct build_characteristics + "
          "characteristic_to_factor_return calls are bit-identical to composing them by hand. "
          "Validated.")


if __name__ == "__main__":
    test_factor_panel_matches_hand_composed_reference()
