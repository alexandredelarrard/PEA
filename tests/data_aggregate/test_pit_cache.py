"""
`PitFrames` must be a pure memoization of the point-in-time accessors: same frames, fewer
computations.

The cube's fundamentals sub-step hands ONE filing history to five builders, and its extras
sub-step hands one to three more. Today each builder re-pivots the same fields off the same
frame -- `sharesOutstanding` ~7 times, the daily market cap 6 times -- because
`fundamentals_to_daily` is a pure function and nobody owned a cache. Sharing one
`PitFrames` removes that, and this test is the proof that sharing changes no number:
every accessor is compared bit-for-bit against the free function it wraps.

It also pins the IDENTITY GUARANTEE. The cache is keyed only by field name, so it is valid
solely for the (history, trading_index, close) it was built with -- and the cube sub-steps
deliberately use DIFFERENT warm-up windows. A silently reused cache would be a correctness
bug, so `assert_matches` must raise rather than return a frame computed elsewhere.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_aggregate.utils.common.pit import (
    PitFrames,
    daily_market_cap,
    fiscal_apply_to_daily,
    fiscal_change_to_daily,
    fundamentals_to_daily,
    infer_yoy_periods,
)
from tests.data_aggregate.aggregate_fingerprint import fundamentals

FIELDS = ["sharesOutstanding", "totalRevenue", "netIncome", "freeCashflow",
          "stockholdersEquity", "totalDebt", "employees", "pretaxIncome"]


@pytest.fixture(scope="module")
def inputs() -> tuple[pd.DataFrame, pd.DatetimeIndex, pd.DataFrame]:
    """The same frozen real filing history the fingerprint uses (22 names, all 11 GICS
    sectors, 15y of quarterly filings) plus a seeded price panel."""
    fund = fundamentals()
    tickers = sorted(fund["ticker"].unique())
    idx = pd.bdate_range("2019-01-02", "2026-06-30")
    rng = np.random.default_rng(7)
    close = pd.DataFrame(
        100.0 * np.exp(np.cumsum(rng.normal(0.0004, 0.018, (len(idx), len(tickers))), axis=0)),
        index=idx, columns=tickers)
    return fund, idx, close


def test_pit_accessors_are_bit_identical_to_the_free_functions(inputs):
    fund, idx, close = inputs
    pit = PitFrames(fund, idx, close)

    for field in FIELDS:
        pd.testing.assert_frame_equal(pit.daily(field), fundamentals_to_daily(fund, field, idx),
                                      check_exact=True, check_dtype=True, check_names=True)
        # __call__ is the FieldGetter alias, so capital.py can be handed a PitFrames
        pd.testing.assert_frame_equal(pit(field), pit.daily(field), check_exact=True)

    pd.testing.assert_frame_equal(pit.market_cap, daily_market_cap(fund, close),
                                  check_exact=True, check_dtype=True)
    assert pit.yoy_periods == infer_yoy_periods(fund)

    # change(): periods=None must mean "one fiscal year of filings for THIS history"
    pd.testing.assert_frame_equal(
        pit.change("totalRevenue"),
        fiscal_change_to_daily(fund, "totalRevenue", idx, kind="pct", periods=pit.yoy_periods),
        check_exact=True)
    pd.testing.assert_frame_equal(
        pit.change("netIncome", kind="diff", periods=1),
        fiscal_change_to_daily(fund, "netIncome", idx, kind="diff", periods=1),
        check_exact=True)

    def yoy(s: pd.Series) -> pd.Series:
        return s.pct_change(periods=4)

    pd.testing.assert_frame_equal(
        pit.applied("totalRevenue", "yoy4", yoy),
        fiscal_apply_to_daily(fund, "totalRevenue", idx, yoy), check_exact=True)

    print("\n=== SANITY CHECK: PitFrames == the free point-in-time functions ===")
    print(f"  {len(FIELDS)} fields + market_cap + change(pct/diff) + applied() over "
          f"{len(fund)} filings x {len(idx)} trading days")
    print("  CONCLUSION: every accessor is bit-identical (check_exact=True) to the function "
          "it memoizes, so sharing one cache across builders cannot move a number. Validated.")


def test_repeated_access_computes_once(inputs):
    """The actual saving: the 6 `daily_market_cap` call sites and the ~7
    `sharesOutstanding` pivots collapse to one each."""
    fund, idx, close = inputs
    pit = PitFrames(fund, idx, close)

    calls = {"pivots": 0}
    real = fundamentals_to_daily

    import src.data_aggregate.utils.common.pit as pit_mod
    def counting(hist, field, index):
        calls["pivots"] += 1
        return real(hist, field, index)

    pit_mod.fundamentals_to_daily = counting
    try:
        # simulate the real access pattern: 8 fields read repeatedly by 5 builders
        for _ in range(5):
            for field in FIELDS:
                pit.daily(field)
        for _ in range(6):                                  # the 6 daily_market_cap sites
            _ = pit.market_cap
    finally:
        pit_mod.fundamentals_to_daily = real

    stats = pit.stats()
    assert stats["fields"] == len(FIELDS), stats
    assert stats["accesses"] == 5 * len(FIELDS), stats
    assert stats["hits"] == 4 * len(FIELDS), stats
    assert stats["market_cap"] == 1, stats
    # market_cap pivots sharesOutstanding through the module-level function once
    assert calls["pivots"] == len(FIELDS) + 1, (
        f"expected {len(FIELDS)} field pivots + 1 for market_cap, got {calls['pivots']}")

    print("\n=== SANITY CHECK: PitFrames computes each frame once ===")
    print(f"  {stats['accesses']} accesses over {stats['fields']} fields -> "
          f"{stats['fields']} pivots ({stats['hits']} cache hits)")
    print(f"  6 market_cap reads -> {stats['market_cap']} computation")
    print("  CONCLUSION: the ~7 sharesOutstanding pivots and 6 market-cap computations per "
          "run collapse to one each. Validated.")


def test_cache_refuses_a_different_window(inputs):
    """The identity guarantee. Sub-steps use different warm-up trims, so serving a cache
    built on another window would be a correctness bug, not a slow path."""
    fund, idx, close = inputs
    pit = PitFrames(fund, idx, close)
    pit.assert_matches(idx, close)                          # the window it was built on: fine

    with pytest.raises(ValueError, match="build one cache per warm-up window"):
        pit.assert_matches(idx[500:], close)
    with pytest.raises(ValueError, match="different `close` frame"):
        pit.assert_matches(idx, close.iloc[:, :3])

    print("\n=== SANITY CHECK: PitFrames rejects a foreign window ===")
    print(f"  built on {len(idx)} days; assert_matches({len(idx[500:])} days) raises, "
          "as does a re-shaped close frame")
    print("  CONCLUSION: a cache cannot silently leak across warm-up windows. Validated.")


def test_absent_history_behaves_like_an_absent_field(inputs):
    """A None/empty history must degrade exactly like `fundamentals_to_daily` does for a
    field that was never reported -- an empty frame on the trading index -- so each
    builder's own `if history is None` guard keeps its current behaviour."""
    _, idx, close = inputs
    empty = PitFrames(None, idx, close)
    assert empty.empty
    for field in ("totalRevenue", "sharesOutstanding"):
        got = empty.daily(field)
        pd.testing.assert_frame_equal(got, pd.DataFrame(index=idx), check_exact=True)
        assert not empty.has(field)
    assert empty.market_cap.empty and empty.yoy_periods == 1
    # and a field genuinely absent from a REAL history behaves the same way
    fund, _, _ = inputs
    real = PitFrames(fund, idx, close)
    pd.testing.assert_frame_equal(real.daily("noSuchTag"),
                                  fundamentals_to_daily(fund, "noSuchTag", idx),
                                  check_exact=True)
    assert not real.has("noSuchTag") and real.has("totalRevenue")

    print("\n=== SANITY CHECK: absent history / absent field ===")
    print("  None history and an unreported tag both yield an empty frame on the trading "
          "index, matching fundamentals_to_daily; has() distinguishes them. Validated.")
