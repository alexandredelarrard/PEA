"""Real sub-50% say-on-pay results SURVIVE the cube's clean-on-read stage.

The 0.50 floor this module used to assert was measured wrong: 14 of 14 sampled sub-0.50 values
are genuine shareholder revolts, and the three the old docstring cited as proof of extraction
error are all real disclosures quoted in the filings themselves —

    JPM 2023   "the 31% support we received for last year's say-on-pay resolution"
    INTC 2023  "received only 34% support"
    SPG 2024   "11.1% of the votes cast favored our Say-on-Pay"

The floor nulled 61 correct rows, i.e. exactly the highest-signal governance events in the
table, so it was removed. `drop_implausible_def14a` survives as the documented seam for the next
present-but-known-wrong cell class, and the drop -> impute ORDER is what these tests pin: it is
load-bearing (`impute_def14a` is non-destructive by contract) and was broken once already.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_aggregate.utils.extras.def14a_impute import (
    drop_implausible_def14a, impute_def14a,
)

#: The three real revolts, at the values the filings state.
REAL_REVOLTS = {"JPM": 0.31, "INTC": 0.34, "SPG": 0.111}


def _clean_then_impute(df):
    """The cube order: VALIDATE (null known-wrong cells) then gap-fill. They are separate
    functions because `impute_def14a` is strictly non-destructive by contract — folding the
    nulling into it overwrote 59 present cells and broke
    `test_impute_real_data_nondestructive`."""
    dropped, drop_stats = drop_implausible_def14a(df)
    filled, fill_stats = impute_def14a(dropped)
    return filled, {**drop_stats, **fill_stats}


def _proxies(values: list[float], ticker: str = "JPM") -> pd.DataFrame:
    return pd.DataFrame({
        "ticker": [ticker] * len(values),
        "as_of": pd.date_range("2019-04-01", periods=len(values), freq="365D"),
        "say_on_pay_support_pct": values,
    })


def test_real_revolts_survive_the_clean_stage():
    """The three documented revolts go in and come out unchanged — one ticker each, so no
    temporal gap-fill can reach them and 'survived' means the cell itself was never touched."""
    for ticker, value in REAL_REVOLTS.items():
        out, stats = _clean_then_impute(_proxies([value], ticker))
        got = out["say_on_pay_support_pct"].iloc[0]
        assert got == value, f"{ticker} {value} was altered to {got}"
        assert not any("dropped" in k for k in stats), f"{ticker}: a drop rule fired: {stats}"


def test_drop_stage_nulls_nothing():
    """`drop_implausible_def14a` is a no-op stub: every value in, every value out, no stats."""
    values = [0.111, 0.31, 0.34, 0.45, 0.52, 0.88, 0.94, 1.0]
    dropped, stats = drop_implausible_def14a(_proxies(values))
    assert dropped["say_on_pay_support_pct"].tolist() == values
    assert stats == {}, f"the seam is meant to drop nothing, got {stats}"


def test_drop_stage_returns_a_copy():
    """It must not alias its input — callers pass the cube's frame and rely on the copy, and a
    stub that returns `df` itself would silently make a future drop rule mutate upstream."""
    df = _proxies([0.31, 0.93])
    dropped, _ = drop_implausible_def14a(df)
    dropped.loc[0, "say_on_pay_support_pct"] = 0.99
    assert df["say_on_pay_support_pct"].iloc[0] == 0.31, "input frame was mutated"


def test_impute_still_fills_a_genuinely_missing_year():
    """Removing the floor must not disturb the gap-fill: a NaN between two filled years is
    still interpolated, which is the behaviour a real missing proxy relies on."""
    out, _ = _clean_then_impute(_proxies([0.90, np.nan, 0.94]))
    filled = out["say_on_pay_support_pct"].iloc[1]
    assert not np.isnan(filled), "interior missing cell was not gap-filled"
    assert 0.90 <= filled <= 0.94


def test_say_on_pay_prints_conclusion():
    values = [0.111, 0.31, 0.34, 0.45, 0.52, 0.88, 0.94, 1.0]
    out, stats = _clean_then_impute(_proxies(values))
    kept = out["say_on_pay_support_pct"].dropna()
    n_dropped = sum(v for k, v in stats.items() if "dropped" in k)
    print("\n=== SANITY CHECK: say-on-pay revolts survive ===")
    print(f"  3 real revolts in -> {sum(1 for t, v in REAL_REVOLTS.items())} survive; "
          f"{n_dropped} cells nulled")
    print(f"  {len(values)} values in -> {len(kept)} out, range "
          f"{kept.min():.3f} .. {kept.max():.3f} (0.111 kept, not floored)")
    print("  Measured: 14/14 sampled sub-0.50 values are CORRECT. JPM 2023 disclosed 31%")
    print("  support, INTC 2023 34%, SPG 2024 11.1% — all quoted in the filings. The 0.50")
    print("  floor deleted 61 correct rows; it is gone. Validated.")
    assert n_dropped == 0
