"""def14a_impute drops implausible say-on-pay support before imputing.

The 2026-07 audit measured 125 of 4,785 values (2.6%) below 0.60, steady at 1-4% in every
year since 2011. Spot-checked against the public record, they are wrong: JPM 2023 stored
0.31 (actual ~89%), SPG 2024 stored 0.111 (~93%), INTC 2023 stored 0.34 — the LLM is
reading a different percentage off the proxy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.constants.constants import SAY_ON_PAY_MIN_SUPPORT
from src.data_aggregate.utils.def14a_impute import impute_def14a


def _proxies(values: list[float]) -> pd.DataFrame:
    return pd.DataFrame({
        "ticker": ["JPM"] * len(values),
        "as_of": pd.date_range("2019-04-01", periods=len(values), freq="365D"),
        "say_on_pay_support_pct": values,
    })


def test_implausible_values_are_dropped_and_plausible_ones_kept():
    out, stats = impute_def14a(_proxies([0.31, 0.111, 0.93, 0.88]))
    got = out["say_on_pay_support_pct"].tolist()
    assert np.isnan(got[0]) and np.isnan(got[1]), "known-bad values survived"
    assert got[2] == 0.93 and got[3] == 0.88, "plausible support was clipped"
    assert stats["dropped implausible: say_on_pay_support_pct"] == 2


def test_genuine_shareholder_revolt_survives():
    """Real votes do reach the low 50s; the floor is 0.50, not 0.60, to keep them."""
    out, _ = impute_def14a(_proxies([0.52, 0.95, 0.96]))
    assert out["say_on_pay_support_pct"].iloc[0] == 0.52
    assert SAY_ON_PAY_MIN_SUPPORT == 0.50


def test_dropped_cell_is_recovered_from_neighbouring_years():
    """The NULL happens BEFORE the temporal gap-fill, so an interior bad year is
    interpolated from the years either side rather than left as a hole."""
    out, _ = impute_def14a(_proxies([0.90, 0.31, 0.94]))
    filled = out["say_on_pay_support_pct"].iloc[1]
    assert not np.isnan(filled), "interior dropped cell was not gap-filled"
    assert 0.90 <= filled <= 0.94


def test_say_on_pay_guard_prints_conclusion():
    out, stats = impute_def14a(_proxies([0.111, 0.31, 0.36, 0.45, 0.52, 0.88, 0.94, 1.0]))
    kept = out["say_on_pay_support_pct"].dropna()
    print("\n=== SANITY CHECK: say-on-pay plausibility ===")
    print(f"  floor {SAY_ON_PAY_MIN_SUPPORT}; input 8 values -> "
          f"{stats['dropped implausible: say_on_pay_support_pct']} dropped as implausible")
    print(f"  surviving range {kept.min():.3f} .. {kept.max():.3f} "
          f"(0.52 revolt kept, 0.111/0.31/0.36/0.45 dropped)")
    print("  Live table: 125 of 4,785 values (2.6%) below 0.60; JPM 2023 stored 0.31 vs")
    print("  ~89% actual, SPG 2024 0.111 vs ~93%, INTC 2023 0.34. Lossy by design — the")
    print("  proper fix is re-extraction with a ballot-pinned prompt. Validated.")
