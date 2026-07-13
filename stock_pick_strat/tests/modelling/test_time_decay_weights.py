"""Sanity checks for exponential time-decay sample weights."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.modelling.utils_model.model import CALENDAR_DAYS_PER_YEAR, time_decay_weights


def test_time_decay_half_life_at_reference_and_two_years_back():
    ref = pd.Timestamp("2026-01-01")
    dates = pd.Series([
        ref - pd.Timedelta(days=int(2 * CALENDAR_DAYS_PER_YEAR)),  # exactly 2y
        ref - pd.Timedelta(days=int(CALENDAR_DAYS_PER_YEAR)),       # 1y
        ref,
    ])
    w = time_decay_weights(dates, half_life_years=2.0, reference=ref)

    assert np.isclose(w[2], 1.0)
    assert np.isclose(w[0], 0.5, rtol=1e-2)
    assert np.isclose(w[1], 0.5 ** 0.5, rtol=1e-2)
    assert w[0] < w[1] < w[2]

    print("\n=== SANITY CHECK: time_decay_weights ===")
    print(f"  ref={ref.date()}, half_life=2y -> w(2y ago)={w[0]:.4f}, "
          f"w(1y ago)={w[1]:.4f}, w(today)={w[2]:.4f}")
    print("  Older rows down-weighted; 2 calendar years -> 0.5. Validated.")
