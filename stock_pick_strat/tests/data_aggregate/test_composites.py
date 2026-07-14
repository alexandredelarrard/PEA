"""Tests for the thematic composite-signal builder (composites.build_composites):
sign inversion, NaN-tolerant averaging, and the ADDITIVE guarantee (raw features
are kept -> no information is lost).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_aggregate.utils.composites import build_composites


def _panel():
    dates = pd.bdate_range("2020-01-01", periods=1)
    n = 5
    tickers = [f"T{i}" for i in range(n)]
    return pd.DataFrame({
        "date": [dates[0]] * n,
        "ticker": tickers,
        "f_val": [0.1, 0.3, 0.5, 0.7, 0.9],           # higher = better
        "f_bad": [0.9, 0.7, 0.5, 0.3, 0.1],           # perfectly anti-correlated with f_val
        "f_sparse": [1.0, np.nan, np.nan, np.nan, 5.0],  # mostly missing
    })


def test_composites_sign_inversion_nan_tolerant_and_additive():
    panel = _panel()
    groups = {
        "value": ["f_val"],
        "risk": ["-f_bad"],                            # inverted -> equals z(f_val)
        "mix": ["f_val", "-f_bad", "f_sparse"],
    }
    out = build_composites(panel, groups, method="zscore")

    # ADDITIVE: every raw feature is still there (no information lost)
    assert {"f_val", "f_bad", "f_sparse"}.issubset(out.columns)
    # composites added
    assert {"comp_value", "comp_risk", "comp_mix"}.issubset(out.columns)

    # f_bad = 1 - f_val -> z(-f_bad) == z(f_val), so inverting recovers the value sign
    assert np.allclose(out["comp_value"].to_numpy(), out["comp_risk"].to_numpy(), atol=1e-9)

    # NaN-tolerant: comp_mix is defined for EVERY row even though f_sparse is mostly NaN
    assert out["comp_mix"].notna().all()

    # input not mutated
    assert "comp_value" not in panel.columns

    print("\n=== SANITY CHECK: composite signals ===")
    print(f"  comp_value == comp_risk (inverted anti-correlated member): "
          f"{np.allclose(out['comp_value'], out['comp_risk'])}")
    print(f"  comp_mix non-null on all rows despite a mostly-NaN member; raw features kept. Validated.")


def test_composites_disabled_and_missing_members_are_safe():
    panel = _panel()
    # no groups -> unchanged
    assert build_composites(panel, {}).equals(panel)
    # a group whose members are all absent -> no column created, no crash
    out = build_composites(panel, {"ghost": ["f_does_not_exist", "-f_also_missing"]})
    assert "comp_ghost" not in out.columns
    assert {"f_val", "f_bad"}.issubset(out.columns)

    print("\n=== SANITY CHECK: composites degrade safely ===")
    print("  empty groups -> panel unchanged; all-missing group -> skipped, no crash. Validated.")
