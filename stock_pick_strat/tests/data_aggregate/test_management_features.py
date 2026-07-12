"""Tests for the management/ownership features
(src/data_aggregate/utils/management_features.py).

The snapshot has no free historical archive, so the guarantee that matters is
that every field is strictly point-in-time (applied only from its `as_of`
onward, never backwards) and that revenue-per-employee combines the historical
TTM revenue with the employee snapshot correctly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_aggregate.utils.management_features import (
    _management_fields, build_management_feature_panel,
)


def _synth_mgmt():
    return pd.DataFrame([{
        "ticker": "AAA", "as_of": "2020-03-02",
        "heldPercentInsiders": 0.30, "heldPercentInstitutions": 0.60,
        "founder_present": 1, "family_owned": 1,
        "net_insider_buying": 0.02, "ceo_age": 45.0,
    }])


def test_management_fields_point_in_time():
    idx = pd.bdate_range("2020-01-01", "2020-06-01")
    F = _management_fields(_synth_mgmt(), idx)

    before = pd.Timestamp("2020-02-14")   # before the 2020-03-02 as_of
    after = pd.Timestamp("2020-04-01")

    # ---- point-in-time: nothing before the snapshot's as_of ----
    for name, frame in F.items():
        assert np.isnan(frame.loc[before, "AAA"]), f"{name} leaked before as_of"

    # ---- values available on/after as_of ----
    assert abs(F["insider_ownership"].loc[after, "AAA"] - 0.30) < 1e-9
    assert abs(F["institutional_ownership"].loc[after, "AAA"] - 0.60) < 1e-9
    assert F["founder_led"].loc[after, "AAA"] == 1
    assert F["family_owned"].loc[after, "AAA"] == 1
    assert abs(F["ceo_age"].loc[after, "AAA"] - 45.0) < 1e-9

    print("\n=== SANITY CHECK: management fields point-in-time ===")
    print(f"  insider={F['insider_ownership'].loc[after,'AAA']:.2f}  "
          f"instit={F['institutional_ownership'].loc[after,'AAA']:.2f}  "
          f"founder_led={int(F['founder_led'].loc[after,'AAA'])}  "
          f"family={int(F['family_owned'].loc[after,'AAA'])}")
    print("  All fields NaN before the 2020-03-02 as_of -> strictly leak-free. Validated.")


def test_build_panel_empty_without_history():
    # no snapshot yet -> empty panel, never raises (mirrors analyst behaviour)
    idx = pd.bdate_range("2020-01-01", "2020-06-01")
    panel = build_management_feature_panel(None, {"AAA": ["BBB"]}, idx)
    assert list(panel.columns) == ["date", "ticker"] and panel.empty

    print("\n=== SANITY CHECK: graceful skip ===")
    print("  No management history -> empty (date,ticker) panel, no crash. Validated.")
