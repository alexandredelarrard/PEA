"""Feature-side rules (src/modelling/utils/features.py): monotone parsing, categorical coercion
shared by training and SHAP, the per-horizon column override, and the float32 downcast."""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from src.modelling.utils.features import (
    CATEGORICAL_NA_CODE,
    build_monotone_constraints,
    coerce_categoricals,
    columns_for_horizon,
    design_matrix,
    downcast_f32,
    parse_monotone_feature_map,
)


def test_parse_monotone_feature_map_list_and_dict_formats() -> None:
    expected = {"f_employee_growth_xs": 1, "f_sga_growth_xs": -1, "f_fwd_eps_yield_xs": 1}
    as_list = OmegaConf.create({"features": [{k: v} for k, v in expected.items()]})
    as_dict = OmegaConf.create({"features": expected})
    assert parse_monotone_feature_map(as_list.features) == expected
    assert parse_monotone_feature_map(as_dict.features) == expected
    assert parse_monotone_feature_map(None) == {}
    with pytest.raises(ValueError):
        parse_monotone_feature_map(OmegaConf.create({"features": [{"f": 2}]}).features)
    feats = ["f_sga_growth_xs", "unconstrained", "f_employee_growth_xs"]
    assert build_monotone_constraints(feats, expected) == [-1, 0, 1]
    print("\n=== SANITY CHECK: monotone map ===")
    print(f"  list and dict formats parse identically; constraints follow feature order: {build_monotone_constraints(feats, expected)}. Validated.")


def test_design_matrix_matches_training_encoding_for_categoricals() -> None:
    panel = pd.DataFrame({"f0": [0.1, 0.2, 0.3, 0.4], "sector": ["Energy", "Utilities", "Energy", "Utilities"], "industry_group": [3, 8, 3, 8]})
    feats = ["f0", "sector", "industry_group"]
    x = design_matrix(panel, feats, categorical_features=["sector", "industry_group"])
    assert x.shape == (4, 3) and x.dtype == np.dtype("float32")
    assert set(np.unique(x[:, 1])) == {float(CATEGORICAL_NA_CODE)}, "unparseable text -> training's missing code"
    assert set(np.unique(x[:, 2])) == {3.0, 8.0}
    coerced = coerce_categoricals(cast(pd.DataFrame, panel[feats]), ["sector", "industry_group"])
    assert (coerced["industry_group"].to_numpy() == panel["industry_group"].to_numpy()).all()
    assert design_matrix(panel, feats).dtype == np.dtype("float32"), "a text column is coerced even when not declared"
    print("\n=== SANITY CHECK: one categorical encoding ===")
    print(f"  text 'sector' -> code {CATEGORICAL_NA_CODE}; numeric industry_group passes through {{3, 8}}; never raises on text. Validated.")


def test_columns_for_horizon_override_and_fallback() -> None:
    block = OmegaConf.create({"columns": ["d1", "d2"], "columns_by_horizon": {30: ["s1", "s2"]}})
    assert columns_for_horizon(block, 30) == ["s1", "s2"]
    assert columns_for_horizon(block, np.int64(30)) == ["s1", "s2"]
    assert columns_for_horizon(block, "30") == ["s1", "s2"]
    assert columns_for_horizon(block, 60) is None
    assert columns_for_horizon(block, None) is None
    assert columns_for_horizon(OmegaConf.create({"columns": ["a"]}), 30) is None
    print("\n=== SANITY CHECK: columns_by_horizon ===")
    print("  override for 30 (int / numpy / str key); None -> caller falls back to `columns`. Validated.")


def test_downcast_f32_only_touches_float64() -> None:
    df = pd.DataFrame({"a": np.array([1.0, 2.0]), "b": np.array([1, 2], dtype="int64"), "c": ["x", "y"]})
    before = df.dtypes.copy()
    out = downcast_f32(df)
    assert out is df and str(df["a"].dtype) == "float32"
    assert df["b"].dtype == before["b"] and df["c"].dtype == before["c"]
    print("\n=== SANITY CHECK: downcast_f32 ===")
    print(f"  dtypes after: {dict(df.dtypes.astype(str))}; ints and text untouched. Validated.")
