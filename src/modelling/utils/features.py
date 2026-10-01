"""
features.py  (src/modelling/utils/features.py)
----------------------------------------------
Feature-side rules shared by every model family and the diagnostics: the ONE categorical
coercion (training and SHAP must see the same encoding), the float32 design matrix, the
monotone-constraint map, the per-horizon column override, and the float32 downcast.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from omegaconf import ListConfig, OmegaConf
from pandas.api.types import is_numeric_dtype

VALID_MONOTONE_DIRECTIONS = frozenset({-1, 0, 1})
CATEGORICAL_NA_CODE = -1  # LightGBM categorical code standing for "missing"


def coerce_categoricals(frame: pd.DataFrame, categorical_features: list[str] | None) -> pd.DataFrame:
    """Copy of `frame` with each categorical column coerced to a NUMERIC code (unparseable or
    missing -> `CATEGORICAL_NA_CODE`). Shared by the LightGBM Dataset (then int32) and the
    SHAP / PDP design matrix (float32), so an explanation never sees a different encoding."""
    out = frame.copy()
    for c in categorical_features or []:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(CATEGORICAL_NA_CODE)
    return out


def design_matrix(panel: pd.DataFrame, feats: list[str], categorical_features: list[str] | None = None) -> np.ndarray:
    """`panel[feats]` as a float32 matrix in `feats` order, categoricals coerced to their numeric
    codes. Any non-numeric column is treated as a categorical, so text can never raise here."""
    cats = list(categorical_features or [])
    non_numeric = [c for c in feats if c in panel.columns and not is_numeric_dtype(panel[c])]
    return coerce_categoricals(panel[feats], list(dict.fromkeys(cats + non_numeric))).to_numpy(dtype="float32")


def parse_monotone_feature_map(features_cfg: dict | ListConfig | None) -> dict[str, int]:
    """Parse a `monotonic.features` block: a mapping `{feature: direction}` or a list of
    single-key mappings `[{feature: direction}, ...]`; directions in {-1, 0, +1}."""
    if features_cfg is None:
        return {}
    out: dict[str, int] = {}
    if OmegaConf.is_dict(features_cfg):
        for name, direction in features_cfg.items():
            d = int(direction)
            if d not in VALID_MONOTONE_DIRECTIONS:
                raise ValueError(f"Invalid monotone direction {d} for {name}")
            out[str(name)] = d
        return out

    for item in features_cfg:
        if not OmegaConf.is_dict(item):
            raise ValueError("monotonic.features must be a mapping or a list of single-key mappings like `- f_sales_yield_xs: 1`")
        if len(item) != 1:
            raise ValueError(f"Each monotone list entry must contain exactly one feature and direction, got {dict(item)}")
        name, direction = next(iter(item.items()))
        d = int(direction)
        if d not in VALID_MONOTONE_DIRECTIONS:
            raise ValueError(f"Invalid monotone direction {d} for {name}")
        out[str(name)] = d
    return out


def build_monotone_constraints(feats: list[str], feature_map: dict[str, int]) -> list[int]:
    """LightGBM `monotone_constraints` vector aligned to `feats` order (0 = unconstrained)."""
    return [int(feature_map.get(f, 0)) for f in feats]


def columns_for_horizon(block: Any, horizon: Any) -> list[str] | None:
    """`block.columns_by_horizon[<horizon>]` when the family declares a list for this horizon,
    else None (the caller falls back to `block.columns`). Keys compare as ints, so a YAML `30:`
    matches a numpy / int / str horizon."""
    by_h = block.get("columns_by_horizon") if block else None
    if not by_h or horizon is None:
        return None
    try:
        h = int(horizon)
    except (TypeError, ValueError):
        return None
    for k in by_h.keys():
        try:
            if int(k) == h and by_h.get(k):
                return list(by_h.get(k))
        except (TypeError, ValueError):
            continue
    return None


def downcast_f32(df: pd.DataFrame) -> pd.DataFrame:
    """float64 columns -> float32 in place (halves a training panel; ranks need no float64)."""
    f64 = df.select_dtypes(include=["float64"]).columns
    if len(f64):
        df[f64] = df[f64].astype("float32")
    return df
