"""Reusable, config-driven modelling transformers.

Model families (`BaseModel` subclasses) are registered by the names `model.ensemble` uses.
`Monitor` and `Backtest` live in their own modules and are imported from there, so unpickling a
fitted member never pulls in SHAP / matplotlib.
"""

from __future__ import annotations

from src.modelling.transformers.base import BaseModel
from src.modelling.transformers.lightgbm_model import LightGBMModel
from src.modelling.transformers.linear_regression import LinearRegression
from src.modelling.transformers.random_forest import RandomForestModel

MODEL_FAMILIES: dict[str, type[BaseModel]] = {
    "lgbm": LightGBMModel,
    "lightgbm": LightGBMModel,
    "random_forest": RandomForestModel,
    "elasticnet": LinearRegression,
    "ridge": LinearRegression,
}


def model_class(family: str) -> type[BaseModel]:
    """The transformer class for a `model.ensemble` name; unknown names are a config error."""
    try:
        return MODEL_FAMILIES[family]
    except KeyError:
        raise ValueError(f"unknown model family {family!r} in model.ensemble; expected one of {sorted(MODEL_FAMILIES)}") from None


__all__ = ["BaseModel", "LightGBMModel", "LinearRegression", "MODEL_FAMILIES", "RandomForestModel", "model_class"]
