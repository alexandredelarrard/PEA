"""
families.py  (src/modelling/transformers/families.py)
-----------------------------------------------------
Registry of the model families (`BaseModel` subclasses) by the names `model.ensemble` uses, and `model_class`
to resolve one. Importing it loads every family; code that needs only one family imports its module directly.
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
