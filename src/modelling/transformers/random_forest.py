"""
random_forest.py  (src/modelling/transformers/random_forest.py)
---------------------------------------------------------------
`RandomForestModel`: LightGBM in `boosting: rf` mode (`random_forest:` block) — bagged,
INDEPENDENT trees, a fixed tree count and no early stopping (more trees only lower variance).
Unlike scikit-learn's forest it handles the cube's NaNs, native categoricals and monotone
constraints, all borrowed from the `lgbm:` block, as is the feature list when
`random_forest.columns` / `columns_by_horizon` do not set one.
"""

from __future__ import annotations

import lightgbm as lgb
import pandas as pd
from omegaconf import DictConfig

from src.modelling.transformers.lightgbm_model import DEFAULT_SEED, LightGBMModel, train_booster
from src.modelling.utils.features import columns_for_horizon


class RandomForestModel(LightGBMModel):
    config_key = "random_forest"

    @classmethod
    def configured_columns(cls, config: DictConfig, horizon: int | None) -> list[str]:
        """`random_forest.columns_by_horizon[h]` → `random_forest.columns` → the LightGBM set."""
        block = cls.block(config)
        override = columns_for_horizon(block, horizon)
        if override:
            return override
        if block and block.get("columns"):
            return list(block.columns)
        return LightGBMModel.configured_columns(config, horizon)

    def _fit(self, train: pd.DataFrame, valid: pd.DataFrame | None) -> lgb.Booster:
        rf = self.block(self._config)
        params = {
            "objective": "regression",
            "metric": "rmse",
            "boosting": rf.get("boosting", "rf"),
            "bagging_fraction": float(rf.get("bagging_fraction", 0.7)),
            "bagging_freq": int(rf.get("bagging_freq", 1)),
            "feature_fraction": float(rf.get("feature_fraction", 0.6)),
            "max_depth": int(rf.get("max_depth", 8)),
            "num_leaves": int(rf.get("num_leaves", 127)),
            "min_child_samples": int(rf.get("min_child_samples", 50)),
            "lambda_l1": float(rf.get("lambda_l1", 0.0)),
            "lambda_l2": float(rf.get("lambda_l2", 1.0)),
            "verbosity": -1,
            "seed": int(self._config.get("seed", DEFAULT_SEED)),
            "deterministic": True,
            "force_row_wise": True,
        }
        self._objective(params)
        mono = self._monotone_constraints(self.features)
        if mono is not None and len(mono) == len(self.features):
            params["monotone_constraints"] = mono
        return train_booster(
            train,
            self.features,
            self.label_column,
            valid_panel=None,
            params=params,
            num_boost_round=int(rf.get("num_boost_round", 500)),
            categorical_features=self.categoricals or None,
            half_life_years=self.half_life_years,
        )
