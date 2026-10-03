"""
base.py  (src/modelling/transformers/base.py)
---------------------------------------------
`BaseModel`: the sklearn-style contract every model family implements.

A model is built from `(context, config, family, horizon)` and reads its OWN YAML block
(`config[<config_key>]`: hyperparameters, `columns` / `columns_by_horizon`, `task`) plus the shared
`model.*` keys (label column, time-decay weights). `fit` resolves the feature list as the
configured columns that are present in the training panel (meta / target columns can never be
features), in configured order; `predict` scores a panel row for row; `evaluate` reports the
daily cross-sectional IC (regression) or AUC (classification). A fitted model pickles to one
file without its context, config or logger, and unpickles to its own class, ready to predict.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, ClassVar, Self

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from src.context import Context
from src.data_aggregate.utils.assemble.cube import is_meta_column
from src.data_store.schema import Table
from src.modelling.utils.artifacts import load_member
from src.modelling.utils.features import columns_for_horizon
from src.modelling.utils.metrics import auc, daily_ic
from src.modelling.utils.panel import load_frame

TASK_REGRESSION = "regression"
TASK_CLASSIFICATION = "classification"
_UNPICKLED = ("_context", "_config", "_log")


def half_life_years(config: DictConfig) -> float | None:
    """`model.weight_decay.half_life_years` when time-decay weighting is enabled, else None."""
    wd = config.model.get("weight_decay")
    return float(wd.half_life_years) if (wd and wd.get("enabled", False)) else None


class BaseModel:
    config_key: ClassVar[str]
    supports_shap: ClassVar[bool] = False
    supports_classification: ClassVar[bool] = True

    def __init__(self, context: Context, config: DictConfig, family: str, horizon: int | None = None) -> None:
        self._context = context
        self._config = config
        self._log = logging.getLogger(type(self).__module__)
        self.family = str(family)
        self.horizon = None if horizon is None else int(horizon)
        self.label_column = str(config.model.label_column)
        self.task = str(self.block(config).get("task", TASK_REGRESSION))
        if self.task not in (TASK_REGRESSION, TASK_CLASSIFICATION):
            raise ValueError(f"{self.config_key}.task must be '{TASK_REGRESSION}' or '{TASK_CLASSIFICATION}', got {self.task!r}")
        if self.task == TASK_CLASSIFICATION and not self.supports_classification:
            raise ValueError(f"the '{self.family}' family has no classification objective; set {self.config_key}.task: {TASK_REGRESSION}")
        self.half_life_years = half_life_years(config)
        self.features: list[str] = []
        self.categoricals: list[str] = []
        self.model: Any = None

    # ---- configuration --------------------------------------------------------------- #
    @classmethod
    def block(cls, config: DictConfig) -> Any:
        """This family's YAML block (`{}` when absent)."""
        return config.get(cls.config_key) or {}

    @classmethod
    def configured_columns(cls, config: DictConfig, horizon: int | None) -> list[str]:
        """Numeric feature names for `horizon`: `columns_by_horizon[h]`, else `columns`."""
        block = cls.block(config)
        override = columns_for_horizon(block, horizon)
        if override:
            return override
        return list(block.columns) if (block and block.get("columns")) else []

    @classmethod
    def configured_categoricals(cls, config: DictConfig) -> list[str]:
        """Categorical feature names; none for families without native categoricals."""
        return []

    # ---- data ------------------------------------------------------------------------- #
    def load_data(
        self,
        table: Table,
        *,
        columns: list[str] | None = None,
        where: dict | None = None,
        since: object = None,
        until: object = None,
        downcast: bool = True,
    ) -> pd.DataFrame | None:
        """Projected read of `table` through `context.store`. Default projection: `date`,
        `ticker` and this model's configured features that the table has. Training reads are
        float32 (`downcast`); keep float64 for scoring."""
        store = self._context.store
        if columns is None:
            present = set(store.columns(table))
            wanted = ["date", "ticker"] + self.configured_columns(self._config, self.horizon) + self.configured_categoricals(self._config)
            columns = [c for c in dict.fromkeys(wanted) if c in present]
        return load_frame(store, table, columns=columns, where=where, since=since, until=until, downcast=downcast)

    def _usable(self, column: str, frame: pd.DataFrame) -> bool:
        return column in frame.columns and not is_meta_column(column) and column != self.label_column

    # ---- sklearn-style surface ----------------------------------------------------------- #
    def fit(self, train: pd.DataFrame, valid: pd.DataFrame | None = None) -> Self:
        """Fit on `train`; `valid` is the chronological early-stopping tail (ignored by
        families that do not early-stop)."""
        numeric = [c for c in self.configured_columns(self._config, self.horizon) if self._usable(c, train)]
        self.categoricals = [c for c in self.configured_categoricals(self._config) if self._usable(c, train)]
        self.features = numeric + self.categoricals
        if self.task == TASK_CLASSIFICATION:
            self._check_binary_label(train)
        self.model = self._fit(train, valid)
        return self

    def predict(self, panel: pd.DataFrame) -> pd.Series:
        """Raw score (regression) or positive-class probability (classification), aligned to
        `panel`'s index."""
        if self.model is None:
            raise RuntimeError(f"{self.family} h{self.horizon} is not fitted")
        return pd.Series(self._predict(panel), index=panel.index, name="score")

    def evaluate(self, panel: pd.DataFrame, predictions: pd.Series | None = None) -> dict[str, float]:
        """Daily cross-sectional IC summary (`daily_ic`, IR annualised by the horizon) for
        regression; `{"auc", "n_rows"}` for classification."""
        preds = self.predict(panel) if predictions is None else predictions
        if self.task == TASK_CLASSIFICATION:
            return {"auc": auc(panel[self.label_column].to_numpy(), preds.to_numpy()), "n_rows": float(len(panel))}
        return daily_ic(panel, preds, self.label_column, horizon=self.horizon or 1)

    def importance(self) -> pd.Series:
        """Per-feature importance (comparable within one model only)."""
        raise NotImplementedError

    # ---- persistence ------------------------------------------------------------------- #
    @classmethod
    def load(cls, path: Path) -> Self:
        obj = load_member(path)
        if not isinstance(obj, cls):
            raise TypeError(f"{path} holds a {type(obj).__name__}, not a {cls.__name__}")
        return obj

    def __getstate__(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if k not in _UNPICKLED}

    def __setstate__(self, state: dict[str, Any]) -> None:
        # an unpickled member predicts; it carries no context / config (fit and load_data need them)
        self.__dict__.update(state)
        self._log = logging.getLogger(type(self).__module__)

    # ---- family hooks -------------------------------------------------------------------- #
    def _fit(self, train: pd.DataFrame, valid: pd.DataFrame | None) -> Any:
        raise NotImplementedError

    def _predict(self, panel: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def _check_binary_label(self, train: pd.DataFrame) -> None:
        values = set(np.unique(train[self.label_column].dropna().to_numpy()))
        if not values <= {0, 1}:
            raise ValueError(
                f"task: classification needs a {{0, 1}} label, but '{self.label_column}' has {len(values)} distinct values "
                f"(e.g. {sorted(values)[:5]}); a rank / z-score label is a regression target"
            )

    def __repr__(self) -> str:
        return f"{type(self).__name__}(family={self.family!r}, horizon={self.horizon}, task={self.task!r}, n_features={len(self.features)})"
