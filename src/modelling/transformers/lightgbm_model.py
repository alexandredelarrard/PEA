"""
lightgbm_model.py  (src/modelling/transformers/lightgbm_model.py)
-----------------------------------------------------------------
`LightGBMModel`: the gradient-boosted member (`lgbm:` block), plus `train_booster`, the one
LightGBM training routine both tree families go through.

Training is deterministic (`seed`, `deterministic`, `force_row_wise`): multithreaded LightGBM
otherwise sums gradients in a nondeterministic order and early stopping flips between rounds.
Early stopping uses the chronological validation tail, with `eval_metric: rmse` (built-in) or
`ic` (mean daily cross-sectional Spearman IC, the ranking metric the strategy uses).
Categoricals are native LightGBM categorical splits on the integer codes of
`coerce_categoricals`; monotone constraints come from `lgbm.monotonic` aligned to the trained
feature order. `task: classification` switches to the `binary` objective on a {0, 1} label.
"""

from __future__ import annotations

from functools import partial
from typing import Any, cast

import lightgbm as lgb
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from src.modelling.transformers.base import TASK_CLASSIFICATION, BaseModel
from src.modelling.utils.cv import time_decay_weights
from src.modelling.utils.features import build_monotone_constraints, coerce_categoricals, parse_monotone_feature_map

EARLY_STOPPING_ROUNDS = 40
DEFAULT_SEED = 42  # overridden by the pipeline's global `seed`


def _group_sizes(panel: pd.DataFrame) -> list[int]:
    return panel.groupby("date", sort=False).size().tolist()


def _graded_labels(panel: pd.DataFrame, label_name: str) -> np.ndarray:
    # lambdarank expects integer relevance in [0, n_levels); 31 levels -> 0..30
    return np.clip((panel[label_name].to_numpy() * 30).round().astype(int), 0, 30)


def _build_dataset(
    params: dict,
    panel: pd.DataFrame,
    feats: list,
    label_name: str,
    weights: np.ndarray | None = None,
    categorical_features: list[str] | None = None,
) -> lgb.Dataset:
    # With categoricals, a DataFrame keeps the int category codes as int (native categorical
    # splits); otherwise the fast all-float numpy path.
    if categorical_features:
        cat = set(categorical_features)
        x = coerce_categoricals(panel[feats], categorical_features)
        num = [f for f in feats if f not in cat]
        if num:
            x[num] = x[num].astype("float32")
        for c in categorical_features:
            x[c] = x[c].astype("int32")
        kw: dict = {"feature_name": feats, "categorical_feature": list(categorical_features)}
    else:
        x = panel[feats].to_numpy(dtype="float32")
        kw = {"feature_name": feats}
    if weights is not None:
        kw["weight"] = weights
    if params["objective"] == "lambdarank":
        return lgb.Dataset(x, label=_graded_labels(panel, label_name), group=_group_sizes(panel), **kw)
    return lgb.Dataset(x, label=panel[label_name].to_numpy(dtype="float32"), **kw)


def _ic_eval_factory(val_dates: np.ndarray, val_label: np.ndarray, min_names: int = 5):
    """LightGBM custom eval: mean DAILY cross-sectional Spearman IC on the validation set
    (higher is better). Label ranks are precomputed once; each round ranks only the predictions."""
    dates = np.asarray(val_dates)
    y = np.asarray(val_label, dtype=float)
    if len(dates) == 0:
        groups, y_ranks = [], []
    else:
        _, inv = np.unique(dates, return_inverse=True)
        groups = [np.where(inv == g)[0] for g in range(int(inv.max()) + 1)]
        y_ranks = [pd.Series(y[idx]).rank().to_numpy() for idx in groups]

    return partial(_daily_ic_eval, groups=groups, y_ranks=y_ranks, min_names=min_names)


def _rank_ic(preds: np.ndarray, y_ranks: np.ndarray) -> float | None:
    """Spearman IC of one date's predictions against its precomputed label ranks; None when
    either side is constant or the correlation is not finite."""
    pr = pd.Series(preds).rank().to_numpy()
    if not (pr.std() > 0 and y_ranks.std() > 0):
        return None
    ic = float(np.corrcoef(pr, y_ranks)[0, 1])
    return ic if np.isfinite(ic) else None


def _daily_ic_eval(preds: Any, _data: Any, *, groups: list[np.ndarray], y_ranks: list[np.ndarray], min_names: int) -> tuple[str, float, bool]:
    """LightGBM feval: mean daily IC over dates with more than `min_names` rows (0.0 when none)."""
    preds = np.asarray(preds, dtype=float)
    ics = [_rank_ic(preds[idx], yr) for idx, yr in zip(groups, y_ranks, strict=False) if len(idx) > min_names]
    finite = [ic for ic in ics if ic is not None]
    return "daily_ic", (float(np.mean(finite)) if finite else 0.0), True


def train_booster(
    panel: pd.DataFrame,
    feats: list,
    label_name: str = "y",
    params: dict | None = None,
    num_boost_round: int = 400,
    valid_panel: pd.DataFrame | None = None,
    early_stopping_rounds: int = EARLY_STOPPING_ROUNDS,
    half_life_years: float | None = None,
    eval_metric: str = "rmse",
    categorical_features: list[str] | None = None,
    sample_weight: np.ndarray | None = None,
) -> lgb.Booster:
    """Fit one LightGBM booster. `params` update the deterministic regression defaults below.
    With `half_life_years`, TRAINING rows get exponential time-decay weights (most recent = 1.0);
    validation stays unweighted so early stopping reflects recent out-of-sample fit. An explicit
    `sample_weight` wins over the decay weights."""
    default = dict(
        objective="regression",
        metric="rmse",
        learning_rate=0.03,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=10,
        lambda_l1=0.0,
        lambda_l2=5.0,
        verbosity=-1,
        n_jobs=-2,
        seed=DEFAULT_SEED,
        deterministic=True,
        force_row_wise=True,
    )
    if params:
        default.update(params)

    if sample_weight is not None:
        train_w = np.asarray(sample_weight, dtype=float)
    else:
        train_w = time_decay_weights(panel["date"], half_life_years) if half_life_years is not None else None
    train_set = _build_dataset(default, panel, feats, label_name, train_w, categorical_features=categorical_features)
    valid_sets = []
    callbacks = []
    feval = None
    if valid_panel is not None and not valid_panel.empty:
        valid_sets = [_build_dataset(default, valid_panel, feats, label_name, categorical_features=categorical_features)]
        if eval_metric == "ic":
            default["metric"] = "None"  # early stopping keys on the custom IC only
            feval = _ic_eval_factory(valid_panel["date"].to_numpy(), valid_panel[label_name].to_numpy())
        callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping_rounds))

    booster = lgb.train(default, train_set, num_boost_round=num_boost_round, valid_sets=valid_sets or None, feval=feval, callbacks=callbacks or None)
    cast(Any, booster).feature_names = feats
    return booster


class LightGBMModel(BaseModel):
    """Gradient-boosted trees (`lgbm:` block), early-stopped on the validation tail."""

    config_key = "lgbm"
    supports_shap = True

    @classmethod
    def lgbm_block(cls, config: DictConfig) -> Any:
        """The `lgbm:` block — also the source of the random forest's categoricals, monotone
        map and column fallback."""
        return config.get("lgbm") or {}

    @classmethod
    def configured_categoricals(cls, config: DictConfig) -> list[str]:
        block = cls.lgbm_block(config)
        return list(block.categoricals) if (block and block.get("categoricals") is not None) else []

    def _monotone_constraints(self, feats: list[str]) -> list[int] | None:
        """`lgbm.monotonic` aligned to `feats`. WARNs only for a constrained name that is not
        even in `lgbm.columns` (a typo); a listed name that is absent from the cube is inactive."""
        block = self.lgbm_block(self._config)
        mono = block.get("monotonic") if block else None
        if not mono or not mono.get("enabled", False):
            return None
        feature_map = parse_monotone_feature_map(mono.get("features"))
        if not feature_map:
            self._log.warning("lgbm.monotonic.enabled but no features configured")
            return None
        constraints = build_monotone_constraints(feats, feature_map)
        allow = set(LightGBMModel.configured_columns(self._config, None))
        absent = [f for f in feature_map if f not in feats]
        typos = [f for f in absent if f not in allow]
        if typos:
            self._log.warning("Monotone constraint on feature(s) not in inputs.columns (typo?): %s", typos)
        elif absent:
            self._log.info("%d monotone constraint(s) inactive (feature not in the current cube; applies after a rebuild)", len(absent))
        return constraints

    def _objective(self, params: dict) -> dict:
        if self.task == TASK_CLASSIFICATION:
            params.update({"objective": "binary", "metric": "binary_logloss"})
        return params

    def _fit(self, train: pd.DataFrame, valid: pd.DataFrame | None) -> lgb.Booster:
        c = self.lgbm_block(self._config)
        params = {
            "learning_rate": c.get("learning_rate"),
            "max_depth": c.get("max_depth"),
            "num_leaves": c.get("num_leaves", 31),
            # subsample (= bagging_fraction) only takes effect when bagging_freq > 0
            "subsample": c.get("subsample"),
            "bagging_freq": c.get("bagging_freq", 0),
            "colsample_bytree": c.get("colsample_bytree"),
            "min_child_samples": c.get("min_child_samples"),
            "lambda_l1": c.get("lambda_l1"),
            "lambda_l2": c.get("lambda_l2"),
            "seed": int(self._config.get("seed", DEFAULT_SEED)),
            "deterministic": True,
            "force_row_wise": True,
        }
        kw: dict[str, Any] = {
            "params": self._objective(params),
            "num_boost_round": c.get("num_boost_round"),
            "early_stopping_rounds": int(c.get("early_stopping_rounds", EARLY_STOPPING_ROUNDS)),
            "eval_metric": c.get("eval_metric") or "rmse",
        }
        monotone = self._monotone_constraints(self.features)
        if monotone is not None:
            kw["params"]["monotone_constraints"] = monotone
        if self.half_life_years is not None:
            kw["half_life_years"] = self.half_life_years
        return train_booster(train, self.features, self.label_column, valid_panel=valid, categorical_features=self.categoricals or None, **kw)

    def _predict(self, panel: pd.DataFrame) -> np.ndarray:
        # the frame as-is (not a float32 array): integer category codes keep the categorical
        # bins learned at training time
        return np.asarray(self.model.predict(panel[self.features]))

    def importance(self) -> pd.Series:
        """LightGBM gain importance, in trained-feature order."""
        return pd.Series(dict(zip(self.features, self.model.feature_importance(importance_type="gain"), strict=False)), dtype=float)

    # The booster travels as its LightGBM text model (the format the `.txt` artifacts used), not as
    # a pickled `Booster` object: that pickle references lightgbm / numpy internals and breaks when
    # the trainer (Airflow, Python 3.12) and the scorer (app / CLI) run different library versions.
    def __getstate__(self) -> dict[str, Any]:
        state = super().__getstate__()
        if isinstance(self.model, lgb.Booster):
            state["model"] = self.model.model_to_string()
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        super().__setstate__(state)
        if isinstance(self.model, str):
            self.model = lgb.Booster(model_str=self.model)
            cast(Any, self.model).feature_names = self.features
