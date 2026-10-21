"""Shared known-truth fixtures for the transformer tests: a config factory with the three family
blocks and synthetic daily cross-sections whose label is a rank of a weak linear signal (plus a
categorical shift and sparse NaNs where a test asks for them)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from src.modelling.utils.artifacts import configuration_snapshot, write_metadata
from src.modelling.utils.assessment import write_json

SEED = 4325
NUMERIC = [f"f{j}" for j in range(6)]


def make_config(**overrides: Any) -> DictConfig:
    """Config with `model`, `lgbm`, `random_forest`, `linear` blocks; `overrides` deep-merge."""
    base = {
        "seed": SEED,
        "model": {"label_column": "y", "ensemble": ["elasticnet", "lgbm", "random_forest"], "weight_decay": {"enabled": False, "half_life_years": 5}},
        "lgbm": {
            "task": "regression",
            "learning_rate": 0.05,
            "max_depth": 4,
            "num_leaves": 15,
            "subsample": 0.8,
            "bagging_freq": 1,
            "colsample_bytree": 0.8,
            "min_child_samples": 20,
            "lambda_l1": 1.0,
            "lambda_l2": 1.0,
            "num_boost_round": 120,
            "early_stopping_rounds": 20,
            "eval_metric": "rmse",
            "categoricals": ["sector"],
            "monotonic": {"enabled": True, "features": [{"f0": 1}, {"f2": -1}]},
            "columns": [*NUMERIC, "sector"],
        },
        "random_forest": {
            "task": "regression",
            "num_boost_round": 40,
            "bagging_fraction": 0.6,
            "feature_fraction": 0.6,
            "max_depth": 6,
            "num_leaves": 31,
        },
        "linear": {"task": "regression", "alpha": 0.001, "l1_ratio": 0.5, "max_iter": 500, "tol": 1e-6, "columns": ["f2", "f0", "f1"]},
    }
    return OmegaConf.merge(OmegaConf.create(base), OmegaConf.create(overrides))  # type: ignore[return-value]


def ctx(store: Any = None) -> Any:
    """The context surface a transformer touches (`store` only), standing in for a `Context`."""
    return SimpleNamespace(store=store)


def completed_run(root: Path, run_id: str, completed_at: str, *, folds: int = 3, horizons: list[int] | None = None) -> tuple[Path, DictConfig]:
    """Build known-truth manifest state without fitting estimators or requiring market data."""
    horizons = horizons or [60]
    config = make_config(
        model={
            "horizons": horizons,
            "target_type": "zscore",
            "cv": {"n_splits": folds, "embargo": None},
            "assessment": {"output_dir": str(root), "evaluation_start": "2022-01-03", "read_chunksize": 200},
            "diagnostics": {"shap_sample": 30, "dependence_sample": 10, "pdp_grid": 30, "pdp_background": 20, "rolling_window": 252},
        },
        train={"start_date": "1999-01-01", "end_date": "2021-12-31", "outcome_cutoff": "2021-12-31"},
    )
    run = root / run_id
    write_json(run / "training/configuration.json", configuration_snapshot(config))
    state = {
        "workflow": "fresh-assessment-v1",
        "run_id": run_id,
        "horizons": horizons,
        "model_types": list(config.model.ensemble),
        "feature_cols": ["x"],
        "categorical_cols": [],
        "target_type": "zscore",
        "label_column": "y",
        "training_complete": True,
        "training_completed_at_utc": completed_at,
        "training_as_of": "2021-12-31",
        "observation_frontier": "2024-12-31",
        "folds": {str(h): list(range(1, folds + 1)) for h in horizons},
    }
    write_json(run / "metadata.json", state)
    write_metadata(
        run / "cutoff",
        {
            "run_id": run_id,
            "role": "cutoff",
            "full_history": False,
            "horizons": horizons,
            "model_types": state["model_types"],
            "feature_cols": ["x"],
            "categorical_cols": [],
            "training_as_of": "2021-12-31",
        },
    )
    return run, config


def signal_panel(n_days: int = 160, n_tickers: int = 60, seed: int = 0, sparse: bool = True) -> pd.DataFrame:
    """Daily cross-sections: y = per-day rank of 0.6 f0 + 0.35 f1 - 0.25 f2 + sector shift +
    noise; `sector` is an int16 code; with `sparse`, f5 is NaN on ~40 % of rows."""
    rng = np.random.default_rng(seed)
    frames = []
    for d in pd.bdate_range("2016-01-01", periods=n_days):
        x = rng.normal(0, 1, (n_tickers, len(NUMERIC)))
        sector = np.arange(n_tickers) % 5
        raw = 0.6 * x[:, 0] + 0.35 * x[:, 1] - 0.25 * x[:, 2] + np.where(sector == 3, 0.8, 0.0) + rng.normal(0, 1.2, n_tickers)
        block = pd.DataFrame(x, columns=NUMERIC)
        if sparse:
            block.loc[rng.random(n_tickers) < 0.4, "f5"] = np.nan
        block["sector"] = sector.astype("int16")
        block.insert(0, "y", pd.Series(raw).rank(pct=True).to_numpy())
        block.insert(0, "ticker", [f"T{i:03d}" for i in range(n_tickers)])
        block.insert(0, "date", d)
        frames.append(block)
    return pd.concat(frames, ignore_index=True)
