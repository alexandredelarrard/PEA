"""Saved fitted contracts remain independent of the current fitting YAML."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest
from omegaconf import DictConfig, OmegaConf

from src.modelling.utils.artifacts import configuration_snapshot
from src.validate.checks import modelling
from tests.modelling.model_fixtures import make_config


@pytest.mark.parametrize(
    "field,value",
    [
        ("train.outcome_cutoff", "2020-12-31"),
        ("model.target_type", "rank"),
        ("model.horizons", [60]),
        ("model.ensemble", ["lgbm"]),
        ("model.cv.test_windows", [["2019-01-01", "2021-12-31"]]),
        ("linear.columns_by_horizon", {60: ["f1"]}),
        ("lgbm.monotonic.features", {"f0": -1}),
        ("random_forest.columns_by_horizon", {90: ["f1"]}),
        ("seed", 9),
    ],
)
def test_current_fitting_changes_do_not_block_saved_models(field: str, value: Any) -> None:
    config = OmegaConf.merge(
        make_config(),
        OmegaConf.create(
            {"model": {"target_type": "zscore", "horizons": [30, 60, 90], "cv": {"n_splits": 5}}, "train": {"outcome_cutoff": "2021-12-31"}}
        ),
    )
    assert isinstance(config, DictConfig)
    training = {
        "workflow": "fresh-assessment-v1",
        "horizons": [30, 60, 90],
        "model_types": list(config.model.ensemble),
        "feature_cols": ["f0", "f1"],
        "categorical_cols": ["sector"],
        "target_type": config.model.target_type,
        "label_column": config.model.label_column,
        "folds": {str(h): list(range(1, 6)) for h in (30, 60, 90)},
    }
    cutoff = {
        **training,
        "target_type": config.model.target_type,
        "label_column": config.model.label_column,
    }
    snapshot = configuration_snapshot(config)
    findings = []
    modelling._reconcile_configuration(config, training, cutoff, findings, snapshot=snapshot)
    assert not findings
    OmegaConf.update(config, field, value, merge=False, force_add=True)
    modelling._reconcile_configuration(config, training, cutoff, findings, snapshot=snapshot)
    assert not findings
    print(f"SANITY: changed current fitting control {field} cannot invalidate an assessment of saved fitted models.")


@pytest.mark.parametrize("field,value", [("target_type", "rank"), ("horizons", [60]), ("ensemble", ["lgbm"]), ("label_column", "wrong")])
def test_saved_configuration_must_match_actual_declared_models(field: str, value: Any) -> None:
    config = make_config(model={"target_type": "zscore", "horizons": [60], "cv": {"n_splits": 3}})
    training = {
        "workflow": "fresh-assessment-v1",
        "horizons": [60],
        "model_types": list(config.model.ensemble),
        "feature_cols": ["f0"],
        "categorical_cols": [],
        "target_type": "zscore",
        "label_column": "y",
        "folds": {"60": [1, 2, 3]},
    }
    cutoff = deepcopy(training)
    snapshot = configuration_snapshot(config)
    snapshot["model"][field] = [30] if field == "horizons" else value
    findings = []
    modelling._reconcile_configuration(config, training, cutoff, findings, snapshot=snapshot)
    assert findings
    print(f"SANITY: saved {field} differs from the actual fitted inventory and is independently rejected.")
