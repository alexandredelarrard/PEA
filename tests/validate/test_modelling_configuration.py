"""Current fixed training configuration must match the exact saved run independently."""

from __future__ import annotations

from typing import Any

import pytest
from omegaconf import DictConfig, OmegaConf

from src.modelling.utils.artifacts import selection_signature
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
def test_changed_fitting_control_is_rejected(field: str, value: Any) -> None:
    config = OmegaConf.merge(
        make_config(), OmegaConf.create({"model": {"target_type": "zscore", "horizons": [30, 60, 90]}, "train": {"outcome_cutoff": "2021-12-31"}})
    )
    assert isinstance(config, DictConfig)
    signature = selection_signature(config, ["f0", "f1", "sector"], [30, 60, 90])
    training = {"selection_signature": signature, "horizons": [30, 60, 90], "model_types": list(config.model.ensemble)}
    cutoff = {"selection_signature": signature, "feature_cols": ["f0", "f1"], "categorical_cols": ["sector"]}
    findings = []
    modelling._reconcile_configuration(config, training, cutoff, findings)
    assert not findings
    OmegaConf.update(config, field, value, merge=False, force_add=True)
    modelling._reconcile_configuration(config, training, cutoff, findings)
    assert findings
    print(f"SANITY: independent training-configuration reconciliation rejects changed {field}, before any model or source replay.")
