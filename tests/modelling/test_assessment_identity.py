"""Assessment cache identity covers diagnostics, saved models and float64 scoring inputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from omegaconf import OmegaConf

from src.modelling.steps.step_assess_long_short import StepAssessLongShort


@pytest.mark.parametrize(
    "field,value",
    [
        ("seed", 8),
        ("evaluation_start", "2022-02-01"),
        ("shap_sample", 121),
        ("dependence_sample", 11),
        ("pdp_grid", 31),
        ("pdp_background", 21),
        ("rolling_window", 253),
    ],
)
def test_assessment_identity_refuses_changed_diagnostic_controls(field: str, value: Any) -> None:
    step = object.__new__(StepAssessLongShort)
    step._config = OmegaConf.create(
        {
            "seed": 7,
            "model": {
                "diagnostics": {"shap_sample": 120, "dependence_sample": 10, "pdp_grid": 30, "pdp_background": 20, "rolling_window": 252},
                "assessment": {"evaluation_start": "2022-01-03"},
            },
        }
    )
    step._options = step._config.model.assessment
    training = {
        "run_id": "20261018_42",
        "cutoff_metadata": {"sha256": "one"},
        "folds": {"30": {"1": {"sha256": "two"}}},
        "calendar_file": {"sha256": "three"},
    }
    before = step._identity(training)
    if field == "seed":
        step._config.seed = value
    elif field == "evaluation_start":
        step._options.evaluation_start = value
    else:
        step._config.model.diagnostics[field] = value
    assert step._identity(training) != before
    assert all(Path(name).name != "assessment_report.py" for name in before["source_hashes"])
    print(f"SANITY: changing {field} invalidates saved analysis, while the offline renderer can be repaired without approving stale diagnostics.")


def test_retired_search_and_fitting_methods_are_absent() -> None:
    assert all(not hasattr(StepAssessLongShort, name) for name in ("_search", "_candidate", "_development", "_fit_selected", "_runtime_config"))
    print("SANITY: the operational assessment class has no feature search, candidate, development trainer or selected-refit path.")
