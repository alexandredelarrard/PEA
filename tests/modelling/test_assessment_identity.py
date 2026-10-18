"""Known-truth resume identities; no database access or model fitting."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pandas as pd
import pytest

from src.context import Context
from src.modelling.steps import step_assess_long_short as assessment
from src.modelling.steps.step_assess_long_short import StepAssessLongShort
from src.modelling.utils.assessment import frame_digest, write_json
from tests.modelling.model_fixtures import make_config


@pytest.fixture
def candidate_state(tmp_path: Path) -> tuple[StepAssessLongShort, pd.DataFrame, dict[str, Any]]:
    config = make_config(
        model={
            "target_type": "zscore",
            "cv": {"n_splits": 5, "embargo": None},
            "diagnostics": {"shap_sample": 12},
            "backtest": {"enabled": False},
            "assessment": {
                "horizon": 60,
                "outcome_cutoff": "2021-12-31",
                "evaluation_start": "2022-01-03",
                "test_windows": [["2000-01-01", "2021-12-31"]],
                "prune_fraction": 0.2,
                "max_rounds": 10,
                "patience": 2,
                "read_chunksize": 25000,
                "categorical_features": ["sector"],
                "excluded_columns": ["close"],
                "resume": True,
            },
        },
        train={},
        build_cube={"output": {}},
    )
    step = object.__new__(StepAssessLongShort)
    step._config, step._options, step._output = config, config.model.assessment, tmp_path
    step._log = logging.getLogger(__name__)
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-02", "2020-01-02"]),
            "ticker": ["A", "B"],
            "x": [1.0, 2.0],
            "unused": [3.0, 4.0],
            "y": [0.2, -0.2],
            "label_end": pd.to_datetime(["2020-03-27", "2020-03-27"]),
        }
    )
    step._development_input_digest = frame_digest(panel, ["x", "unused", "y", "label_end"])
    result = {
        "id": "l0",
        "family": "lgbm",
        "features": ["x"],
        "categoricals": [],
        "identity": {
            "development_input_digest": step._development_input_digest,
            "controls": step._controls(),
            "fixed_parameters": step._fixed_parameters(),
        },
    }
    write_json(tmp_path / "_out" / "l0" / "result.json", result)
    return step, panel, result


@pytest.mark.parametrize("control", ["shap_sample", "categorical_features", "excluded_columns", "prune_fraction", "seed", "fixed", "shared"])
def test_candidate_resume_refuses_changed_controls(candidate_state: tuple[StepAssessLongShort, pd.DataFrame, dict[str, Any]], control: str) -> None:
    step, panel, _ = candidate_state
    fit_sources = step._fit_source_identity()
    if control == "shap_sample":
        step._config.model.diagnostics.shap_sample += 1
    elif control in {"categorical_features", "excluded_columns"}:
        step._options[control].append("unused")
    elif control == "prune_fraction":
        step._options.prune_fraction = 0.3
    elif control == "seed":
        step._config.seed += 1
    elif control == "fixed":
        step._config.lgbm.num_leaves += 1
    else:
        step._config.model.weight_decay.enabled = True
    assert step._fit_source_identity() == fit_sources
    with pytest.raises(ValueError, match="identity mismatch"):
        step._candidate("l0", "lgbm", ["x"], [], panel, panel)
    print(f"SANITY: changed {control} refuses cached selection while baseline fit-source identity stays unchanged.")


def _changed_implementation(*args: Any, **kwargs: Any) -> None:
    raise AssertionError("Changed selection implementation must never execute on cache reuse")


@pytest.mark.parametrize("owner,name", [(assessment, "prune_pool"), (assessment, "shap_row_values"), (StepAssessLongShort, "_search")])
def test_candidate_resume_refuses_changed_selection_source(
    candidate_state: tuple[StepAssessLongShort, pd.DataFrame, dict[str, Any]], owner: Any, name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    step, panel, _ = candidate_state
    monkeypatch.setattr(owner, name, _changed_implementation)
    with pytest.raises(ValueError, match="identity mismatch"):
        step._candidate("l0", "lgbm", ["x"], [], panel, panel)
    print(f"SANITY: changed {name} cannot mix cached importance with a new selection implementation.")


@pytest.mark.parametrize("change", ["input", "missing_identity", "changed_identity", "family", "categories"])
def test_candidate_resume_refuses_changed_input_or_cached_identity(
    candidate_state: tuple[StepAssessLongShort, pd.DataFrame, dict[str, Any]], change: str
) -> None:
    step, panel, result = candidate_state
    if change == "input":
        panel.loc[0, "unused"] = 9.0
        step._development_input_digest = frame_digest(panel, ["x", "unused", "y", "label_end"])
    elif change == "missing_identity":
        result.pop("identity")
    elif change == "changed_identity":
        result["identity"]["development_input_digest"] = "stale"
    elif change == "family":
        result["family"] = "random_forest"
    else:
        result["categoricals"] = ["x"]
    write_json(step._output / "_out" / "l0" / "result.json", result)
    with pytest.raises(ValueError, match="identity mismatch"):
        step._candidate("l0", "lgbm", ["x"], [], panel, panel)
    print(f"SANITY: {change} refuses cached candidate, including a changed development feature outside this candidate pool.")


def test_candidate_resume_reuses_unchanged_identity_and_ignores_report_repairs(
    candidate_state: tuple[StepAssessLongShort, pd.DataFrame, dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    step, panel, result = candidate_state
    before = (step._output / "_out" / "l0" / "result.json").read_bytes()
    monkeypatch.setattr(StepAssessLongShort, "_report", _changed_implementation)
    monkeypatch.setattr(StepAssessLongShort, "_evaluation", _changed_implementation)
    assert step._candidate("l0", "lgbm", ["x"], [], panel, panel) == result
    assert step._candidate("l0", "lgbm", ["x"], [], panel.iloc[::-1], panel) == result
    assert (step._output / "_out" / "l0" / "result.json").read_bytes() == before
    print("SANITY: unchanged experiment reuses byte-identical candidate evidence; report/evaluation repairs leave selection identity stable.")


def test_candidate_persists_identity_before_future_reuse(
    candidate_state: tuple[StepAssessLongShort, pd.DataFrame, dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    step, panel, _ = candidate_state
    step._context = cast(Context, SimpleNamespace())
    cv = SimpleNamespace(oos=panel.assign(pred=0.1, fold=1), fold_models=[], folds=[], rounds={})

    class NoFitStep:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def _prepare_training(self) -> None:
            pass

        def _cross_validate(self, *args: Any, **kwargs: Any) -> Any:
            return cv

    def importance(*args: Any, **kwargs: Any) -> dict[str, float]:
        return {"x": 1.0}

    monkeypatch.setattr(assessment, "StepLongShort", NoFitStep)
    monkeypatch.setattr(step, "_importance", importance)
    result = step._candidate("l1", "lgbm", ["x"], [], panel, panel)
    saved = json.loads((step._output / "_out" / "l1" / "result.json").read_text(encoding="utf-8"))
    assert saved["identity"] == result["identity"]
    assert saved["identity"]["development_input_digest"] == step._development_input_digest
    assert saved["identity"]["controls"] == step._controls()
    assert saved["identity"]["fixed_parameters"] == step._fixed_parameters()
    assert step._candidate("l1", "lgbm", ["x"], [], panel, panel) == saved
    print("SANITY: a newly completed candidate persists complete input/selection/shared/fixed identity and reuses it without fitting.")
