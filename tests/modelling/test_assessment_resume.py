"""Verified reuse refuses changed float64 inputs, original model bytes and explanation files."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from src.modelling.steps import step_assess_long_short as assessment
from src.modelling.steps.step_assess_long_short import StepAssessLongShort
from src.modelling.transformers.linear_regression import LinearRegression
from src.modelling.utils.artifacts import member_inventory, member_path, save_member, write_metadata


@pytest.fixture
def scored_window(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[StepAssessLongShort, pd.DataFrame, dict[str, Any]]:
    step = object.__new__(StepAssessLongShort)
    step._output, step._log, step._store = tmp_path, logging.getLogger(__name__), assessment._ReadOnlyStore(object())
    step._config = OmegaConf.create(
        {"seed": 7, "model": {"diagnostics": {"shap_sample": 4, "dependence_sample": 4, "pdp_grid": 3, "pdp_background": 4, "rolling_window": 2}}}
    )
    step._options = OmegaConf.create({"read_chunksize": 10})
    member = object.__new__(LinearRegression)
    member.features, member.categoricals = ["x", "missing"], []
    member.family, member.horizon, member.task = "elasticnet", 60, "regression"
    member.coef_, member.mean_, member.std_ = np.array([2.0, 0.0]), np.zeros(2), np.ones(2)
    member.intercept_, member.model = 0.5, "fitted"
    directory = tmp_path / "cutoff"
    save_member(member, member_path(directory, 60, "elasticnet"))
    write_metadata(directory, {"horizons": [60], "model_types": ["elasticnet"], "member_files": member_inventory(directory, [60], ["elasticnet"])})
    panel = pd.DataFrame(
        {"date": pd.to_datetime(["2022-01-03"] * 4), "ticker": list("ABCD"), "x": [1.000000000001, 2.0, 3.0, 4.0], "missing": np.nan}
    )
    monkeypatch.setattr(assessment, "load_frame", lambda *args, **kwargs: panel.copy())
    state: dict[str, Any] = {"windows": {}}
    step._score_window(60, 0, panel.date.min(), panel.date.max(), 4, {"model_types": ["elasticnet"]}, state)
    return step, panel, state


def test_matching_window_resume_does_not_predict_or_explain(
    scored_window: tuple[StepAssessLongShort, pd.DataFrame, dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    step, panel, state = scored_window

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("A complete matching window must reuse its verified artifacts")

    monkeypatch.setattr(assessment, "ensemble_predict", forbidden)
    monkeypatch.setattr(step, "_explain_member", forbidden)
    before = {path: path.read_bytes() for path in step._output.rglob("*") if path.is_file()}
    result = step._score_window(60, 0, panel.date.min(), panel.date.max(), 4, {"model_types": ["elasticnet"]}, state)
    assert len(result) == 4
    assert before == {path: path.read_bytes() for path in step._output.rglob("*") if path.is_file()}
    print("SANITY: matching saved window reuses scores and explanations byte for byte without prediction or explanation work.")


@pytest.mark.parametrize("changed", ["float64_input", "model_bytes", "sample_bytes", "curve_bytes", "metadata_bytes"])
def test_window_resume_rejects_changed_inputs_or_artifacts(
    changed: str, scored_window: tuple[StepAssessLongShort, pd.DataFrame, dict[str, Any]]
) -> None:
    step, panel, state = scored_window
    if changed == "float64_input":
        old = panel.x.to_numpy(dtype=float)[0]
        panel.loc[0, "x"] = old + 1e-12
        assert np.float32(old) == np.float32(panel.x.to_numpy(dtype=float)[0])
    else:
        relative = {
            "model_bytes": "cutoff/model_h60_elasticnet.pkl",
            "sample_bytes": "assessment/h60/fold0/sample.parquet",
            "curve_bytes": "assessment/h60/fold0/elasticnet/curves.json",
            "metadata_bytes": "assessment/h60/fold0/metadata.json",
        }[changed]
        with (step._output / relative).open("ab") as handle:
            handle.write(b"changed")
    with pytest.raises(ValueError, match="changed|identity"):
        step._score_window(60, 0, panel.date.min(), panel.date.max(), 4, {"model_types": ["elasticnet"]}, state)
    print(f"SANITY: {changed} cannot silently reuse a stale saved score/explanation.")
