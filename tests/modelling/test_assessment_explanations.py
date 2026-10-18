"""Actual configured explanation path with deterministic score and attribution identities."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from src.modelling.transformers import assessment_diagnostics
from src.modelling.transformers.base import BaseModel
from src.utils.config import read_config


@pytest.mark.parametrize("pdp", [True, False], ids=["configured-pdp", "shap-only"])
def test_explain_tree_uses_actual_config_and_preserves_score_artifacts(pdp: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = read_config("./configs")
    assert config.model.diagnostics.pdp_grid == 30 and config.model.diagnostics.pdp_background == 2000
    assert "pdp_grid_points" not in config.model.diagnostics
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2021-09-20"] * 4 + ["2021-09-21"] * 4),
            "ticker": ["AAPL", "JPM", "O", "XOM"] * 2,
            "x": [0.1234567890123, 1.0, 2.0, 3.0, 0.5, 1.5, 2.5, 3.5],
            "sector": [0, 1, 2, 0, 1, 2, 0, 1],
        }
    )
    original = panel.copy(deep=True)
    matrix = panel[["x", "sector"]].to_numpy(dtype="float64")
    phi = np.column_stack([2 * matrix[:, 0], -matrix[:, 1]])
    base = np.full(len(panel), 0.25)
    predicted = phi.sum(axis=1) + base
    calls: list[tuple[int, int, int, bool, int]] = []

    def native_predict(values: np.ndarray, *, pred_contrib: bool = False) -> np.ndarray:
        np.testing.assert_array_equal(values, matrix)
        return np.column_stack([phi, base]) if pred_contrib else predicted.copy()

    def member_predict(frame: pd.DataFrame) -> pd.Series:
        pd.testing.assert_frame_equal(frame, original)
        return pd.Series(predicted, index=frame.index)

    def shap_values(booster: Any, values: np.ndarray, features: list[str], *, sample: int, logger: logging.Logger) -> tuple[np.ndarray, np.ndarray]:
        assert booster is model and features == ["x", "sector"] and sample == 0
        np.testing.assert_array_equal(values, matrix)
        return phi.copy(), np.arange(len(values))

    def pdp_values(
        booster: Any, values: np.ndarray, feature: int, grid_points: int, sample: int, *, categorical: bool, seed: int
    ) -> tuple[np.ndarray, np.ndarray]:
        assert booster is model and pdp
        np.testing.assert_array_equal(values, matrix)
        calls.append((feature, grid_points, sample, categorical, seed))
        grid = np.asarray([0.0, 1.0])
        means = 2 * grid - matrix[:, 1].mean() + 0.25 if feature == 0 else 2 * matrix[:, 0].mean() - grid + 0.25
        return grid, means

    model = SimpleNamespace(
        params={"monotone_constraints": [1, 0]},
        predict=native_predict,
        dump_model=lambda **_: {"average_output": False, "objective": "regression"},
    )
    member = cast(BaseModel, SimpleNamespace(features=["x", "sector"], categoricals=["sector"], family="lgbm", model=model, predict=member_predict))
    monkeypatch.setattr(assessment_diagnostics, "shap_row_values", shap_values)
    monkeypatch.setattr(assessment_diagnostics, "partial_dependence", pdp_values)

    result = assessment_diagnostics.explain_tree(member, panel, tmp_path, config, logging.getLogger(__name__), pdp=pdp)

    assert calls == ([(0, 30, 2000, False, int(config.seed)), (1, 30, 2000, True, int(config.seed))] if pdp else [])
    assert result["rows"] == result["background_rows"] == len(panel)
    assert result["additivity_max_error"] == 0 and result["unit_scale"] == 1
    stored = json.loads((tmp_path / "curves.json").read_text(encoding="utf-8"))
    assert stored.keys() == result.keys()
    for index, feature in enumerate(member.features):
        curve = result["features"][feature]
        assert curve["importance"] == pytest.approx(np.abs(phi[:, index]).mean())
        assert curve["constraint"] == [1, 0][index]
        np.testing.assert_array_equal(stored["features"][feature]["binned"]["mean"], curve["binned"]["mean"])
        if pdp:
            np.testing.assert_array_equal(stored["features"][feature]["pdp"]["x"], [0.0, 1.0])
            np.testing.assert_array_equal(stored["features"][feature]["pdp"]["mean"], curve["pdp"]["mean"])
            assert np.diff(curve["pdp"]["mean"])[0] == pytest.approx([2, -1][index])
        else:
            assert stored["features"][feature]["pdp"]["x"] is None and stored["features"][feature]["pdp"]["mean"] is None
    saved_shap = pd.read_parquet(tmp_path / "shap.parquet")
    pd.testing.assert_frame_equal(pd.read_parquet(tmp_path / "x.parquet"), original)
    pd.testing.assert_frame_equal(saved_shap[["date", "ticker"]], original[["date", "ticker"]])
    np.testing.assert_array_equal(saved_shap[["x", "sector"]].to_numpy(), phi)
    np.testing.assert_array_equal(saved_shap[["x", "sector", "base"]].sum(axis=1), saved_shap.prediction)
    np.testing.assert_array_equal(saved_shap.prediction, predicted)
    pd.testing.assert_frame_equal(panel, original)
    print(
        f"SANITY: actual config pdp={pdp} forwards the canonical grid/background/seed; scores, additive SHAP, signed PDP and persisted artifacts agree."
    )
