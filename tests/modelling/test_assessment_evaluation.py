"""Saved original models are the only source of assessment predictions and explanations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from src.data_store.schema import Tables
from src.modelling.steps import step_assess_long_short as assessment
from src.modelling.steps.step_assess_long_short import StepAssessLongShort
from src.modelling.steps.step_long_short import StepLongShort
from src.modelling.transformers import BaseModel, LightGBMModel
from tests.modelling.test_step_long_short import _setup


def test_assessment_reads_original_folds_without_fitting(tmp_path: Path, sqlite_store: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    trainer, spy, config = _setup(
        tmp_path,
        sqlite_store,
        save=False,
        model={
            "target_type": "zscore",
            "horizons": [30],
            "cv": {"n_splits": 5},
            "assessment": {"read_chunksize": 5000},
            "diagnostics": {"shap_sample": 20, "dependence_sample": 10, "pdp_grid": 3, "pdp_background": 10, "rolling_window": 3},
        },
        lgbm={"num_boost_round": 20},
        random_forest={"num_boost_round": 5},
        build_cube={"output": {"save_predictions": False, "save_cv_results": False, "save_signal": False}},
    )
    cube = sqlite_store.load(Tables.cube).rename(columns={"target_rank_h30": "target_zscore_h30"})
    sqlite_store.drop(Tables.cube)
    sqlite_store.replace(Tables.cube, cube)
    dates = sorted(cube.date.unique())
    config.train.end_date = str(pd.Timestamp(dates[-120]).date())
    config.train.outcome_cutoff = str(pd.Timestamp(dates[-90]).date())
    config.model.assessment.evaluation_start = str(pd.Timestamp(dates[-88]).date())
    result = trainer.run()
    assert result.run_dir is not None
    before = {path: path.read_bytes() for path in result.run_dir.rglob("*.pkl")}

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("Assessment attempted fitting or CV")

    for name in ["run_train", "_fit_members", "_fit_final", "_cross_validate"]:
        monkeypatch.setattr(StepLongShort, name, forbidden)
    monkeypatch.setattr(BaseModel, "fit", forbidden)
    monkeypatch.setattr(LightGBMModel, "fit_fixed_rounds", forbidden)

    def renderer(run: Path, cfg: Any) -> Path:
        path = run / "report.html"
        path.write_text("<html>renderer contract stub</html>", encoding="utf-8")
        return path

    monkeypatch.setattr(assessment, "render_assessment", renderer)
    spy.loads.clear()
    report = StepAssessLongShort(context=trainer._context, config=config).run()
    assert report == result.run_dir / "report.html"
    assert before == {path: path.read_bytes() for path in result.run_dir.rglob("*.pkl")}
    output = result.run_dir / "assessment/h30"
    predictions = pd.read_parquet(output / "predictions.parquet")
    assert set(predictions.fold) == {0, 1, 2, 3, 4, 5}
    assert {"elasticnet", "lgbm", "random_forest", "ensemble", "y", "label_end"} <= set(predictions)
    assert len(list(output.glob("fold*/elasticnet/curves.json"))) == 6
    assert len(list(output.glob("fold*/lgbm/shap.parquet"))) == 6
    first_label = next(i for i, row in enumerate(spy.loads) if any(c.startswith("target_") for c in row["columns"]))
    assert first_label == 6, "all six complete feature-only universes must be scored before any target read"
    assert all(not row["where"] for row in spy.loads[:first_label])
    assert not (tmp_path / "models").exists()
    print(
        "SANITY: assessment loads five original folds plus cutoff, scores feature-only before labels, persists every family effect and never fits or changes model bytes; HTML rendering tested separately."
    )
