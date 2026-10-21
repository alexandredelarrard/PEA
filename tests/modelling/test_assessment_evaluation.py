"""Replay serialized known-truth coefficients through fresh assessment without estimator fitting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.data_store.schema import Tables
from src.modelling.steps.step_assess_long_short import StepAssessLongShort
from src.modelling.steps.step_long_short import StepLongShort
from src.modelling.transformers import BaseModel, LinearRegression
from src.modelling.utils.metrics import daily_ic
from src.validate.checks.modelling import check_modelling, check_training_readiness
from tests.modelling.test_step_long_short import _setup


def test_assessment_reads_original_models_and_recomputes_every_score_without_fitting(
    tmp_path: Path, sqlite_store: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    trainer, spy, config = _setup(
        tmp_path,
        sqlite_store,
        model={
            "target_type": "zscore",
            "horizons": [60],
            "ensemble": ["elasticnet"],
            "cv": {"n_splits": 3},
            "assessment": {"read_chunksize": 5000},
            "diagnostics": {"shap_sample": 20, "dependence_sample": 10, "pdp_grid": 3, "pdp_background": 10, "rolling_window": 3},
        },
    )
    cube = sqlite_store.load(Tables.cube).rename(columns={"target_rank_h60": "target_zscore_h60"})
    cube.loc[cube.ticker == "T000", "target_zscore_h60"] = np.nan
    sqlite_store.drop(Tables.cube)
    sqlite_store.replace(Tables.cube, cube)
    dates = sorted(cube.date.unique())
    config.train.end_date = str(pd.Timestamp(dates[-180]).date())
    config.train.outcome_cutoff = str(pd.Timestamp(dates[-120]).date())
    config.model.assessment.evaluation_start = str(pd.Timestamp(dates[-118]).date())

    def known_member(horizon: int, train: pd.DataFrame, valid: pd.DataFrame | None, rounds: int | None = None) -> dict:
        member = LinearRegression(trainer._context, config, "elasticnet", horizon)
        member.features = ["f2", "f0", "f1"]
        member.mean_, member.std_, member.coef_ = np.zeros(3), np.ones(3), np.array([-0.2, 0.6, 0.3])
        member.intercept_, member.model = 0.25, "fitted"
        return {"elasticnet": member}

    monkeypatch.setattr(trainer, "_fit_members", known_member)
    result = trainer.run_train()
    assert check_training_readiness(out=result.run_dir, config=config).status == "pass"
    before = {path: path.read_bytes() for path in result.run_dir.rglob("*.pkl")}

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("Assessment attempted fitting or CV")

    monkeypatch.setattr(BaseModel, "fit", forbidden)
    monkeypatch.setattr(LinearRegression, "_fit", forbidden)
    monkeypatch.setattr(StepLongShort, "_cross_validate", forbidden)
    original_predict = BaseModel.predict
    calls = []

    def record_predict(self: Any, panel: pd.DataFrame) -> pd.Series:
        calls.append(len(panel))
        return original_predict(self, panel)

    monkeypatch.setattr(BaseModel, "predict", record_predict)
    spy.loads.clear()
    report = StepAssessLongShort(trainer._context, config).run()
    output = result.run_dir / "assessment/h60"
    first_scores = pd.read_parquet(output / "fold1/scores.parquet")
    predictions = pd.read_parquet(output / "predictions.parquet")
    assert set(predictions.fold) == {0, 1, 2, 3}
    assert (predictions.ticker == "T000").any()
    assert predictions.loc[predictions.ticker == "T000", "y"].isna().all()
    assert predictions.loc[predictions.ticker == "T000", "ensemble"].notna().all()
    first_label = next(i for i, row in enumerate(spy.loads) if any(column.startswith("target_") for column in row["columns"]))
    assert first_label == 4 and all(not row["where"] for row in spy.loads[:first_label])
    for fold, audit in enumerate(result.cv[60].folds, 1):
        panel = predictions.loc[predictions.fold == fold]
        metrics = daily_ic(panel, panel.ensemble, horizon=60)
        assert metrics["mean_ic"] == pytest.approx(audit["mean_ic"])
        assert metrics["ic_ir"] == pytest.approx(audit["ic_ir"])
    checked = check_modelling(trainer._context, out=result.run_dir, config=config)
    assert checked.status == "pass", checked.to_json()
    changed = predictions.copy()
    changed["elasticnet"] += 1
    changed.to_parquet(output / "predictions.parquet", index=False)
    rejected = check_modelling(trainer._context, out=result.run_dir, config=config)
    assert rejected.status == "fail", "Rank-preserving changes must still disagree with saved native scores"
    predictions.to_parquet(output / "predictions.parquet", index=False)
    inputs = pd.read_parquet(output / "fold1/elasticnet/x.parquet")
    inputs.loc[0, "f0"] += 1
    inputs.to_parquet(output / "fold1/elasticnet/x.parquet", index=False)
    rejected = check_modelling(trainer._context, out=result.run_dir, config=config)
    assert rejected.status == "fail", "Explanations must use the independently source-checked sample"
    assert any("source-checked sample" in str(finding) for finding in rejected.findings)
    count = len(calls)
    corrupted = first_scores.copy()
    corrupted["ensemble"] = 999
    corrupted.to_parquet(output / "fold1/scores.parquet", index=False)
    config.model.ensemble = ["lgbm"]
    config.model.horizons = [30, 90]
    config.linear.columns = ["unrelated_current_feature"]
    assert StepAssessLongShort(trainer._context, config, run_id=result.run_id).run() == report
    assert len(calls) > count
    pd.testing.assert_frame_equal(pd.read_parquet(output / "fold1/scores.parquet"), first_scores)
    assert before == {path: path.read_bytes() for path in result.run_dir.rglob("*.pkl")}
    assert json.loads((result.run_dir / "assessment/metadata.json").read_text())["complete"]
    print(
        "SANITY: original members score every title before outcomes, reproduce fold IC/IR, pass independent source/native/effect validation and recompute corrupted old scores without fitting."
    )
