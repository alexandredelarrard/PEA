"""The report contains only three offline plot areas and all retained effects at every horizon."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pandas as pd
import pytest
from omegaconf import OmegaConf

from src.modelling.transformers.assessment_report import render_assessment
from src.modelling.utils.artifacts import configuration_snapshot, file_identity
from src.modelling.utils.assessment import write_json
from tests.modelling.model_fixtures import make_config


def _report_artifacts(run: Path) -> None:
    families = ["elasticnet", "lgbm", "random_forest"]
    config = make_config(model={"horizons": [30, 60, 90], "target_type": "zscore", "cv": {"n_splits": 5}})
    write_json(run / "training/configuration.json", configuration_snapshot(config))
    write_json(
        run / "metadata.json",
        {
            "run_id": run.name,
            "horizons": [30, 60, 90],
            "model_types": families,
            "training_complete": True,
            "workflow": "fresh-assessment-v1",
            "training_completed_at_utc": "2026-10-21T10:00:00+00:00",
            "observation_frontier": "2023-12-29",
            "training_as_of": "2021-12-31",
            "feature_cols": ["x", "zero"],
            "categorical_cols": ["category"],
            "target_type": "zscore",
            "label_column": "y",
            "folds": {str(h): list(range(1, 6)) for h in (30, 60, 90)},
        },
    )
    state = {
        "run_id": run.name,
        "horizons": {},
        "windows": {},
        "complete": True,
        "observation_frontier": "2026-10-20",
        "assessment_completed_at_utc": "2026-10-21T11:00:00+00:00",
    }
    for horizon in (30, 60, 90):
        output = run / f"assessment/h{horizon}"
        frames = []
        for population in ("validation", "backtest"):
            for model in [*families, "ensemble"]:
                frames.append(
                    pd.DataFrame(
                        {
                            "date": pd.to_datetime(["2022-01-03", "2022-01-04"]),
                            "model": model,
                            "population": population,
                            "horizon": horizon,
                            "ic": [0.1, 0.2],
                            "rolling_ic": [None, 0.15],
                            "rolling_ir": [None, 0.3],
                            "paired": 4,
                            "universe": 5,
                        }
                    )
                )
        output.mkdir(parents=True, exist_ok=True)
        pd.concat(frames, ignore_index=True).to_parquet(output / "daily.parquet", index=False)
        state["horizons"][str(horizon)] = {"effective_fold_test_end": "2021-10-06"}
        state["windows"][str(horizon)] = {}
        for fold in range(6):
            folder = output / f"fold{fold}"
            files = {}
            for family in families:
                curves = {"kind": "linear" if family == "elasticnet" else "tree", "rows": 2, "features": {}}
                for feature, importance in [("x", 0.123), ("zero", 0.0), ("category", 0.5)]:
                    curves["features"][feature] = {
                        "importance": importance,
                        "coefficient": -importance,
                        "categorical": feature == "category" and family != "elasticnet",
                        "observed": 2,
                        "sample_rows": 2,
                        "scatter": {"x": [0, 1], "raw_x": [10, 12], "row": [0, 1], "shap": [0, 0.2], "date": ["2022-01-03", "2022-01-04"]},
                        "binned": {"x": [0, 1], "raw_x": [10, 12], "mean": [0, 0.2], "count": [1, 1]},
                        "pdp": {"x": [0, 1] if fold == 0 else None, "mean": [0.1, 0.2] if fold == 0 else None},
                    }
                target = folder / family / "curves.json"
                write_json(target, curves)
                files[f"{family}/curves.json"] = file_identity(target)
            write_json(folder / "metadata.json", {"files": files})
            state["windows"][str(horizon)][str(fold)] = file_identity(folder / "metadata.json")
    write_json(run / "assessment/metadata.json", state)


def test_report_is_offline_and_covers_every_horizon_family_fold_and_zero_feature(tmp_path: Path) -> None:
    _report_artifacts(tmp_path)
    before = {path: path.read_bytes() for path in (tmp_path / "assessment").rglob("curves.json")}
    config = OmegaConf.create({"model": {"target_type": "zscore"}})
    output = render_assessment(tmp_path, config)
    text = output.read_text(encoding="utf-8")
    raw = re.search(r'<script id="assessment-data" type="application/json">(.*?)</script>', text, re.S)
    assert raw is not None
    payload = json.loads(raw.group(1), parse_constant=lambda token: pytest.fail(f"Non-standard JSON {token}"))
    assert payload["horizons"] == [30, 60, 90]
    assert set(payload["analysis"]) == {"30", "60", "90"}
    for horizon in payload["analysis"].values():
        assert set(horizon["temporal"]) == {"validation", "backtest"}
        assert set(horizon["effects"]) == {"elasticnet", "lgbm", "random_forest"}
        for family in horizon["effects"].values():
            assert set(family) == {str(fold) for fold in range(6)}
            for fold, curve in family.items():
                assert set(curve["features"]) == {"x", "zero", "category"}
                assert curve["features"]["zero"]["importance"] == 0
                assert ("scatter" in curve["features"]["x"]) is (fold == "0")
                if fold == "0":
                    assert curve["features"]["x"]["scatter"] == {"x": [0, 1], "shap": [0, 0.2], "raw_x": [10, 12]}
    assert text.count("<section ") == 2 and 'id="importance"' in text and 'id="effect"' in text and 'id="temporal"' in text
    assert "<table" not in text and 'id="research"' not in text
    assert "Signed linear contribution" in text and "Signed SHAP dependence" in text and "fixed cutoff model" in text
    assert " (training-standardized)" in text and " (raw input)" in text
    assert "Raw input %{customdata[0]" in text and "Raw input median %{customdata[0]" in text
    assert "Saved beta %{customdata:+.6g}" in text and "Saved beta %{customdata[1]:+.6g}" in text
    assert not re.search(r"<(?:script|link)\b[^>]*(?:src|href)=[\x22\x27](?:https?:)?//", text)
    assert before == {path: path.read_bytes() for path in (tmp_path / "assessment").rglob("curves.json")}
    first = hashlib.sha256(output.read_bytes()).hexdigest()
    render_assessment(tmp_path, config)
    assert hashlib.sha256(output.read_bytes()).hexdigest() == first
    print(
        "SANITY: two sections/three chart areas cover all horizons, original folds, tree and linear effects including zero importance; embedded data/assets rerender deterministically without changing evidence."
    )


@pytest.mark.parametrize("missing", ["training/configuration.json", "assessment/h90/daily.parquet", "assessment/h60/fold4/lgbm/curves.json"])
def test_report_refuses_incomplete_required_chart_evidence(tmp_path: Path, missing: str) -> None:
    _report_artifacts(tmp_path)
    (tmp_path / missing).unlink()
    with pytest.raises((FileNotFoundError, ValueError)):
        render_assessment(tmp_path, OmegaConf.create({}))
    assert not (tmp_path / "report.html").exists()
    print(f"SANITY: required missing chart artifact {missing} cannot produce a complete report.")
