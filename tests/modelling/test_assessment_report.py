"""Artifact-only HTML rendering: complete retained pools, offline assets and strict evidence JSON."""

import json
import re
from pathlib import Path

import pandas as pd
import pytest
from omegaconf import OmegaConf

from src.modelling.transformers.assessment_report import render_assessment
from src.modelling.utils.assessment import write_json


def _report_artifacts(root: Path) -> None:
    out = root / "_out"
    selected = {
        family: {"id": "b0", "features": ["x"], "categoricals": [], "metrics": {"ic_ir": 0.2, "mean_ic": 0.1, "eligible": True}}
        for family in ["elasticnet", "lgbm", "random_forest"]
    }
    write_json(
        out / "selection.json",
        {
            "frozen": True,
            "frozen_at": "2026-10-17",
            "selected": selected,
            "features": ["x"],
            "families": {family: {"best": candidate, "candidates": []} for family, candidate in selected.items()},
        },
    )
    write_json(
        out / "b0" / "result.json",
        {"features": {family: ["x"] for family in selected}, "metrics": {family: item["metrics"] for family, item in selected.items()}},
    )
    write_json(out / "final" / "result.json", {})
    write_json(out / "eval" / "result.json", {"frozen_at": "2026-10-17", "metrics": {}})
    write_json(out / "eval" / "linear.json", {"features": ["x"], "coefficient": [0.2], "mean": [0], "std": [1], "intercept": 0})
    write_json(out / "eval" / "market_calendar_years.json", {"2022": -0.1, "2023": 0.2})
    write_json(out / "eligibility.json", {"rows": 20, "features": [{"feature": "x", "role": "numeric"}]})
    dates = pd.to_datetime(["2022-01-03", "2023-01-03"])
    for fold in range(6):
        quality = pd.DataFrame({"year": [2022, 2023], "feature": ["x", "x"], "observed": [8, 9], "rows": [10, 10]})
        quality.to_parquet(out / "eval" / f"q{fold}.parquet", index=False)
        for family in ["lgbm", "random_forest"]:
            write_json(
                out / "eval" / f"{family}_{fold}" / "curves.json", {"features": {"x": {"importance": 0.123, "pdp": {"x": [0, 1], "mean": [0, 0.2]}}}}
            )
    daily = pd.DataFrame({"date": dates, "model": "ensemble", "ic": [0.1, 0.2]})
    for name in ["development", "sealed"]:
        daily.to_parquet(out / "eval" / f"daily_{name}.parquet", index=False)
    returns = pd.DataFrame(
        {
            "date": dates,
            "model": "ensemble",
            "period": "sealed",
            "selected": 2,
            "supported": 2,
            "cohort_return": [0.1, 0.2],
            "market_return": [0.05, 0.15],
            "excess_return": [0.05, 0.05],
        }
    )
    returns.to_parquet(out / "eval" / "matched_returns.parquet", index=False)


def test_report_is_offline_and_covers_exact_frozen_feature_pool(tmp_path: Path) -> None:
    _report_artifacts(tmp_path)
    config = OmegaConf.create({"model": {"target_type": "zscore"}})
    output = render_assessment(tmp_path, config)
    text = output.read_text(encoding="utf-8")
    raw = re.search(r'<script id="assessment-data" type="application/json">(.*?)</script>', text, re.S)
    assert raw is not None
    payload = json.loads(raw.group(1), parse_constant=lambda token: pytest.fail(f"Non-standard JSON {token}"))
    assert payload["explanations"]["lgbm"]["0"]["features"]["x"]["importance"] == 0.123
    assert payload["market"][0]["mean_excess"] == pytest.approx(0.05)
    assert not re.search(r'<(?:script|link|img)[^>]+(?:src|href)=["\']https?://', text)
    assert 'id="family"' in text and 'id="feature"' in text
    assert text.index('id="diagnosis"') < text.index('id="research"')
    assert "document.documentElement.dataset.reportReady" in text
    again = render_assessment(tmp_path, config)
    assert again.read_text(encoding="utf-8") == text
    print(
        "SANITY: offline HTML contains exact retained-feature evidence, matched metrics, selectors and final research step; rerender is deterministic and artifact-only."
    )


def test_report_refuses_missing_retained_variable(tmp_path: Path) -> None:
    _report_artifacts(tmp_path)
    write_json(tmp_path / "_out" / "eval" / "lgbm_5" / "curves.json", {"features": {}})
    with pytest.raises(ValueError, match="coverage"):
        render_assessment(tmp_path, OmegaConf.create({"model": {"target_type": "zscore"}}))
    print("SANITY: a missing original-fold explanation cannot silently disappear from the variable selector.")
