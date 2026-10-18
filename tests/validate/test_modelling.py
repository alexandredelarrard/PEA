"""Independent saved-run arithmetic, strict inventory and incomplete-evidence refusal."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from src.context import Context
from src.modelling.utils.assessment import write_json
from src.validate.checks import modelling


def test_independent_rank_ic_skips_undefined_dates() -> None:
    frame = pd.DataFrame(
        {
            "date": np.repeat(pd.to_datetime(["2022-01-03", "2022-01-04"]), 4),
            "ticker": list("ABCD") * 2,
            "y": [0, 1, 2, 3] * 2,
            "m": [0, 1, 2, 3, 0, 0, 0, 0],
        }
    )
    result = modelling.independent_daily(frame, "m")
    assert result.tolist() == pytest.approx([1.0])
    print("SANITY: independent rank IC measures the known positive day and leaves the constant-prediction day undefined.")


def test_incomplete_assessment_abstains_without_store_access(tmp_path: Path) -> None:
    class Forbidden:
        def __getattr__(self, name: str) -> Any:
            pytest.fail("Incomplete artifacts must not read a source table")

    result = modelling.check_modelling(cast(Context, SimpleNamespace(store=Forbidden())), out=tmp_path, config=OmegaConf.create({}))
    assert result.status == "abstain"
    print("SANITY: missing run evidence abstains before any database access; absence is never a zero or pass.")


@pytest.mark.parametrize("name", ["../escape.json", "C:/escape.json"])
def test_independent_inventory_rejects_escaping_paths(tmp_path: Path, name: str) -> None:
    with pytest.raises(ValueError, match="path|escape|inside"):
        modelling._inventory(tmp_path, {name: {"bytes": 0, "sha256": "0" * 64}})
    print("SANITY: an independent validator refuses an escaping artifact reference before opening a file.")


def test_independent_inventory_rejects_changed_bytes(tmp_path: Path) -> None:
    path = tmp_path / "x"
    path.write_bytes(b"abc")
    identity = {"bytes": 3, "sha256": hashlib.sha256(b"abc").hexdigest()}
    modelling._inventory(tmp_path, {"x": identity})
    path.write_bytes(b"def")
    with pytest.raises(ValueError, match="hash|identity|changed"):
        modelling._inventory(tmp_path, {"x": identity})
    print("SANITY: same-sized artifact corruption fails independent SHA256 reconciliation.")


@pytest.mark.parametrize("mutation", ["none", "coefficient_order", "signed_values", "importance", "pdp_response", "pdp_grid"])
def test_independent_linear_effect_reconciliation(tmp_path: Path, mutation: str) -> None:
    member = SimpleNamespace(
        family="elasticnet",
        horizon=60,
        features=["x", "negative", "zero"],
        categoricals=[],
        coef_=np.array([2.0, -1.0, 0.0]),
        mean_=np.array([1.0, 2.0, 0.0]),
        std_=np.array([1.0, 2.0, 1.0]),
        intercept_=0.25,
    )
    x = pd.DataFrame(
        {
            "date": pd.to_datetime(["2022-01-03"] * 4),
            "ticker": list("ABCD"),
            "x": [0.0, 1.0, 2.0, np.nan],
            "negative": [0.0, 2.0, 4.0, 0.0],
            "zero": [0.0, 1.0, 2.0, 3.0],
        }
    )
    raw = x[["date", "ticker"]].copy()
    raw[member.features] = np.array([[-2.0, 1.0, 0.0], [0.0, 0.0, 0.0], [2.0, -1.0, 0.0], [0.0, 1.0, 0.0]])
    raw["base"], raw["prediction"] = 0.25, [-0.75, 0.25, 1.25, 1.25]
    coefficients = pd.DataFrame({"feature": member.features, "coefficient": member.coef_, "mean": member.mean_, "std": member.std_})
    curves = {
        "features": {
            "x": {"importance": 2.0, "pdp": {"x": [0.0, 1.0, 2.0], "mean": [-1.5, 0.5, 2.5]}},
            "negative": {"importance": 1.0, "pdp": {"x": [0.0, 2.0, 4.0], "mean": [1.25, 0.25, -0.75]}},
            "zero": {"importance": 0.0, "pdp": {"x": [0.0, 1.0, 2.0, 3.0], "mean": [0.5] * 4}},
        }
    }
    if mutation == "coefficient_order":
        coefficients = coefficients.iloc[::-1]
    elif mutation == "signed_values":
        raw.loc[0, "x"] = 1.0
    elif mutation == "importance":
        curves["features"]["zero"]["importance"] = 2.0
    elif mutation == "pdp_response":
        curves["features"]["negative"]["pdp"]["mean"][0] = 10.0
    elif mutation == "pdp_grid":
        curves["features"]["x"]["pdp"]["x"][0] = -100.0
    x.to_parquet(tmp_path / "x.parquet", index=False)
    raw.to_parquet(tmp_path / "contributions.parquet", index=False)
    coefficients.to_parquet(tmp_path / "coefficients.parquet", index=False)
    write_json(tmp_path / "curves.json", curves)
    config = OmegaConf.create({"seed": 7, "model": {"diagnostics": {"pdp_grid": 30, "pdp_background": 4}}})
    findings = []
    modelling._effect_checks(tmp_path, member, config, True, findings)
    assert bool(findings) is (mutation != "none")
    print(
        f"SANITY: independent saved-beta/linear additivity/abs-importance/native PDP checks detect mutation={mutation}, including a true zero coefficient and missing input."
    )


@pytest.mark.parametrize("horizon", [30, 60, 90])
@pytest.mark.parametrize("mutation", ["none", "drop_day", "rolling", "paired", "wrong_horizon"])
def test_independent_metrics_preserve_undefined_sessions_and_horizon_scaling(tmp_path: Path, horizon: int, mutation: str) -> None:
    frames, daily_frames, stats = [], [], {}
    for population, year, fold in [("validation", 2000, 1), ("backtest", 2022, 0)]:
        dates = pd.bdate_range(f"{year}-01-03", periods=4)
        frames.append(
            pd.DataFrame(
                {
                    "date": np.repeat(dates, 4),
                    "ticker": list("ABCD") * 4,
                    "y": [0.0, 1.0, 2.0, 3.0] * 4,
                    "m": [0, 1, 2, 3, 3, 2, 1, 0, 0, 2, 1, 3, 0, 0, 0, 0],
                    "fold": fold,
                }
            )
        )
        daily_frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "model": "m",
                    "population": population,
                    "horizon": horizon,
                    "fold": fold,
                    "ic": [1.0, -1.0, 0.8, np.nan],
                    "rolling_ic": [np.nan, 0.0, -0.1, np.nan],
                    "rolling_ir": [np.nan, 0.0, -0.1 / 0.9 * np.sqrt(252 / horizon), np.nan],
                    "paired": 4,
                    "universe": 4,
                    "prediction_unique": [4, 4, 4, 1],
                    "target_unique": 4,
                }
            )
        )
        ic = np.array([1.0, -1.0, 0.8])
        stats[population] = {"m": {"mean_ic": ic.mean(), "ic_std": ic.std(), "ic_ir": ic.mean() / ic.std() * np.sqrt(252 / horizon), "n_days": 3}}
    predictions, daily = pd.concat(frames, ignore_index=True), pd.concat(daily_frames, ignore_index=True)
    if mutation == "drop_day":
        daily = daily.drop(index=3)
    elif mutation == "rolling":
        daily.loc[2, "rolling_ir"] = 999
    elif mutation == "paired":
        daily.loc[0, "paired"] = 3
    elif mutation == "wrong_horizon":
        daily.loc[0, "horizon"] = 1
    predictions.to_parquet(tmp_path / "predictions.parquet", index=False)
    daily.to_parquet(tmp_path / "daily.parquet", index=False)
    write_json(tmp_path / "metrics.json", stats)
    findings = []
    modelling._reconcile_metrics(tmp_path, horizon, ["m"], 2, findings)
    assert bool(findings) is (mutation != "none")
    print(
        f"SANITY: independent h{horizon} IC/population dispersion/full-window arithmetic detects mutation={mutation}, preserving the undefined fourth session."
    )
