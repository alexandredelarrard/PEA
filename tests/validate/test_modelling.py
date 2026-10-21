"""Independent saved-run arithmetic, strict inventory and incomplete-evidence refusal."""

from __future__ import annotations

import base64
import copy
import gzip
import hashlib
import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from src.context import Context
from src.modelling.transformers.assessment_report import render_assessment
from src.modelling.utils.assessment import write_json
from src.modelling.utils.assessment_metrics import native_error_diagnostics, universe_diagnostics
from src.validate.checks import modelling
from tests.modelling.test_assessment_report import _report_artifacts


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


@pytest.mark.parametrize(
    "mutation",
    [
        "none",
        "coefficient_order",
        "signed_values",
        "importance",
        "pdp_response",
        "pdp_grid",
        "scatter_x",
        "bin_x",
        "raw_hover",
        "beta_sign",
        "bin_mean",
        "bin_count",
        "row",
    ],
)
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
    for feature, signed_x, raw_x, binned_x, binned_raw, means, counts in [
        ("x", [-1.0, 0.0, 1.0, None], [0.0, 1.0, 2.0, None], [-1.0, 0.0, 1.0], [0.0, 1.0, 2.0], [-2.0, 0.0, 2.0], [1, 1, 1]),
        ("negative", [-1.0, 0.0, 1.0, -1.0], [0.0, 2.0, 4.0, 0.0], [-1.0, 0.0, 1.0], [0.0, 2.0, 4.0], [1.0, 0.0, -1.0], [2, 1, 1]),
        ("zero", [0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 2.0, 3.0], [0.0] * 4, [1] * 4),
    ]:
        curve = curves["features"][feature]
        curve["coefficient"] = member.coef_[member.features.index(feature)]
        curve["scatter"] = {"x": signed_x, "raw_x": raw_x, "row": [0, 1, 2, 3], "shap": raw[feature].tolist()}
        curve["binned"] = {"x": binned_x, "raw_x": binned_raw, "mean": means, "count": counts}
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
    elif mutation in {"scatter_x", "bin_x"}:
        key = "scatter" if mutation == "scatter_x" else "binned"
        curves["features"]["x"][key]["x"] = [0.0, 1.0, 2.0, None] if key == "scatter" else [0.0, 1.0, 2.0]
    elif mutation == "raw_hover":
        curves["features"]["x"]["scatter"]["raw_x"][0] = -1.0
    elif mutation == "beta_sign":
        curves["features"]["negative"]["coefficient"] = 1.0
    elif mutation in {"bin_mean", "bin_count"}:
        curves["features"]["x"]["binned"][mutation.removeprefix("bin_")][0] = 99
    elif mutation == "row":
        curves["features"]["x"]["scatter"]["row"][0] = -1
    x.to_parquet(tmp_path / "x.parquet", index=False)
    raw.to_parquet(tmp_path / "contributions.parquet", index=False)
    coefficients.to_parquet(tmp_path / "coefficients.parquet", index=False)
    write_json(tmp_path / "curves.json", curves)
    config = OmegaConf.create({"seed": 7, "model": {"diagnostics": {"dependence_sample": 4, "pdp_grid": 30, "pdp_background": 4}}})
    findings = []
    modelling._effect_checks(tmp_path, member, config, True, findings)
    assert bool(findings) is (mutation != "none")
    print(
        f"SANITY: independent signed coordinates/raw hover/beta/bin support/additivity/importance/PDP checks detect mutation={mutation}, including zero beta and missing input."
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


def test_independent_tail_metrics_reject_wrong_selection_or_summary(tmp_path: Path) -> None:
    panels, daily, summaries = [], [], {}
    for population, year, fold in [("validation", 2001, 1), ("backtest", 2022, 0)]:
        index = np.tile(np.arange(24), 3)
        panel = pd.DataFrame(
            {
                "date": np.repeat(pd.bdate_range(f"{year}-01-03", periods=3), 24),
                "ticker": [f"T{i:02}" for i in index],
                "m": index.astype(float),
                "y": np.sin(index),
                "fold": fold,
            }
        )
        panel.loc[panel.ticker == "T23", "y"] = np.nan
        summaries[population], rows = universe_diagnostics(panel, ["m"], 60, 2, (10, 50, 100))
        rows["population"], rows["horizon"] = population, 60
        panels.append(panel)
        daily.append(rows)
    predictions, baseline = pd.concat(panels, ignore_index=True), pd.concat(daily, ignore_index=True)
    predictions.to_parquet(tmp_path / "predictions.parquet", index=False)
    for mutation in ("none", "early_label_filter", "paired", "universe", "rolling", "global", "drop_day"):
        changed, stats = baseline.copy(), copy.deepcopy(summaries)
        if mutation == "early_label_filter":
            wrong = []
            for population, panel in zip(("validation", "backtest"), panels, strict=True):
                _, rows = universe_diagnostics(panel.dropna(subset=["y"]), ["m"], 60, 2, (10, 50, 100))
                rows["population"], rows["horizon"] = population, 60
                wrong.append(rows)
            changed = pd.concat(wrong, ignore_index=True)
        elif mutation in ("paired", "universe", "rolling"):
            changed.loc[1, "rolling_ir" if mutation == "rolling" else mutation] = 999.0
        elif mutation == "global":
            stats["backtest"]["10"]["m"]["mean_ic"] = 999.0
        elif mutation == "drop_day":
            changed = changed.drop(index=0)
        changed.to_parquet(tmp_path / "universe_daily.parquet", index=False)
        write_json(tmp_path / "universe_metrics.json", stats)
        findings = []
        for size in (10, 50, 100):
            modelling._reconcile_metrics(tmp_path, 60, ["m"], 2, findings, size=size)
        assert bool(findings) is (mutation != "none"), mutation
    print(
        "SANITY: independent cohort arithmetic rejects prefiltered outcomes, altered counts/rolling/global summaries and dropped sessions; All's calculation contract stays separate."
    )


def test_independent_native_errors_reject_corruption(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "date": pd.bdate_range("2022-01-03", periods=6),
            "ticker": ["A", "A", "B", "A", "A", "B"],
            "fold": [1, 1, 1, 0, 0, 0],
            "raw_m": [2.0, 2.0, 1.0, 2.0, 2.0, np.inf],
            "m": np.nan,
            "y": [0.0, 4.0, np.nan, 0.0, 4.0, 1.0],
        }
    )
    frame.to_parquet(tmp_path / "predictions.parquet", index=False)
    baseline = native_error_diagnostics(frame, ["m"])
    measured = modelling._independent_native_errors(frame, ["m"])
    assert measured.loc[measured.ticker == "A", "rmse"].tolist() == [2.0, 2.0]
    for mutation in ("none", "n", "mae", "bias", "rmse", "missing", "duplicate", "unsupported"):
        changed = baseline.copy()
        if mutation in ("n", "mae", "bias", "rmse"):
            changed.loc[0, mutation] = 999.0
        elif mutation == "missing":
            changed = changed.iloc[1:]
        elif mutation == "duplicate":
            changed = pd.concat([changed, changed.iloc[:1]], ignore_index=True)
        elif mutation == "unsupported":
            changed.loc[changed.n == 0, "rmse"] = 0.0
        changed.to_parquet(tmp_path / "ticker_errors.parquet", index=False)
        findings = []
        modelling._native_error_checks(tmp_path, ["m"], findings)
        assert bool(findings) is (mutation != "none"), mutation
    print(
        "SANITY: independent native errors preserve constants/zero support and reject arithmetic, missing/duplicated ticker and unsupported-as-zero corruption."
    )


@pytest.mark.parametrize("eligible_count", [40, 125, 211])
@pytest.mark.parametrize("mutation", ["none", "legacy_union", "duplicate"])
def test_independent_report_ties_reject_truncated_population(tmp_path: Path, eligible_count: int, mutation: str) -> None:
    panels, errors, native, blocks = [], [], {}, []
    worst = list(range(eligible_count - 1, max(-1, eligible_count - 101), -1))
    best = list(range(min(50, max(0, eligible_count - 100))))
    selection = [(i, "worst") for i in worst] + [(i, "best") for i in best]
    if mutation == "legacy_union":
        selection = [(i, "worst") for i in range(min(100, eligible_count))]
    elif mutation == "duplicate":
        selection[-1] = selection[0]
    for population, year, fold in [("validation", 2001, 1), ("backtest", 2022, 0)]:
        dates = pd.bdate_range(f"{year}-01-03", periods=3)
        panels.append(
            pd.DataFrame(
                {
                    "date": np.repeat(dates, eligible_count),
                    "ticker": [f"T{i:03}" for i in range(eligible_count)] * 3,
                    "fold": fold,
                    "y": 0.0,
                    "raw_m": 1.0,
                }
            )
        )
        errors.append(
            pd.DataFrame(
                {
                    "population": population,
                    "model": "m",
                    "ticker": [f"T{i:03}" for i in range(eligible_count)],
                    "n": 3,
                    "mae": 1.0,
                    "bias": 1.0,
                    "rmse": 1.0,
                }
            )
        )
        block = f"native-60-{population}-m"
        native[population] = {
            "m": {
                "date": dates.tolist(),
                "fold": [fold] * 3,
                "eligible_tickers": eligible_count,
                "tickers": [{"ticker": f"T{i:03}", "group": group, "n": 3, "mae": 1.0, "bias": 1.0, "rmse": 1.0} for i, group in selection],
                "block": block,
            }
        }
        values = np.tile(np.array([[1.0] * 3, [0.0] * 3], dtype="<f8"), (len(selection), 1, 1))
        encoded = base64.b64encode(gzip.compress(values.tobytes(), mtime=0)).decode("ascii")
        blocks.append(f'<script id="{block}" type="application/octet-stream">{encoded}</script>')
    output = tmp_path / "assessment/h60"
    output.mkdir(parents=True)
    pd.concat(panels, ignore_index=True).to_parquet(output / "predictions.parquet", index=False)
    pd.concat(errors, ignore_index=True).to_parquet(output / "ticker_errors.parquet", index=False)
    findings = []
    modelling._native_payload_checks(tmp_path, 60, ["m"], native, "".join(blocks), findings)
    assert bool(findings) is (mutation != "none")
    population_failure = mutation == "duplicate" or (mutation == "legacy_union" and eligible_count > 100)
    assert any(finding.observed == "HTML error-display population size differs" for finding in findings) is population_failure
    print(
        f"SANITY: independent tied-RMSE check for{eligible_count} titles accepts the full opposite-end selection and rejects mutation={mutation}; population cardinality is checked separately."
    )


def test_independent_report_rejects_wrong_ticker_series_or_rmse(tmp_path: Path) -> None:
    _report_artifacts(tmp_path, n_tickers=211)
    text = render_assessment(tmp_path).read_text(encoding="utf-8")
    match = re.search(r'<script id="assessment-data" type="application/json">(.*?)</script>', text, re.S)
    assert match is not None
    baseline = json.loads(match.group(1))["analysis"]["60"]["native"]
    families = ["elasticnet", "lgbm", "random_forest"]
    for mutation in ("none", "n", "mae", "bias", "rmse", "group", "order", "membership", "date", "history"):
        native, document = copy.deepcopy(baseline), text
        meta = native["backtest"]["lgbm"]
        if mutation in ("n", "mae", "bias", "rmse"):
            meta["tickers"][0][mutation] = 999.0
        elif mutation == "group":
            meta["tickers"][0]["group"] = "best"
        elif mutation == "order":
            meta["tickers"][0], meta["tickers"][1] = meta["tickers"][1], meta["tickers"][0]
        elif mutation == "membership":
            meta["tickers"][0]["ticker"] = "unselected"
        elif mutation == "date":
            meta["date"][0] = "1999-01-01 00:00:00"
        elif mutation == "history":
            block = re.search(r'<script id="' + re.escape(meta["block"]) + r'" type="application/octet-stream">(.*?)</script>', text, re.S)
            assert block is not None
            values = np.frombuffer(gzip.decompress(base64.b64decode(block.group(1))), dtype="<f8").copy()
            values[0] += 1.0
            encoded = base64.b64encode(gzip.compress(values.tobytes(), mtime=0)).decode("ascii")
            document = document.replace(block.group(0), block.group(0).replace(block.group(1), encoded))
        findings = []
        modelling._native_payload_checks(tmp_path, 60, families, native, document, findings)
        assert bool(findings) is (mutation != "none"), mutation
    print(
        "SANITY: independent report reconciliation rejects wrong100worst/50best membership/order/groups, MAE/bias/RMSE/count, dates and native history bytes; every selected daily value is checked."
    )
