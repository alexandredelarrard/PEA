"""The report contains only three offline plot areas and all retained effects at every horizon."""

from __future__ import annotations

import base64
import gzip
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.modelling.transformers.assessment_report import _error_display, render_assessment
from src.modelling.utils.artifacts import configuration_snapshot, file_identity
from src.modelling.utils.assessment import write_json
from src.modelling.utils.assessment_metrics import native_error_diagnostics, temporal_diagnostics, universe_diagnostics
from src.modelling.utils.metrics import per_day_zscore
from tests.modelling.model_fixtures import make_config


def _report_artifacts(run: Path, n_tickers: int = 6) -> None:
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
        frames, tails, predictions, metrics, universe_metrics = [], [], [], {}, {}
        for population in ("validation", "backtest"):
            dates = pd.to_datetime(
                ["2001-01-03", "2001-01-04", "2001-06-01"] if population == "validation" else ["2022-01-03", "2022-01-04", "2022-01-05"]
            )
            index = np.tile(np.arange(n_tickers), len(dates))
            panel = pd.DataFrame(
                {
                    "date": np.repeat(dates, n_tickers),
                    "ticker": [f"T{i:03}" for i in index],
                    "y": (index - n_tickers / 2) / n_tickers,
                    "fold": np.repeat([1, 1, 2] if population == "validation" else [0, 0, 0], n_tickers),
                }
            )
            for number, model in enumerate(families):
                shift = horizon // 30 * 13 + number * 71 + (101 if population == "backtest" else 0)
                residual = ((index + shift) % n_tickers) // 2 / 10.0
                panel[f"raw_{model}"] = panel.y + residual * np.repeat([1.0, -1.0, 1.0], n_tickers)
                panel[model] = per_day_zscore(panel[f"raw_{model}"].to_numpy(), panel.date.to_numpy())
            panel.loc[panel.ticker == f"T{n_tickers - 1:03}", "y"] = np.nan
            panel["ensemble"] = panel[families].mean(axis=1)
            metrics[population], daily = temporal_diagnostics(panel, [*families, "ensemble"], horizon, 2)
            universe_metrics[population], cohort = universe_diagnostics(panel, [*families, "ensemble"], horizon, 2, (10, 50, 100))
            daily["population"], daily["horizon"] = population, horizon
            cohort["population"], cohort["horizon"] = population, horizon
            frames.append(daily)
            tails.append(cohort)
            predictions.append(panel)
        output.mkdir(parents=True, exist_ok=True)
        pd.concat(frames, ignore_index=True).to_parquet(output / "daily.parquet", index=False)
        pd.concat(tails, ignore_index=True).to_parquet(output / "universe_daily.parquet", index=False)
        full = pd.concat(predictions, ignore_index=True)
        full.to_parquet(output / "predictions.parquet", index=False)
        native_error_diagnostics(full, families).to_parquet(output / "ticker_errors.parquet", index=False)
        write_json(output / "metrics.json", metrics)
        write_json(output / "universe_metrics.json", universe_metrics)
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
    output = render_assessment(tmp_path)
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
    assert text.count("<section ") == 3
    assert all(f'id="{chart}"' in text for chart in ("importance", "effect", "temporal", "target_prediction", "ticker_rmse"))
    assert "<table" not in text and 'id="research"' not in text
    assert "Signed linear contribution" in text and "Signed SHAP dependence" in text and "fixed cutoff model" in text
    assert " (training-standardized)" in text and " (raw input)" in text
    assert "Raw input %{customdata[0]" in text and "Raw input median %{customdata[0]" in text
    assert "Saved beta %{customdata:+.6g}" in text and "Saved beta %{customdata[1]:+.6g}" in text
    assert not re.search(r"<(?:script|link)\b[^>]*(?:src|href)=[\x22\x27](?:https?:)?//", text)
    assert before == {path: path.read_bytes() for path in (tmp_path / "assessment").rglob("curves.json")}
    first = hashlib.sha256(output.read_bytes()).hexdigest()
    render_assessment(tmp_path)
    assert hashlib.sha256(output.read_bytes()).hexdigest() == first
    print(
        "SANITY: three sections/five chart areas cover all horizons, original folds, native errors and retained variable effects; offline rendering is deterministic without changing evidence."
    )


@pytest.mark.parametrize(
    "missing",
    [
        "training/configuration.json",
        "assessment/h90/daily.parquet",
        "assessment/h60/fold4/lgbm/curves.json",
        "assessment/h60/universe_daily.parquet",
        "assessment/h60/ticker_errors.parquet",
    ],
)
def test_report_refuses_incomplete_required_chart_evidence(tmp_path: Path, missing: str) -> None:
    _report_artifacts(tmp_path)
    (tmp_path / missing).unlink()
    with pytest.raises((FileNotFoundError, ValueError)):
        render_assessment(tmp_path)
    assert not (tmp_path / "report.html").exists()
    print(f"SANITY: required missing chart artifact {missing} cannot produce a complete report.")


def _read_payload(run: Path) -> tuple[dict, str]:
    text = render_assessment(run).read_text(encoding="utf-8")
    raw = re.search(r'<script id="assessment-data" type="application/json">(.*?)</script>', text, re.S)
    assert raw is not None
    return json.loads(raw.group(1)), text


def _native_values(meta: dict, text: str) -> np.ndarray:
    block = re.search(r'<script id="' + re.escape(meta["block"]) + r'" type="application/octet-stream">(.*?)</script>', text, re.S)
    assert block is not None
    return np.frombuffer(gzip.decompress(base64.b64decode(block.group(1))), dtype="<f8").reshape(len(meta["tickers"]), 2, len(meta["date"]))


def test_report_filters_cover_tail_universes_and_native_ticker_errors(tmp_path: Path) -> None:
    _report_artifacts(tmp_path)
    payload, text = _read_payload(tmp_path)
    for analysis in payload["analysis"].values():
        for population in ("validation", "backtest"):
            assert set(analysis["temporal"][population]) == {"all", "10", "50", "100"}
            for models in analysis["temporal"][population].values():
                assert set(models) == {*payload["families"], "ensemble"}
                assert models["elasticnet"]["summary"]["n_days"] == 3
            assert set(analysis["native"][population]) == set(payload["families"])
            assert all(meta["tickers"] for meta in analysis["native"][population].values())
    assert all(f'<option value="{size}">' in text for size in ("all", "10", "50", "100"))
    assert '"change"' in text and "async function temporal(" in text
    assert "async function targetPrediction(" in text and "async function tickerRmse(" in text
    print(
        "SANITY: every horizon/sample/cohort has precomputed model curves and summaries; native charts offer individual families only, with separate renderers."
    )


def test_report_preserves_native_history_dates_gaps_and_empty_states(tmp_path: Path) -> None:
    _report_artifacts(tmp_path)
    output = tmp_path / "assessment/h60"
    predictions = pd.read_parquet(output / "predictions.parquet")
    missing = (predictions.fold == 1) & (predictions.date == predictions.date.min()) & (predictions.ticker == "T000")
    predictions.loc[missing, "raw_lgbm"] = np.nan
    predictions.loc[predictions.fold == 0, "raw_elasticnet"] = np.nan
    predictions.to_parquet(output / "predictions.parquet", index=False)
    native_error_diagnostics(predictions, ["elasticnet", "lgbm", "random_forest"]).to_parquet(output / "ticker_errors.parquet", index=False)
    payload, text = _read_payload(tmp_path)
    meta = payload["analysis"]["60"]["native"]["validation"]["lgbm"]
    assert pd.to_datetime(meta["date"]).tolist() == sorted(predictions.loc[predictions.fold > 0, "date"].unique())
    assert meta["fold"] == [1, 1, 2]
    values = _native_values(meta, text)
    ticker_index = [row["ticker"] for row in meta["tickers"]].index("T000")
    assert np.isnan(values[ticker_index, :, 0]).all()
    assert payload["analysis"]["60"]["native"]["backtest"]["elasticnet"]["tickers"] == []
    assert "No comparable native predictions" in text and "connectgaps:false" in text
    print(
        "SANITY: native histories preserve formation dates, null comparison slots and original fold changes; unsupported selections stay empty, never zero errors."
    )


@pytest.mark.parametrize("eligible_count", [40, 125, 211])
def test_error_display_ties_fill_the_available_population(eligible_count: int) -> None:
    errors = pd.DataFrame({"ticker": [f"T{i:03}" for i in range(eligible_count)], "rmse": 1.0, "n": 3})
    selected = _error_display(errors.sample(frac=1, random_state=7))
    worst = [f"T{i:03}" for i in range(eligible_count - 1, max(-1, eligible_count - 101), -1)]
    best = [f"T{i:03}" for i in range(min(50, max(0, eligible_count - 100)))]
    print(f"TIED RMSE: eligible={eligible_count}, selected={len(selected)}, required={min(eligible_count, 150)}")
    assert len(selected) == min(eligible_count, 150)
    assert selected.ticker.tolist() == [*worst, *best]
    assert selected.group.tolist() == ["worst"] * len(worst) + ["best"] * len(best)
    assert selected.ticker.is_unique
    print("SANITY: tied RMSE uses descending tickers at the worst end and ascending tickers at the best end, retaining the full available cap.")


def test_error_display_mixed_ties_exclude_nonfinite_and_missing_support() -> None:
    errors = pd.DataFrame({"ticker": [f"T{i:03}" for i in range(160)], "rmse": np.arange(160) // 40, "n": 3})
    unsupported = pd.DataFrame(
        {
            "ticker": ["nan", "inf", "negative_inf", "zero_support", "missing_support"],
            "rmse": [np.nan, np.inf, -np.inf, 999, 999],
            "n": [3, 3, 3, 0, np.nan],
        }
    )
    selected = _error_display(pd.concat([errors, unsupported], ignore_index=True).sample(frac=1, random_state=7))
    assert selected.ticker.tolist() == [*[f"T{i:03}" for i in range(159, 59, -1)], *[f"T{i:03}" for i in range(50)]]
    assert selected.group.tolist() == ["worst"] * 100 + ["best"] * 50
    assert selected.ticker.is_unique and np.isfinite(selected.rmse).all() and (selected.n > 0).all()
    print("SANITY: mixed RMSE ties fill150 distinct titles, retain valid zero errors, and exclude nonfinite errors and zero/missing support.")


def test_report_embeds_only_worst100_best50_tickers_per_selection(tmp_path: Path) -> None:
    _report_artifacts(tmp_path, n_tickers=211)
    payload, text = _read_payload(tmp_path)
    selections = []
    for h, analysis in payload["analysis"].items():
        evidence = pd.read_parquet(tmp_path / f"assessment/h{h}/ticker_errors.parquet")
        predictions = pd.read_parquet(tmp_path / f"assessment/h{h}/predictions.parquet")
        for population, models in analysis["native"].items():
            for model, meta in models.items():
                eligible = evidence.loc[
                    (evidence.population == population) & (evidence.model == model) & np.isfinite(evidence.rmse) & (evidence.n > 0)
                ]
                order = eligible.sort_values(["rmse", "ticker"]).ticker.tolist()
                worst, best = order[-100:][::-1], order[:50]
                expected = list(dict.fromkeys([*worst, *best]))
                assert [row["ticker"] for row in meta["tickers"]] == expected and len(expected) == 150
                assert [row["group"] for row in meta["tickers"]] == ["worst"] * 100 + ["best"] * 50
                selections.append(tuple(expected))
                values = _native_values(meta, text)
                assert values.shape == (150, 2, 3)
                observed = predictions.loc[(predictions.fold > 0) == (population == "validation")].set_index(["ticker", "date"])
                histories = np.asarray([observed.loc[ticker].sort_index()[[f"raw_{model}", "y"]].to_numpy().T for ticker in expected])
                np.testing.assert_allclose(values, histories)
                assert analysis["temporal"][population]["all"][model]["universe"] == [211] * 3
    assert len(set(selections)) > 3
    print(
        "SANITY: all18model/horizon/sample groups independently export only100worst+50best native histories with deterministic ties; unsupported titles excluded and All IC keeps211titles."
    )
