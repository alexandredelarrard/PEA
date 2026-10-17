"""Score-consistent held-out tree explanations and observed-domain marginal response curves."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from src.modelling.transformers.base import BaseModel
from src.modelling.transformers.monitor import partial_dependence, shap_row_values, shap_score_scale
from src.modelling.utils.assessment import balanced_sample_positions, summarize_ic, write_json
from src.modelling.utils.assessment_metrics import trend_summary
from src.modelling.utils.features import design_matrix
from src.modelling.utils.metrics import daily_ic_series


def explain_tree(member: BaseModel, panel: pd.DataFrame, output: Path, config: DictConfig, logger: logging.Logger, *, pdp: bool) -> dict[str, Any]:
    """Persist all sampled signed contributions; plot sampling never changes importance or trends."""
    features = member.features
    matrix = design_matrix(panel, features, member.categoricals, downcast=False)
    got = shap_row_values(member.model, matrix, features, sample=0, logger=logger)
    if got is None:
        raise RuntimeError("Required SHAP explanation failed")
    phi, positions = got
    native = np.asarray(member.model.predict(matrix, pred_contrib=True)) * shap_score_scale(member.model)
    predicted = member.predict(panel).to_numpy()
    error = float(np.max(np.abs(native.sum(axis=1) - predicted)))
    if not np.array_equal(positions, np.arange(len(panel))) or not np.allclose(native.sum(axis=1), predicted, atol=1e-7, rtol=1e-6):
        raise ValueError("Native contribution and base sum do not reconstruct actual member predictions")
    if not np.allclose(phi, native[:, :-1], atol=1e-7, rtol=1e-6):
        raise ValueError("TreeExplainer does not match native prediction contributions")
    output.mkdir(parents=True, exist_ok=True)
    values = pd.DataFrame(phi, columns=features)
    values.insert(0, "ticker", panel.ticker.to_numpy())
    values.insert(0, "date", panel.date.to_numpy())
    values["base"], values["prediction"] = native[:, -1], predicted
    values.to_parquet(output / "shap.parquet", index=False)
    panel[["date", "ticker", *features]].to_parquet(output / "x.parquet", index=False)
    options = config.model.diagnostics
    chosen = balanced_sample_positions(panel, int(options.dependence_sample), int(config.seed))
    importance = pd.DataFrame(np.abs(phi), columns=features).groupby(panel.date.to_numpy()).mean().mean()
    constraints = member.model.params.get("monotone_constraints", [])
    if isinstance(constraints, str):
        constraints = [int(value) for value in constraints.split(",") if value.strip()]
    if len(constraints) != len(features):
        constraints = member.model.dump_model(num_iteration=1).get("monotone_constraints", [0] * len(features))
    curves = {}
    for index, feature in enumerate(features):
        col = matrix[:, index]
        categorical = feature in member.categoricals
        valid = np.isfinite(col) & ((col >= 0) if categorical else True)
        binned = _binned_dependence(col, phi[:, index], categorical)
        grid, means = (
            partial_dependence(
                member.model, matrix, index, int(options.pdp_grid_points), int(options.pdp_background), categorical=categorical, seed=int(config.seed)
            )
            if pdp
            else (None, None)
        )
        curves[feature] = {
            "importance": float(importance[feature]),
            "observed": int(valid.sum()),
            "sample_rows": len(panel),
            "categorical": categorical,
            "constraint": int(constraints[index]) if constraints else 0,
            "scatter": {"x": col[chosen], "shap": phi[chosen, index], "date": panel.date.iloc[chosen].astype(str).tolist()},
            "binned": binned,
            "shap_trend": {"shape": "unordered categorical"} if categorical else trend_summary(binned["x"], binned["mean"]),
            "pdp": {
                "x": grid,
                "mean": means,
                "trend": {"shape": "unordered categorical"}
                if categorical
                else trend_summary(grid, means)
                if grid is not None
                else {"shape": "unsupported"},
            },
            "missing_shap_mean": float(np.mean(phi[~valid, index])) if (~valid).any() else None,
        }
        if pdp:
            logger.info("PDP %s: %d/%d (%s)", member.family, index + 1, len(features), feature)
    result = {
        "features": curves,
        "additivity_max_error": error,
        "rows": len(panel),
        "unit_scale": shap_score_scale(member.model),
        "background_rows": min(int(options.pdp_background), len(panel)),
        "raw_shap_artifact": str(output / "shap.parquet"),
        "interpretation": "PDP replaces one input and averages with other observed inputs fixed; correlated-feature combinations can be unsupported. SHAP allocates model predictions, not causal effects.",
    }
    write_json(output / "curves.json", result)
    return result


def _binned_dependence(x: np.ndarray, phi: np.ndarray, categorical: bool) -> dict[str, Any]:
    good = np.isfinite(x) & ((x >= 0) if categorical else True)
    if not good.any():
        return {"x": [], "mean": [], "count": []}
    frame = pd.DataFrame({"x": x[good], "phi": phi[good]})
    bins = frame.x if categorical else pd.qcut(frame.x, min(10, frame.x.nunique()), duplicates="drop")
    grouped = frame.groupby(bins, observed=True).agg(x=("x", "median"), mean=("phi", "mean"), count=("phi", "size"))
    return {key: grouped[key].to_numpy() for key in ["x", "mean", "count"]}


def _summaries(frame: pd.DataFrame, models: list[str], horizon: int) -> dict[str, Any]:
    paired = frame.loc[np.isfinite(frame.y)]
    return {name: summarize_ic(daily_ic_series(paired, "y", name), horizon) for name in models}


def period_evidence(
    predictions: pd.DataFrame,
    models: list[str],
    periods: dict[str, list[str]],
    comparison_years: list[int],
    prior_scores: pd.DataFrame | None = None,
    *,
    horizon: int = 60,
) -> dict[str, Any]:
    """Diagnose fixed predictions; future-observed support is never a ranking or selection rule."""
    years = predictions.date.dt.year
    comparison = predictions.loc[years.isin(comparison_years)]
    supported = comparison.loc[np.isfinite(comparison.y)]
    n_dates = comparison.date.nunique()
    complete = supported.groupby("ticker").date.nunique()
    tickers = sorted(complete.index[complete == n_dates].astype(str)) if n_dates else []
    fixed = supported.loc[supported.ticker.isin(tickers)]
    cohorts = {str(year): _summaries(fixed.loc[fixed.date.dt.year == year], models, horizon) for year in comparison_years}
    population = {}
    sectors = []
    for year, frame in predictions.groupby(years):
        population[str(year)] = {
            "score_rows": len(frame),
            "label_rows": int(np.isfinite(frame.y).sum()),
            "dates": frame.date.nunique(),
            "tickers": frame.ticker.nunique(),
        }
        for sector, group in frame.groupby("sector", dropna=False):
            for model, stats in _summaries(group, models, horizon).items():
                sectors.append({"year": year, "sector": sector, "model": model, "rows": len(group), "tickers": group.ticker.nunique(), **stats})
    earlier = {}
    if prior_scores is not None and not prior_scores.empty:
        paired = prior_scores.merge(predictions[["date", "ticker", "y"]], on=["date", "ticker"], how="left", validate="one_to_one")
        earlier = _summaries(paired, models, horizon)
    return {
        "periods": {
            name: _summaries(predictions.loc[predictions.date.between(pd.Timestamp(bounds[0]), pd.Timestamp(bounds[1]))], models, horizon)
            for name, bounds in periods.items()
        },
        "year_population": population,
        "sector_year": sectors,
        "fixed_cohort": {
            "years": cohorts,
            "comparison_years": comparison_years,
            "tickers": tickers,
            "n_dates": n_dates,
            "rows": len(fixed),
            "policy": "Retrospective diagnostic: identical names with finite labels on every compared formation date; no model refitting or ranking change.",
        },
        "earlier_model": earlier,
        "earlier_model_policy": "The preceding original fold model scores the later comparison year without refitting; this diagnoses refresh sensitivity, not a selected alternative model.",
        "attribution_limit": "Coverage, distribution, sector and fixed-cohort comparisons are diagnostic associations. A causal source-defect claim requires dated source evidence.",
    }
