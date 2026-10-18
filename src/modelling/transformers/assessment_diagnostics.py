"""Score-consistent held-out tree explanations and observed-domain marginal response curves."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from src.modelling.transformers.base import BaseModel
from src.modelling.transformers.linear_regression import LinearRegression
from src.modelling.transformers.monitor import _sample_idx, partial_dependence, shap_row_values, shap_score_scale
from src.modelling.utils.assessment import balanced_sample_positions, write_json
from src.modelling.utils.features import design_matrix


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
                member.model, matrix, index, int(options.pdp_grid), int(options.pdp_background), categorical=categorical, seed=int(config.seed)
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
            "pdp": {"x": grid, "mean": means},
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
    bins = frame.x if categorical or frame.x.nunique() <= 10 else pd.qcut(frame.x, 10, duplicates="drop")
    grouped = frame.groupby(bins, observed=True).agg(x=("x", "median"), mean=("phi", "mean"), count=("phi", "size"))
    return {key: grouped[key].to_numpy() for key in ["x", "mean", "count"]}


def explain_linear(
    member: LinearRegression, panel: pd.DataFrame, output: Path, config: DictConfig, logger: logging.Logger, *, pdp: bool
) -> dict[str, Any]:
    """Explain saved standardized coefficients exactly, without fitting or conditional SHAP."""
    features = member.features
    matrix = panel[features].to_numpy(dtype=float)
    standardized = np.nan_to_num((matrix - member.mean_) / member.std_, nan=0.0)
    phi = standardized * member.coef_
    predicted = member.predict(panel).to_numpy()
    reconstructed = phi.sum(axis=1) + member.intercept_
    if not np.allclose(reconstructed, predicted, atol=1e-10, rtol=1e-10):
        raise ValueError("Saved coefficients and scaling do not reconstruct linear predictions")
    output.mkdir(parents=True, exist_ok=True)
    values = pd.DataFrame(phi, columns=features)
    values.insert(0, "ticker", panel.ticker.to_numpy())
    values.insert(0, "date", panel.date.to_numpy())
    values["base"], values["prediction"] = member.intercept_, predicted
    values.to_parquet(output / "contributions.parquet", index=False)
    panel[["date", "ticker", *features]].to_parquet(output / "x.parquet", index=False)
    coefficients = pd.DataFrame({"feature": features, "coefficient": member.coef_, "mean": member.mean_, "std": member.std_})
    coefficients.to_parquet(output / "coefficients.parquet", index=False)
    options = config.model.diagnostics
    chosen = balanced_sample_positions(panel, int(options.dependence_sample), int(config.seed))
    background = _sample_idx(len(panel), int(options.pdp_background), int(config.seed))
    curves = {}
    for index, feature in enumerate(features):
        col = matrix[:, index]
        valid = np.isfinite(col)
        signed_x = (col - member.mean_[index]) / member.std_[index]
        binned = _binned_dependence(signed_x, phi[:, index], False)
        binned["raw_x"] = np.asarray(binned["x"]) * member.std_[index] + member.mean_[index]
        grid, means = None, None
        if pdp and valid.any():
            grid = np.unique(col[valid])
            if len(grid) > int(options.pdp_grid):
                grid = np.unique(np.quantile(col[valid], np.linspace(0.02, 0.98, int(options.pdp_grid))))
            means = (
                predicted[background].mean() - phi[background, index].mean() + member.coef_[index] * (grid - member.mean_[index]) / member.std_[index]
            )
        curves[feature] = {
            "importance": float(abs(member.coef_[index])),
            "coefficient": float(member.coef_[index]),
            "observed": int(valid.sum()),
            "sample_rows": len(panel),
            "categorical": False,
            "scatter": {
                "x": signed_x[chosen],
                "raw_x": col[chosen],
                "row": chosen,
                "shap": phi[chosen, index],
                "date": panel.date.iloc[chosen].astype(str).tolist(),
            },
            "binned": binned,
            "pdp": {"x": grid, "mean": means},
            "missing_shap_mean": float(np.mean(phi[~valid, index])) if (~valid).any() else None,
        }
    result = {
        "kind": "linear",
        "features": curves,
        "additivity_max_error": float(np.max(np.abs(reconstructed - predicted))),
        "rows": len(panel),
        "background_rows": len(background),
        "intercept": float(member.intercept_),
        "interpretation": "Signed linear x uses the saved training mean/std; contribution is saved beta times that input. Missing input contributes zero and has no observed x/bin. Raw inputs remain in hover and PDP; importance is abs(beta). These are model responses, not causal effects.",
    }
    write_json(output / "curves.json", result)
    logger.info("Linear contributions: %s rows, %s retained variables", len(panel), len(features))
    return result
