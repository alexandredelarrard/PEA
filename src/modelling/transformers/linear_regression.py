"""
linear_regression.py  (src/modelling/transformers/linear_regression.py)
-----------------------------------------------------------------------
`LinearRegression`: the cross-sectional linear member (`linear:` block), family `elasticnet` or
`ridge`. Low signal-to-noise return targets are the classic home of linear models and the
benchmark a tree ensemble must beat. Pure numpy (no scikit-learn):
  * ridge      -- closed form, optionally time-decay weighted (pure L2);
  * elasticnet -- weighted cyclic coordinate descent with L1 + L2: L1 selects, L2 shares weight
                  across a collinear cluster instead of arbitrarily keeping one name.
Features are standardised with the TRAIN mean / std; a missing value is imputed to the mean
(0 after standardising) — LightGBM handles NaN natively, a linear model cannot.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from src.context import Context
from src.modelling.transformers.base import BaseModel
from src.modelling.utils.cv import time_decay_weights

_log = logging.getLogger(__name__)
LINEAR_FAMILIES = ("elasticnet", "ridge")
_FITTED_ARRAYS = ("coef_", "mean_", "std_")


def _standardize(features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Column-standardise, tolerant of ALL-NaN columns (no coverage in a fold) without
    np.nanmean / np.nanstd warnings. An all-NaN or constant column gets mean 0, std 1 and becomes
    all zeros after imputation, i.e. contributes nothing."""
    features = np.asarray(features, dtype=float)
    finite = np.isfinite(features)
    n = finite.sum(axis=0)
    safe_n = np.where(n > 0, n, 1)
    filled = np.where(finite, features, 0.0)
    mean = filled.sum(axis=0) / safe_n
    mean = np.where(n > 0, mean, 0.0)
    var = np.where(finite, (features - mean) ** 2, 0.0).sum(axis=0) / safe_n
    std = np.sqrt(var)
    std = np.where(np.isfinite(std) & (std > 0), std, 1.0)
    standardized = np.nan_to_num((features - mean) / std, nan=0.0)
    return standardized, mean, std


def _enet_coordinate_descent(
    standardized: np.ndarray, y: np.ndarray, w: np.ndarray, lam: float, l1_ratio: float, max_iter: int, tol: float
) -> np.ndarray:
    """Weighted elastic net by cyclic coordinate descent (glmnet-style) on
        (1/2sw) Σ w_i (y_i - Xs_i·β)^2 + lam*[ l1_ratio*||β||_1 + (1-l1_ratio)/2*||β||_2^2 ],
    with the soft-threshold update β_j = S(ρ_j, lam*l1_ratio) / (z_j + lam*(1-l1_ratio))."""
    _, k = standardized.shape
    sw = float(w.sum())
    z = np.array([float((w * standardized[:, j] ** 2).sum() / sw) for j in range(k)])
    z = np.where(z > 0, z, 1.0)
    l1, l2 = lam * l1_ratio, lam * (1.0 - l1_ratio)

    beta = np.zeros(k)
    r = y.astype(float).copy()  # residual = y - Xs @ beta (beta = 0)
    for _ in range(max_iter):
        max_step = 0.0
        for j in range(k):
            bj = beta[j]
            rho = float((w * standardized[:, j] * r).sum() / sw) + bj * z[j]
            if rho > l1:
                nj = (rho - l1) / (z[j] + l2)
            elif rho < -l1:
                nj = (rho + l1) / (z[j] + l2)
            else:
                nj = 0.0
            if nj != bj:
                r += standardized[:, j] * (bj - nj)  # keep the residual in sync
                beta[j] = nj
                max_step = max(max_step, abs(nj - bj))
        if max_step < tol:
            break
    return beta


def _sample_weights(panel: pd.DataFrame, n: int, half_life_years: float | None) -> np.ndarray:
    return time_decay_weights(panel["date"], half_life_years).astype(float) if half_life_years is not None else np.ones(n)


def fit_ridge(
    panel: pd.DataFrame, feats: list[str], label_name: str, alpha: float, half_life_years: float | None
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Closed-form (optionally time-decay weighted) ridge on standardised features:
    coef = (Xs' W Xs + alpha I)^-1 Xs' W (y - y_bar). Returns (coef, intercept, mean, std)."""
    features = panel[feats].to_numpy(dtype=float)
    y = panel[label_name].to_numpy(dtype=float)
    standardized, mean, std = _standardize(features)
    w = _sample_weights(panel, len(y), half_life_years)
    y_bar = float(np.average(y, weights=w))
    weighted_transpose = standardized.T * w
    system = weighted_transpose @ standardized + float(alpha) * np.eye(standardized.shape[1])
    coef = np.linalg.solve(system, weighted_transpose @ (y - y_bar))
    return coef, y_bar, mean, std


def fit_elasticnet(
    panel: pd.DataFrame, feats: list[str], label_name: str, alpha: float, l1_ratio: float, max_iter: int, tol: float, half_life_years: float | None
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Elastic net on standardised features. `alpha` is the overall penalty on the (1/2n) loss
    (glmnet scale, ~1e-4..1e-1), `l1_ratio` the L1 share (0 = ridge, 1 = lasso). WARNs when every
    coefficient is zero: `alpha` is then too high for the target's scale and the member predicts
    a constant. Returns (coef, intercept, mean, std)."""
    features = panel[feats].to_numpy(dtype=float)
    y = panel[label_name].to_numpy(dtype=float)
    standardized, mean, std = _standardize(features)
    w = _sample_weights(panel, len(y), half_life_years)
    y_bar = float(np.average(y, weights=w))
    coef = _enet_coordinate_descent(standardized, y - y_bar, w, float(alpha), float(l1_ratio), int(max_iter), float(tol))
    if int(np.count_nonzero(np.abs(coef) > 0)) == 0:
        _log.warning(
            "elastic-net is DEGENERATE: all %d coefficients are zero (alpha=%.4g too high for the target scale -> every |rho| < alpha*l1_ratio=%.4g). "
            "The model will predict a CONSTANT; lower `alpha` in linear_modelling.yml.",
            len(coef),
            alpha,
            alpha * l1_ratio,
        )
    return coef, y_bar, mean, std


class LinearRegression(BaseModel):
    """Standardising elastic-net / ridge member; categoricals are not used (numeric only)."""

    config_key = "linear"
    supports_classification = False

    def __init__(self, context: Context, config: DictConfig, family: str, horizon: int | None = None) -> None:
        super().__init__(context, config, family, horizon)
        if self.family not in LINEAR_FAMILIES:
            raise ValueError(f"LinearRegression family must be one of {LINEAR_FAMILIES}, got {self.family!r}")
        self.coef_ = np.zeros(0)
        self.intercept_ = 0.0
        self.mean_ = np.zeros(0)
        self.std_ = np.ones(0)

    def _fit(self, train: pd.DataFrame, valid: pd.DataFrame | None) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
        c = self.block(self._config)
        alpha = float(c.get("alpha", 1e-3))
        if self.family == "ridge":
            fitted = fit_ridge(train, self.features, self.label_column, alpha, self.half_life_years)
        else:
            fitted = fit_elasticnet(
                train,
                self.features,
                self.label_column,
                alpha=alpha,
                l1_ratio=float(c.get("l1_ratio", 0.5)),
                max_iter=int(c.get("max_iter", 1000)),
                tol=float(c.get("tol", 1e-6)),
                half_life_years=self.half_life_years,
            )
        self.coef_, self.intercept_, self.mean_, self.std_ = fitted
        return fitted

    def _predict(self, panel: pd.DataFrame) -> np.ndarray:
        features = np.asarray(panel[self.features], dtype=float)
        standardized = np.nan_to_num((features - self.mean_) / self.std_, nan=0.0)
        return standardized @ self.coef_ + self.intercept_

    def importance(self) -> pd.Series:
        """|coefficient| per feature (features are standardised, so comparable)."""
        return pd.Series(dict(zip(self.features, np.abs(self.coef_), strict=False)), dtype=float)

    # Fitted arrays travel as plain Python floats (exact float64 round trip), not numpy objects:
    # a numpy pickle is tied to the numpy version that wrote it.
    def __getstate__(self) -> dict[str, Any]:
        state = super().__getstate__()
        for name in _FITTED_ARRAYS:
            state[name] = np.asarray(state[name], dtype=float).tolist()
        state["intercept_"] = float(state["intercept_"])
        state["model"] = None if self.model is None else "fitted"
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        super().__setstate__(state)
        for name in _FITTED_ARRAYS:
            setattr(self, name, np.asarray(getattr(self, name), dtype=float))
        if self.model is not None:
            self.model = (self.coef_, self.intercept_, self.mean_, self.std_)
