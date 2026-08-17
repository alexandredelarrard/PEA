"""
baselines.py  (src/modelling/utils_model/baselines.py)
------------------------------------------------------
Linear cross-sectional baselines (ridge / elastic-net).

Low signal-to-noise cross-sectional return targets are the classic home of
linear models: low variance and robust when estimation error dominates, and the
standard benchmark any tree/GBDT alpha must beat. Two pure-numpy estimators (no
scikit-learn dependency):
  * ridge      -- closed-form, optionally time-decay weighted (fast; pure L2).
  * elasticnet -- coordinate descent with BOTH L1 and L2. Preferred with many
                  correlated features: L1 selects, L2 shares weight across a
                  collinear cluster instead of arbitrarily keeping one name.
Both expose the same `predict(X)` interface as a LightGBM booster, so the
modelling/backtest code treats them interchangeably.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.modelling.long_short.utils.model import time_decay_weights

_log = logging.getLogger(__name__)


class LinearModel:
    """Standardizing linear model with a booster-compatible `predict(X)`.

    Features are standardized with the TRAIN mean/std (stored), and missing
    values are imputed to the feature mean (0 after standardizing) at predict
    time -- LightGBM handles NaN natively, a linear model cannot."""

    def __init__(self, coef, intercept, mean, std, feature_names, kind):
        self.coef = np.asarray(coef, dtype=float)
        self.intercept = float(intercept)
        self.mean = np.asarray(mean, dtype=float)
        self.std = np.asarray(std, dtype=float)
        self.feature_names = list(feature_names)
        self.kind = kind

    def predict(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        Xs = np.nan_to_num((X - self.mean) / self.std, nan=0.0)
        return Xs @ self.coef + self.intercept


def _standardize(X: np.ndarray):
    """Column-standardize, tolerant of ALL-NaN columns (a feature with no
    coverage in a fold) -- computed without np.nanmean/np.nanstd so it never
    emits 'Mean of empty slice' / 'Degrees of freedom <= 0' RuntimeWarnings.
    An all-NaN or constant column gets mean 0, std 1 -> becomes all zeros after
    NaN-imputation, i.e. contributes nothing (correct)."""
    X = np.asarray(X, dtype=float)
    finite = np.isfinite(X)
    n = finite.sum(axis=0)
    safe_n = np.where(n > 0, n, 1)
    filled = np.where(finite, X, 0.0)
    mean = filled.sum(axis=0) / safe_n
    mean = np.where(n > 0, mean, 0.0)
    var = np.where(finite, (X - mean) ** 2, 0.0).sum(axis=0) / safe_n
    std = np.sqrt(var)
    std = np.where(np.isfinite(std) & (std > 0), std, 1.0)
    Xs = np.nan_to_num((X - mean) / std, nan=0.0)
    return Xs, mean, std


def train_ridge(panel: pd.DataFrame, feats: list, label_name: str = "y",
                alpha: float = 10.0, half_life_years: float | None = None) -> LinearModel:
    """Closed-form (optionally time-decay weighted) ridge on standardized features:
        coef = (Xs' W Xs + alpha I)^-1 Xs' W (y - y_bar)."""
    X = panel[feats].to_numpy(dtype=float)
    y = panel[label_name].to_numpy(dtype=float)
    Xs, mean, std = _standardize(X)

    w = (time_decay_weights(panel["date"], half_life_years).astype(float)
         if half_life_years is not None else np.ones(len(y)))
    y_bar = float(np.average(y, weights=w))
    yc = y - y_bar

    XtW = Xs.T * w
    A = XtW @ Xs + float(alpha) * np.eye(Xs.shape[1])
    b = XtW @ yc
    coef = np.linalg.solve(A, b)
    return LinearModel(coef, y_bar, mean, std, feats, "ridge")


def _enet_coordinate_descent(Xs: np.ndarray, y: np.ndarray, w: np.ndarray,
                             lam: float, l1_ratio: float,
                             max_iter: int, tol: float) -> np.ndarray:
    """Weighted elastic-net via cyclic coordinate descent (glmnet-style) on the
    objective
        (1/2sw) Σ w_i (y_i - Xs_i·β)^2 + lam*[ l1_ratio*||β||_1 + (1-l1_ratio)/2*||β||_2^2 ].
    Standard soft-threshold update per coordinate:
        β_j = S(ρ_j, lam*l1_ratio) / (z_j + lam*(1-l1_ratio)).
    L1 sets weak/redundant coefficients to EXACTLY zero (feature selection); L2
    shares weight smoothly across a correlated cluster instead of picking one
    arbitrarily -- the reason elastic net beats pure lasso (and pure ridge) when
    features are collinear."""
    n, k = Xs.shape
    sw = float(w.sum())
    z = np.array([float((w * Xs[:, j] ** 2).sum() / sw) for j in range(k)])
    z = np.where(z > 0, z, 1.0)
    l1, l2 = lam * l1_ratio, lam * (1.0 - l1_ratio)

    beta = np.zeros(k)
    r = y.astype(float).copy()                 # residual = y - Xs @ beta (beta = 0)
    for _ in range(max_iter):
        max_step = 0.0
        for j in range(k):
            bj = beta[j]
            rho = float((w * Xs[:, j] * r).sum() / sw) + bj * z[j]
            if rho > l1:
                nj = (rho - l1) / (z[j] + l2)
            elif rho < -l1:
                nj = (rho + l1) / (z[j] + l2)
            else:
                nj = 0.0
            if nj != bj:
                r += Xs[:, j] * (bj - nj)       # keep residual in sync
                beta[j] = nj
                max_step = max(max_step, abs(nj - bj))
        if max_step < tol:
            break
    return beta


def train_elasticnet(panel: pd.DataFrame, feats: list, label_name: str = "y",
                     alpha: float = 1e-3, l1_ratio: float = 0.5,
                     max_iter: int = 1000, tol: float = 1e-6,
                     half_life_years: float | None = None) -> LinearModel:
    """Elastic net (L1 + L2) via pure-numpy coordinate descent -- no scikit-learn
    dependency. `alpha` is the overall penalty on the normalized (1/2n) loss
    (glmnet scale, so ~1e-4..1e-1), `l1_ratio` the L1 fraction (0 = ridge,
    1 = lasso)."""
    X = panel[feats].to_numpy(dtype=float)
    y = panel[label_name].to_numpy(dtype=float)
    Xs, mean, std = _standardize(X)
    w = (time_decay_weights(panel["date"], half_life_years).astype(float)
         if half_life_years is not None else np.ones(len(y)))
    y_bar = float(np.average(y, weights=w))
    coef = _enet_coordinate_descent(Xs, y - y_bar, w, float(alpha), float(l1_ratio),
                                    int(max_iter), float(tol))
    # Guard the silent-degeneracy failure mode: if `alpha` is too high for the
    # target's scale, every feature's gradient |rho| falls below the L1 threshold
    # (alpha*l1_ratio) and ALL coefficients soft-threshold to exactly zero -> the
    # model predicts a CONSTANT (just y_bar) and contributes nothing to the
    # ensemble. This is what killed the linear member before; warn loudly instead
    # of shipping a dead model.
    n_nonzero = int(np.count_nonzero(np.abs(coef) > 0))
    if n_nonzero == 0:
        _log.warning(
            "elastic-net is DEGENERATE: all %d coefficients are zero (alpha=%.4g too "
            "high for the target scale -> every |rho| < alpha*l1_ratio=%.4g). The model "
            "will predict a CONSTANT; lower `alpha` in linear_modelling.yml.",
            len(coef), alpha, alpha * l1_ratio)
    return LinearModel(coef, y_bar, mean, std, feats, "elasticnet")


def train_linear(panel: pd.DataFrame, feats: list, label_name: str = "y",
                 kind: str = "elasticnet", alpha: float = 1e-3, l1_ratio: float = 0.5,
                 max_iter: int = 1000, tol: float = 1e-6,
                 half_life_years: float | None = None) -> LinearModel:
    if kind == "ridge":
        return train_ridge(panel, feats, label_name, alpha=alpha,
                           half_life_years=half_life_years)
    return train_elasticnet(panel, feats, label_name, alpha=alpha, l1_ratio=l1_ratio,
                            max_iter=max_iter, tol=tol, half_life_years=half_life_years)


def linear_importance(model: LinearModel) -> dict:
    """|coefficient| per feature (features are standardized, so comparable)."""
    return dict(zip(model.feature_names, np.abs(model.coef)))
