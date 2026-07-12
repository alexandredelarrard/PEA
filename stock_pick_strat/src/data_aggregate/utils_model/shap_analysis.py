"""
SHAP explainability for LightGBM rankers.

Computes SHAP values on a held-out CV fold, saves feature importance and
partial-dependence-style plots for the top features.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.data_aggregate.utils_model import model as ml


def _sample_panel(panel: pd.DataFrame, max_samples: int, seed: int = 42) -> pd.DataFrame:
    if len(panel) <= max_samples:
        return panel
    return panel.sample(n=max_samples, random_state=seed)


def _compute_shap(
    booster,
    panel: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    x = panel[feature_cols].to_numpy(dtype="float32")
    explainer = shap.TreeExplainer(booster)
    shap_values = explainer.shap_values(x)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    return shap_values, x


def shap_feature_importance(
    booster,
    panel: pd.DataFrame,
    feature_cols: list[str],
    *,
    max_samples: int = 5000,
) -> pd.Series:
    """Mean |SHAP| per feature, descending."""
    panel = _sample_panel(panel, max_samples)
    shap_values, _ = _compute_shap(booster, panel, feature_cols)
    mean_abs = np.abs(shap_values).mean(axis=0)
    return pd.Series(mean_abs, index=feature_cols).sort_values(ascending=False)


def _plot_dependence(
    ax: plt.Axes,
    feat: str,
    x: np.ndarray,
    shap_values: np.ndarray,
    feature_cols: list[str],
) -> None:
    idx = feature_cols.index(feat)
    xv = x[:, idx]
    sv = shap_values[:, idx]
    ax.scatter(xv, sv, s=4, alpha=0.25, c="steelblue", edgecolors="none")
    order = np.argsort(xv)
    xv_s, sv_s = xv[order], sv[order]
    window = max(50, len(xv_s) // 50)
    if len(xv_s) > window:
        trend = pd.Series(sv_s).rolling(window, center=True, min_periods=1).mean()
        ax.plot(xv_s, trend.to_numpy(), color="crimson", lw=1.5)
    ax.set_xlabel(feat, fontsize=8)
    ax.set_ylabel("SHAP", fontsize=8)
    ax.tick_params(labelsize=7)


def save_shap_analysis(
    booster,
    panel: pd.DataFrame,
    feature_cols: list[str],
    out_dir: Path,
    *,
    top_n: int = 20,
    max_samples: int = 5000,
    horizon: int | None = None,
) -> pd.Series:
    """
    Save SHAP + LightGBM feature importance and top-N dependence plots.

    Returns the SHAP importance series (mean |SHAP|).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"h{horizon}_" if horizon is not None else ""

    panel = _sample_panel(panel, max_samples)
    shap_values, x = _compute_shap(booster, panel, feature_cols)

    shap_imp = pd.Series(
        np.abs(shap_values).mean(axis=0), index=feature_cols,
    ).sort_values(ascending=False)
    gain_imp = pd.Series(ml.feature_importance(booster, feature_cols)).sort_values(
        ascending=False,
    )

    imp_df = pd.DataFrame({
        "feature": shap_imp.index,
        "shap_mean_abs": shap_imp.values,
        "lgbm_gain": [gain_imp.get(f, np.nan) for f in shap_imp.index],
    })
    imp_df.to_csv(out_dir / f"{prefix}shap_feature_importance.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.35)))
    shap_imp.head(top_n).iloc[::-1].plot(kind="barh", ax=ax, color="steelblue")
    ax.set_xlabel("mean |SHAP value|")
    title = f"Top {top_n} features by SHAP (last CV fold)"
    if horizon is not None:
        title += f" — horizon {horizon}"
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}shap_feature_importance.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.35)))
    gain_imp.head(top_n).iloc[::-1].plot(kind="barh", ax=ax, color="darkorange")
    ax.set_xlabel("LightGBM gain")
    ax.set_title(title.replace("SHAP", "LightGBM gain"))
    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}lgbm_feature_importance.png", dpi=150)
    plt.close(fig)

    top_features = shap_imp.head(top_n).index.tolist()
    ncols = 4
    nrows = int(np.ceil(top_n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.2 * nrows))
    for ax, feat in zip(np.atleast_1d(axes).flatten(), top_features):
        _plot_dependence(ax, feat, x, shap_values, feature_cols)
        ax.set_title(feat, fontsize=9)
    for ax in np.atleast_1d(axes).flatten()[len(top_features):]:
        ax.axis("off")

    dep_title = f"SHAP partial dependence — top {top_n} (last CV fold)"
    if horizon is not None:
        dep_title += f" — horizon {horizon}"
    fig.suptitle(dep_title, fontsize=12)
    fig.tight_layout()
    fig.savefig(
        out_dir / f"{prefix}shap_dependence_top{top_n}.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)

    return shap_imp
