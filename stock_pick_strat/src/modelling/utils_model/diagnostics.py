"""
diagnostics.py  (src/modelling/utils_model/diagnostics.py)
----------------------------------------------------------
Per-run, per-horizon model-diagnosis artifacts. `save_run_diagnostics` lays out:

    <OUTPUT_DIR>/diagnostics/<run_stamp>/
        h<H>/
            pdp/pdp_01_<feature>.png ...        top-N individual partial-dependence plots
            shap_importance.png / .csv          top features by mean|SHAP| (SHAP only)
            feature_importance.xlsx (or .csv)   LightGBM gain (+ SHAP) importance, this horizon
            ic_over_time.png / ic_over_time.csv  OOS daily IC, CV folds concatenated over time

Design notes
  * Partial dependence is computed directly from the LightGBM booster (vary one
    feature across its own quantile grid, hold the rest, average the score), so
    it needs no extra dependency.
  * SHAP and the Excel engine are OPTIONAL: if `shap` (or an .xlsx writer) is not
    installed the corresponding artifact is skipped / falls back to CSV and the
    rest still render -- the pipeline never fails on a missing optional dependency.
  * The IC-over-time curve uses the concatenated out-of-sample CV predictions
    (each walk-forward fold contributes its own later test window), giving a true
    time-dependent skill curve rather than a single in-sample number.
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")           # headless: save figures without a display
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np               # noqa: E402
import pandas as pd              # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from src.modelling.utils_model import model as ml  # noqa: E402


def _safe(name: str) -> str:
    """Filesystem-safe version of a feature name (e.g. 'beta_USD/EUR')."""
    return re.sub(r"[^0-9A-Za-z._-]+", "_", name)


def _sample_rows(x: np.ndarray, sample: int, seed: int = 42) -> np.ndarray:
    if sample and len(x) > sample:
        rng = np.random.default_rng(seed)
        return x[rng.choice(len(x), sample, replace=False)]
    return x


# --------------------------------------------------------------------------- #
# Partial dependence (booster-only, no extra deps)                             #
# --------------------------------------------------------------------------- #
def partial_dependence(booster, x: np.ndarray, feat_idx: int,
                       grid_points: int = 30, sample: int = 2000) -> tuple:
    """1-D partial dependence: average model score as `feat_idx` is swept across
    its own 2-98% quantile grid, all other features held at their real values."""
    x = _sample_rows(x, sample)
    col = x[:, feat_idx]
    finite = col[np.isfinite(col)]
    if finite.size == 0:
        return None, None
    grid = np.unique(np.quantile(finite, np.linspace(0.02, 0.98, grid_points)))
    if grid.size < 2:
        return None, None
    means = np.empty(grid.size, dtype=float)
    work = x.copy()
    for i, g in enumerate(grid):
        work[:, feat_idx] = g
        means[i] = float(booster.predict(work).mean())
    return grid, means


def save_pdp_plot(grid: np.ndarray, means: np.ndarray, feat: str,
                  path: Path, horizon, feature_values: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(grid, means, color="steelblue", lw=2, marker="o", ms=3)
    # rug of the real feature distribution (deciles) along the x-axis
    fin = feature_values[np.isfinite(feature_values)]
    if fin.size:
        for q in np.quantile(fin, np.linspace(0.1, 0.9, 9)):
            ax.axvline(q, color="grey", alpha=0.15, lw=0.8)
    ax.set_xlabel(feat)
    ax.set_ylabel("average model score")
    ax.set_title(f"Partial dependence — {feat} (horizon {horizon})", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# SHAP importance (optional dependency)                                        #
# --------------------------------------------------------------------------- #
def shap_importance(booster, x: np.ndarray, feature_cols: list[str],
                    sample: int = 2000) -> pd.Series | None:
    """Mean |SHAP| per feature (descending), or None if `shap` is unavailable."""
    try:
        import shap  # optional; imported lazily so the module loads without it
    except Exception:
        return None
    try:
        xs = _sample_rows(x, sample)
        values = shap.TreeExplainer(booster).shap_values(xs)
        if isinstance(values, list):
            values = values[0]
        return pd.Series(np.abs(values).mean(axis=0),
                         index=feature_cols).sort_values(ascending=False)
    except Exception:
        return None


def save_shap_importance_plot(shap_imp: pd.Series, path: Path, horizon, top_n: int) -> None:
    fig, ax = plt.subplots(figsize=(9, max(4, top_n * 0.32)))
    shap_imp.head(top_n).iloc[::-1].plot(kind="barh", ax=ax, color="seagreen")
    ax.set_xlabel("mean |SHAP value|")
    ax.set_title(f"Top {top_n} features by SHAP — horizon {horizon}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Feature-importance table (Excel, CSV fallback)                               #
# --------------------------------------------------------------------------- #
def save_importance_table(gain_imp: pd.Series, shap_imp: pd.Series | None,
                          out_dir: Path) -> Path:
    """Per-horizon importance to Excel; falls back to CSV if no .xlsx engine."""
    df = pd.DataFrame({"lgbm_gain": gain_imp.astype(float)})
    total = df["lgbm_gain"].sum()
    df["lgbm_gain_pct"] = df["lgbm_gain"] / total if total else np.nan
    if shap_imp is not None:
        df["shap_mean_abs"] = shap_imp.reindex(df.index)
    df = df.sort_values("lgbm_gain", ascending=False)
    df.index.name = "feature"

    xlsx = out_dir / "feature_importance.xlsx"
    try:
        df.to_excel(xlsx)          # needs openpyxl / xlsxwriter
        return xlsx
    except Exception:
        csv = out_dir / "feature_importance.csv"
        df.to_csv(csv)
        return csv


# --------------------------------------------------------------------------- #
# Out-of-sample IC over time (CV folds concatenated)                           #
# --------------------------------------------------------------------------- #
def daily_ic_series(oos: pd.DataFrame, label_name: str, pred_col: str = "pred") -> pd.Series:
    """Per-day cross-sectional Spearman IC over the concatenated OOS predictions."""
    rows = {}
    for d, g in oos.groupby("date", sort=True):
        if g[pred_col].nunique() > 2 and g[label_name].nunique() > 2:
            ic, _ = spearmanr(g[pred_col], g[label_name])
            if np.isfinite(ic):
                rows[d] = ic
    return pd.Series(rows, name="ic").sort_index()


def save_ic_curve(oos: pd.DataFrame, label_name: str, out_dir: Path,
                  horizon, roll: int = 21) -> pd.Series:
    ic = daily_ic_series(oos, label_name)
    if ic.empty:
        return ic
    ic.index = pd.to_datetime(ic.index)
    ic.to_csv(out_dir / "ic_over_time.csv")

    mean_ic = ic.mean()
    hit = float((ic > 0).mean())
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    ax1.plot(ic.index, ic.values, color="steelblue", lw=0.6, alpha=0.5, label="daily IC")
    if len(ic) >= roll:
        ax1.plot(ic.index, ic.rolling(roll, min_periods=1).mean().values,
                 color="crimson", lw=1.6, label=f"{roll}d rolling mean")
    ax1.axhline(0, color="black", lw=0.8)
    ax1.axhline(mean_ic, color="green", ls="--", lw=1,
                label=f"mean IC={mean_ic:+.4f}")
    ax1.set_ylabel("daily IC")
    ax1.set_title(f"Out-of-sample IC over time — horizon {horizon} "
                  f"(mean={mean_ic:+.4f}, hit-rate={hit:.0%}, n={len(ic)} days)", fontsize=10)
    ax1.legend(fontsize=8, loc="upper left")
    ax2.plot(ic.index, ic.cumsum().values, color="darkorange", lw=1.4)
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_ylabel("cumulative IC")
    ax2.set_xlabel("date")
    fig.tight_layout()
    fig.savefig(out_dir / "ic_over_time.png", dpi=140)
    plt.close(fig)
    return ic


# --------------------------------------------------------------------------- #
# Orchestration                                                                #
# --------------------------------------------------------------------------- #
def save_horizon_diagnostics(horizon, booster, panel: pd.DataFrame,
                             feature_cols: list[str], out_dir: Path,
                             oos_predictions: pd.DataFrame | None,
                             label_name: str = "y", top_n: int = 15,
                             shap_sample: int = 2000, pdp_grid: int = 30,
                             logger=None) -> dict:
    """Write every diagnostic artifact for one horizon. Returns a small summary."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    x = panel[feature_cols].to_numpy(dtype="float32")

    gain_imp = pd.Series(ml.feature_importance(booster, feature_cols)).sort_values(ascending=False)
    shap_imp = shap_importance(booster, x, feature_cols, sample=shap_sample)

    # rank for the PDP selection: SHAP if available, else LightGBM gain
    ranking = shap_imp if shap_imp is not None else gain_imp
    top_feats = list(ranking.head(top_n).index)

    # 1. top-N individual partial-dependence plots
    pdp_dir = out_dir / "pdp"
    pdp_dir.mkdir(exist_ok=True)
    n_pdp = 0
    for rank, feat in enumerate(top_feats, 1):
        grid, means = partial_dependence(booster, x, feature_cols.index(feat),
                                         grid_points=pdp_grid, sample=shap_sample)
        if grid is None:
            continue
        save_pdp_plot(grid, means, feat, pdp_dir / f"pdp_{rank:02d}_{_safe(feat)}.png",
                      horizon, x[:, feature_cols.index(feat)])
        n_pdp += 1

    # 2. SHAP importance (SHAP only) — skipped if shap unavailable
    if shap_imp is not None:
        save_shap_importance_plot(shap_imp, out_dir / "shap_importance.png", horizon, top_n)
        shap_imp.rename("shap_mean_abs").to_csv(out_dir / "shap_importance.csv")
    elif logger is not None:
        logger.warning("h%s diagnostics: shap not available -> SHAP importance skipped", horizon)

    # 3. per-horizon feature-importance table (Excel, CSV fallback)
    imp_path = save_importance_table(gain_imp, shap_imp, out_dir)

    # 4. OOS IC over time (CV folds concatenated)
    ic = None
    if oos_predictions is not None and not oos_predictions.empty:
        ic = save_ic_curve(oos_predictions, label_name, out_dir, horizon)
    elif logger is not None:
        logger.warning("h%s diagnostics: no OOS predictions -> IC-over-time skipped", horizon)

    return {
        "n_pdp": n_pdp,
        "shap_available": shap_imp is not None,
        "importance_path": imp_path.name,
        "ic_days": 0 if ic is None else int(len(ic)),
        "ic_mean": float(ic.mean()) if ic is not None and len(ic) else float("nan"),
    }


def save_run_diagnostics(run_dir: Path, models: dict, panels: dict,
                         feature_cols: list[str], oos_predictions: dict,
                         label_name: str = "y", top_n: int = 15,
                         shap_sample: int = 2000, pdp_grid: int = 30,
                         logger=None) -> Path:
    """Create <run_dir>/h<H>/ diagnostics for every trained horizon."""
    run_dir = Path(run_dir)
    for h, booster in models.items():
        summary = save_horizon_diagnostics(
            horizon=h, booster=booster, panel=panels[h],
            feature_cols=feature_cols, out_dir=run_dir / f"h{h}",
            oos_predictions=oos_predictions.get(h),
            label_name=label_name, top_n=top_n,
            shap_sample=shap_sample, pdp_grid=pdp_grid, logger=logger,
        )
        if logger is not None:
            logger.info("  h%s diagnostics: %s PDPs, shap=%s, IC days=%s (mean IC=%+.4f)",
                        h, summary["n_pdp"], summary["shap_available"],
                        summary["ic_days"], summary["ic_mean"])
    return run_dir
