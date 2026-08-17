"""
diagnostics.py  (src/modelling/long_short/utils/diagnostics.py)
---------------------------------------------------------------
Per-run, per-horizon model-diagnosis artifacts. `save_run_diagnostics` lays out:

    <OUTPUT_DIR>/diagnostics/<run_stamp>/
        kpis.json / kpis.csv                     RUN-level key KPIs, one row per (horizon, member)
        h<H>/
            kpis.json                            this horizon's KPIs (IC, IC_IR, top features)
            ic_over_time.png / .csv              OOS daily IC, CV folds concatenated over time
            <member>/                            ONE folder per BOOSTER member (lgbm, random_forest)
                pdp/pdp_01_<feature>.png ...     top-N individual partial-dependence plots
                shap_values.parquet              the RAW per-row SHAP matrix (date, ticker, features)
                shap_importance.png / .csv       top features by mean|SHAP|
                feature_importance.xlsx (or .csv) LightGBM gain (+ SHAP) importance

Design notes
  * Partial dependence is computed directly from the LightGBM booster (vary one
    feature across its own quantile grid, hold the rest, average the score), so
    it needs no extra dependency.
  * SHAP is computed ONCE per member: `shap_row_values` returns the raw matrix,
    and the mean|SHAP| ranking is derived from it. The raw matrix is persisted
    (parquet) because the ranking alone cannot answer "why is THIS name long
    today" -- the per-row attribution can.
  * The Excel engine is OPTIONAL (falls back to CSV). `shap` is a DECLARED
    dependency, so its absence is a real environment fault and is logged as a
    WARNING with the install hint rather than passed over in silence -- a silently
    missing SHAP artifact is exactly how this went unnoticed before.
  * The IC-over-time curve uses the concatenated out-of-sample CV predictions
    (each walk-forward fold contributes its own later test window), giving a true
    time-dependent skill curve rather than a single in-sample number. It is a
    property of the ENSEMBLE, so it sits at the horizon level, not per member.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")           # headless: save figures without a display
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np               # noqa: E402
import pandas as pd              # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from src.modelling.long_short.utils import model as ml  # noqa: E402

try:                             # declared dependency (pyproject + requirements-airflow),
    import shap                  # guarded ONLY so a stale venv degrades to "no SHAP artifact"
except ImportError:              # instead of killing the whole training run at import time.
    shap = None                  # `shap_row_values` logs a WARNING with the install hint.


def _safe(name: str) -> str:
    """Filesystem-safe version of a feature name (e.g. 'beta_USD/EUR')."""
    return re.sub(r"[^0-9A-Za-z._-]+", "_", name)


def _sample_idx(n_rows: int, sample: int, seed: int = 42) -> np.ndarray:
    """Row POSITIONS to diagnose. Returns indices (not the rows themselves) so the
    SHAP matrix can be joined back to the panel's date/ticker keys."""
    if sample and n_rows > sample:
        return np.random.default_rng(seed).choice(n_rows, sample, replace=False)
    return np.arange(n_rows)


def _sample_rows(x: np.ndarray, sample: int, seed: int = 42) -> np.ndarray:
    return x[_sample_idx(len(x), sample, seed)]


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
# SHAP: raw per-row values + the mean|SHAP| ranking derived from them           #
# --------------------------------------------------------------------------- #
def shap_row_values(booster, x: np.ndarray, feature_cols: list[str],
                    sample: int = 2000, seed: int = 42,
                    logger=None) -> tuple[np.ndarray, np.ndarray] | None:
    """`(shap_matrix, row_idx)` for a sampled subset of `x`, or None when unavailable.

    `shap_matrix` is (n_sampled, n_features) of signed SHAP contributions and `row_idx`
    are the sampled row POSITIONS in `x`, so a caller can attach the panel's date/ticker.
    Computed once and shared by the persisted matrix, the mean|SHAP| ranking and the PDP
    feature selection -- SHAP on a 12k x 500 panel is the expensive part of a run.

    Failures are LOGGED, never raised: diagnostics must not fail a training run. But they
    are logged at WARNING (with the install hint) because `shap` is a declared dependency,
    so its absence means the environment is wrong -- most likely the Airflow venv."""
    if shap is None:
        if logger is not None:
            logger.warning("SHAP artifacts skipped: `shap` is not installed in this "
                           "environment (declared in pyproject / requirements-airflow.txt "
                           "-- reinstall the venv: `poetry install`).")
        return None
    try:
        idx = _sample_idx(len(x), sample, seed)
        values = shap.TreeExplainer(booster).shap_values(x[idx])
        if isinstance(values, list):                 # multi-output -> first output
            values = values[0]
        values = np.asarray(values)
        if values.shape[1] == len(feature_cols) + 1:  # trailing base-value column
            values = values[:, :-1]
        return values, idx
    except Exception as exc:                          # noqa: BLE001
        if logger is not None:
            logger.warning("SHAP computation failed (%s: %s) -> SHAP artifacts skipped",
                           type(exc).__name__, exc)
        return None


def shap_importance_from_values(values: np.ndarray,
                                feature_cols: list[str]) -> pd.Series:
    """Mean |SHAP| per feature, descending -- the global ranking."""
    return pd.Series(np.abs(values).mean(axis=0),
                     index=feature_cols).sort_values(ascending=False)


def save_shap_values(values: np.ndarray, row_idx: np.ndarray, feature_cols: list[str],
                     panel: pd.DataFrame, path: Path) -> Path:
    """Persist the RAW SHAP matrix keyed by (date, ticker) -> parquet.

    This is what makes an attribution question answerable after the run ("which features
    pushed this name into the long book on this date?"). The mean|SHAP| CSV alone averages
    that away. Parquet (not CSV) because it is float32 x n_features x n_sampled rows."""
    out = pd.DataFrame(values.astype("float32"), columns=list(feature_cols))
    keys = panel.iloc[row_idx]
    for key in ("ticker", "date"):                    # insert at front: date, ticker, ...
        if key in keys.columns:
            out.insert(0, key, keys[key].to_numpy())
    out.to_parquet(path, index=False)
    return path


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
# Key KPIs (machine-readable, per horizon and per run)                          #
# --------------------------------------------------------------------------- #
def _jsonable(value):
    """numpy scalars / NaN -> plain JSON (NaN is not valid JSON; emit null)."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(float(value)) else float(value)
    return value


def save_horizon_kpis(out_dir: Path, kpis: dict) -> Path:
    """One horizon's KPIs -> `kpis.json`.

    Written per horizon INSIDE the loop so an interrupted run keeps the horizons it
    already finished (a full training run is hours; losing all its KPIs because the
    last horizon failed is the difference between a usable and a wasted run)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "kpis.json"
    path.write_text(json.dumps(kpis, indent=2, default=_jsonable), encoding="utf-8")
    return path


def save_run_kpis(run_dir: Path, run_kpis: dict, logger=None) -> Path:
    """Run-level KPIs -> `kpis.json` (nested, full detail) + `kpis.csv` (FLAT, one row per
    (horizon, member): mean_ic / ic_ir / blend weight / rows / OOS IC).

    The CSV is the artifact to eyeball or diff between runs; the JSON keeps the nested
    detail (top features per member, train window, config echo)."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "kpis.json"
    path.write_text(json.dumps(run_kpis, indent=2, default=_jsonable), encoding="utf-8")

    rows = []
    for h, hk in sorted((run_kpis.get("horizons") or {}).items(), key=lambda kv: int(kv[0])):
        common = {
            "horizon": int(h),
            "blend_weight": hk.get("blend_weight"),
            "n_rows": hk.get("n_rows"), "n_tickers": hk.get("n_tickers"),
            "n_days": hk.get("n_days"),
            "oos_ic_mean": hk.get("oos_ic_mean"), "oos_ic_hit_rate": hk.get("oos_ic_hit_rate"),
            "oos_ic_days": hk.get("oos_ic_days"),
        }
        rows.append({**common, "member": "ENSEMBLE",
                     "cv_mean_ic": hk.get("cv_mean_ic"), "cv_ic_ir": hk.get("cv_ic_ir"),
                     "n_features": None, "n_pdp": None, "shap_available": None})
        for name, mk in sorted((hk.get("members") or {}).items()):
            rows.append({**common, "member": name,
                         "cv_mean_ic": mk.get("cv_mean_ic"), "cv_ic_ir": mk.get("cv_ic_ir"),
                         "n_features": mk.get("n_features"), "n_pdp": mk.get("n_pdp"),
                         "shap_available": mk.get("shap_available")})
    csv_path = run_dir / "kpis.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    if logger is not None:
        logger.info("Diagnostics KPIs: %d row(s) across %d horizon(s) -> %s",
                    len(rows), len(run_kpis.get("horizons") or {}), csv_path)
    return path


# --------------------------------------------------------------------------- #
# Orchestration                                                                #
# --------------------------------------------------------------------------- #
def save_member_diagnostics(horizon, member: str, booster, panel: pd.DataFrame,
                            feature_cols: list[str], out_dir: Path,
                            top_n: int = 15, shap_sample: int = 2000,
                            pdp_grid: int = 30, logger=None) -> dict:
    """PDP + SHAP (raw values, ranking, plot) + gain table for ONE booster member.

    Every booster member is worth its own folder: with `ensemble: [elasticnet, lgbm,
    random_forest]` the two tree members can rank features very differently, and only
    diagnosing one of them (or, as before, a member name that no longer exists) hides that."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # the SAME encoding the model was fitted on (categoricals -> numeric codes), so SHAP and
    # PDP explain the model rather than a re-encoding of the panel
    x = ml.design_matrix(panel, feature_cols)

    gain_imp = pd.Series(ml.feature_importance(booster, feature_cols)).sort_values(ascending=False)

    shap_imp, n_shap = None, 0
    got = shap_row_values(booster, x, feature_cols, sample=shap_sample, logger=logger)
    if got is not None:
        values, row_idx = got
        shap_imp = shap_importance_from_values(values, feature_cols)
        save_shap_values(values, row_idx, feature_cols, panel, out_dir / "shap_values.parquet")
        save_shap_importance_plot(shap_imp, out_dir / "shap_importance.png", horizon, top_n)
        shap_imp.rename("shap_mean_abs").to_csv(out_dir / "shap_importance.csv")
        n_shap = int(len(row_idx))

    # rank for the PDP selection: SHAP if available, else gain
    ranking = shap_imp if shap_imp is not None else gain_imp
    top_feats = list(ranking.head(top_n).index)

    pdp_dir = out_dir / "pdp"
    pdp_dir.mkdir(exist_ok=True)
    n_pdp = 0
    for rank, feat in enumerate(top_feats, 1):
        j = feature_cols.index(feat)
        grid, means = partial_dependence(booster, x, j, grid_points=pdp_grid,
                                         sample=shap_sample)
        if grid is None:
            continue
        save_pdp_plot(grid, means, feat,
                      pdp_dir / f"pdp_{rank:02d}_{_safe(feat)}.png", horizon, x[:, j])
        n_pdp += 1

    imp_path = save_importance_table(gain_imp, shap_imp, out_dir)
    return {
        "member": member,
        "n_features": len(feature_cols),
        "n_pdp": n_pdp,
        "shap_available": shap_imp is not None,
        "shap_rows": n_shap,
        "importance_path": imp_path.name,
        "top_features_shap": list(ranking.head(top_n).index) if shap_imp is not None else [],
        "top_features_gain": list(gain_imp.head(top_n).index),
    }


def save_horizon_diagnostics(horizon, booster, panel: pd.DataFrame,
                             feature_cols: list[str], out_dir: Path,
                             oos_predictions: pd.DataFrame | None,
                             label_name: str = "y", top_n: int = 15,
                             shap_sample: int = 2000, pdp_grid: int = 30,
                             logger=None, boosters: dict | None = None,
                             feature_cols_by_member: dict | None = None,
                             kpis: dict | None = None) -> dict:
    """Every artifact for one horizon: the OOS IC curve (ensemble-level) + one folder per
    booster member + `kpis.json`.

    `boosters` / `feature_cols_by_member` carry the FULL member set; the single
    `booster` / `feature_cols` pair is the one-member shorthand (kept so a caller with
    just one model -- tests, ad-hoc analysis -- needs no dict plumbing)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    members = dict(boosters) if boosters else {"model": booster}
    feats_by = dict(feature_cols_by_member or {})

    # 1. OOS IC over time — a property of the ENSEMBLE, so once per horizon
    ic = None
    if oos_predictions is not None and not oos_predictions.empty:
        ic = save_ic_curve(oos_predictions, label_name, out_dir, horizon)
    elif logger is not None:
        logger.warning("h%s diagnostics: no OOS predictions -> IC-over-time skipped", horizon)

    # 2. per-member PDP / SHAP / importance. A single member keeps the FLAT layout
    #    (h<H>/pdp/...) so existing readers of that path still work; two or more get
    #    their own sub-folder, since they would otherwise overwrite each other.
    flat = len(members) == 1
    member_summaries = {}
    for name, b in members.items():
        member_summaries[name] = save_member_diagnostics(
            horizon=horizon, member=name, booster=b, panel=panel,
            feature_cols=list(feats_by.get(name, feature_cols)),
            out_dir=out_dir if flat else out_dir / _safe(name),
            top_n=top_n, shap_sample=shap_sample, pdp_grid=pdp_grid, logger=logger,
        )

    first = next(iter(member_summaries.values()))
    summary = {
        "horizon": int(horizon),
        "n_rows": int(len(panel)),
        "n_tickers": int(panel["ticker"].nunique()) if "ticker" in panel else None,
        "n_days": int(panel["date"].nunique()) if "date" in panel else None,
        "oos_ic_days": 0 if ic is None else int(len(ic)),
        "oos_ic_mean": float(ic.mean()) if ic is not None and len(ic) else float("nan"),
        "oos_ic_hit_rate": float((ic > 0).mean()) if ic is not None and len(ic) else float("nan"),
        "members": member_summaries,
        # flat back-compat keys, so the old summary contract still holds for one member
        "n_pdp": first["n_pdp"],
        "shap_available": first["shap_available"],
        "importance_path": first["importance_path"],
        "ic_days": 0 if ic is None else int(len(ic)),
        "ic_mean": float(ic.mean()) if ic is not None and len(ic) else float("nan"),
    }
    if kpis:                                    # CV IC / blend weight known by the caller
        for key, value in kpis.items():
            if key == "members":
                for name, mk in (value or {}).items():
                    summary["members"].setdefault(name, {}).update(mk)
            else:
                summary[key] = value
    save_horizon_kpis(out_dir, summary)
    return summary


def save_run_diagnostics(run_dir: Path, models: dict, panels: dict,
                         feature_cols: list[str], oos_predictions: dict,
                         label_name: str = "y", top_n: int = 15,
                         shap_sample: int = 2000, pdp_grid: int = 30,
                         logger=None) -> Path:
    """Create <run_dir>/h<H>/ diagnostics for every trained horizon.

    `models[h]` is either ONE booster or a `{member: booster}` dict -- the per-horizon
    ensemble, which is what `StepModelling` passes."""
    run_dir = Path(run_dir)
    summaries = {}
    for h, entry in models.items():
        boosters = entry if isinstance(entry, dict) else None
        summaries[int(h)] = save_horizon_diagnostics(
            horizon=h, booster=None if boosters else entry, panel=panels[h],
            feature_cols=feature_cols, out_dir=run_dir / f"h{h}",
            oos_predictions=oos_predictions.get(h), boosters=boosters,
            label_name=label_name, top_n=top_n,
            shap_sample=shap_sample, pdp_grid=pdp_grid, logger=logger,
        )
        if logger is not None:
            s = summaries[int(h)]
            logger.info("  h%s diagnostics: %s, IC days=%s (mean IC=%+.4f)", h,
                        ", ".join(f"{n}: {m['n_pdp']} PDPs shap={m['shap_available']}"
                                  for n, m in s["members"].items()),
                        s["ic_days"], s["ic_mean"])
    save_run_kpis(run_dir, {"horizons": summaries}, logger=logger)
    return run_dir
