"""
monitor.py  (src/modelling/transformers/monitor.py)
---------------------------------------------------
`Monitor`: every training metric and report artifact of one run, under

    <OUTPUT_DIR>/diagnostics/<run_stamp>/
        kpis.json / kpis.csv                 RUN level: one CSV row per (horizon, member)
        h<H>/
            kpis.json                        this horizon: OOS IC (+ drawdown), CV IC, members
            ic_over_time.png / .csv          OOS daily IC, rolling mean, cumulative IC, drawdown
            backtest_*.csv / backtest.png    label-only quantile backtest (see backtest.py)
            coef_importance_<member>.csv     |coef| of each linear member
            [<member>/]                      per BOOSTER member; flat in h<H>/ when there is one
                pdp/pdp_NN_<feature>.png     top-N partial-dependence plots
                shap_values.parquet          RAW per-row SHAP matrix keyed by (date, ticker)
                shap_importance.png / .csv   top features by mean|SHAP|
                shap_dependence.png          feature value vs SHAP for the top features
                feature_importance.xlsx/csv  LightGBM gain (+ mean|SHAP|)

Reporting is best-effort: a failed artifact is logged at WARNING and never fails training.
`scripts/dod/modelling_report.py` reads this layout and these KPI keys: members come from
`kpis.json["members"]`, a member with `shap_available` / `n_pdp` is a booster, and the member
folder is flat exactly when the horizon has one booster.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless: save figures without a display
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from omegaconf import DictConfig  # noqa: E402

from src.context import Context  # noqa: E402
from src.modelling.transformers.backtest import Backtest, BacktestResult  # noqa: E402
from src.modelling.transformers.base import BaseModel  # noqa: E402
from src.modelling.utils.artifacts import safe_filename  # noqa: E402
from src.modelling.utils.features import design_matrix  # noqa: E402
from src.modelling.utils.metrics import daily_ic_series, max_drawdown  # noqa: E402

try:  # declared dependency; guarded only so a stale venv degrades to "no SHAP artifact"
    import shap
except ImportError:
    shap = None

_DEPENDENCE_PANELS = 16  # features in the SHAP dependence grid (4 x 4)
_ROLL = 21


# --------------------------------------------------------------------------- #
# Partial dependence (booster-only)                                           #
# --------------------------------------------------------------------------- #
def _sample_idx(n_rows: int, sample: int, seed: int = 42) -> np.ndarray:
    """Row POSITIONS to diagnose, so the SHAP matrix can be joined back to date/ticker keys."""
    if sample and n_rows > sample:
        return np.random.default_rng(seed).choice(n_rows, sample, replace=False)
    return np.arange(n_rows)


def partial_dependence(booster: Any, x: np.ndarray, feat_idx: int, grid_points: int = 30, sample: int = 2000) -> tuple:
    """1-D partial dependence: mean model score as `feat_idx` sweeps its own 2-98 % quantile grid,
    every other feature held at its real value."""
    x = x[_sample_idx(len(x), sample)]
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


def _save_pdp_plot(grid: np.ndarray, means: np.ndarray, feat: str, path: Path, horizon: int, feature_values: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(grid, means, color="steelblue", lw=2, marker="o", ms=3)
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
# SHAP                                                                        #
# --------------------------------------------------------------------------- #
def shap_row_values(
    booster: Any, x: np.ndarray, feature_cols: list[str], sample: int = 2000, seed: int = 42, logger: logging.Logger | None = None
) -> tuple[np.ndarray, np.ndarray] | None:
    """`(shap_matrix, row_idx)` for a sampled subset of `x`, or None when unavailable; computed
    once and shared by the persisted matrix, the ranking, the dependence plot and the PDP choice."""
    if shap is None:
        if logger is not None:
            logger.warning(
                "SHAP artifacts skipped: `shap` is not installed in this environment (declared in pyproject -- reinstall the venv: `poetry install`)."
            )
        return None
    try:
        idx = _sample_idx(len(x), sample, seed)
        values = shap.TreeExplainer(booster).shap_values(x[idx])
        if isinstance(values, list):  # multi-output -> first output
            values = values[0]
        values = np.asarray(values)
        if values.shape[1] == len(feature_cols) + 1:  # trailing base-value column
            values = values[:, :-1]
        return values, idx
    except Exception as exc:  # noqa: BLE001 - SHAP is a reporting library boundary
        if logger is not None:
            logger.warning("SHAP computation failed (%s: %s) -> SHAP artifacts skipped", type(exc).__name__, exc)
        return None


def shap_importance_from_values(values: np.ndarray, feature_cols: list[str]) -> pd.Series:
    """Mean |SHAP| per feature, descending."""
    return pd.Series(np.abs(values).mean(axis=0), index=feature_cols).sort_values(ascending=False)


def save_shap_values(values: np.ndarray, row_idx: np.ndarray, feature_cols: list[str], panel: pd.DataFrame, path: Path) -> Path:
    """The RAW SHAP matrix keyed by (date, ticker) -> parquet: answers "why is THIS name long today"."""
    out = pd.DataFrame(values.astype("float32"), columns=list(feature_cols))
    keys = panel.iloc[row_idx]
    for key in ("ticker", "date"):
        if key in keys.columns:
            out.insert(0, key, keys[key].to_numpy())
    out.to_parquet(path, index=False)
    return path


def _save_shap_importance_plot(shap_imp: pd.Series, path: Path, horizon: int, top_n: int) -> None:
    fig, ax = plt.subplots(figsize=(9, max(4, top_n * 0.32)))
    shap_imp.head(top_n).iloc[::-1].plot(kind="barh", ax=ax, color="seagreen")
    ax.set_xlabel("mean |SHAP value|")
    ax.set_title(f"Top {top_n} features by SHAP — horizon {horizon}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _save_shap_dependence_plot(values: np.ndarray, x_sampled: np.ndarray, feature_cols: list[str], top: list[str], path: Path, horizon: int) -> None:
    """Grid of feature value vs SHAP contribution (with a rolling-mean trend) for `top` features."""
    feats = top[:_DEPENDENCE_PANELS]
    ncols = 4
    nrows = int(np.ceil(len(feats) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.2 * nrows))
    flat = np.atleast_1d(axes).flatten()
    for ax, feat in zip(flat, feats, strict=False):
        j = feature_cols.index(feat)
        xv, sv = x_sampled[:, j], values[:, j]
        ax.scatter(xv, sv, s=4, alpha=0.25, c="steelblue", edgecolors="none")
        keep = np.isfinite(xv)
        order = np.argsort(xv[keep])
        xs, ss = xv[keep][order], sv[keep][order]
        window = max(50, len(xs) // 50)
        if len(xs) > window:
            ax.plot(xs, pd.Series(ss).rolling(window, center=True, min_periods=1).mean().to_numpy(), color="crimson", lw=1.5)
        ax.set_title(feat, fontsize=9)
        ax.tick_params(labelsize=7)
    for ax in flat[len(feats) :]:
        ax.axis("off")
    fig.suptitle(f"SHAP dependence — top {len(feats)} features (horizon {horizon})", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Importance table, IC curve, KPI files                                       #
# --------------------------------------------------------------------------- #
def _save_importance_table(gain_imp: pd.Series, shap_imp: pd.Series | None, out_dir: Path) -> Path:
    """Per-member importance to Excel; CSV when no .xlsx engine is available."""
    df = pd.DataFrame({"lgbm_gain": gain_imp.astype(float)})
    total = df["lgbm_gain"].sum()
    df["lgbm_gain_pct"] = df["lgbm_gain"] / total if total else np.nan
    if shap_imp is not None:
        df["shap_mean_abs"] = shap_imp.reindex(df.index)
    df = df.sort_values("lgbm_gain", ascending=False)
    df.index.name = "feature"
    xlsx = out_dir / "feature_importance.xlsx"
    try:
        df.to_excel(xlsx)
        return xlsx
    except Exception:  # noqa: BLE001 - optional engine
        csv = out_dir / "feature_importance.csv"
        df.to_csv(csv)
        return csv


def save_ic_curve(oos: pd.DataFrame, label_name: str, out_dir: Path, horizon: int, roll: int = _ROLL) -> pd.Series:
    """OOS daily IC -> `ic_over_time.csv` + a 3-panel plot (daily IC + rolling mean, cumulative IC,
    cumulative-IC drawdown)."""
    ic = daily_ic_series(oos, label_name)
    if ic.empty:
        return ic
    ic.index = pd.to_datetime(ic.index)
    ic.to_csv(out_dir / "ic_over_time.csv")
    cum = ic.cumsum()
    depth, length = max_drawdown(cum)
    mean_ic, hit = ic.mean(), float((ic > 0).mean())
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 9), sharex=True, gridspec_kw={"height_ratios": [3, 2, 1]})
    ax1.plot(ic.index, ic.values, color="steelblue", lw=0.6, alpha=0.5, label="daily IC")
    if len(ic) >= roll:
        ax1.plot(ic.index, ic.rolling(roll, min_periods=1).mean().values, color="crimson", lw=1.6, label=f"{roll}d rolling mean")
    ax1.axhline(0, color="black", lw=0.8)
    ax1.axhline(mean_ic, color="green", ls="--", lw=1, label=f"mean IC={mean_ic:+.4f}")
    ax1.set_ylabel("daily IC")
    ax1.set_title(f"Out-of-sample IC over time — horizon {horizon} (mean={mean_ic:+.4f}, hit-rate={hit:.0%}, n={len(ic)} days)", fontsize=10)
    ax1.legend(fontsize=8, loc="upper left")
    ax2.plot(ic.index, cum.values, color="darkorange", lw=1.4)
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_ylabel("cumulative IC")
    ax3.fill_between(ic.index, (cum - cum.cummax().clip(lower=0.0)).values, 0.0, color="firebrick", alpha=0.4)
    ax3.set_ylabel("IC drawdown")
    ax3.set_title(f"max cumulative-IC drawdown {depth:+.3f} over {length} days", fontsize=9)
    ax3.set_xlabel("date")
    fig.tight_layout()
    fig.savefig(out_dir / "ic_over_time.png", dpi=140)
    plt.close(fig)
    return ic


def _jsonable(value: Any) -> Any:
    """numpy scalars / NaN -> plain JSON (NaN is not valid JSON; emit null)."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating | float):
        return None if not np.isfinite(float(value)) else float(value)
    return value


def save_horizon_kpis(out_dir: Path, kpis: dict) -> Path:
    """One horizon's KPIs -> `kpis.json`, written inside the horizon loop so an interrupted run
    keeps the horizons it finished."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "kpis.json"
    path.write_text(json.dumps(kpis, indent=2, default=_jsonable), encoding="utf-8")
    return path


def save_run_kpis(run_dir: Path, run_kpis: dict, logger: logging.Logger | None = None) -> Path:
    """Run-level `kpis.json` (nested) + `kpis.csv` (FLAT, one row per (horizon, member)): the file
    to diff between runs."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "kpis.json"
    path.write_text(json.dumps(run_kpis, indent=2, default=_jsonable), encoding="utf-8")
    rows = []
    for h, hk in sorted((run_kpis.get("horizons") or {}).items(), key=lambda kv: int(kv[0])):
        bt = hk.get("backtest") or {}
        common = {
            "horizon": int(h),
            "blend_weight": hk.get("blend_weight"),
            "n_rows": hk.get("n_rows"),
            "n_tickers": hk.get("n_tickers"),
            "n_days": hk.get("n_days"),
            "oos_ic_mean": hk.get("oos_ic_mean"),
            "oos_ic_hit_rate": hk.get("oos_ic_hit_rate"),
            "oos_ic_days": hk.get("oos_ic_days"),
        }
        extra = {
            "oos_ic_max_drawdown": hk.get("oos_ic_max_drawdown"),
            "bt_spread_mean": bt.get("spread_mean"),
            "bt_spread_ir": bt.get("spread_ir"),
            "bt_hit_rate": bt.get("hit_rate"),
            "bt_max_drawdown": bt.get("max_drawdown"),
            "bt_monotonicity": bt.get("monotonicity"),
        }
        rows.append(
            {
                **common,
                "member": "ENSEMBLE",
                "cv_mean_ic": hk.get("cv_mean_ic"),
                "cv_ic_ir": hk.get("cv_ic_ir"),
                "n_features": None,
                "n_pdp": None,
                "shap_available": None,
                **extra,
            }
        )
        for name, mk in sorted((hk.get("members") or {}).items()):
            rows.append(
                {
                    **common,
                    "member": name,
                    "cv_mean_ic": mk.get("cv_mean_ic"),
                    "cv_ic_ir": mk.get("cv_ic_ir"),
                    "n_features": mk.get("n_features"),
                    "n_pdp": mk.get("n_pdp"),
                    "shap_available": mk.get("shap_available"),
                    **extra,
                }
            )
    csv_path = run_dir / "kpis.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    if logger is not None:
        logger.info("Diagnostics KPIs: %d row(s) across %d horizon(s) -> %s", len(rows), len(run_kpis.get("horizons") or {}), csv_path)
    return path


def save_member_diagnostics(
    horizon: int,
    member: str,
    booster: Any,
    panel: pd.DataFrame,
    feature_cols: list[str],
    out_dir: Path,
    top_n: int = 15,
    shap_sample: int = 2000,
    pdp_grid: int = 30,
    logger: logging.Logger | None = None,
) -> dict:
    """PDP + SHAP (raw values, ranking, plots) + gain table for ONE booster member, on the SAME
    encoding the model was fitted on (categoricals -> numeric codes)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    x = design_matrix(panel, feature_cols)
    gain_imp = pd.Series(dict(zip(feature_cols, booster.feature_importance(importance_type="gain"), strict=False))).sort_values(ascending=False)

    shap_imp, n_shap = None, 0
    got = shap_row_values(booster, x, feature_cols, sample=shap_sample, logger=logger)
    if got is not None:
        values, row_idx = got
        shap_imp = shap_importance_from_values(values, feature_cols)
        save_shap_values(values, row_idx, feature_cols, panel, out_dir / "shap_values.parquet")
        _save_shap_importance_plot(shap_imp, out_dir / "shap_importance.png", horizon, top_n)
        shap_imp.rename("shap_mean_abs").to_csv(out_dir / "shap_importance.csv")
        _save_shap_dependence_plot(values, x[row_idx], feature_cols, list(shap_imp.index), out_dir / "shap_dependence.png", horizon)
        n_shap = int(len(row_idx))

    ranking = shap_imp if shap_imp is not None else gain_imp  # PDP selection: SHAP, else gain
    top_feats = list(ranking.head(top_n).index)
    pdp_dir = out_dir / "pdp"
    pdp_dir.mkdir(exist_ok=True)
    n_pdp = 0
    for rank, feat in enumerate(top_feats, 1):
        j = feature_cols.index(feat)
        grid, means = partial_dependence(booster, x, j, grid_points=pdp_grid, sample=shap_sample)
        if grid is None:
            continue
        _save_pdp_plot(grid, means, feat, pdp_dir / f"pdp_{rank:02d}_{safe_filename(feat)}.png", horizon, x[:, j])
        n_pdp += 1

    imp_path = _save_importance_table(gain_imp, shap_imp, out_dir)
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


# --------------------------------------------------------------------------- #
# Monitor                                                                     #
# --------------------------------------------------------------------------- #
class Monitor:
    """Collects one training run's metrics and writes its report under a run-stamped folder.

    Artifacts are written only when the context saves (`context.save`) and
    `model.diagnostics.enabled`; metrics are still computed and logged otherwise."""

    def __init__(self, context: Context, config: DictConfig) -> None:
        self._context = context
        self._log = logging.getLogger(type(self).__module__)
        diag = config.model.get("diagnostics", {}) or {}
        self.enabled = bool(diag.get("enabled", True))
        self.top_n = int(diag.get("top_n_features", 15))
        self.shap_sample = int(diag.get("shap_sample", 2000))
        self.pdp_grid = int(diag.get("pdp_grid", 30))
        self.run_stamp: str | None = None
        self.summaries: dict[int, dict] = {}

    def start_run(self) -> str | None:
        """Open a new run: a timestamp folder when the context saves, else no artifacts."""
        self.run_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") if self._context.save else None
        self.summaries = {}
        return self.run_stamp

    @property
    def run_dir(self) -> Path:
        return Path(self._context.paths["OUTPUT_DIR"]) / "diagnostics" / str(self.run_stamp)

    @property
    def active(self) -> bool:
        return self.enabled and self.run_stamp is not None

    def horizon_report(
        self, horizon: int, panel: pd.DataFrame, oos: pd.DataFrame | None, members: dict[str, BaseModel], kpis: dict, label_name: str
    ) -> dict | None:
        """Every artifact for one horizon while its panel is resident: the ensemble OOS IC curve
        (+ drawdown), one folder per booster member, |coef| tables for linear members, and
        `kpis.json` merged with the caller's CV `kpis`. Returns the summary (None when inactive)."""
        if not self.active:
            return None
        boosters = {name: m for name, m in members.items() if m.supports_shap}
        if not boosters:
            self._log.warning(
                "horizon %s diagnostics: no tree member in the ensemble %s -> no SHAP/PDP possible (add 'lgbm' or 'random_forest' to model.ensemble)",
                horizon,
                list(members),
            )
        out_dir = self.run_dir / f"h{horizon}"
        try:
            summary = self._horizon_summary(horizon, panel, oos, members, boosters, kpis, label_name, out_dir)
        except Exception as exc:  # noqa: BLE001 - reporting is best-effort
            self._log.warning("horizon %s diagnostics failed: %s", horizon, exc)
            return None
        self.summaries[int(horizon)] = summary
        self._log.info(
            "horizon %s diagnostics -> %s | %s | OOS IC %+.4f over %d days (max IC drawdown %+.3f)",
            horizon,
            out_dir,
            ", ".join(
                f"{n}: {m.get('n_pdp', 0)} PDPs, shap={m.get('shap_available', False)}({m.get('shap_rows', 0)} rows)"
                for n, m in summary["members"].items()
                if n in boosters
            ),
            summary["oos_ic_mean"],
            summary["oos_ic_days"],
            summary["oos_ic_max_drawdown"],
        )
        return summary

    def _horizon_summary(
        self,
        horizon: int,
        panel: pd.DataFrame,
        oos: pd.DataFrame | None,
        members: dict[str, BaseModel],
        boosters: dict[str, BaseModel],
        kpis: dict,
        label_name: str,
        out_dir: Path,
    ) -> dict:
        out_dir.mkdir(parents=True, exist_ok=True)
        ic = None
        if oos is not None and not oos.empty:
            ic = save_ic_curve(oos, label_name, out_dir, horizon)
        else:
            self._log.warning("h%s diagnostics: no OOS predictions -> IC-over-time skipped", horizon)

        flat = len(boosters) == 1  # one booster keeps the flat h<H>/ layout readers rely on
        member_summaries: dict[str, dict] = {}
        for name, m in boosters.items():
            member_summaries[name] = save_member_diagnostics(
                horizon=horizon,
                member=name,
                booster=m.model,
                panel=panel,
                feature_cols=list(m.model.feature_name()),
                out_dir=out_dir if flat else out_dir / safe_filename(name),
                top_n=self.top_n,
                shap_sample=self.shap_sample,
                pdp_grid=self.pdp_grid,
                logger=self._log,
            )
        for name, m in members.items():
            if name not in boosters:
                m.importance().rename("abs_coef").sort_values(ascending=False).rename_axis("feature").to_csv(
                    out_dir / f"coef_importance_{safe_filename(name)}.csv"
                )

        ic_s = ic if ic is not None else pd.Series(dtype=float)
        has_ic = len(ic_s) > 0
        depth, length = max_drawdown(ic_s.cumsum()) if has_ic else (float("nan"), 0)
        summary: dict[str, Any] = {
            "horizon": int(horizon),
            "n_rows": int(len(panel)),
            "n_tickers": int(panel["ticker"].nunique()) if "ticker" in panel else None,
            "n_days": int(panel["date"].nunique()) if "date" in panel else None,
            "oos_ic_days": int(len(ic_s)),
            "oos_ic_mean": float(ic_s.mean()) if has_ic else float("nan"),
            "oos_ic_hit_rate": float((ic_s > 0).mean()) if has_ic else float("nan"),
            "members": member_summaries,
        }
        if member_summaries:  # flat back-compat keys of the first booster
            first = next(iter(member_summaries.values()))
            summary.update({"n_pdp": first["n_pdp"], "shap_available": first["shap_available"], "importance_path": first["importance_path"]})
        summary.update(
            {
                "ic_days": summary["oos_ic_days"],
                "ic_mean": summary["oos_ic_mean"],
                "oos_ic_max_drawdown": depth,
                "oos_ic_drawdown_days": length,
            }
        )
        for key, value in kpis.items():  # CV IC / train window known by the caller
            if key == "members":
                for name, mk in (value or {}).items():
                    summary["members"].setdefault(name, {}).update(mk)
            else:
                summary[key] = value
        save_horizon_kpis(out_dir, summary)
        return summary

    def backtest_report(self, horizon: int, result: BacktestResult) -> None:
        """Log the label-only backtest; save its files and attach its summary when active."""
        s = result.summary
        self._log.info(
            "horizon %s backtest (%d buckets, %d days): spread %+.4f (IR %+.2f), hit %.0f%%, max DD %+.3f, monotonicity %+.2f, top turnover %.0f%%",
            horizon,
            int(s.get("n_quantiles", 0)),
            int(s.get("n_days", 0)),
            s.get("spread_mean", float("nan")),
            s.get("spread_ir", float("nan")),
            100 * s.get("hit_rate", float("nan")),
            s.get("max_drawdown", float("nan")),
            s.get("monotonicity", float("nan")),
            100 * s.get("turnover_top", float("nan")),
        )
        if not self.active:
            return
        try:
            Backtest.save(result, self.run_dir / f"h{horizon}", horizon)
        except Exception as exc:  # noqa: BLE001 - reporting is best-effort
            self._log.warning("horizon %s backtest artifacts failed: %s", horizon, exc)
        if int(horizon) in self.summaries:
            self.summaries[int(horizon)]["backtest"] = s

    def run_report(self, weights: dict[int, float], model_types: list[str], target_type: str) -> Path | None:
        """Run-level `kpis.json` / `kpis.csv` across horizons, once the blend weights are known."""
        if not self.summaries or self.run_stamp is None:
            return None
        for h, summary in self.summaries.items():
            summary["blend_weight"] = float(weights.get(h, float("nan")))
        try:
            return save_run_kpis(
                self.run_dir,
                {"run_stamp": self.run_stamp, "model_types": list(model_types), "target_type": target_type, "horizons": self.summaries},
                logger=self._log,
            )
        except Exception as exc:  # noqa: BLE001 - reporting is best-effort
            self._log.warning("run-level diagnostics KPIs failed: %s", exc)
            return None

    def feature_importance(self, models: dict[int, dict[str, BaseModel]]) -> pd.Series | None:
        """Importance aggregated over every (horizon, member): each member's importance is first
        normalised to sum 1 (gain and |coef| are not on one scale), then summed and normalised.
        Logs the top 15 and the peer-relative fundamentals (`f_*`) share."""
        try:
            imp: dict[str, float] = {}
            for members in models.values():
                for model in members.values():
                    s = model.importance()
                    tot = s.sum()
                    if tot > 0:
                        s = s / tot
                    for f, g in s.items():
                        imp[str(f)] = imp.get(str(f), 0.0) + float(g)
            imp_s = pd.Series(imp, dtype=float).sort_values(ascending=False)
            imp_s = imp_s / imp_s.sum()
            self._log.info("Top features by gain:\n%s", imp_s.head(15).round(4).to_string())
            fund_share = float(imp_s[[f for f in imp_s.index if str(f).startswith("f_")]].sum())
            self._log.info("Peer-relative fundamentals share of importance: %.1f%%", 100 * fund_share)
            return imp_s
        except Exception as exc:  # noqa: BLE001 - reporting is best-effort
            self._log.warning("Feature importance unavailable: %s", exc)
            return None
