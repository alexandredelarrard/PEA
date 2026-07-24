"""
ls_model.py  (src/strategies/utils/ls_model.py)
-----------------------------------------------
Shared L/S MODEL signal builder: load the trained ensemble artifacts, project the cube (feature
cols only, OOS window date >= train_end), score each horizon's ensemble, blend across horizons →
per-name combined z-signal, and load the equity return / price panels. Used by BOTH the
market-neutral L/S sleeve (`step_ls`) and the long-only sleeve (`step_eq_long_only`) so the
model/signal is defined once and neither strategy depends on the other's step.
"""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass

import numpy as np
import pandas as pd
import lightgbm as lgb
from omegaconf import DictConfig
from sqlalchemy import text

from src.context import Context
from src.data_aggregate.utils import data_utils as du
from src.data_aggregate.utils.cube import panel_from_cube
from src.modelling.long_short.utils import model as ml


@dataclass
class SignalBundle:
    signal: pd.DataFrame        # date x ticker combined cross-sectional z-signal
    stock_ret: pd.DataFrame     # date x equity daily returns (market/macro + ^index dropped)
    spy_ret: pd.Series          # market benchmark daily return
    close: pd.DataFrame         # date x ticker close prices (all tickers; for share blotters)
    backtest_start: pd.Timestamp
    end: pd.Timestamp
    train_ic: dict
    horizons: list


def _load_models(context: Context, cube_cfg: DictConfig, model_cfg: DictConfig):
    models_dir = context.paths["MODELS_DIR"]
    meta_path = models_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"No models at {models_dir}. Train the L/S models first.")
    meta = json.loads(meta_path.read_text())
    target_type = meta.get("target_type", model_cfg.get("target_type", "rank"))
    model_types = list(meta.get("model_types") or [meta.get("model_type", "lightgbm")])
    horizons = list(cube_cfg.targets.horizons)
    models: dict = {}
    for h in horizons:
        members = {}
        for kind in model_types:
            p = ml.member_model_path(models_dir, h, kind)
            if not p.exists():
                continue
            if kind in ml.BOOSTER_MEMBER_KINDS:
                b = lgb.Booster(model_file=str(p)); b.feature_names = b.feature_name()
                members[kind] = b
            else:
                with p.open("rb") as f:
                    members[kind] = pickle.load(f)
        if members:
            models[h] = members
    if not models:
        raise FileNotFoundError(f"No saved model files found in {models_dir}.")
    return meta, models, target_type, horizons


def _cube_columns(context: Context) -> set[str]:
    with context.store.engine.connect() as c:
        return set(pd.read_sql(text("SELECT column_name FROM information_schema.columns "
                                     "WHERE table_name = 'cube'"), c)["column_name"])


def _project_cube(context: Context, meta: dict, models: dict, target_type: str,
                  start: pd.Timestamp, end) -> tuple[pd.DataFrame, str]:
    cube_cols = _cube_columns(context)
    target_col = (f"target_{target_type}" if f"target_{target_type}" in cube_cols
                  else "target" if "target" in cube_cols else None)
    if target_col is None:
        raise KeyError(f"Target column 'target_{target_type}' not in cube; rebuild the cube.")
    want = list(dict.fromkeys(meta["feature_cols"] + meta.get("categorical_cols", [])))
    load_cols = list(dict.fromkeys(["date", "ticker", "target_horizon", target_col]
                                   + [c for c in want if c in cube_cols]))
    horizons = sorted(int(h) for h in models)
    where = [f'target_horizon IN ({",".join(str(h) for h in horizons)})', f"date >= '{start.date()}'"]
    if end is not None:
        where.append(f"date <= '{pd.Timestamp(end).date()}'")
    q = text(f"SELECT {', '.join(chr(34)+c+chr(34) for c in load_cols)} FROM cube WHERE " + " AND ".join(where))
    with context.store.engine.connect() as c:
        return pd.read_sql(q, c, parse_dates=["date"]), target_col


def _returns(context: Context, config: DictConfig, cube_cfg: DictConfig, model_cfg: DictConfig,
             start: pd.Timestamp):
    buffer = int(2.2 * (int(model_cfg.get("beta_window", 63)) + int(model_cfg.get("vol_window", 63))) + 30)
    cutoff = (start - pd.Timedelta(days=buffer)).date()
    with context.store.engine.connect() as c:
        long = pd.read_sql(text("SELECT * FROM prices WHERE date >= :cut"), c,
                           params={"cut": str(cutoff)}, parse_dates=["date"])
    close = du.extract_field(du.prices_long_to_multiindex(long), "Close")
    mkt = cube_cfg.market_ticker
    rets = du.daily_returns(close)
    idx_syms = [c for c in rets.columns if str(c).startswith("^")]   # drop indices (^VIX, ^GSPC…)
    drop = [mkt] + list(config.data_extract.get("other_tickers", [])) + idx_syms
    return close, rets.drop(columns=drop, errors="ignore"), rets[mkt]


def build_signal(context: Context, config: DictConfig, end=None) -> SignalBundle:
    """Load the ensemble, project the OOS cube, score + blend horizons -> combined z-signal, and
    load the equity returns/prices. `config.strategy_ls` holds the model windows/blend params."""
    cube_cfg, model_cfg = config.build_cube, config.strategy_ls
    meta, models, target_type, _ = _load_models(context, cube_cfg, model_cfg)
    start = pd.Timestamp(meta["train_end"])
    train_ic = {int(k): float(v) for k, v in meta.get("train_ic_ir", {}).items()}
    cube, _ = _project_cube(context, meta, models, target_type, start, end)
    close, stock_ret, spy_ret = _returns(context, config, cube_cfg, model_cfg, start)
    end_ts = pd.Timestamp(end) if end is not None else pd.Timestamp(cube["date"].max())

    blended = None
    for h, members in models.items():
        panel = panel_from_cube(cube, horizon=h, label_name=meta["label_column"],
                                feature_cols=meta["feature_cols"] + meta.get("categorical_cols", []),
                                target_type=target_type)
        panel = panel[(panel["date"] >= start) & (panel["date"] <= end_ts)]
        if panel.empty:
            continue
        scores, _ = ml.ensemble_predict(members, panel, meta["feature_cols"])
        df = panel[["date", "ticker"]].copy()
        df["z"] = pd.Series(scores.to_numpy(), index=panel.index)
        df["z"] = df.groupby("date")["z"].transform(
            lambda s: (s - s.mean()) / (s.std() if s.std() > 0 else np.nan))
        blended = df.rename(columns={"z": f"z_{h}"}) if blended is None else \
            blended.merge(df.rename(columns={"z": f"z_{h}"}), on=["date", "ticker"], how="outer")

    zc = [f"z_{h}" for h in models if blended is not None and f"z_{h}" in blended.columns]
    if not zc:
        raise RuntimeError("build_signal: no horizon produced a signal in the OOS window.")
    hs = [int(c.split("_")[1]) for c in zc]
    ir = {h: train_ic.get(h, np.nan) for h in hs}
    if str(model_cfg.get("blend", "ir")) == "equal":
        bw = {h: 1.0 / len(hs) for h in hs}
    else:
        bw = ml.optimal_forecast_weights({h: blended[f"z_{h}"].to_numpy() for h in hs}, ir,
                                         shrink=float(model_cfg.get("blend_shrink", 0.5)))
    w = np.array([bw[h] for h in hs])
    z = blended[zc].to_numpy(); mask = ~np.isnan(z)
    wsum = np.where(mask, w, 0).sum(axis=1)
    blended["combined"] = np.where(wsum > 0,
                                   np.nansum(np.where(mask, z * w, 0), axis=1) / np.where(wsum > 0, wsum, 1),
                                   np.nan)
    signal = blended.pivot(index="date", columns="ticker", values="combined")
    signal.index = pd.to_datetime(signal.index)
    return SignalBundle(signal=signal, stock_ret=stock_ret, spy_ret=spy_ret, close=close,
                        backtest_start=start, end=end_ts, train_ic=train_ic, horizons=list(models))
