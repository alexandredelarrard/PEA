"""
step_backtest.py  (optimizer construction)
------------------------------------------
Same load/predict/report as before, but `simulate()` now uses
simulate_portfolio_opt: a dollar+beta-neutral, inverse-variance, vol-targeted
alpha sleeve traded with a turnover-aware partial step, plus a deliberate SPY
market sleeve. Replaces the top/bottom-decile equal-weight book.

KEY CHANGE in predict_and_blend: the signal fed to construction is now the
IR-weighted combined Z-SCORE (magnitude preserved), NOT the percentile rank --
the optimizer uses magnitude, and z-scores make the mean-variance tilt meaningful.
"""
from __future__ import annotations

import json
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import lightgbm as lgb
from omegaconf import DictConfig

from src.utils.step import Step
from src.context import Context
from src.data_aggregate.utils import data_utils as du
from src.data_aggregate.utils.cube import panel_from_cube
from src.modelling.utils_model import model as ml
from src.post_processing.utils.metrics import compute_metrics
from src.post_processing.utils.plot_analysis import plot_equity
from src.post_processing.utils.strategies_opt import simulate_portfolio_opt


class StepBacktest(Step):

    def __init__(self, context: Context, config: DictConfig):
        super().__init__(context=context, config=config)
        self._cfg = config.backtest
        self._cube_cfg = config.build_cube

    def run(self):
        self.load_models()
        self.load_cube_and_returns()
        self.predict_and_blend()
        self.simulate()
        self.report()

    def load_models(self):
        models_dir = self._context.paths["MODELS_DIR"]
        meta_path = models_dir / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"No models at {models_dir}. Run StepModelling first.")
        meta = json.loads(meta_path.read_text())
        self.feature_cols = meta["feature_cols"]
        self.label_column = meta["label_column"]
        # match the target the model was trained on (rank/zscore); config fallback
        self.target_type = meta.get("target_type",
                                    self._config.model.get("target_type", "rank"))
        self.train_ic = {int(k): float(v) for k, v in meta.get("train_ic_ir", {}).items()}
        # ensemble members trained per horizon (back-compat with single model_type)
        self.model_types = list(meta.get("model_types")
                                or [meta.get("model_type", "lightgbm")])
        self.backtest_start = pd.Timestamp(meta["train_end"])
        self.horizons = list(self._cube_cfg.targets.horizons)
        self.models = {}
        for h in self.horizons:
            members = {}
            for kind in self.model_types:
                if kind == "lightgbm":
                    p = models_dir / f"model_h{h}_{kind}.txt"
                    if p.exists():
                        members[kind] = lgb.Booster(model_file=str(p))
                else:                               # pickled linear baseline
                    p = models_dir / f"model_h{h}_{kind}.pkl"
                    if p.exists():
                        with p.open("rb") as f:
                            members[kind] = pickle.load(f)
            if members:
                self.models[h] = members            # {kind: model} per horizon
        if not self.models:
            raise FileNotFoundError(f"No saved models found in {models_dir}.")
        self._log.info("Loaded ensemble %s for horizons %s; backtest starts %s",
                       self.model_types, list(self.models.keys()), self.backtest_start.date())

    def load_cube_and_returns(self):
        self.cube = pd.read_parquet(self._context.paths["CUBE_PATH"])
        self.end = (pd.Timestamp(self._cfg.end) if self._cfg.get("end")
                    else pd.Timestamp(self.cube["date"].max()))
        prices_long = pd.read_parquet(self._context.paths["PRICES_PATH"])
        raw = du.prices_long_to_multiindex(prices_long)
        close = du.extract_field(raw, "Close")
        mkt = self._cube_cfg.market_ticker
        rets = du.daily_returns(close)
        self.spy_ret = rets[mkt]
        drop = [mkt] + list(self._config.data_extract.get("other_tickers", []))
        self.stock_ret = rets.drop(columns=drop, errors="ignore")
        self._log.info("Backtest window %s -> %s", self.backtest_start.date(), self.end.date())

    def _blend_weights(self) -> dict:
        if self._cfg.get("blend", "ir") == "equal" or not self.train_ic:
            return {h: 1.0 / len(self.models) for h in self.models}
        irs = {h: max(0.0, self.train_ic.get(h, 0.0)) for h in self.models}
        tot = sum(irs.values())
        return ({h: irs[h] / tot for h in self.models} if tot > 0
                else {h: 1.0 / len(self.models) for h in self.models})

    def predict_and_blend(self):
        weights = self._blend_weights()
        self._log.info("Blend weights: %s", {h: round(w, 3) for h, w in weights.items()})
        blended = None
        for h, models in self.models.items():
            panel = panel_from_cube(self.cube, horizon=h, label_name=self.label_column,
                                    feature_cols=self.feature_cols,
                                    target_type=self.target_type)
            panel = panel[(panel["date"] >= self.backtest_start) & (panel["date"] <= self.end)]
            if panel.empty:
                continue
            df = panel[["date", "ticker"]].copy()
            # ensemble = per-day-standardized average of the trained families
            scores, zs = ml.ensemble_predict(models, panel, self.feature_cols)
            df["z"] = scores.to_numpy()
            df["z"] = df.groupby("date")["z"].transform(
                lambda s: (s - s.mean()) / (s.std() if s.std() > 0 else np.nan))
            df = df.rename(columns={"z": f"z_{h}"})
            blended = df if blended is None else blended.merge(df, on=["date", "ticker"], how="outer")

        zc = [f"z_{h}" for h in self.models if f"z_{h}" in blended.columns]
        w = np.array([weights[int(c.split('_')[1])] for c in zc])
        z = blended[zc].to_numpy()
        mask = ~np.isnan(z)
        wsum = np.where(mask, w, 0).sum(axis=1)
        
        # IR-weighted combined z-score; KEEP magnitude (no percentile rank) for the optimizer
        blended["combined"] = np.where(
            wsum > 0,
            np.nansum(np.where(mask, z * w, 0), axis=1) / np.where(wsum > 0, wsum, 1),
            np.nan)
        self.signal = blended.pivot(index="date", columns="ticker", values="combined")
        self.signal.index = pd.to_datetime(self.signal.index)
        self._log.info("Signal (combined z) matrix: %s days x %s tickers", *self.signal.shape)

    def simulate(self):
        c = self._cfg
        self.daily = simulate_portfolio_opt(
            self.signal, self.stock_ret, self.spy_ret,
            starting_capital=float(c.starting_capital),
            market_weight=c.get("market_weight", 0.5),
            target_ann_vol=c.get("target_ann_vol", 0.08),
            beta_neutral=c.get("beta_neutral", True),
            pos_cap=c.get("pos_cap", 0.03),
            gross_cap=c.get("gross_cap", 3.0),
            step=c.get("step", 0.35),
            no_trade_band=c.get("no_trade_band", 0.0),
            beta_window=c.get("beta_window", 63),
            vol_window=c.get("vol_window", 63),
            fee_bps=c.fee_bps, spread_bps=c.spread_bps,
            rebalance_freq=c.get("rebalance_freq", 1))
        self.metrics = compute_metrics(self.daily, rf_annual=c.get("risk_free_rate", 0.0))

    def report(self):
        m = self.metrics
        c = self._cfg

        self._log.info("=== Backtest results ===")
        self._log.info("Strategy: total %.1f%%  ann %.1f%%  vol %.1f%%  Sharpe %.2f  maxDD %.1f%%",
                       m["total_return"]*100, m["ann_return"]*100, m["ann_vol"]*100,
                       m["sharpe"], m["max_drawdown"]*100)
        self._log.info("SPY:      total %.1f%%  ann %.1f%%  Sharpe %.2f  maxDD %.1f%%",
                       m["spy_total_return"]*100, m["spy_ann_return"]*100,
                       m["spy_sharpe"], m["spy_max_drawdown"]*100)
        self._log.info("Avg daily turnover %.3f  avg daily cost %.4f%%",
                       m["avg_daily_turnover"], m["avg_daily_cost"]*100)

        # --- per-sleeve diagnostics: show whether each construction param bites ---
        d = self.daily
        if {"alpha_ret", "mkt_ret", "alpha_gross", "alpha_max_w"}.issubset(d.columns):
            ann = np.sqrt(252.0)
            alpha_vol = float(d["alpha_ret"].std() * ann)
            mkt_vol = float(d["mkt_ret"].std() * ann)
            avg_gross = float(d["alpha_gross"].mean())
            avg_maxw = float(d["alpha_max_w"].mean())
            gross_cap = float(c.get("gross_cap", 3.0))
            pos_cap = float(c.get("pos_cap", 0.05))
            self._log.info(
                "Alpha sleeve: realized vol %.1f%% (target %.1f%%)  avg gross %.2f/%.1f  "
                "avg max|w| %.3f/%.3f", alpha_vol*100,
                float(c.get("target_ann_vol", 0.08))*100, avg_gross, gross_cap, avg_maxw, pos_cap)
            self._log.info(
                "Market sleeve: weight %.2f -> realized vol %.1f%%  (dominates total risk "
                "when the alpha sleeve is smaller)", float(c.get("market_weight", 0.5)), mkt_vol*100)
            # flag knobs that are configured so loosely they never activate
            if avg_gross < 0.8 * gross_cap:
                self._log.info("  note: gross_cap (%.1f) never binds (avg gross %.2f) -> "
                               "it is a slack safety rail, not an active knob here", gross_cap, avg_gross)
            if avg_maxw < 0.8 * pos_cap:
                self._log.info("  note: pos_cap (%.3f) never binds (avg max|w| %.3f) -> "
                               "slack safety rail; tighten it to shape concentration", pos_cap, avg_maxw)

        out_dir = self._context.paths["OUTPUT_DIR"] / "backtest"
        out_dir.mkdir(parents=True, exist_ok=True)
        self.daily.to_parquet(out_dir / "backtest_daily.parquet")
        pd.DataFrame([m]).to_csv(out_dir / "backtest_metrics.csv", index=False)
        plot_equity(self.daily, m, out_dir / "portfolio_vs_spy.png")
        self._log.info("Saved backtest outputs to %s", out_dir)