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
from sqlalchemy import text

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
        self.save_predictions()      # flat files in /data: each method + the ensemble
        self.compare_strategies()    # OOS construction comparison vs SP500
        self.simulate()
        self.report()

    def load_models(self):
        models_dir = self._context.paths["MODELS_DIR"]
        meta_path = models_dir / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"No models at {models_dir}. Run StepModelling first.")
        meta = json.loads(meta_path.read_text())
        self.feature_cols = meta["feature_cols"]
        # LightGBM-only categoricals (sector/industry) must be in the scored panel too;
        # ensemble_predict picks each model's own feature_names, so the linear member
        # still ignores them. Older models (no categoricals) -> empty list.
        self.categorical_cols = meta.get("categorical_cols", [])
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
                        b = lgb.Booster(model_file=str(p))
                        # A Booster reloaded from file loses the custom `.feature_names`
                        # attribute that ensemble_predict reads (train_ranker set it to the
                        # model's OWN 60 features incl. the sector/industry_group
                        # categoricals). Without this it falls back to the 58-col numeric
                        # union -> LightGBM raises "58 vs 60 features". Restore it from the
                        # model file's own feature names.
                        b.feature_names = b.feature_name()
                        members[kind] = b
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

    # ------------------------------------------------------------------ #
    # Cube read: ONLY the columns the models score on, ONLY the OOS window #
    # ------------------------------------------------------------------ #
    def _cube_columns(self) -> set[str]:
        """Column names present in the `cube` table (no data scan)."""
        q = text("SELECT column_name FROM information_schema.columns "
                 "WHERE table_name = 'cube'")
        with self._context.store.engine.connect() as c:
            return set(pd.read_sql(q, c)["column_name"])

    def _target_column(self, cube_cols: set[str]) -> str:
        """Stored target column for the model's target_type (legacy `target` fallback)."""
        col = f"target_{self.target_type}"
        if col in cube_cols:
            return col
        if "target" in cube_cols:
            return "target"
        raise KeyError(f"Target column '{col}' not in cube; rebuild with "
                       f"'{self.target_type}' in build_cube.targets.labels.")

    def _load_cube_projected(self):
        """Load ONLY what prediction needs: the index + the model's feature/categorical
        columns + the target column, restricted to the ensemble's horizons and the OOS
        window (date >= train_end). Avoids pulling the full ~159-col x millions-of-rows
        cube — the scored panels (and thus the signal) are identical."""
        cube_cols = self._cube_columns()
        self.target_col = self._target_column(cube_cols)
        want = list(dict.fromkeys(self.feature_cols + self.categorical_cols))
        feats = [c for c in want if c in cube_cols]
        missing = [c for c in want if c not in cube_cols]
        meta = ["date", "ticker", "target_horizon", self.target_col]
        load_cols = list(dict.fromkeys(meta + feats))
        horizons = sorted(int(h) for h in self.models)
        projected = ", ".join(f'"{c}"' for c in load_cols)
        where = [f'target_horizon IN ({",".join(str(h) for h in horizons)})',
                 f"date >= '{self.backtest_start.date()}'"]
        if self._end_cfg is not None:
            where.append(f"date <= '{self._end_cfg.date()}'")
        q = text(f'SELECT {projected} FROM cube WHERE ' + " AND ".join(where))
        with self._context.store.engine.connect() as c:
            self.cube = pd.read_sql(q, c, parse_dates=["date"])
        self._log.info("Loaded projected cube: %d rows x %d cols (features %d/%d requested, "
                       "horizons=%s, date>=%s)", len(self.cube), len(load_cols), len(feats),
                       len(want), horizons, self.backtest_start.date())
        if missing:
            self._log.info("backtest: %d model feature(s) absent from cube (scored as NaN "
                           "by LightGBM): %s", len(missing), missing)

    def _load_prices_since(self, cutoff) -> pd.DataFrame:
        """Prices from `cutoff` forward only — enough history to warm up the rolling
        risk model before the OOS window, not the full 15y series."""
        q = text("SELECT * FROM prices WHERE date >= :cut")
        with self._context.store.engine.connect() as c:
            return pd.read_sql(q, c, params={"cut": str(cutoff)}, parse_dates=["date"])

    def load_cube_and_returns(self):
        self._end_cfg = pd.Timestamp(self._cfg.end) if self._cfg.get("end") else None
        self._load_cube_projected()
        self.end = self._end_cfg if self._end_cfg is not None else pd.Timestamp(self.cube["date"].max())

        # returns for the risk model + P&L: load prices only from a warmup buffer before
        # the window (the rolling beta/vol need ~beta_window+vol_window trading days first)
        buffer_days = int(2.2 * (int(self._cfg.get("beta_window", 63))
                                 + int(self._cfg.get("vol_window", 63))) + 30)
        cutoff = (self.backtest_start - pd.Timedelta(days=buffer_days)).date()
        prices_long = self._load_prices_since(cutoff)
        raw = du.prices_long_to_multiindex(prices_long)
        close = du.extract_field(raw, "Close")
        mkt = self._cube_cfg.market_ticker
        rets = du.daily_returns(close)
        self.spy_ret = rets[mkt]
        drop = [mkt] + list(self._config.data_extract.get("other_tickers", []))
        self.stock_ret = rets.drop(columns=drop, errors="ignore")
        self._log.info("Backtest window %s -> %s (prices loaded from %s; %d return days)",
                       self.backtest_start.date(), self.end.date(), cutoff, len(rets))

    def _horizon_blend_weights(self, signals: dict, ir: dict) -> dict:
        """Correlation-aware, shrinkage-regularized optimal combination of the
        per-horizon forecasts (see model.optimal_forecast_weights). Falls back to
        equal weights when blend='equal'. Unlike the old max(0, IR)/sum(IR), a
        horizon whose CV IR is NaN is treated as a neutral prior (not dropped), so
        e.g. the 90d horizon still participates instead of getting weight 0."""
        if self._cfg.get("blend", "ir") == "equal" or not signals:
            return {h: 1.0 / len(signals) for h in signals} if signals else {}
        return ml.optimal_forecast_weights(
            signals, ir, shrink=float(self._cfg.get("blend_shrink", 0.5)))

    def predict_and_blend(self):
        # 1) build each horizon's per-day-standardized ensemble z-signal FIRST, so
        #    the blend weights can see the signals' cross-correlation
        blended = None
        member_frames = []
        for h, models in self.models.items():
            panel = panel_from_cube(self.cube, horizon=h, label_name=self.label_column,
                                    feature_cols=self.feature_cols + self.categorical_cols,
                                    target_type=self.target_type)
            panel = panel[(panel["date"] >= self.backtest_start) & (panel["date"] <= self.end)]
            if panel.empty:
                self._log.warning("Horizon %s: no panel rows in the backtest window "
                                  "[%s, %s] -> excluded from the blend.",
                                  h, self.backtest_start.date(), self.end.date())
                continue
            df = panel[["date", "ticker"]].copy()
            # ensemble = per-day-standardized average of the trained families; keep each
            # member's per-day-standardized prediction for the flat prediction files
            scores, members = ml.ensemble_predict(models, panel, self.feature_cols)
            df["z"] = scores.to_numpy()
            df["z"] = df.groupby("date")["z"].transform(
                lambda s: (s - s.mean()) / (s.std() if s.std() > 0 else np.nan))
            mf = panel[["date", "ticker"]].copy()
            mf["horizon"] = h
            for name, mz in members.items():
                mf[f"pred_{name}"] = mz.to_numpy()          # individual method (per-day z)
            mf["pred_ensemble"] = df["z"].to_numpy()        # ensembling method (per-day z)
            member_frames.append(mf)
            df = df.rename(columns={"z": f"z_{h}"})
            blended = df if blended is None else blended.merge(df, on=["date", "ticker"], how="outer")

        self.member_predictions = (pd.concat(member_frames, ignore_index=True)
                                   if member_frames else pd.DataFrame())

        zc = [f"z_{h}" for h in self.models if f"z_{h}" in (blended.columns if blended is not None else [])]
        if not zc:
            raise RuntimeError("No horizon produced a signal in the backtest window.")
        hs = [int(c.split("_")[1]) for c in zc]

        # 2) correlation-aware, shrinkage-regularized optimal horizon weights
        signals = {h: blended[f"z_{h}"].to_numpy() for h in hs}
        ir = {h: self.train_ic.get(h, np.nan) for h in hs}
        weights = self._horizon_blend_weights(signals, ir)
        self.blend_weights = weights
        self._log.info("Per-horizon CV IC_IR: %s",
                       {h: (round(ir[h], 3) if np.isfinite(ir[h]) else None) for h in hs})
        self._log.info("Blend weights (corr-aware, shrunk): %s",
                       {h: round(w, 3) for h, w in weights.items()})

        # 3) combine: weighted, NaN-tolerant average of the horizon z-scores
        #    (KEEP magnitude -- the optimizer uses it, not a percentile rank)
        w = np.array([weights[h] for h in hs])
        z = blended[zc].to_numpy()
        mask = ~np.isnan(z)
        wsum = np.where(mask, w, 0).sum(axis=1)
        blended["combined"] = np.where(
            wsum > 0,
            np.nansum(np.where(mask, z * w, 0), axis=1) / np.where(wsum > 0, wsum, 1),
            np.nan)
        self.signal = blended.pivot(index="date", columns="ticker", values="combined")
        self.signal.index = pd.to_datetime(self.signal.index)
        # tidy long form of the blended cross-horizon signal for the flat prediction file
        self.blended_signal_long = blended[["date", "ticker", "combined"]].dropna(subset=["combined"]).copy()
        self._log.info("Signal (combined z) matrix: %s days x %s tickers", *self.signal.shape)

    def save_predictions(self):
        """Write the OOS predictions as flat local files under data/predictions/:
          * backtest_member_predictions.{parquet,csv}: one row per (date, ticker,
            horizon) with EACH individual method (pred_elasticnet, pred_lightgbm, ...)
            AND the ensembling method (pred_ensemble), all per-day cross-sectional z.
          * backtest_blended_signal.{parquet,csv}: the cross-horizon blended signal
            (combined z + its per-day percentile rank) actually fed to construction."""
        out = self._context.paths["DATA_STORE"] / "predictions"
        out.mkdir(parents=True, exist_ok=True)
        mp = getattr(self, "member_predictions", pd.DataFrame())
        methods = [c for c in mp.columns if c.startswith("pred_")] if not mp.empty else []
        if not mp.empty:
            mp = mp.sort_values(["date", "horizon", "ticker"]).reset_index(drop=True)
            mp.to_parquet(out / "backtest_member_predictions.parquet", index=False)
            mp.to_csv(out / "backtest_member_predictions.csv", index=False)
        bl = getattr(self, "blended_signal_long", pd.DataFrame())
        if not bl.empty:
            bl = bl.copy()
            bl["signal_rank"] = bl.groupby("date")["combined"].rank(pct=True)
            bl = bl.sort_values(["date", "ticker"]).reset_index(drop=True)
            bl.to_parquet(out / "backtest_blended_signal.parquet", index=False)
            bl.to_csv(out / "backtest_blended_signal.csv", index=False)
        self._log.info("Saved flat prediction files to %s (methods=%s, %d member rows, "
                       "%d blended rows)", out, methods, len(mp), len(bl))

    # ------------------------------------------------------------------ #
    # OOS comparison of construction strategies (all on the SAME signal)   #
    # ------------------------------------------------------------------ #
    def _opt_common(self, sector_map) -> dict:
        c = self._cfg
        return dict(
            signal=self.signal, stock_ret=self.stock_ret, spy_ret=self.spy_ret,
            starting_capital=float(c.starting_capital),
            target_ann_vol=c.get("target_ann_vol", 0.1), beta_neutral=c.get("beta_neutral", True),
            pos_cap=c.get("pos_cap", 0.05), gross_cap=c.get("gross_cap", 3.0),
            step=c.get("step", 0.35), no_trade_band=c.get("no_trade_band", 0.0),
            beta_window=c.get("beta_window", 63), vol_window=c.get("vol_window", 63),
            fee_bps=c.fee_bps, spread_bps=c.spread_bps,
            rebalance_freq=c.get("rebalance_freq", 63),
            sector_map=sector_map, sector_neutral=bool(c.get("sector_neutral", False)))

    def _regime_kwargs(self) -> dict:
        c = self._cfg
        return dict(regime_target_vol=c.get("regime_target_vol", 0.15),
                    regime_vol_window=c.get("regime_vol_window", 63),
                    regime_scale_floor=c.get("regime_scale_floor", 0.3),
                    regime_scale_cap=c.get("regime_scale_cap", 1.5))

    def compare_strategies(self):
        """Backtest several construction strategies on the OOS signal and report each
        one's return vs SP500. Same trained models + signal + window throughout — only
        the portfolio construction differs, so the table isolates the construction."""
        if not self._cfg.get("compare", True):
            return
        c = self._cfg
        sector_map = self._sector_map() if c.get("sector_neutral", False) else None
        oc, reg = self._opt_common(sector_map), self._regime_kwargs()
        mw = float(c.get("compare_market_weight", 1.0))     # SPY sleeve for the "with spy" variant
        variants = {
            "optimizer_volscaled": lambda: simulate_portfolio_opt(
                **oc, market_weight=0.0, vol_scaling=True, **reg),               # saved default, no SPY
            "optimizer_volscaled_with_spy": lambda: simulate_portfolio_opt(
                **oc, market_weight=mw, vol_scaling=True, **reg),                # + SPY beta sleeve
        }
        rows, last_m = [], None
        for name, fn in variants.items():
            try:
                d = fn()
                m = compute_metrics(d, rf_annual=c.get("risk_free_rate", 0.0))
            except Exception as e:                                  # noqa: BLE001
                self._log.warning("strategy '%s' failed: %s", name, e)
                continue
            last_m = m
            rows.append({"strategy": name, "total_%": round(m["total_return"] * 100, 1),
                         "ann_%": round(m["ann_return"] * 100, 1),
                         "vol_%": round(m["ann_vol"] * 100, 1), "sharpe": round(m["sharpe"], 2),
                         "maxDD_%": round(m["max_drawdown"] * 100, 1),
                         "turnover": round(m["avg_daily_turnover"], 3)})
        if last_m is not None:
            rows.append({"strategy": "SP500 (SPY)", "total_%": round(last_m["spy_total_return"] * 100, 1),
                         "ann_%": round(last_m["spy_ann_return"] * 100, 1),
                         "vol_%": round(last_m["spy_ann_vol"] * 100, 1),
                         "sharpe": round(last_m["spy_sharpe"], 2),
                         "maxDD_%": round(last_m["spy_max_drawdown"] * 100, 1), "turnover": np.nan})
        self.comparison = pd.DataFrame(rows)
        out_dir = self._context.paths["OUTPUT_DIR"] / "backtest"
        out_dir.mkdir(parents=True, exist_ok=True)
        self.comparison.to_csv(out_dir / "strategy_comparison.csv", index=False)
        self._log.info("=== OOS strategy comparison (%s -> %s) vs SP500 ===\n%s",
                       self.backtest_start.date(), self.end.date(),
                       self.comparison.to_string(index=False))

    def _sector_map(self) -> dict:
        """ticker -> GICS group for sector-neutral construction. Uses the
        `industry_group` column (24-level) from tickers.csv, falling back to
        `sector` (11) if that column is absent (older tickers.csv)."""
        tk = self._context.store.load("sp500_tickers")
        if tk.empty:
            return {}
        col = ("industry_group" if "industry_group" in tk.columns
               else "sector" if "sector" in tk.columns else None)
        if col is None:
            return {}
        return dict(zip(tk["ticker"], tk[col]))

    def simulate(self):
        c = self._cfg
        sector_neutral = bool(c.get("sector_neutral", False))
        sector_map = self._sector_map() if sector_neutral else None
        if sector_neutral:
            n_groups = len(set(sector_map.values())) if sector_map else 0
            self._log.info("Sector-neutral construction ON: %s groups (industry-group level)",
                           n_groups)
        vol_scaling = bool(c.get("vol_scaling", False))
        reg = self._regime_kwargs() if vol_scaling else {}
        if vol_scaling:
            self._log.info("Vol-regime de-risking ON: exposure = clip(%.2f / trailing SPY "
                           "vol, %.2f, %.2f) over a %dd window",
                           reg["regime_target_vol"], reg["regime_scale_floor"],
                           reg["regime_scale_cap"], reg["regime_vol_window"])
        self.daily = simulate_portfolio_opt(
            **self._opt_common(sector_map),
            market_weight=c.get("market_weight", 0.5),
            vol_scaling=vol_scaling, **reg)
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
            if "regime_scale" in d.columns and bool(c.get("vol_scaling", False)):
                rsc = d["regime_scale"]
                self._log.info("Vol-regime overlay: avg exposure x%.2f (min %.2f in the "
                               "highest-vol days, max %.2f in calm) -> book de-risks through "
                               "vol spikes", float(rsc.mean()), float(rsc.min()), float(rsc.max()))
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