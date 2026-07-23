"""
step_ls.py  (src/strategies/step_ls.py)
---------------------------------------
Market-neutral equity LONG/SHORT (alpha) strategy sleeve — SELF-CONTAINED (absorbs the old
StepBacktest prediction + construction; depends on no other step). Flow:
  1. read `strategy_ls` construction config + PortfolioInputs (capital, target vol, window)
  2. load the trained ensemble artifacts (models_dir + metadata.json) -> OOS starts at train_end
  3. PREDICT: project the cube (feature cols only, date >= train_end), score each horizon's
     ensemble, blend across horizons -> per-name combined z-signal
  4. CONSTRUCT: dollar/beta/sector-neutral inverse-variance optimizer (strategies_opt,
     market_weight=0 -> pure alpha), sized to the portfolio's target vol
  5. metrics -> daily P&L + positions -> StrategyResult
"""
from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd
import lightgbm as lgb
from sqlalchemy import text

from src.strategies.base import Strategy, PortfolioInputs, StrategyResult
from src.data_aggregate.utils import data_utils as du
from src.data_aggregate.utils.cube import panel_from_cube
from src.modelling.long_short.utils import model as ml
from src.strategies.utils.strategies_opt import simulate_portfolio_opt
from src.utils.risk_parity import series_metrics


class LongShortStrategy(Strategy):
    name = "ls_equity"

    def run(self, inputs: PortfolioInputs) -> StrategyResult:

        self._inputs = inputs
        self._cfg = self._config.strategy_ls
        self._cube_cfg = self._config.build_cube
        self._end_cfg = inputs.end
        self.load_models()
        self.load_cube_and_returns()
        self.predict_and_blend()
        daily = self.simulate()
        if daily is None or daily.empty:
            self._log.warning("ls_equity: empty alpha book (no tradeable signal in window)")
            return StrategyResult(self.name, pd.Series(dtype=float), series_metrics(pd.Series(dtype=float)))
        ret = daily["net_ret"].astype(float)
        if inputs.start is not None:
            ret = ret[ret.index >= inputs.start]
        self._log.info("ls_equity sleeve: %d OOS days from %s, ann-vol %.1f%%",
                       len(ret), self.backtest_start.date(), float(ret.std() * (252 ** 0.5)) * 100)
        extra = {"signal_shape": tuple(self.signal.shape)}
        if inputs.analysis:
            extra["analysis"] = self._analyze(ret)
        return StrategyResult(name=self.name, returns=ret,
                              metrics=series_metrics(ret, inputs.risk_free_rate),
                              positions=None, extra=extra)

    def _analyze(self, ret: pd.Series) -> dict:
        """IC + Sharpe/maxDD + market-neutrality (beta to SP, corr to energy) plots."""
        from src.strategies.analysis.ls_analysis import analyze_ls
        from src.strategies.analysis.common import load_market_refs
        refs = load_market_refs(self._context.store)
        horizon = int(self._cube_cfg.targets.get("primary_horizon", self.horizons[0]))
        out_dir = self._context.paths["OUTPUT_DIR"] / "ls_equity" / "analysis"
        m = analyze_ls(ret, self.signal, self.stock_ret, self.spy_ret,
                       refs.get("energy"), out_dir, horizon=horizon)
        self._log.info("ls_equity analysis: IC %.3f (IR %.2f), full beta_SP %.2f, corr_energy %.2f -> %s",
                       m["ic_mean"], m["ic_ir"], m["full_beta_sp"], m["full_corr_energy"], out_dir)
        return m

    # ------------------------------------------------------------------ #
    def load_models(self):
        models_dir = self._context.paths["MODELS_DIR"]
        meta_path = models_dir / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"No models at {models_dir}. Train the L/S models first.")
        meta = json.loads(meta_path.read_text())
        self.feature_cols = meta["feature_cols"]
        self.categorical_cols = meta.get("categorical_cols", [])
        self.label_column = meta["label_column"]
        self.target_type = meta.get("target_type", self._cfg.get("target_type", "rank"))
        self.train_ic = {int(k): float(v) for k, v in meta.get("train_ic_ir", {}).items()}
        self.model_types = list(meta.get("model_types") or [meta.get("model_type", "lightgbm")])
        self.backtest_start = pd.Timestamp(meta["train_end"])
        self.horizons = list(self._cube_cfg.targets.horizons)
        self.models = {}
        for h in self.horizons:
            members = {}
            for kind in self.model_types:
                p = ml.member_model_path(models_dir, h, kind)
                if not p.exists():
                    continue
                if kind in ml.BOOSTER_MEMBER_KINDS:
                    b = lgb.Booster(model_file=str(p))
                    b.feature_names = b.feature_name()
                    members[kind] = b
                else:
                    with p.open("rb") as f:
                        members[kind] = pickle.load(f)
            if members:
                self.models[h] = members
        if not self.models:
            raise FileNotFoundError(f"No saved models found in {models_dir}.")
        self._log.info("ls_equity: loaded ensemble %s for horizons %s; OOS from %s",
                       self.model_types, list(self.models.keys()), self.backtest_start.date())

    def _cube_columns(self) -> set[str]:
        q = text("SELECT column_name FROM information_schema.columns WHERE table_name = 'cube'")
        with self._context.store.engine.connect() as c:
            return set(pd.read_sql(q, c)["column_name"])

    def _target_column(self, cube_cols: set[str]) -> str:
        col = f"target_{self.target_type}"
        if col in cube_cols:
            return col
        if "target" in cube_cols:
            return "target"
        raise KeyError(f"Target column '{col}' not in cube; rebuild with '{self.target_type}'.")

    def _load_cube_projected(self):
        cube_cols = self._cube_columns()
        self.target_col = self._target_column(cube_cols)
        want = list(dict.fromkeys(self.feature_cols + self.categorical_cols))
        feats = [c for c in want if c in cube_cols]
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
            
        self._log.info("ls_equity: projected cube %d rows x %d cols (horizons=%s, date>=%s)",
                       len(self.cube), len(load_cols), horizons, self.backtest_start.date())

    def _load_prices_since(self, cutoff) -> pd.DataFrame:
        q = text("SELECT * FROM prices WHERE date >= :cut")
        with self._context.store.engine.connect() as c:
            return pd.read_sql(q, c, params={"cut": str(cutoff)}, parse_dates=["date"])

    def load_cube_and_returns(self):
        self._load_cube_projected()
        self.end = self._end_cfg if self._end_cfg is not None else pd.Timestamp(self.cube["date"].max())
        buffer_days = int(2.2 * (int(self._cfg.get("beta_window", 63))
                                 + int(self._cfg.get("vol_window", 63))) + 30)
        cutoff = (self.backtest_start - pd.Timedelta(days=buffer_days)).date()
        raw = du.prices_long_to_multiindex(self._load_prices_since(cutoff))
        close = du.extract_field(raw, "Close")
        mkt = self._cube_cfg.market_ticker
        rets = du.daily_returns(close)
        self.spy_ret = rets[mkt]
        drop = [mkt] + list(self._config.data_extract.get("other_tickers", []))
        self.stock_ret = rets.drop(columns=drop, errors="ignore")

    def _horizon_blend_weights(self, signals: dict, ir: dict) -> dict:
        if self._cfg.get("blend", "ir") == "equal" or not signals:
            return {h: 1.0 / len(signals) for h in signals} if signals else {}
        return ml.optimal_forecast_weights(signals, ir, shrink=float(self._cfg.get("blend_shrink", 0.5)))

    def predict_and_blend(self):
        blended = None
        for h, models in self.models.items():
            panel = panel_from_cube(self.cube, horizon=h, label_name=self.label_column,
                                    feature_cols=self.feature_cols + self.categorical_cols,
                                    target_type=self.target_type)
            panel = panel[(panel["date"] >= self.backtest_start) & (panel["date"] <= self.end)]
            if panel.empty:
                continue
            df = panel[["date", "ticker"]].copy()
            scores, _ = ml.ensemble_predict(models, panel, self.feature_cols)
            df["z"] = scores.to_numpy()
            df["z"] = df.groupby("date")["z"].transform(
                lambda s: (s - s.mean()) / (s.std() if s.std() > 0 else np.nan))
            df = df.rename(columns={"z": f"z_{h}"})
            blended = df if blended is None else blended.merge(df, on=["date", "ticker"], how="outer")

        zc = [f"z_{h}" for h in self.models if f"z_{h}" in (blended.columns if blended is not None else [])]
        if not zc:
            raise RuntimeError("ls_equity: no horizon produced a signal in the window.")
        hs = [int(c.split("_")[1]) for c in zc]
        signals = {h: blended[f"z_{h}"].to_numpy() for h in hs}
        ir = {h: self.train_ic.get(h, np.nan) for h in hs}
        weights = self._horizon_blend_weights(signals, ir)
        w = np.array([weights[h] for h in hs])
        z = blended[zc].to_numpy()
        mask = ~np.isnan(z)
        wsum = np.where(mask, w, 0).sum(axis=1)
        blended["combined"] = np.where(wsum > 0,
                                       np.nansum(np.where(mask, z * w, 0), axis=1) / np.where(wsum > 0, wsum, 1),
                                       np.nan)
        self.signal = blended.pivot(index="date", columns="ticker", values="combined")
        self.signal.index = pd.to_datetime(self.signal.index)
        self._log.info("ls_equity: signal matrix %s days x %s tickers", *self.signal.shape)

    def _sector_map(self) -> dict:
        tk = self._context.store.load("sp500_tickers")
        if tk.empty:
            return {}
        col = ("industry_group" if "industry_group" in tk.columns
               else "sector" if "sector" in tk.columns else None)
        return dict(zip(tk["ticker"], tk[col])) if col else {}

    def simulate(self):
        c, inp = self._cfg, self._inputs
        sector_neutral = bool(c.get("sector_neutral", False))
        sector_map = self._sector_map() if sector_neutral else None
        return simulate_portfolio_opt(
            signal=self.signal, stock_ret=self.stock_ret, spy_ret=self.spy_ret,
            starting_capital=float(inp.capital), target_ann_vol=float(inp.target_vol),
            beta_neutral=c.get("beta_neutral", True), pos_cap=c.get("pos_cap", 0.05),
            gross_cap=c.get("gross_cap", 3.0), step=c.get("step", 0.35),
            no_trade_band=c.get("no_trade_band", 0.0), beta_window=c.get("beta_window", 63),
            vol_window=c.get("vol_window", 63),
            fee_bps=float(c.get("fee_bps", inp.fee_bps)), spread_bps=float(c.get("spread_bps", inp.spread_bps)),
            rebalance_freq=c.get("rebalance_freq", 63), sector_map=sector_map,
            sector_neutral=sector_neutral, market_weight=0.0, vol_scaling=False)
