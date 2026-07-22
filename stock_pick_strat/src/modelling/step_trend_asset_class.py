"""
step_trend_asset_class.py  (src/modelling/step_trend_asset_class.py)
--------------------------------------------------------------------
StepTrendAssetClass — builds the multi-asset TIME-SERIES-MOMENTUM (trend / CTA) sleeve: a
directional cross-asset-class book (long assets trending up, short those trending down), sized by
volatility. It is the positive-skew / "crisis-alpha" diversifier for the long-biased equity book —
price-only, NO macro inputs (see src/modelling/utils_trend_asset/trend_signal.py).

fit()     — on the TRAIN window, build positions and calibrate a scalar so the sleeve realizes
            `sleeve_vol_target` annual vol; persist the model (params + calibration) as JSON.
predict() — apply to any window -> per-asset weights + daily NET sleeve returns (calibrated).
run()     — load prices for the configured asset universe, fit on train, predict on full history,
            save daily returns to the `trend_asset_returns` table + the model artifact.
"""
from __future__ import annotations

import json

import pandas as pd
from omegaconf import DictConfig
from sqlalchemy import bindparam, text

from src.constants.constants import TREND_ASSET_MODEL_FILE, TREND_ASSET_RETURNS_TABLE
from src.context import Context
from src.data_aggregate.utils import data_utils as du
from src.modelling.utils_trend_asset import (
    apply_class_budget,
    carry_forecast,
    combine_signals,
    combined_forecast,
    realized_ann_vol,
    sleeve_returns,
    value_forecast,
    vol_scaled_positions,
    vol_target_scalar,
)
from src.utils.step import Step


class StepTrendAssetClass(Step):

    def __init__(self, context: Context, config: DictConfig):
        super().__init__(context=context, config=config)
        self._cfg = config.trend_asset
        self._train_end = pd.Timestamp(config.train.end_date)
        self._scalar: float = 1.0
        self.model: dict = {}

    # ------------------------------------------------------------------ #
    def run(self) -> None:
        close = self.load_prices()
        carry = self._load_carry()
        self.fit(close, carry)
        returns, positions = self.predict(close, carry)
        self.save(returns, positions)

    # ------------------------------------------------------------------ #
    def load_prices(self) -> pd.DataFrame:
        """Wide close matrix (date x asset) for the configured trend universe, from `prices`."""
        assets = [str(a) for a in self._cfg.assets]
        q = text("SELECT date, ticker, close FROM prices WHERE ticker IN :assets").bindparams(
            bindparam("assets", expanding=True))
        with self._context.store.engine.connect() as c:
            long = pd.read_sql(q, c, params={"assets": assets}, parse_dates=["date"])
        if long.empty:
            raise ValueError(f"trend sleeve: no prices found for assets {assets}")
        close = du.extract_field(du.prices_long_to_multiindex(long), "Close")
        present = [a for a in assets if a in close.columns]
        missing = [a for a in assets if a not in present]
        if missing:
            self._log.warning("trend sleeve: %d asset(s) absent from `prices`, dropped: %s",
                              len(missing), missing)
        self.assets = present
        self.close = close[present].sort_index()
        self._log.info("trend sleeve universe: %s (%d days %s -> %s)", present,
                       len(self.close), self.close.index.min().date(), self.close.index.max().date())
        return self.close

    # ------------------------------------------------------------------ #
    def _forecast(self, close: pd.DataFrame, carry: pd.DataFrame | None) -> pd.DataFrame:
        """Per-class DIRECTION as a blend of orthogonal signals: TREND (3-12m momentum) + VALUE
        (long-horizon reversal) + optional CARRY (rates). Weights from `trend_asset.signals`.
        The value leg is negatively correlated with trend -> steadies the whipsaw; carry adds a
        rates-driven view (wired via _load_carry / the macro table)."""
        vw, cap = int(self._cfg.vol_window), float(self._cfg.signal_cap)
        sw = dict(self._cfg.get("signals", {}) or {"trend": 1.0})
        forecasts = {"trend": combined_forecast(close, list(self._cfg.lookbacks), vw, cap)}
        if float(sw.get("value", 0.0)) > 0:
            forecasts["value"] = value_forecast(close, int(self._cfg.get("value_lookback", 1260)), vw, cap)
        if carry is not None and float(sw.get("carry", 0.0)) > 0:
            forecasts["carry"] = carry_forecast(carry, close, vw, cap)
        self._log.info("trend sleeve signals: %s (weights %s)", list(forecasts),
                       {k: sw.get(k, 1.0) for k in forecasts})
        return combine_signals(forecasts, sw, cap)

    def _positions(self, close: pd.DataFrame, carry: pd.DataFrame | None = None) -> pd.DataFrame:
        """Vol-scaled weights (date x asset) from the multi-signal forecast, then risk-budgeted
        ACROSS asset classes so the sleeve times equity / bonds / commodities / FX on equal footing."""
        w = vol_scaled_positions(self._forecast(close, carry), close, int(self._cfg.vol_window),
                                 float(self._cfg.per_asset_vol_target))
        amap = dict(self._cfg.get("asset_classes", {}) or {})
        if amap:
            w = apply_class_budget(w, amap, dict(self._cfg.get("class_budgets", {}) or {}))
        return w

    def _load_carry(self) -> pd.DataFrame | None:
        """Best-effort per-asset annualized carry from the `macro` rates table (bond curve slope,
        FX short-rate differential). Returns None until a `trend_asset.carry_map` is configured —
        the sleeve then runs trend+value only. (Hook for the phase-2 rates/curve wiring.)"""
        cmap = dict(self._cfg.get("carry_map", {}) or {})
        if not cmap:
            self._log.info("trend sleeve: no carry_map configured — running trend+value only.")
            return None
        try:
            macro = self._context.store.load("macro")
        except Exception as e:                                      # noqa: BLE001
            self._log.warning("trend sleeve: macro table unavailable (%s) — carry off.", e)
            return None
        return _build_carry(macro, cmap)

    def fit(self, close: pd.DataFrame, carry: pd.DataFrame | None = None) -> dict:
        """Calibrate the sleeve-level vol scalar on the TRAIN window and persist the model."""
        tr = close[close.index <= self._train_end]
        rets = sleeve_returns(self._positions(tr, carry), tr, float(self._cfg.fee_bps),
                              float(self._cfg.spread_bps), int(self._cfg.rebalance_freq))
        self._scalar = vol_target_scalar(rets["ret"], float(self._cfg.sleeve_vol_target))
        self.model = {"assets": list(self._cfg.assets), "lookbacks": list(self._cfg.lookbacks),
                      "vol_window": int(self._cfg.vol_window), "signal_cap": float(self._cfg.signal_cap),
                      "per_asset_vol_target": float(self._cfg.per_asset_vol_target),
                      "sleeve_vol_target": float(self._cfg.sleeve_vol_target),
                      "rebalance_freq": int(self._cfg.rebalance_freq),
                      "vol_scalar": round(self._scalar, 4),
                      "train_end": str(self._train_end.date()),
                      "train_realized_ann_vol": round(realized_ann_vol(rets["ret"]), 4)}
        self._log.info("trend sleeve fit: train ann-vol %.3f -> calibration scalar %.3f "
                       "(target %.2f)", self.model["train_realized_ann_vol"], self._scalar,
                       float(self._cfg.sleeve_vol_target))
        return self.model

    def predict(self, close: pd.DataFrame,
                carry: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Calibrated per-asset weights + daily NET sleeve returns over the full window."""
        weights = self._positions(close, carry) * self._scalar
        rets = sleeve_returns(weights, close, float(self._cfg.fee_bps),
                              float(self._cfg.spread_bps), int(self._cfg.rebalance_freq))
        oos = rets[rets.index > self._train_end]["ret"]
        self._log.info("trend sleeve predict: full ann-vol %.3f, Sharpe %.2f (OOS ann-vol %.3f, "
                       "Sharpe %.2f)", realized_ann_vol(rets["ret"]),
                       _sharpe(rets["ret"]), realized_ann_vol(oos), _sharpe(oos))
        return rets, weights

    # ------------------------------------------------------------------ #
    def save(self, returns: pd.DataFrame, positions: pd.DataFrame) -> None:
        out = returns.rename_axis("date").reset_index()
        out["date"] = pd.to_datetime(out["date"]).dt.date.astype(str)
        self._context.store.replace(TREND_ASSET_RETURNS_TABLE, out)
        models_dir = self._context.paths["MODELS_DIR"]
        models_dir.mkdir(parents=True, exist_ok=True)
        (models_dir / TREND_ASSET_MODEL_FILE).write_text(json.dumps(self.model, indent=2))
        self._log.info("saved %d trend-sleeve daily returns to `%s` + model %s",
                       len(out), TREND_ASSET_RETURNS_TABLE, models_dir / TREND_ASSET_MODEL_FILE)


def _sharpe(ret: pd.Series) -> float:
    r = ret.dropna()
    return float(r.mean() / r.std() * (252.0 ** 0.5)) if len(r) > 2 and r.std() > 0 else 0.0


def _build_carry(macro: pd.DataFrame, carry_map: dict) -> pd.DataFrame | None:
    """Per-asset annualized carry (date x asset) from the `macro` rates table. `carry_map` is
    asset -> list of [series, sign]; carry = sum(sign * rate)/100 (rates in %). E.g. bond curve
    slope TLT: [[DGS10, 1], [DGS3MO, -1]]; FX EURUSD: [[ECBDFR, 1], [DGS3MO, -1]]."""
    m = macro.copy()
    if "variable" in m.columns and "value" in m.columns:            # long -> wide
        m = m.pivot_table(index="date", columns="variable", values="value")
    elif "date" in m.columns:
        m = m.set_index("date")
    m.index = pd.to_datetime(m.index)
    out: dict[str, pd.Series] = {}
    for asset, terms in carry_map.items():
        s = None
        for series, sign in terms:
            if series in m.columns:
                s = (0.0 if s is None else s) + float(sign) * m[series].astype(float) / 100.0
        if s is not None:
            out[asset] = s
    return pd.DataFrame(out) if out else None
