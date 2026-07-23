"""
allocation.py  (src/modelling/long_book/allocation.py)
------------------------------------------------------
The LONG BOOK "model": a long-only % allocation across the macro asset classes
(equity / gold / energy / 10Y-bond-TR / FX) with CASH as the residual/funding leg. There is
no ML here — the "prediction" is the risk-parity target weights + overlays:
  1. MIX   -- ERC / inverse-vol risk-parity (correlation-aware), EWMA or simple covariance;
  2. TREND -- a long-only overlay that scales an asset toward cash on a downtrend;
  3. TILT  -- a VIX-aware risk-on regime tilt (more equity/energy when calm, less in stress);
  4. SIZE  -- global leverage to a vol target (fixed cap or a 1x-stress→2x-calm responsive cap).
Reuses the shared risk-parity primitives (src/utils/risk_parity) and trend overlay
(src/utils/trend); consumed by the long-book strategy step.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.trend import trend_scale_long_only
from src.utils.risk_parity import base_weights, risk_on_score, series_metrics

_ANN: float = 252.0
CASH: str = "cash"


def asset_returns_from_macro(df: pd.DataFrame, include_fx: bool = True
                             ) -> tuple[pd.DataFrame, pd.Series]:
    """`macro_asset_prices` rows -> (risky daily returns [equity, gold, energy, bond, (fx)],
    cash daily return). Total-return legs -> pct_change; cash_rate (annual %) -> daily riskless
    drift (rate/100/252); FX -> pct_change (dropped if entirely absent)."""
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values("date").set_index("date")
    risky: dict[str, pd.Series] = {}
    if "equity_tr" in d.columns:
        risky["equity"] = d["equity_tr"].pct_change(fill_method=None)
    if "gold" in d.columns:
        risky["gold"] = d["gold"].pct_change(fill_method=None)
    if "energy" in d.columns and d["energy"].notna().any():
        risky["energy"] = d["energy"].pct_change(fill_method=None)
    if "bond_10y_tr" in d.columns:
        risky["bond"] = d["bond_10y_tr"].pct_change(fill_method=None)
    if include_fx and "fx_usdeur" in d.columns and d["fx_usdeur"].notna().any():
        risky["fx"] = d["fx_usdeur"].pct_change(fill_method=None)
    risky_df = pd.DataFrame(risky)
    cash = (d["cash_rate"].astype(float) / 100.0 / _ANN
            if "cash_rate" in d.columns else pd.Series(0.0, index=d.index))
    return risky_df, cash.rename(CASH)


def allocation_backtest(rets: pd.DataFrame, cash_ret: pd.Series, *,
                        scheme: str = "erc", vol_window: int = 63, rebalance_freq: int = 21,
                        trend_enabled: bool = True, trend_lookbacks: tuple[int, ...] = (63, 126, 252),
                        trend_scheme: str = "linear", trend_floor: float = 0.0,
                        trend_vol_window: int = 63, trend_cap: float = 2.0,
                        portfolio_vol_target: float = 0.10, max_leverage: float = 2.0,
                        fee_bps: float = 2.0, spread_bps: float = 8.0,
                        cov_mode: str = "std", cov_halflife: int = 42, vol_mode: str = "std",
                        lever_on: str = "scaled", risk_on: bool = False, vix: pd.Series | None = None,
                        offensive: tuple[str, ...] = ("equity", "energy"),
                        off_share_range: tuple[float, float] = (0.15, 0.85),
                        lev_responsive: bool = False, lev_min: float = 1.0,
                        lev_max: float = 2.0) -> dict[str, object]:
    """Run the layered long-book allocation. Returns net_ret / gross_ret / weights (levered) /
    cash_weight / leverage / turnover / cost / contrib / scale / alloc (pre-lev) / alloc_cash / score."""
    rets = rets.sort_index()
    cash_ret = cash_ret.reindex(rets.index).fillna(0.0)

    score = risk_on_score(rets, vix) if (risk_on or lev_responsive) else None
    W = base_weights(rets, vol_window, scheme, rebalance_freq, cov_mode=cov_mode,
                     cov_halflife=cov_halflife, score=(score if risk_on else None),
                     offensive=offensive, off_share_range=off_share_range)

    if trend_enabled:
        prices = (1.0 + rets.fillna(0.0)).cumprod()
        scale = trend_scale_long_only(prices, list(trend_lookbacks), trend_vol_window,
                                      trend_scheme, trend_floor, trend_cap)
        scale = scale.reindex_like(W).fillna(1.0).clip(lower=0.0, upper=1.0)
    else:
        scale = pd.DataFrame(1.0, index=W.index, columns=W.columns)
    w_risky = (W * scale).fillna(0.0)                       # sum <= 1; remainder -> cash

    # global vol target; target the BASE book's vol when lever_on='base' so trend-to-cash de-risks
    book = (W.fillna(0.0) if lever_on == "base" else w_risky)
    held0 = book.shift(1)
    pre_ret = (held0 * rets).sum(axis=1) + (1.0 - held0.sum(axis=1)) * cash_ret
    if vol_mode == "ewma":
        pv = np.sqrt(pre_ret.pow(2).ewm(halflife=cov_halflife).mean()).shift(1) * np.sqrt(_ANN)
    else:
        pv = pre_ret.rolling(vol_window, min_periods=max(10, vol_window // 2)).std().shift(1) * np.sqrt(_ANN)
    raw_lev = portfolio_vol_target / pv
    if lev_responsive and score is not None:
        cap = (lev_min + (lev_max - lev_min) * score.reindex(raw_lev.index)).clip(lev_min, lev_max)
        lev = np.minimum(raw_lev, cap.fillna(lev_min))
    else:
        lev = raw_lev.clip(upper=max_leverage)
    lev = lev.where(np.isfinite(lev)).fillna(1.0)
    w_lev = w_risky.mul(lev, axis=0)
    cash_w = 1.0 - w_lev.sum(axis=1)

    w_held = w_lev.shift(1)
    contrib = w_held * rets
    gross = contrib.sum(axis=1) + cash_w.shift(1) * cash_ret
    turnover = (w_lev - w_lev.shift(1)).abs().sum(axis=1)
    cost = turnover * (fee_bps + spread_bps) / 1e4
    net = (gross - cost)
    alloc_cash = 1.0 - w_risky.sum(axis=1)

    return {"net_ret": net.iloc[1:], "gross_ret": gross.iloc[1:], "weights": w_lev,
            "cash_weight": cash_w, "leverage": lev, "turnover": turnover.iloc[1:],
            "cost": cost.iloc[1:], "contrib": contrib.iloc[1:], "scale": scale,
            "alloc": w_risky, "alloc_cash": alloc_cash, "score": score}


def per_asset_metrics(rets: pd.DataFrame, cash_ret: pd.Series,
                      rf_annual: float = 0.0) -> pd.DataFrame:
    """Standalone buy-and-hold return / vol / Sharpe / maxDD per risky asset + cash."""
    rows = {a: series_metrics(rets[a], rf_annual) for a in rets.columns}
    rows[CASH] = series_metrics(cash_ret, rf_annual)
    return pd.DataFrame(rows).T[["ann_return", "ann_vol", "sharpe", "max_drawdown"]]


def sweep_trend_params(rets: pd.DataFrame, cash_ret: pd.Series, grid: list[dict],
                       base_kwargs: dict, rf_annual: float = 0.0) -> pd.DataFrame:
    """Backtest a grid of overlay params (each dict overrides base_kwargs) -> comparison table."""
    rows = []
    for override in grid:
        clean = {k: v for k, v in override.items() if not k.startswith("_")}
        res = allocation_backtest(rets, cash_ret, **{**base_kwargs, **clean})
        m = series_metrics(res["net_ret"], rf_annual)
        label = override.get("_label") or ", ".join(f"{k}={v}" for k, v in clean.items())
        rows.append({"config": label, **{k: round(v, 3) for k, v in m.items()}})
    return pd.DataFrame(rows)
