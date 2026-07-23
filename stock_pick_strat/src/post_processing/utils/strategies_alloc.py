"""
strategies_alloc.py  (src/post_processing/utils/strategies_alloc.py)
--------------------------------------------------------------------
Multi-asset ALLOCATION backtest: allocate % of a LONG-ONLY book across risky sleeves
(equity, gold, 10Y bond, [FX], [L/S alpha]) with CASH as the residual / funding leg.

Three layers, kept separate (each can be turned off to see its marginal value):
  1. MIX     -- base risk-parity weights across the risky assets. Two schemes:
                * inverse_vol : w_i ∝ 1/σ_i  (ignores correlation)
                * erc         : Equal Risk Contribution — equalizes each asset's
                                contribution to PORTFOLIO vol (correlations included);
                                more diversified when correlations are heterogeneous.
  2. TREND   -- a LONG-ONLY overlay (src/utils/trend.trend_scale_long_only) that scales
                each risky weight toward `trend_floor` when the asset rolls over; the freed
                weight goes to CASH. This is the crisis de-risking mechanism.
  3. SIZE    -- one global leverage that scales the risky book to `portfolio_vol_target`
                (capped by `max_leverage`); cash absorbs the rest (borrow if levered >1).
Fees are charged on weight turnover. Everything is point-in-time (weights at t use data
up to t-1, applied to day-t returns) so there is no look-ahead.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.trend import trend_scale_long_only

_ANN: float = 252.0
CASH: str = "cash"


# --------------------------------------------------------------------------- #
# Inputs                                                                       #
# --------------------------------------------------------------------------- #
def asset_returns_from_macro(df: pd.DataFrame, include_fx: bool = True
                             ) -> tuple[pd.DataFrame, pd.Series]:
    """`macro_asset_prices` rows -> (risky daily returns [equity, gold, bond, (fx)], cash
    daily return). Total-return legs -> pct_change; cash_rate (annual %) -> daily riskless
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


# --------------------------------------------------------------------------- #
# Weighting schemes                                                            #
# --------------------------------------------------------------------------- #
def erc_weights(cov: np.ndarray, budget: np.ndarray | None = None,
                max_iter: int = 10_000, tol: float = 1e-10) -> np.ndarray:
    """Equal-Risk-Contribution weights (long-only, sum=1) via cyclical coordinate descent
    (Griveau-Billion / Spinu). Per coordinate it solves the risk-parity fixed point
        σ_ii x_i² + (Σ_{j≠i} σ_ij x_j) x_i − b_i = 0
    then normalizes, so every asset ends with the SAME contribution to portfolio vol
    (b_i = equal budget). Assets with a non-positive variance (degenerate) get weight 0."""
    n = cov.shape[0]
    if n == 0:
        return np.empty(0)
    diag = np.diag(cov).astype(float)
    good = diag > 0
    if not good.any():
        return np.full(n, 1.0 / n)
    b = (np.ones(n) / good.sum()) if budget is None else np.asarray(budget, float)
    b = np.where(good, b, 0.0)
    x = np.where(good, np.sqrt(np.where(good, b, 0.0)) / np.sqrt(np.where(good, diag, 1.0)), 0.0)
    ssum = x.sum()
    x = x / ssum if ssum > 0 else np.where(good, 1.0 / good.sum(), 0.0)
    for _ in range(max_iter):
        x_prev = x.copy()
        for i in range(n):
            if not good[i]:
                x[i] = 0.0
                continue
            a1 = float(cov[i, :] @ x - cov[i, i] * x[i])        # Σ_{j≠i} σ_ij x_j
            sii = float(cov[i, i])
            x[i] = (-a1 + np.sqrt(max(a1 * a1 + 4.0 * sii * b[i], 0.0))) / (2.0 * sii)
        if np.max(np.abs(x - x_prev)) < tol:
            break
    ssum = x.sum()
    return x / ssum if ssum > 0 else np.full(n, 1.0 / n)


def risk_contributions(cov: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Fractional risk contribution per asset: RC_i = w_i (Σw)_i / (wᵀΣw). Sums to 1.
    Used to VERIFY ERC (all entries ≈ 1/n) vs inverse-vol (unequal when correlated)."""
    port_var = float(w @ cov @ w)
    if port_var <= 0:
        return np.full_like(w, np.nan, dtype=float)
    return (w * (cov @ w)) / port_var


def _cov_window(window_rets: pd.DataFrame, min_obs: int = 20) -> tuple[np.ndarray, list[str]]:
    """Annualized SIMPLE covariance over the assets with >= `min_obs` obs in the window."""
    cols = [c for c in window_rets.columns if window_rets[c].notna().sum() >= min_obs]
    if not cols:
        return np.empty((0, 0)), []
    cov = window_rets[cols].dropna(how="any").cov().to_numpy() * _ANN
    return cov, cols


def _ewma_cov(window_rets: pd.DataFrame, halflife: int, min_obs: int = 20) -> tuple[np.ndarray, list[str]]:
    """Annualized EXPONENTIALLY-WEIGHTED covariance (recent days weighted more, no 63d cliff).
    Weights decay with `halflife` days; reacts faster to a fresh vol spike than a flat window."""
    cols = [c for c in window_rets.columns if window_rets[c].notna().sum() >= min_obs]
    r = window_rets[cols].dropna(how="any") if cols else window_rets.iloc[:0]
    if len(r) < min_obs:
        return np.empty((0, 0)), []
    n = len(r)
    lam = 0.5 ** (1.0 / float(halflife))
    w = lam ** np.arange(n)[::-1]
    w = w / w.sum()
    x = r.to_numpy() - np.average(r.to_numpy(), axis=0, weights=w)
    cov = (x * w[:, None]).T @ x
    return cov * _ANN, cols


def risk_on_score(rets: pd.DataFrame, vix: pd.Series | None = None, equity: str = "equity",
                  trend_win: int = 252, vol_hl: int = 42, z_win: int = 756) -> pd.Series:
    """Point-in-time RISK-ON score in [0,1]: HIGH when crash-probability is low (equity in an
    uptrend AND vol/VIX low vs their own recent history), LOW in stress. Equal-weight blend of:
      * equity 12m trend up (1/0),
      * equity EWMA-vol LOW vs its trailing z-score,
      * (if `vix` given) VIX LOW vs its trailing z-score.
    Shifted 1 day so date t uses only info up to t-1."""
    px = (1.0 + rets[equity].fillna(0.0)).cumprod()
    trend_on = (px.pct_change(trend_win) > 0).astype(float)
    evol = np.sqrt(rets[equity].pow(2).ewm(halflife=vol_hl).mean()) * np.sqrt(_ANN)
    ez = (evol - evol.rolling(z_win, min_periods=252).mean()) / evol.rolling(z_win, min_periods=252).std()
    parts = [trend_on, (0.5 - 0.5 * ez).clip(0.0, 1.0)]
    if vix is not None:
        v = vix.reindex(rets.index).ffill()
        vz = (v - v.rolling(z_win, min_periods=252).mean()) / v.rolling(z_win, min_periods=252).std()
        parts.append((0.5 - 0.5 * vz).clip(0.0, 1.0))
    s = sum(parts) / float(len(parts))
    return s.clip(0.0, 1.0).shift(1)


def _tilted_budget(cols: list[str], score: float, offensive: tuple[str, ...],
                   off_range: tuple[float, float]) -> np.ndarray:
    """Regime-tilted ERC risk budgets: give the OFFENSIVE sleeves (equity/energy) a larger risk
    share when `score` (risk-on) is high, the DEFENSIVE sleeves (bond/gold/fx) more when low.
    off_share = clip(0.2 + 0.6*score, *off_range); split equally within each group. Sums to 1."""
    lo, hi = off_range
    off = [c for c in cols if c in offensive]
    deff = [c for c in cols if c not in offensive]
    if not off or not deff:                                   # only one group live -> equal
        return np.ones(len(cols)) / len(cols)
    off_share = float(np.clip(0.2 + 0.6 * score, lo, hi))
    b = np.zeros(len(cols))
    for c in off:
        b[cols.index(c)] = off_share / len(off)
    for c in deff:
        b[cols.index(c)] = (1.0 - off_share) / len(deff)
    return b / b.sum()


def base_weights(rets: pd.DataFrame, window: int, scheme: str, rebalance_freq: int, *,
                 cov_mode: str = "std", cov_halflife: int = 42, score: pd.Series | None = None,
                 offensive: tuple[str, ...] = ("equity", "energy"),
                 off_share_range: tuple[float, float] = (0.15, 0.85)) -> pd.DataFrame:
    """date x asset base MIX weights (sum=1 across the live risky assets), point-in-time:
    recompute every `rebalance_freq` days on the trailing `window` (strictly up to t-1),
    hold (ffill) in between. `scheme` in {erc, inverse_vol}; `cov_mode` in {std, ewma}.
    If `score` (a risk-on series in [0,1]) is given, ERC/inverse-vol budgets are REGIME-TILTED
    toward the offensive sleeves when risk-on is high (see _tilted_budget)."""
    idx = rets.index
    reb = np.zeros(len(idx), dtype=bool)
    reb[::max(1, int(rebalance_freq))] = True
    W = pd.DataFrame(index=idx, columns=rets.columns, dtype=float)
    for pos, t in enumerate(idx):
        if not reb[pos]:
            continue
        win = rets.iloc[max(0, pos - window):pos]                  # exclusive of t
        cov, cols = (_ewma_cov(win, cov_halflife) if cov_mode == "ewma" else _cov_window(win))
        if not cols:
            continue
        budget = None
        if score is not None:
            s = score.get(t, np.nan)
            s = 0.5 if not np.isfinite(s) else float(s)
            budget = _tilted_budget(cols, s, offensive, off_share_range)
        if scheme == "erc":
            w = erc_weights(cov, budget=budget)
        elif scheme == "inverse_vol":
            vol = np.sqrt(np.diag(cov))
            inv = np.where(vol > 0, 1.0 / vol, 0.0)
            if budget is not None:
                inv = inv * budget                                 # tilt inverse-vol too
            w = inv / inv.sum() if inv.sum() > 0 else np.full(len(cols), 1.0 / len(cols))
        else:
            raise ValueError(f"unknown scheme '{scheme}' (use erc | inverse_vol)")
        W.loc[t, cols] = w
    return W.ffill()


# --------------------------------------------------------------------------- #
# Backtest                                                                     #
# --------------------------------------------------------------------------- #
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
    """Run the layered allocation. Returns a dict with:
      net_ret, gross_ret (Series), weights (date x risky asset, levered), cash_weight (Series),
      leverage, turnover, cost (Series), contrib (date x asset return contribution), scale, score.

    Risk model / regime knobs:
      * cov_mode {std|ewma}, vol_mode {std|ewma}, cov_halflife -- the vol estimator (EWMA reacts
        faster, no window cliff);
      * risk_on + vix -- REGIME TILT: lift equity/energy risk budget when crash-prob is low
        (equity uptrend + low vol/VIX), cut it in stress (see risk_on_score / _tilted_budget);
      * lever_on {scaled|base} -- target the vol of the trend-SCALED book (legacy) or the BASE
        book (so trend-to-cash genuinely cuts exposure instead of the governor re-levering)."""
    rets = rets.sort_index()
    cash_ret = cash_ret.reindex(rets.index).fillna(0.0)

    # 1) base risk-parity mix (optionally EWMA cov + regime-tilted risk budgets).
    #    The risk-on score also drives the leverage cap (below), so compute it whenever
    #    either the budget tilt OR the responsive leverage cap is on.
    score = risk_on_score(rets, vix) if (risk_on or lev_responsive) else None
    W = base_weights(rets, vol_window, scheme, rebalance_freq, cov_mode=cov_mode,
                     cov_halflife=cov_halflife, score=(score if risk_on else None),
                     offensive=offensive, off_share_range=off_share_range)

    # 2) long-only trend overlay -> scale risky weights toward cash on downtrends
    if trend_enabled:
        prices = (1.0 + rets.fillna(0.0)).cumprod()
        scale = trend_scale_long_only(prices, list(trend_lookbacks), trend_vol_window,
                                      trend_scheme, trend_floor, trend_cap)
        scale = scale.reindex_like(W).fillna(1.0).clip(lower=0.0, upper=1.0)
    else:
        scale = pd.DataFrame(1.0, index=W.index, columns=W.columns)
    w_risky = (W * scale).fillna(0.0)                       # sum <= 1; remainder -> cash

    # 3) global vol target: leverage to hit portfolio_vol_target. Target the BASE book's vol when
    #    lever_on='base' so trend-to-cash reduces exposure (the governor doesn't re-lever it back).
    book = (W.fillna(0.0) if lever_on == "base" else w_risky)
    held0 = book.shift(1)
    pre_ret = (held0 * rets).sum(axis=1) + (1.0 - held0.sum(axis=1)) * cash_ret
    if vol_mode == "ewma":
        pv = np.sqrt(pre_ret.pow(2).ewm(halflife=cov_halflife).mean()).shift(1) * np.sqrt(_ANN)
    else:
        pv = pre_ret.rolling(vol_window, min_periods=max(10, vol_window // 2)).std().shift(1) * np.sqrt(_ANN)
    raw_lev = portfolio_vol_target / pv
    if lev_responsive and score is not None:
        # leverage CEILING scales with the regime: lev_min in stress (score→0), lev_max in
        # calm (score→1). So the book can lever up when crash-prob is low and is forced
        # unlevered when vol/VIX spike. Warmup (no score) -> conservative lev_min.
        cap = (lev_min + (lev_max - lev_min) * score.reindex(raw_lev.index)).clip(lev_min, lev_max)
        lev = np.minimum(raw_lev, cap.fillna(lev_min))
    else:
        lev = raw_lev.clip(upper=max_leverage)
    lev = lev.where(np.isfinite(lev)).fillna(1.0)
    w_lev = w_risky.mul(lev, axis=0)
    cash_w = 1.0 - w_lev.sum(axis=1)                        # negative => borrowing (levered)

    # 4) daily NET return with fees on weight turnover
    w_held = w_lev.shift(1)
    contrib = w_held * rets                                 # per-asset return contribution
    gross = contrib.sum(axis=1) + cash_w.shift(1) * cash_ret
    turnover = (w_lev - w_lev.shift(1)).abs().sum(axis=1)
    cost = turnover * (fee_bps + spread_bps) / 1e4
    net = (gross - cost)

    # pre-leverage ALLOCATION (sums to 1 with cash) — the intuitive weight-evolution view
    alloc_cash = 1.0 - w_risky.sum(axis=1)

    return {"net_ret": net.iloc[1:], "gross_ret": gross.iloc[1:], "weights": w_lev,
            "cash_weight": cash_w, "leverage": lev, "turnover": turnover.iloc[1:],
            "cost": cost.iloc[1:], "contrib": contrib.iloc[1:], "scale": scale,
            "alloc": w_risky, "alloc_cash": alloc_cash, "score": score}


# --------------------------------------------------------------------------- #
# Metrics                                                                      #
# --------------------------------------------------------------------------- #
def series_metrics(ret: pd.Series, rf_annual: float = 0.0) -> dict[str, float]:
    """Annualized return / vol / Sharpe / max-drawdown of a daily return series."""
    r = ret.dropna()
    if len(r) < 2:
        return {"ann_return": np.nan, "ann_vol": np.nan, "sharpe": np.nan, "max_drawdown": np.nan}
    equity = (1.0 + r).cumprod()
    ann = float(equity.iloc[-1] ** (_ANN / len(r)) - 1.0)
    vol = float(r.std() * np.sqrt(_ANN))
    sharpe = float((r.mean() - rf_annual / _ANN) / r.std() * np.sqrt(_ANN)) if r.std() > 0 else np.nan
    peak = equity.cummax()
    mdd = float(((equity - peak) / peak).min())
    return {"ann_return": ann, "ann_vol": vol, "sharpe": sharpe, "max_drawdown": mdd}


def per_asset_metrics(rets: pd.DataFrame, cash_ret: pd.Series,
                      rf_annual: float = 0.0) -> pd.DataFrame:
    """Standalone buy-and-hold return / vol / Sharpe / maxDD per risky asset + cash."""
    rows = {a: series_metrics(rets[a], rf_annual) for a in rets.columns}
    rows[CASH] = series_metrics(cash_ret, rf_annual)
    return pd.DataFrame(rows).T[["ann_return", "ann_vol", "sharpe", "max_drawdown"]]


def daily_frame(net_ret: pd.Series, benchmark_ret: pd.Series, turnover: pd.Series,
                cost: pd.Series, starting_capital: float = 1_000_000.0) -> pd.DataFrame:
    """Assemble the daily frame `compute_metrics` / plotting expect (portfolio vs benchmark)."""
    idx = net_ret.dropna().index
    b = benchmark_ret.reindex(idx).fillna(0.0)
    df = pd.DataFrame(index=idx)
    df["net_ret"] = net_ret.reindex(idx).fillna(0.0)
    df["portfolio_value"] = (1.0 + df["net_ret"]).cumprod() * starting_capital
    df["spy_value"] = (1.0 + b).cumprod() * starting_capital
    df["turnover"] = turnover.reindex(idx).fillna(0.0)
    df["cost"] = cost.reindex(idx).fillna(0.0)
    return df


def sweep_trend_params(rets: pd.DataFrame, cash_ret: pd.Series, grid: list[dict],
                       base_kwargs: dict, rf_annual: float = 0.0) -> pd.DataFrame:
    """Backtest a grid of trend-overlay params (each dict overrides base_kwargs) and return
    a comparison table (ann/vol/Sharpe/maxDD) so the default is chosen from evidence."""
    rows = []
    for override in grid:
        clean = {k: v for k, v in override.items() if not k.startswith("_")}
        res = allocation_backtest(rets, cash_ret, **{**base_kwargs, **clean})
        m = series_metrics(res["net_ret"], rf_annual)
        label = override.get("_label") or ", ".join(f"{k}={v}" for k, v in clean.items())
        rows.append({"config": label, **{k: round(v, 3) for k, v in m.items()}})
    return pd.DataFrame(rows)
