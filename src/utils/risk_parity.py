"""
risk_parity.py  (src/utils/risk_parity.py)
------------------------------------------
SHARED risk-parity / weighting primitives, used by the portfolio sleeve blender
(src/portfolio/step_portfolio.py), the long-only equity book (src/strategies/utils/long_only.py)
and the sleeves' metrics — so there is one definition of ERC / EWMA covariance / point-in-time
weighting and no cross-package import.

  * erc_weights           -- Equal-Risk-Contribution weights (cyclical coordinate descent)
  * risk_contributions    -- fractional risk contribution per asset (ERC verification)
  * cov_window / ewma_cov -- annualized simple / EWMA covariance over a trailing window
  * base_weights          -- date x asset point-in-time MIX weights (erc | inverse_vol)
  * series_metrics        -- ann return / vol / Sharpe / maxDD of a daily return series
  * daily_frame           -- assemble the portfolio-vs-benchmark daily frame for metrics/plots
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd

_ANN: float = 252.0


# --------------------------------------------------------------------------- #
# Weighting schemes                                                            #
# --------------------------------------------------------------------------- #
def erc_weights(cov: np.ndarray, budget: np.ndarray | None = None, max_iter: int = 10_000, tol: float = 1e-10) -> np.ndarray:
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
            a1 = float(cov[i, :] @ x - cov[i, i] * x[i])  # Σ_{j≠i} σ_ij x_j
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


def cov_window(window_rets: pd.DataFrame, min_obs: int = 20) -> tuple[np.ndarray, list[str]]:
    """Annualized SIMPLE covariance over the assets with >= `min_obs` obs in the window."""
    cols = [c for c in window_rets.columns if window_rets[c].notna().sum() >= min_obs]
    if not cols:
        return np.empty((0, 0)), []
    complete = cast(pd.DataFrame, window_rets[cols].dropna(how="any"))
    cov = complete.cov().to_numpy() * _ANN
    return cov, cols


def ewma_cov(window_rets: pd.DataFrame, halflife: int, min_obs: int = 20) -> tuple[np.ndarray, list[str]]:
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


def base_weights(
    rets: pd.DataFrame,
    window: int,
    scheme: str,
    rebalance_freq: int,
    *,
    cov_mode: str = "std",
    cov_halflife: int = 42,
) -> pd.DataFrame:
    """date x asset base MIX weights (sum=1 across the live assets), point-in-time: recompute
    every `rebalance_freq` days on the trailing `window` (strictly up to t-1), hold (ffill) in
    between. `scheme` in {erc, inverse_vol}; `cov_mode` in {std, ewma}."""
    idx = rets.index
    reb = np.zeros(len(idx), dtype=bool)
    reb[:: max(1, int(rebalance_freq))] = True
    weights = pd.DataFrame(index=idx, columns=rets.columns, dtype=float)
    for pos, t in enumerate(idx):
        if not reb[pos]:
            continue
        win = rets.iloc[max(0, pos - window) : pos]  # exclusive of t
        cov, cols = ewma_cov(win, cov_halflife) if cov_mode == "ewma" else cov_window(win)
        if not cols:
            continue
        if scheme == "erc":
            w = erc_weights(cov)
        elif scheme == "inverse_vol":
            vol = np.sqrt(np.diag(cov))
            inv = np.where(vol > 0, 1.0 / vol, 0.0)
            w = inv / inv.sum() if inv.sum() > 0 else np.full(len(cols), 1.0 / len(cols))
        else:
            raise ValueError(f"unknown scheme '{scheme}' (use erc | inverse_vol)")
        weights.loc[t, cols] = w
    return weights.ffill()


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


def daily_frame(
    net_ret: pd.Series, benchmark_ret: pd.Series, turnover: pd.Series, cost: pd.Series, starting_capital: float = 1_000_000.0
) -> pd.DataFrame:
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
