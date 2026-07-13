"""
betas.py  (src/data_aggregate/utils/betas.py)  -- MULTI-FACTOR, ROBUST-FILTER
-----------------------------------------------------------------------------
Rolling ridge betas of each stock on market + sector + style + macro/commodity
factors. Change vs the previous version: the hardcoded try/except that dropped
['d_ig_spread','d_hy_spread','d_cpi_yoy','d_fed_balance_sheet'] is replaced by a
PRINCIPLED filter (`filter_daily_factors`) that drops any factor which does not
move at daily frequency (exact-zero on > max_zero_frac of days). This generically
removes stale low-frequency series (monthly CPI, weekly Fed balance sheet) whose
daily change is ~always zero and therefore cannot support an estimable beta --
without hardcoding names, and it is logged so nothing is dropped silently.

Timing rule unchanged: betas at t use only data up to t; applied to FORWARD
factor returns in targets.py.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from src.data_aggregate.utils.macro_factors import filter_daily_factors


def estimate_betas_for_stock(
    y: pd.Series,                 # stock daily returns
    shared: pd.DataFrame,         # market + style + commodity + macro (already filtered)
    sector: pd.Series,            # this stock's sector daily return
    window: int = 63,
    min_obs: int = 40,
    ridge: float = 5.0,
    step: int = 5,
) -> pd.DataFrame:
    """Rolling ridge betas for one stock. Regressors = shared factors + sector."""
    X = pd.concat([shared, sector.rename("sector")], axis=1)
    data = pd.concat([y.rename("y"), X], axis=1).dropna()
    if len(data) < min_obs:
        return pd.DataFrame(index=y.index, columns=[f"beta_{c}" for c in X.columns])

    cols = list(X.columns)
    yv = data["y"].to_numpy()
    Xv = data[cols].to_numpy()
    dates = data.index
    n = len(data)

    betas = np.full((n, len(cols)), np.nan)
    for t in range(min_obs - 1, n, step):
        lo = max(0, t - window + 1)
        Xw = Xv[lo:t + 1]
        yw = yv[lo:t + 1]
        if len(yw) < min_obs:
            continue
        mu = Xw.mean(0)
        sd = Xw.std(0)
        sd[sd < 1e-12] = np.nan
        Xs = (Xw - mu) / sd
        ok = ~np.isnan(Xs).any(0)                     # drop degenerate-in-window factors
        if ok.sum() == 0:
            continue
        Xs_ok = np.nan_to_num(Xs[:, ok])
        yc = yw - yw.mean()
        K = Xs_ok.shape[1]
        A = Xs_ok.T @ Xs_ok + ridge * np.eye(K)
        b_std = np.linalg.solve(A, Xs_ok.T @ yc)
        b_raw = np.zeros(len(cols))
        b_raw[np.where(ok)[0]] = b_std / sd[ok]
        betas[t, :] = b_raw

    out = pd.DataFrame(betas, index=dates, columns=[f"beta_{c}" for c in cols])
    out = out.ffill()

    # Interpretable univariate market beta (for reporting / optimizer neutrality).
    if "market" in shared.columns:
        m = shared["market"].reindex(y.index)
        roll = pd.concat({"y": y, "m": m}, axis=1).rolling(window, min_periods=min_obs)
        out["beta_market_simple"] = (roll["m"].cov(y) / m.rolling(window, min_periods=min_obs).var())

    return out.reindex(y.index)


def estimate_all_betas(
    stock_returns: pd.DataFrame,      # date x ticker (stocks only)
    factor_panel: pd.DataFrame,       # date x factor (market + style + commodity + macro)
    sector_returns: pd.DataFrame,     # date x ticker, per-stock sector return
    window: int = 63,
    min_obs: int = 40,
    ridge: float = 5.0,
    step: int = 5,
    filter_factors: bool = True,
) -> dict:
    """
    Multi-factor rolling ridge betas for every stock.

    `filter_factors`: drop non-daily-moving factors (stale monthly/weekly macro)
    ONCE, up front, so every stock regresses on the same clean factor set. This
    replaces the old hardcoded try/except drop. If you already filtered the panel
    in step_build_cube (via macro_factors.assemble_factor_panel), it's a no-op.
    """
    if filter_factors:
        factor_panel, dropped = filter_daily_factors(factor_panel)
        if dropped:
            print(f"[estimate_all_betas] excluded stale factors from regression: {dropped}")

    betas = {}
    for ticker in stock_returns.columns:
        if ticker not in sector_returns.columns:
            continue
        betas[ticker] = estimate_betas_for_stock(
            y=stock_returns[ticker],
            shared=factor_panel,
            sector=sector_returns[ticker],
            window=window,
            min_obs=min_obs,
            ridge=ridge,
            step=step,
        )
    return betas