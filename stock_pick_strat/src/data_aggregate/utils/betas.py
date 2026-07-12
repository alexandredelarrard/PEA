"""
betas.py  (src/data_aggregate/utils/betas.py)  -- MULTI-FACTOR REWRITE
----------------------------------------------------------------------
Estimate each stock's loading on EVERY common factor by rolling regression of
its daily returns on:
    market + sector(per-stock) + size + value + momentum + quality + resvol
    + macro betas (d_yield_10y, d_vix, d_cpi_yoy, d_fed_balance_sheet, ...)

Two changes from the market+sector-only version:

1. RIDGE, not OLS. With ~10 correlated regressors (market, sector and momentum
   co-move heavily) on a 63-day window, plain OLS betas are wildly unstable and
   the split of a shared move across collinear factors is arbitrary. Ridge
   (penalty in STANDARDIZED space so every factor is penalized evenly)
   stabilizes the loadings and replaces the old ad-hoc shrink_weight.

2. Per-window standardization done via correlation matrices so we get
   RAW-SCALE betas back -- they can be applied directly to raw forward factor
   returns in targets.py, with no scaler to carry around.

Timing rule is unchanged and paramount: betas at t use only data up to t; they
are applied to FORWARD factor returns. Never fit on the forward window.

Efficiency: the factor-factor correlation matrix is identical across stocks, so
we compute it once per date; only the factor-vs-stock covariances are per-stock.
`step` lets you re-estimate every N days (e.g. weekly) and forward-fill, which
is both faster and steadier.

Output: {ticker: DataFrame indexed by date, columns = beta_<factor> for every
shared factor plus 'beta_sector'}.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def _rolling_moments(panel: pd.DataFrame, window: int, min_obs: int):
    """Rolling mean and std for each factor column (point-in-time, ending at t)."""
    roll = panel.rolling(window, min_periods=min_obs)
    return roll.mean(), roll.std()


def estimate_betas_for_stock(
    y: pd.Series,                 # stock daily returns
    shared: pd.DataFrame,         # market + style + macro (date x factor)
    sector: pd.Series,            # this stock's sector daily return
    window: int = 63,
    min_obs: int = 40,
    ridge: float = 5.0,
    step: int = 5,
) -> pd.DataFrame:
    """
    Rolling ridge betas for one stock. Regressors = shared factors + sector.
    Returns DataFrame (date x ['beta_<factor>'... , 'beta_sector']).
    """

    X = pd.concat([shared, sector.rename("sector")], axis=1)

    # drop fullna coll from X 
    try:
        X = X.drop(columns = ['d_ig_spread', 'd_hy_spread', 'd_cpi_yoy', 'd_fed_balance_sheet'])
    except:
        pass
    
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
        # standardize regressors in-window (mean 0, unit std) -> ridge is even
        mu = Xw.mean(0)
        sd = Xw.std(0)
        sd[sd < 1e-12] = np.nan
        Xs = (Xw - mu) / sd
        ok = ~np.isnan(Xs).any(0)                     # drop degenerate factors
        if ok.sum() == 0:
            continue
        Xs_ok = np.nan_to_num(Xs[:, ok])
        yc = yw - yw.mean()
        K = Xs_ok.shape[1]
        A = Xs_ok.T @ Xs_ok + ridge * np.eye(K)
        b_std = np.linalg.solve(A, Xs_ok.T @ yc)      # standardized betas
        b_raw = np.zeros(len(cols))
        b_raw[np.where(ok)[0]] = b_std / sd[ok]       # back to raw scale
        betas[t, :] = b_raw

    out = pd.DataFrame(betas, index=dates, columns=[f"beta_{c}" for c in cols])
    out = out.ffill()                                  # hold between re-estimations

    # Interpretable UNIVARIATE market beta (cov/var), separate from the
    # regularized joint loadings above. The joint betas are stabilized by ridge
    # and split collinear exposure, so beta_market there is NOT a literal market
    # beta -- use this column for reporting and the optimizer's beta-neutrality
    # constraint. It does not affect residualization (that uses the joint fit).
    if "market" in shared.columns:
        m = shared["market"].reindex(y.index)
        roll = pd.concat({"y": y, "m": m}, axis=1).rolling(window, min_periods=min_obs)
        out["beta_market_simple"] = (roll["m"].cov(y) / m.rolling(window, min_periods=min_obs).var())

    return out.reindex(y.index)


def estimate_all_betas(
    stock_returns: pd.DataFrame,      # date x ticker (stocks only)
    factor_panel: pd.DataFrame,       # date x factor (market + style + macro)
    sector_returns: pd.DataFrame,     # date x ticker, per-stock sector return
    window: int = 63,
    min_obs: int = 40,
    ridge: float = 5.0,
    step: int = 5,
) -> dict:
    """
    Multi-factor rolling ridge betas for every stock.

    NOTE the signature change vs the old (mkt_ret, sector_ret, shrink_weight):
    it now takes the full `factor_panel` (which already includes 'market') and a
    `ridge` strength. Update step_build_cube.estimate_betas accordingly.
    """
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
