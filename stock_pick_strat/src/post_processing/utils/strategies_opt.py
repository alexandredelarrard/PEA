"""
strategies_opt.py  (src/post_processing/utils/strategies_opt.py)
----------------------------------------------------------------
State-of-the-art construction to replace the top/bottom-decile equal-weight
long/short book. That naive book was volatile and expensive because it (a)
equal-weighted extreme-signal names -> concentrated in high-vol stocks, (b)
was only DOLLAR-neutral, not BETA-neutral -> carried net market exposure that
whipsawed it, and (c) jumped to a fresh target every day -> ~2.9x daily
turnover -> costs that swamped the signal.

This builds, each rebalance:

  ALPHA SLEEVE  (market-neutral, self-financing long/short)
    w* = argmax_w  a'w - (lambda/2) w' D w   s.t.  1'w = 0  and  beta'w = 0
    -> closed form:  w* proportional to  D^{-1} * (GLS residual of a on [1, beta])
       i.e. residualize the alpha against a constant AND market beta in the
       inverse-variance metric, then inverse-variance weight. This is the
       Grinold-Kahn characteristic portfolio with dollar + beta neutrality and
       an (idiosyncratic) risk model D. No external solver needed.
    Then: position-cap, re-neutralize, and VOLATILITY-TARGET the whole sleeve to
    a fixed ex-ante vol so the book's risk is stable through time.

  MARKET SLEEVE
    a deliberate, constant `market_weight` in SPY. Because the alpha sleeve is
    beta-neutral, TOTAL portfolio beta = market_weight exactly -- the market bet
    is sized on purpose, not smuggled in through the signal.

  TURNOVER CONTROL (Garleanu-Pedersen)
    don't jump to w*; trade PARTIALLY toward it:  w = w_prev + step*(w* - w_prev)
    plus an optional no-trade band. `step` in (0,1]; smaller = slower book =
    less turnover. This is the closed-form optimal-trading-rate idea and is the
    single biggest cost reducer.

All the heavy math is in pure functions (optimize_day, vol_target_scale) so it
is unit-tested; only the day loop touches state.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_ANN = 252.0


# --------------------------------------------------------------------------- #
# Pure building blocks                                                         #
# --------------------------------------------------------------------------- #
def _neutralize(a: np.ndarray, X: np.ndarray, w_metric: np.ndarray) -> np.ndarray:
    """GLS residual of `a` on columns of `X` using diagonal weights `w_metric`.
    Zeros out a's components along each column of X in that metric."""
    W = w_metric
    XtW = X.T * W                       # (k,N)
    coef = np.linalg.solve(XtW @ X + 1e-12 * np.eye(X.shape[1]), XtW @ a)
    return a - X @ coef


def optimize_day(alpha: np.ndarray, beta: np.ndarray, var: np.ndarray,
                 beta_neutral: bool = True, pos_cap: float | None = 0.03) -> np.ndarray:
    """
    Closed-form mean-variance long/short weights with dollar (and optional beta)
    neutrality and a diagonal idiosyncratic-risk model.

    alpha : centered cross-sectional score (magnitude matters; z-scores ideal)
    beta  : per-name market beta
    var   : per-name idiosyncratic DAILY variance (>0)
    Returns weights summing to ~0 (dollar-neutral) and beta'w ~ 0 if beta_neutral.
    Scale is arbitrary here (set later by vol targeting).
    """
    n = len(alpha)
    var = np.clip(var, np.nanpercentile(var[np.isfinite(var)], 5) if np.isfinite(var).any() else 1e-6, None)
    invd = 1.0 / var
    ones = np.ones(n)
    X = np.column_stack([ones, beta]) if beta_neutral else ones.reshape(-1, 1)

    a_res = _neutralize(alpha, X, invd)
    w = invd * a_res                       # D^{-1} * residual alpha

    if pos_cap is not None and n > 0:
        scale = np.sum(np.abs(w))
        if scale > 0:
            w = w / scale                  # normalize gross to 1 before capping
        w = np.clip(w, -pos_cap, pos_cap)
        w = _neutralize(w, X, np.ones(n))  # restore neutrality after clipping
    return w


def enforce_pos_cap(w: np.ndarray, beta: np.ndarray, beta_neutral: bool,
                    pos_cap: float | None) -> np.ndarray:
    """Clip |w| to `pos_cap` and restore neutrality, applied to the FINAL
    (vol-targeted) weights. The cap inside `optimize_day` is a PRE-scale shape
    limit; vol targeting then rescales the book, so without this a name can end
    up above pos_cap. This makes pos_cap the true max |weight| per name on the
    traded book. When the book is diversified enough that no name reaches the cap
    (the usual case for a large universe) this is a no-op."""
    if pos_cap is None:
        return w
    n = len(w)
    X = np.column_stack([np.ones(n), beta]) if beta_neutral else np.ones((n, 1))
    return _neutralize(np.clip(w, -pos_cap, pos_cap), X, np.ones(n))


def vol_target_scale(w: np.ndarray, var: np.ndarray, target_ann_vol: float,
                     gross_cap: float = 3.0) -> np.ndarray:
    """Scale weights so the sleeve's ex-ante annualized idiosyncratic vol equals
    `target_ann_vol`, subject to a gross-leverage cap."""
    daily_var = float(np.sum((w ** 2) * var))
    ann_vol = np.sqrt(max(daily_var, 1e-18) * _ANN)
    if ann_vol <= 0:
        return w
    k = target_ann_vol / ann_vol
    w = w * k
    gross = np.sum(np.abs(w))
    if gross > gross_cap:
        w = w * (gross_cap / gross)
    return w


# --------------------------------------------------------------------------- #
# Rolling risk inputs (self-contained: no dependency on cube betas)            #
# --------------------------------------------------------------------------- #
def rolling_beta_var(stock_ret: pd.DataFrame, spy_ret: pd.Series,
                     beta_window: int = 63, vol_window: int = 63):
    """Trailing market beta and idiosyncratic daily variance per name (point-in-
    time: uses only past returns). idio var = var(stock) - beta^2 * var(spy)."""
    var_spy = spy_ret.rolling(beta_window).var()
    cov = stock_ret.rolling(beta_window).cov(spy_ret)
    beta = cov.div(var_spy, axis=0)
    tot_var = stock_ret.rolling(vol_window).var()
    idio_var = (tot_var - beta.pow(2).mul(var_spy, axis=0)).clip(lower=1e-8)
    return beta, idio_var


# --------------------------------------------------------------------------- #
# Backtest engine with the optimizer + turnover-aware stepping                 #
# --------------------------------------------------------------------------- #
def simulate_portfolio_opt(
    signal: pd.DataFrame,            # date x ticker cross-sectional score (z or rank)
    stock_ret: pd.DataFrame,
    spy_ret: pd.Series,
    starting_capital: float = 1_000_000,
    market_weight: float = 0.5,      # deliberate SPY beta sleeve
    target_ann_vol: float = 0.08,    # alpha-sleeve ex-ante vol target
    beta_neutral: bool = True,
    pos_cap: float = 0.03,
    gross_cap: float = 3.0,
    step: float = 0.35,              # partial trade toward target (turnover control)
    no_trade_band: float = 0.0,      # skip per-name trades smaller than this
    beta_window: int = 63,
    vol_window: int = 63,
    fee_bps: float = 1.0,
    spread_bps: float = 5.0,
    rebalance_freq: int = 1,
) -> pd.DataFrame:
    cost_rate = (fee_bps + spread_bps) / 1e4
    beta_df, var_df = rolling_beta_var(stock_ret, spy_ret, beta_window, vol_window)

    dates = sorted(d for d in signal.index
                   if d in stock_ret.index and d in spy_ret.index
                   and d in beta_df.index)
    tickers = list(stock_ret.columns)
    prev_w = pd.Series(0.0, index=tickers + ["SPY"])
    V = spy_V = starting_capital
    rows = []

    target_alpha = pd.Series(0.0, index=tickers)
    for i in range(len(dates) - 1):
        t, t1 = dates[i], dates[i + 1]

        if i % rebalance_freq == 0:
            s = signal.loc[t].dropna()
            common = [tk for tk in s.index
                      if tk in beta_df.columns
                      and np.isfinite(beta_df.loc[t, tk]) and np.isfinite(var_df.loc[t, tk])]
            if len(common) >= 10:
                a = s[common].to_numpy(float)
                a = (a - a.mean()) / (a.std() if a.std() > 0 else 1.0)   # centered z
                b = beta_df.loc[t, common].to_numpy(float)
                v = var_df.loc[t, common].to_numpy(float)
                w_star = optimize_day(a, b, v, beta_neutral, pos_cap)
                w_star = vol_target_scale(w_star, v, target_ann_vol, gross_cap)
                # enforce pos_cap on the FINAL weights (vol targeting rescales the
                # pre-scale cap applied inside optimize_day)
                w_star = enforce_pos_cap(w_star, b, beta_neutral, pos_cap)
                target_alpha = pd.Series(0.0, index=tickers)
                target_alpha[common] = w_star

        # partial step toward target (Garleanu-Pedersen), then no-trade band
        aim = prev_w[tickers] + step * (target_alpha - prev_w[tickers])
        if no_trade_band > 0:
            delta = aim - prev_w[tickers]
            aim = prev_w[tickers] + delta.where(delta.abs() >= no_trade_band, 0.0)

        w = pd.Series(0.0, index=tickers + ["SPY"])
        w[tickers] = aim.values
        w["SPY"] = market_weight

        turnover = (w - prev_w).abs().sum()
        cost = turnover * cost_rate
        r_stocks = stock_ret.loc[t1, tickers].fillna(0.0)
        r_spy = spy_ret.loc[t1] if np.isfinite(spy_ret.loc[t1]) else 0.0
        # split the P&L into its two sleeves so each param's effect is observable
        alpha_ret = float((w[tickers] * r_stocks).sum())
        mkt_ret = float(w["SPY"] * r_spy)
        gross = alpha_ret + mkt_ret
        net = gross - cost
        V *= (1.0 + net)
        spy_V *= (1.0 + r_spy)
        rows.append({"date": t1, "gross_ret": gross, "cost": cost, "net_ret": net,
                     "turnover": turnover, "portfolio_value": V, "spy_value": spy_V,
                     # sleeve diagnostics (make the construction params visible)
                     "alpha_ret": alpha_ret, "mkt_ret": mkt_ret,
                     "alpha_gross": float(w[tickers].abs().sum()),
                     "alpha_max_w": float(w[tickers].abs().max())})
        prev_w = w

    return pd.DataFrame(rows).set_index("date")