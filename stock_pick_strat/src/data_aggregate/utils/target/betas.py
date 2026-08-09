"""
betas.py  (src/data_aggregate/utils/target/betas.py)
-----------------------------------------------------------------------------
Rolling multi-factor betas of each stock on market + style + macro/commodity
factors. These betas are what `targets.compute_epsilon` subtracts to turn a
forward return into an IDIOSYNCRATIC one, so every choice here is judged on one
question: how much common-factor exposure SURVIVES in epsilon?

Measured on the live panel (491 tickers, 2011-2026, h=60; the number is the
correlation of the residual with the forward market return -- 0 is neutral):

    raw forward return (no neutralization)              +0.426
    shrink every beta toward 0, lambda fixed at 5.0     +0.0755
    THIS module                                         +0.0112

FOUR THINGS THIS MODULE DOES THAT A TEXTBOOK ROLLING RIDGE DOES NOT
-------------------------------------------------------------------
1. THE MARKET BETA SHRINKS TOWARD 1.0, NOT 0.0.  Ridge is a Gaussian prior
   centred on the shrinkage target. Equity market beta is centred on 1.0, so
   shrinking it toward 0 is not regularization, it is a BIAS: the fitted beta
   comes back systematically too low and the difference stays inside epsilon as
   unhedged market. That is the single biggest term above (+0.0755 -> +0.0112,
   a 6.7x cut). Implemented as the generalized ridge
       min_b ||y - Xb||^2 + lambda ||b - b0||^2,   b0 = (1, 0, 0, ... 0)
   which is algebraically the plain ridge on  y~ = y - 1.0 * market  with the
   1.0 added back afterwards -- exactly the substitution b = b0 + d. Style,
   macro, commodity and FX betas keep the 0.0 prior, which IS their natural
   centre. A per-name prior (each stock's own trailing 252d beta, with and
   without Vasicek shrinkage) was tried and is NOT better: it helps the
   low-beta cohort by ~0.02 and gives that back on the aggregate. Flat 1.0 wins.

2. THE PENALTY SCALES WITH THE SAMPLE.  X'X grows like N, so a FIXED lambda
   means the shrinkage RATIO 1/(1 + lambda/N) drifts with the window: at the old
   lambda=5.0 a stock was regularized 58% harder on its first 40 observations
   than in steady state at 63. `ridge_alpha` IS that ratio -- lambda =
   ridge_alpha * N -- and because the regressors are standardized (Z'Z has
   diagonal N) an orthogonal factor is shrunk by exactly 1/(1 + ridge_alpha) at
   every window length. The 0.08 default is the old 5.0/63, so steady-state
   shrinkage is unchanged and only the warm-up is fixed. A sweep over
   0.02 -> 0.60 is flat between 0.08 and 0.15 and clearly worse outside it.

3. NO STAIRCASE: betas are re-fit EVERY day.  Refitting every 5th day and
   forward-filling left 75% of days bit-identical and put all the movement on
   one day in five -- a step function, both as a cube feature and as any
   downstream risk input. The fix is not to smooth the staircase (an EWMA over
   the steps costs ~7 days of extra lag ON TOP of the 5 days of staleness, and
   measurably hurts: corr(eps, market) 0.0112 -> 0.0141) but to remove it:
   `step=1`. That is affordable because the design matrix is SHARED by every
   stock, so each window is factorized ONCE and applied to the whole universe as
   a single matmul instead of being re-derived per ticker. The whole universe at
   step=1 takes ~11s against ~150s for the per-ticker loop at step=5.
   `smooth_halflife` stays available for experiments but defaults OFF.

4. A ONE-DAY HOLE IN A FACTOR NO LONGER DELETES THAT FACTOR'S BETA.  Dropping
   any column with a single NaN in the window silently zeroed the FX beta on
   10.4% of all windows (12 missing days in USD/EUR), gold on 4.1% and oil on
   2.4% -- and, worse, moved every OTHER coefficient in those windows, because
   the regression had lost a regressor. Missing days are now imputed to the
   window mean (0 in standardized space); a factor is dropped only when it is
   genuinely absent (< `min_factor_frac` of the window) or constant.

TIMING CONTRACT (unchanged -- and it is NOT a look-ahead)
---------------------------------------------------------
beta_t is fitted on the window ENDING at t. `targets.compute_epsilon` applies it
to `forward_return(close, h) = close[t+h]/close[t] - 1`, i.e. to the returns of
days t+1..t+h. The estimation window and the label window DO NOT OVERLAP, so
there is no in-sample fitting bias in the label and no extra lag is required
(`tests/data_aggregate/test_betas.py::test_betas_have_no_lookahead` pins this,
`::test_label_window_does_not_overlap_estimation_window` pins the non-overlap).
The one place a lag WOULD be needed is a live trading rule keyed off beta_t; the
L/S optimizer does not read this table, it re-derives its own trailing beta in
`strategies/utils/strategies_opt.py::rolling_beta_var`.

`beta_market` vs `beta_market_simple`
-------------------------------------
Different objects; they correlate only ~0.51 on the live panel. `beta_market` is
a MULTIVARIATE (partial) loading -- market exposure holding the other factors
fixed -- and the panel contains series that are themselves largely market
(`d_vix` correlates -0.81 with market, `resvol` -0.58), so the two split the
market between them. `beta_market_simple` is the plain univariate cov/var and is
the interpretable TOTAL exposure; it is what the L/S optimizer neutralizes on.
Making epsilon orthogonal to the whole factor space needs the multivariate
vector; reporting and hedging want the univariate one. Both are emitted -- do
not substitute one for the other.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.data_aggregate.utils.target.factors import filter_daily_factors

logger = logging.getLogger(__name__)

# the market column `assemble_factor_panel` writes, and the univariate companion
# column this module adds alongside the fitted loadings.
MARKET_FACTOR = "market"
BETA_MARKET_SIMPLE = "beta_market_simple"


def _window_design(Xw: np.ndarray, ridge_alpha: float, min_factor_obs: int):
    """Factorize ONE window's design matrix -- shared by every stock in the panel.

    Returns `(P, idx, sd)`: `idx` the estimable factor columns, `sd` their
    in-window standard deviations, and `P` the (len(idx), N) operator such that
    `P @ y_centered` is the ridge solution in STANDARDIZED units. `None` when no
    factor is estimable in this window.
    """
    n_win = Xw.shape[0]
    # nanstd only on columns that HAVE enough observations: an all-NaN column
    # would otherwise warn ("Degrees of freedom <= 0") on every empty window.
    enough = np.flatnonzero(np.count_nonzero(~np.isnan(Xw), axis=0) >= min_factor_obs)
    if enough.size == 0:
        return None
    sd = np.nanstd(Xw[:, enough], axis=0)
    keep = sd > 1e-12
    if not keep.any():
        return None

    idx = enough[keep]
    sd_ok = sd[keep]
    Z = (Xw[:, idx] - np.nanmean(Xw[:, idx], axis=0)) / sd_ok
    Z = np.nan_to_num(Z)                      # a missing day -> the window mean
    A = Z.T @ Z + ridge_alpha * n_win * np.eye(idx.size)
    return np.linalg.solve(A, Z.T), idx, sd_ok


def estimate_all_betas(
    stock_returns: pd.DataFrame,      # date x ticker (stocks only)
    factor_panel: pd.DataFrame,       # date x factor (market + style + commodity + macro)
    window: int = 63,
    min_obs: int = 40,
    ridge_alpha: float = 0.08,
    step: int = 1,
    market_prior: float = 1.0,
    min_factor_frac: float = 0.5,
    ffill_limit: int | None = 21,
    smooth_halflife: int = 0,
    filter_factors: bool = True,
) -> dict[str, pd.DataFrame]:
    """Rolling multi-factor betas for every stock -> `{ticker: DataFrame}`.

    PANEL-VECTORIZED: every stock regresses on the SAME factors, so each window's
    design matrix is factorized once (`_window_design`) and applied to the whole
    cross-section in one matmul. Only a stock whose window is not fully observed
    (a recent listing, whose first windows hold fewer than `window` returns) takes
    the exact per-stock path -- correctness, not an approximation.

    window / min_obs : trailing window, and the observations needed to emit a beta.
    ridge_alpha      : shrinkage RATIO; lambda = ridge_alpha * N (see module doc).
    step             : re-fit every `step` days. 1 = daily, i.e. no staircase.
    market_prior     : what `beta_market` shrinks toward (1.0). Every other beta
                       shrinks toward 0.0.
    min_factor_frac  : a factor needs this fraction of the window observed to be
                       estimable; below it its beta is 0.0 ("simply not
                       neutralized here"), the same convention `compute_epsilon`
                       uses for a missing forward factor -- NaN would propagate
                       and delete the whole target row.
    ffill_limit      : carry a beta over at most this many days with no estimable
                       window; past that it is a stale lie and stays NaN.
    smooth_halflife  : optional EWMA over the fitted betas. OFF by default -- once
                       `step` is 1 it only adds lag, and lag costs neutrality.
    filter_factors   : drop non-daily-moving factors (stale monthly/weekly macro)
                       ONCE, up front, so every stock sees the same clean set.
    """
    if filter_factors:
        factor_panel, dropped = filter_daily_factors(factor_panel)
        if dropped:
            logger.warning("excluded stale factors from regression: %s", dropped)

    cols = list(factor_panel.columns)
    if not cols or stock_returns.empty:
        raise RuntimeError("betas need a non-empty factor panel and return frame")

    dates = stock_returns.index
    tickers = list(stock_returns.columns)
    n_dates, n_stocks, n_fac = len(dates), len(tickers), len(cols)
    mkt = cols.index(MARKET_FACTOR) if MARKET_FACTOR in cols else -1

    X = factor_panel.reindex(dates).to_numpy(float)
    Y = stock_returns.to_numpy(float)
    seen_all = np.isfinite(Y)

    # (date, ticker, factor). LOCAL and freed before returning; the long-form
    # frame the caller builds next is the larger allocation.
    fitted = np.full((n_dates, n_stocks, n_fac), np.nan)

    for t in range(min_obs - 1, n_dates, step):
        lo = max(0, t - window + 1)
        Xw = X[lo:t + 1]
        n_win = t - lo + 1
        design = _window_design(Xw, ridge_alpha,
                                max(2, int(round(min_factor_frac * n_win))))
        if design is None:
            continue
        P, idx, sd_ok = design
        market_ok = mkt >= 0 and mkt in idx
        xm = np.nan_to_num(Xw[:, mkt]) if market_ok else None

        Yw = Y[lo:t + 1]
        seen = seen_all[lo:t + 1]
        n_valid = seen.sum(0)
        estimable = n_valid >= min_obs
        if not estimable.any():
            continue

        # -- steady state: fully observed windows, ONE solve for the whole panel
        full = np.flatnonzero(estimable & (n_valid == n_win))
        if full.size:
            Yb = Yw[:, full]
            if market_ok:
                Yb = Yb - market_prior * xm[:, None]      # shrink market toward the prior
            Yb = Yb - Yb.mean(0)
            block = np.zeros((n_fac, full.size))
            block[idx] = (P @ Yb) / sd_ok[:, None]        # standardized -> raw units
            if market_ok:
                block[mkt] += market_prior                # add the prior back
            fitted[t, full] = block.T

        # -- ragged: a stock whose window is only partly observed (recent listing)
        for j in np.flatnonzero(estimable & (n_valid != n_win)):
            rows = np.flatnonzero(seen[:, j])
            sub = _window_design(Xw[rows], ridge_alpha,
                                 max(2, int(round(min_factor_frac * rows.size))))
            if sub is None:
                continue
            P_j, idx_j, sd_j = sub
            yj = Yw[rows, j]
            market_ok_j = mkt >= 0 and mkt in idx_j
            if market_ok_j:
                yj = yj - market_prior * np.nan_to_num(Xw[rows, mkt])
            row = np.zeros(n_fac)
            row[idx_j] = (P_j @ (yj - yj.mean())) / sd_j
            if market_ok_j:
                row[mkt] += market_prior
            fitted[t, j] = row

    beta_cols = [f"beta_{c}" for c in cols]
    simple = _univariate_market_beta(stock_returns, factor_panel, window, min_obs)

    out: dict[str, pd.DataFrame] = {}
    for j, ticker in enumerate(tickers):
        block = fitted[:, j, :]
        if not np.isfinite(block).any():
            # too little history for a single window -> OMIT the ticker rather than
            # write an all-NaN block. `compute_epsilon` already skips a ticker that
            # is absent from the dict, and the part table stays free of empty rows.
            continue
        df = pd.DataFrame(block, index=dates, columns=beta_cols)
        df = df.ffill(limit=ffill_limit) if ffill_limit else df.ffill()
        if smooth_halflife:
            df = df.ewm(halflife=smooth_halflife, min_periods=1).mean()
        if simple is not None:
            df[BETA_MARKET_SIMPLE] = simple[ticker]
        out[ticker] = df
    del fitted

    logger.info("betas: %s/%s tickers x %s factors (window=%s step=%s ridge_alpha=%s "
                "market_prior=%s)", len(out), n_stocks, n_fac, window, step,
                ridge_alpha, market_prior)
    return out


def _univariate_market_beta(stock_returns: pd.DataFrame, factor_panel: pd.DataFrame,
                            window: int, min_obs: int) -> pd.DataFrame | None:
    """Plain rolling cov/var market beta, for every stock at once.

    NOT the same object as the fitted `beta_market` (see the module docstring):
    this is the TOTAL market exposure the L/S optimizer neutralizes on, while
    `beta_market` is the partial loading that makes epsilon orthogonal to the
    whole factor space.
    """
    if MARKET_FACTOR not in factor_panel.columns:
        return None
    m = factor_panel[MARKET_FACTOR].reindex(stock_returns.index)
    var = m.rolling(window, min_periods=min_obs).var()
    cov = stock_returns.rolling(window, min_periods=min_obs).cov(m)
    return cov.div(var, axis=0)


def estimate_betas_for_stock(
    y: pd.Series,                 # stock daily returns
    shared: pd.DataFrame,         # market + style + commodity + macro (already filtered)
    window: int = 63,
    min_obs: int = 40,
    ridge_alpha: float = 0.08,
    step: int = 1,
    market_prior: float = 1.0,
    **kwargs,
) -> pd.DataFrame:
    """Single-stock view of `estimate_all_betas` (tests, notebooks, one-off checks).

    A one-column panel takes the SAME code path as the full universe, so this can
    never drift from what the pipeline computes.
    """
    name = y.name if y.name is not None else "STOCK"
    kwargs.setdefault("filter_factors", False)
    return estimate_all_betas(y.rename(name).to_frame(), shared,
                              window=window, min_obs=min_obs,
                              ridge_alpha=ridge_alpha, step=step,
                              market_prior=market_prior, **kwargs)[name]
