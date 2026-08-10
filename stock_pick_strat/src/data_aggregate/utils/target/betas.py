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

GICS SECTOR IS A BETA HERE, NOT ONLY A DEMEAN LATER
----------------------------------------------------
`per_stock_factors` carries each stock's OWN GICS sector basket (leave-one-out,
market-neutralized -- built by `factors.gics_sector_excess_returns`), so every
name gets a fitted `beta_sector` and `compute_epsilon` strips the sector exposure
the stock actually has. The cross-sectional demeaning in `targets.py` is kept as
well, and the two are complements rather than alternatives:

  * the BETA removes the stock's individual loading, which is genuinely dispersed
    (sd 0.52, 1%-99% range -0.61 .. +2.02) -- the demeaning cannot, because
    subtracting a group mean implicitly assumes the same loading for everyone;
  * the DEMEANING guarantees an EXACT zero group mean on the day, with no
    estimation error and no staleness, and it also covers `industry_group`, which
    deliberately has no beta of its own (a ~20-name basket over 63 days produces a
    beta with sd 0.86 and a 1%-99% range of -1.7 .. +3.3; subtracting that noise
    put sector exposure BACK into the residual, sector R^2 0.017 -> 0.046).

Measured on the live panel (h=60, 491 names), adding the sector beta moves the
sector share of epsilon's cross-sectional variance 9.3% -> 1.7% and the residual/
market correlation 0.0077 -> 0.0039; the label's own sector R^2 is 0.001 either
way because the demeaning finishes the job. Dropping the demeaning and keeping
only the beta leaves the LABEL at 1.8% sector and 8.9% industry_group -- i.e. the
model would learn industry timing the optimizer then neutralizes away.

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


def _window_design(window_factors: np.ndarray, ridge_alpha: float, min_factor_obs: int,
                   market_col: int = -1, ridge_alpha_market: float | None = None):
    """Factorize ONE window's shared design -- reused for every stock in the panel.

    Returns `(ridge_hat, factor_idx, factor_std, standardized_factors)`:
      - `factor_idx`            columns of `window_factors` that are estimable
      - `factor_std`            their in-window standard deviations
      - `standardized_factors`  the window, centered/scaled to those columns
      - `ridge_hat`             the (len(factor_idx), n_obs) operator such that
                                `ridge_hat @ y_centered` is the ridge fit, in
                                STANDARDIZED units
    `None` when no factor is estimable in this window.

    `ridge_alpha_market` shrinks the market column ALONE (see `estimate_all_betas`); `None`
    keeps one alpha for every factor, which reproduces the single-alpha gram bit-for-bit.
    """

    n_obs = window_factors.shape[0]

    # only consider columns with enough observations: an all-NaN column would
    # otherwise warn ("Degrees of freedom <= 0") on every empty window.
    observed_enough = np.flatnonzero(
        np.count_nonzero(~np.isnan(window_factors), axis=0) >= min_factor_obs)
    if observed_enough.size == 0:
        return None

    col_std = np.nanstd(window_factors[:, observed_enough], axis=0)
    non_constant = col_std > 1e-12
    if not non_constant.any():
        return None

    # normalize each factor (Factor - mean / std)
    factor_idx = observed_enough[non_constant]
    factor_std = col_std[non_constant]
    standardized_factors = (window_factors[:, factor_idx]
                            - np.nanmean(window_factors[:, factor_idx], axis=0)) / factor_std
    standardized_factors = np.nan_to_num(standardized_factors)   # a missing day -> the window mean

    alphas = np.full(factor_idx.size, ridge_alpha)
    if ridge_alpha_market is not None:
        alphas[factor_idx == market_col] = ridge_alpha_market
    gram = (standardized_factors.T @ standardized_factors
            + n_obs * np.diag(alphas))
    ridge_hat = np.linalg.solve(gram, standardized_factors.T)
    return ridge_hat, factor_idx, factor_std, standardized_factors


def _standardize_sector_window(window_sector: np.ndarray):
    """Standardize the per-stock sector regressor within one window.

    Returns `(sector_z, sector_std, usable)`: `usable` flags the stocks where the
    sector column is fully present and non-constant over the window -- everyone
    else gets `beta_sector = 0.0` instead of a fitted value.
    """
    present = np.isfinite(window_sector).all(axis=0)
    std = np.where(present, window_sector.std(axis=0), 0.0)
    usable = std > 1e-12
    safe_std = np.where(usable, std, 1.0)
    sector_z = np.nan_to_num((window_sector - np.nanmean(window_sector, axis=0)) / safe_std)
    return sector_z, safe_std, usable


def _fold_in_sector(ridge_hat, standardized_factors, shared_beta, y, sector_z,
                    ridge_alpha: float, n_win: int):
    """Add the per-stock sector regressor to an already-factorized shared design.

    Each stock's GICS-sector basket is a DIFFERENT column, so the shared design no
    longer covers the whole regression -- but it covers all of it except that one
    column, so the augmented ridge solves in closed form off the SAME
    factorization via the Schur complement (bordered system). That keeps the
    solve one matmul per window for the ENTIRE cross-section instead of one solve
    per (stock, window): with the sector column the universe costs ~45s rather than
    the ~10 min a per-stock loop would.

    `y` is the (already prior-shifted, centered) return block `shared_beta` was fit
    on. Returns `(shared_beta_adjusted, sector_beta)`, both in standardized units.
    """
    lam = ridge_alpha * n_win
    cross = standardized_factors.T @ sector_z    # shared-factor / sector cross-product
    hat_sector = ridge_hat @ sector_z            # shared ridge-hat applied to the sector column
    denom = (sector_z ** 2).sum(0) - (cross * hat_sector).sum(0) + lam
    numer = (sector_z * y).sum(0) - (cross * shared_beta).sum(0)
    sector_beta = numer / denom
    shared_beta = shared_beta - hat_sector * sector_beta
    return shared_beta, sector_beta


def _fit_full_window(shared_design, window_factors, window_returns, window_sector,
                     market_col: int, market_prior: float, ridge_alpha: float,
                     n_win: int, n_factors: int) -> np.ndarray:
    """One ridge solve for every stock whose window is FULLY observed (the fast path)."""

    ridge_hat, factor_idx, factor_std, standardized_factors = shared_design
    has_market = market_col >= 0 and market_col in factor_idx

    y = window_returns
    if has_market:
        market_return = np.nan_to_num(window_factors[:, market_col])
        y = y - market_prior * market_return[:, None]      # shrink market toward its prior
    y = y - y.mean(0)                                      # center, like the standardized factors

    shared_beta = ridge_hat @ y                             # standardized shared betas
    n_out = n_factors + (1 if window_sector is not None else 0)
    block = np.zeros((n_out, window_returns.shape[1]))

    if window_sector is not None:
        sector_z, sector_std, sector_ok = _standardize_sector_window(window_sector)
        shared_beta, sector_beta = _fold_in_sector(
            ridge_hat, standardized_factors, shared_beta, y, sector_z, ridge_alpha, n_win)
        block[n_factors] = np.where(sector_ok, sector_beta / sector_std, 0.0)

    block[factor_idx] = shared_beta / factor_std[:, None]   # standardized -> raw units
    if has_market:
        block[market_col] += market_prior                   # add the prior back
    return block.T


def _fit_ragged_stock(window_factors, stock_return, observed, sector_column,
                      market_col: int, market_prior: float, ridge_alpha: float,
                      min_factor_frac: float, n_factors: int,
                      ridge_alpha_market: float | None = None) -> np.ndarray:
    """Exact one-off ridge for a stock whose window is only partly observed (a
    recent listing) -- rare enough that a per-stock solve is cheap. The sector
    column, when present, is simply appended to this stock's own design."""
    rows = np.flatnonzero(observed)
    design_cols = window_factors[rows]
    if sector_column is not None:
        design_cols = np.column_stack([design_cols, sector_column[rows]])

    n_out = n_factors + (1 if sector_column is not None else 0)
    # the sector column is appended LAST, so `market_col` still indexes the market factor
    design = _window_design(design_cols, ridge_alpha,
                            max(2, int(round(min_factor_frac * rows.size))),
                            market_col, ridge_alpha_market)
    if design is None:
        return np.full(n_out, np.nan)

    ridge_hat, factor_idx, factor_std, _ = design
    y = stock_return[rows]
    has_market = market_col >= 0 and market_col in factor_idx
    if has_market:
        y = y - market_prior * np.nan_to_num(window_factors[rows, market_col])

    row = np.zeros(n_out)
    row[factor_idx] = (ridge_hat @ (y - y.mean())) / factor_std
    if has_market:
        row[market_col] += market_prior
    return row


def _assemble_output(fitted: np.ndarray, dates, tickers, beta_cols: list[str],
                     ffill_limit: int | None, simple_market_beta) -> dict[str, pd.DataFrame]:
    """Turn the raw (date, ticker, beta) array into one DataFrame per ticker."""
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
        if simple_market_beta is not None:
            df["beta_market_simple"] = simple_market_beta[ticker]
        out[ticker] = df
    return out


def estimate_all_betas(
    stock_returns: pd.DataFrame,      # date x ticker (stocks only)
    factor_panel: pd.DataFrame,       # date x factor (market + style + commodity + macro)
    per_stock_factors: pd.DataFrame | None = None,
    window: int = 126,
    min_obs: int = 80,
    # The TUNED values live in configs/build_cube.yml, which is what the pipeline reads. These
    # signature defaults stay NEUTRAL on purpose: a non-None `ridge_alpha_market` default makes
    # `ridge_alpha` inert for the market column, so every test and notebook caller that varies
    # `ridge_alpha` alone would silently measure nothing (it did -- three cases in
    # tests/data_aggregate/test_betas.py returned the same beta for alpha 0.0, 0.08 and 0.25).
    ridge_alpha: float = 0.08,
    ridge_alpha_market: float | None = None,
    step: int = 1,
    market_prior: float = 1.0,
    min_factor_frac: float = 0.5,
    ffill_limit: int | None = 21,
    filter_factors: bool = True,
) -> dict[str, pd.DataFrame]:
    """Rolling multi-factor betas for every stock -> `{ticker: DataFrame}`.

    PANEL-VECTORIZED: every stock regresses on the SAME factors, so each window's
    design matrix is factorized once (`_window_design`) and applied to the whole
    cross-section in one matmul. Only a stock whose window is not fully observed
    (a recent listing, whose first windows hold fewer than `window` returns) takes
    the exact per-stock path -- correctness, not an approximation.

    per_stock_factors : date x ticker frame of the ONE regressor that differs per
                       stock -- each stock's own GICS sector excess return. Emitted
                       as `beta_sector`, solved off the same shared factorization
                       (`_fold_in_sector`), so the cross-section still costs one
                       solve per window. `None` -> no sector beta is fitted.
    window / min_obs : trailing window, and the observations needed to emit a beta.
    ridge_alpha      : shrinkage RATIO; lambda = ridge_alpha * N (see module doc).
    ridge_alpha_market : the market column's OWN ratio; `None` -> it shares `ridge_alpha`.
                       Two knobs because the two priors are not comparable: market shrinks
                       toward 1.0 and every other factor toward 0.0, and the amount of
                       shrinkage each NEEDS is set by how well its beta forecasts the window
                       it hedges. Measured on the live panel at window=63, the cross-sectional
                       slope of the realized in-window beta on the ex-ante one is 0.43 for
                       market but 0.03 for `d_vix` -- i.e. the macro loadings are almost pure
                       noise and must be shrunk an order of magnitude harder. Under-shrinking
                       makes the hedge OVER-shoot in proportion to a name's distance from the
                       mean, which is a factor bet the label then hands to the model.
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
    filter_factors   : drop non-daily-moving factors (stale monthly/weekly macro)
                       ONCE, up front, so every stock sees the same clean set.
    """
    if filter_factors:
        factor_panel, dropped = filter_daily_factors(factor_panel)
        if dropped:
            logger.warning("excluded stale factors from regression: %s", dropped)

    factor_names = list(factor_panel.columns)
    if not factor_names or stock_returns.empty:
        raise RuntimeError("betas need a non-empty factor panel and return frame")

    dates = stock_returns.index
    tickers = list(stock_returns.columns)
    n_dates, n_stocks, n_factors = len(dates), len(tickers), len(factor_names)
    market_col = factor_names.index("market") if "market" in factor_names else -1
    has_sector = per_stock_factors is not None

    factor_values = factor_panel.reindex(dates).to_numpy(float)
    stock_return_values = stock_returns.to_numpy(float)
    is_observed = np.isfinite(stock_return_values)
    sector_values = (per_stock_factors.reindex(index=dates, columns=tickers).to_numpy(float)
                     if has_sector else None)

    n_beta_cols = n_factors + (1 if has_sector else 0)
    # (date, ticker, beta). LOCAL and freed before returning; the long-form frame
    # the caller builds next is the larger allocation.
    fitted = np.full((n_dates, n_stocks, n_beta_cols), np.nan)

    for t in range(min_obs - 1, n_dates, step):
        lo = max(0, t - window + 1)
        n_win = t - lo + 1
        window_factors = factor_values[lo:t + 1]

        design = _window_design(window_factors, ridge_alpha,
                                max(2, int(round(min_factor_frac * n_win))),
                                market_col, ridge_alpha_market)
        if design is None:
            continue

        window_observed = is_observed[lo:t + 1]
        n_valid = window_observed.sum(0)
        estimable = n_valid >= min_obs
        if not estimable.any():
            continue

        window_returns = stock_return_values[lo:t + 1]
        window_sector = sector_values[lo:t + 1] if has_sector else None

        # steady state: fully observed windows, ONE ridge solve for the whole panel
        full = np.flatnonzero(estimable & (n_valid == n_win))
        if full.size:
            fitted[t, full] = _fit_full_window(
                design, window_factors, window_returns[:, full],
                window_sector[:, full] if has_sector else None,
                market_col, market_prior, ridge_alpha, n_win, n_factors)

        # ragged: a stock whose window is only partly observed (a recent listing)
        for stock_idx in np.flatnonzero(estimable & (n_valid != n_win)):
            fitted[t, stock_idx] = _fit_ragged_stock(
                window_factors, window_returns[:, stock_idx], window_observed[:, stock_idx],
                window_sector[:, stock_idx] if has_sector else None,
                market_col, market_prior, ridge_alpha, min_factor_frac, n_factors,
                ridge_alpha_market)

    beta_cols = [f"beta_{c}" for c in factor_names] + (["beta_sector"] if has_sector else [])
    simple_market_beta = _univariate_market_beta(stock_returns, factor_panel, window, min_obs)
    out = _assemble_output(fitted, dates, tickers, beta_cols, ffill_limit, simple_market_beta)
    del fitted

    logger.info("betas: %s/%s tickers x %s factors (window=%s step=%s ridge_alpha=%s "
                "ridge_alpha_market=%s market_prior=%s)", len(out), n_stocks, n_factors,
                window, step, ridge_alpha, ridge_alpha_market, market_prior)
    return out


def _univariate_market_beta(stock_returns: pd.DataFrame, factor_panel: pd.DataFrame,
                            window: int, min_obs: int) -> pd.DataFrame | None:
    """Plain rolling cov/var market beta, for every stock at once.

    NOT the same object as the fitted `beta_market` (see the module docstring):
    this is the TOTAL market exposure the L/S optimizer neutralizes on, while
    `beta_market` is the partial loading that makes epsilon orthogonal to the
    whole factor space.
    """
    if "market" not in factor_panel.columns:
        return None
    m = factor_panel["market"].reindex(stock_returns.index)
    var = m.rolling(window, min_periods=min_obs).var()
    cov = stock_returns.rolling(window, min_periods=min_obs).cov(m)
    return cov.div(var, axis=0)


def estimate_betas_for_stock(
    y: pd.Series,                 # stock daily returns
    shared: pd.DataFrame,         # market + style + commodity + macro (already filtered)
    sector: pd.Series | None = None,   # this stock's GICS sector excess return
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
    per_stock_factors = sector.rename(name).to_frame()

    return estimate_all_betas(y.rename(name).to_frame(), shared, per_stock_factors,
                              window=window, min_obs=min_obs,
                              ridge_alpha=ridge_alpha, step=step,
                              market_prior=market_prior, **kwargs)[name]
