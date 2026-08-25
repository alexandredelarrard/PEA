"""
betas.py  (src/data_aggregate/utils/target/betas.py)
-----------------------------------------------------------------------------
Rolling multi-factor betas of each stock on market + style + macro/commodity
factors. These betas are what `targets.compute_epsilon` subtracts to turn a
forward return into an IDIOSYNCRATIC one, so every choice here is judged on one
question: how much common-factor exposure SURVIVES in epsilon?

Measured on the live panel (491 tickers, 2011-2026, h=60) at the THEN-CURRENT
window=63 / ridge_alpha=0.08; the number is the correlation of the residual with the
forward market return -- 0 is neutral:

    raw forward return (no neutralization)              +0.426
    shrink every beta toward 0, lambda fixed at 5.0     +0.0755
    THIS module                                         +0.0112

Do NOT tune against that metric: it is a name-average, so it is mathematically
insensitive to a hedge that over-shoots on high-beta names and under-shoots on low-beta
ones. Item 2 below gives the metric that is not.

TWO KINDS OF REGRESSOR
----------------------
GLOBAL factors (`global_factors`) are one series each, IDENTICAL for every stock:
market, the style baskets, oil/gold/FX, the macro changes. Their design matrix is
shared, so it is factorized once per window (`_factorize_global_design`) and applied
to the whole cross-section in a single matmul.

The PER-STOCK regressor (`stock_sector_factor`) is a DIFFERENT column for every
stock -- each name's own GICS sector basket. It is folded into that same
factorization by an exact bordered solve (`_solve_with_sector_column`), so the
cross-section still costs one solve per window rather than one per (stock, window).

WHY SECTOR IS THE ONLY PER-STOCK BETA. A beta comes from a TIME-SERIES regression, so
its regressor must VARY DAY TO DAY. A characteristic does not: a stock's own
`log(market cap)`, its 12-1 momentum, its earnings yield are all near-flat over 126
days, and `_factorize_global_design` would discard such a column as constant
(`col_std > 1e-12`). That is precisely why a characteristic is first turned into a
factor RETURN -- a long-short basket, in `factors.characteristic_to_factor_return` --
before any beta is fitted, and why neutralizing the CHARACTERISTIC itself belongs in
the cross-sectional projection in `targets.py` and not here. A loading on the basket
is not a substitute: `beta_size` explains only R^2 0.26 of `-log(mcap)` across names.
The sector basket is the one per-stock quantity that is already a daily return series.

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
   every window length.
   HOW TO TUNE IT: against the CALIBRATION slope, not against the residual/market
   correlation above. Regress the REALIZED in-window beta on the EX-ANTE beta across
   the cross-section and drive that slope to 1.0 -- below 1.0 the hedge over-shoots
   (it hands the label a factor bet in proportion to a name's distance from the mean),
   above 1.0 it under-shoots. The shipped `ridge_alpha: 1.5` /
   `ridge_alpha_market: 0.48` land at 1.04 (market) and 1.14 (size) over 56
   non-overlapping 60-day windows; the old 0.08 at window=63 sat at 0.43, i.e. it
   over-hedged by more than half. The SIGNATURE defaults stay at those old neutral
   values deliberately -- see `estimate_all_betas`.

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

4. A ONE-DAY HOLE IN A FACTOR NO LONGER DELETES THAT FACTOR'S BETA.  Dropping
   any column with a single NaN in the window silently zeroed the FX beta on
   10.4% of all windows (12 missing days in USD/EUR), gold on 4.1% and oil on
   2.4% -- and, worse, moved every OTHER coefficient in those windows, because
   the regression had lost a regressor. Missing days are now imputed to the
   window mean (0 in standardized space); a factor is dropped only when it is
   genuinely absent (< `min_factor_frac` of the window) or constant.

GICS SECTOR IS A BETA HERE, NOT ONLY A DEMEAN LATER
----------------------------------------------------
`stock_sector_factor` carries each stock's OWN GICS sector basket (leave-one-out,
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

ONE MARKET BETA
---------------
`beta_market` is a MULTIVARIATE (partial) loading -- market exposure holding the
other factors fixed -- and that is the coefficient epsilon's decomposition
REQUIRES: the pieces only add up if every beta comes from the same joint fit. The
panel contains series that are themselves largely market (`d_vix` correlates -0.81
with market, `resvol` -0.58), so the partial loading is genuinely not the total
exposure.

A univariate cov/var twin was emitted beside it as `beta_market_simple` for a long
time and has been REMOVED, because nothing read it: no modelling allow-list mentions
`beta_*` (features come from explicit `columns:` lists), the L/S optimizer computes
its own 63-day beta (`strategies_opt.rolling_beta_var`, `strategy_ls.yml
beta_window`), and `compute_epsilon` never subtracted it -- there is no
`market_simple` factor column to multiply. Its ONLY effect was as a regressor in
`targets.py`'s label projection, where at the current window/alpha it correlated
0.945 with `beta_market` and carried the same leak (free IC 0.0088 vs 0.0084): a
duplicate of a column already in the design.
"""
from __future__ import annotations

import logging
from typing import NamedTuple

import numpy as np
import pandas as pd

from src.data_aggregate.utils.target.factors import filter_daily_factors

logger = logging.getLogger(__name__)


class GlobalDesign(NamedTuple):
    """ONE window's factorized GLOBAL design, shared by every stock in the panel.

    `ridge_operator @ y_centered` IS the ridge fit in STANDARDIZED units, so the whole
    cross-section is fitted by one matmul against the `(n_obs, n_stocks)` return block.
    """
    ridge_operator: np.ndarray        # (n_estimable_factors, n_obs)
    factor_idx: np.ndarray            # which `global_factors` columns are estimable here
    factor_std: np.ndarray            # their in-window standard deviations
    standardized_factors: np.ndarray  # the window, centered/scaled to those columns


def _factorize_global_design(window_factors: np.ndarray, ridge_alpha: float,
                             min_factor_obs: int, market_col_idx: int = -1,
                             ridge_alpha_market: float | None = None) -> GlobalDesign | None:
    """Factorize ONE window's GLOBAL design -- reused for every stock in the panel.

    `None` when no factor is estimable in this window. Note the `col_std > 1e-12` filter:
    a constant column is dropped, which is why a CHARACTERISTIC can never be a regressor
    here (see the module docstring's TWO KINDS OF REGRESSOR).

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
        alphas[factor_idx == market_col_idx] = ridge_alpha_market

    gram = (standardized_factors.T @ standardized_factors
            + n_obs * np.diag(alphas))
    ridge_operator = np.linalg.solve(gram, standardized_factors.T)
    return GlobalDesign(ridge_operator, factor_idx, factor_std, standardized_factors)


def _standardize_sector_over_window(window_sector: np.ndarray):
    """Standardize the per-stock sector regressor ALONG TIME, within one window.

    The axis matters: this divides by the std over the window's ~126 DAYS, per stock, to
    put the column in the same units as `_factorize_global_design`'s standardized factors
    so that one `ridge_alpha` means the same shrinkage for every regressor. It is NOT the
    cross-sectional z of `xs.py` and the two are not interchangeable -- `xs_z` standardizes
    across TICKERS on a date, and it winsorizes, which would bias a fitted coefficient.

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


def _solve_with_sector_column(ridge_operator, standardized_factors, global_beta, y, sector_z,
                              ridge_alpha: float, n_win: int):
    """Extend an already-factorized GLOBAL design with the one per-stock column.

    Each stock's GICS-sector basket is a DIFFERENT column, so the shared design no
    longer covers the whole regression -- but it covers all of it except that one
    column, so the augmented ridge solves in closed form off the SAME factorization
    via the Schur complement (bordered system).

    This is EXACT, not an approximation: the result is what a full per-stock solve of
    the augmented system would return, by the block-inverse identity. It is done this
    way purely for cost -- 15 of the 16 columns are shared, so the cross-section stays
    one solve per window instead of one per (stock, window): ~45s for the universe
    against the ~10 min a per-stock loop takes. Every line below is a column-wise
    vector op over the stocks, so `denom` is a length-n_stocks vector, not an inverse.

    `y` is the (already prior-shifted, centered) return block `global_beta` was fit
    on. Returns `(global_beta_adjusted, sector_beta)`, both in standardized units.
    """
    lam = ridge_alpha * n_win
    cross = standardized_factors.T @ sector_z    # global-factor / sector cross-product
    hat_sector = ridge_operator @ sector_z       # global operator applied to the sector column
    denom = (sector_z ** 2).sum(0) - (cross * hat_sector).sum(0) + lam
    numer = (sector_z * y).sum(0) - (cross * global_beta).sum(0)
    sector_beta = numer / denom
    global_beta = global_beta - hat_sector * sector_beta
    return global_beta, sector_beta


def _fit_complete_stocks(global_design: GlobalDesign, window_factors, window_returns,
                         window_sector, market_col_idx: int, market_prior: float,
                         ridge_alpha: float, n_win: int, n_factors: int) -> np.ndarray:
    """ONE ridge solve covering EVERY stock whose window is fully observed (the fast path).

    `window_sector is None` -> no sector beta is fitted at all (a caller that passed no
    per-stock regressor); the returned block is then `n_factors` rows, not `n_factors + 1`.
    """

    ridge_operator, factor_idx, factor_std, standardized_factors = global_design
    has_market = market_col_idx >= 0 and market_col_idx in factor_idx

    y = window_returns
    if has_market:
        market_return = np.nan_to_num(window_factors[:, market_col_idx])
        y = y - market_prior * market_return[:, None]      # shrink market toward its prior
    y = y - y.mean(0)                                      # center, like the standardized factors

    global_beta = ridge_operator @ y                        # standardized GLOBAL betas
    n_out = n_factors + (1 if window_sector is not None else 0)
    block = np.zeros((n_out, window_returns.shape[1]))

    if window_sector is not None:
        sector_z, sector_std, sector_ok = _standardize_sector_over_window(window_sector)
        global_beta, sector_beta = _solve_with_sector_column(
            ridge_operator, standardized_factors, global_beta, y, sector_z, ridge_alpha, n_win)
        block[n_factors] = np.where(sector_ok, sector_beta / sector_std, 0.0)

    block[factor_idx] = global_beta / factor_std[:, None]   # standardized -> raw units
    if has_market:
        block[market_col_idx] += market_prior               # add the prior back

    return block.T


def _fit_partial_stock(window_factors, stock_return, observed, sector_column,
                       market_col_idx: int, market_prior: float, ridge_alpha: float,
                       min_factor_frac: float, n_factors: int,
                       ridge_alpha_market: float | None = None) -> np.ndarray:
    """Exact one-off ridge for ONE stock whose window is only partly observed (a
    recent listing) -- rare enough that a per-stock solve is cheap, so here the sector
    column is simply appended to this stock's own design instead of being bordered in."""
    rows = np.flatnonzero(observed)
    design_cols = window_factors[rows]
    if sector_column is not None:
        design_cols = np.column_stack([design_cols, sector_column[rows]])

    n_out = n_factors + (1 if sector_column is not None else 0)
    # the sector column is appended LAST, so `market_col_idx` still indexes the market factor
    design = _factorize_global_design(design_cols, ridge_alpha,
                                      max(2, int(round(min_factor_frac * rows.size))),
                                      market_col_idx, ridge_alpha_market)
    if design is None:
        return np.full(n_out, np.nan)

    ridge_operator, factor_idx, factor_std, _ = design
    y = stock_return[rows]
    has_market = market_col_idx >= 0 and market_col_idx in factor_idx
    if has_market:
        y = y - market_prior * np.nan_to_num(window_factors[rows, market_col_idx])

    row = np.zeros(n_out)
    row[factor_idx] = (ridge_operator @ (y - y.mean())) / factor_std
    if has_market:
        row[market_col_idx] += market_prior
    return row


def _assemble_output(fitted: np.ndarray, dates, tickers, beta_cols: list[str],
                     ffill_limit: int | None) -> dict[str, pd.DataFrame]:
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
        out[ticker] = df.ffill(limit=ffill_limit) if ffill_limit else df.ffill()
    return out


def estimate_all_betas(
    stock_returns: pd.DataFrame,          # date x ticker (stocks only)
    global_factors: pd.DataFrame,         # date x factor, SAME for every stock
    stock_sector_factor: pd.DataFrame | None = None,   # date x ticker, DIFFERENT per stock
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

    PANEL-VECTORIZED: every stock regresses on the SAME global factors, so each window's
    design is factorized once (`_factorize_global_design`) and applied to the whole
    cross-section in one matmul (`_fit_complete_stocks`). Only a stock whose window is not
    fully observed (a recent listing, whose first windows hold fewer than `window` returns)
    takes the per-stock path (`_fit_partial_stock`) -- correctness, not an approximation.

    global_factors   : date x factor. ONE series per factor, identical for every stock:
                       market + style baskets + commodity/FX + macro changes.
    stock_sector_factor : date x ticker frame of the ONE regressor that DIFFERS per
                       stock -- each stock's own GICS sector excess return. Emitted
                       as `beta_sector`, solved off the same global factorization
                       (`_solve_with_sector_column`), so the cross-section still costs one
                       solve per window. `None` -> no sector beta is fitted at all, which
                       several callers (the fingerprint harnesses, `conftest`'s
                       `real_pipeline`) legitimately want.
                       Only a DAILY-VARYING quantity can go here -- see the module
                       docstring on why a characteristic cannot.
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
        global_factors, dropped = filter_daily_factors(global_factors)
        if dropped:
            logger.warning("excluded stale factors from regression: %s", dropped)

    factor_names = list(global_factors.columns)
    if not factor_names or stock_returns.empty:
        raise RuntimeError("betas need a non-empty factor panel and return frame")

    dates = stock_returns.index
    tickers = list(stock_returns.columns)
    n_dates, n_stocks, n_factors = len(dates), len(tickers), len(factor_names)
    market_col_idx = factor_names.index("market") if "market" in factor_names else -1
    # the sector regressor is OPTIONAL: the pipeline always passes it, but five callers
    # (both fingerprint harnesses, conftest's real_pipeline, two target tests) legitimately
    # fit betas on the global panel alone, so every sector-shaped object below is guarded.
    has_sector = stock_sector_factor is not None

    # transform to numpy
    factor_values = global_factors.reindex(dates).to_numpy(float)
    stock_return_values = stock_returns.to_numpy(float)
    is_observed = np.isfinite(stock_return_values)
    sector_values = (stock_sector_factor.reindex(index=dates, columns=tickers).to_numpy(float)
                     if has_sector else None)

    n_beta_cols = n_factors + (1 if has_sector else 0)
    # (date, ticker, beta). LOCAL and freed before returning; the long-form frame
    # the caller builds next is the larger allocation.
    fitted = np.full((n_dates, n_stocks, n_beta_cols), np.nan)

    for t in range(min_obs - 1, n_dates, step):
        lo = max(0, t - window + 1)
        n_win = t - lo + 1
        window_factors = factor_values[lo:t + 1]

        global_design = _factorize_global_design(window_factors, ridge_alpha,
                                                 max(2, int(round(min_factor_frac * n_win))),
                                                 market_col_idx, ridge_alpha_market)
        if global_design is None:
            continue

        window_observed = is_observed[lo:t + 1]
        n_valid = window_observed.sum(0)
        estimable = n_valid >= min_obs
        if not estimable.any():
            continue

        window_returns = stock_return_values[lo:t + 1]
        window_sector = sector_values[lo:t + 1] if has_sector else None

        # STEADY STATE: every stock whose window is fully observed, in one ridge solve
        complete = np.flatnonzero(estimable & (n_valid == n_win))
        if complete.size:
            fitted[t, complete] = _fit_complete_stocks(
                global_design, window_factors, window_returns[:, complete],
                window_sector[:, complete] if has_sector else None,
                market_col_idx, market_prior, ridge_alpha, n_win, n_factors)

        # PARTIAL: a stock whose window is only partly observed (a recent listing)
        for stock_idx in np.flatnonzero(estimable & (n_valid != n_win)):
            fitted[t, stock_idx] = _fit_partial_stock(
                window_factors, window_returns[:, stock_idx], window_observed[:, stock_idx],
                window_sector[:, stock_idx] if has_sector else None,
                market_col_idx, market_prior, ridge_alpha, min_factor_frac, n_factors,
                ridge_alpha_market)

    beta_cols = [f"beta_{c}" for c in factor_names] + (["beta_sector"] if has_sector else [])
    out = _assemble_output(fitted, dates, tickers, beta_cols, ffill_limit)
    del fitted

    logger.info("betas: %s/%s tickers x %s factors (window=%s step=%s ridge_alpha=%s "
                "ridge_alpha_market=%s market_prior=%s)", len(out), n_stocks, n_factors,
                window, step, ridge_alpha, ridge_alpha_market, market_prior)
    return out


def estimate_betas_for_stock(
    y: pd.Series,                      # stock daily returns
    global_factors: pd.DataFrame,      # market + style + commodity + macro (already filtered)
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
    sector_frame = sector.rename(name).to_frame() if sector is not None else None

    return estimate_all_betas(y.rename(name).to_frame(), global_factors, sector_frame,
                              window=window, min_obs=min_obs,
                              ridge_alpha=ridge_alpha, step=step,
                              market_prior=market_prior, **kwargs)[name]
