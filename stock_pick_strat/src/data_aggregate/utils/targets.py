"""
targets.py  (src/data_aggregate/utils/targets.py)  -- MULTI-FACTOR REWRITE
--------------------------------------------------------------------------
epsilon_i(t) = fwd_ret_i(t->t+h)
               - beta_market_i  * fwd_market(t->t+h)
               - beta_sector_i  * fwd_sector_i(t->t+h)
               - sum_style  beta_style_i  * fwd_style(t->t+h)
               - sum_macro  beta_macro_i  * fwd_macro_change(t->t+h)

i.e. strip market, sector, AND every style + macro factor, leaving the pure
firm-vs-factor-matched-peers move. Then rank cross-sectionally per day.

Forward factor returns over the SAME t->t+h window:
  * market / style : compounded factor return over the window (one series each,
                     applied via each stock's loading)
  * sector         : per-stock peer-basket forward return (as before)
  * macro          : cumulative CHANGE over the window (level[t+h]-level[t]),
                     matching how the macro beta was estimated on daily changes
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from src.data_aggregate.utils.factors import momentum_characteristic


def forward_return(prices: pd.DataFrame, horizon: int) -> pd.DataFrame:
    return prices.shift(-horizon) / prices - 1.0


def forward_compound(daily: pd.Series | pd.DataFrame, horizon: int):
    """Compounded forward return over t+1..t+h from a daily-return series/frame."""
    log1p = np.log1p(daily)
    fwd = np.expm1(log1p[::-1].rolling(horizon, min_periods=horizon).sum()[::-1].shift(-1))
    return fwd


def forward_cumchange(level_change: pd.Series | pd.DataFrame, horizon: int):
    """Cumulative forward change over t+1..t+h from a daily-CHANGE series/frame."""
    fwd = level_change[::-1].rolling(horizon, min_periods=horizon).sum()[::-1].shift(-1)
    return fwd


def forward_sector_return(stock_returns, peer_dict, horizon):
    fwd_cum = forward_compound(stock_returns, horizon)
    sector_fwd = pd.DataFrame(index=stock_returns.index,
                              columns=stock_returns.columns, dtype="float64")
    for ticker, peers in peer_dict.items():
        if not peers or ticker not in fwd_cum.columns:
            continue
        cols = [p for p in peers if p in fwd_cum.columns]
        if not cols:
            continue
        # NaN-tolerant weighted mean (see compute_sector_returns): a raw matrix
        # product would drop the whole date if any single peer is missing.
        w = pd.Series({p: float(peers[p]) for p in cols}, dtype="float64")
        w = w / w.sum()
        weighted = fwd_cum[cols].mul(w, axis=1).sum(axis=1, min_count=1)
        denom = fwd_cum[cols].notna().mul(w, axis=1).sum(axis=1)
        sector_fwd[ticker] = weighted.div(denom.where(denom > 0))
    return sector_fwd


def compute_epsilon(
    close: pd.DataFrame,             # stocks only
    stock_returns: pd.DataFrame,     # stocks only, daily
    peer_dict: dict,
    betas: dict,                     # {ticker: DataFrame beta_<factor>...}
    factor_panel: pd.DataFrame,      # market + style + macro daily
    macro_cols: list,                # which factor_panel columns are macro CHANGES
    horizon: int,
) -> pd.DataFrame:
    """
    Multi-factor forward residual for every stock at every date, one horizon.
    """
    fwd_stock = forward_return(close, horizon)
    fwd_sector = forward_sector_return(stock_returns, peer_dict, horizon)

    # Precompute forward returns of every SHARED factor (same for all stocks).
    style_market_cols = [c for c in factor_panel.columns if c not in macro_cols]
    fwd_shared = {}
    for c in style_market_cols:                       # returns -> compound
        fwd_shared[c] = forward_compound(factor_panel[c], horizon)
    for c in macro_cols:                              # changes -> cumulative sum
        fwd_shared[c] = forward_cumchange(factor_panel[c], horizon)
    fwd_shared = pd.DataFrame(fwd_shared)

    eps = pd.DataFrame(index=close.index, columns=close.columns, dtype="float64")
    for ticker in close.columns:
        if ticker not in betas or ticker not in fwd_sector.columns:
            continue
        b = betas[ticker].reindex(close.index)
        resid = fwd_stock[ticker].copy()
        # subtract every shared factor's forward return * its loading
        for c in fwd_shared.columns:
            bc = f"beta_{c}"
            if bc in b.columns:
                resid = resid - b[bc] * fwd_shared[c]
        # subtract sector
        if "beta_sector" in b.columns:
            resid = resid - b["beta_sector"] * fwd_sector[ticker]
        eps[ticker] = resid
    return eps


def cross_sectional_rank(eps: pd.DataFrame, min_names: int = 20) -> pd.DataFrame:
    valid = eps.notna().sum(axis=1) >= min_names
    ranked = eps.rank(axis=1, pct=True, method="average")
    ranked[~valid] = np.nan
    return ranked


def cross_sectional_zscore(eps: pd.DataFrame, min_names: int = 20) -> pd.DataFrame:
    valid = eps.notna().sum(axis=1) >= min_names
    z = eps.sub(eps.mean(axis=1), axis=0).div(eps.std(axis=1), axis=0)
    z[~valid] = np.nan
    return z


def _apply_label(eps: pd.DataFrame, label: str, min_names: int) -> pd.DataFrame:
    """Turn the raw factor-neutral residual into a modelling target.
      rank    -> cross-sectional percentile in [0,1] (robust, scale-free)
      zscore  -> cross-sectional z-score (mean 0, std 1 per day; keeps magnitude)
      epsilon -> the raw residual itself
    """
    if label == "rank":
        return cross_sectional_rank(eps, min_names)
    if label == "zscore":
        return cross_sectional_zscore(eps, min_names)
    if label == "epsilon":
        return eps
    raise ValueError("label must be 'rank', 'zscore', or 'epsilon'")


def cross_sectional_neutralize(
    values: pd.DataFrame,
    factor: pd.DataFrame,
) -> pd.DataFrame:
    """Per-day residual of `values` regressed cross-sectionally on `factor`
    (single regressor, with intercept). Makes each row (date) orthogonal to
    `factor`, so a target built on the residual no longer tilts toward that
    characteristic.

    The factor is z-scored and clipped per day so a handful of extreme names
    (e.g. a stock up 500%) cannot dominate the slope. NaN-safe: a name is
    neutralized only where BOTH the value and the factor are present; names
    whose factor is missing (e.g. < 252 days of history) are left untouched so
    they still enter the cross-sectional ranking.
    """
    f = factor.reindex_like(values)
    mu = f.mean(axis=1)
    sd = f.std(axis=1).replace(0.0, np.nan)
    z = f.sub(mu, axis=0).div(sd, axis=0).clip(-4.0, 4.0)

    mask = values.notna() & z.notna()
    v = values.where(mask)
    x = z.where(mask)

    vc = v.sub(v.mean(axis=1), axis=0)
    xc = x.sub(x.mean(axis=1), axis=0)
    denom = (xc * xc).sum(axis=1).replace(0.0, np.nan)
    beta = ((vc * xc).sum(axis=1) / denom).fillna(0.0)
    resid = vc.sub(xc.mul(beta, axis=0))

    # Keep un-neutralizable names (factor NaN) as their demeaned value so they
    # still rank; ranking is invariant to the per-day demeaning.
    values_demeaned = values.sub(values.mean(axis=1), axis=0)
    return resid.where(mask, values_demeaned)


def build_targets(
    close: pd.DataFrame,
    stock_returns: pd.DataFrame,
    peer_dict: dict,
    betas: dict,
    factor_panel: pd.DataFrame,
    macro_cols: list,
    horizons=(5, 10, 20, 60),
    label: str = "rank",
    min_names: int = 20,
    neutralize_momentum: bool = True,
) -> dict:
    """
    NOTE signature change vs the old (close, market_close, ...): market is now
    inside factor_panel, and we pass macro_cols so forward macro factors are
    treated as cumulative CHANGES not compounded returns. Update
    step_build_cube.build_targets accordingly.

    `neutralize_momentum`: after computing the multi-factor residual epsilon,
    cross-sectionally orthogonalize it against the 12-1 momentum characteristic
    each day. Subtracting beta*factor only strips the market-wide momentum move
    and leaves each stock's idiosyncratic momentum, so without this the target
    still tilts toward past winners (corr with beta_momentum ~ +0.6). This makes
    the target orthogonal to momentum by construction.
    """
    mom_char = momentum_characteristic(close) if neutralize_momentum else None

    out = {}
    for h in horizons:
        eps = compute_epsilon(close, stock_returns, peer_dict, betas,
                              factor_panel, macro_cols, h)
        if neutralize_momentum:
            eps = cross_sectional_neutralize(eps, mom_char)
        out[h] = _apply_label(eps, label, min_names)
    return out


def build_targets_multi(
    close: pd.DataFrame,
    stock_returns: pd.DataFrame,
    peer_dict: dict,
    betas: dict,
    factor_panel: pd.DataFrame,
    macro_cols: list,
    horizons=(5, 10, 20, 60),
    labels=("rank", "zscore"),
    min_names: int = 20,
    neutralize_momentum: bool = True,
) -> dict:
    """Like `build_targets`, but computes the (expensive) factor-neutral residual
    ONCE per horizon and emits SEVERAL target versions from it, so the cube can
    store e.g. both the rank and the z-score target and the modelling step can
    pick which one to train on without a cube rebuild.

    Returns {horizon: {label: DataFrame(date x ticker)}}.
    """
    mom_char = momentum_characteristic(close) if neutralize_momentum else None
    out: dict[int, dict[str, pd.DataFrame]] = {}
    for h in horizons:
        eps = compute_epsilon(close, stock_returns, peer_dict, betas,
                              factor_panel, macro_cols, h)
        if neutralize_momentum:
            eps = cross_sectional_neutralize(eps, mom_char)
        out[h] = {lab: _apply_label(eps, lab, min_names) for lab in labels}
    return out
