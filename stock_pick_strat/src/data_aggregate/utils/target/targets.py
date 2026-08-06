"""
targets.py  (src/data_aggregate/utils/targets.py)  -- MULTI-FACTOR REWRITE
--------------------------------------------------------------------------
epsilon_i(t) = fwd_ret_i(t->t+h)
               - beta_market_i  * fwd_market(t->t+h)
               - sum_style  beta_style_i  * fwd_style(t->t+h)
               - sum_macro  beta_macro_i  * fwd_macro_change(t->t+h)

i.e. strip market AND every style + macro factor, leaving the residual to be
neutralized to the GICS sector + industry_group (`cross_sectional_group_neutralize`).
Then rank cross-sectionally per day.

Forward factor returns over the SAME t->t+h window:
  * market / style : compounded factor return over the window (one series each,
                     applied via each stock's loading)
  * macro          : cumulative CHANGE over the window (level[t+h]-level[t]),
                     matching how the macro beta was estimated on daily changes
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from src.data_aggregate.utils.common.prices import (
    forward_compound,
    forward_cumchange,
    forward_return,
    momentum_characteristic,
)
from src.data_aggregate.utils.common.xs import XS_CLIP_CHARACTERISTIC, XS_CLIP_LABEL, xs_rank_pct, xs_z


def compute_epsilon(
    close: pd.DataFrame,             # stocks only
    betas: dict,                     # {ticker: DataFrame beta_<factor>...}
    factor_panel: pd.DataFrame,      # market + style + macro daily
    macro_cols: list,                # which factor_panel columns are macro CHANGES
    horizon: int,
) -> pd.DataFrame:
    """
    Multi-factor forward residual for every stock at every date, one horizon. Sector /
    industry_group neutralization happens separately, cross-sectionally, on the residual
    this returns (see `cross_sectional_group_neutralize`), not inside this function.
    """
    fwd_stock = forward_return(close, horizon)

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
        if ticker not in betas:
            continue
        b = betas[ticker].reindex(close.index)
        resid = fwd_stock[ticker].copy()
        # Subtract every shared factor's forward return * its loading. A SHARED
        # factor is the same series for every stock, so a single missing value
        # (e.g. a data gap in oil / gold / USD-EUR, whose calendar differs from
        # equities) would otherwise NaN the residual for the WHOLE cross-section
        # and drop the entire date. Fill ONLY the factor's forward return with 0
        # on those dates -> that factor is simply not neutralized there (a tiny
        # approximation, since commodity/FX betas are small for most equities),
        # while the target stays defined. We fill the factor, NOT the product, so
        # a missing BETA still propagates NaN (early-history dates keep their old
        # behaviour); and the stock's OWN forward return is never filled, so the
        # genuine tail (no future price yet) is still correctly undefined.
        for c in fwd_shared.columns:
            bc = f"beta_{c}"
            if bc in b.columns:
                resid = resid - b[bc] * fwd_shared[c].fillna(0.0)
        eps[ticker] = resid
    return eps


def cross_sectional_rank(eps: pd.DataFrame, min_names: int = 20) -> pd.DataFrame:
    valid = eps.notna().sum(axis=1) >= min_names
    ranked = xs_rank_pct(eps)
    ranked[~valid] = np.nan
    return ranked


def cross_sectional_zscore(eps: pd.DataFrame, min_names: int = 20,
                           clip: float | None = XS_CLIP_LABEL) -> pd.DataFrame:
    """Cross-sectional z-score per day. Residual returns are fat-tailed, so the
    z-score is winsorized to +-`clip` (default 3): without it a handful of
    extreme names dominate an RMSE loss and make the target hard/unstable to fit.
    """
    valid = eps.notna().sum(axis=1) >= min_names
    z = xs_z(eps, clip=clip)
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
    # zero_sd_to_nan: on a day with no factor dispersion the slope is undefined, so this
    # call site yields NaN rather than the fabricated +/-clip the unguarded sites produce.
    z = xs_z(factor.reindex_like(values), clip=XS_CLIP_CHARACTERISTIC, zero_sd_to_nan=True)

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


def cross_sectional_group_neutralize(
    values: pd.DataFrame,
    group_map: dict[str, str],
    unknown: str = "__UNK__",
) -> pd.DataFrame:
    """Per-day, subtract each cross-sectional GROUP mean (a GICS sector or industry)
    so the residual has ~zero mean within every group on every day -> the target
    carries NO sector/industry tilt and group membership can no longer PREDICT it.

    `group_map` is {ticker: group_label}. Names with no group are left untouched
    (they keep whatever neutralization ran before). NaN-safe: a day's group mean
    skips missing names. Compose calls (sector, then the nested industry_group) to
    neutralize both levels; industry is nested in sector, so the industry pass — run
    last — is what makes the target neutral to BOTH."""
    if values.empty or not group_map:
        return values
    labels = pd.Series({c: (group_map.get(c) or unknown) for c in values.columns})
    out = values.copy()
    for g, idx in labels.groupby(labels).groups.items():
        if g == unknown:
            continue
        cols = list(idx)
        out[cols] = values[cols].sub(values[cols].mean(axis=1), axis=0)   # per-date group demean
    return out


def _neutralize_sector_industry(
    eps: pd.DataFrame,
    sector_groups: dict[str, dict[str, str]] | None,
) -> pd.DataFrame:
    """Sequentially demean `eps` within GICS sector then industry_group (each level's
    {ticker: label} under `sector_groups[level]`). No-op when `sector_groups` is None
    (no sector/industry neutralization is applied at all)."""
    for level in ("sector", "industry_group"):
        gm = (sector_groups or {}).get(level)
        if gm:
            eps = cross_sectional_group_neutralize(eps, gm)
    return eps


def build_targets_multi(
    close: pd.DataFrame,
    betas: dict,
    factor_panel: pd.DataFrame,
    macro_cols: list,
    horizons=(5, 10, 20, 60),
    labels=("rank", "zscore"),
    min_names: int = 20,
    neutralize_momentum: bool = True,
    sector_groups: dict[str, dict[str, str]] | None = None,
) -> dict:
    """Compute the (expensive) factor-neutral residual ONCE per horizon and emit
    SEVERAL target versions from it, so the cube can store e.g. both the rank and the
    z-score target and the modelling step can pick which one to train on without a
    cube rebuild.

    `neutralize_momentum`: after computing the multi-factor residual epsilon,
    cross-sectionally orthogonalize it against the 12-1 momentum characteristic each
    day (makes the target orthogonal to momentum by construction).

    `sector_groups` = {"sector": {ticker: gics_sector}, "industry_group": {...}}. When
    given, the residual is neutralized to the ACTUAL GICS sector + industry_group
    (per-day within-group demeaning, applied LAST) so sector / industry membership can
    no longer predict the target (else they'd dominate the model, a sign the target was
    not sector-neutral). `None` means no group neutralization is applied at all.

    Returns {horizon: {label: DataFrame(date x ticker)}}.

    (A single-label `build_targets` twin used to sit alongside this, duplicating the
    whole epsilon -> neutralize -> label loop for one label. Nothing in `src/` called
    it; pass `labels=("rank",)` instead.)
    """
    mom_char = momentum_characteristic(close) if neutralize_momentum else None
    out: dict[int, dict[str, pd.DataFrame]] = {}
    for h in horizons:
        eps = compute_epsilon(close, betas, factor_panel, macro_cols, h)
        if neutralize_momentum:
            eps = cross_sectional_neutralize(eps, mom_char)
        eps = _neutralize_sector_industry(eps, sector_groups)      # last -> neutral to both
        out[h] = {lab: _apply_label(eps, lab, min_names) for lab in labels}
    return out
