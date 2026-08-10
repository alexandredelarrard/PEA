"""
targets.py  (src/data_aggregate/utils/targets.py)  -- MULTI-FACTOR REWRITE
--------------------------------------------------------------------------
epsilon_i(t) = fwd_ret_i(t->t+h)
               - beta_market_i  * fwd_market(t->t+h)
               - beta_sector_i  * fwd_gics_sector_excess_i(t->t+h)
               - sum_style  beta_style_i  * fwd_style(t->t+h)
               - sum_macro  beta_macro_i  * fwd_macro_change(t->t+h)

i.e. strip market, the stock's own GICS sector tilt, AND every style + macro factor.

That subtraction is only HALF the job, and the missing half was measured: the residual stays
predictable from the very loadings that were subtracted (a signal built from nothing but a
name's market beta earned rank-IC +0.073, t +10.3, against the old label). So each label is
then TRANSFORMED (rank / zscore) and PROJECTED cross-sectionally orthogonal to those loadings
plus momentum plus LOG MARKET CAP plus GICS industry_group, jointly, and transformed again to
restore its scale -- `_neutral_label`. Projecting the transformed label rather than epsilon is
deliberate: see that function for why the ordering carries most of the effect.

Log market cap is in that design because a LOADING does not span a CHARACTERISTIC: `beta_size`
is a loading on the size basket's RETURN and explains only R^2 0.26 of `-log(mcap)` across
names, which left `-log_mcap` earning free rank-IC +0.0380 (t +7.4) at h=60. Subtracting
`beta_size * fwd_size` removes co-movement with the small-cap basket; it does not remove the
premium for BEING small. See `_neutralizing_design`.

The sector term appears TWICE on purpose: `beta_sector_i` removes the stock's own loading
(dispersed across names, and a group demean cannot capture that), while the industry indicator
block guarantees an exact zero group mean on the day and covers industry_group, which has no
beta. See `betas.py` for the measurements.

Forward factor returns over the SAME t->t+h window:
  * market / style : compounded factor return over the window (one series each,
                     applied via each stock's loading)
  * sector         : per-stock GICS basket excess return, compounded (same shape)
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
    trailing_vol,
)
from src.data_aggregate.utils.common.xs import (
    XS_CLIP_CHARACTERISTIC, XS_CLIP_LABEL, xs_group_dummies, xs_project_out, xs_rank_pct, xs_z,
)


def compute_epsilon(
    close: pd.DataFrame,             # stocks only
    betas: dict,                     # {ticker: DataFrame beta_<factor>...}
    factor_panel: pd.DataFrame,      # market + style + macro daily
    macro_cols: list,                # which factor_panel columns are macro CHANGES
    horizon: int,
    sector_excess: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Multi-factor forward residual for every stock at every date, one horizon.

    `sector_excess` = (date x ticker) DAILY frame of each stock's OWN GICS sector basket
    (`factors.gics_sector_excess_returns`) -- the one regressor that differs per stock --
    applied through the `beta_sector` fitted in `betas.py`. SAME shape `estimate_all_betas`
    takes, so one object travels the whole path. Forward-compounded over the SAME t->t+h
    window as the shared return factors, so the beta and the thing it multiplies are the
    same object.

    This strips each stock's OWN sector loading. The GICS indicator block in the label
    projection that runs afterwards is NOT redundant with it: this removes the stock's
    individual sector beta, that one guarantees an exact zero group mean on the day AND covers
    industry_group, which has no beta of its own. Measured on the live panel, the sector beta
    cuts the sector share of epsilon's cross-sectional variance from 9.3% to 1.7%; the
    indicator block then takes the LABEL to 0.0%.
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

    # the sector basket is a daily RETURN frame -> compounded, like the style factors.
    # OPTIONAL for the same reason as `betas.estimate_all_betas`'s sector regressor: the
    # fingerprint harnesses build a label with no sector term at all.
    fwd_sector = forward_compound(sector_excess, horizon) if sector_excess is not None else None

    eps = pd.DataFrame(index=close.index, columns=close.columns, dtype="float64")
    for ticker in close.columns:
        if ticker not in betas:
            continue
        
        b = betas[ticker].reindex(close.index)
        resid = fwd_stock[ticker].copy()
        # the stock's OWN sector basket -> strip its individual sector loading
        if (fwd_sector is not None and "beta_sector" in b.columns
                and ticker in fwd_sector.columns):
            resid = resid - b["beta_sector"] * fwd_sector[ticker].reindex(close.index).fillna(0.0)

        # Subtract every shared factor's forward return * its loading. A SHARED factor is the
        # same series for every stock, so ONE missing value (e.g. a data gap in oil / gold /
        # USD-EUR, whose calendars differ from equities) would otherwise NaN the residual for
        # the WHOLE cross-section and drop the entire date. Fill ONLY the factor's forward
        # return with 0 -> that factor is simply not neutralized there, while the target stays
        # defined. We fill the FACTOR, not the product, so a missing BETA still propagates NaN
        # (early-history dates stay undefined); and the stock's own forward return is never
        # filled, so the genuine tail (no future price yet) remains correctly NaN.
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


def _beta_frame(betas: dict, column: str, like: pd.DataFrame) -> pd.DataFrame | None:
    """One beta column pivoted to a (date x ticker) frame on `like`'s grid, None if unfitted."""
    fitted = {t: b[column] for t, b in betas.items() if column in b.columns}
    if not fitted:
        return None
    return pd.DataFrame(fitted).reindex(index=like.index, columns=like.columns)


def fitted_beta_columns(betas: dict) -> list[str]:
    """Every beta column present anywhere in `betas` -- the projection design, DERIVED.

    Hand-listing which loadings to neutralize was arbitrary and unsafe: a factor added to the
    panel is hedged by `compute_epsilon` automatically, but its loading kept leaking into the
    LABEL until somebody remembered to edit the list. Measured on the live panel, the loadings
    that had been left out leaked 0.0000-0.0095 against a 0.0026-0.0088 band for the ones that
    were in -- i.e. the exclusions bought nothing, so there is no reason to curate them.
    """
    return sorted({c for b in betas.values() for c in b.columns})


def _neutralizing_design(
    close: pd.DataFrame,
    betas: dict,
    sector_groups: dict[str, dict[str, str]] | None,
    with_momentum: bool,
    market_cap: pd.DataFrame | None,
) -> tuple[list[pd.DataFrame], pd.DataFrame | None]:
    """The regressors the LABEL is made orthogonal to: every fitted factor loading, the 12-1
    momentum characteristic, log market cap, and the GICS industry_group indicators.

    The loadings ARE the betas this same step just fitted, so nothing new has to be loaded.
    Each is z-scored ONCE here rather than once per (horizon, label); a name with no fitted
    beta gets 0 -- the day's average exposure -- so it is simply not neutralized on that
    factor instead of dropping out of the fit.

    `market_cap` (from `pit.daily_market_cap`) adds the SIZE CHARACTERISTIC. It is not
    redundant with `beta_size`: that is a loading on the size BASKET's return and explains only
    R^2 0.26 of `-log(mcap)` across names, so 59% of the size ordering was unspanned by the
    whole design and `-log_mcap` earned free rank-IC +0.0380 (t +7.4) at h=60 against the
    label. Adding this takes that to +0.0051. Presence of the frame is the switch.
    """

    frames = [_beta_frame(betas, c, close) for c in fitted_beta_columns(betas)]

    # neutralize momentum at stock level
    if with_momentum:
        frames.append(momentum_characteristic(close))

    if market_cap is not None:
        frames.append(np.log(market_cap))     # sign is irrelevant to a projection

    # zero_sd_to_nan: on a day with no dispersion the loading is undefined, so this yields NaN
    # (-> 0 below) rather than the fabricated +/-clip an unguarded z would produce.
    # reindex BEFORE fillna, and in that order: `daily_market_cap` returns a COLUMN SUBSET (a
    # ticker with no filing history is absent, not NaN), while `xs_project_out` runs its own
    # `reindex_like` AFTER this fill -- so an unaligned exposure would reach `np.linalg.lstsq`
    # carrying NaN and raise. This reindex is what makes the later one harmless.
    exposures = [xs_z(f, clip=XS_CLIP_CHARACTERISTIC, zero_sd_to_nan=True)
                 .reindex_like(close).fillna(0.0)
                 for f in frames if f is not None]

    # industry_group is NESTED in sector, so industry indicators span both levels; fall back to
    # sector when only that level is supplied.
    groups = ((sector_groups or {}).get("industry_group")
              or (sector_groups or {}).get("sector"))
    dummies = xs_group_dummies(groups, close.columns) if groups else None
    return exposures, dummies


def _neutral_label(eps: pd.DataFrame, label: str, min_names: int,
                   exposures: list[pd.DataFrame],
                   dummies: pd.DataFrame | None) -> pd.DataFrame:
    """Transform -> project the exposures out -> transform again.

    The projection sits on the TRANSFORMED label, not on epsilon, and that ordering is the
    whole point: the rank/z transform is NON-LINEAR and the residual is right-skewed for
    volatile names (cross-sectional skew 0.07 -> 0.62 across vol quintiles), so a name's
    MEDIAN residual is negative while its mean is zero. Neutralizing epsilon therefore removes
    only half the tilt (free IC 0.073 -> 0.036) where neutralizing the label removes it
    (-> 0.016).

    The second `_apply_label` restores each label's own scale, which the projection destroys:
    rank back to a [0,1] percentile, zscore back to mean 0 / sd 1 / +-3.
    """
    y = _apply_label(eps, label, min_names)
    y = xs_project_out(y, exposures, dummies)
    return _apply_label(y, label, min_names)


def vol_standardize_epsilon(eps: pd.DataFrame, stock_ret: pd.DataFrame, horizon: int,
                            window: int = 63) -> pd.DataFrame:
    """Epsilon per unit of the name's OWN trailing risk, i.e. an information ratio.

    Without it the label's magnitude is a volatility artefact -- sd(epsilon) correlates +0.90
    with trailing vol and the top-vol decile carries 3.5x the dispersion of the bottom -- which
    an RMSE loss on the raw or z-scored label reads as signal.
    """
    sigma = trailing_vol(stock_ret, window, min_periods=window // 2) * np.sqrt(horizon)
    return eps / sigma.where(sigma > 0).reindex_like(eps)


def build_targets_multi(
    close: pd.DataFrame,
    betas: dict,
    factor_panel: pd.DataFrame,
    macro_cols: list,
    horizons=(5, 10, 20, 60),
    labels=("rank", "zscore", "epsilon"),
    min_names: int = 20,
    neutralize_momentum: bool = True,
    sector_groups: dict[str, dict[str, str]] | None = None,
    sector_excess: pd.DataFrame | None = None,
    stock_ret: pd.DataFrame | None = None,
    vol_standardize: bool = False,
    market_cap: pd.DataFrame | None = None,
) -> dict:
    """Compute the (expensive) factor-neutral residual ONCE per horizon and emit
    SEVERAL target versions from it, so the cube can store e.g. both the rank and the
    z-score target.

    Subtracting `beta * factor` leaves the label still PREDICTABLE from the exposures, for two
    reasons: the hedge is a noisy forecast of the window it hedges, and
    a beta-hedged high-beta name genuinely under-earns in a rising market (the low-risk
    anomaly). Measured on the live panel, a signal built from nothing but a name's market beta
    earned a rank-IC of +0.073 (t +10.3) against the old label -- free IC a model would happily
    learn, and which the L/S optimizer then neutralizes back out. So every label is finally
    PROJECTED orthogonal to EVERY fitted loading + momentum + log market cap + GICS industry,
    jointly, by `_neutral_label`. The loading list is derived, not configured -- see
    `fitted_beta_columns`.

    `neutralize_momentum` adds the 12-1 momentum characteristic to that same design.

    `market_cap` adds log market cap, i.e. the size CHARACTERISTIC as opposed to the size
    loading. Unlike the beta and sector tilts, a size tilt is neutralized NOWHERE downstream
    (`strategy_ls.yml` enforces `beta_neutral` and `sector_neutral` only), so it would flow
    straight into the live book. See `_neutralizing_design`.

    `sector_groups` = {"sector": {ticker: gics_sector}, "industry_group": {...}} -> the group
    indicator block, so sector / industry membership cannot predict the target. `None` means no
    group neutralization at all.

    `vol_standardize` divides epsilon by the name's trailing risk first (needs `stock_ret`);
    off by default because it does NOT reduce the exposure tilt once the projection is in
    place -- it only homogenises magnitude, which matters for an RMSE loss on the zscore /
    epsilon labels but not for the rank one.

    Returns {horizon: {label: DataFrame(date x ticker)}}.

    (A single-label `build_targets` twin used to sit alongside this, duplicating the
    whole epsilon -> neutralize -> label loop for one label. Nothing in `src/` called
    it; pass `labels=("rank",)` instead.)
    """

    # horizon-independent, so it is built ONCE for all (horizon, label) pairs
    exposures, dummies = _neutralizing_design(close, betas, sector_groups,
                                              neutralize_momentum, market_cap)
    out: dict[int, dict[str, pd.DataFrame]] = {}
    for h in horizons:
        eps = compute_epsilon(close, betas, factor_panel, macro_cols, h,
                              sector_excess=sector_excess)
        if vol_standardize:
            eps = vol_standardize_epsilon(eps, stock_ret, h)
        out[h] = {lab: _neutral_label(eps, lab, min_names, exposures, dummies)
                  for lab in labels}
    return out
