"""
xs.py  (src/data_aggregate/utils/common/xs.py)
---------------------------------------------
CROSS-SECTIONAL operations: every per-day, across-tickers transform in one place.

`clip` is a REQUIRED argument of `xs_z` with no default, so a call site always states its bound.
There are two bounds:

    3.0  `XS_CLIP_LABEL`  the modelling LABEL: a handful of extreme residuals must not dominate
                          the loss the model is fitted against.
    5.0  `Z_CLIP`         every z a FEATURE or REGRESSOR uses: the own-history z (`self_history_z`),
                          the peer z of `panel.peer_relative` and the factor/neutralization
                          characteristic.

Likewise the ZERO-DISPERSION policy is explicit rather than incidental. On a day where
every name shares one value, `sd == 0`, so an unguarded divide yields +/-inf which `clip`
then turns into a FABRICATED +/-clip; the two guarded call sites instead produce NaN.
`zero_sd_to_nan` selects between the two.

The GROUP-PRESENCE FLOOR (`MIN_GROUP_SIZE_FOR_NEUTRALIZATION`) is the same spirit applied to
`xs_project_out`'s indicator block: a named threshold instead of silently accepting a
degenerate fit. A one-hot column with a single present member fits that member's residual
EXACTLY, so the "neutralized" label is identically zero and ships as a valid-looking
`target_rank ~ 0.50`; two members split it and each keeps exactly half. That is not a
neutralization, it is a deletion, so a group too thin to support a demean is dropped from
the day's design rather than allowed to absorb the name whole.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

# the two clip bounds, named so a call site documents its intent
XS_CLIP_LABEL = 3.0  # modelling target (targets.py)
Z_CLIP = 5.0  # every feature or regressor z: own-history, peer, factor characteristic

#: How many PRESENT members a design CELL needs on a day before the group block may split the
MIN_GROUP_SIZE_FOR_NEUTRALIZATION = 5


def xs_rank_pct(df: pd.DataFrame) -> pd.DataFrame:
    """THE cross-sectional percentile rank in [0, 1], per date across tickers.

    `method="average"` is pandas' default and is stated explicitly because the four
    call sites this replaces were split between spelling it and omitting it."""
    return df.rank(axis=1, pct=True, method="average")


def zscore[Pandas: (pd.DataFrame, pd.Series)](x: Pandas, centre: Pandas, scale: Pandas, clip: float | None = None) -> Pandas:
    """THE z-score arithmetic every leg shares: `(x - centre) / scale`, optionally clipped to +/-`clip`.

    The caller supplies the centre (peer weighted mean, universe mean or own trailing mean) and the
    scale, already guarded for its own zero-dispersion policy; inf handling, input trimming and
    output winsorizing stay with the caller, in its own order."""
    z = x.sub(centre, axis=0).div(scale, axis=0)
    return z if clip is None else z.clip(-clip, clip)


def xs_z(df: pd.DataFrame, clip: float | None, *, zero_sd_to_nan: bool = False, eps: float = 1e-12) -> pd.DataFrame:
    """THE cross-sectional z-score, per date across tickers.

    `clip` is required (see the module docstring); pass `None` for unclipped.
    `zero_sd_to_nan=False` reproduces the UNGUARDED behaviour of `factors._xs_z` /
    `features.cross_sectional_standardize` / `targets.cross_sectional_zscore`
    (sd == 0 -> +/-inf -> clip -> +/-clip); `True` is the guarded behaviour a REGRESSOR needs
    (sd == 0 -> NaN), used by `targets._neutralizing_design`.

    ⚠ The peer std floor (`panel.peer_relative`'s `std_floor_ratio`) IS DELIBERATELY NOT APPLIED
    HERE: it is a fraction of THE DAY'S CROSS-SECTIONAL STD, and `sd` on this line IS that std,
    so the floor could never bind. It is only meaningful for a SUBGROUP dispersion (a peer
    basket) measured against a wider reference.

    Inputs are not trimmed: this function standardizes the MODELLING LABEL
    (`targets.cross_sectional_zscore`) as well as characteristics."""
    mu = df.mean(axis=1)
    sd = df.std(axis=1)
    if zero_sd_to_nan:
        sd = sd.replace(0.0, np.nan)
    else:
        # Guard against sd being zero or near-zero due to floating point precision
        sd = sd.mask(sd < eps, np.nan)
    return zscore(df, mu, sd, clip)


def xs_group_dummies(group_map: dict[str, str], tickers: pd.Index) -> pd.DataFrame:
    """Ticker x group one-hot membership, to be used as a regressor block in `xs_project_out`.

    Static (no date axis): GICS membership is keyed by ticker only. An unmapped name lands in a
    shared `__UNK__` column so every name sits in exactly one group and the block spans the
    constant -- which is what makes the projection's residual exactly zero-mean per group.

    No reference level is dropped, deliberately: the FULL one-hot block is what forces an exact
    zero group mean rather than a mean relative to some arbitrary base group. The price of that
    is a column per group whatever the group's size, which is why `xs_project_out` needs a
    per-day presence floor (`min_group_size`) -- the block is built once, statically, and cannot
    know how many of a group's members are present on any given day.
    """
    labels = pd.Series([group_map.get(t, "__UNK__") for t in tickers], index=tickers)
    return pd.get_dummies(labels, dtype=float)


def _floored_group_block(grp: np.ndarray, min_group_size: int) -> np.ndarray:
    """One day's one-hot group block with every column too thin to support a demean removed.

    ⚠ THE RULE IS ABOUT DESIGN CELLS, NOT COLUMNS, and the difference is the whole reason this
    is a function rather than one filter line. `_day_residual` centers both sides, so there is
    an IMPLICIT INTERCEPT: the one-hot block plus that intercept PARTITIONS the day's names, and
    dropping a group's column does not take its members out of the design -- it moves them into
    the partition's COMPLEMENT cell, which is then demeaned exactly like any other cell. Drop a
    solo group's column while every other group clears the floor and the complement is a cell of
    ONE, fitted exactly, residual identically 0.0 -- the original defect, unmoved. Measured on a
    synthetic solo panel: 1.1e-16 with the column dropped, which is zero.

    So the complement is checked too, and the smallest surviving group is folded into it until
    it clears the floor as well. Every cell in the returned design therefore holds at least
    `min_group_size` names, or the block is empty and no group demean happens that day at all
    (which is the correct answer when no partition of the day can be supported).
    """
    counts = grp.sum(axis=0)
    keep = counts >= min_group_size
    total = grp.shape[0]
    # the complement cell = every present name whose group column was dropped
    while keep.any() and 0 < total - counts[keep].sum() < min_group_size:
        smallest = np.flatnonzero(keep)[np.argmin(counts[keep])]
        keep[smallest] = False
    return grp[:, keep]


def _day_residual(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """OLS residual of ONE day's cross-section on `x`.

    Both sides are demeaned, so the intercept is implicit. `lstsq` (not the normal equations)
    because it returns the exact projection even when `x` is rank-deficient -- a GICS group
    with no name on that day is an all-zero column, and that is the normal case at the edges.
    """
    y_centered = y - y.mean()
    x_centered = x - x.mean(axis=0)
    coef, *_ = np.linalg.lstsq(x_centered, y_centered, rcond=None)
    return y_centered - x_centered @ coef


def xs_project_out(
    values: pd.DataFrame, exposures: list[pd.DataFrame], dummies: pd.DataFrame | None = None, *, min_group_size: int | None = None
) -> pd.DataFrame:
    """Per-day cross-sectional residual of `values` on `exposures` + `dummies`, JOINTLY.

    The multivariate sibling of a single-factor neutralization, and the difference is not
    cosmetic: the regressors are mutually correlated (market beta vs trailing vol is +0.6), so
    removing them one at a time leaves the result orthogonal to none of them.

    `exposures` are (date x ticker) frames, ALREADY standardized by the caller (once, rather
    than once per label); `dummies` is the static ticker x group block from `xs_group_dummies`.

    `min_group_size` is the per-day presence floor on every DESIGN CELL of the group block --
    see `_floored_group_block`, which is where the subtlety lives. It closes the defect the
    whole-day guard below cannot see: `present.sum() > design.shape[1]` asks whether the WHOLE
    fit is over-determined, so a day with 400 names and one solo group sails through it while
    that one name is still fitted exactly by its own indicator column, to 0.0.

    A name below the floor is not DROPPED, only un-grouped: it keeps every other regressor and
    is simply not forced to a group zero mean that day, so no new NaN is introduced.
    `None` (the default) reproduces the unfloored behaviour exactly, bit for bit.
    """
    stacked = np.stack([e.reindex_like(values).to_numpy(float) for e in exposures], axis=2) if exposures else np.zeros((*values.shape, 0))
    group_block = dummies.reindex(values.columns).to_numpy(float) if dummies is not None else np.zeros((values.shape[1], 0))
    y = values.to_numpy(float)
    out = np.full_like(y, np.nan)
    for i in range(len(y)):
        present = np.isfinite(y[i])
        grp = group_block[present]
        if min_group_size is not None and grp.shape[1]:
            grp = _floored_group_block(grp, min_group_size)
        design = np.column_stack([stacked[i][present], grp])
        if present.sum() > design.shape[1]:  # else the fit is exact and says nothing
            out[i, present] = _day_residual(y[i][present], design)
    return pd.DataFrame(out, index=values.index, columns=values.columns)


def xs_standardize(feat: pd.DataFrame, method: Literal["rank", "zscore"], clip: float = XS_CLIP_LABEL) -> pd.DataFrame:
    """Standardize one feature within each day (across stocks).
    'rank'   -> percentile in [0,1] (robust to outliers)
    'zscore' -> demean/divide by cross-sectional std, clipped at +/-`clip`
    """
    if method == "rank":
        return xs_rank_pct(feat)
    if method == "zscore":
        return xs_z(feat, clip=clip)
    raise ValueError("method must be 'rank' or 'zscore'")


#: The trailing window for a SELF-HISTORY z, and the minimum history before one is emitted.
#: 1260 trading days is ~5 calendar years; 252 is ~1 year. These set the fundamentals part's
#: warm-up (`parts.py` pins 1260 for exactly this reason), so changing them changes that too.
HIST_WINDOW = 1260
HIST_MIN_PERIODS = 252


def self_history_z(
    field_df: pd.DataFrame,
    window: int = HIST_WINDOW,
    min_periods: int = HIST_MIN_PERIODS,
    output_since: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Time-series z-score of each ticker versus its OWN trailing `window`:

        z(t) = (x(t) - trailing_mean(t)) / trailing_std(t)

    The value is clipped to +-`Z_CLIP`.

    The rolling window is trailing (right-edge = today), so it uses only current-and-past
    values -- strictly point-in-time, no look-ahead. On a valuation YIELD a high z means the
    firm is currently cheaper than its own historical norm (mean-reversion long side); on a
    governance LEVEL it means the firm is unusual BY ITS OWN STANDARDS, which is a different
    question from unusual versus its peers.

    ⚠ ITS PRECONDITION IS WITHIN-FIRM MOVEMENT, and that is not a formality. The denominator is
    the firm's own trailing std, so on a characteristic that is effectively fixed per firm the
    std is zero and the z is NaN (guarded by `std > 0`) -- measured on governance fields
    2026-09-08, the trailing std is zero on 63.0% of firm-days for
    `board_dissent_breadth_20` and 23.3% for `insider_ownership_pct`, so a self-history view
    of those is mostly absent and, where it does exist, inverts the sign of the raw signal.
    Check the zero-std rate before adopting it for a new field.

    ⚠ WHY IT LIVES HERE. It is the fifth standardizer this module exists to hold in one place,
    and its consumers are the shared panel builder (fundamentals valuation panel, governance board
    tenure) and the short-flow families.
    """
    if field_df is None or field_df.empty:
        return pd.DataFrame()
    mean = field_df.rolling(window, min_periods=min_periods).mean()
    std = field_df.rolling(window, min_periods=min_periods).std()
    z = zscore(field_df, mean, std.where(std > 0), Z_CLIP)
    if output_since is not None:
        z = z.loc[z.index >= pd.Timestamp(output_since)]
    return z
