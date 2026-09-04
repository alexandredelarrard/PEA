"""
xs.py  (src/data_aggregate/utils/common/xs.py)
---------------------------------------------
CROSS-SECTIONAL operations: every per-day, across-tickers transform in one place.

Five separate implementations of "standardize within a date" had accumulated --
`factors._xs_z` (clip 4), `features.cross_sectional_standardize` (rank | z clip 3),
`targets.cross_sectional_zscore` (clip 3 + a min_names gate), the inline z inside
`targets`' neutralization design (clip 4, zero-sd -> NaN) and
`composites._xs_standardize` (long panel, groupby("date")) -- plus four independent
`rank(axis=1, pct=True)` call sites.

THE CLIPS ARE NOT AN ACCIDENT AND ARE NOT HARMONISED HERE. They encode three different
jobs, so `clip` is a REQUIRED argument with no default -- nobody can unify them by
forgetting to pass one:

    3.0  the LABEL          RMSE stability: a handful of extreme residuals must not
                            dominate the loss the model is fitted against.
    4.0  a CHARACTERISTIC   factor/regression weights: keep the tail, bound its leverage.
    8.0  the PEER z         `panel.peer_relative`, already winsorised at 1%/99% downstream.

Likewise the ZERO-DISPERSION policy is explicit rather than incidental. On a day where
every name shares one value, `sd == 0`, so an unguarded divide yields +/-inf which `clip`
then turns into a FABRICATED +/-clip; the two guarded call sites instead produce NaN.
Both behaviours are reproduced exactly (`zero_sd_to_nan`) because changing either would
move live numbers -- that is a separate, declared decision, not a refactor.

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

_WINSOR_LO, _WINSOR_HI = 0.01, 0.99

# the three clip policies, named so a call site documents its intent
XS_CLIP_LABEL = 3.0             # modelling target (targets.py)
XS_CLIP_CHARACTERISTIC = 4.0    # factor characteristic / regressor (factors.py, composites)
XS_CLIP_PEER = 8.0             # peer-relative z (panel.py), winsorised again downstream

#: How many PRESENT members a design CELL needs on a day before the group block may split the
#: day's names on it (`_floored_group_block` -- read it before changing this, the rule is about
#: cells and not columns). See the module docstring for the defect it closes: a 1-member cell
#: forces an exactly-zero residual and a 2-member cell an exactly-halved one -- measured 5,230
#: exactly-zero and 16,960 halved cells at h=60, `F` losing 48.7% of its history and `CSGP`
#: 21.5%.
#:
#: The bar is 5 rather than the 2 or 3 that would close only the arithmetically degenerate
#: cases: a 3- or 4-member demean still removes a third to a quarter of the name's own variance
#: as if it were a group effect, which is the same defect with a smaller coefficient. The cost
#: is the other side of that trade -- below the floor the name still gets the FULL rest of the
#: projection (betas, momentum, log market cap) and is simply not forced to a group-zero mean.
#: No cell is nulled by this rule.
MIN_GROUP_SIZE_FOR_NEUTRALIZATION = 5


def winsorize_xs(df: pd.DataFrame, lo: float = _WINSOR_LO,
                 hi: float = _WINSOR_HI) -> pd.DataFrame:
    """Clip each ROW (date) to its cross-sectional [lo, hi] quantiles across tickers.
    NaN-safe: an all-NaN row yields NaN bounds -> clip is a no-op there."""
    if df is None or df.empty:
        return df
    lo_q = df.quantile(lo, axis=1)
    hi_q = df.quantile(hi, axis=1)
    return df.clip(lower=lo_q, upper=hi_q, axis=0)


def xs_rank_pct(df: pd.DataFrame) -> pd.DataFrame:
    """THE cross-sectional percentile rank in [0, 1], per date across tickers.

    `method="average"` is pandas' default and is stated explicitly because the four
    call sites this replaces were split between spelling it and omitting it."""
    return df.rank(axis=1, pct=True, method="average")


def xs_z(df: pd.DataFrame, clip: float | None, *,
         zero_sd_to_nan: bool = False,
         eps: float = 1e-12) -> pd.DataFrame:
    """THE cross-sectional z-score, per date across tickers.

    `clip` is required (see the module docstring); pass `None` for unclipped.
    `zero_sd_to_nan=False` reproduces the UNGUARDED behaviour of `factors._xs_z` /
    `features.cross_sectional_standardize` / `targets.cross_sectional_zscore`
    (sd == 0 -> +/-inf -> clip -> +/-clip); `True` is the guarded behaviour a REGRESSOR needs
    (sd == 0 -> NaN), used by `targets._neutralizing_design`."""
    mu = df.mean(axis=1)
    sd = df.std(axis=1)
    if zero_sd_to_nan:
        sd = sd.replace(0.0, np.nan)
    else:
        # Guard against sd being zero or near-zero due to floating point precision
        sd = sd.mask(sd < eps, np.nan)
    z = df.sub(mu, axis=0).div(sd, axis=0)
    return z if clip is None else z.clip(-clip, clip)


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


def xs_project_out(values: pd.DataFrame, exposures: list[pd.DataFrame],
                   dummies: pd.DataFrame | None = None, *,
                   min_group_size: int | None = None) -> pd.DataFrame:
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
    stacked = (np.stack([e.reindex_like(values).to_numpy(float) for e in exposures], axis=2)
               if exposures else np.zeros((*values.shape, 0)))
    group_block = (dummies.reindex(values.columns).to_numpy(float)
                   if dummies is not None else np.zeros((values.shape[1], 0)))
    y = values.to_numpy(float)
    out = np.full_like(y, np.nan)
    for i in range(len(y)):
        present = np.isfinite(y[i])
        grp = group_block[present]
        if min_group_size is not None and grp.shape[1]:
            grp = _floored_group_block(grp, min_group_size)
        design = np.column_stack([stacked[i][present], grp])
        if present.sum() > design.shape[1]:          # else the fit is exact and says nothing
            out[i, present] = _day_residual(y[i][present], design)
    return pd.DataFrame(out, index=values.index, columns=values.columns)


def xs_standardize(feat: pd.DataFrame, method: Literal["rank", "zscore"],
                   clip: float = XS_CLIP_LABEL) -> pd.DataFrame:
    """Standardize one feature within each day (across stocks).
      'rank'   -> percentile in [0,1] (robust to outliers)
      'zscore' -> demean/divide by cross-sectional std, clipped at +/-`clip`
    """
    if method == "rank":
        return xs_rank_pct(feat)
    if method == "zscore":
        return xs_z(feat, clip=clip)
    raise ValueError("method must be 'rank' or 'zscore'")


def long_xs_standardize(panel: pd.DataFrame, cols: list[str], method: str,
                        clip: float) -> pd.DataFrame:
    """Cross-sectionally standardize each column within each date, on a LONG panel.

    Kept separate from `xs_z`/`xs_rank_pct` on purpose: this operates on a long
    (date, ticker) frame via `groupby("date")` rather than a wide date x ticker one, the
    rank branch rescales to ~[-1, 1] instead of [0, 1], and the z branch has its OWN
    zero-dispersion guard (`s.std() if s.std() > 0 else np.nan`). Three real differences,
    so unifying it would change the composites. Was `composites._xs_standardize`."""
    g = panel.groupby("date")[cols]
    if method == "rank":
        return (g.rank(pct=True) - 0.5) * 2.0            # -> ~[-1, 1], mean ~0
    z = g.transform(lambda s: (s - s.mean()) / (s.std() if s.std() > 0 else np.nan))
    return z.clip(-clip, clip)
