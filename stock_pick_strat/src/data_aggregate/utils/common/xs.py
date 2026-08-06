"""
xs.py  (src/data_aggregate/utils/common/xs.py)
---------------------------------------------
CROSS-SECTIONAL operations: every per-day, across-tickers transform in one place.

Five separate implementations of "standardize within a date" had accumulated --
`factors._xs_z` (clip 4), `features.cross_sectional_standardize` (rank | z clip 3),
`targets.cross_sectional_zscore` (clip 3 + a min_names gate), the inline z inside
`targets.cross_sectional_neutralize` (clip 4, zero-sd -> NaN) and
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
         zero_sd_to_nan: bool = False) -> pd.DataFrame:
    """THE cross-sectional z-score, per date across tickers.

    `clip` is required (see the module docstring); pass `None` for unclipped.
    `zero_sd_to_nan=False` reproduces the UNGUARDED behaviour of `factors._xs_z` /
    `features.cross_sectional_standardize` / `targets.cross_sectional_zscore`
    (sd == 0 -> +/-inf -> clip -> +/-clip); `True` reproduces the guarded behaviour of
    `targets.cross_sectional_neutralize` (sd == 0 -> NaN)."""
    mu = df.mean(axis=1)
    sd = df.std(axis=1)
    if zero_sd_to_nan:
        sd = sd.replace(0.0, np.nan)
    z = df.sub(mu, axis=0).div(sd, axis=0)
    return z if clip is None else z.clip(-clip, clip)


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
