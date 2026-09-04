"""
prices.py  (src/data_aggregate/utils/common/prices.py)
-----------------------------------------------------
Price-derived primitives shared by the momentum features, the style factors and the
targets. Each was implemented twice, in modules that do not import each other:

    momentum 12-1     `features.mom_12_1` re-derived the expression that
                      `factors.momentum_characteristic` already claimed, in its own
                      docstring, to be the single source of truth for. The feature the
                      model trains on and the factor the target is neutralised against
                      were two code paths that only happened to agree.
    trailing vol      `features.vol_21` / `vol_63` vs `factors.resvol` (same rolling std,
                      negated).
    forward compound  the reverse-rolling log-sum trick in `targets.forward_compound` and
                      again inside `features`'s seasonality block -- a leak-sensitive
                      routine written twice, with different `min_periods` policies.

NOT here any more: `price_column_returns`. Its whole job was remapping factor name -> price
COLUMN ({"oil": "CL=F"}) while the commodity/FX series sat inside the `prices` panel. Those
series now live in `prices_macro` under their factor names, so the remap is the identity and
`StepCubeTarget._asset_factors` just takes the pct_change -- and it no longer silently skips a
missing column, which is how a factor could vanish from the panel with no error.

The differing `min_periods` policies are PARAMETERS here, not harmonised: the target must
not accept a partial forward window (that would silently be a shorter horizon), while the
seasonality feature averages five prior years and would lose its newest year at the sample
edge if it demanded a full one. Both call sites pass their own value explicitly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_aggregate.utils.common.level_basis import mask_seam_windows

#: The lookback `momentum_characteristic` reads, as `(back, skip)` for `mask_seam_windows`:
#: the value at `t` reads `close_total` at positions `t-252` and `t-21` only, so it straddles a
#: seam at `s` for `t` in `[s+21, s+251]`. One consumer (below), so it stays here.
MOMENTUM_SEAM_WINDOW = (252, 21)


def momentum_characteristic(stock_close: pd.DataFrame, *,
                            seams: dict[str, list[pd.Timestamp]] | None = None
                            ) -> pd.DataFrame:
    """12-1 price momentum characteristic (skip the most recent month).

    Single source of truth, now actually shared: the momentum style factor, the
    `mom_12_1` model feature AND the target's momentum neutralization all call this.
    Point-in-time (uses only past prices), so it never leaks future information.

    ⚠ ALL THREE CONSUMERS MUST PASS `seams` (from `level_basis.measure_seams`), and they do not
    share a VALUE -- each recomputes this call on its own frames, so a fix applied at one call
    site leaves the other two fabricating. Where a registered `null_ret` seam sits inside the
    252-day window, `close_total`'s two bases are mixed and the ratio is not a return at all:
    unmasked, `DHR` reads rank 0.9934 for 231 consecutive trading days after 2016-07-05, which
    is the top decile of a 460-name cross-section on a number the market never produced. The
    target neutralizes against that same fabricated exposure.

    `seams=None` means no mask and reproduces the pre-mask value bit-for-bit, so a synthetic
    fixture with no register stays a one-liner.
    """
    mom = stock_close.shift(21) / stock_close.shift(252) - 1.0
    if not seams:
        return mom
    return mask_seam_windows(mom, seams, *MOMENTUM_SEAM_WINDOW)


def trailing_vol(returns: pd.DataFrame, window: int,
                 min_periods: int | None = None) -> pd.DataFrame:
    """Trailing realized volatility = rolling std of daily returns.

    `features` uses it directly (`vol_21`, `vol_63`); `factors` NEGATES it for the
    low-volatility style characteristic (low vol = the long side)."""
    return returns.rolling(window, min_periods=min_periods).std()


def forward_return(total_return_index: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Simple forward return over the next `horizon` rows, from a TOTAL-RETURN INDEX.

    ⚠ NEVER hand this an equity price series. `close_split` is a PRICE, so its ratio is a
    price return and excludes every dividend -- which is exactly the defect that made the
    labels short high-yielders (MO reads 1.24x over the sample where the truth is 20.2x).
    The stock labels were rebuilt on `forward_compound(stock_ret, h)` for that reason and no
    longer call this.

    It survives for the series where a ratio of levels IS a total return by construction:
    the macro legs (`equity_tr`, `bond_10y_tr`), which are cumulative-return indices. For
    anything with a separate dividend stream, use `forward_compound` on daily returns."""
    return total_return_index.shift(-horizon) / total_return_index - 1.0


def forward_compound(daily: pd.Series | pd.DataFrame, horizon: int,
                     min_periods: int | None = None):
    """Compounded forward return over t+1..t+h from a daily-return series/frame."""
    safe = daily.clip(lower=-0.999999)
    log1p = np.log1p(safe)
    mp = horizon if min_periods is None else min_periods
    fwd = np.expm1(log1p[::-1].rolling(horizon, min_periods=mp).sum()[::-1].shift(-1))
    return fwd


def forward_cumchange(level_change: pd.Series | pd.DataFrame, horizon: int,
                      min_periods: int | None = None):
    """Cumulative forward change over t+1..t+h from a daily-CHANGE series/frame."""
    mp = horizon if min_periods is None else min_periods
    return level_change[::-1].rolling(horizon, min_periods=mp).sum()[::-1].shift(-1)


