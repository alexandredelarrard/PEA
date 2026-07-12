"""
intrinsic.py  (src/data_aggregate/utils/intrinsic.py)
-----------------------------------------------------
A transparent, point-in-time INTRINSIC VALUE estimate from the firm's own cash
generation -- the "what is this business worth on its future cash flows" anchor
that we then compare against the market price AND against the sell-side analyst
estimates.

Model: two-stage discounted cash flow on trailing-twelve-month free cash flow.
    * Stage 1 (explicit): grow base FCF at `growth` for `years`, discount at `r`.
    * Stage 2 (terminal): a Gordon perpetuity at `terminal_growth`.
        V = Σ_{t=1..N} FCF·(1+g)^t / (1+r)^t
          + [ FCF·(1+g)^N·(1+g_term) / (r - g_term) ] / (1+r)^N

Everything is built from fundamentals that are already keyed on their SEC filing
date (via `fundamentals_to_daily`), so the estimate is point-in-time and never
uses a cash-flow number before it was public. `growth` is the firm's own TTM
revenue growth, winsorized to a sane band so a single blow-out quarter cannot
produce an absurd valuation. Where base FCF <= 0 the DCF is undefined (returned
as NaN) -- a business burning cash has no meaningful cash-flow intrinsic value.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from src.data_aggregate.utils.factors import fundamentals_to_daily, daily_market_cap


def two_stage_dcf(
    base_cf: pd.DataFrame,
    growth: pd.DataFrame,
    discount_rate: float,
    terminal_growth: float,
    years: int,
) -> pd.DataFrame:
    """Two-stage DCF value of a cash-flow stream. `base_cf` and `growth` are
    aligned wide frames (date x ticker). Returns the intrinsic equity value.
    NaN where base_cf <= 0 (cash-burning -> no cash-flow value)."""
    if base_cf.empty:
        return pd.DataFrame()
    r = float(discount_rate)
    gt = float(terminal_growth)
    if r <= gt:
        raise ValueError("discount_rate must exceed terminal_growth")

    g = growth.reindex_like(base_cf)
    cf_pos = base_cf.where(base_cf > 0)

    pv = pd.DataFrame(0.0, index=base_cf.index, columns=base_cf.columns)
    for t in range(1, years + 1):
        pv = pv + cf_pos * (1.0 + g) ** t / (1.0 + r) ** t
    terminal_cf = cf_pos * (1.0 + g) ** years * (1.0 + gt)
    terminal_value = terminal_cf / (r - gt)
    pv = pv + terminal_value / (1.0 + r) ** years
    return pv.where(cf_pos.notna())


def intrinsic_value_daily(
    fund_hist: pd.DataFrame,
    close: pd.DataFrame | None,
    idx: pd.DatetimeIndex,
    discount_rate: float = 0.10,
    terminal_growth: float = 0.025,
    years: int = 5,
    growth_cap: float = 0.15,
    growth_floor: float = -0.10,
) -> dict:
    """Point-in-time intrinsic value from TTM free cash flow.

    Returns wide daily frames:
      * total     : intrinsic equity value (currency)
      * per_share : total / shares outstanding
      * yield      : total / market cap  ( >1 => cheaper than intrinsic )
    `yield` and `per_share` require `close`; `total` is always returned.
    """
    base_fcf = fundamentals_to_daily(fund_hist, "freeCashflow", idx)   # TTM, PIT
    if base_fcf.empty:
        return {}
    rev_growth = fundamentals_to_daily(fund_hist, "revenueGrowth", idx)
    g = rev_growth.reindex_like(base_fcf).clip(growth_floor, growth_cap)
    g = g.fillna(terminal_growth)                                      # no growth info -> conservative

    total = two_stage_dcf(base_fcf, g, discount_rate, terminal_growth, years)
    out = {"total": total}

    shares = fundamentals_to_daily(fund_hist, "sharesOutstanding", idx)
    if not shares.empty:
        cols = total.columns.intersection(shares.columns)
        sh = shares[cols].where(shares[cols] > 0)
        out["per_share"] = (total[cols] / sh).replace([np.inf, -np.inf], np.nan)

    if close is not None:
        mcap = daily_market_cap(fund_hist, close)
        if not mcap.empty:
            cols = total.columns.intersection(mcap.columns)
            out["yield"] = (total[cols] / mcap[cols]).replace([np.inf, -np.inf], np.nan)
    return out
