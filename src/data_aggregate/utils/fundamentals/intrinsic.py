"""Point-in-time two-stage cash-flow DCF values against supported saved capitalization."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from src.data_aggregate.utils.common.pit import fundamentals_to_daily
from src.data_aggregate.utils.price.frames import PriceFrames
from src.utils.price.per_share import same_capital_scope

_QUARTERLY_MAX_AGE_DAYS = 185


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
    prices: PriceFrames | None,
    idx: pd.DatetimeIndex,
    settings: Mapping[str, Any] | None = None,
) -> dict[str, pd.DataFrame]:
    """Return issuer DCF total/yield, plus per-security value only with matching coverage."""
    options = settings or {}
    df_cashflow = fundamentals_to_daily(fund_hist, "freeCashflow", idx, max_age_days=_QUARTERLY_MAX_AGE_DAYS)
    if df_cashflow.empty:
        return {}
    df_growth = fundamentals_to_daily(fund_hist, "revenueGrowth", idx, max_age_days=_QUARTERLY_MAX_AGE_DAYS)
    df_growth = df_growth.reindex_like(df_cashflow).clip(float(options.get("growth_floor", -0.10)), float(options.get("growth_cap", 0.15)))
    terminal_growth = float(options.get("terminal_growth", 0.025))
    df_growth = df_growth.fillna(terminal_growth)
    df_total = two_stage_dcf(df_cashflow, df_growth, float(options.get("discount_rate", 0.10)), terminal_growth, int(options.get("years", 5)))
    fields = {"total": df_total}
    if prices is None or prices.mcap_pit is None:
        return fields

    # Carrying daily capital never refreshes its original public-share observation age.
    df_eligible = fundamentals_to_daily(fund_hist, "sharesOutstanding", idx, max_age_days=_QUARTERLY_MAX_AGE_DAYS)
    df_cap = prices.mcap_pit.reindex(index=idx, columns=df_total.columns)
    df_cap = df_cap.where(df_eligible.reindex_like(df_cap).gt(0) & np.isfinite(df_cap) & df_cap.gt(0))
    fields["yield"] = (df_total / df_cap).replace([np.inf, -np.inf], np.nan)
    if prices.shares_outstanding_pit is not None:
        df_count = prices.shares_outstanding_pit.reindex_like(df_total)
        df_supported = same_capital_scope(prices.close_pit, df_count, df_cap)
        fields["per_share"] = (df_total / df_count.where(df_count.gt(0))).where(df_supported).replace([np.inf, -np.inf], np.nan)
    return fields
