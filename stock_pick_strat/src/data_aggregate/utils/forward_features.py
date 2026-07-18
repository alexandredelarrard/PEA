"""
forward_features.py  (src/data_aggregate/utils/forward_features.py)
-------------------------------------------------------------------
Forward-looking valuation feature from the accruing yfinance snapshot
(`fundamentals_snapshot`):

    forward_earnings_yield = 1 / forward P/E   (forward E/P; higher = cheaper)

Point-in-time from each snapshot `as_of` and forward-filled, so it is leak-free.
Like the other snapshot-based features, coverage is ~empty historically and
accrues as `fundamentals_snapshot` is collected over successive run days.
"""
from __future__ import annotations

import pandas as pd

from src.data_aggregate.utils.factors import fundamentals_to_daily
from src.data_aggregate.utils.fundamental_features import build_peer_relative_panel


def build_forward_valuation_panel(
    snapshot: pd.DataFrame | None,
    peer_dict: dict,
    trading_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Long-format `f_forward_earnings_yield_{vs_peers,xs}` panel. Empty when the
    snapshot is unavailable or has no usable forward P/E yet."""
    if (snapshot is None or snapshot.empty
            or "as_of" not in snapshot.columns or "forwardPE" not in snapshot.columns):
        return pd.DataFrame(columns=["date", "ticker"])

    s = snapshot.copy()
    fpe = pd.to_numeric(s["forwardPE"], errors="coerce")
    s["forward_earnings_yield"] = (1.0 / fpe).where(fpe > 0)   # only sensible positive P/E

    daily = fundamentals_to_daily(s, "forward_earnings_yield", trading_index)
    fields: dict[str, pd.DataFrame] = {}
    if not daily.empty and daily.notna().any().any():
        fields["forward_earnings_yield"] = daily
    return build_peer_relative_panel(fields, peer_dict)
