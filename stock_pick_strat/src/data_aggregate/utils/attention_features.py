"""
attention_features.py  (src/data_aggregate/utils/attention_features.py)
-----------------------------------------------------------------------
Retail-ATTENTION features from a per-ticker attention time series (Wikipedia
pageviews or Google Trends search interest). Source-agnostic: both extractors
emit a long frame [date, ticker, <value_col>] and this builder turns it into
peer-relative features under a source `prefix`.

Signal (Da-Engelberg-Gao "In Search of Attention"): a SPIKE in attention vs the
name's own baseline flags retail-driven attention (often short-term overpricing
-> subsequent reversal). Everything is point-in-time (trailing windows only):

    <prefix>_attn_spike   log recent (5d) avg / trailing (63d) avg attention
    <prefix>_attn_level   log trailing (21d) average attention (attention level)

Weekly series (Google Trends) are forward-filled onto the daily trading calendar
(bounded ffill), so the value on date t is the last observation on/before t.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_aggregate.utils.fundamental_features import build_peer_relative_panel


def _attention_fields(attn_hist: pd.DataFrame, idx: pd.DatetimeIndex,
                      prefix: str, value_col: str, ffill_limit: int = 7) -> dict:
    piv = attn_hist.pivot_table(index="date", columns="ticker",
                                values=value_col, aggfunc="sum")
    piv.index = pd.to_datetime(piv.index).normalize()
    # onto the trading calendar; bounded ffill bridges weekly->daily and small gaps
    piv = piv.reindex(piv.index.union(idx)).ffill(limit=ffill_limit).reindex(idx)
    A = piv.clip(lower=0.0)

    recent = A.rolling(5, min_periods=3).mean()
    base = A.rolling(63, min_periods=20).mean()
    spike = (np.log1p(recent) - np.log1p(base)).replace([np.inf, -np.inf], np.nan)
    level = np.log1p(A.rolling(21, min_periods=10).mean()).replace([np.inf, -np.inf], np.nan)
    return {f"{prefix}_attn_spike": spike, f"{prefix}_attn_level": level}


def build_attention_feature_panel(
    attention_history: pd.DataFrame | None,
    peer_dict: dict,
    trading_index: pd.DatetimeIndex,
    prefix: str,
    value_col: str = "value",
) -> pd.DataFrame:
    """Long-format attention feature panel (`f_<prefix>_attn_*_vs_peers`, `_xs`).
    Empty if no attention history is available."""
    if (attention_history is None or attention_history.empty
            or value_col not in attention_history.columns):
        return pd.DataFrame(columns=["date", "ticker"])
    fields = _attention_fields(attention_history, trading_index, prefix, value_col)
    return build_peer_relative_panel(fields, peer_dict)
