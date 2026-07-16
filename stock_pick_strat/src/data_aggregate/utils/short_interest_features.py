"""
short_interest_features.py  (src/data_aggregate/utils/short_interest_features.py)
---------------------------------------------------------------------------------
Short-selling-pressure features from FINRA RegSHO daily short volume
(fetch_short_interest): [date, ticker, short_volume, total_volume].

    short_vol_ratio      trailing-21d mean of short_volume / total_volume
                         (high = heavily shorted -> short-interest anomaly:
                          predicts lower forward returns)
    short_vol_ratio_chg  recent (5d) minus baseline (63d) ratio
                         (rising short pressure)

Point-in-time: each day's RegSHO file is disseminated the NEXT morning, so the
ratio is lagged one trading day (`shift(1)`) before use. If a true short-INTEREST
positions series is present (columns `short_interest`, `avg_daily_volume`), a
`days_to_cover` field is added as well.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_aggregate.utils.fundamental_features import build_peer_relative_panel


def _short_fields(hist: pd.DataFrame, idx: pd.DatetimeIndex) -> dict:
    daily_short = hist.pivot_table(index="date", columns="ticker",
                                   values="short_volume", aggfunc="sum")
    daily_total = hist.pivot_table(index="date", columns="ticker",
                                   values="total_volume", aggfunc="sum")
    daily_short.index = pd.to_datetime(daily_short.index).normalize()
    daily_total.index = pd.to_datetime(daily_total.index).normalize()
    daily_short = daily_short.reindex(idx)
    daily_total = daily_total.reindex(idx)

    ratio = (daily_short / daily_total.where(daily_total > 0)).replace([np.inf, -np.inf], np.nan)
    F: dict[str, pd.DataFrame] = {}
    # lag one trading day: day-t short volume is only public on t+1
    F["short_vol_ratio"] = ratio.rolling(21, min_periods=5).mean().shift(1)
    F["short_vol_ratio_chg"] = (ratio.rolling(5, min_periods=3).mean()
                                - ratio.rolling(63, min_periods=20).mean()).shift(1)

    # optional: true short-interest positions if present in the same history
    if {"short_interest", "avg_daily_volume"}.issubset(hist.columns):
        si = hist.pivot_table(index="date", columns="ticker",
                              values="short_interest", aggfunc="last")
        adv = hist.pivot_table(index="date", columns="ticker",
                               values="avg_daily_volume", aggfunc="last")
        si.index = pd.to_datetime(si.index).normalize()
        adv.index = pd.to_datetime(adv.index).normalize()
        dtc = (si.reindex(idx).ffill() / adv.reindex(idx).ffill().where(lambda x: x > 0))
        F["days_to_cover"] = dtc.replace([np.inf, -np.inf], np.nan)
    return F


def build_short_interest_feature_panel(
    short_history: pd.DataFrame | None,
    peer_dict: dict,
    trading_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Long-format short-pressure feature panel (`f_<name>_vs_peers`, `f_<name>_xs`).
    Empty if no short-volume history is available."""
    if (short_history is None or short_history.empty
            or not {"short_volume", "total_volume"}.issubset(short_history.columns)):
        return pd.DataFrame(columns=["date", "ticker"])
    fields = _short_fields(short_history, trading_index)
    return build_peer_relative_panel(fields, peer_dict)
