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

Also folds in SEC Fails-to-Deliver (`fails_history`, from fetch_fails_to_deliver):
    fails_to_deliver_ratio  trailing-21d fails / trailing-21d traded volume
                            (settlement stress / naked-short pressure)
    fails_to_deliver_chg    recent (5d) minus baseline (63d) fails ratio
FTD files are published ~1-2 months after the settlement period, so the signal is
lagged `_FTD_PUB_LAG` trading days to its (conservative) availability date.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_aggregate.utils.panel import build_peer_relative_panel

_FTD_PUB_LAG = 40   # ~2 months of trading days: SEC FTD files are published well after the
                    # settlement period, so lag the signal to its (conservative) availability.


def _fails_fields(fails_hist: pd.DataFrame, idx: pd.DatetimeIndex,
                  volume: pd.DataFrame | None) -> dict:
    """Fails-to-deliver pressure. A security is listed only on days it had fails, so
    absent -> 0. Normalize by trailing average traded volume ('fails as a share of
    volume') and lag to the FTD publication date. Falls back to raw smoothed fails
    (still peer-relativized) when no volume frame is supplied."""
    fails = fails_hist.pivot_table(index="date", columns="ticker",
                                   values="fails_quantity", aggfunc="sum")
    fails.index = pd.to_datetime(fails.index).normalize()
    fails = fails.reindex(idx).fillna(0.0)                     # no listing that day = 0 fails
    fails_sm = fails.rolling(21, min_periods=5).mean()
    if volume is not None and not volume.empty:
        vol = volume.reindex(idx).reindex(columns=fails.columns)
        denom = vol.rolling(21, min_periods=5).mean()
        ratio = (fails_sm / denom.where(denom > 0)).replace([np.inf, -np.inf], np.nan)
    else:
        ratio = fails_sm
    F: dict[str, pd.DataFrame] = {}
    F["fails_to_deliver_ratio"] = ratio.shift(_FTD_PUB_LAG)
    chg = (fails.rolling(5, min_periods=3).mean() - fails.rolling(63, min_periods=20).mean())
    if volume is not None and not volume.empty:
        chg = (chg / vol.rolling(63, min_periods=20).mean().where(lambda x: x > 0))
    F["fails_to_deliver_chg"] = chg.replace([np.inf, -np.inf], np.nan).shift(_FTD_PUB_LAG)
    return F


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
    fails_history: pd.DataFrame | None = None,
    volume: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Long-format short-pressure + fails-to-deliver feature panel
    (`f_<name>_vs_peers`, `f_<name>_xs`). Empty if neither source is available.
    `volume` (daily traded volume) normalizes the fails ratio."""
    fields: dict = {}
    if (short_history is not None and not short_history.empty
            and {"short_volume", "total_volume"}.issubset(short_history.columns)):
        fields.update(_short_fields(short_history, trading_index))
    if (fails_history is not None and not fails_history.empty
            and "fails_quantity" in fails_history.columns):
        fields.update(_fails_fields(fails_history, trading_index, volume))
    if not fields:
        return pd.DataFrame(columns=["date", "ticker"])
    return build_peer_relative_panel(fields, peer_dict)
