"""Stock versus existing peer-basket total returns after an observed earnings report."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_aggregate.utils.fundamentals.earnings_features import reported_events


def disclosure_return_fields(
    earnings_history: pd.DataFrame,
    stock_returns: pd.DataFrame,
    peer_returns: pd.DataFrame,
    *,
    reaction_sessions: int = 3,
    max_age_days: int = 185,
) -> dict[str, pd.DataFrame]:
    """Compound both legs separately; a missing session never becomes a zero return."""
    if reaction_sessions != 3:
        raise ValueError("the earnings_abret_3d feature schema requires three reaction sessions")
    idx = stock_returns.index
    if not isinstance(idx, pd.DatetimeIndex):
        raise TypeError("disclosure returns require a DatetimeIndex")
    columns = stock_returns.columns.intersection(peer_returns.columns)
    since = pd.DataFrame(np.nan, index=idx, columns=columns)
    reaction = since.copy()
    events = reported_events(earnings_history, idx)
    # An earlier report's return anchor is outside this grid and cannot be reconstructed.
    if len(idx):
        events = events.loc[events["earnings_date"] >= idx[0]]
    for ticker, rows in events.groupby("ticker", sort=False):
        if ticker not in columns:
            continue
        rows = rows.drop_duplicates("as_of", keep="last")
        starts = idx.searchsorted(rows["as_of"])
        ends = np.r_[starts[1:], len(idx)]
        own = stock_returns[ticker].to_numpy(dtype=float)
        peers = peer_returns[ticker].reindex(idx).to_numpy(dtype=float)
        for start, end, report_date in zip(starts, ends, rows["earnings_date"], strict=True):
            own_leg, peer_leg = own[start:end], peers[start:end]
            valid = np.isfinite(own_leg) & np.isfinite(peer_leg) & (own_leg >= -1) & (peer_leg >= -1)
            complete = np.logical_and.accumulate(valid)
            abnormal = np.cumprod(1 + own_leg) - np.cumprod(1 + peer_leg)
            abnormal[~np.isfinite(abnormal)] = np.nan
            fresh = (idx[start:end] - report_date) <= pd.Timedelta(days=max_age_days)
            since.loc[idx[start:end], ticker] = np.where(complete & fresh, abnormal, np.nan)
            if end - start >= reaction_sessions and complete[reaction_sessions - 1]:
                target = idx[start + reaction_sessions - 1 : end]
                reaction.loc[target, ticker] = np.where(fresh[reaction_sessions - 1 :], abnormal[reaction_sessions - 1], np.nan)
    return {"earnings_abret_3d": reaction, "earnings_abret_since_report": since}
