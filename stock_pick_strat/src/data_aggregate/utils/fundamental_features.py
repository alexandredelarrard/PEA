"""
fundamental_features.py  (src/data_aggregate/utils/fundamental_features.py)
---------------------------------------------------------------------------
The "firm vs its direct competitors" fundamental signals -- the differentiator
of this strategy. For every fundamental characteristic we express the stock
RELATIVE TO ITS PEER BASKET (from the peer dict), not in absolute terms:

    rel_i(t) = (X_i(t) - peer_mean_i(t)) / peer_std_i(t)

so "cheap vs its competitors" and "growing faster than its competitors" become
features, and a market-wide value/growth level does NOT (that is the crowded
factor we already strip from the label). These peer-relative fundamentals are
largely orthogonal to the broad style factors, so they survive residualization
and are where real firm-specific edge lives.

Derived characteristics built from the raw yfinance fields:
    earnings_yield = 1 / trailingPE
    fcf_yield      = freeCashflow / marketCap
    ebitda_yield   = ebitda / enterpriseValue
    plus direct fields: revenueGrowth, earningsGrowth, grossMargins,
    operatingMargins, profitMargins, returnOnEquity, debtToEquity

Availability caveat (same as factors.py): with snapshot-only fundamentals these
are near-static and only populated from when you began collecting; backfill via
SimFin for real history.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from src.data_aggregate.utils.factors import fundamentals_to_daily


def _derived_fields(fund_hist: pd.DataFrame, idx: pd.DatetimeIndex) -> dict:
    """Build daily wide frames for each characteristic (raw + derived)."""
    F = {}

    def daily(field):
        return fundamentals_to_daily(fund_hist, field, idx)

    pe = daily("trailingPE")
    if not pe.empty:
        F["earnings_yield"] = 1.0 / pe.where(pe > 0)

    fcf, mcap = daily("freeCashflow"), daily("marketCap")
    if not fcf.empty and not mcap.empty:
        F["fcf_yield"] = (fcf / mcap.where(mcap > 0)).replace([np.inf, -np.inf], np.nan)

    ebitda, ev = daily("ebitda"), daily("enterpriseValue")
    if not ebitda.empty and not ev.empty:
        F["ebitda_yield"] = (ebitda / ev.where(ev > 0)).replace([np.inf, -np.inf], np.nan)

    for field in ["revenueGrowth", "earningsGrowth", "grossMargins",
                  "operatingMargins", "profitMargins", "returnOnEquity",
                  "debtToEquity"]:
        f = daily(field)
        if not f.empty:
            F[field] = f
    return F


def _peer_relative(field_df: pd.DataFrame, peer_dict: dict) -> pd.DataFrame:
    """
    (stock - peer_weighted_mean) / peer_weighted_std, per date, per stock.
    Self excluded by construction of the peer dict.
    """
    rel = pd.DataFrame(index=field_df.index, columns=field_df.columns, dtype="float64")
    for ticker, peers in peer_dict.items():
        if not peers or ticker not in field_df.columns:
            continue
        cols = [p for p in peers if p in field_df.columns]
        if len(cols) < 3:
            continue
        w = np.array([peers[p] for p in cols], dtype="float64")
        w = w / w.sum()
        peer_vals = field_df[cols]
        pmean = peer_vals.mul(w, axis=1).sum(axis=1, min_count=1)
        # weighted std around the weighted mean
        var = (peer_vals.sub(pmean, axis=0) ** 2).mul(w, axis=1).sum(axis=1, min_count=1)
        pstd = np.sqrt(var).replace(0, np.nan)
        rel[ticker] = (field_df[ticker] - pmean) / pstd
    return rel


def build_fundamental_feature_panel(
    fundamentals_history: pd.DataFrame | None,
    peer_dict: dict,
    trading_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Long-format panel: ['date','ticker', f_<char>_vs_peers, ...] plus a
    cross-sectional rank of each characteristic (f_<char>_xs) as a secondary
    view. Empty frame if no fundamentals available.
    """
    if fundamentals_history is None or fundamentals_history.empty:
        return pd.DataFrame(columns=["date", "ticker"])

    fields = _derived_fields(fundamentals_history, trading_index)
    if not fields:
        return pd.DataFrame(columns=["date", "ticker"])

    long_frames = []
    for name, fdf in fields.items():
        # peer-relative (firm vs direct competitors)
        rel = _peer_relative(fdf, peer_dict)
        s = rel.stack()
        s.index.set_names(["date", "ticker"], inplace=True)
        long_frames.append(s.rename(f"f_{name}_vs_peers"))

        # cross-sectional rank (firm vs whole universe) -- secondary view
        xs = fdf.rank(axis=1, pct=True, method="average")
        s2 = xs.stack()
        s2.index.set_names(["date", "ticker"], inplace=True)
        long_frames.append(s2.rename(f"f_{name}_xs"))

    panel = pd.concat(long_frames, axis=1).reset_index()
    return panel
