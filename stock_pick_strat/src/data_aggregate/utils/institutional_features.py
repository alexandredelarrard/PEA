"""
institutional_features.py  (src/data_aggregate/utils/institutional_features.py)
--------------------------------------------------------------------------------
13F institutional-ownership features: per stock, aggregate all managers' long
positions each quarter and measure quarter-over-quarter accumulation / breadth
changes. Point-in-time via a 45-day filing lag (positions as-of quarter-end are
only public by the filing deadline), then forward-filled daily.

Input `holdings` is manager-grain long:
    [cik, period, ticker, shares, value_usd]   (filing_date optional)
`period` is the quarter-end (period-of-report).

Features (per stock, per quarter):
    inst_holders      # of managers holding the stock            (ownership breadth)
    inst_breadth_chg  Δ holders vs prior quarter = new - exit    (Chen-Hong-Stein)
    inst_shares_chg   % change in aggregate shares held          (accumulation)
    new_buyers        # managers initiating a position
    exiters           # managers fully exiting
    cluster_buying    (increasers - decreasers) / holders        (smart-money consensus)
    inst_ownership_pct aggregate 13F shares / shares outstanding  (needs fundamentals)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_aggregate.utils.factors import fundamentals_to_daily
from src.data_aggregate.utils.fundamental_features import build_peer_relative_panel

_FILING_LAG_DAYS = 45   # 13F must be filed within 45 days of quarter-end


def _quarter_features(holdings: pd.DataFrame) -> pd.DataFrame:
    """Manager-grain 13F -> one row per (ticker, quarter) with the QoQ features,
    stamped `as_of = quarter-end + 45 days` (leak-free availability date)."""
    h = holdings.copy()
    h["period"] = pd.to_datetime(h["period"]).dt.normalize()
    h = h.dropna(subset=["ticker", "cik", "period"])
    h["shares"] = pd.to_numeric(h["shares"], errors="coerce").fillna(0.0)
    h["value_usd"] = pd.to_numeric(h.get("value_usd", 0.0), errors="coerce").fillna(0.0)
    # amendments: keep the last-filed row per (ticker, manager, quarter)
    if "filing_date" in h.columns:
        h = h.sort_values("filing_date")
    h = h.drop_duplicates(["ticker", "cik", "period"], keep="last")

    rows = []
    for ticker, tdf in h.groupby("ticker"):
        prev: dict = {}
        for p in sorted(tdf["period"].unique()):
            cur_rows = tdf[tdf["period"] == p]
            cur = dict(zip(cur_rows["cik"], cur_rows["shares"]))
            cur_ciks, prev_ciks = set(cur), set(prev)
            holders = len(cur_ciks)
            both = cur_ciks & prev_ciks
            inc = sum(1 for c in both if cur[c] > prev[c])
            dec = sum(1 for c in both if cur[c] < prev[c])
            inst_shares = float(sum(cur.values()))
            prev_shares = float(sum(prev.values())) if prev else np.nan
            has_prev = len(prev_ciks) > 0
            rows.append({
                "ticker": ticker,
                "as_of": pd.Timestamp(p) + pd.Timedelta(days=_FILING_LAG_DAYS),
                "inst_holders": float(holders),
                "inst_shares": inst_shares,
                "inst_value": float(cur_rows["value_usd"].sum()),
                "new_buyers": float(len(cur_ciks - prev_ciks)) if has_prev else np.nan,
                "exiters": float(len(prev_ciks - cur_ciks)) if has_prev else np.nan,
                "inst_breadth_chg": float(holders - len(prev_ciks)) if has_prev else np.nan,
                "inst_shares_chg": (inst_shares / prev_shares - 1.0)
                                   if (has_prev and prev_shares > 0) else np.nan,
                "cluster_buying": ((inc - dec) / holders) if holders > 0 else np.nan,
            })
            prev = cur
    return pd.DataFrame(rows)


def build_institutional_feature_panel(
    holdings: pd.DataFrame | None,
    peer_dict: dict,
    trading_index: pd.DatetimeIndex,
    shares_out_history: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Long-format 13F feature panel (`f_<name>_vs_peers`, `f_<name>_xs`). Empty
    if no holdings. `shares_out_history` (fundamentals with `sharesOutstanding`)
    enables inst_ownership_pct."""
    need = {"cik", "period", "ticker", "shares"}
    if holdings is None or holdings.empty or not need.issubset(holdings.columns):
        return pd.DataFrame(columns=["date", "ticker"])

    qf = _quarter_features(holdings)
    if qf.empty:
        return pd.DataFrame(columns=["date", "ticker"])

    feats = ["inst_holders", "inst_breadth_chg", "inst_shares_chg",
             "new_buyers", "exiters", "cluster_buying"]
    fields = {f: fundamentals_to_daily(qf, f, trading_index) for f in feats}

    if shares_out_history is not None and not shares_out_history.empty:
        inst_sh = fundamentals_to_daily(qf, "inst_shares", trading_index)
        shares = fundamentals_to_daily(shares_out_history, "sharesOutstanding", trading_index)
        if not shares.empty and shares.notna().any().any():
            fields["inst_ownership_pct"] = (
                inst_sh / shares.where(shares > 0)).replace([np.inf, -np.inf], np.nan)

    return build_peer_relative_panel(fields, peer_dict)
