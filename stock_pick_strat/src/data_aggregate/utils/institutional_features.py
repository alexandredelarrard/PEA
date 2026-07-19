"""
institutional_features.py  (src/data_aggregate/utils/institutional_features.py)
--------------------------------------------------------------------------------
13F institutional-ownership features: per stock, aggregate all managers' reported
positions each quarter and measure quarter-over-quarter accumulation, breadth,
concentration, option positioning, and the dollar weight of institutions in the
name.

POINT-IN-TIME (critical): a 13F reports positions as of the quarter-END but is
only public up to the SEC filing deadline ~45 days later (a March-31 position set
is not knowable until ~mid-May). Every quarter's aggregate is therefore stamped
`as_of = quarter-end + _FILING_LAG_DAYS` and only forward-filled onto trading days
FROM that date — so a backtest never sees a quarter's 13F before it was public.

Input `holdings` is manager-grain long (one row per manager x security x quarter):
    [cik, period, ticker, shares, value_usd, call_value, put_value]  (filing_date optional)
`period` is the quarter-end (period-of-report); `value_usd` is long-equity value
(already era-adjusted to dollars upstream), `call_value`/`put_value` the option
exposure the manager reported on the name.

Features (per stock, per quarter; all peer-relativized downstream):
    inst_holders        # of managers holding                       (ownership breadth)
    inst_breadth_chg    Δ holders vs prior quarter = new - exit      (Chen-Hong-Stein)
    inst_shares_chg     % QoQ change in aggregate shares held        (VOLUME accumulation)
    inst_value_chg      % QoQ change in aggregate long value         (VALUE accumulation)
    new_buyers/exiters  # managers initiating / fully exiting
    new_buyer_ratio     new_buyers / holders                         (fresh-money breadth)
    cluster_buying      (increasers - decreasers) / holders          (smart-money consensus)
    inst_concentration  Herfindahl of managers' value shares         (crowding / fragility)
    net_options_ratio   (call_value - put_value) / total invested    (options sentiment)
    net_options_ratio_chg  QoQ Δ in net_options_ratio                (sentiment shift)
    inst_ownership_pct  aggregate 13F shares / shares outstanding     (needs fundamentals)
    inst_value_to_mcap  aggregate 13F long value / market cap         (institutional WEIGHT)
    inst_flow_to_mcap   net QoQ $ flow / market cap                   (size-scaled $ accumulation)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_aggregate.utils.factors import daily_market_cap, fundamentals_to_daily
from src.data_aggregate.utils.fundamental_features import build_peer_relative_panel

_FILING_LAG_DAYS = 45   # 13F must be filed within 45 days of quarter-end (leak-free floor)


def _quarter_features(holdings: pd.DataFrame) -> pd.DataFrame:
    """Manager-grain 13F -> one row per (ticker, quarter) with the QoQ features,
    stamped `as_of = quarter-end + 45 days` (leak-free availability date)."""
    h = holdings.copy()
    h["period"] = pd.to_datetime(h["period"]).dt.normalize()
    h = h.dropna(subset=["ticker", "cik", "period"])
    for c in ("shares", "value_usd", "call_value", "put_value"):
        h[c] = pd.to_numeric(h[c], errors="coerce").fillna(0.0) if c in h.columns \
            else pd.Series(0.0, index=h.index)
    # amendments: keep the last-filed row per (ticker, manager, quarter)
    if "filing_date" in h.columns:
        h = h.sort_values("filing_date")
    h = h.drop_duplicates(["ticker", "cik", "period"], keep="last")

    rows = []
    for ticker, tdf in h.groupby("ticker"):
        prev: dict = {}
        prev_value, prev_opt = np.nan, np.nan
        for p in sorted(tdf["period"].unique()):
            cur_rows = tdf[tdf["period"] == p]
            cur = dict(zip(cur_rows["cik"], cur_rows["shares"]))
            cur_ciks, prev_ciks = set(cur), set(prev)
            holders = len(cur_ciks)
            both = cur_ciks & prev_ciks
            inc = sum(1 for c in both if cur[c] > prev[c])
            dec = sum(1 for c in both if cur[c] < prev[c])
            has_prev = len(prev_ciks) > 0

            inst_shares = float(sum(cur.values()))
            prev_shares = float(sum(prev.values())) if prev else np.nan
            inst_value = float(cur_rows["value_usd"].sum())
            call_v = float(cur_rows["call_value"].sum())
            put_v = float(cur_rows["put_value"].sum())
            total_invested = inst_value + call_v + put_v            # long equity + option exposure
            opt_ratio = ((call_v - put_v) / total_invested) if total_invested > 0 else np.nan
            # crowding: Herfindahl of managers' VALUE shares (high = few dominant holders)
            mv = cur_rows.groupby("cik")["value_usd"].sum()
            tot_mv = float(mv.sum())
            hhi = float(((mv / tot_mv) ** 2).sum()) if tot_mv > 0 else np.nan
            new_b = float(len(cur_ciks - prev_ciks)) if has_prev else np.nan

            rows.append({
                "ticker": ticker,
                "as_of": pd.Timestamp(p) + pd.Timedelta(days=_FILING_LAG_DAYS),
                "inst_holders": float(holders),
                "inst_shares": inst_shares,
                "inst_value": inst_value,
                # net QoQ dollar flow (long value); NaN on the first observed quarter
                "inst_value_flow": (inst_value - prev_value)
                                   if (has_prev and np.isfinite(prev_value)) else np.nan,
                "new_buyers": new_b,
                "exiters": float(len(prev_ciks - cur_ciks)) if has_prev else np.nan,
                "inst_breadth_chg": float(holders - len(prev_ciks)) if has_prev else np.nan,
                "inst_shares_chg": (inst_shares / prev_shares - 1.0)
                                   if (has_prev and prev_shares > 0) else np.nan,
                "inst_value_chg": (inst_value / prev_value - 1.0)
                                  if (has_prev and np.isfinite(prev_value) and prev_value > 0) else np.nan,
                "cluster_buying": ((inc - dec) / holders) if holders > 0 else np.nan,
                "new_buyer_ratio": (new_b / holders)
                                   if (holders > 0 and np.isfinite(new_b)) else np.nan,
                "net_options_ratio": opt_ratio,
                "net_options_ratio_chg": (opt_ratio - prev_opt)
                                         if (has_prev and np.isfinite(prev_opt)
                                             and np.isfinite(opt_ratio)) else np.nan,
                "inst_concentration": hhi,
            })
            prev, prev_value, prev_opt = cur, inst_value, opt_ratio
    return pd.DataFrame(rows)


def build_institutional_feature_panel(
    holdings: pd.DataFrame | None,
    peer_dict: dict,
    trading_index: pd.DatetimeIndex,
    shares_out_history: pd.DataFrame | None = None,
    stock_close: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Long-format 13F feature panel (`f_<name>_vs_peers`, `f_<name>_xs`). Empty if
    no holdings. `shares_out_history` (fundamentals with `sharesOutstanding`) enables
    inst_ownership_pct; `shares_out_history` + `stock_close` enable the value/market-cap
    weight and the size-scaled net-flow (via a point-in-time daily market cap)."""
    need = {"cik", "period", "ticker", "shares"}
    if holdings is None or holdings.empty or not need.issubset(holdings.columns):
        return pd.DataFrame(columns=["date", "ticker"])

    qf = _quarter_features(holdings)
    if qf.empty:
        return pd.DataFrame(columns=["date", "ticker"])

    feats = ["inst_holders", "inst_breadth_chg", "inst_shares_chg", "inst_value_chg",
             "new_buyers", "exiters", "cluster_buying", "new_buyer_ratio",
             "net_options_ratio", "net_options_ratio_chg", "inst_concentration"]
    fields = {f: fundamentals_to_daily(qf, f, trading_index) for f in feats}

    have_shares = shares_out_history is not None and not shares_out_history.empty
    if have_shares:
        # ownership % by SHARES (aggregate 13F shares / shares outstanding)
        inst_sh = fundamentals_to_daily(qf, "inst_shares", trading_index)
        shares = fundamentals_to_daily(shares_out_history, "sharesOutstanding", trading_index)
        if not shares.empty and shares.notna().any().any():
            fields["inst_ownership_pct"] = (
                inst_sh / shares.where(shares > 0)).replace([np.inf, -np.inf], np.nan)

    if have_shares and stock_close is not None and not stock_close.empty:
        # institutional WEIGHT by VALUE and size-scaled net $ flow, via a point-in-time
        # daily market cap (ffilled sharesOutstanding x daily close).
        mcap = daily_market_cap(shares_out_history, stock_close)
        if not mcap.empty:
            mpos = mcap.where(mcap > 0)
            inst_val = fundamentals_to_daily(qf, "inst_value", trading_index)
            iv = (inst_val / mpos).replace([np.inf, -np.inf], np.nan)
            if iv.notna().any().any():
                fields["inst_value_to_mcap"] = iv
            flow = fundamentals_to_daily(qf, "inst_value_flow", trading_index)
            fm = (flow / mpos).replace([np.inf, -np.inf], np.nan)
            if fm.notna().any().any():
                fields["inst_flow_to_mcap"] = fm

    return build_peer_relative_panel(fields, peer_dict)
