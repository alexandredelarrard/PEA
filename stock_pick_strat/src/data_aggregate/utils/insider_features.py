"""
insider_features.py  (src/data_aggregate/utils/insider_features.py)
-------------------------------------------------------------------
Insider-trading signal from the SEC Insider Transactions Data Sets (Forms 3/4/5,
table `insider_transactions`). Distinct from the 13F institutional signal: this is
the issuer's OWN officers / directors trading their stock. The durable alpha is in
OPEN-MARKET PURCHASES (transaction code `P`) — especially CLUSTER buying (several
insiders buying at once) — while sales (`S`) are noisy (insiders sell for liquidity
/ diversification). Non-discretionary codes (grants `A`, option exercise `M`, tax
withholding `F`, gifts `G`) are ignored.

POINT-IN-TIME: a Form 4 must be filed within ~2 business days of the trade, so we
stamp everything on `filing_date` and aggregate on a TRAILING window — a backtest
on day d only ever sees transactions already filed by d.

Features (per stock, trailing window; all peer-relativized downstream):
    insider_net_buy_ratio        (buy$ - sell$) / (buy$ + sell$)   in [-1, 1]
    insider_buy_sell_count_ratio  buy_count / (buy_count + sell_count)
    insider_buy_count            # open-market purchases in the window (cluster buying)
    insider_net_buy_to_mcap      (buy$ - sell$) / market cap        (size-scaled conviction)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_aggregate.utils.factors import daily_market_cap
from src.data_aggregate.utils.fundamental_features import build_peer_relative_panel

_WINDOW_DAYS = 180   # ~6-month trailing window (insider signals work at 3-12m horizons)


def _rolling_flow(txns: pd.DataFrame, mask: pd.Series, calendar: pd.DatetimeIndex,
                  idx: pd.DatetimeIndex, value: bool, window_days: int) -> pd.DataFrame:
    """Trailing `window_days`-calendar-day sum of a transaction flow (value or
    count) per ticker, sampled onto the trading index. Zero-filled between filing
    days so the rolling sum is a true trailing total; point-in-time by construction
    (each day only sees transactions filed on/before it)."""
    d = txns[mask]
    if d.empty:
        return pd.DataFrame(index=idx)
    if value:
        piv = d.groupby(["day", "ticker"])["value_usd"].sum().unstack("ticker")
    else:
        piv = d.groupby(["day", "ticker"]).size().unstack("ticker")
    piv = piv.reindex(calendar).fillna(0.0).rolling(f"{window_days}D").sum()
    return piv.reindex(idx)


def build_insider_feature_panel(
    insider: pd.DataFrame | None,
    peer_dict: dict,
    trading_index: pd.DatetimeIndex,
    shares_out_history: pd.DataFrame | None = None,
    stock_close: pd.DataFrame | None = None,
    window_days: int = _WINDOW_DAYS,
) -> pd.DataFrame:
    """Long-format insider-trading feature panel (`f_<name>_vs_peers`, `f_<name>_xs`).
    Empty if no insider transactions. `shares_out_history` + `stock_close` enable the
    market-cap-scaled net buy (via a point-in-time daily market cap)."""
    need = {"ticker", "filing_date", "transaction_code", "value_usd"}
    if insider is None or insider.empty or not need.issubset(insider.columns):
        return pd.DataFrame(columns=["date", "ticker"])

    t = insider.copy()
    t["ticker"] = t["ticker"].astype(str).str.upper()
    code = t["transaction_code"].astype(str).str.upper().str.strip()
    t["is_buy"] = code == "P"                              # open-market purchase
    t["is_sell"] = code == "S"                             # open-market sale
    t = t[t["is_buy"] | t["is_sell"]]
    t["day"] = pd.to_datetime(t["filing_date"], errors="coerce").dt.normalize()
    t["value_usd"] = pd.to_numeric(t["value_usd"], errors="coerce").fillna(0.0)
    t = t.dropna(subset=["day", "ticker"])
    if t.empty:
        return pd.DataFrame(columns=["date", "ticker"])

    calendar = pd.date_range(t["day"].min(),
                             max(t["day"].max(), trading_index.max()), freq="D")
    buy_val = _rolling_flow(t, t["is_buy"], calendar, trading_index, True, window_days)
    sell_val = _rolling_flow(t, t["is_sell"], calendar, trading_index, True, window_days)
    buy_cnt = _rolling_flow(t, t["is_buy"], calendar, trading_index, False, window_days)
    sell_cnt = _rolling_flow(t, t["is_sell"], calendar, trading_index, False, window_days)

    # align the four flows to a common ticker set + the trading index, zero-filled,
    # so column-mismatched arithmetic (an all-buys name absent from the sells frame)
    # doesn't propagate NaN. Leak-free is preserved: a pre-window row has 0 buy+sell,
    # so the net-buy denominator is 0 -> NaN (no signal before any Form 4 is filed).
    cols = (buy_val.columns.union(sell_val.columns)
            .union(buy_cnt.columns).union(sell_cnt.columns))
    bv, sv, bc, sc = (f.reindex(columns=cols, index=trading_index).fillna(0.0)
                      for f in (buy_val, sell_val, buy_cnt, sell_cnt))

    fields: dict[str, pd.DataFrame] = {}
    denom_v = bv + sv
    fields["insider_net_buy_ratio"] = ((bv - sv) / denom_v.where(denom_v > 0)
                                       ).replace([np.inf, -np.inf], np.nan)
    denom_c = bc + sc
    fields["insider_buy_sell_count_ratio"] = (bc / denom_c.where(denom_c > 0)
                                              ).replace([np.inf, -np.inf], np.nan)
    fields["insider_buy_count"] = bc      # # purchases in the window (0 where none)

    if (shares_out_history is not None and not shares_out_history.empty
            and stock_close is not None and not stock_close.empty):
        mcap = daily_market_cap(shares_out_history, stock_close)
        if not mcap.empty:
            nbm = ((bv - sv) / mcap.where(mcap > 0)).replace([np.inf, -np.inf], np.nan)
            if nbm.notna().any().any():
                fields["insider_net_buy_to_mcap"] = nbm

    return build_peer_relative_panel(fields, peer_dict)
