"""
analyst_features.py  (src/data_aggregate/utils/analyst_features.py)
-------------------------------------------------------------------
Sell-side ANALYST-ESTIMATE signals, expressed peer-relative like every other
feature. Two families that the literature shows carry information:

  * Expectations LEVEL   -> forward earnings/revenue yields and growth the
                            street is pricing in.
  * Expectations CHANGE  -> estimate revisions and rating tilt (the revision /
                            drift effect: stocks whose estimates are being
                            raised tend to keep outperforming).

And the piece the user specifically asked for -- ESTIMATES vs our own INTRINSIC
value:
    est_vs_intrinsic = (intrinsic_per_share - analyst_price_target) / price
  > 0  our DCF sees more value than the street's target  (they are behind)
  < 0  the street is more bullish than the firm's cash flows justify  (fade)

LOOK-AHEAD / DATA NOTE (important):
yfinance only exposes a CURRENT snapshot of estimates -- there is no free
10-year archive. `fetch_analyst_estimates` appends each pull to
`analyst_estimates_history.parquet`, so a genuine point-in-time history only
accrues GOING FORWARD. These features are built strictly point-in-time (each
value forward-filled from its real `as_of`), so they are leak-free -- but until
the history accrues they are populated for only the most recent dates and
cannot yet demonstrate backtest predictive power. Do NOT broadcast today's
snapshot across history: that would be look-ahead.
"""

from __future__ import annotations
import pandas as pd

from src.data_aggregate.utils.factors import fundamentals_to_daily
from src.data_aggregate.utils.intrinsic import intrinsic_value_daily
from src.data_aggregate.utils.fundamental_features import _ratio, build_peer_relative_panel


def _analyst_fields(
    analyst_hist: pd.DataFrame,
    idx: pd.DatetimeIndex,
    close: pd.DataFrame,
    fund_hist: pd.DataFrame | None,
    intrinsic_cfg: dict | None,
) -> dict:
    """Daily wide frames (date x ticker), point-in-time from each `as_of`."""
    F: dict[str, pd.DataFrame] = {}
    intrinsic_cfg = intrinsic_cfg or {}
    price = close.reindex(idx)

    def daily(field):
        return fundamentals_to_daily(analyst_hist, field, idx)

    eps0y, eps1y = daily("eps_est_0y"), daily("eps_est_+1y")
    rev0y, rev1y = daily("rev_est_0y"), daily("rev_est_+1y")
    tgt = daily("price_target_mean")
    up30 = daily("eps_revisions_+1y_upLast30days")
    dn30 = daily("eps_revisions_+1y_downLast30days")
    tr_cur, tr_90 = daily("eps_trend_+1y_current"), daily("eps_trend_+1y_90daysAgo")
    sb, b = daily("rec_strongBuy"), daily("rec_buy")
    h, sl, ss = daily("rec_hold"), daily("rec_sell"), daily("rec_strongSell")

    # ---- expectation LEVELS ----
    if not eps0y.empty:
        F["est_fwd_earnings_yield"] = _ratio(eps0y, price)          # forward E/P
    if not eps1y.empty and not eps0y.empty:
        F["est_eps_growth"] = _ratio(eps1y, eps0y, positive_den=True) - 1.0
    if not rev1y.empty and not rev0y.empty:
        F["est_rev_growth"] = _ratio(rev1y, rev0y, positive_den=True) - 1.0
    if not tgt.empty:
        F["est_target_upside"] = _ratio(tgt, price) - 1.0

    # ---- expectation CHANGES (revisions / drift) ----
    if not up30.empty and not dn30.empty:
        tot = up30.add(dn30, fill_value=0.0)
        F["est_revision_ratio"] = (up30.sub(dn30, fill_value=0.0)) / tot.where(tot > 0)
    if not tr_cur.empty and not tr_90.empty:
        denom = tr_90.abs().where(tr_90.abs() > 0)
        F["est_eps_trend_3m"] = (tr_cur - tr_90) / denom
    if not sb.empty:
        tot = (sb.add(b, fill_value=0).add(h, fill_value=0)
                 .add(sl, fill_value=0).add(ss, fill_value=0))
        score = (2 * sb.fillna(0) + b.fillna(0) - sl.fillna(0) - 2 * ss.fillna(0))
        F["est_rec_score"] = score / tot.where(tot > 0)

    # ---- ESTIMATES vs INTRINSIC value ----
    if fund_hist is not None and not tgt.empty:
        iv = intrinsic_value_daily(fund_hist, close, idx, **intrinsic_cfg)
        ips = iv.get("per_share")
        if ips is not None and not ips.empty:
            cols = ips.columns.intersection(tgt.columns).intersection(price.columns)
            if len(cols):
                p = price[cols].where(price[cols] > 0)
                intrinsic_upside = ips[cols] / p - 1.0
                analyst_upside = tgt[cols] / p - 1.0
                F["est_vs_intrinsic"] = intrinsic_upside - analyst_upside
    return F


def build_analyst_feature_panel(
    analyst_history: pd.DataFrame | None,
    peer_dict: dict,
    trading_index: pd.DatetimeIndex,
    stock_close: pd.DataFrame | None = None,
    fundamentals_history: pd.DataFrame | None = None,
    intrinsic_cfg: dict | None = None,
) -> pd.DataFrame:
    """Long-format analyst-estimate feature panel (`f_est_*_vs_peers`,
    `f_est_*_xs`). Empty if estimates or prices are unavailable."""
    if (analyst_history is None or analyst_history.empty
            or stock_close is None or "as_of" not in analyst_history.columns):
        return pd.DataFrame(columns=["date", "ticker"])

    fields = _analyst_fields(analyst_history, trading_index, stock_close,
                             fundamentals_history, intrinsic_cfg)
    return build_peer_relative_panel(fields, peer_dict)
