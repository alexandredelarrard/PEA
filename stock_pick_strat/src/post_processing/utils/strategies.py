import pandas as pd
import numpy as np

# =========================================================================== #
# PURE PORTFOLIO ENGINE (unchanged, unit-tested)                              #
# =========================================================================== #
def _target_weights_for_day(sig_row, tickers, market_weight, alpha_weight,
                            long_q, short_q, rank_weight):
    w = pd.Series(0.0, index=tickers + ["SPY"])
    w["SPY"] = market_weight
    s = sig_row.dropna()
    if len(s) >= 10:
        hi, lo = s.quantile(1.0 - long_q), s.quantile(short_q)
        longs, shorts = s[s >= hi].index, s[s <= lo].index
        if len(longs) > 0:
            if rank_weight:
                lw = s[longs] - s[longs].min() + 1e-9
                w[longs] = alpha_weight * (lw / lw.sum()).values
            else:
                w[longs] = alpha_weight / len(longs)
        if len(shorts) > 0:
            if rank_weight:
                sw = s[shorts].max() - s[shorts] + 1e-9
                w[shorts] = -alpha_weight * (sw / sw.sum()).values
            else:
                w[shorts] = -alpha_weight / len(shorts)
    return w


def simulate_portfolio(signal, stock_ret, spy_ret, starting_capital=1_000_000,
                       market_weight=0.20, alpha_weight=0.80, long_q=0.10,
                       short_q=0.10, rank_weight=False, fee_bps=1.0,
                       spread_bps=5.0, rebalance_freq=1):
    cost_rate = (fee_bps + spread_bps) / 1e4
    dates = sorted(d for d in signal.index if d in stock_ret.index and d in spy_ret.index)
    tickers = list(stock_ret.columns)
    prev_w = pd.Series(0.0, index=tickers + ["SPY"])
    V = spy_V = starting_capital
    rows = []
    for i in range(len(dates) - 1):
        t, t1 = dates[i], dates[i + 1]
        if i % rebalance_freq == 0:
            w = _target_weights_for_day(signal.loc[t], tickers, market_weight,
                                        alpha_weight, long_q, short_q, rank_weight)
        else:
            w = prev_w.copy()
        turnover = (w - prev_w).abs().sum()
        cost = turnover * cost_rate
        r_stocks = stock_ret.loc[t1, tickers].fillna(0.0)
        r_spy = spy_ret.loc[t1] if np.isfinite(spy_ret.loc[t1]) else 0.0
        gross = float((w[tickers] * r_stocks).sum() + w["SPY"] * r_spy)
        net = gross - cost
        V *= (1.0 + net)
        spy_V *= (1.0 + r_spy)
        rows.append({"date": t1, "gross_ret": gross, "cost": cost, "net_ret": net,
                     "turnover": turnover, "portfolio_value": V, "spy_value": spy_V})
        prev_w = w
    return pd.DataFrame(rows).set_index("date")