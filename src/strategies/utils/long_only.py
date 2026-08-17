"""
long_only.py  (src/strategies/utils/long_only.py)
-------------------------------------------------
Long-only TOP-QUANTILE (rank) equity book: hold the best-ranked `top_n` names by the model
signal, weighted by inverse-vol / ERC / equal, rebalanced daily. A HOLD-BAND (buffer) keeps a
held name until it drops out of the top `top_n · buffer_mult` — so names only turn over when they
genuinely leave the top, cutting the daily churn. Fully invested long (weights sum to 1, no
shorts) so the book carries full market beta (~1) — it is the long leg of the L/S, retail-viable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategies.utils.strategies_opt import shrunk_idio_cov
from src.utils.risk_parity import erc_weights

_ANN = 252.0


def _weights(sel: list[str], t, stock_ret: pd.DataFrame, vol: pd.DataFrame,
             weighting: str, vol_window: int, cov_shrink: float) -> pd.Series:
    """Long-only weights over the selected names (sum to 1): inverse_vol | erc | equal."""
    n = len(sel)
    if weighting == "equal" or n == 1:
        return pd.Series(np.ones(n) / n, index=sel)
    if weighting == "erc":
        win = stock_ret.loc[:t, sel].tail(vol_window)
        cov = shrunk_idio_cov(win.to_numpy(float), win.var().to_numpy(float), cov_shrink)
        return pd.Series(erc_weights(cov), index=sel)
    v = vol.loc[t, sel].to_numpy(float)                    # inverse-vol (default)
    inv = np.where(v > 0, 1.0 / v, 0.0)
    w = inv / inv.sum() if inv.sum() > 0 else np.ones(n) / n
    return pd.Series(w, index=sel)


def long_only_book(signal: pd.DataFrame, stock_ret: pd.DataFrame, capital: float, *,
                   top_n: int = 50, buffer_mult: float = 2.0, weighting: str = "inverse_vol",
                   vol_window: int = 63, rebalance_freq: int = 1, fee_bps: float = 1.0,
                   spread_bps: float = 5.0, cov_shrink: float = 0.5, min_names: int = 10) -> dict:
    """Daily long-only top-N book. Returns dict: net_ret / gross_ret / turnover (Series),
    weights (date x ticker held-weight panel), n_holdings (Series)."""
    vol = stock_ret.rolling(vol_window, min_periods=max(20, vol_window // 2)).std()
    dates = sorted(d for d in signal.index if d in stock_ret.index)
    tickers = list(stock_ret.columns)
    cost_rate = (fee_bps + spread_bps) / 1e4
    exit_rank = max(int(top_n * float(buffer_mult)), top_n)

    prev_w = pd.Series(0.0, index=tickers)
    held: set[str] = set()
    target = pd.Series(0.0, index=tickers)
    rows, weights_hist = [], {}
    for i in range(len(dates) - 1):
        t, t1 = dates[i], dates[i + 1]
        if i % max(1, rebalance_freq) == 0:
            s = signal.loc[t].dropna()
            cand = [tk for tk in s.index if tk in vol.columns
                    and np.isfinite(vol.loc[t, tk]) and vol.loc[t, tk] > 0]
            if len(cand) >= min_names:
                ranked = s[cand].sort_values(ascending=False)          # best signal first
                rank = {tk: r for r, tk in enumerate(ranked.index)}
                keep = [tk for tk in held if rank.get(tk, 10 ** 9) < exit_rank]   # hold-band buffer
                sel = list(dict.fromkeys(list(ranked.index[:top_n]) + keep))
                sel = sorted(sel, key=lambda tk: rank[tk])[:exit_rank]            # cap to the buffer size
                held = set(sel)
                w = _weights(sel, t, stock_ret, vol, weighting, vol_window, cov_shrink)
                target = pd.Series(0.0, index=tickers); target[sel] = w.values

        turnover = float((target - prev_w).abs().sum())
        r = stock_ret.loc[t1].reindex(tickers).fillna(0.0)
        gross = float((target * r).sum())                              # held from t into t1
        net = gross - turnover * cost_rate
        weights_hist[t1] = target.copy()
        rows.append({"date": t1, "gross_ret": gross, "net_ret": net, "turnover": turnover,
                     "n_holdings": int((target != 0).sum())})
        prev_w = target

    out = pd.DataFrame(rows).set_index("date")
    weights = pd.DataFrame(weights_hist).T
    weights = weights.loc[:, (weights != 0).any(axis=0)]               # keep only ever-held names
    return {"net_ret": out["net_ret"], "gross_ret": out["gross_ret"], "turnover": out["turnover"],
            "n_holdings": out["n_holdings"], "weights": weights}
