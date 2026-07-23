"""
accuracy.py  (src/post_processing/utils/accuracy.py)
----------------------------------------------------
Horizon-forward, cross-sectional directional accuracy of the blended signal, used
by the Streamlit app. The signal targets 30/60/90-day moves, so accuracy is scored
over the FORECAST horizon (not next-day noise) and CROSS-SECTIONALLY (did the name
out/under-perform its peers in the predicted direction).

NaN-tolerance is the whole point of `forward_return`: daily returns (pct_change)
carry a leading NaN and scattered holes (suspensions, missing closes, recent IPOs).
Requiring the FULL horizon window to be non-NaN (rolling min_periods == horizon)
blanks almost every name's forward return, which drops it, and once < 10 names
remain the ENTIRE date is dropped from the table -- the "chart shows only a few
days" bug. We instead require a fraction of the window (`min_frac`) so a few gaps
no longer blank a name, while a genuinely data-poor name (too few observations)
stays NaN and is correctly skipped.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def forward_return(daily_ret: pd.DataFrame | pd.Series, horizon: int,
                   min_frac: float = 0.6):
    """NaN-tolerant compounded forward return over t+1..t+horizon.

    Sums the AVAILABLE daily log-returns in the forward window (a missing day
    contributes nothing = no-change) as long as at least ``min_frac * horizon``
    observations are present; otherwise NaN. Equivalent to price[t+h]/price[t]-1
    when the window has no gaps. Only names with too little data in the window --
    and the genuine tail (fewer than the required observations ahead) -- stay NaN.
    """
    safe = daily_ret.clip(lower=-0.999999)
    logr = np.log1p(safe)
    min_periods = max(1, int(round(horizon * min_frac)))
    fwd_log = (logr[::-1]
               .rolling(horizon, min_periods=min_periods).sum()[::-1]
               .shift(-1))
    return np.expm1(fwd_log)


def compute_horizon_accuracy(bt, horizon: int, active_thresh: float = 0.1,
                             min_frac: float = 0.6) -> pd.DataFrame:
    """Per signal date, directional accuracy of the signal over the forecast
    horizon, measured CROSS-SECTIONALLY (relative to the universe).

    Columns: hit_rate_% (sign match vs the peer-relative forward return),
    correct/total active picks, long_short_fwd_% (realized horizon return of
    predicted-longs minus predicted-shorts), spy_fwd_% (market's horizon return).
    """
    signal: pd.DataFrame = bt.signal                      # date x ticker, combined z
    fwd = forward_return(bt.stock_ret, horizon, min_frac)  # date x ticker
    spy_fwd = forward_return(bt.spy_ret, horizon, min_frac)

    rows = []
    for date in signal.index.sort_values():
        if date not in fwd.index:
            continue
        s = signal.loc[date].dropna()
        f = fwd.loc[date].reindex(s.index).dropna()
        common = s.index.intersection(f.index)
        if len(common) < 10:
            continue
        s = s[common]
        f = f[common]
        rel = f - f.mean()                    # cross-sectional (market-neutral) fwd

        mask = s.abs() > active_thresh        # evaluate only conviction names
        n = int(mask.sum())
        if n == 0:
            continue
        correct = int(((s[mask] > 0) == (rel[mask] > 0)).sum())

        longs = f[mask & (s > 0)]
        shorts = f[mask & (s < 0)]
        ls_spread = (longs.mean() - shorts.mean()) if len(longs) and len(shorts) else np.nan
        spy_v = spy_fwd.get(date, np.nan)
        if hasattr(spy_v, "iloc"):
            spy_v = float(spy_v.iloc[0])

        rows.append({
            "date": date,
            "hit_rate_%": round(correct / n * 100, 1),
            "correct_picks": correct,
            "total_active_picks": n,
            "long_short_fwd_%": round(ls_spread * 100, 3) if np.isfinite(ls_spread) else np.nan,
            "spy_fwd_%": round(spy_v * 100, 3) if np.isfinite(spy_v) else np.nan,
        })

    return pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame()


def horizon_accuracy_summary(bt, horizons: list[int]) -> pd.DataFrame:
    """Compact per-horizon summary: avg hit rate, share of dates with a positive
    long/short spread, and the mean realized long/short spread."""
    out = []
    for h in horizons:
        acc = compute_horizon_accuracy(bt, h)
        if acc.empty:
            out.append({"horizon": h, "avg_hit_rate_%": np.nan,
                        "pct_dates_positive_%": np.nan, "avg_long_short_%": np.nan,
                        "n_dates": 0})
            continue
        spread = acc["long_short_fwd_%"].dropna()
        out.append({
            "horizon": h,
            "avg_hit_rate_%": round(acc["hit_rate_%"].mean(), 2),
            "pct_dates_positive_%": round((spread > 0).mean() * 100, 1) if len(spread) else np.nan,
            "avg_long_short_%": round(spread.mean(), 3) if len(spread) else np.nan,
            "n_dates": len(acc),
        })
    return pd.DataFrame(out).set_index("horizon")
