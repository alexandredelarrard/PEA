"""
trend_analysis.py  (src/strategies/analysis/trend_analysis.py)
--------------------------------------------------------------
Dedicated analysis for the trend / CTA sleeve: rolling Sharpe + drawdown, the net long/short
exposure per asset over time (is it actually flipping direction?), and the CRISIS-ALPHA check —
cumulative trend vs S&P + rolling beta to the S&P (should be ~0 and turn negative in sell-offs).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.strategies.analysis.common import rolling_sharpe, drawdown_series, rolling_beta


def analyze_trend(returns: pd.Series, positions: pd.DataFrame | None, spy_ret: pd.Series,
                  out_dir, window: int = 126) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    r = returns.astype(float).dropna()
    sp = spy_ret.reindex(r.index).fillna(0.0)
    beta_sp = rolling_beta(r, sp, window)
    full_beta = float(r.cov(sp) / sp.var()) if sp.var() > 0 else np.nan

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    a1, a2, a3 = axes
    a1.plot(r.index, (1 + r).cumprod(), color="#d62728", lw=1.8, label="Trend/CTA (cum)")
    a1.plot(r.index, (1 + sp).cumprod(), color="#ff7f0e", lw=1.2, ls="--", label="S&P (cum)")
    a1.set_ylabel("Growth of $1"); a1.legend(fontsize=9); a1.grid(True, alpha=0.3)
    a1.set_title(f"Trend crisis-alpha — cumulative vs S&P (full beta_SP={full_beta:+.2f}; "
                 "profits when SP falls ⇒ diversifier)")
    if positions is not None and not positions.empty:
        for c in positions.columns:
            a2.plot(positions.index, positions[c], lw=1.0, label=c)
        a2.axhline(0, color="k", lw=0.8, ls="--")
    a2.set_ylabel("net position (long/short)"); a2.legend(ncol=5, fontsize=8); a2.grid(True, alpha=0.3)
    a2.set_title("Net long/short exposure per asset (flips with the trend)")
    rs = rolling_sharpe(r, window)
    a3.plot(rs.index, rs, color="#2ca02c", lw=1.4, label=f"{window}d rolling Sharpe")
    a3.axhline(0, color="k", lw=0.8, ls="--"); a3.set_ylabel("rolling Sharpe"); a3.legend(loc="upper left", fontsize=9)
    a3b = a3.twinx(); dd = drawdown_series(r) * 100
    a3b.fill_between(dd.index, dd, 0, color="#d62728", alpha=0.2); a3b.set_ylabel("drawdown (%)")
    a3.grid(True, alpha=0.3); a3.set_title("Rolling Sharpe (lhs) & drawdown (rhs)")
    fig.tight_layout(); fig.savefig(out_dir / "trend_analysis.png", dpi=110); plt.close(fig)
    return {"full_beta_sp": full_beta,
            "avg_rolling_beta_sp": float(beta_sp.mean()) if not beta_sp.dropna().empty else np.nan}
