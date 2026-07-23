"""
analysis.py  (src/portfolio/analysis.py)
----------------------------------------
Portfolio-level analysis: the CORRELATION EVOLUTION of the strategy sleeves — rolling
correlation for each sleeve PAIR and the overall (average pairwise) correlation over time.
This is the diversification health check for the blend: are the sleeves staying independent
(and does that break in stress)? Saves a figure + returns the full-sample sleeve matrix.
"""
from __future__ import annotations

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.strategies.analysis.common import rolling_pairwise_corr

_SLEEVE_COLORS = {"ls_equity-long_book": "#1f77b4", "ls_equity-trend_cta": "#9467bd",
                  "long_book-trend_cta": "#2ca02c"}


def analyze_portfolio(sleeve_rets: pd.DataFrame, out_dir, window: int = 126) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    full_corr = sleeve_rets.corr()
    pair_corr, avg_corr = rolling_pairwise_corr(sleeve_rets, window)

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    for name, s in pair_corr.items():
        a1.plot(s.index, s, lw=1.3, label=name, color=_SLEEVE_COLORS.get(name))
    a1.axhline(0, color="k", lw=0.8, ls="--"); a1.set_ylim(-1, 1)
    a1.set_ylabel(f"{window}d correlation"); a1.legend(fontsize=9); a1.grid(True, alpha=0.3)
    a1.set_title("Sleeve correlation over time (per pair) — lower ⇒ more diversified blend")
    a2.plot(avg_corr.index, avg_corr, color="#1f77b4", lw=1.6, label="avg pairwise sleeve corr")
    a2.axhline(0, color="k", lw=0.8, ls="--"); a2.set_ylim(-1, 1)
    a2.set_ylabel("avg pairwise corr"); a2.legend(fontsize=9); a2.grid(True, alpha=0.3)
    a2.set_title("Overall (average pairwise) sleeve correlation")
    fig.tight_layout(); fig.savefig(out_dir / "sleeve_correlation_evolution.png", dpi=110); plt.close(fig)

    full_corr.to_csv(out_dir / "sleeve_corr_matrix.csv")
    return {"full_corr": full_corr, "avg_pairwise_corr": float(avg_corr.mean())}
