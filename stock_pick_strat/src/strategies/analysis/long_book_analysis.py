"""
long_book_analysis.py  (src/strategies/analysis/long_book_analysis.py)
----------------------------------------------------------------------
Dedicated analysis for the long-book (multi-asset allocation) sleeve: the CORRELATION over time
between each asset class AND the overall (average pairwise) correlation — the diversification
health check (are the classes staying independent, and does that break in stress?). Saves a
figure (rolling per-pair correlations + overall avg) + returns the full-sample matrix.
"""
from __future__ import annotations

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.strategies.analysis.common import rolling_pairwise_corr


def analyze_long_book(asset_rets: pd.DataFrame, out_dir, window: int = 126) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    full_corr = asset_rets.corr()
    pair_corr, avg_corr = rolling_pairwise_corr(asset_rets, window)

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    for name, s in pair_corr.items():
        a1.plot(s.index, s, lw=1.0, label=name)
    a1.axhline(0, color="k", lw=0.8, ls="--"); a1.set_ylim(-1, 1)
    a1.set_ylabel(f"{window}d correlation"); a1.legend(ncol=4, fontsize=8); a1.grid(True, alpha=0.3)
    a1.set_title("Long-book asset-class correlations over time (per pair)")
    a2.plot(avg_corr.index, avg_corr, color="#1f77b4", lw=1.6, label="avg pairwise corr")
    a2.axhline(0, color="k", lw=0.8, ls="--"); a2.set_ylim(-0.5, 1)
    a2.set_ylabel("avg pairwise corr"); a2.legend(fontsize=9); a2.grid(True, alpha=0.3)
    a2.set_title("Overall (average pairwise) correlation — diversification health")
    fig.tight_layout(); fig.savefig(out_dir / "long_book_correlations.png", dpi=110); plt.close(fig)

    full_corr.to_csv(out_dir / "long_book_corr_matrix.csv")
    return {"full_corr": full_corr, "avg_pairwise_corr": float(avg_corr.mean())}
