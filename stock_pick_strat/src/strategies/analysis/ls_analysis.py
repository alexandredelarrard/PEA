"""
ls_analysis.py  (src/strategies/analysis/ls_analysis.py)
--------------------------------------------------------
Dedicated analysis for the market-neutral L/S sleeve:
  * day-by-day cross-sectional IC (rank correlation of signal vs forward return) + IC_IR;
  * rolling Sharpe + drawdown of the L/S book;
  * MARKET-NEUTRALITY / IDIOSYNCRASY checks — rolling beta of the L/S returns to the S&P
    (should hover ~0) and rolling correlation to ENERGY (should hover ~0), plus the
    cumulative L/S curve overlaid on SP + energy (should look uncorrelated).
Saves two figures + returns a metrics dict.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.strategies.analysis.common import (
    daily_ic, rolling_sharpe, drawdown_series, rolling_beta, rolling_corr)

_ANN = 252.0


def analyze_ls(returns: pd.Series, signal: pd.DataFrame, stock_ret: pd.DataFrame,
               spy_ret: pd.Series, energy_ret: pd.Series | None, out_dir,
               horizon: int = 30, window: int = 126) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    ic = daily_ic(signal, stock_ret, horizon)
    ic_mean = float(ic.mean()) if len(ic) else np.nan
    ic_ir = float(ic.mean() / ic.std()) if len(ic) > 2 and ic.std() > 0 else np.nan
    ic_hit = float((ic > 0).mean()) if len(ic) else np.nan

    r = returns.astype(float).dropna()
    beta_sp = rolling_beta(r, spy_ret.reindex(r.index), window)
    corr_en = (rolling_corr(r, energy_ret.reindex(r.index), window)
               if energy_ret is not None else pd.Series(dtype=float))
    full_beta = float(r.cov(spy_ret.reindex(r.index)) / spy_ret.reindex(r.index).var())
    full_corr_en = float(r.corr(energy_ret.reindex(r.index))) if energy_ret is not None else np.nan

    # --- fig 1: neutrality / idiosyncrasy ---
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    a1.plot(r.index, (1 + r).cumprod(), color="#1f77b4", lw=1.8, label="L/S (cum)")
    sp = spy_ret.reindex(r.index).fillna(0.0)
    a1.plot(r.index, (1 + sp).cumprod(), color="#ff7f0e", lw=1.2, ls="--", label="S&P (cum)")
    if energy_ret is not None:
        en = energy_ret.reindex(r.index).fillna(0.0)
        a1.plot(r.index, (1 + en).cumprod(), color="#d62728", lw=1.0, ls=":", label="Energy (cum)")
    a1.set_ylabel("Growth of $1"); a1.legend(fontsize=9); a1.grid(True, alpha=0.3)
    a1.set_title(f"L/S neutrality — cumulative vs market/energy (full beta_SP={full_beta:+.2f}, "
                 f"corr_energy={full_corr_en:+.2f})")
    a2.axhline(0, color="k", lw=0.8, ls="--")
    a2.plot(beta_sp.index, beta_sp, color="#ff7f0e", lw=1.3, label=f"{window}d beta to S&P (target ~0)")
    if not corr_en.dropna().empty:
        a2.plot(corr_en.index, corr_en, color="#d62728", lw=1.1, label=f"{window}d corr to Energy")
    a2.set_ylim(-1, 1); a2.set_ylabel("beta / corr"); a2.legend(fontsize=9); a2.grid(True, alpha=0.3)
    a2.set_title("Rolling market-beta & energy-correlation (≈0 ⇒ neutral / idiosyncratic)")
    fig.tight_layout(); fig.savefig(out_dir / "ls_neutrality.png", dpi=110); plt.close(fig)

    # --- fig 2: IC + rolling Sharpe/drawdown ---
    fig, (b1, b2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    if len(ic):
        b1.bar(ic.index, ic, color="#90caf9", width=2, alpha=0.5, label="daily IC")
        b1.plot(ic.index, ic.rolling(21, min_periods=5).mean(), color="#1565c0", lw=1.5,
                label="21d mean IC")
    b1.axhline(0, color="red", lw=1, ls="--")
    b1.set_ylabel(f"IC (h={horizon}d)"); b1.legend(fontsize=9); b1.grid(True, alpha=0.3)
    b1.set_title(f"Day-by-day cross-sectional IC — mean {ic_mean:+.3f}, IC_IR {ic_ir:+.2f}, "
                 f"%>0 {ic_hit*100:.0f}%")
    rs = rolling_sharpe(r, window)
    b2.plot(rs.index, rs, color="#2ca02c", lw=1.4, label=f"{window}d rolling Sharpe")
    b2.axhline(0, color="k", lw=0.8, ls="--"); b2.set_ylabel("rolling Sharpe"); b2.legend(loc="upper left", fontsize=9)
    b2b = b2.twinx(); dd = drawdown_series(r) * 100
    b2b.fill_between(dd.index, dd, 0, color="#1f77b4", alpha=0.25); b2b.set_ylabel("drawdown (%)")
    b2.grid(True, alpha=0.3); b2.set_title("Rolling Sharpe (lhs) & drawdown (rhs)")
    fig.tight_layout(); fig.savefig(out_dir / "ls_ic_sharpe.png", dpi=110); plt.close(fig)

    return {"ic_mean": ic_mean, "ic_ir": ic_ir, "ic_pct_positive": ic_hit,
            "full_beta_sp": full_beta, "full_corr_energy": full_corr_en,
            "avg_rolling_beta_sp": float(beta_sp.mean()) if not beta_sp.dropna().empty else np.nan}
