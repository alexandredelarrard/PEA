"""
super_investors_analysis.py  (src/strategies/analysis/super_investors_analysis.py)
----------------------------------------------------------------------------------
Analysis for the 13F replication sleeve: growth of the mirrored book vs SPY, drawdown of
both, and the invested-vs-cash split over time (which is also the visual proof the sleeve
never levers -- invested weight stays at or below 100%).

Three stacked panels share one x-axis and each carries a SINGLE y-scale: a twin/secondary
axis would let two different units share one set of gridlines, which makes the crossings
between the two series read as meaningful when they are an artifact of the scaling.
"""
from __future__ import annotations

import logging
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.strategies.analysis.common import drawdown_series
from src.strategies.utils.replication import replicate_superinvestors

logger = logging.getLogger(__name__)

# Categorical slots 1-3 of the validated reference palette, in fixed order. These three
# clear the all-pairs colour-vision gates; they are never re-ordered or cycled per chart.
_PORTFOLIO = "#2a78d6"     # slot 1, blue
_BENCHMARK = "#eb6834"     # slot 2, orange
_INVESTED = "#1baf7a"      # slot 3, aqua
_GRID = {"alpha": 0.25, "lw": 0.6}
_ANN = 252.0
_MATERIAL_WEIGHT = 0.005     # a position below 0.5% of the book is not a position
_SINGLE_STOCK_FLAG = 0.25    # flag a book that is one name for a quarter of its days


def analyze_super_investors(returns: pd.Series, spy_ret: pd.Series, weights: pd.DataFrame,
                            out_dir, capital: float = 1_000_000.0,
                            label: str = "13F replication",
                            filename: str = "super_investors_analysis.png",
                            title: str | None = None) -> dict:
    """Write the 3-panel chart and return the headline comparison numbers. `label` / `filename`
    / `title` are parameterized so the same figure serves the pooled cohort book and each
    individual manager's own portfolio."""
    out_dir.mkdir(parents=True, exist_ok=True)
    r = returns.astype(float).dropna()
    sp = spy_ret.reindex(r.index).fillna(0.0)
    eq, eq_sp = (1 + r).cumprod() * capital, (1 + sp).cumprod() * capital
    dd, dd_sp = drawdown_series(r) * 100, drawdown_series(sp) * 100
    # Breadth, not cash. A full weight replication is ~100% invested every day, so a cash panel
    # is a flat line carrying no information -- whereas HOW FEW NAMES the book holds is the
    # single most important caveat on these numbers: `sec13f_hr` keeps only S&P-500-universe
    # holdings, so a manager's non-index positions are invisible and some replicated "books"
    # collapse to one stock (ShawSpring is 100% CVNA then 100% XYZ for most of 2019-2021).
    w = weights.reindex(r.index).fillna(0.0)
    n_holdings = (w.abs() > _MATERIAL_WEIGHT).sum(axis=1)
    top_weight = w.max(axis=1) * 100
    # excess-of-zero Sharpe, matching the per-manager summary table's definition
    sharpe = float(r.mean() / r.std() * np.sqrt(_ANN)) if r.std() > 0 else np.nan
    sharpe_sp = float(sp.mean() / sp.std() * np.sqrt(_ANN)) if sp.std() > 0 else np.nan
    mdd, mdd_sp = float(dd.min()), float(dd_sp.min())

    fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True,
                                     gridspec_kw={"height_ratios": [2, 1, 1]})
    a1.plot(eq.index, eq, color=_PORTFOLIO, lw=2.0, label=label)
    a1.plot(eq_sp.index, eq_sp, color=_BENCHMARK, lw=2.0, label="SPY")
    a1.set_ylabel("portfolio value (€)")
    a1.set_yscale("log")            # 10+ years of compounding: equal % moves must look equal
    a1.legend(loc="upper left", frameon=False, fontsize=10)
    a1.grid(True, **_GRID)
    # Sharpe and max drawdown always ride in the title, on both the pooled and the per-manager
    # figure -- the caller supplies only the identifying headline, so neither can omit them.
    headline = title or (f"Superinvestor 13F replication vs SPY — "
                         f"€{capital:,.0f} invested {r.index[0].date()}, "
                         f"ending €{eq.iloc[-1]:,.0f} vs €{eq_sp.iloc[-1]:,.0f}")
    # a book that is one stock for a big share of its life is not a manager's track record --
    # say so on the figure itself, not only in a column of a CSV nobody opens
    single = float((n_holdings <= 1).mean())
    warn = (f"   ·   ⚠ SINGLE-STOCK {single:.0%} of days" if single >= _SINGLE_STOCK_FLAG else "")
    a1.set_title(f"{headline}\nSharpe {sharpe:.2f} vs SPY {sharpe_sp:.2f}   ·   "
                 f"max drawdown {mdd:.1f}% vs SPY {mdd_sp:.1f}   ·   "
                 f"median {int(n_holdings.median())} positions{warn}", fontsize=11)

    a2.plot(dd.index, dd, color=_PORTFOLIO, lw=1.6, label=label)
    a2.plot(dd_sp.index, dd_sp, color=_BENCHMARK, lw=1.6, label="SPY")
    # mark the single worst day rather than labelling the curve -- one direct label, at the
    # point the reader is actually looking for
    trough = dd.idxmin()
    a2.plot([trough], [mdd], marker="o", ms=6, color=_PORTFOLIO, zorder=5)
    a2.annotate(f"{mdd:.1f}%", xy=(trough, mdd), xytext=(6, 4), textcoords="offset points",
                fontsize=9, color=_PORTFOLIO, fontweight="bold")
    a2.set_ylabel("drawdown (%)")
    a2.legend(loc="lower left", frameon=False, fontsize=9)
    a2.grid(True, **_GRID)

    a3.plot(n_holdings.index, n_holdings, color=_INVESTED, lw=1.6, drawstyle="steps-post",
            label=f"positions held (>{_MATERIAL_WEIGHT:.1%} of book)")
    a3.axhline(1, color="#52514e", lw=1.0, ls="--")
    a3.text(n_holdings.index[int(len(n_holdings) * 0.02)], 1.15,
            "1 = single-stock book", fontsize=8, color="#52514e", va="bottom")
    a3.set_ylim(0, max(2.0, float(n_holdings.max()) * 1.08))
    a3.set_ylabel("number of positions")
    a3.legend(loc="upper left", frameon=False, fontsize=9)
    a3.grid(True, **_GRID)

    for ax in (a1, a2, a3):
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / filename, dpi=110)
    plt.close(fig)

    te = float((r - sp).std() * np.sqrt(_ANN))
    return {"total_return": float(eq.iloc[-1] / capital - 1.0),
            "spy_total_return": float(eq_sp.iloc[-1] / capital - 1.0),
            "sharpe": sharpe, "spy_sharpe": sharpe_sp,
            "max_drawdown_pct": mdd, "spy_max_drawdown_pct": mdd_sp,
            "median_positions": int(n_holdings.median()),
            "pct_days_single_stock": single, "max_name_weight": float(top_weight.max() / 100),
            "beta_spy": float(r.cov(sp) / sp.var()) if sp.var() > 0 else np.nan,
            "tracking_error": te,
            "info_ratio": float((r - sp).mean() * _ANN / te) if te > 0 else np.nan}


def _slug(cik: str, name: str) -> str:
    """Filesystem-safe `cik_name` stem, e.g. `0001067983_warren_buffett_berkshire_hathaway`."""
    clean = re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")[:60]
    return f"{cik}_{clean}" if clean else str(cik)


def analyze_super_investors_by_cik(panel: pd.DataFrame, prices: pd.DataFrame,
                                   spy_ret: pd.Series, roster: dict[str, str], out_dir,
                                   capital: float = 1_000_000.0, fee_bps: float = 2.0,
                                   spread_bps: float = 8.0, execution_lag: int = 1,
                                   seed_min_names: int = 0,
                                   start=None, end=None) -> pd.DataFrame:
    """Replay EACH manager's 13F book as its own standalone portfolio and chart it against SPY.

    `panel` must be the `by_cik=True` panel (one row per cik x ticker x day). Every manager is
    run through the same unlevered mirror as the pooled book, seeded and benchmarked on ITS OWN
    window -- a manager who only starts filing in 2019 is compared against SPY over 2019+, not
    against the full-history SPY, so the excess column is window-matched and comparable across
    managers even though their windows are not.

    `seed_min_names` is much lower than the pooled book's: one manager holds a handful of names,
    not 50. A manager who never reaches even that, or whose names are unpriceable, is skipped
    with a warning rather than aborting the sweep. Returns the summary table, best IR first.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for cik, sub in panel.groupby("cik", sort=False):
        name = roster.get(cik, cik)
        try:
            res = replicate_superinvestors(
                sub.drop(columns=["cik"]), prices, capital=capital, fee_bps=fee_bps,
                spread_bps=spread_bps, start=start, end=end, execution_lag=execution_lag,
                seed_min_names=seed_min_names)
        except (ValueError, KeyError) as e:
            logger.warning("super_investors: skipping %s (%s) -- %s", cik, name, e)
            continue
        r, diag = res["returns"], res["diagnostics"]
        if len(r) < 252:                       # under a year of history: metrics are noise
            logger.info("super_investors: skipping %s (%s) -- only %d trading days",
                        cik, name, len(r))
            continue
        stats = analyze_super_investors(
            r, spy_ret, res["weights"], out_dir, capital=capital, label=name[:40],
            filename=f"cik_{_slug(cik, name)}.png",
            title=(f"{name} — 13F replication vs SPY | €{capital:,.0f} from "
                   f"{r.index[0].date()} → €{diag['final_equity']:,.0f}"))
        ann = float((1 + r).prod() ** (_ANN / len(r)) - 1.0)
        sp = spy_ret.reindex(r.index).fillna(0.0)
        ann_spy = float((1 + sp).prod() ** (_ANN / len(sp)) - 1.0)
        # Sharpe / max-DD come straight from the figure's own numbers, so the table and the
        # chart title can never quote different values for the same manager
        rows.append({"cik": cik, "name": name, "start": r.index[0].date(),
                     "end": r.index[-1].date(), "days": len(r),
                     "n_names_seed": diag["seed_names"],
                     "ann_return": ann, "spy_ann_return": ann_spy, "excess_ann": ann - ann_spy,
                     "sharpe": stats["sharpe"], "spy_sharpe": stats["spy_sharpe"],
                     "max_dd_pct": stats["max_drawdown_pct"],
                     "spy_max_dd_pct": stats["spy_max_drawdown_pct"],
                     # how much of a "track record" this really is (see the universe caveat)
                     "median_positions": stats["median_positions"],
                     "pct_days_single_stock": stats["pct_days_single_stock"],
                     "max_name_weight": stats["max_name_weight"],
                     "beta_spy": stats["beta_spy"],
                     "tracking_error": stats["tracking_error"], "info_ratio": stats["info_ratio"],
                     "final_equity": diag["final_equity"],
                     "max_leverage": diag["max_leverage"]})
    summary = pd.DataFrame(rows)
    if summary.empty:
        logger.warning("super_investors: no manager produced a usable per-cik backtest.")
        return summary
    summary = summary.sort_values("info_ratio", ascending=False).reset_index(drop=True)
    summary.to_csv(out_dir / "per_cik_summary.csv", index=False)
    _plot_cik_leaderboard(summary, out_dir)
    return summary


def _plot_cik_leaderboard(summary: pd.DataFrame, out_dir, top_n: int = 25) -> None:
    """Horizontal bar of annualized EXCESS return vs each manager's own-window SPY.

    Diverging polarity (beat / lagged the benchmark) around a zero baseline, so the two hues
    are the two signs of one measure -- not two categories. Managers are ordered by the value
    being encoded, which is the one ordering a reader can verify from the chart itself."""
    d = summary.dropna(subset=["excess_ann"]).copy()
    d = pd.concat([d.nlargest(top_n // 2 + top_n % 2, "excess_ann"),
                   d.nsmallest(top_n // 2, "excess_ann")]).drop_duplicates("cik")
    d = d.sort_values("excess_ann")
    if d.empty:
        return
    colors = [_BENCHMARK if v < 0 else _INVESTED for v in d["excess_ann"]]
    fig, ax = plt.subplots(figsize=(11, max(5, 0.32 * len(d))))
    labels = [f"{n[:44]}" for n in d["name"]]
    ax.barh(labels, d["excess_ann"] * 100, color=colors, height=0.72)
    ax.axvline(0, color="#52514e", lw=1.0)
    ax.set_xlabel("annualized excess return vs SPY over the manager's own window (pp)")
    ax.set_title(f"Superinvestor 13F replication — per-manager excess vs SPY "
                 f"({len(summary)} managers, best/worst {len(d)} shown)")
    ax.grid(True, axis="x", **_GRID)
    ax.tick_params(axis="y", labelsize=8)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "per_cik_excess_vs_spy.png", dpi=110)
    plt.close(fig)
