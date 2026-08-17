"""
Streamlit PORTFOLIO dashboard.

Run from stock_pick_strat/ with:
    streamlit run app/app.py

Layout:
  * Sidebar — PORTFOLIO-LEVEL levers (sleeve set, blend scheme, global vol target / leverage,
    per-sleeve target vol, window, capital, fees). One "Run" builds the whole book.
  * Main panel — PORTFOLIO results FIRST (KPIs vs SP, per-strategy Sharpe table, sleeve
    correlation matrix, equity curve, dynamic $-allocation, sleeve-correlation evolution).
  * Tabs — one PER STRATEGY: its KPIs + analysis metrics + analysis plots (L/S: IC / Sharpe /
    market-neutrality; long_book: asset-class correlation; trend: crisis-alpha / exposure), so
    you can check how accurate / well-behaved each sleeve is.

The models are assumed pre-trained (StepModelling). L/S is out-of-sample from the model train_end.
"""
from __future__ import annotations

import sys
import os
import json
from pathlib import Path

# Ensure imports resolve from stock_pick_strat/
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np
import pandas as pd
import streamlit as st
from omegaconf import OmegaConf

from src.context import get_config_context
from src.portfolio import StepPortfolio

st.set_page_config(page_title="PEA — Portfolio Dashboard", layout="wide",
                   initial_sidebar_state="expanded")
st.title("PEA — Multi-Strategy Portfolio Dashboard")


@st.cache_resource(show_spinner=False)
def get_context():
    config, context = get_config_context("./configs", use_cache=False, save=True)
    return config, context


base_config, context = get_context()
_PB = base_config.portfolio


def model_train_end() -> str | None:
    """The trained L/S ensemble's train_end (metadata.json) — the L/S sleeve is out-of-sample
    only from this date. The backtest start must equal it for a clean OOS L/S."""
    meta = context.paths["MODELS_DIR"] / "metadata.json"
    if not meta.exists():
        return None
    try:
        return str(json.loads(meta.read_text()).get("train_end"))
    except Exception:
        return None


TRAIN_END = model_train_end()

# model-dependent equity sleeves (need the trained ensemble; OOS from its train_end)
MODEL_SLEEVES = ("ls_equity", "eq_long_only")
ALL_SLEEVES = ["ls_equity", "eq_long_only", "long_book", "trend_cta"]

# friendly per-sleeve blurb (what to look for in its analysis tab)
SLEEVE_INFO = {
    "ls_equity": ("Market-neutral equity L/S",
                  "Check: IC > 0 and stable (predictive), and rolling **beta-to-SP ≈ 0** / "
                  "**corr-to-energy ≈ 0** (market-neutral, idiosyncratic)."),
    "eq_long_only": ("Long-only top-N equity (no shorts)",
                     "Long the model's best-ranked names (top-N, hold-band). Check IC > 0; "
                     "**beta-to-SP ≈ 1** here (it's a long book / smart-beta tilt, retail-viable)."),
    "long_book": ("Multi-asset long book (ERC)",
                  "Check: the asset classes stay lowly/negatively correlated over time "
                  "(diversification holds; watch stress spikes)."),
    "trend_cta": ("Trend / CTA (long-short)",
                  "Check: profits when SP falls (crisis-alpha, beta-to-SP ≈ 0 / negative) and "
                  "positions flip long/short with the trend."),
}


# ---------------------------------------------------------------------------
# Sidebar — portfolio-level levers
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Portfolio levers")
    sleeves = st.multiselect("Strategies (sleeves)", ALL_SLEEVES,
                             default=[str(s) for s in _PB.get("sleeves", ALL_SLEEVES)])
    st.subheader("Blend across sleeves")
    scheme = st.selectbox("Weighting scheme", ["erc", "inverse_vol"],
                          index=0 if str(_PB.get("scheme", "erc")) == "erc" else 1,
                          help="ERC = correlation-aware equal-risk; inverse_vol = risk parity")
    cov_mode = st.selectbox("Sleeve covariance", ["ewma", "std"],
                            index=0 if str(_PB.get("cov_mode", "ewma")) == "ewma" else 1)
    rebalance_freq = st.selectbox("Rebalance freq (days)", [5, 21, 42, 63],
                                  index=[5, 21, 42, 63].index(int(_PB.get("rebalance_freq", 21))))

    st.subheader("Global risk")
    portfolio_vol_target = st.slider("Global vol target (annual)", 0.02, 0.30,
                                     float(_PB.get("portfolio_vol_target", 0.10)), 0.01, format="%.2f")
    max_leverage = st.slider("Global max leverage", 1.0, 3.0, float(_PB.get("max_leverage", 2.0)), 0.25)
    sleeve_target_vol = st.slider("Per-sleeve target vol", 0.02, 0.30,
                                  float(_PB.get("sleeve_target_vol", 0.10)), 0.01, format="%.2f",
                                  help="Reference vol each sleeve is scaled to before blending")

    st.subheader("Window & capital")
    if TRAIN_END:
        st.caption(f"⚙ Model **train-end = {TRAIN_END}** — the L/S sleeve is out-of-sample from this "
                   f"date. Keep Start = this for a clean OOS L/S.")
    else:
        st.error("No trained L/S model found (metadata.json). Train StepModelling first, "
                 "otherwise the `ls_equity` sleeve will be dropped from the blend.")
    # start defaults to the model train-end so L/S is OOS-aligned and always present
    start = st.text_input("Start date (YYYY-MM-DD)", value=TRAIN_END or str(_PB.get("start", "2023-01-01")))
    end = st.text_input("End date (blank = last common)", value="")
    starting_capital = st.number_input("Starting capital ($)", 1, 1_000_000_000,
                                        int(_PB.get("starting_capital", 1_000_000)), 100_000, format="%d")
    fee_bps = st.number_input("Fee (bps)", 0.0, 20.0, float(_PB.get("fee_bps", 2.0)), 0.5)
    spread_bps = st.number_input("Spread (bps)", 0.0, 40.0, float(_PB.get("spread_bps", 8.0)), 0.5)
    plot_analysis = st.checkbox("Generate per-strategy analysis plots", value=True)

    run_btn = st.button("▶  Run Portfolio Backtest", type="primary", use_container_width=True)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
def run_portfolio(params: dict) -> StepPortfolio:
    cfg = OmegaConf.merge(base_config, OmegaConf.create({"portfolio": params}))
    step = StepPortfolio(context=context, config=cfg)
    step.run()
    return step


def ls_model_ready(start: str) -> bool:
    """True when the model-dependent equity sleeves (ls_equity / eq_long_only) can run OOS for
    this backtest: either none is selected, or a model exists trained EXACTLY for this period
    (train_end == backtest start). A missing/misaligned model must NOT be used — block instead."""
    if not any(s in sleeves for s in MODEL_SLEEVES):
        return True
    return TRAIN_END is not None and bool(start) and start == TRAIN_END


def _clear_logs():
    context.log_buffer.seek(0); context.log_buffer.truncate(0)


# ---------------------------------------------------------------------------
# Render — portfolio overview
# ---------------------------------------------------------------------------
def render_overview(step: StepPortfolio):
    m, d = step.metrics, step.daily
    st.header("Portfolio results")
    st.caption(f"Window {d.index.min().date()} → {d.index.max().date()} "
               f"({len(d):,} trading days ≈ {len(d)/252:.1f}y) · scheme "
               f"**{str(step._cfg.get('scheme','erc')).upper()}** · global vol target "
               f"{float(step._cfg.get('portfolio_vol_target',0.10))*100:.0f}% · avg leverage "
               f"{float(step.blended['leverage'].mean()):.2f}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Portfolio Sharpe", f"{m['sharpe']:.2f}", delta=f"{m['sharpe']-m['spy_sharpe']:.2f} vs SP")
    c2.metric("Ann. Return", f"{m['ann_return']*100:.1f}%",
              delta=f"{(m['ann_return']-m['spy_ann_return'])*100:.1f}% vs SP")
    c3.metric("Ann. Volatility", f"{m['ann_vol']*100:.1f}%",
              delta=f"{(m['ann_vol']-m['spy_ann_vol'])*100:.1f}% vs SP", delta_color="inverse")
    c4.metric("Max Drawdown", f"{m['max_drawdown']*100:.1f}%",
              delta=f"{(m['max_drawdown']-m['spy_max_drawdown'])*100:.1f}% vs SP", delta_color="inverse")

    st.subheader("Per-strategy Sharpe vs global portfolio (+ dynamic $ allocation)")
    st.dataframe(step.summary, use_container_width=True)

    tp = getattr(step, "trades_path", None)
    if tp and Path(tp).exists():
        with open(tp, "rb") as fh:
            st.download_button("⬇  Download daily trade blotter (Excel — one sheet per sleeve)",
                               fh.read(), file_name="trades.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    cc1, cc2 = st.columns([1, 1])
    with cc1:
        st.subheader("Sleeve correlation")
        st.dataframe(step.sleeve_corr.style.format("{:.2f}").background_gradient(cmap="RdBu_r", vmin=-1, vmax=1),
                     use_container_width=True)
    with cc2:
        st.subheader("Diversification (avg pairwise sleeve corr)")
        _img(step, "portfolio/analysis/sleeve_correlation_evolution.png")

    st.subheader("Portfolio value vs SP500 (+ standalone sleeves)")
    _img(step, "portfolio/portfolio_vs_sp.png")
    st.subheader("Dynamic capital allocation across strategies")
    _img(step, "portfolio/sleeve_weights.png")


def render_strategy_tabs(step: StepPortfolio):
    st.header("Per-strategy analysis")
    names = list(step.results.keys())
    if not names:
        return
    tabs = st.tabs([SLEEVE_INFO.get(n, (n, ""))[0] for n in names])
    for tab, n in zip(tabs, names):
        with tab:
            res = step.results[n]
            title, blurb = SLEEVE_INFO.get(n, (n, ""))
            st.caption(blurb)
            mm = res.metrics
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Sharpe", f"{mm['sharpe']:.2f}")
            k2.metric("Ann. Return", f"{mm['ann_return']*100:.1f}%")
            k3.metric("Ann. Vol", f"{mm['ann_vol']*100:.1f}%")
            k4.metric("Max Drawdown", f"{mm['max_drawdown']*100:.1f}%")

            an = (res.extra or {}).get("analysis", {})
            if an:
                st.markdown("**Analysis KPIs**")
                fmt = {k: (f"{v:.3f}" if isinstance(v, (int, float)) and np.isfinite(v) else "—")
                       for k, v in an.items() if not isinstance(v, pd.DataFrame)}
                st.dataframe(pd.DataFrame([fmt]), use_container_width=True, hide_index=True)

            adir = context.paths["OUTPUT_DIR"] / n / "analysis"
            pngs = sorted(adir.glob("*.png")) if adir.exists() else []
            if not pngs:
                st.info("No analysis plots (enable 'Generate per-strategy analysis plots' and re-run).")
            for p in pngs:
                st.image(str(p), use_container_width=True)


def _img(step, rel: str):
    p = context.paths["OUTPUT_DIR"] / rel
    if p.exists():
        st.image(str(p), use_container_width=True)
    else:
        st.info(f"missing plot: {rel}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
# The equity model sleeves (ls_equity / eq_long_only) need a model TRAINED FOR THIS PERIOD
# (train_end == backtest start). If it's missing or trained for a different period, we do NOT run
# with the wrong model — we warn and block, so results are a clean out-of-sample test.
_start = start.strip() or "2023-01-01"
_ls_ready = ls_model_ready(_start)
_model_sel = [s for s in MODEL_SLEEVES if s in sleeves]
if _model_sel and not _ls_ready:
    if TRAIN_END is None:
        st.warning(f"⚠ No trained equity model found. Train it for this period first "
                   f"(`train.end_date = {_start}`, then StepModelling), or deselect {_model_sel}. "
                   f"**The backtest will not run with a missing model.**")
    else:
        st.warning(f"⚠ The equity model is trained to **train_end = {TRAIN_END}**, but the backtest "
                   f"starts **{_start}** — not trained for this period. Retrain with "
                   f"`train.end_date = {_start}` (then StepModelling), set Start = {TRAIN_END}, or "
                   f"deselect {_model_sel}. **The backtest will not run with a misaligned model.**")

if run_btn:
    # guard: never run with a missing / misaligned equity model
    if _model_sel and not _ls_ready:
        st.error(f"Backtest blocked: the equity model for {_model_sel} is missing or not trained "
                 f"for this period (see the warning above). Fix the alignment or deselect them.")
        st.stop()
    params = {
        "sleeves": list(sleeves) or ALL_SLEEVES,
        "start": _start,
        "end": end.strip() or None,
        "scheme": scheme, "cov_mode": cov_mode, "rebalance_freq": int(rebalance_freq),
        "portfolio_vol_target": float(portfolio_vol_target), "max_leverage": float(max_leverage),
        "sleeve_target_vol": float(sleeve_target_vol), "starting_capital": int(starting_capital),
        "fee_bps": float(fee_bps), "spread_bps": float(spread_bps), "plot_analysis": bool(plot_analysis),
    }
    _clear_logs()
    try:
        with st.spinner("Running portfolio backtest… (L/S reads the model + scores the cube)"):
            st.session_state["portfolio"] = run_portfolio(params)
        st.session_state["portfolio_logs"] = context.log_buffer.getvalue()
    except Exception as exc:
        st.exception(exc)
        st.stop()

step = st.session_state.get("portfolio")
if step is not None:
    dropped = getattr(step, "dropped_sleeves", []) or []
    if dropped:
        hint = ""
        if any(s in dropped for s in MODEL_SLEEVES):
            hint = (f" — the equity model sleeves need model PREDICTIONS and are out-of-sample from "
                    f"the model train-end ({TRAIN_END}); ensure the models exist and the window "
                    f"overlaps [train-end, end]. Retrain if the start doesn't match the train-end.")
        st.error(f"⚠ Sleeve(s) dropped (no data in window): **{dropped}**{hint}")
    render_overview(step)
    render_strategy_tabs(step)
    with st.expander("Show run logs"):
        st.code(st.session_state.get("portfolio_logs", ""), language="log")
else:
    st.info("Set the portfolio levers in the sidebar, then click **▶ Run Portfolio Backtest**. "
            "Portfolio KPIs + correlations show first; per-strategy analysis tabs appear below.")
