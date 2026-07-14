"""
Streamlit backtest dashboard.

Run from stock_pick_strat/ with:
    streamlit run app/app.py

Behaviour:
  * Sidebar exposes every backtest parameter.
  * Backtest start date is pinned to modellling.yml `train.end_date` (the model
    train-end), so the backtest is always strictly out-of-sample.
  * If trained models already exist AND their stored train_end matches the current
    config, they are reused. Otherwise StepModelling is run first and the central
    panel shows a "models are being retrained" banner with live logs.
  * Live pipeline logs are streamed in the central panel while the backtest runs,
    then replaced by the results once computation finishes.
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
from omegaconf import OmegaConf

from src.context import get_config_context
from src.modelling.step_modelling import StepModelling
from src.post_processing.step_backtest import StepBacktest


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="PEA Backtest Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("PEA — Backtest Dashboard")


# ---------------------------------------------------------------------------
# Context (built once; log buffer + logging handlers must not accumulate)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_context():
    """Load base config + Context exactly once per Streamlit process."""
    config, context = get_config_context("./configs", use_cache=False, save=True)
    return config, context


# ---------------------------------------------------------------------------
# Sidebar — backtest parameters
# ---------------------------------------------------------------------------
base_config, context = get_context()
train_end = str(base_config.train.end_date)

st.sidebar.header("Backtest Parameters")
st.sidebar.caption(f"Backtest start pinned to model train-end: **{train_end}**")

with st.sidebar:
    st.subheader("Capital & Window")
    starting_capital = st.number_input(
        "Starting Capital ($)", min_value=10_000, max_value=100_000_000,
        value=1_000_000, step=100_000, format="%d"
    )
    end_date = st.text_input(
        "End Date (YYYY-MM-DD, blank = max)", value="",
        help="Leave blank to use the last available date in the cube"
    )

    st.subheader("Portfolio Construction")
    market_weight = st.slider(
        "Market Weight (SPY sleeve)", 0.0, 1.0, 0.5, 0.05,
        help="Fraction of portfolio allocated to SPY buy-and-hold"
    )
    target_ann_vol = st.slider(
        "Target Annual Vol (alpha sleeve)", 0.01, 0.40, 0.10, 0.01,
        format="%.2f"
    )
    beta_neutral = st.checkbox("Beta-Neutral Alpha Sleeve", value=True)
    pos_cap = st.slider(
        "Position Cap |w|", 0.01, 0.30, 0.08, 0.01, format="%.2f",
        help="Max absolute weight per stock in alpha sleeve"
    )
    gross_cap = st.slider(
        "Gross Cap (sum|w|)", 0.5, 6.0, 3.0, 0.25, format="%.2f",
        help="Max gross leverage for alpha sleeve"
    )

    st.subheader("Turnover Control")
    step = st.slider(
        "Trade Step (partial fill fraction)", 0.05, 1.0, 0.4, 0.05,
        help="Lower = slower turnover = cheaper"
    )
    no_trade_band = st.slider(
        "No-Trade Band", 0.0, 0.05, 0.0, 0.005, format="%.3f"
    )
    rebalance_freq = st.selectbox(
        "Rebalance Frequency (days)", [1, 5, 21, 42, 63], index=4
    )

    st.subheader("Risk Windows")
    beta_window = st.selectbox("Beta Window (days)", [21, 42, 63, 126, 252], index=2)
    vol_window = st.selectbox("Vol Window (days)", [21, 42, 63, 126, 252], index=2)

    st.subheader("Costs & Blending")
    fee_bps = st.number_input("Fee (bps)", min_value=0.0, max_value=20.0, value=1.0, step=0.5)
    spread_bps = st.number_input("Spread (bps)", min_value=0.0, max_value=30.0, value=5.0, step=0.5)
    blend = st.selectbox("Horizon Blend", ["ir", "equal"], index=0)
    risk_free_rate = st.slider("Risk-Free Rate (annual)", 0.0, 0.10, 0.03, 0.005, format="%.3f")

    force_retrain = st.checkbox(
        "Force retrain models", value=False,
        help="Retrain even if up-to-date models already exist"
    )

    run_btn = st.button("▶  Run Backtest", type="primary", use_container_width=True)


# ---------------------------------------------------------------------------
# Retrain detection + log streaming helpers
# ---------------------------------------------------------------------------
def needs_retrain(ctx, expected_train_end: str) -> tuple[bool, str]:
    """
    Retrain when: no metadata, no model files, or the stored train_end differs
    from the current modellling.yml train.end_date.
    """
    models_dir = ctx.paths["MODELS_DIR"]
    meta_path = models_dir / "metadata.json"
    if not meta_path.exists():
        return True, "no trained models found"
    try:
        meta = json.loads(meta_path.read_text())
    except Exception:
        return True, "model metadata unreadable"
    stored = str(meta.get("train_end", ""))
    if stored != expected_train_end:
        return True, f"train_end changed ({stored or 'unset'} -> {expected_train_end})"
    horizons = meta.get("horizons", [])
    model_types = meta.get("model_types", [])
    for h in horizons:
        for kind in model_types:
            ext = "txt" if kind == "lightgbm" else "pkl"
            if not (models_dir / f"model_h{h}_{kind}.{ext}").exists():
                return True, f"missing model file for h{h}/{kind}"
    return False, f"up-to-date (train_end={stored})"


def _clear_log_buffer(ctx):
    ctx.log_buffer.seek(0)
    ctx.log_buffer.truncate(0)


def _show_progress(placeholder, ctx, status_msg: str):
    """Render the current status + tail of the log buffer into the placeholder."""
    logs = ctx.log_buffer.getvalue()
    tail = "\n".join(logs.splitlines()[-60:]) or "(waiting…)"
    with placeholder.container():
        st.info(status_msg)
        st.code(tail, language="log")


def _run_phase(steps, placeholder, ctx, phase: str):
    """Run an ordered list of (label, callable), streaming logs after each step."""
    n = len(steps)
    for i, (label, fn) in enumerate(steps, start=1):
        _show_progress(placeholder, ctx, f"{phase} — running: {label}  ({i}/{n})")
        fn()
        _show_progress(placeholder, ctx, f"{phase} — done: {label}  ({i}/{n})")


# ---------------------------------------------------------------------------
# Build overridden config + run the full pipeline
# ---------------------------------------------------------------------------
def build_run_config(overrides: dict):
    patch = OmegaConf.create({"backtest": overrides})
    return OmegaConf.merge(base_config, patch)


def run_pipeline(params: dict, do_retrain: bool, placeholder):
    config = build_run_config(params)

    # ---- optional retrain -------------------------------------------------
    if do_retrain:
        model = StepModelling(context=context, config=config)
        _run_phase(
            [
                ("load cube", model.load_cube),
                ("build panels", model.build_panels),
                ("cross-validate horizons", model.cross_validate_all_horizons),
                ("train final models", model.train_final_models),
                ("save models", model.save_models),
                ("blend & generate signal", model.blend_and_generate_signal),
                ("feature importance", model.log_feature_importance),
                ("diagnostics", model.save_diagnostics),
                ("save outputs", model.save_outputs),
            ],
            placeholder, context, phase="Retraining models",
        )

    # ---- backtest ---------------------------------------------------------
    bt = StepBacktest(context=context, config=config)
    bt.load_models()
    # Pin the backtest start to the model train-end (out-of-sample guarantee)
    bt.backtest_start = pd.Timestamp(train_end)
    bt._log.info("Backtest start pinned to model train-end: %s", train_end)

    _run_phase(
        [
            ("load cube & returns", bt.load_cube_and_returns),
            ("predict & blend signal", bt.predict_and_blend),
            ("simulate portfolio", bt.simulate),
        ],
        placeholder, context, phase="Backtest",
    )
    return bt


# ---------------------------------------------------------------------------
# Accuracy helper — per-day hit rate
# ---------------------------------------------------------------------------
def compute_daily_accuracy(bt: StepBacktest) -> pd.DataFrame:
    """
    For each day: fraction of alpha positions where sign(signal) == sign(next-day return).
    """
    signal: pd.DataFrame = bt.signal          # date x ticker, z-score
    stock_ret: pd.DataFrame = bt.stock_ret    # date x ticker, daily return

    rows = []
    dates = signal.index.sort_values()

    for i, date in enumerate(dates):
        if i + 1 >= len(dates):
            break
        next_date = dates[i + 1]
        if next_date not in stock_ret.index:
            continue

        sig_row = signal.loc[date].dropna()
        ret_row = stock_ret.loc[next_date].reindex(sig_row.index).dropna()

        common = sig_row.index.intersection(ret_row.index)
        if common.empty:
            continue

        s = sig_row[common]
        r = ret_row[common]

        # only evaluate where signal is non-trivial (|z| > 0.1 to avoid noise)
        mask = s.abs() > 0.1
        if mask.sum() == 0:
            continue

        correct = ((s[mask] > 0) == (r[mask] > 0)).sum()
        total = mask.sum()
        hit_rate = correct / total

        # portfolio outperformed SPY that day?
        daily_row = bt.daily.loc[bt.daily.index == next_date]
        strat_ret = float(daily_row["net_ret"].iloc[0]) if not daily_row.empty else np.nan
        spy_ret_val = bt.spy_ret.get(next_date, np.nan)
        if hasattr(spy_ret_val, "iloc"):
            spy_ret_val = float(spy_ret_val.iloc[0])

        rows.append({
            "date": next_date,
            "hit_rate_%": round(hit_rate * 100, 1),
            "correct_picks": int(correct),
            "total_active_picks": int(total),
            "strategy_return_%": round(strat_ret * 100, 3) if not np.isnan(strat_ret) else np.nan,
            "spy_return_%": round(spy_ret_val * 100, 3) if not np.isnan(spy_ret_val) else np.nan,
            "outperformed_spy": (strat_ret > spy_ret_val) if not (np.isnan(strat_ret) or np.isnan(spy_ret_val)) else None,
        })

    return pd.DataFrame(rows).set_index("date")


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def plot_equity(daily: pd.DataFrame) -> plt.Figure:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1]})

    ax1.plot(daily.index, daily["portfolio_value"] / daily["portfolio_value"].iloc[0],
             label="Strategy", linewidth=1.8, color="#1f77b4")
    ax1.plot(daily.index, daily["spy_value"] / daily["spy_value"].iloc[0],
             label="SPY", linewidth=1.5, color="#ff7f0e", linestyle="--")
    ax1.set_ylabel("Growth of $1")
    ax1.legend()
    ax1.set_title("Portfolio vs SPY — Equity Curve")
    ax1.grid(True, alpha=0.3)

    # drawdown
    port_eq = daily["portfolio_value"].to_numpy()
    peak = np.maximum.accumulate(port_eq)
    dd = (port_eq - peak) / peak * 100
    spy_eq = daily["spy_value"].to_numpy()
    spy_peak = np.maximum.accumulate(spy_eq)
    spy_dd = (spy_eq - spy_peak) / spy_peak * 100

    ax2.fill_between(daily.index, dd, 0, alpha=0.4, color="#1f77b4", label="Strategy DD")
    ax2.plot(daily.index, spy_dd, color="#ff7f0e", linewidth=1.0, linestyle="--", label="SPY DD")
    ax2.set_ylabel("Drawdown (%)")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_hit_rate(acc: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 3))
    rolling = acc["hit_rate_%"].rolling(21, min_periods=5).mean()
    ax.bar(acc.index, acc["hit_rate_%"], color="#90caf9", alpha=0.5, width=1, label="Daily")
    ax.plot(rolling.index, rolling, color="#1565c0", linewidth=1.5, label="21-day MA")
    ax.axhline(50, color="red", linewidth=1, linestyle="--", label="50% (random)")
    ax.set_ylabel("Hit Rate (%)")
    ax.set_title("Daily Directional Accuracy — Signal vs Next-Day Return")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def render_results(bt: StepBacktest):
    daily = bt.daily
    metrics = bt.metrics

    # ---- Metrics KPI row ----
    st.subheader("Performance Summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Return (Strategy)",
                f"{metrics['total_return']*100:.1f}%",
                delta=f"{(metrics['total_return'] - metrics['spy_total_return'])*100:.1f}% vs SPY")
    col2.metric("Ann. Return",
                f"{metrics['ann_return']*100:.1f}%",
                delta=f"{(metrics['ann_return'] - metrics['spy_ann_return'])*100:.1f}% vs SPY")
    col3.metric("Sharpe Ratio",
                f"{metrics['sharpe']:.2f}",
                delta=f"{metrics['sharpe'] - metrics['spy_sharpe']:.2f} vs SPY")
    col4.metric("Max Drawdown",
                f"{metrics['max_drawdown']*100:.1f}%",
                delta=f"{(metrics['max_drawdown'] - metrics['spy_max_drawdown'])*100:.1f}% vs SPY",
                delta_color="inverse")

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Ann. Volatility", f"{metrics['ann_vol']*100:.1f}%")
    col6.metric("SPY Total Return", f"{metrics['spy_total_return']*100:.1f}%")
    col7.metric("Avg Daily Turnover", f"{metrics['avg_daily_turnover']:.3f}")
    col8.metric("Avg Daily Cost", f"{metrics['avg_daily_cost']*100:.4f}%")

    # ---- Equity curve ----
    st.subheader("Equity Curve vs SPY")
    fig_eq = plot_equity(daily)
    st.pyplot(fig_eq)
    plt.close(fig_eq)

    # ---- Daily accuracy ----
    st.subheader("Daily Prediction Accuracy")
    with st.spinner("Computing per-day hit rates…"):
        acc = compute_daily_accuracy(bt)

    if acc.empty:
        st.warning("Could not compute daily accuracy — no overlapping signal/return data.")
        return

    avg_hit = acc["hit_rate_%"].mean()
    pct_beat = (acc["outperformed_spy"].dropna().sum()
                / acc["outperformed_spy"].dropna().count() * 100)

    acol1, acol2, acol3 = st.columns(3)
    acol1.metric("Avg Daily Hit Rate", f"{avg_hit:.1f}%",
                 help="Fraction of active signal positions with correct direction next day")
    acol2.metric("Days Outperforming SPY", f"{pct_beat:.1f}%")
    acol3.metric("Total Evaluated Days", f"{len(acc):,}")

    fig_hr = plot_hit_rate(acc)
    st.pyplot(fig_hr)
    plt.close(fig_hr)

    st.subheader("Daily Detail Table")

    def _colour_hit(val):
        if isinstance(val, float):
            if val >= 60:
                return "background-color: #c8e6c9"
            elif val < 45:
                return "background-color: #ffcdd2"
        return ""

    def _colour_out(val):
        if val is True:
            return "background-color: #c8e6c9"
        elif val is False:
            return "background-color: #ffcdd2"
        return ""

    display = acc.copy()
    display.index = display.index.strftime("%Y-%m-%d")

    styled = (
        display.style
        .applymap(_colour_hit, subset=["hit_rate_%"])
        .applymap(_colour_out, subset=["outperformed_spy"])
        .format({
            "hit_rate_%": "{:.1f}",
            "strategy_return_%": "{:+.3f}",
            "spy_return_%": "{:+.3f}",
        }, na_rep="—")
    )
    st.dataframe(styled, height=420, use_container_width=True)

    csv = acc.reset_index().to_csv(index=False)
    st.download_button(
        "Download daily accuracy CSV",
        data=csv,
        file_name="backtest_daily_accuracy.csv",
        mime="text/csv",
    )


# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------
if run_btn:
    params = {
        "end": end_date.strip() if end_date.strip() else None,
        "starting_capital": int(starting_capital),
        "market_weight": float(market_weight),
        "target_ann_vol": float(target_ann_vol),
        "beta_neutral": bool(beta_neutral),
        "pos_cap": float(pos_cap),
        "gross_cap": float(gross_cap),
        "step": float(step),
        "no_trade_band": float(no_trade_band),
        "rebalance_freq": int(rebalance_freq),
        "beta_window": int(beta_window),
        "vol_window": int(vol_window),
        "fee_bps": float(fee_bps),
        "spread_bps": float(spread_bps),
        "blend": blend,
        "risk_free_rate": float(risk_free_rate),
    }

    retrain_flag, reason = needs_retrain(context, train_end)
    do_retrain = bool(force_retrain or retrain_flag)

    # banner: retrain vs reuse
    if do_retrain:
        st.warning(f"🔄 Models are being retrained ({'forced' if force_retrain else reason}). "
                   f"Backtest results will appear once training completes.")
    else:
        st.success(f"✓ Using existing trained models — {reason}. No retraining needed.")

    _clear_log_buffer(context)
    log_placeholder = st.empty()   # live logs stream here, then get replaced

    try:
        bt = run_pipeline(params, do_retrain, log_placeholder)
    except Exception as exc:
        _show_progress(log_placeholder, context, f"❌ Run failed: {exc}")
        st.exception(exc)
        st.stop()

    # replace live logs with the results
    log_placeholder.empty()
    render_results(bt)

    with st.expander("Show full run logs"):
        st.code(context.log_buffer.getvalue(), language="log")

else:
    st.info("Configure parameters in the sidebar, then click **▶ Run Backtest**.")
    ok, reason = needs_retrain(context, train_end)
    if ok:
        st.caption(f"Note: models will be retrained on first run — {reason}.")
    else:
        st.caption(f"Trained models are {reason}; the next run will reuse them.")
