"""
Streamlit backtest dashboard.

Run from stock_pick_strat/ with:
    streamlit run app/app.py

Behaviour:
  * Sidebar exposes every backtest parameter.
  * Backtest start date == the trained model's end date, so the backtest is always
    strictly out-of-sample. The desired start is modellling.yml `train.end_date`.
  * If the existing model's end date is aligned (same day) with the desired backtest
    start, the models are reused. If it is not aligned — earlier OR later — or the
    models are absent/incomplete, StepModelling is retrained first and the central
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
from src.modelling.utils_model import model as ml
from src.post_processing.step_backtest import StepBacktest
from src.post_processing.utils.accuracy import (
    compute_horizon_accuracy,
    horizon_accuracy_summary,
)


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
# forecast horizons the model was actually trained on (30/60/90 by default)
MODEL_HORIZONS = [int(h) for h in base_config.build_cube.targets.horizons]
PRIMARY_HORIZON = int(base_config.build_cube.targets.get("primary_horizon",
                                                         MODEL_HORIZONS[0]))

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

    st.subheader("Prediction Accuracy")
    accuracy_horizon = st.selectbox(
        "Accuracy horizon (trading days)", MODEL_HORIZONS,
        index=MODEL_HORIZONS.index(PRIMARY_HORIZON),
        help="Directional accuracy is measured over the model's FORECAST horizon "
             "(the signal targets 30/60/90-day moves), not next-day noise. "
             "Changing this re-renders instantly — no re-run needed.",
    )

    run_btn = st.button("▶  Run Backtest", type="primary", use_container_width=True)


# ---------------------------------------------------------------------------
# Retrain detection + log streaming helpers
# ---------------------------------------------------------------------------
def needs_retrain(ctx, desired_start: str) -> tuple[bool, str]:
    """
    The backtest must start exactly at the model's end date to stay out-of-sample.
    Retrain when that alignment is broken: the trained model's end date is earlier
    OR later than the desired backtest start, or when models are absent/incomplete.
    Dates are compared normalized (day granularity), not as raw strings.
    """
    models_dir = ctx.paths["MODELS_DIR"]
    meta_path = models_dir / "metadata.json"
    if not meta_path.exists():
        return True, "no trained models found"
    try:
        meta = json.loads(meta_path.read_text())
    except Exception:
        return True, "model metadata unreadable"

    stored = meta.get("train_end")
    if not stored:
        return True, "model train_end missing from metadata"

    model_end = pd.Timestamp(stored).normalize()
    want = pd.Timestamp(desired_start).normalize()
    if model_end < want:
        return True, (f"model end {model_end.date()} is EARLIER than desired backtest "
                      f"start {want.date()} — not aligned, retrain")
    if model_end > want:
        return True, (f"model end {model_end.date()} is LATER than desired backtest "
                      f"start {want.date()} — not aligned, retrain")

    # ensemble COMPOSITION changed in the config (e.g. random_forest added / removed) ->
    # retrain so every chosen member is trained + saved (and none goes stale)
    desired_members = list(base_config.model.get("ensemble")
                           or [base_config.model.get("type", "lightgbm")])
    saved_members = list(meta.get("model_types", []))
    if set(desired_members) != set(saved_members):
        return True, (f"ensemble changed: config wants {sorted(desired_members)}, saved "
                      f"models are {sorted(saved_members)} — retrain to (re)save all members")

    # dates + ensemble aligned -> make sure every chosen member's file is actually present
    # (member_model_path picks .txt for booster kinds incl. random_forest, .pkl for linear)
    for h in meta.get("horizons", []):
        for kind in saved_members:
            if not ml.member_model_path(models_dir, h, kind).exists():
                return True, f"missing model file for h{h}/{kind}"

    return False, (f"model end aligned with backtest start ({model_end.date()}); "
                   f"ensemble {sorted(saved_members)} present")


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
    # load_models() sets bt.backtest_start = the MODEL's end date (meta train_end);
    # that IS the backtest start, so we keep it rather than overriding with config.
    bt.load_models()
    model_end = pd.Timestamp(bt.backtest_start).normalize()
    want = pd.Timestamp(train_end).normalize()
    if model_end != want:
        # should not happen after the retrain check; surface loudly if it does
        bt._log.warning("Backtest start %s != model end date %s — alignment check failed",
                        want.date(), model_end.date())
    bt._log.info("Backtest start = model end date: %s", model_end.date())

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


def plot_hit_rate(acc: pd.DataFrame, horizon: int) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 3))
    rolling = acc["hit_rate_%"].rolling(21, min_periods=5).mean()
    ax.bar(acc.index, acc["hit_rate_%"], color="#90caf9", alpha=0.5, width=1,
           label="Per-date")
    ax.plot(rolling.index, rolling, color="#1565c0", linewidth=1.5, label="21-obs MA")
    ax.axhline(50, color="red", linewidth=1, linestyle="--", label="50% (random)")
    ax.set_ylabel("Hit Rate (%)")
    ax.set_title(f"Cross-Sectional Directional Accuracy — Signal vs {horizon}-Day "
                 f"Forward Return")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def render_horizon_blend(bt: StepBacktest, model_horizons: list[int]):
    """Per-horizon CV IC_IR (from the trained model's metadata) alongside the
    actual blend weight the strategy used — makes it explicit whether e.g. the
    90d horizon contributes, and why."""
    st.subheader("Horizon Blend (ensemble → combined signal)")
    train_ic = getattr(bt, "train_ic", {}) or {}
    weights = getattr(bt, "blend_weights", {}) or {}
    rows = []
    for h in model_horizons:
        ir = train_ic.get(h, np.nan)
        rows.append({
            "horizon": h,
            "cv_ic_ir": round(float(ir), 3) if ir is not None and np.isfinite(ir) else np.nan,
            "blend_weight": round(float(weights.get(h, 0.0)), 3),
            "used": "yes" if weights.get(h, 0.0) > 1e-6 else "NO",
        })
    tbl = pd.DataFrame(rows).set_index("horizon")
    st.dataframe(
        tbl.style.format({"cv_ic_ir": "{:.3f}", "blend_weight": "{:.3f}"}, na_rep="—"),
        use_container_width=True,
    )
    dropped = [h for h in model_horizons if weights.get(h, 0.0) <= 1e-6]
    if dropped:
        st.caption(f"Horizons with ~zero weight: {dropped}. With the correlation-aware "
                   "blend, a NaN CV IC_IR no longer forces weight 0 — a zero here means "
                   "the horizon had no panel in the window or a non-positive IR.")


def render_signal_coverage(bt: StepBacktest, model_horizons: list[int]):
    """Signal coverage diagnostic. The backtest can ONLY trade / be scored on dates
    that carry a signal, and a signal exists only where the cube has a defined
    target (panel_from_cube drops target-NaN rows). Sparse coverage here is the
    single explanation for: sporadic accuracy-table dates, a late first trade, and
    empty long horizons. This makes the coverage visible instead of implicit."""
    sig = bt.signal
    dates = pd.DatetimeIndex(sig.index).sort_values()
    st.subheader("Signal coverage (why trades/rows appear where they do)")
    if len(dates) == 0:
        st.error("The signal is EMPTY — the cube has no defined target in the backtest "
                 "window. Nothing can trade. Rebuild the cube with complete price data.")
        return
    names_per_date = sig.notna().sum(axis=1)
    gaps = dates.to_series().diff().dt.days.dropna()
    active = names_per_date[names_per_date >= 10]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Signal dates", f"{len(dates):,}")
    c2.metric("First → last", f"{dates.min().date()} → {dates.max().date()}")
    c3.metric("Median gap (cal. days)", f"{gaps.median():.0f}" if len(gaps) else "—")
    c4.metric("Largest gap (cal. days)", f"{gaps.max():.0f}" if len(gaps) else "—")
    st.caption(f"Median names/date with a signal: {int(names_per_date.median())}; "
               f"dates with ≥10 names (tradeable): {len(active)} "
               f"(first {active.index.min().date() if len(active) else '—'}).")

    if len(gaps) and gaps.max() > 7:
        st.warning(
            f"The signal is SPARSE — consecutive signal dates are up to "
            f"{gaps.max():.0f} calendar days apart (a dense daily signal would be ~1–4). "
            f"A signal exists only on dates where the cube has a defined target, so a "
            f"sparse/gappy cube (missing prices → missing targets) is why the detail "
            f"table shows only scattered dates, why the first trade is late (the book "
            f"can't trade before a tradeable signal exists), and why long horizons are "
            f"empty. Fix = heal the price/factor gaps (re-run extraction) and REBUILD "
            f"the cube; these KPIs are reporting the data faithfully."
        )


def render_results(bt: StepBacktest, accuracy_horizon: int, model_horizons: list[int]):
    daily = bt.daily
    metrics = bt.metrics

    # ---- Backtest span (surfaces "expected 2y but only N days" data-coverage gaps)
    span_days = len(daily)
    if span_days:
        d0, d1 = daily.index.min(), daily.index.max()
        st.caption(f"**Backtest span:** {d0.date()} → {d1.date()}  "
                   f"({span_days:,} trading days ≈ {span_days/252:.1f}y). "
                   f"Signal matrix: {bt.signal.shape[0]}×{bt.signal.shape[1]} (dates×tickers).")
        if span_days < max(model_horizons):
            st.warning(
                f"The backtest window ({span_days} days) is shorter than the longest "
                f"forecast horizon ({max(model_horizons)}d): horizons ≥ the window "
                f"length cannot be scored (their forward return needs future data that "
                f"isn't in the window). This is a DATA-COVERAGE issue — the cube barely "
                f"extends past the model train-end ({train_end}) — not a metric bug. "
                f"Extend the cube/prices past {train_end} to evaluate longer horizons."
            )

    render_signal_coverage(bt, model_horizons)

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

    # ---- Horizon blend (which horizons actually feed the signal, and why) ----
    render_horizon_blend(bt, model_horizons)

    # ---- Equity curve ----
    st.subheader("Equity Curve vs SPY")
    fig_eq = plot_equity(daily)
    st.pyplot(fig_eq)
    plt.close(fig_eq)

    # ---- Prediction accuracy AT THE FORECAST HORIZON ----
    st.subheader("Prediction Accuracy (forecast horizon)")
    st.caption(
        "The signal targets 30/60/90-day moves, so accuracy is measured over the "
        "forecast horizon and CROSS-SECTIONALLY (did the name beat/lag its peers in "
        "the predicted direction), not against next-day noise. A daily-horizon hit "
        "rate near 50% is expected and uninformative for this model."
    )

    # cross-horizon summary first — the honest at-a-glance edge
    with st.spinner("Scoring accuracy across horizons…"):
        summary = horizon_accuracy_summary(bt, model_horizons)
    st.markdown("**Directional accuracy by horizon**")
    st.dataframe(
        summary.style.format({
            "avg_hit_rate_%": "{:.2f}", "pct_dates_positive_%": "{:.1f}",
            "avg_long_short_%": "{:+.3f}", "n_dates": "{:d}",
        }, na_rep="—"),
        use_container_width=True,
    )

    # detailed panel for the chosen horizon
    with st.spinner(f"Computing {accuracy_horizon}-day hit rates…"):
        acc = compute_horizon_accuracy(bt, accuracy_horizon)

    if acc.empty:
        st.warning("Could not compute horizon accuracy — no overlapping signal/return "
                   "data (need ≥ horizon days of forward returns).")
        return

    avg_hit = acc["hit_rate_%"].mean()
    spread = acc["long_short_fwd_%"].dropna()
    pct_pos = (spread > 0).mean() * 100 if len(spread) else np.nan

    acol1, acol2, acol3 = st.columns(3)
    acol1.metric(f"Avg {accuracy_horizon}-Day Hit Rate", f"{avg_hit:.2f}%",
                 delta=f"{avg_hit - 50:.2f}% vs coin-flip",
                 help="Share of conviction names whose sign matched the peer-relative "
                      f"{accuracy_horizon}-day forward return")
    acol2.metric("Dates w/ Positive Long-Short", f"{pct_pos:.1f}%")
    acol3.metric("Mean Long-Short Spread",
                 f"{spread.mean():+.3f}%" if len(spread) else "—",
                 help=f"Realized {accuracy_horizon}-day return of predicted longs "
                      f"minus predicted shorts")

    fig_hr = plot_hit_rate(acc, accuracy_horizon)
    st.pyplot(fig_hr)
    plt.close(fig_hr)

    st.subheader(f"Detail Table — {accuracy_horizon}-day horizon")

    def _colour_hit(val):
        if isinstance(val, (int, float)):
            if val >= 54:
                return "background-color: #c8e6c9"
            elif val < 46:
                return "background-color: #ffcdd2"
        return ""

    def _colour_spread(val):
        if isinstance(val, (int, float)) and np.isfinite(val):
            return "background-color: #c8e6c9" if val > 0 else "background-color: #ffcdd2"
        return ""

    display = acc.copy()
    display.index = display.index.strftime("%Y-%m-%d")

    styled = (
        display.style
        .map(_colour_hit, subset=["hit_rate_%"])
        .map(_colour_spread, subset=["long_short_fwd_%"])
        .format({
            "hit_rate_%": "{:.1f}",
            "long_short_fwd_%": "{:+.3f}",
            "spy_fwd_%": "{:+.3f}",
        }, na_rep="—")
    )
    st.dataframe(styled, height=420, use_container_width=True)

    csv = acc.reset_index().to_csv(index=False)
    st.download_button(
        f"Download {accuracy_horizon}-day accuracy CSV",
        data=csv,
        file_name=f"backtest_accuracy_h{accuracy_horizon}.csv",
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

    log_placeholder.empty()
    # cache the finished backtest so changing the accuracy horizon re-renders the
    # results WITHOUT re-running (or retraining) the whole pipeline
    st.session_state["bt"] = bt
    st.session_state["run_logs"] = context.log_buffer.getvalue()

# Render whenever a completed backtest is available (fresh run or a horizon flip)
bt = st.session_state.get("bt")
if bt is not None:
    render_results(bt, int(accuracy_horizon), MODEL_HORIZONS)
    with st.expander("Show full run logs"):
        st.code(st.session_state.get("run_logs", ""), language="log")
else:
    st.info("Configure parameters in the sidebar, then click **▶ Run Backtest**.")
    ok, reason = needs_retrain(context, train_end)
    if ok:
        st.caption(f"Note: models will be retrained on first run — {reason}.")
    else:
        st.caption(f"Trained models are {reason}; the next run will reuse them.")
