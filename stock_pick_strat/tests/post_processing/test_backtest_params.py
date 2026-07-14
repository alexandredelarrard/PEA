"""Tests that every backtest CONSTRUCTION parameter actually moves the strategy
(src/post_processing/utils/strategies_opt.py).

Motivated by "the params seem to have no effect": these lock in that each knob
changes the traded book, and that pos_cap is now enforced on the FINAL weights
(vol targeting used to be able to scale a name back above the cap).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.post_processing.utils.strategies_opt import simulate_portfolio_opt


def _synth(seed: int = 0, T: int = 300, N: int = 60):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=T)
    tickers = [f"T{i:02d}" for i in range(N)]
    betas = rng.uniform(0.5, 1.5, N)
    mkt = rng.normal(0.0004, 0.010, T)
    idio = rng.normal(0.0, 0.015, (T, N))
    stock = betas[None, :] * mkt[:, None] + idio
    stock_ret = pd.DataFrame(stock, index=dates, columns=tickers)
    spy_ret = pd.Series(mkt, index=dates)
    sig = np.full((T, N), np.nan)
    for t in range(T - 1):
        sig[t] = idio[t + 1] + rng.normal(0, 0.02, N)   # noisy predictor of next idio
    return pd.DataFrame(sig, index=dates, columns=tickers), stock_ret, spy_ret


_SIG, _STK, _SPY = _synth()
_BASE = dict(starting_capital=1e6, market_weight=0.0, target_ann_vol=0.08,
             beta_neutral=True, pos_cap=0.05, gross_cap=5.0, step=0.35,
             beta_window=63, vol_window=63, fee_bps=1.0, spread_bps=5.0)


def _run(**over):
    return simulate_portfolio_opt(_SIG, _STK, _SPY, **{**_BASE, **over})


def _alpha_vol(d):
    return float(d["alpha_ret"].std() * np.sqrt(252))


def test_diagnostic_columns_present():
    d = _run()
    for col in ("alpha_ret", "mkt_ret", "alpha_gross", "alpha_max_w"):
        assert col in d.columns, f"missing sleeve-diagnostic column {col}"
    print("\n=== SANITY CHECK: sleeve diagnostics present ===")
    print(f"  columns include alpha_ret/mkt_ret/alpha_gross/alpha_max_w -> "
          f"avg alpha gross={d['alpha_gross'].mean():.2f}, avg max|w|={d['alpha_max_w'].mean():.3f}")


def test_each_construction_param_moves_the_book():
    base = _run()
    to0 = base["turnover"].mean()

    # target_ann_vol: higher risk budget -> higher realized alpha vol + bigger gross.
    # Use a loose pos_cap here so the per-name cap does not clip the scale-up
    # (on a small synthetic universe pos_cap=0.05 would itself bind).
    lo = _run(pos_cap=0.5, target_ann_vol=0.08)
    hi = _run(pos_cap=0.5, target_ann_vol=0.16)
    assert _alpha_vol(hi) > _alpha_vol(lo) * 1.2
    assert hi["alpha_gross"].mean() > lo["alpha_gross"].mean() * 1.2

    # step: faster trading -> more turnover
    assert _run(step=1.0)["turnover"].mean() > to0 * 1.2

    # gross_cap: a tight cap shrinks the book's gross (and its vol)
    assert _run(target_ann_vol=0.16, gross_cap=0.3)["alpha_gross"].mean() < 0.35

    # beta_neutral: toggling changes the weights -> different P&L
    assert not np.allclose(_run(beta_neutral=False)["net_ret"].to_numpy(),
                           base["net_ret"].to_numpy())

    print("\n=== SANITY CHECK: every construction param moves the book ===")
    print(f"  target_ann_vol 0.08->0.16: alpha vol {_alpha_vol(lo):.3f}->{_alpha_vol(hi):.3f}; "
          f"step 0.35->1.0 turnover {to0:.3f}->{_run(step=1.0)['turnover'].mean():.3f}")
    print("  gross_cap=0.3 caps gross; beta_neutral toggles the P&L -> all active. Validated.")


def test_pos_cap_binds_on_final_weights():
    # loose cap -> no name reaches it (diversified book); tight cap -> it binds
    loose = _run(pos_cap=0.5)["alpha_max_w"].mean()
    tight = _run(pos_cap=0.01)["alpha_max_w"].mean()

    assert loose < 0.5, "loose pos_cap should not bind on a diversified book"
    # tight cap must actually constrain the FINAL weights (was violated when the
    # cap was applied pre-vol-scale): max|w| ~ pos_cap, not pos_cap * vol_scale
    assert tight <= 0.01 + 5e-3, f"pos_cap=0.01 not enforced on final book (max|w|={tight:.4f})"
    assert tight < loose

    print("\n=== SANITY CHECK: pos_cap enforced on final weights ===")
    print(f"  avg max|w|: pos_cap=0.5 -> {loose:.3f} (slack); pos_cap=0.01 -> {tight:.4f} "
          f"(binds ~0.01). Cap now applies AFTER vol targeting. Validated.")


def test_realized_risk_is_frequency_invariant():
    """Regression: the held book's realized vol / gross must NOT collapse as the
    rebalance frequency rises. Before the fix, vol targeting was applied to the
    TARGET w* only, so the partial-step book at rebalance_freq=1 (market_weight=0)
    carried ~44% of the intended gross -> it looked like it barely traded. The fix
    risk-targets the ACTUALLY-HELD book each day, making realized risk frequency-
    invariant while turnover still falls with less-frequent rebalancing."""
    def _stats(freq):
        d = _run(target_ann_vol=0.08, rebalance_freq=freq)
        return (float(d["alpha_gross"].mean()),
                float(d["alpha_ret"].std() * np.sqrt(252)),
                float(d["turnover"].mean()))

    g1, v1, to1 = _stats(1)
    g63, v63, to63 = _stats(63)

    # realized alpha vol tracks the 0.08 target at BOTH frequencies (within ~35%)
    for v in (v1, v63):
        assert 0.05 <= v <= 0.11, f"realized alpha vol {v:.3f} not near target 0.08"
    # daily rebalancing no longer shrinks the book vs quarterly (ratio near 1)
    assert g1 > 0.7 * g63, f"book gross collapses at freq=1 (g1={g1:.2f}, g63={g63:.2f})"
    # turnover control still works: more frequent rebalancing => more turnover
    assert to1 > to63

    print("\n=== SANITY CHECK: realized risk is rebalance-frequency invariant ===")
    print(f"  market_weight=0: freq=1 -> gross {g1:.2f}, alpha_vol {v1:.3f}, turnover {to1:.3f}")
    print(f"                   freq=63-> gross {g63:.2f}, alpha_vol {v63:.3f}, turnover {to63:.3f}")
    print(f"  book gross ratio freq1/freq63 = {g1/g63:.2f} (was ~0.44 before the fix); "
          f"realized vol tracks the 0.08 target at both. The alpha sleeve no longer "
          f"goes inert at daily rebalancing when market_weight=0. Validated.")


def test_degenerate_signal_warns(caplog=None):
    """A signal with no cross-sectional dispersion yields an empty alpha book;
    with market_weight=0 the whole portfolio is inert. The engine must WARN
    rather than silently return a flat curve."""
    import logging
    T, N = 120, 40
    dates = pd.bdate_range("2020-01-01", periods=T)
    tickers = [f"T{i:02d}" for i in range(N)]
    stock_ret = pd.DataFrame(np.random.default_rng(0).normal(0, 0.01, (T, N)),
                             index=dates, columns=tickers)
    spy_ret = pd.Series(np.random.default_rng(1).normal(0, 0.01, T), index=dates)
    flat_sig = pd.DataFrame(1.0, index=dates, columns=tickers)  # zero dispersion

    logger = logging.getLogger("src.post_processing.utils.strategies_opt")
    records = []
    handler = logging.Handler()
    handler.emit = lambda r: records.append(r.getMessage())
    logger.addHandler(handler); logger.setLevel(logging.WARNING)
    try:
        d = simulate_portfolio_opt(flat_sig, stock_ret, spy_ret, market_weight=0.0,
                                   target_ann_vol=0.08, beta_neutral=True, pos_cap=0.05,
                                   gross_cap=5.0, step=0.35)
    finally:
        logger.removeHandler(handler)

    assert float(d["alpha_gross"].mean()) < 1e-6, "flat signal should give empty book"
    assert any("never established a position" in m for m in records), \
        "degenerate/inert alpha book must emit a warning"

    print("\n=== SANITY CHECK: degenerate signal is flagged, not silent ===")
    print(f"  zero-dispersion signal + market_weight=0 -> avg alpha gross "
          f"{d['alpha_gross'].mean():.2e}; engine logged the inactivity warning. Validated.")
