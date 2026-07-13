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
