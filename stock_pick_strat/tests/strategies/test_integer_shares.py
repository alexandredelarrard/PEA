"""
Integer-share optimizer (src/strategies/utils/integer_shares.py). Validates that the MILP turns a
continuous market-neutral target into a WHOLE-SHARE book that (1) has integer shares, (2) keeps the
long/short signs, (3) stays dollar/beta-neutral within tolerance, (4) keeps gross within ±tol, and
(5) tracks the target closely at realistic capital.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategies.utils.integer_shares import integerize, book_stats


def _target(n=30, seed=0):
    rng = np.random.default_rng(seed)
    names = [f"S{i:02d}" for i in range(n)]
    w = np.concatenate([np.full(n // 2, 1.0), np.full(n // 2, -1.0)]) / n   # dollar-neutral, gross 1
    price = pd.Series(rng.uniform(20, 300, n), index=names)
    beta = pd.Series(rng.normal(1.0, 0.05, n), index=names)                 # ~1 so beta-neutral≈dollar-neutral
    return pd.Series(w, index=names), price, beta


def test_integerize_is_integer_neutral_and_gross_bounded():
    w, price, beta = _target()
    cap = 5_000_000
    shares = integerize(w, price, cap, beta=beta, gross_tol=0.02, dollar_tol=0.005, beta_tol=0.02)

    assert np.allclose(shares.values, np.round(shares.values)), "shares must be integers"
    longs, shorts = w[w > 0].index, w[w < 0].index
    assert (shares[longs] >= 0).all() and (shares[shorts] <= 0).all(), "long/short signs preserved"
    st = book_stats(shares, price, cap, beta)
    assert abs(st["net_frac"]) <= 0.005 + 1e-9, f"dollar-neutral within tol (got {st['net_frac']:.4f})"
    assert abs(st["beta_frac"]) <= 0.02 + 1e-9, f"beta-neutral within tol (got {st['beta_frac']:.4f})"
    assert 0.98 - 1e-6 <= st["gross_frac"] <= 1.02 + 1e-6, f"gross within ±2% (got {st['gross_frac']:.4f})"

    print("\n=== SANITY CHECK: integer-share MILP ===")
    print(f"  integer shares, signs preserved; net={st['net_frac']*100:+.3f}% beta={st['beta_frac']*100:+.2f}% "
          f"gross={st['gross_frac']*100:.1f}% (target 100%), {st['n_pos']} positions. Validated.")


def test_long_fractional_shorts_integer_at_100k():
    """Retail case at $100k: SHORTS must be whole shares, LONGS may stay fractional. The fractional
    longs absorb the short-side rounding, so the book is neutral AND tracks tighter than full-integer."""
    w, price, beta = _target(seed=1)
    cap = 100_000
    longs, shorts = w[w > 0].index, w[w < 0].index
    sh = integerize(w, price, cap, beta=beta, gross_tol=0.02, dollar_tol=0.005,
                    beta_tol=0.02, long_fractional=True)

    assert np.allclose(sh[shorts].values, np.round(sh[shorts].values)), "shorts must be whole shares"
    assert (sh[longs] >= 0).all() and (sh[shorts] <= 0).all(), "long/short signs preserved"
    st = book_stats(sh, price, cap, beta)
    assert abs(st["net_frac"]) <= 0.005 + 1e-9, f"dollar-neutral within tol (got {st['net_frac']:.4f})"
    assert abs(st["beta_frac"]) <= 0.02 + 1e-9, f"beta-neutral within tol (got {st['beta_frac']:.4f})"
    assert 0.98 - 1e-6 <= st["gross_frac"] <= 1.02 + 1e-6, f"gross within ±2% (got {st['gross_frac']:.4f})"

    def tracking(long_fractional):
        s = integerize(w, price, cap, beta=beta, long_fractional=long_fractional)
        return float((s * price - w * cap).abs().sum() / (w.abs() * cap).sum())
    err_frac, err_full = tracking(True), tracking(False)
    assert err_frac <= err_full + 1e-9, "fractional longs cannot track worse than full-integer"
    n_frac_long = int((~np.isclose(sh[longs].values, np.round(sh[longs].values))).sum())

    print("\n=== SANITY CHECK: fractional longs / integer shorts @ $100k ===")
    print(f"  shorts all whole shares; {n_frac_long}/{len(longs)} longs fractional. "
          f"net={st['net_frac']*100:+.3f}% beta={st['beta_frac']*100:+.2f}% gross={st['gross_frac']*100:.1f}%.")
    print(f"  L1 tracking/gross: fractional-long {err_frac*100:.2f}%  vs  full-integer {err_full*100:.2f}%. Validated.")


def test_tracking_tightens_with_capital():
    w, price, beta = _target(seed=3)
    # larger capital -> each name is many shares -> integer book tracks the continuous target better
    def tracking(cap):
        sh = integerize(w, price, cap, beta=beta)
        cont = w * cap
        return float((sh * price - cont).abs().sum() / (w.abs() * cap).sum())   # L1 error / gross
    err_small, err_big = tracking(2_000_000), tracking(50_000_000)
    assert err_big < err_small, "tracking error should shrink as capital (shares per name) grows"
    print("\n=== SANITY CHECK: integer tracking vs capital ===")
    print(f"  L1 tracking error/gross: $2M -> {err_small*100:.2f}%   $50M -> {err_big*100:.2f}%. Validated.")


if __name__ == "__main__":
    test_integerize_is_integer_neutral_and_gross_bounded()
    test_long_fractional_shorts_integer_at_100k()
    test_tracking_tightens_with_capital()
