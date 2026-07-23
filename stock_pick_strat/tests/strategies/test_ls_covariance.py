"""
Correlation-aware (shrunk-covariance) risk model for the L/S optimizer
(src/strategies/utils/strategies_opt.py). Validates:
  1. it REDUCES EXACTLY to the diagonal inverse-variance book when Σ = diag(var) (shrink→1);
  2. it DOWN-WEIGHTS a correlated cluster — two highly-correlated longs get less combined weight
     than an equally-strong uncorrelated name (the whole point of accounting for correlation);
  3. the shrunk covariance is PD (invertible) even with N > T.
"""
from __future__ import annotations

import numpy as np

from src.strategies.utils.strategies_opt import optimize_day, shrunk_idio_cov, vol_target_scale


def test_cov_reduces_to_diagonal_at_full_shrink():
    rng = np.random.default_rng(0)
    n = 12
    alpha = rng.normal(0, 1, n)
    beta = rng.normal(1, 0.3, n)
    var = np.full(n, 3e-4)                 # constant -> the diagonal path's 5th-pctile clip is a no-op
    w_diag = optimize_day(alpha, beta, var, beta_neutral=True, pos_cap=None, cov=None)
    w_cov = optimize_day(alpha, beta, var, beta_neutral=True, pos_cap=None, cov=np.diag(var))
    assert np.allclose(w_diag, w_cov, atol=1e-10), "Σ=diag(var) must reproduce inverse-variance"
    print("\n=== SANITY CHECK: covariance model reduces to diagonal ===")
    print(f"  max|w_cov - w_diag| = {np.max(np.abs(w_cov - w_diag)):.2e} (≈0). Validated.")


def test_cov_downweights_correlated_cluster():
    # 4 names, dollar-neutral: THREE longs {0,1,2} + one short {3}. Longs 0&1 are 0.9-correlated;
    # long 2 is INDEPENDENT (same alpha=1 as 0,1). No beta/sector neutrality, no cap -> raw shape.
    alpha = np.array([1.0, 1.0, 1.0, -3.0])
    beta = np.zeros(4)
    sig2 = 4e-4
    var = np.full(4, sig2)
    R = np.eye(4); R[0, 1] = R[1, 0] = 0.9
    cov = sig2 * R
    w_diag = optimize_day(alpha, beta, var, beta_neutral=False, pos_cap=None, cov=None)
    w_cov = optimize_day(alpha, beta, var, beta_neutral=False, pos_cap=None, cov=cov)

    # diagonal: the two correlated longs and the independent long are weighted EQUALLY (same alpha)
    assert abs(abs(w_diag[0]) - abs(w_diag[2])) < 1e-9
    # covariance: each correlated long is DOWN-WEIGHTED vs the equally-strong INDEPENDENT long
    assert abs(w_cov[0]) < 0.98 * abs(w_cov[2]), "correlated cluster must be down-weighted"
    print("\n=== SANITY CHECK: correlated cluster down-weighting ===")
    print(f"  diagonal   |w_corr0|={abs(w_diag[0]):.3f}  |w_indep2|={abs(w_diag[2]):.3f} (equal)")
    print(f"  covariance |w_corr0|={abs(w_cov[0]):.3f}  |w_indep2|={abs(w_cov[2]):.3f} "
          f"(correlated long down-weighted vs independent). Validated.")


def test_shrunk_cov_pd_even_when_n_gt_t():
    rng = np.random.default_rng(1)
    T, N = 40, 120                                   # N > T -> sample cov singular
    resid = rng.normal(0, 0.02, (T, N))
    idio = rng.uniform(1e-4, 5e-4, N)
    Sig = shrunk_idio_cov(resid, idio, shrink=0.5)
    assert Sig.shape == (N, N)
    assert np.min(np.linalg.eigvalsh(Sig)) > 0, "shrunk covariance must be positive-definite"
    # shrink=1 -> diagonal target (up to the tiny ridge)
    Sig1 = shrunk_idio_cov(resid, idio, shrink=1.0)
    assert np.allclose(np.diag(Sig1), idio, atol=1e-8) and np.allclose(Sig1 - np.diag(np.diag(Sig1)), 0, atol=1e-8)
    # vol_target_scale uses wᵀΣw when cov given
    w = rng.normal(0, 1, N)
    s = vol_target_scale(w.copy(), idio, 0.10, gross_cap=1e9, cov=Sig)
    assert np.isfinite(s).all()
    print("\n=== SANITY CHECK: shrunk covariance PD (N>T) ===")
    print(f"  N={N} > T={T}: min eigenvalue = {np.min(np.linalg.eigvalsh(Sig)):.2e} > 0; "
          f"shrink=1 → diagonal. Validated.")


if __name__ == "__main__":
    test_cov_reduces_to_diagonal_at_full_shrink()
    test_cov_downweights_correlated_cluster()
    test_shrunk_cov_pd_even_when_n_gt_t()
