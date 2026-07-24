"""
integer_shares.py  (src/strategies/utils/integer_shares.py)
-----------------------------------------------------------
Round a continuous long/short target to INTEGER SHARES (you can't hold/short a fraction of a
share). This is the classic **round-lot / integer-constrained portfolio** problem — a
mixed-integer program (solved here with HiGHS via scipy.optimize.milp):

  minimize   Σ |shares_i·price_i − target_$_i|                (L1 tracking error to the target)
  over       shares_i INTEGER, sign fixed to the target       (keeps the long/short structure)
  s.t.       |Σ shares_i·price_i|          ≤ dollar_tol·capital   (dollar-neutral)
             |Σ shares_i·price_i·beta_i|   ≤ beta_tol·capital     (beta-neutral)
             |Σ_{i∈g} shares_i·price_i|    ≤ sector_tol·capital   (sector-neutral, per group)
             (1−gross_tol)·G ≤ Σ|shares_i·price_i| ≤ (1+gross_tol)·G   (gross within ±tol of target)

Relaxing gross / neutrality by 1–2% is what makes an integer solution reachable while keeping the
same portfolio STRUCTURE — the solver freely trades off which names to round up/down (implicit
substitution among names) to satisfy the neutrality constraints at least tracking-error cost.
Falls back to plain nearest-integer rounding if the solver is unavailable or the MILP is infeasible.

`long_fractional=True` models the common retail reality: you CAN buy fractional shares (longs stay
continuous) but CANNOT short fractions (only the SHORT leg is integer). It's a smaller MIP — fewer
integer vars — and tracks the target far better, because the continuous longs freely absorb the
short-side rounding to restore dollar/beta/sector-neutrality (only shorts pay a discretization cost).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy import sparse


def _round(d: np.ndarray, p: np.ndarray, sign: np.ndarray, long_fractional: bool) -> np.ndarray:
    """Fallback: nearest-integer share count (signed); if `long_fractional`, LONGS keep their exact
    fractional share and only SHORTS are rounded (you can buy fractional but not short fractional)."""
    sh = d / p
    to_int = (sign < 0) if long_fractional else np.ones(len(sh), dtype=bool)
    sh[to_int] = np.round(sh[to_int])
    return sh


def _solve_milp(w, p, d, b, groups, capital, gross_tol, dollar_tol, beta_tol,
                sector_tol, share_cap_mult, time_limit, long_fractional) -> np.ndarray:
    n = len(w)
    sign = np.sign(w)
    sp = sign * p                                   # signed price: v_i = sp_i · k_i
    G = float(np.abs(d).sum())                      # target gross $
    kmax = np.ceil(share_cap_mult * np.abs(d) / p) + 2.0
    # vars: k_0..k_{n-1} (share magnitude ≥0) then t_0..t_{n-1} (L1 aux ≥0). Only SHORT k's are
    # integer when long_fractional (fractional longs allowed, fractional shorts not); else all k integer.
    k_int = ((sign < 0).astype(float) if long_fractional else np.ones(n))
    c = np.concatenate([np.zeros(n), np.ones(n)])
    integrality = np.concatenate([k_int, np.zeros(n)])
    bounds = Bounds(np.concatenate([np.zeros(n), np.zeros(n)]),
                    np.concatenate([kmax, np.full(n, np.inf)]))

    rows, lbs, ubs = [], [], []
    I = sparse.identity(n, format="csr")
    Ksp = sparse.diags(sp)                          # k -> v (signed value)
    # tracking:  sp·k − t ≤ d   and   −sp·k − t ≤ −d   (=> t ≥ |v − d|)
    rows += [sparse.hstack([Ksp, -I]), sparse.hstack([-Ksp, -I])]
    lbs += [np.full(n, -np.inf), np.full(n, -np.inf)]; ubs += [d, -d]
    zt = sparse.csr_matrix((1, n))
    def row_k(vec):                                  # a single constraint on k only
        return sparse.hstack([sparse.csr_matrix(vec.reshape(1, -1)), zt])
    rows.append(row_k(sp));         lbs.append(-dollar_tol * capital); ubs.append(dollar_tol * capital)
    rows.append(row_k(sp * b));     lbs.append(-beta_tol * capital);   ubs.append(beta_tol * capital)
    rows.append(row_k(p));          lbs.append((1 - gross_tol) * G);   ubs.append((1 + gross_tol) * G)
    for g in groups:                                 # per-sector dollar neutrality
        m = np.zeros(n); m[g] = 1.0
        rows.append(row_k(sp * m)); lbs.append(-sector_tol * capital); ubs.append(sector_tol * capital)

    A = sparse.vstack(rows, format="csr")
    lb = np.concatenate([np.atleast_1d(x) for x in lbs])
    ub = np.concatenate([np.atleast_1d(x) for x in ubs])
    res = milp(c=c, integrality=integrality, bounds=bounds,
               constraints=LinearConstraint(A, lb, ub),
               options={"time_limit": float(time_limit), "mip_rel_gap": 1e-3})
    if not res.success or res.x is None:
        raise RuntimeError(f"integer MILP infeasible/failed: {res.message}")
    k = res.x[:n]
    # clean solver tolerance on the INTEGER vars only; leave fractional longs untouched
    k = np.where(k_int > 0, np.round(k), k) if long_fractional else np.round(k)
    return sign * k


def integerize(target_w: pd.Series, price: pd.Series, capital: float,
               beta: pd.Series | None = None, sector: pd.Series | None = None, *,
               gross_tol: float = 0.02, dollar_tol: float = 0.005, beta_tol: float = 0.02,
               sector_tol: float = 0.03, share_cap_mult: float = 3.0, time_limit: float = 10.0,
               method: str = "milp", long_fractional: bool = False) -> pd.Series:
    """Signed SHARES per name best-matching `target_w`·capital (index = target_w.index).
    Names with no price or zero target get 0 shares. When `long_fractional` is True only the SHORT
    leg is constrained to whole shares (you can buy fractional but not short fractional); the LONG
    leg stays continuous and absorbs the short-side rounding to keep the book neutral."""
    idx = target_w.index
    names = [i for i in idx if i in price.index and np.isfinite(price[i]) and price[i] > 0
             and np.isfinite(target_w[i]) and abs(target_w[i]) > 0]
    out = pd.Series(0.0, index=idx)
    if not names:
        return out
    w = target_w[names].to_numpy(float)
    p = price[names].to_numpy(float)
    d = w * float(capital)
    b = (beta[names].to_numpy(float) if beta is not None else np.zeros(len(names)))
    b = np.nan_to_num(b, nan=0.0)
    groups = []
    if sector is not None:
        sec = [str(sector.get(i, "__NA__")) if hasattr(sector, "get") else str(sector[i]) for i in names]
        for u in dict.fromkeys(sec):
            groups.append([j for j, s in enumerate(sec) if s == u])
    sign = np.sign(w)
    if method == "milp":
        try:
            shares = _solve_milp(w, p, d, b, groups, float(capital), gross_tol, dollar_tol,
                                 beta_tol, sector_tol, share_cap_mult, time_limit, long_fractional)
        except Exception:                                       # noqa: BLE001 - fall back to rounding
            shares = _round(d, p, sign, long_fractional)
    else:
        shares = _round(d, p, sign, long_fractional)
    out[names] = shares
    return out


def book_stats(shares: pd.Series, price: pd.Series, capital: float,
               beta: pd.Series | None = None) -> dict:
    """Diagnostics of an integer book: net $ (dollar-neutrality), beta $, gross, #positions."""
    v = (shares * price.reindex(shares.index)).fillna(0.0)
    b = beta.reindex(shares.index).fillna(0.0) if beta is not None else pd.Series(0.0, index=shares.index)
    return {"net_usd": float(v.sum()), "net_frac": float(v.sum() / capital),
            "beta_usd": float((v * b).sum()), "beta_frac": float((v * b).sum() / capital),
            "gross_usd": float(v.abs().sum()), "gross_frac": float(v.abs().sum() / capital),
            "n_pos": int((shares != 0).sum())}
