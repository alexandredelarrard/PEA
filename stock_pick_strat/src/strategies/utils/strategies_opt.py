"""
strategies_opt.py  (src/post_processing/utils/strategies_opt.py)
----------------------------------------------------------------
State-of-the-art construction to replace the top/bottom-decile equal-weight
long/short book. That naive book was volatile and expensive because it (a)
equal-weighted extreme-signal names -> concentrated in high-vol stocks, (b)
was only DOLLAR-neutral, not BETA-neutral -> carried net market exposure that
whipsawed it, and (c) jumped to a fresh target every day -> ~2.9x daily
turnover -> costs that swamped the signal.

This builds, each rebalance:

  ALPHA SLEEVE  (market-neutral, self-financing long/short)
    w* = argmax_w  a'w - (lambda/2) w' D w   s.t.  1'w = 0  and  beta'w = 0
    -> closed form:  w* proportional to  D^{-1} * (GLS residual of a on [1, beta])
       i.e. residualize the alpha against a constant AND market beta in the
       inverse-variance metric, then inverse-variance weight. This is the
       Grinold-Kahn characteristic portfolio with dollar + beta neutrality and
       an (idiosyncratic) risk model D. No external solver needed.
    Then: position-cap, re-neutralize, and VOLATILITY-TARGET the whole sleeve to
    a fixed ex-ante vol so the book's risk is stable through time.

  MARKET SLEEVE
    a deliberate, constant `market_weight` in SPY. Because the alpha sleeve is
    beta-neutral, TOTAL portfolio beta = market_weight exactly -- the market bet
    is sized on purpose, not smuggled in through the signal.

  TURNOVER CONTROL (Garleanu-Pedersen)
    don't jump to w*; trade PARTIALLY toward it:  w = w_prev + step*(w* - w_prev)
    plus an optional no-trade band. `step` in (0,1]; smaller = slower book =
    less turnover. This is the closed-form optimal-trading-rate idea and is the
    single biggest cost reducer.

All the heavy math is in pure functions (optimize_day, vol_target_scale) so it
is unit-tested; only the day loop touches state.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

_log = logging.getLogger(__name__)
_ANN = 252.0


# --------------------------------------------------------------------------- #
# Pure building blocks                                                         #
# --------------------------------------------------------------------------- #
def _neutralize(a: np.ndarray, X: np.ndarray, w_metric: np.ndarray) -> np.ndarray:
    """GLS residual of `a` on columns of `X` using diagonal weights `w_metric`.
    Zeros out a's components along each column of X in that metric."""
    W = w_metric
    XtW = X.T * W                       # (k,N)
    coef = np.linalg.solve(XtW @ X + 1e-12 * np.eye(X.shape[1]), XtW @ a)
    return a - X @ coef


def _sector_dummies(sector_labels) -> np.ndarray:
    """One-hot (n,k) matrix of sector membership. Missing labels go to a shared
    '__NA__' bucket so every name is constrained to exactly one group (the columns
    span the constant, so per-sector dollar-neutrality implies global neutrality)."""
    labels = ["__NA__" if (l is None or (isinstance(l, float) and np.isnan(l))) else str(l)
              for l in sector_labels]
    uniq = list(dict.fromkeys(labels))
    idx = {u: i for i, u in enumerate(uniq)}
    D = np.zeros((len(labels), len(uniq)))
    for r, l in enumerate(labels):
        D[r, idx[l]] = 1.0
    return D


def _neutralizer_X(n: int, beta: np.ndarray, beta_neutral: bool,
                   sector_labels=None) -> np.ndarray:
    """Design matrix the weights are made orthogonal to: sector one-hots (or a
    single constant for plain dollar-neutrality) plus the market beta. Sector
    one-hots => the book is dollar-neutral WITHIN each sector (net sector exposure
    ~ 0) -- the industry-group neutrality top market-neutral funds enforce."""
    if sector_labels is not None and len(sector_labels) == n:
        base = _sector_dummies(sector_labels)          # spans the constant
    else:
        base = np.ones((n, 1))
    if beta_neutral:
        return np.column_stack([base, np.asarray(beta, float).reshape(-1, 1)])
    return base


def _optimize_cov(alpha: np.ndarray, X: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Full-covariance characteristic portfolio (correlation-aware generalization of the diagonal
    case): w = Σ⁻¹ (alpha − X (Xᵀ Σ⁻¹ X)⁻¹ Xᵀ Σ⁻¹ alpha) — residualize alpha against the
    neutrality columns X in the Σ⁻¹ metric, then weight by Σ⁻¹ (not D⁻¹). Correlated names are
    jointly down-weighted so the book doesn't double-count shared (idiosyncratic) risk."""
    Sinv_a = np.linalg.solve(cov, alpha)                    # Σ⁻¹ a
    Sinv_X = np.linalg.solve(cov, X)                        # Σ⁻¹ X
    M = X.T @ Sinv_X                                        # Xᵀ Σ⁻¹ X
    coef = np.linalg.solve(M + 1e-12 * np.eye(X.shape[1]), X.T @ Sinv_a)
    return Sinv_a - Sinv_X @ coef                          # Σ⁻¹ (a − X coef)


def optimize_day(alpha: np.ndarray, beta: np.ndarray, var: np.ndarray,
                 beta_neutral: bool = True, pos_cap: float | None = 0.03,
                 sector_labels=None, cov: np.ndarray | None = None) -> np.ndarray:
    """
    Closed-form mean-variance long/short weights with dollar (and optional beta
    and sector) neutrality.

    Risk model: DIAGONAL idiosyncratic variance `var` (D⁻¹ inverse-variance weighting) when
    `cov` is None; the FULL (shrunk) idiosyncratic covariance `cov` (Σ⁻¹, correlation-aware)
    when given — the latter is the ERC-like upgrade that accounts for stock-stock correlation.

    alpha : centered cross-sectional score (magnitude matters; z-scores ideal)
    beta  : per-name market beta
    var   : per-name idiosyncratic DAILY variance (>0)  [diagonal risk model + fallback]
    cov   : optional NxN idiosyncratic DAILY covariance (same name order as alpha)
    sector_labels : per-name group (e.g. GICS industry group). When given, weights
        are made orthogonal to each sector dummy -> net exposure per sector ~ 0.
    Returns weights summing to ~0 (dollar-neutral, and per-sector if labelled) and
    beta'w ~ 0 if beta_neutral. Scale is arbitrary here (set later by vol targeting).
    """
    n = len(alpha)
    X = _neutralizer_X(n, beta, beta_neutral, sector_labels)

    if cov is not None:
        w = _optimize_cov(alpha, X, cov)                   # Σ⁻¹ * residual alpha (correlation-aware)
    else:
        var = np.clip(var, np.nanpercentile(var[np.isfinite(var)], 5) if np.isfinite(var).any() else 1e-6, None)
        invd = 1.0 / var
        w = invd * _neutralize(alpha, X, invd)             # D⁻¹ * residual alpha (inverse-variance)

    if pos_cap is not None and n > 0:
        scale = np.sum(np.abs(w))
        if scale > 0:
            w = w / scale                  # normalize gross to 1 before capping
        w = np.clip(w, -pos_cap, pos_cap)
        w = _neutralize(w, X, np.ones(n))  # restore neutrality after clipping
    return w


def enforce_pos_cap(w: np.ndarray, beta: np.ndarray, beta_neutral: bool,
                    pos_cap: float | None, sector_labels=None) -> np.ndarray:
    """Clip |w| to `pos_cap` and restore neutrality (dollar / beta / sector),
    applied to the FINAL (vol-targeted) weights. The cap inside `optimize_day` is
    a PRE-scale shape limit; vol targeting then rescales the book, so without this
    a name can end up above pos_cap. When the book is diversified enough that no
    name reaches the cap (the usual case) this is a no-op."""
    if pos_cap is None:
        return w
    X = _neutralizer_X(len(w), beta, beta_neutral, sector_labels)
    return _neutralize(np.clip(w, -pos_cap, pos_cap), X, np.ones(len(w)))


def vol_target_scale(w: np.ndarray, var: np.ndarray, target_ann_vol: float,
                     gross_cap: float = 3.0, cov: np.ndarray | None = None) -> np.ndarray:
    """Scale weights so the sleeve's ex-ante annualized idiosyncratic vol equals
    `target_ann_vol`, subject to a gross-leverage cap. Uses the full covariance
    (wᵀΣw, correlation-aware) when `cov` is given, else the diagonal sum(w²·var)."""
    daily_var = float(w @ cov @ w) if cov is not None else float(np.sum((w ** 2) * var))
    ann_vol = np.sqrt(max(daily_var, 1e-18) * _ANN)
    if ann_vol <= 0:
        return w
    k = target_ann_vol / ann_vol
    w = w * k
    gross = np.sum(np.abs(w))
    if gross > gross_cap:
        w = w * (gross_cap / gross)
    return w


def shrunk_idio_cov(resid_window: np.ndarray, idio_var: np.ndarray,
                    shrink: float) -> np.ndarray:
    """Ledoit-Wolf-style shrunk idiosyncratic covariance (NxN, DAILY): a convex blend of the
    sample residual covariance and the DIAGONAL idiosyncratic-variance target,
        Σ = shrink · diag(idio_var) + (1 − shrink) · sample_cov(residuals),
    plus a tiny ridge for numerical PD. With N > T the sample cov is rank-deficient, so
    `shrink` (toward the diagonal the diagonal-model already uses) makes Σ invertible and
    reduces EXACTLY to the inverse-variance risk model at shrink = 1."""
    x = np.nan_to_num(np.asarray(resid_window, float), nan=0.0)
    S = np.cov(x, rowvar=False)
    S = np.atleast_2d(S)
    D = np.diag(np.asarray(idio_var, float))
    n = D.shape[0]
    Sig = float(shrink) * D + (1.0 - float(shrink)) * S
    return Sig + 1e-10 * np.eye(n)                          # ridge -> guaranteed PD


def rebalance_idio_cov(stock_ret: pd.DataFrame, spy_ret: pd.Series, t, common: list[str],
                       beta_vec: np.ndarray, idio_var: np.ndarray, window: int,
                       shrink: float) -> np.ndarray:
    """Shrunk idiosyncratic covariance for `common` names from the trailing `window` daily
    returns up to t: residual_k = r_k − beta_k·r_spy (market removed), then `shrunk_idio_cov`."""
    hist = stock_ret.loc[:t, common].tail(window)
    spy_h = spy_ret.reindex(hist.index).fillna(0.0).to_numpy(float)
    resid = hist.to_numpy(float) - np.outer(spy_h, np.asarray(beta_vec, float))
    return shrunk_idio_cov(resid, idio_var, shrink)


def risk_target_book(aim: pd.Series, beta_row: pd.Series, var_row: pd.Series,
                     target_ann_vol: float, gross_cap: float, pos_cap: float | None,
                     beta_neutral: bool, sector_map: dict | None = None) -> pd.Series:
    """Re-scale the ACTUALLY-HELD book `aim` to the vol target and re-apply the
    caps, using the current day's risk model.

    The optimizer sizes the *target* w* to the vol budget, but the book actually
    held is the partial-step `aim`, whose gross (and therefore realized vol) is
    lower than w* -- and shrinks further the more often we rebalance, because the
    partial step averages successive (noisy) targets. Without this, realized risk
    and book size depend on `rebalance_freq` instead of the vol target. Applying
    the vol target to the held book makes realized risk frequency-invariant.

    Only names with a finite beta AND variance today are risk-scaled; the rest of
    `aim` is preserved. A degenerate (all-zero) book is returned unchanged."""
    names = [tk for tk in aim.index
             if np.isfinite(var_row.get(tk, np.nan))
             and np.isfinite(beta_row.get(tk, np.nan))]
    if not names:
        return aim
    w = aim[names].to_numpy(float)
    if not np.isfinite(w).any() or np.sum(np.abs(w)) == 0.0:
        return aim
    v = var_row[names].to_numpy(float)
    b = beta_row[names].to_numpy(float)
    sec = [sector_map.get(tk) for tk in names] if sector_map else None
    w = vol_target_scale(w, v, target_ann_vol, gross_cap)
    w = enforce_pos_cap(w, b, beta_neutral, pos_cap, sector_labels=sec)
    out = aim.copy()
    out[names] = w
    return out


# --------------------------------------------------------------------------- #
# Rolling risk inputs (self-contained: no dependency on cube betas)            #
# --------------------------------------------------------------------------- #
def regime_vol_scale(spy_ret: pd.Series, window: int = 63, target_vol: float = 0.15,
                     floor: float = 0.3, cap: float = 1.5,
                     min_periods: int | None = None) -> pd.Series:
    """Point-in-time exposure multiplier that DE-RISKS the whole book when the market
    is volatile -- a standard volatility-control overlay (exposure inversely
    proportional to realized vol, the user's "weights ~ 1/vol" idea applied to the
    sleeve, not just per name):

        scale_t = clip( target_vol / trailing_ann_vol(SPY)_t , floor, cap )

    Trailing SPY vol at t uses only returns up to close t, so the multiplier applied
    to the book held into the t->t+1 return is leak-free. In calm markets the ratio
    saturates at `cap` (lever up modestly); in a vol spike (2020, 2022) it collapses
    toward `floor`, so realized portfolio vol and drawdowns become far less sensitive
    to the vol regime. Warmup (no vol yet) -> neutral 1.0."""
    mp = int(min_periods) if min_periods is not None else max(20, window // 2)
    vol = spy_ret.rolling(window, min_periods=mp).std() * np.sqrt(_ANN)
    scale = (target_vol / vol).clip(lower=floor, upper=cap)
    return scale.fillna(1.0)


def rolling_beta_var(stock_ret: pd.DataFrame, spy_ret: pd.Series,
                     beta_window: int = 63, vol_window: int = 63,
                     min_obs: int | None = None):
    """Trailing market beta and idiosyncratic daily variance per name (point-in-
    time: uses only past returns). idio var = var(stock) - beta^2 * var(spy).

    NaN-tolerant. Daily returns (pct_change) always start with a NaN and carry
    scattered holes (suspensions, missing closes, recent IPOs); with the default
    rolling `min_periods == window` a SINGLE NaN anywhere in the trailing window
    blanks the estimate, so `beta_df` comes back almost entirely NaN and the day
    loop's `len(common) >= 10` check never passes -> the book never trades. And a
    NaN in `spy_ret` would blank the WHOLE cross-section for those windows. We set
    `min_periods` (default ~half the window, floored at 20, matching
    build_cube.betas.min_obs) so a window with a few gaps still yields a beta from
    the observations it does have; a genuinely data-poor name (too few obs) stays
    NaN and is correctly skipped."""
    b_min = int(min_obs) if min_obs is not None else max(20, beta_window // 2)
    v_min = int(min_obs) if min_obs is not None else max(20, vol_window // 2)
    var_spy = spy_ret.rolling(beta_window, min_periods=b_min).var()
    cov = stock_ret.rolling(beta_window, min_periods=b_min).cov(spy_ret)
    beta = cov.div(var_spy, axis=0)
    tot_var = stock_ret.rolling(vol_window, min_periods=v_min).var()
    idio_var = (tot_var - beta.pow(2).mul(var_spy, axis=0)).clip(lower=1e-8)
    return beta, idio_var


# --------------------------------------------------------------------------- #
# Backtest engine with the optimizer + turnover-aware stepping                 #
# --------------------------------------------------------------------------- #
def simulate_portfolio_opt(
    signal: pd.DataFrame,            # date x ticker cross-sectional score (z or rank)
    stock_ret: pd.DataFrame,
    spy_ret: pd.Series,
    starting_capital: float = 1_000_000,
    market_weight: float = 0.5,      # deliberate SPY beta sleeve
    target_ann_vol: float = 0.08,    # alpha-sleeve ex-ante vol target
    beta_neutral: bool = True,
    pos_cap: float = 0.03,
    gross_cap: float = 3.0,
    step: float = 0.35,              # partial trade toward target (turnover control)
    no_trade_band: float = 0.0,      # skip per-name trades smaller than this
    beta_window: int = 63,
    vol_window: int = 63,
    fee_bps: float = 1.0,
    spread_bps: float = 5.0,
    rebalance_freq: int = 1,
    sector_map: dict | None = None,   # ticker -> group (e.g. GICS industry group)
    sector_neutral: bool = False,     # enforce net sector exposure ~ 0
    risk_model: str = "diagonal",     # diagonal (inverse-variance) | covariance (correlation-aware)
    cov_shrink: float = 0.5,          # shrink toward the diagonal for the covariance risk model
    vol_scaling: bool = False,        # de-risk the whole book when the market is volatile
    regime_target_vol: float = 0.15,  # SPY ann-vol at which exposure multiplier = 1
    regime_vol_window: int = 63,
    regime_scale_floor: float = 0.3,  # min exposure multiplier (deep vol spike)
    regime_scale_cap: float = 1.5,    # max exposure multiplier (calm markets)
    collect_weights: bool = False,    # stash the per-day held per-name weights on out.attrs["weights"]
) -> pd.DataFrame:
    cost_rate = (fee_bps + spread_bps) / 1e4
    weights_hist: dict = {} if collect_weights else None
    beta_df, var_df = rolling_beta_var(stock_ret, spy_ret, beta_window, vol_window)
    smap = sector_map if (sector_neutral and sector_map) else None
    reg_scale = (regime_vol_scale(spy_ret, regime_vol_window, regime_target_vol,
                                  regime_scale_floor, regime_scale_cap)
                 if vol_scaling else None)

    dates = sorted(d for d in signal.index
                   if d in stock_ret.index and d in spy_ret.index
                   and d in beta_df.index)
    tickers = list(stock_ret.columns)
    prev_w = pd.Series(0.0, index=tickers + ["SPY"])
    V = spy_V = starting_capital
    rows = []

    target_alpha = pd.Series(0.0, index=tickers)
    for i in range(len(dates) - 1):
        t, t1 = dates[i], dates[i + 1]

        if i % rebalance_freq == 0:
            s = signal.loc[t].dropna()
            common = [tk for tk in s.index
                      if tk in beta_df.columns
                      and np.isfinite(beta_df.loc[t, tk]) and np.isfinite(var_df.loc[t, tk])]
            if len(common) >= 10:
                a = s[common].to_numpy(float)
                a = (a - a.mean()) / (a.std() if a.std() > 0 else 1.0)   # centered z
                b = beta_df.loc[t, common].to_numpy(float)
                v = var_df.loc[t, common].to_numpy(float)
                sec = [smap.get(tk) for tk in common] if smap else None
                # correlation-aware risk model: build the shrunk idiosyncratic covariance for
                # today's tradeable names (else None -> diagonal inverse-variance, as before)
                cov = (rebalance_idio_cov(stock_ret, spy_ret, t, common, b, v, vol_window, cov_shrink)
                       if risk_model == "covariance" else None)
                w_star = optimize_day(a, b, v, beta_neutral, pos_cap, sector_labels=sec, cov=cov)
                w_star = vol_target_scale(w_star, v, target_ann_vol, gross_cap, cov=cov)
                # enforce pos_cap on the FINAL weights (vol targeting rescales the
                # pre-scale cap applied inside optimize_day)
                w_star = enforce_pos_cap(w_star, b, beta_neutral, pos_cap, sector_labels=sec)
                target_alpha = pd.Series(0.0, index=tickers)
                target_alpha[common] = w_star

        # partial step toward target (Garleanu-Pedersen), then no-trade band
        aim = prev_w[tickers] + step * (target_alpha - prev_w[tickers])
        if no_trade_band > 0:
            delta = aim - prev_w[tickers]
            aim = prev_w[tickers] + delta.where(delta.abs() >= no_trade_band, 0.0)

        # risk-target the ACTUALLY-HELD book so realized vol/gross does not depend
        # on rebalance_freq (the partial step shrinks gross vs the target w*).
        aim = risk_target_book(aim, beta_df.loc[t], var_df.loc[t],
                               target_ann_vol, gross_cap, pos_cap, beta_neutral,
                               sector_map=smap)

        w = pd.Series(0.0, index=tickers + ["SPY"])
        w[tickers] = aim.values
        w["SPY"] = market_weight

        # market-regime vol overlay: shrink BOTH sleeves when SPY vol is high so the
        # portfolio de-risks through vol spikes (point-in-time, uses vol up to t).
        rs = float(reg_scale.get(t, 1.0)) if reg_scale is not None else 1.0
        if rs != 1.0:
            w = w * rs

        turnover = (w - prev_w).abs().sum()
        cost = turnover * cost_rate
        r_stocks = stock_ret.loc[t1, tickers].fillna(0.0)
        r_spy = spy_ret.loc[t1] if np.isfinite(spy_ret.loc[t1]) else 0.0
        # split the P&L into its two sleeves so each param's effect is observable
        alpha_ret = float((w[tickers] * r_stocks).sum())
        mkt_ret = float(w["SPY"] * r_spy)
        gross = alpha_ret + mkt_ret
        net = gross - cost
        V *= (1.0 + net)
        spy_V *= (1.0 + r_spy)
        rows.append({"date": t1, "gross_ret": gross, "cost": cost, "net_ret": net,
                     "turnover": turnover, "portfolio_value": V, "spy_value": spy_V,
                     # sleeve diagnostics (make the construction params visible)
                     "alpha_ret": alpha_ret, "mkt_ret": mkt_ret,
                     "alpha_gross": float(w[tickers].abs().sum()),
                     "alpha_max_w": float(w[tickers].abs().max()),
                     "regime_scale": rs})
        if weights_hist is not None:
            weights_hist[t1] = w[tickers].copy()               # held book (t->t1), for the blotter
        prev_w = w

    out = pd.DataFrame(rows).set_index("date")
    if weights_hist:
        out.attrs["weights"] = pd.DataFrame(weights_hist).T     # date x ticker held weights

    # Surface a silently-inactive alpha book instead of returning a flat curve.
    # (With market_weight=0 the SPY sleeve no longer masks a dead alpha sleeve.)
    if not out.empty and float(out["alpha_gross"].mean()) < 1e-6:
        _log.warning(
            "Alpha sleeve never established a position (avg gross ~0). Likely a "
            "degenerate signal (no cross-sectional dispersion) or < 10 names with a "
            "finite beta/variance on rebalance days. With market_weight=%.2f the "
            "portfolio trades effectively nothing.", market_weight)

    return out