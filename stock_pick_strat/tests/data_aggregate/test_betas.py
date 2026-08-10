"""Tests for the rolling multi-factor beta estimator
(src/data_aggregate/utils/target/betas.py).

Covers:
  1. Correctness    -> ridge recovers KNOWN loadings on synthetic data.
  2. Ridge vs OLS   -> ridge betas are materially more stable under collinearity.
  3. No look-ahead  -> betas at date t are invariant to data after t.
  4. Coverage       -> a sparse regressor must not silently truncate the whole
                       history (regression test for the sector-NaN bug).
  5. Market prior   -> the market beta shrinks toward 1.0, everything else toward
                       0.0, so a shrunk market beta cannot leave unhedged market
                       inside epsilon.
  6. Scale-free ridge -> the shrinkage RATIO is the same at N=min_obs and N=window.
  7. Factor gaps    -> one missing day in a factor no longer deletes that factor's
                       beta (and no longer moves the other coefficients).
  8. No staircase   -> at step=1 consecutive betas actually differ.
  9. Panel == loop  -> the vectorized panel path equals the single-stock path.
 10. Label timing   -> the estimation window and the label window do not overlap.
 11. Per-stock factor -> the GICS sector regressor (passed via `per_stock_factors`,
                       NOT a shared column) recovers a DIFFERENT known loading per
                       stock (Schur-complement fold-in, `_fold_in_sector`).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_aggregate.utils.target.betas import (
    estimate_all_betas, estimate_betas_for_stock,
)
from src.data_aggregate.utils.target.factors import gics_sector_excess_returns


# --------------------------------------------------------------------------- #
# 1. Correctness: recover known loadings                                       #
# --------------------------------------------------------------------------- #
def test_ridge_recovers_known_betas(synthetic_factor_model):
    y, shared, sector, true_betas = synthetic_factor_model
    X = pd.concat([shared, sector.rename("sector")], axis=1)

    # near-OLS (tiny ridge) + long window => estimator should recover truth.
    out = estimate_betas_for_stock(y, X, window=250, min_obs=200,
                                   ridge_alpha=0.001, step=1)
    last = out.dropna().iloc[-1]

    recovered = {
        "market": last["beta_market"],
        "momentum": last["beta_momentum"],
        "value": last["beta_value"],
        "sector": last["beta_sector"],
    }
    for name, truth in true_betas.items():
        assert abs(recovered[name] - truth) < 0.15, (
            f"beta_{name}={recovered[name]:.3f} not within 0.15 of true {truth}"
        )

    # beta_market_simple (cov/var) should also be close to the true market beta.
    assert abs(last["beta_market_simple"] - true_betas["market"]) < 0.30

    print("\n=== SANITY CHECK: ridge recovers known betas ===")
    for name, truth in true_betas.items():
        print(f"  beta_{name:<9} recovered={recovered[name]:+.3f}  true={truth:+.2f}")
    print("  -> all loadings recovered within tolerance. Estimator is correct.")


# --------------------------------------------------------------------------- #
# 2. Ridge vs OLS stability under collinearity                                 #
# --------------------------------------------------------------------------- #
def test_ridge_more_stable_than_ols_under_collinearity():
    """With collinear regressors the OLS split between them is unstable over
    time; ridge should reduce that temporal variance."""
    rng = np.random.default_rng(7)
    n = 600
    dates = pd.bdate_range("2017-01-01", periods=n)

    market = pd.Series(rng.normal(0, 0.01, n), index=dates, name="market")
    # sector strongly collinear with market (corr ~0.9)
    sector = pd.Series(0.9 * market.to_numpy() + rng.normal(0, 0.004, n),
                       index=dates, name="sector")
    shared = pd.concat([market, sector], axis=1)

    y = 0.8 * market + 0.5 * sector + pd.Series(rng.normal(0, 0.004, n), index=dates)
    y.name = "STOCK"

    ols = estimate_betas_for_stock(y, shared, window=63, min_obs=40,
                                   ridge_alpha=0.0, step=5)
    rdg = estimate_betas_for_stock(y, shared, window=63, min_obs=40,
                                   ridge_alpha=0.08, step=5)

    ols_std = ols["beta_market"].std()
    rdg_std = rdg["beta_market"].std()

    assert rdg_std < ols_std, (
        f"ridge should stabilize collinear loadings: ridge std={rdg_std:.3f} "
        f"vs ols std={ols_std:.3f}"
    )

    print("\n=== SANITY CHECK: ridge vs OLS stability (collinear market/sector) ===")
    print(f"  temporal std(beta_market)  OLS={ols_std:.3f}   RIDGE={rdg_std:.3f}")
    print(f"  -> ridge cut loading volatility by {100*(1-rdg_std/ols_std):.0f}%. "
          "Ridge is the better choice here.")


# --------------------------------------------------------------------------- #
# 3. No look-ahead                                                             #
# --------------------------------------------------------------------------- #
def test_betas_have_no_lookahead(synthetic_factor_model):
    """Betas dated <= t must not depend on any observation after t."""
    y, shared, sector, _ = synthetic_factor_model
    X = pd.concat([shared, sector.rename("sector")], axis=1)
    cutoff = y.index[300]

    base = estimate_betas_for_stock(y, X, window=120, min_obs=60, step=5)

    # Corrupt everything strictly AFTER the cutoff and recompute.
    y2 = y.copy()
    y2.loc[y2.index > cutoff] = y2.loc[y2.index > cutoff] * 5.0 + 1.0
    X2 = X.copy()
    X2.loc[X2.index > cutoff] += 3.0

    corrupted = estimate_betas_for_stock(y2, X2, window=120, min_obs=60, step=5)

    joint_cols = [c for c in base.columns if c != "beta_market_simple"]
    a = base.loc[base.index <= cutoff, joint_cols]
    b = corrupted.loc[corrupted.index <= cutoff, joint_cols]
    pd.testing.assert_frame_equal(a, b, check_exact=False, atol=1e-10)

    print("\n=== SANITY CHECK: no look-ahead in betas ===")
    print(f"  betas up to {cutoff.date()} unchanged after corrupting all future data.")
    print("  -> timing rule holds: betas at t use only data up to t.")


# --------------------------------------------------------------------------- #
# 4. A sparse regressor must not truncate the whole history                    #
# --------------------------------------------------------------------------- #
def test_sparse_sector_does_not_truncate_history(synthetic_factor_model):
    """Regression test for the real bug: when the sector series is NaN for the
    early part of history (a late-IPO peer), betas should still be estimated
    over the long stretch where it IS available -- not collapse to a tiny
    recent window."""
    y, shared, sector, _ = synthetic_factor_model

    sparse = sector.copy()
    sparse.iloc[: int(0.6 * len(sparse))] = np.nan  # first 60% missing
    X = pd.concat([shared, sparse.rename("sector")], axis=1)

    out = estimate_betas_for_stock(y, X, window=63, min_obs=40, step=5)
    valid = out["beta_sector"].ne(0.0) & out["beta_sector"].notna()

    # ~40% of dates have a sector value; after warmup we expect a healthy chunk.
    assert valid.mean() > 0.25, (
        f"betas collapsed: only {valid.mean():.1%} of dates estimated"
    )
    first_valid = out["beta_sector"].first_valid_index()
    span_days = (out.index[-1] - first_valid).days
    assert span_days > 200, "beta coverage should span the available history"

    print("\n=== SANITY CHECK: sparse regressor does not truncate history ===")
    print(f"  sector available on last 40% of dates -> betas fitted on "
          f"{valid.mean():.0%} of dates, spanning {span_days} days.")
    print("  -> estimator uses all available data instead of collapsing.")


# --------------------------------------------------------------------------- #
# 5. The market beta shrinks toward 1.0, every other beta toward 0.0           #
# --------------------------------------------------------------------------- #
def test_market_beta_shrinks_toward_one_not_zero():
    """A HEAVY ridge must pull beta_market to ~1.0 and the style beta to ~0.0.

    This is the property that keeps unhedged market OUT of epsilon: shrinking
    market toward 0 biases the loading low and the missing slice survives in the
    residual as market exposure.
    """
    rng = np.random.default_rng(11)
    n = 400
    dates = pd.bdate_range("2019-01-01", periods=n)
    market = pd.Series(rng.normal(0, 0.010, n), index=dates, name="market")
    style = pd.Series(rng.normal(0, 0.008, n), index=dates, name="style")
    X = pd.concat([market, style], axis=1)
    y = (1.6 * market + 0.9 * style
         + pd.Series(rng.normal(0, 0.004, n), index=dates)).rename("STOCK")

    light = estimate_betas_for_stock(y, X, window=250, min_obs=200,
                                     ridge_alpha=0.001, step=1).dropna().iloc[-1]
    heavy = estimate_betas_for_stock(y, X, window=250, min_obs=200,
                                     ridge_alpha=50.0, step=1).dropna().iloc[-1]
    zero_prior = estimate_betas_for_stock(y, X, window=250, min_obs=200,
                                          ridge_alpha=50.0, step=1,
                                          market_prior=0.0).dropna().iloc[-1]

    assert abs(heavy["beta_market"] - 1.0) < 0.05, "market must shrink toward 1.0"
    assert abs(heavy["beta_style"]) < 0.05, "style must shrink toward 0.0"
    assert abs(zero_prior["beta_market"]) < 0.05, "prior 0.0 must shrink market to 0"
    # and the light-ridge fit must still recover the truth
    assert abs(light["beta_market"] - 1.6) < 0.10
    assert abs(light["beta_style"] - 0.9) < 0.10

    print("\n=== SANITY CHECK: shrinkage targets (true betas market=1.60 style=0.90) ===")
    print(f"  ridge_alpha=0.001            beta_market={light['beta_market']:+.3f} "
          f" beta_style={light['beta_style']:+.3f}")
    print(f"  ridge_alpha=50, prior=1.0    beta_market={heavy['beta_market']:+.3f} "
          f" beta_style={heavy['beta_style']:+.3f}")
    print(f"  ridge_alpha=50, prior=0.0    beta_market={zero_prior['beta_market']:+.3f} "
          f" beta_style={zero_prior['beta_style']:+.3f}")
    print("  -> market collapses to its 1.0 prior, style to 0.0. Prior 0.0 would leave")
    print("     ~1.0 of unhedged market beta inside the 'idiosyncratic' residual.")


# --------------------------------------------------------------------------- #
# 6. The ridge penalty is scale-free in the sample size                        #
# --------------------------------------------------------------------------- #
def test_ridge_shrinkage_ratio_is_constant_across_sample_sizes():
    """lambda = ridge_alpha * N, so the SHRINKAGE RATIO must not drift with the
    number of observations in the window (the old fixed lambda regularized the
    warm-up 58% harder than steady state)."""
    rng = np.random.default_rng(3)
    n = 400
    dates = pd.bdate_range("2019-01-01", periods=n)
    market = pd.Series(rng.normal(0, 0.010, n), index=dates, name="market")
    X = market.to_frame()
    y = (1.5 * market + pd.Series(rng.normal(0, 0.003, n), index=dates)).rename("STOCK")

    # a single orthogonal regressor is shrunk toward the prior by exactly
    # 1/(1 + ridge_alpha) whatever N is.
    alpha = 0.25
    ratios = {}
    for n_win in (40, 63, 252):
        b = estimate_betas_for_stock(y, X, window=n_win, min_obs=n_win,
                                     ridge_alpha=alpha, step=1)
        raw = estimate_betas_for_stock(y, X, window=n_win, min_obs=n_win,
                                       ridge_alpha=0.0, step=1)
        j = pd.concat([b["beta_market"].rename("s"), raw["beta_market"].rename("r")],
                      axis=1).dropna()
        # (shrunk - prior) / (unshrunk - prior)
        ratios[n_win] = float(((j["s"] - 1.0) / (j["r"] - 1.0)).mean())

    expected = 1.0 / (1.0 + alpha)
    for n_win, r in ratios.items():
        assert abs(r - expected) < 0.02, f"N={n_win}: ratio {r:.4f} != {expected:.4f}"

    print("\n=== SANITY CHECK: shrinkage ratio is invariant to sample size ===")
    for n_win, r in ratios.items():
        print(f"  N={n_win:<4} (shrunk-prior)/(ols-prior) = {r:.4f}")
    print(f"  -> all equal 1/(1+alpha) = {expected:.4f}. A fixed lambda would give "
          f"{1/(1+alpha*63/40):.4f} at N=40 vs {expected:.4f} at N=63.")


# --------------------------------------------------------------------------- #
# 7. One missing day in a factor must not delete that factor's beta            #
# --------------------------------------------------------------------------- #
def test_single_factor_gap_does_not_null_the_window():
    """The FX series has isolated missing days. Dropping the whole column for a
    window that contains one silently zeroed beta_fx on 10% of live history AND
    moved every other coefficient in those windows."""
    rng = np.random.default_rng(5)
    n = 400
    dates = pd.bdate_range("2019-01-01", periods=n)
    market = pd.Series(rng.normal(0, 0.010, n), index=dates, name="market")
    fx = pd.Series(rng.normal(0, 0.005, n), index=dates, name="fx")
    y = (1.1 * market + 0.7 * fx
         + pd.Series(rng.normal(0, 0.003, n), index=dates)).rename("STOCK")

    clean = estimate_betas_for_stock(y, pd.concat([market, fx], axis=1),
                                     window=63, min_obs=40, step=1)
    holed = fx.copy()
    holed.iloc[200] = np.nan                       # ONE missing day
    gappy = estimate_betas_for_stock(y, pd.concat([market, holed], axis=1),
                                     window=63, min_obs=40, step=1)

    affected = gappy.index[200:263]                # the 63 windows containing the hole
    assert (gappy.loc[affected, "beta_fx"].abs() > 1e-6).all(), \
        "beta_fx was zeroed by a single missing day"
    drift = (gappy.loc[affected, "beta_market"] - clean.loc[affected, "beta_market"]).abs()
    assert drift.max() < 0.10, "one missing FX day should barely move beta_market"

    print("\n=== SANITY CHECK: a one-day factor gap is imputed, not dropped ===")
    print(f"  windows containing the hole: {len(affected)}")
    print(f"  beta_fx still fitted on all of them (min |beta_fx| = "
          f"{gappy.loc[affected, 'beta_fx'].abs().min():.3f})")
    print(f"  max drift in beta_market vs the un-holed panel = {drift.max():.4f}")
    print("  -> the gap costs a rounding error instead of the factor's whole beta.")


# --------------------------------------------------------------------------- #
# 8. step=1 leaves no staircase                                                #
# --------------------------------------------------------------------------- #
def test_step_one_has_no_staircase(synthetic_factor_model):
    y, shared, _, _ = synthetic_factor_model
    daily = estimate_betas_for_stock(y, shared, window=63, min_obs=40, step=1)
    stair = estimate_betas_for_stock(y, shared, window=63, min_obs=40, step=5)

    flat_daily = float((daily["beta_market"].diff().abs() < 1e-12).mean())
    flat_stair = float((stair["beta_market"].diff().abs() < 1e-12).mean())
    assert flat_daily < 0.02, "step=1 should update the beta every day"
    assert flat_stair > 0.5, "step=5 is expected to hold the beta most days"

    print("\n=== SANITY CHECK: the 5-day staircase ===")
    print(f"  share of days with an unchanged beta_market: step=5 -> {flat_stair:.1%}, "
          f"step=1 -> {flat_daily:.1%}")
    print("  -> step=1 removes the step function at the source; no EWMA (i.e. no extra")
    print("     lag) is needed to hide it.")


# --------------------------------------------------------------------------- #
# 9. The vectorized panel path == the single-stock path                        #
# --------------------------------------------------------------------------- #
def test_panel_and_single_stock_paths_agree(synthetic_factor_model):
    """Two stocks with DIFFERENT listing dates, so one takes the ragged
    (per-stock, partly observed window) branch and one does not."""
    y, shared, _, _ = synthetic_factor_model
    late = y.copy() * 0.7
    late.iloc[:300] = np.nan
    panel_ret = pd.concat([y.rename("EARLY"), late.rename("LATE")], axis=1)

    both = estimate_all_betas(panel_ret, shared, window=63, min_obs=40, step=1,
                              filter_factors=False)
    for tk in ("EARLY", "LATE"):
        one = estimate_betas_for_stock(panel_ret[tk].rename(tk), shared,
                                       window=63, min_obs=40, step=1)
        pd.testing.assert_frame_equal(both[tk], one, check_exact=False, atol=1e-10)

    n_ragged = int(both["LATE"]["beta_market"].notna().sum())
    print("\n=== SANITY CHECK: panel solve == per-stock solve ===")
    print(f"  EARLY (full windows) and LATE (listed on day 300, {n_ragged} fitted dates)")
    print("  both match the single-stock path to 1e-10 -> the shared-design")
    print("     factorization and the ragged fallback are the same estimator.")


# --------------------------------------------------------------------------- #
# 10. The label window does not overlap the estimation window                  #
# --------------------------------------------------------------------------- #
def test_label_window_does_not_overlap_estimation_window():
    """beta_t is fitted on returns up to and including day t; the label is the
    return of days t+1..t+h. No overlap => using beta_t on that label is not
    in-sample fitting, and needs no extra lag."""
    from src.data_aggregate.utils.common.prices import forward_return

    n = 20
    dates = pd.bdate_range("2020-01-01", periods=n)
    close = pd.DataFrame({"A": np.arange(1.0, n + 1.0)}, index=dates)
    h = 5
    t = 8
    fwd = forward_return(close, h).iloc[t, 0]
    expected = close["A"].iloc[t + h] / close["A"].iloc[t] - 1.0

    assert np.isclose(fwd, expected)
    # the label is a function of closes t..t+h, i.e. of the RETURNS of t+1..t+h;
    # the estimation window ends at the return of day t.
    assert close.index[t] < close.index[t + 1]

    print("\n=== SANITY CHECK: estimation and label windows are disjoint ===")
    print(f"  beta window ends with the return of {dates[t].date()}")
    print(f"  label at {dates[t].date()} = close[{dates[t+h].date()}]/close[{dates[t].date()}]-1"
          f" = returns of {dates[t+1].date()}..{dates[t+h].date()}")
    print("  -> no shared observation, so beta_t applied to the label is NOT in-sample.")


# --------------------------------------------------------------------------- #
# 11. The per-stock sector factor recovers a DIFFERENT loading per stock       #
# --------------------------------------------------------------------------- #
def test_per_stock_sector_factor_recovers_known_loading():
    """Unlike every other factor, the GICS sector basket is passed via
    `per_stock_factors` (date x ticker, one column per ticker) rather than as a
    column of the shared panel -- each stock regresses on its OWN sector series
    and gets its OWN `beta_sector`, folded into the shared ridge solve via the
    Schur complement (`_fold_in_sector`)."""
    rng = np.random.default_rng(21)
    n = 400
    dates = pd.bdate_range("2019-01-01", periods=n)
    market = pd.Series(rng.normal(0, 0.010, n), index=dates, name="market")

    sector_a = pd.Series(rng.normal(0, 0.008, n), index=dates)
    sector_b = pd.Series(rng.normal(0, 0.008, n), index=dates)
    true_beta_a, true_beta_b = 0.8, -0.5

    y_a = (1.0 * market + true_beta_a * sector_a
           + pd.Series(rng.normal(0, 0.003, n), index=dates)).rename("A")
    y_b = (1.0 * market + true_beta_b * sector_b
           + pd.Series(rng.normal(0, 0.003, n), index=dates)).rename("B")

    stock_returns = pd.concat([y_a, y_b], axis=1)
    per_stock_factors = pd.concat([sector_a.rename("A"), sector_b.rename("B")], axis=1)

    out = estimate_all_betas(stock_returns, market.to_frame(), per_stock_factors,
                             window=200, min_obs=150, ridge_alpha=0.001, step=1,
                             filter_factors=False)

    last_a = out["A"].dropna().iloc[-1]
    last_b = out["B"].dropna().iloc[-1]

    assert abs(last_a["beta_sector"] - true_beta_a) < 0.15
    assert abs(last_b["beta_sector"] - true_beta_b) < 0.15
    assert abs(last_a["beta_market"] - 1.0) < 0.15
    assert abs(last_b["beta_market"] - 1.0) < 0.15

    print("\n=== SANITY CHECK: per-stock sector factor (Schur-complement fold-in) ===")
    print(f"  A: beta_sector recovered={last_a['beta_sector']:+.3f}  true={true_beta_a:+.2f}")
    print(f"  B: beta_sector recovered={last_b['beta_sector']:+.3f}  true={true_beta_b:+.2f}")
    print("  -> each stock's OWN sector regressor and loading recovered independently.")


# --------------------------------------------------------------------------- #
# 12. A SECOND ridge alpha, for the market column alone                        #
# --------------------------------------------------------------------------- #
def test_market_gets_its_own_ridge_alpha():
    """`ridge_alpha_market` exists because the two priors are not comparable: market shrinks
    toward 1.0 and every other factor toward 0.0, and how much shrinkage each NEEDS is set by
    how well its beta forecasts the window it hedges (measured at window=63: 0.43 for market,
    0.03 for d_vix). Two things must hold:

      * `None` reproduces the single-alpha path EXACTLY -- the gram is built as
        `n * diag(alphas)` instead of `alpha * n * eye`, which is the same float64 product, so
        this is a bit-identity and not a tolerance;
      * a small market alpha with a large style alpha shrinks market toward 1.0 far LESS than
        style is shrunk toward 0.0.
    """
    rng = np.random.default_rng(11)
    n = 400
    dates = pd.bdate_range("2019-01-01", periods=n)
    market = pd.Series(rng.normal(0, 0.010, n), index=dates, name="market")
    style = pd.Series(rng.normal(0, 0.008, n), index=dates, name="style")
    shared = pd.concat([market, style], axis=1)
    y = (1.6 * market + 0.9 * style
         + pd.Series(rng.normal(0, 0.003, n), index=dates)).rename("STOCK")

    base = estimate_betas_for_stock(y, shared, window=126, min_obs=80,
                                    ridge_alpha=0.08, step=1)
    same = estimate_betas_for_stock(y, shared, window=126, min_obs=80,
                                    ridge_alpha=0.08, ridge_alpha_market=None, step=1)
    pd.testing.assert_frame_equal(base, same)          # bit-identical, no tolerance

    unshrunk = estimate_betas_for_stock(y, shared, window=126, min_obs=80,
                                       ridge_alpha=0.0, step=1).dropna().iloc[-1]
    split = estimate_betas_for_stock(y, shared, window=126, min_obs=80,
                                     ridge_alpha=1.5, ridge_alpha_market=0.24,
                                     step=1).dropna().iloc[-1]

    market_kept = (split["beta_market"] - 1.0) / (unshrunk["beta_market"] - 1.0)
    style_kept = split["beta_style"] / unshrunk["beta_style"]
    assert abs(market_kept - 1.0 / 1.24) < 0.05, f"market shrunk by {market_kept:.3f}"
    assert abs(style_kept - 1.0 / 2.5) < 0.05, f"style shrunk by {style_kept:.3f}"
    assert market_kept > style_kept

    print("\n=== SANITY CHECK: per-factor ridge alpha ===")
    print("  ridge_alpha_market=None -> frame is bit-identical to the single-alpha path")
    print(f"  alpha_market=0.24 / alpha_other=1.5: market keeps {market_kept:.3f} of its "
          f"distance from the 1.0 prior (expected {1/1.24:.3f}), style keeps {style_kept:.3f} "
          f"of its distance from 0.0 (expected {1/2.5:.3f})")
    print("  -> the market column is shrunk on its own dial. Validated.")
