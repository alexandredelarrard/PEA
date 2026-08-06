"""Sanity checks for the factor-neutral target (src/data_aggregate/utils/targets.py).

These run on REAL data (small sample incl. AMD) because the defects here only
appear with real NaNs / late IPOs:

  1. Coverage        -> a long-listed stock's target must span its history, not
                        collapse to a short recent window (regression test for
                        the sector-NaN -> dropna truncation bug that inflated the
                        long-horizon target).
  2. Cross-sectional -> the daily rank target is centered at ~0.5 by construction.
  3. Per-stock bias  -> after the fix, AMD's mean target sits near 0.5 at every
                        horizon (it used to climb to ~0.7 at h=60).
  4. Momentum-neutral -> the target is cross-sectionally orthogonalized against
                        the 12-1 momentum characteristic each day, so on any
                        given day the ranking carries no momentum tilt.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import warnings

from src.data_aggregate.utils.target.factors import momentum_characteristic
from src.data_aggregate.utils.target.targets import cross_sectional_neutralize, forward_compound


def test_forward_compound_no_log1p_warning_on_sub_minus1_return():
    """A return worse than -100% (e.g. the 2020 negative-oil-futures move) must not
    emit 'invalid value encountered in log1p', and must stay finite (floored)."""
    s = pd.Series([0.01, -1.8, 0.02, 0.03, -0.02, 0.01, np.nan, 0.00])
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)   # any RuntimeWarning -> failure
        fwd = forward_compound(s, 2)
    # windows that don't span the NaN produce finite compounded returns
    assert np.isfinite(fwd.dropna()).all() and fwd.notna().any()

    print("\n=== SANITY CHECK: forward_compound guards log1p ===")
    print("  return -180% floored just above -1 -> no log1p warning, finite forward compound. Validated.")


def _per_stock_mean_beta(betas: dict, col: str) -> pd.Series:
    return pd.Series({t: betas[t][col].mean() for t in betas if col in betas[t]})


def _xs_corr(a: pd.Series, b: pd.Series) -> float:
    common = a.index.intersection(b.index)
    aa = a.loc[common]
    bb = b.loc[common]
    ok = aa.notna() & bb.notna()
    if ok.sum() < 10:
        return np.nan
    return float(np.corrcoef(aa[ok], bb[ok])[0, 1])


# --------------------------------------------------------------------------- #
# 1. Coverage: target must span history, not collapse to a recent window       #
# --------------------------------------------------------------------------- #
def test_amd_target_spans_history(real_pipeline):
    labels = real_pipeline["labels_rank"]
    h = 60
    if "AMD" not in labels[h].columns:
        pytest.skip("AMD not in sampled universe")

    amd = labels[h]["AMD"]
    valid = amd.dropna()
    assert len(valid) > 0, "AMD has no valid target at all"

    span_years = (valid.index.max() - valid.index.min()).days / 365.25
    coverage = amd.loc[amd.first_valid_index():].notna().mean()

    assert span_years > 4.0, (
        f"AMD target only spans {span_years:.1f} years -> history was truncated "
        "(sector-NaN bug). Expected multi-year coverage."
    )
    assert coverage > 0.5, f"AMD target coverage only {coverage:.1%} after first valid date"

    print("\n=== SANITY CHECK: AMD target coverage (h=60) ===")
    print(f"  valid range : {valid.index.min().date()} -> {valid.index.max().date()}")
    print(f"  spans       : {span_years:.1f} years, coverage={coverage:.0%}")
    print("  -> target no longer collapses to a recent window. Bug fixed.")


# --------------------------------------------------------------------------- #
# 2. Cross-sectional centering                                                 #
# --------------------------------------------------------------------------- #
def test_rank_target_is_centered(real_pipeline):
    labels = real_pipeline["labels_rank"]
    for h in real_pipeline["horizons"]:
        daily_mean = labels[h].mean(axis=1)
        m = daily_mean.mean()
        assert abs(m - 0.5) < 0.02, f"h={h}: daily rank mean {m:.3f} not ~0.5"
    print("\n=== SANITY CHECK: rank target centered at 0.5 ===")
    for h in real_pipeline["horizons"]:
        print(f"  h={h:>3}: mean daily xs-rank = {labels[h].mean(axis=1).mean():.3f}")
    print("  -> cross-sectional rank is correctly centered.")


# --------------------------------------------------------------------------- #
# 3. Per-stock bias: AMD near-neutral across horizons after the fix            #
# --------------------------------------------------------------------------- #
def test_amd_not_persistently_top_ranked(real_pipeline):
    labels = real_pipeline["labels_rank"]
    if "AMD" not in labels[60].columns:
        pytest.skip("AMD not in sampled universe")

    print("\n=== SANITY CHECK: AMD mean target by horizon (0.5 = neutral) ===")
    means = {}
    for h in real_pipeline["horizons"]:
        means[h] = labels[h]["AMD"].mean()
        print(f"  h={h:>3}: AMD mean rank = {means[h]:.3f}")

    # Pre-fix this climbed to ~0.71 at h=60; after the fix it must be far tamer.
    assert means[60] < 0.62, (
        f"AMD h=60 mean rank {means[60]:.3f} still strongly biased -> "
        "history-truncation bug likely back."
    )
    print("  -> AMD sits near neutral; no longer 'almost always beating' at h=60.")


# --------------------------------------------------------------------------- #
# 4a. Unit test: the neutralizer removes a cross-sectional factor tilt          #
# --------------------------------------------------------------------------- #
def test_cross_sectional_neutralize_removes_tilt():
    """Deterministic proof the orthogonalization works: build values that are,
    each day, a strong linear function of a factor plus noise; after
    neutralization the per-day cross-sectional correlation with that factor
    must be ~0."""
    rng = np.random.default_rng(3)
    dates = pd.bdate_range("2021-01-01", periods=60)
    tickers = [f"T{i:02d}" for i in range(40)]

    factor = pd.DataFrame(rng.normal(0, 1, (len(dates), len(tickers))),
                          index=dates, columns=tickers)
    values = 2.0 * factor + pd.DataFrame(
        rng.normal(0, 0.3, (len(dates), len(tickers))), index=dates, columns=tickers
    )

    before = values.corrwith(factor, axis=1).mean()
    neutral = cross_sectional_neutralize(values, factor)
    after = neutral.corrwith(factor, axis=1).mean()

    assert abs(before) > 0.9, "sanity: values should start highly correlated"
    assert abs(after) < 1e-6, f"neutralized values still correlate with factor: {after:+.4f}"

    print("\n=== SANITY CHECK: cross_sectional_neutralize removes tilt ===")
    print(f"  per-day corr(values, factor) : before={before:+.3f}  after={after:+.6f}")
    print("  -> orthogonalization drives the cross-sectional correlation to zero.")


# --------------------------------------------------------------------------- #
# 4b. Integration: the real target is momentum-neutral each day                #
# --------------------------------------------------------------------------- #
def test_target_orthogonal_to_momentum_per_day(real_pipeline):
    """The meaningful factor-neutrality criterion for a daily cross-sectional
    picker: on each day the target ranking must carry no momentum tilt."""
    labels = real_pipeline["labels_rank"]
    stock_close = real_pipeline["stock_close"]
    mom = momentum_characteristic(stock_close)

    print("\n=== SANITY CHECK: per-day xs corr(target, momentum characteristic) ===")
    for h in real_pipeline["horizons"]:
        tgt = labels[h]
        daily = tgt.corrwith(mom.reindex_like(tgt), axis=1)
        mean_corr = float(daily.mean())
        print(f"  h={h:>3}: mean daily xs-corr(target, mom_12_1) = {mean_corr:+.3f}")
        assert abs(mean_corr) < 0.05, (
            f"h={h}: target still tilts to momentum per day (corr={mean_corr:+.3f})"
        )
    print("  -> target is orthogonal to the momentum characteristic each day.")


# --------------------------------------------------------------------------- #
# 4c. Note on the cross-STOCK (time-mean) correlation                          #
# --------------------------------------------------------------------------- #
def test_time_mean_momentum_correlation_is_persistent_alpha(real_pipeline):
    """The cross-STOCK correlation between the time-mean target and time-mean
    beta_momentum stays positive AFTER per-day neutralization. That is expected:
    it reflects persistent idiosyncratic alpha (chronic winners keep beating
    their factor-matched peers), which is signal we WANT to keep -- not a
    per-day momentum tilt. This test documents/guards that interpretation."""
    labels = real_pipeline["labels_rank"]
    betas = real_pipeline["betas"]
    h = 60

    tgt_mean = labels[h].mean(axis=0)
    bmom = _per_stock_mean_beta(betas, "beta_momentum")
    corr = _xs_corr(tgt_mean, bmom)

    print("\n=== NOTE: cross-stock time-mean corr(target, beta_momentum) ===")
    print(f"  h={h}: {corr:+.3f}  (persistent alpha, NOT a per-day tilt -> kept)")
    assert corr > 0.0, "persistent-alpha structure unexpectedly vanished"
