"""Combining predictions (src/modelling/utils/ensemble.py).

Members are stubs with `predict(panel)`: the logic under test is pure numpy/pandas — per-day
z-scored averaging, horizon weights, the weighted nan-mean blend, and the long prediction rows."""

from __future__ import annotations

import warnings
from typing import cast

import numpy as np
import pandas as pd

from src.modelling.utils.ensemble import (
    PREDICTION_COLUMNS,
    blend_horizons,
    ensemble_predict,
    ir_horizon_weights,
    optimal_forecast_weights,
    prediction_rows,
    predicts_for,
)
from src.modelling.utils.metrics import daily_ic


class _Linear:
    """prediction = panel[feats] @ w (+ seeded noise, identical on every call)."""

    def __init__(self, w: list[float], feats: tuple[str, ...] = ("f0", "f1"), noise: float = 0.0, seed: int = 0) -> None:
        self.w, self.feats, self.noise, self.seed = np.asarray(w, float), list(feats), noise, seed

    def predict(self, panel: pd.DataFrame) -> np.ndarray:
        out = panel[self.feats].to_numpy(float) @ self.w
        return out + np.random.default_rng(self.seed).normal(0, self.noise, len(out)) if self.noise else out


def _panel(t: int = 40, n: int = 30, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    f0, f1 = rng.normal(0, 1, t * n), rng.normal(0, 1, t * n)
    return pd.DataFrame(
        {
            "date": np.repeat(pd.bdate_range("2021-01-01", periods=t), n),
            "ticker": np.tile([f"T{i:02d}" for i in range(n)], t),
            "f0": f0,
            "f1": f1,
            "y": f0 + rng.normal(0, 1.0, t * n),
        }
    )


def test_ensemble_is_per_day_average_of_member_zscores() -> None:
    panel = pd.DataFrame({"date": pd.to_datetime(["2020-01-01"] * 4), "ticker": list("ABCD"), "f0": [1.0, 2.0, 3.0, 4.0], "f1": 0.0})
    blended, members = ensemble_predict({"up": _Linear([1.0, 0.0]), "down": _Linear([-1.0, 0.0])}, panel)
    assert np.allclose(blended.to_numpy(), 0.0), "opposed members must cancel"
    single, _ = ensemble_predict({"up": _Linear([1.0, 0.0])}, panel)
    expected = (panel["f0"] - panel["f0"].mean()) / panel["f0"].std()
    assert np.allclose(single.to_numpy(), expected.to_numpy())
    print("\n=== SANITY CHECK: per-day z average ===")
    print(f"  opposed members -> {blended.round(6).tolist()}; one member -> its own per-day z (ddof=1). Validated.")


def test_ensemble_returns_members_and_blend_is_their_mean() -> None:
    panel = _panel()
    blended, members = ensemble_predict({"elasticnet": _Linear([1.0, 0.0]), "lgbm": _Linear([1.0, 0.0], noise=2.0, seed=7)}, panel)
    assert list(members) == ["elasticnet", "lgbm"]
    for s in members.values():
        assert s.index.equals(panel.index)
        stats = pd.DataFrame({"date": panel["date"].to_numpy(), "v": s.to_numpy()}).groupby("date")["v"].agg(["mean", "std"])
        assert np.allclose(stats["mean"], 0.0, atol=1e-9) and np.allclose(stats["std"], 1.0, atol=1e-6)
    stacked = np.column_stack([members[n].to_numpy() for n in members])
    assert np.allclose(blended.to_numpy(), np.nanmean(stacked, axis=1), atol=1e-12)
    print("\n=== SANITY CHECK: ensemble returns blend + members ===")
    print(f"  members {list(members)} in mapping order, each per-day z; blend == nanmean(members). Validated.")


def test_per_member_ic_separates_skill_from_noise() -> None:
    panel = _panel()
    _, members = ensemble_predict({"skilled": _Linear([1.0, 0.0]), "noise": _Linear([0.0, 0.0], noise=1.0, seed=3)}, panel)
    skilled, noise = daily_ic(panel, members["skilled"], "y"), daily_ic(panel, members["noise"], "y")
    assert skilled["mean_ic"] > 0.3 and abs(noise["mean_ic"]) < 0.15 and skilled["ic_ir"] > noise["ic_ir"]
    print("\n=== SANITY CHECK: per-member IC ===")
    print(f"  skilled IC {skilled['mean_ic']:+.3f} vs noise IC {noise['mean_ic']:+.3f}. Validated.")


def test_constant_member_contributes_nan_without_warning() -> None:
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01"] * 3 + ["2020-01-02"] * 3 + ["2020-01-03"]),
            "ticker": [f"T{i}" for i in range(7)],
            "f0": [1, 2, 3, 4, 5, 6, 9.0],
            "f1": 0.0,
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        preds, members = ensemble_predict({"elasticnet": _Linear([0.0, 0.0]), "lgbm": _Linear([1.0, 0.0])}, panel)
    assert np.isnan(members["elasticnet"].to_numpy()).all()
    assert np.isfinite(preds.to_numpy()[:6]).all() and np.isnan(preds.to_numpy()[6])
    print("\n=== SANITY CHECK: constant member ===")
    print("  constant member -> NaN; ensemble = the live member; single-name day -> NaN; no warning. Validated.")


def test_ir_horizon_weights_floor_and_fallback() -> None:
    assert ir_horizon_weights({30: 2.0, 60: 1.0, 90: -1.0}) == {30: 2 / 3, 60: 1 / 3, 90: 0.0}
    assert ir_horizon_weights({30: float("nan"), 60: 1.0}) == {30: 0.0, 60: 1.0}
    assert ir_horizon_weights({30: -1.0, 60: 0.0}) == {30: 0.5, 60: 0.5}
    assert list(ir_horizon_weights({60: 1.0, 30: 1.0})) == [60, 30], "keys keep the caller's horizon order"
    print("\n=== SANITY CHECK: IR horizon weights ===")
    print("  w ~ max(0, IR); NaN -> 0; no positive IR -> equal; caller's order kept. Validated.")


def test_blend_horizons_matches_the_inline_formula() -> None:
    rng = np.random.default_rng(5)
    z = rng.normal(size=(200, 3))
    z[rng.random((200, 3)) < 0.3] = np.nan
    w = np.array([0.5, 0.3, 0.2])
    mask = ~np.isnan(z)
    wmat = np.where(mask, w, 0.0)
    wsum = wmat.sum(axis=1)
    inline = np.where(wsum > 0, np.nansum(np.where(mask, z * w, 0.0), axis=1) / np.where(wsum > 0, wsum, 1), np.nan)
    got = blend_horizons(z, w)
    assert np.array_equal(np.isnan(got), np.isnan(inline)) and np.array_equal(got[~np.isnan(got)], inline[~np.isnan(inline)])
    print("\n=== SANITY CHECK: horizon blend ===")
    print(f"  bit-identical to the formula it replaces on 200 rows ({int(np.isnan(got).sum())} all-missing rows -> NaN). Validated.")


def _corr_signals(n: int = 4000, seed: int = 0) -> dict[int, np.ndarray]:
    rng = np.random.default_rng(seed)
    base = rng.normal(0, 1, n)
    return {30: base + rng.normal(0, 0.2, n), 60: base + rng.normal(0, 0.2, n), 90: rng.normal(0, 1, n)}


def test_optimal_forecast_weights_keeps_nan_ir_and_rewards_diversifier() -> None:
    sig = _corr_signals()
    w = optimal_forecast_weights(sig, {30: 2.0, 60: 1.5, 90: float("nan")}, shrink=0.5)
    assert set(w) == {30, 60, 90} and abs(sum(w.values()) - 1.0) < 1e-9 and w[90] > 0.05
    eq = optimal_forecast_weights(sig, {30: 1.0, 60: 1.0, 90: 1.0}, shrink=0.3)
    assert eq[90] > eq[30] and eq[90] > eq[60] and abs(eq[30] - eq[60]) < 0.05
    assert optimal_forecast_weights({30: sig[30]}, {30: 1.0}) == {30: 1.0}
    assert all(abs(v - 1 / 3) < 1e-9 for v in optimal_forecast_weights(sig, {30: np.nan, 60: np.nan, 90: np.nan}).values())
    assert all(abs(v - 1 / 3) < 1e-9 for v in optimal_forecast_weights(sig, {30: -1.0, 60: 0.0, 90: -2.0}).values())
    print("\n=== SANITY CHECK: correlation-aware horizon weights ===")
    print(
        f"  NaN-IR horizon kept (w90={w[90]:.3f}); equal IR -> diversifier up-weighted { ({h: round(v, 3) for h, v in eq.items()}) }; degenerate -> equal. Validated."
    )


def test_predicts_for_is_the_as_of_date_plus_horizon_trading_days() -> None:
    as_of = pd.Timestamp("2026-07-27")  # a Monday
    for h in (1, 5, 30, 60, 90):
        got = predicts_for(as_of, h)
        assert got == as_of + pd.tseries.offsets.BDay(h) and got.weekday() < 5
    assert (predicts_for(as_of, 30) - as_of).days == 42
    print("\n=== SANITY CHECK: predicts_for ===")
    print(f"  as-of {as_of.date()} + h30 trading days -> {predicts_for(as_of, 30).date()} (42 calendar days, never a weekend). Validated.")


def test_prediction_rows_are_long_and_stamped() -> None:
    keys = pd.DataFrame({"date": pd.to_datetime(["2026-07-24"] * 3 + ["2026-07-27"] * 2), "ticker": ["AAA", "BBB", "CCC", "AAA", "BBB"]})
    stamp = cast(pd.Timestamp, pd.Timestamp("2026-07-28 06:00:00"))
    out = prediction_rows(keys, np.array([1.0, 2.0, 3.0, 10.0, 20.0]), 30, "lgbm", stamp)
    assert list(out.columns) == PREDICTION_COLUMNS
    assert (out["horizon"] == 30).all() and (out["model"] == "lgbm").all() and (out["predicted_at"] == stamp).all()
    assert (out["predicts_for"] == out["date"].map(lambda d: cast(pd.Timestamp, d) + pd.tseries.offsets.BDay(30))).all()
    for _, g in out.groupby("date"):
        assert abs(float(g["pred"].mean())) < 1e-9
    d0 = out[out["date"] == pd.Timestamp("2026-07-24")].sort_values("pred")
    assert list(d0["rank"]) == sorted(d0["rank"])
    print("\n=== SANITY CHECK: long prediction rows ===")
    print(out.to_string(index=False))
    print("  per-day z-scored pred, monotone per-day rank, predicts_for per row. Validated.")
