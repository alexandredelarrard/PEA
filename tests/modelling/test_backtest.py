"""Label-only quantile backtest (src/modelling/transformers/backtest.py), known-truth cases:
a perfect / inverted prediction, fixed vs rotating extreme buckets, thin days, the IR
annualisation, and the saved artifacts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from src.modelling.transformers.backtest import Backtest

N_TICKERS = 50


def _frame(n_days: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for d in pd.bdate_range("2022-01-03", periods=n_days):
        y = rng.permutation(N_TICKERS) / (N_TICKERS - 1)  # a rank label in [0, 1]
        rows.append(pd.DataFrame({"date": d, "ticker": [f"T{i:02d}" for i in range(N_TICKERS)], "y": y}))
    return pd.concat(rows, ignore_index=True)


def _bt(n: int = 10) -> Backtest:
    return Backtest(OmegaConf.create({"model": {"backtest": {"n_quantiles": n}}}))


def test_perfect_prediction_gives_monotone_profile_and_no_drawdown() -> None:
    frame = _frame()
    res = _bt().run(frame.assign(pred=frame["y"]), label_col="y", horizon=30)
    profile = res.buckets.mean().to_numpy()
    s = res.summary
    assert (np.diff(profile) > 0).all(), "bucket means must rise strictly with the prediction"
    assert (res.spread > 0).all() and s["hit_rate"] == 1.0
    assert s["max_drawdown"] == 0.0 and s["drawdown_days"] == 0.0 and s["monotonicity"] == pytest.approx(1.0)
    assert s["spread_ir"] == pytest.approx(s["spread_mean"] / s["spread_std"] * np.sqrt(252 / 30))
    print("\n=== SANITY CHECK: perfect prediction ===")
    print(f"  profile {np.round(profile, 3)}; spread {s['spread_mean']:.3f} every day, hit 100%, max DD 0, monotonicity +1. Validated.")


def test_inverted_prediction_loses_every_day() -> None:
    frame = _frame()
    s = _bt().run(frame.assign(pred=-frame["y"]), label_col="y").summary
    assert s["spread_mean"] < 0 and s["hit_rate"] == 0.0 and s["monotonicity"] == pytest.approx(-1.0)
    assert s["max_drawdown"] == pytest.approx(s["cum_spread"]), "a curve that only falls draws down by all of it"
    print("\n=== SANITY CHECK: inverted prediction ===")
    print(f"  spread {s['spread_mean']:+.3f}, hit 0%, monotonicity -1, drawdown == cumulative loss {s['max_drawdown']:+.2f}. Validated.")


def test_turnover_zero_for_a_fixed_ranking_and_one_for_a_rotating_one() -> None:
    frame = _frame()
    idx = frame["ticker"].str[1:].astype(int)
    fixed = _bt().run(frame.assign(pred=idx.astype(float)), label_col="y").summary
    day = frame.groupby("date").ngroup()
    rotating = _bt().run(frame.assign(pred=np.where(day % 2 == 0, idx, -idx).astype(float)), label_col="y").summary
    assert fixed["turnover_top"] == 0.0 and fixed["turnover_bottom"] == 0.0
    assert rotating["turnover_top"] == 1.0 and rotating["turnover_bottom"] == 1.0
    print("\n=== SANITY CHECK: extreme-bucket turnover ===")
    print(
        f"  fixed ranking -> top/bottom turnover {fixed['turnover_top']:.0%}/{fixed['turnover_bottom']:.0%}; alternating ranking -> {rotating['turnover_top']:.0%}/{rotating['turnover_bottom']:.0%}. Validated."
    )


def test_thin_days_and_missing_rows_are_skipped() -> None:
    frame = _frame(n_days=5)
    thin = pd.DataFrame({"date": pd.Timestamp("2023-01-02"), "ticker": ["A", "B", "C"], "y": [0.1, 0.5, 0.9]})
    frame = pd.concat([frame, thin], ignore_index=True)
    frame["pred"] = frame["y"]
    frame.loc[0, "pred"] = np.nan
    res = _bt().run(frame, label_col="y")
    assert res.summary["n_days"] == 5.0 and pd.Timestamp("2023-01-02") not in res.buckets.index
    empty = _bt(5).run(thin.assign(pred=thin["y"]), label_col="y")
    assert empty.summary["n_days"] == 0.0 and empty.spread.empty
    with pytest.raises(ValueError):
        _bt(1)
    print("\n=== SANITY CHECK: thin days ===")
    print(
        "  a 3-name day is skipped with 10 buckets; a NaN prediction row is dropped; no usable day -> empty result; n_quantiles < 2 refused. Validated."
    )


def test_saved_artifacts(tmp_path: Path) -> None:
    frame = _frame()
    res = _bt(5).run(frame.assign(pred=frame["y"] + np.random.default_rng(1).normal(0, 0.3, len(frame))), label_col="y", horizon=60)
    Backtest.save(res, tmp_path, 60)
    buckets = pd.read_csv(tmp_path / "backtest_buckets.csv", index_col=0)
    spread = pd.read_csv(tmp_path / "backtest_spread.csv", index_col=0)
    assert (tmp_path / "backtest.png").exists() and list(buckets.columns) == ["1", "2", "3", "4", "5"]
    assert np.allclose(spread["cum_spread"].to_numpy(), res.spread.cumsum().to_numpy()) and (spread["drawdown"] <= 0).all()
    print("\n=== SANITY CHECK: backtest artifacts ===")
    print(
        f"  buckets.csv {buckets.shape}, spread.csv with cum_spread + drawdown, backtest.png; noisy signal spread IR {res.summary['spread_ir']:+.2f}. Validated."
    )
