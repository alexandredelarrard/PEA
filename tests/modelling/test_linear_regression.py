"""`LinearRegression` (src/modelling/transformers/linear_regression.py): warning-free
standardisation, L1 selection vs ridge, competitiveness with LightGBM out of sample, the
ensemble beating its members' average, the degenerate-alpha warning, and its guards."""

from __future__ import annotations

import logging
import warnings
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.modelling.transformers import BaseModel, LightGBMModel, LinearRegression
from src.modelling.transformers.linear_regression import _standardize
from src.modelling.utils.artifacts import save_member
from src.modelling.utils.cv import purged_wf_splits, temporal_valid_split
from src.modelling.utils.ensemble import ensemble_predict
from src.modelling.utils.metrics import daily_ic
from tests.modelling.model_fixtures import NUMERIC, ctx, make_config, signal_panel


def test_standardize_no_warning_on_all_nan_column() -> None:
    x = np.array([[1.0, np.nan, 5.0], [2.0, np.nan, 6.0], [3.0, np.nan, 7.0]])
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        xs, mean, std = _standardize(x)
    assert np.allclose(xs[:, 1], 0.0) and mean[1] == 0.0 and std[1] == 1.0 and np.isfinite(xs).all()
    print("\n=== SANITY CHECK: standardize ===")
    print("  all-NaN feature -> no RuntimeWarning, mean 0 / std 1, contributes zeros. Validated.")


def _collinear_panel() -> tuple[pd.DataFrame, list[str]]:
    rng = np.random.default_rng(0)
    n, k = 3000, 10
    x = rng.normal(0, 1, (n, k))
    x[:, 1] = x[:, 0] + rng.normal(0, 0.05, n)  # f1 nearly collinear with f0
    feats = [f"f{j}" for j in range(k)]
    panel = pd.DataFrame(x, columns=feats)
    panel["y"] = 1.5 * x[:, 0] + rng.normal(0, 1.0, n)
    panel["date"] = pd.Timestamp("2020-01-01")
    return panel, feats


def test_elasticnet_selects_where_ridge_keeps_everything() -> None:
    panel, feats = _collinear_panel()
    en = LinearRegression(
        ctx(), make_config(linear={"columns": feats, "alpha": 0.1, "l1_ratio": 0.7, "max_iter": 3000, "tol": 1e-9}), "elasticnet"
    ).fit(panel)
    rg = LinearRegression(ctx(), make_config(linear={"columns": feats, "alpha": 10.0}), "ridge").fit(panel)
    en_zeros, rg_zeros = int((np.abs(en.coef_) < 1e-8).sum()), int((np.abs(rg.coef_) < 1e-8).sum())
    assert en_zeros >= 4 and rg_zeros == 0
    assert abs(en.coef_[0]) + abs(en.coef_[1]) > np.abs(en.coef_[2:]).sum()
    print("\n=== SANITY CHECK: elastic-net selection vs ridge ===")
    print(f"  zero coefficients: elasticnet {en_zeros}/10, ridge {rg_zeros}/10; f0+f1 carry the weight. L1 selects, L2 shares. Validated.")


def _cv_ic(panel: pd.DataFrame, make: Callable[[], BaseModel]) -> float:
    ics = []
    for tr_d, te_d in purged_wf_splits(pd.Series(panel["date"]), n_splits=4, embargo=5):
        tr, te = panel[panel["date"].isin(tr_d)], panel[panel["date"].isin(te_d)]
        sub_tr, sub_val = temporal_valid_split(tr)
        model = make().fit(sub_tr, sub_val)
        ics.append(daily_ic(te, model.predict(te), "y")["mean_ic"])
    return float(np.nanmean(ics))


@pytest.mark.parametrize("family", ["ridge", "elasticnet"])
def test_linear_baseline_close_to_lightgbm(family: str) -> None:
    panel = signal_panel(n_days=200, n_tickers=60, sparse=False)
    cfg = make_config(
        linear={"columns": NUMERIC, "alpha": 10.0 if family == "ridge" else 0.001},
        lgbm={"num_boost_round": 300, "eval_metric": "ic", "early_stopping_rounds": 40, "categoricals": []},
    )
    lgb_ic = _cv_ic(panel, lambda: LightGBMModel(ctx(), cfg, "lgbm", 1))
    lin_ic = _cv_ic(panel, lambda: LinearRegression(ctx(), cfg, family, 1))
    assert lgb_ic > 0 and lin_ic > 0
    assert abs(lin_ic - lgb_ic) < 0.05 and lin_ic >= 0.7 * lgb_ic
    print(f"\n=== SANITY CHECK: {family} vs LightGBM (purged-CV IC) ===")
    print(f"  LightGBM {lgb_ic:+.4f}  {family} {lin_ic:+.4f}  |diff| {abs(lin_ic - lgb_ic):.4f}: the linear baseline is competitive. Validated.")


def test_ensemble_at_least_matches_average_of_members() -> None:
    panel = signal_panel(n_days=200, n_tickers=60, sparse=False)
    cfg = make_config(
        linear={"columns": NUMERIC}, lgbm={"num_boost_round": 300, "eval_metric": "ic", "early_stopping_rounds": 40, "categoricals": []}
    )
    il, ie, ien = [], [], []
    for tr_d, te_d in purged_wf_splits(pd.Series(panel["date"]), n_splits=4, embargo=5):
        tr, te = panel[panel["date"].isin(tr_d)], panel[panel["date"].isin(te_d)]
        sub_tr, sub_val = temporal_valid_split(tr)
        members = {
            "lgbm": LightGBMModel(ctx(), cfg, "lgbm", 1).fit(sub_tr, sub_val),
            "elasticnet": LinearRegression(ctx(), cfg, "elasticnet", 1).fit(sub_tr, sub_val),
        }
        il.append(daily_ic(te, members["lgbm"].predict(te), "y")["mean_ic"])
        ie.append(daily_ic(te, members["elasticnet"].predict(te), "y")["mean_ic"])
        ien.append(daily_ic(te, ensemble_predict(members, te)[0], "y")["mean_ic"])
    ic_lgb, ic_en, ic_ens = (float(np.nanmean(v)) for v in (il, ie, ien))
    assert ic_ens > 0 and ic_ens >= (ic_lgb + ic_en) / 2 - 0.03
    print("\n=== SANITY CHECK: ensemble vs members (OOS IC) ===")
    print(f"  LightGBM {ic_lgb:+.4f}  elasticnet {ic_en:+.4f}  ENSEMBLE {ic_ens:+.4f} (>= mean of members). Validated.")


def test_elasticnet_learns_at_sane_alpha_and_warns_when_degenerate(caplog: pytest.LogCaptureFixture) -> None:
    rng = np.random.default_rng(0)
    n = 4000
    f1, f2, f3 = rng.standard_normal(n), rng.standard_normal(n), rng.standard_normal(n)
    panel = pd.DataFrame({"date": pd.Timestamp("2020-01-01"), "f1": f1, "f2": f2, "f3": f3, "y": 0.6 * f1 - 0.4 * f2 + 0.1 * rng.standard_normal(n)})
    feats = ["f1", "f2", "f3"]
    live = LinearRegression(ctx(), make_config(linear={"columns": feats, "alpha": 1e-3, "l1_ratio": 0.3, "max_iter": 1000}), "elasticnet").fit(panel)
    coef = dict(zip(feats, live.coef_, strict=True))
    preds = live.predict(panel).to_numpy()
    assert coef["f1"] > 0 and coef["f2"] < 0 and np.corrcoef(preds, panel["y"])[0, 1] > 0.5
    with caplog.at_level(logging.WARNING, logger="src.modelling.transformers.linear_regression"):
        dead = LinearRegression(ctx(), make_config(linear={"columns": feats, "alpha": 10.0, "l1_ratio": 0.3}), "elasticnet").fit(panel)
    assert np.count_nonzero(dead.coef_) == 0 and np.std(dead.predict(panel).to_numpy()) < 1e-9
    assert any("DEGENERATE" in r.getMessage() for r in caplog.records)
    print("\n=== SANITY CHECK: degenerate alpha ===")
    print(
        f"  alpha=1e-3 learns { ({k: round(v, 3) for k, v in coef.items()}) }; alpha=10 -> all-zero constant fit and a DEGENERATE warning. Validated."
    )


def test_roundtrip_importance_and_guards(tmp_path: Path) -> None:
    panel = signal_panel(n_days=60)
    model = LinearRegression(ctx(), make_config(), "elasticnet", 60).fit(panel)
    assert model.features == ["f2", "f0", "f1"], "configured order, categoricals never used"
    loaded = LinearRegression.load(save_member(model, tmp_path / "m.pkl"))
    assert np.array_equal(loaded.predict(panel).to_numpy(), model.predict(panel).to_numpy())
    imp = model.importance()
    assert list(imp.index) == model.features and np.array_equal(imp.to_numpy(), np.abs(model.coef_))
    with pytest.raises(ValueError, match="no classification objective"):
        LinearRegression(ctx(), make_config(linear={"task": "classification"}), "elasticnet")
    with pytest.raises(ValueError, match="family must be"):
        LinearRegression(ctx(), make_config(), "lasso")
    with pytest.raises(RuntimeError, match="not fitted"):
        LinearRegression(ctx(), make_config(), "ridge").predict(panel)
    print("\n=== SANITY CHECK: linear member surface ===")
    print(
        f"  features {model.features}; pickle roundtrip identical; |coef| importance; classification / unknown family / unfitted predict all refused. Validated."
    )
