"""`Monitor` (src/modelling/transformers/monitor.py): the diagnostics layout and KPI keys the DoD
report reads, IC drawdown, SHAP / dependence / PDP artifacts, the linear-only and inactive cases,
the run-level files, and the aggregated importance."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.modelling.transformers import model_class
from src.modelling.transformers.backtest import Backtest
from src.modelling.transformers.monitor import (
    _UNSAFE_FILENAME_CHARS,
    Monitor,
    partial_dependence,
    save_shap_values,
    shap_importance_from_values,
    shap_row_values,
)
from src.modelling.utils.cv import temporal_valid_split
from src.modelling.utils.ensemble import ensemble_predict
from src.modelling.utils.features import design_matrix
from src.modelling.utils.metrics import daily_ic_series, max_drawdown
from tests.modelling.model_fixtures import ctx, make_config, signal_panel

DIAG = {"model": {"diagnostics": {"enabled": True, "top_n_features": 3, "shap_sample": 300, "pdp_grid": 8}}}


def _fitted(families: list[str], h: int = 30) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    cfg = make_config(model={**DIAG["model"], "ensemble": families})
    panel = signal_panel()
    dates = np.sort(panel["date"].unique())
    train, test = panel[panel["date"] < dates[110]], panel[panel["date"] >= dates[120]]
    sub_tr, sub_val = temporal_valid_split(train)
    members = {fam: model_class(fam)(ctx(), cfg, fam, h).fit(sub_tr, sub_val) for fam in families}
    blended, _ = ensemble_predict(members, test)
    oos = pd.DataFrame({"date": test["date"].to_numpy(), "ticker": test["ticker"].to_numpy(), "pred": blended.to_numpy(), "y": test["y"].to_numpy()})
    return members, panel, oos


def _monitor(tmp_path: Path, save: bool = True) -> Monitor:
    context = SimpleNamespace(save=save, paths={"OUTPUT_DIR": tmp_path})
    monitor = Monitor(context, make_config(**DIAG))  # type: ignore[arg-type]
    monitor.start_run()
    return monitor


KPIS = {
    "cv_mean_ic": 0.05,
    "cv_ic_ir": 1.2,
    "members": {"elasticnet": {"cv_mean_ic": 0.04, "cv_ic_ir": 1.0}, "lgbm": {"cv_mean_ic": 0.05, "cv_ic_ir": 1.1}},
}


def test_three_member_layout_and_dod_contract(tmp_path: Path) -> None:
    members, panel, oos = _fitted(["elasticnet", "lgbm", "random_forest"])
    monitor = _monitor(tmp_path)
    summary = monitor.horizon_report(30, panel, oos, members, KPIS, "y")
    assert summary is not None
    hdir = monitor.run_dir / "h30"
    assert (hdir / "kpis.json").exists() and (hdir / "ic_over_time.png").exists() and (hdir / "ic_over_time.csv").exists()
    assert (hdir / "coef_importance_elasticnet.csv").exists()
    for booster in ("lgbm", "random_forest"):  # two boosters -> one sub-folder each
        mdir = hdir / booster
        assert len(list((mdir / "pdp").glob("pdp_*.png"))) == 3
        for name in ("shap_values.parquet", "shap_importance.png", "shap_importance.csv", "shap_dependence.png"):
            assert (mdir / name).exists(), f"{booster}/{name}"
        assert (mdir / "feature_importance.xlsx").exists() or (mdir / "feature_importance.csv").exists()
        assert summary["members"][booster]["shap_available"] is True and summary["members"][booster]["n_pdp"] == 3
    # DoD contract: the linear member carries CV KPIs only -> recognised as "not a booster"
    assert set(summary["members"]["elasticnet"]) == {"cv_mean_ic", "cv_ic_ir"}
    on_disk = json.loads((hdir / "kpis.json").read_text())
    assert on_disk["cv_mean_ic"] == 0.05 and "n_pdp" in on_disk and on_disk["oos_ic_days"] == summary["oos_ic_days"] > 0
    ic = daily_ic_series(oos, "y")
    depth, length = max_drawdown(ic.cumsum())
    assert on_disk["oos_ic_max_drawdown"] == pytest.approx(depth) and on_disk["oos_ic_drawdown_days"] == length
    print("\n=== SANITY CHECK: Monitor layout (3 members, 2 boosters) ===")
    print("  h30/{lgbm,random_forest}/ with 3 PDPs, SHAP parquet/plots, dependence grid; coef table for elasticnet;")
    print(
        f"  OOS IC {on_disk['oos_ic_mean']:+.3f} over {on_disk['oos_ic_days']} days, max cumulative-IC drawdown {depth:+.3f} over {length} days. Validated."
    )


def test_member_and_feature_file_names_are_filesystem_safe() -> None:
    names = {"beta_USD/EUR 1": "beta_USD_EUR_1", "a b\\c:d": "a_b_c_d", "random_forest": "random_forest", "x..y-z": "x..y-z"}
    got = {name: _UNSAFE_FILENAME_CHARS.sub("_", name) for name in names}
    assert got == names, got
    print("\n=== SANITY CHECK: diagnostics file names ===")
    print(f"  {got}: every run outside [0-9A-Za-z._-] collapses to one underscore. Validated.")


def test_single_booster_is_flat(tmp_path: Path) -> None:
    members, panel, oos = _fitted(["elasticnet", "lgbm"])
    monitor = _monitor(tmp_path)
    summary = monitor.horizon_report(30, panel, oos, members, KPIS, "y")
    assert summary is not None
    hdir = monitor.run_dir / "h30"
    assert (hdir / "shap_values.parquet").exists() and (hdir / "pdp").is_dir() and not (hdir / "lgbm").exists()
    assert summary["n_pdp"] == summary["members"]["lgbm"]["n_pdp"] and summary["shap_available"] is True
    print("\n=== SANITY CHECK: single booster -> flat layout ===")
    print("  h30/ holds the booster artifacts directly (the DoD reader's one-booster rule) and the flat back-compat keys. Validated.")


def test_linear_only_ensemble_warns_and_still_reports(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    members, panel, oos = _fitted(["elasticnet", "ridge"])
    monitor = _monitor(tmp_path)
    with caplog.at_level(logging.WARNING, logger="src.modelling.transformers.monitor"):
        summary = monitor.horizon_report(30, panel, oos, members, {"cv_mean_ic": 0.01, "members": {}}, "y")
    assert summary is not None and summary["members"] == {}
    assert not {"n_pdp", "shap_available", "importance_path"} & set(summary), "no flat booster keys without a booster"
    assert (monitor.run_dir / "h30" / "kpis.json").exists() and (monitor.run_dir / "h30" / "coef_importance_ridge.csv").exists()
    assert any("no tree member" in r.getMessage() for r in caplog.records)
    print("\n=== SANITY CHECK: linear-only ensemble ===")
    print("  WARNs 'no tree member', still writes kpis.json + IC curve + coef tables, and no booster keys. Validated.")


def test_inactive_monitor_writes_nothing(tmp_path: Path) -> None:
    members, panel, oos = _fitted(["lgbm"])
    monitor = _monitor(tmp_path, save=False)
    assert monitor.horizon_report(30, panel, oos, members, KPIS, "y") is None
    res = Backtest(make_config()).run(oos, label_col="y", horizon=30)
    monitor.backtest_report(30, res)
    assert monitor.run_report({30: 1.0}, ["lgbm"], "rank") is None
    assert not any(tmp_path.iterdir())
    print("\n=== SANITY CHECK: inactive monitor ===")
    print("  context.save=False -> metrics logged, no file written. Validated.")


def test_run_report_and_backtest_columns(tmp_path: Path) -> None:
    members, panel, oos = _fitted(["elasticnet", "lgbm"])
    monitor = _monitor(tmp_path)
    monitor.horizon_report(30, panel, oos, members, KPIS, "y")
    monitor.backtest_report(30, Backtest(make_config()).run(oos, label_col="y", horizon=30))
    path = monitor.run_report({30: 0.7}, ["elasticnet", "lgbm"], "rank")
    assert path is not None
    flat = pd.read_csv(monitor.run_dir / "kpis.csv")
    assert list(flat["member"]) == ["ENSEMBLE", "elasticnet", "lgbm"] and (flat["blend_weight"] == 0.7).all()
    assert flat["bt_spread_mean"].notna().all() and (monitor.run_dir / "h30" / "backtest.png").exists()
    run = json.loads(path.read_text())
    assert run["horizons"]["30"]["backtest"]["n_quantiles"] == 10.0
    print("\n=== SANITY CHECK: run-level KPIs ===")
    print(flat[["member", "cv_mean_ic", "oos_ic_mean", "blend_weight", "bt_spread_mean", "bt_hit_rate"]].to_string(index=False))
    print("  ENSEMBLE + one row per member, blend weight and label-only backtest columns. Validated.")


def test_shap_values_are_signed_and_pdp_and_importance(tmp_path: Path) -> None:
    members, panel, _ = _fitted(["lgbm", "elasticnet"])
    lgbm = members["lgbm"]
    x = design_matrix(panel, lgbm.features)
    got = shap_row_values(lgbm.model, x, lgbm.features, sample=200)
    if got is None:
        pytest.skip("shap not installed in this environment")
    values, idx = got
    assert (values < 0).any() and (values > 0).any()
    imp = shap_importance_from_values(values, lgbm.features)
    assert np.allclose(imp.reindex(lgbm.features).to_numpy(), np.abs(values).mean(axis=0))
    save_shap_values(values, idx, lgbm.features, panel, tmp_path / "s.parquet")
    saved = pd.read_parquet(tmp_path / "s.parquet")
    assert list(saved.columns[:2]) == ["date", "ticker"] and saved["ticker"].isin(panel["ticker"]).all()
    grid, means = partial_dependence(lgbm.model, x, lgbm.features.index("f0"), grid_points=10, sample=300)
    assert grid is not None and len(grid) == len(means) >= 2 and means[-1] > means[0], "f0 is constrained increasing"
    agg = _monitor(tmp_path).feature_importance({30: members})
    assert agg is not None and abs(agg.sum() - 1.0) < 1e-12 and agg.index[0] in {"f0", "f1", "f2"}
    print("\n=== SANITY CHECK: SHAP / PDP / importance ===")
    print(f"  SHAP signed, mean|SHAP| == ranking, rows keyed by date+ticker; PDP of f0 rises {means[0]:.3f} -> {means[-1]:.3f};")
    print(f"  aggregated importance sums to 1, top feature {agg.index[0]}. Validated.")
