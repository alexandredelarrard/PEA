"""The per-run diagnostics actually get WRITTEN when StepModelling trains.

The regression this pins: `_horizon_diagnostics` resolved the tree member by NAME
(`.get("lightgbm")`), so the moment `model.ensemble` became `[elasticnet, lgbm,
random_forest]` the lookup returned None and the whole block returned early -- no PDP,
no SHAP, no KPIs, and no error either. Member resolution is now by TYPE
(`isinstance(m, lgb.Booster)`), so a rename or an added tree member cannot switch the
diagnostics off again.

Covered here:
  1. `_booster_members` finds the boosters under the CURRENT config names (`lgbm`,
     `random_forest`) and the legacy `lightgbm`, and ignores the linear members;
  2. a run writes, per horizon: kpis.json, ic_over_time.*, and per booster member the
     PDPs + shap_values.parquet (the RAW per-row matrix) + shap_importance.* + the gain
     table;
  3. the raw SHAP matrix is keyed by (date, ticker) with one column per model feature;
  4. the run-level kpis.csv carries one row per (horizon, member) with the CV IC / IC_IR
     and the blend weight.
"""
from __future__ import annotations

import json
import types
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.modelling.long_short.step_train import StepModelling
from src.modelling.long_short.utils import diagnostics
from src.modelling.long_short.utils.model import predict, purged_wf_splits, train_ranker


def _panel(n_days: int = 90, n_tickers: int = 40, n_feats: int = 5, seed: int = 0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-01", periods=n_days)
    feats = [f"f_feat{j}_xs" for j in range(n_feats)]
    frames = []
    for d in dates:
        X = rng.normal(size=(n_tickers, n_feats))
        sig = X[:, 0] * 0.7 + X[:, 1] * 0.3 - X[:, 2] * 0.2 + rng.normal(scale=0.5, size=n_tickers)
        block = pd.DataFrame(X, columns=feats)
        block.insert(0, "y", sig.argsort().argsort() / (n_tickers - 1))
        block.insert(0, "ticker", [f"T{i:03d}" for i in range(n_tickers)])
        block.insert(0, "date", d)
        frames.append(block)
    return pd.concat(frames, ignore_index=True), feats


def _oos(panel, feats):
    frames = []
    for tr_days, te_days in purged_wf_splits(panel["date"], n_splits=3, embargo=5):
        tr, te = panel[panel["date"].isin(tr_days)], panel[panel["date"].isin(te_days)]
        if tr.empty or te.empty:
            continue
        b = train_ranker(tr, feats, "y", num_boost_round=25)
        frames.append(pd.DataFrame({"date": te["date"].to_numpy(),
                                    "ticker": te["ticker"].to_numpy(),
                                    "pred": predict(b, te, feats).to_numpy(),
                                    "y": te["y"].to_numpy()}))
    return pd.concat(frames, ignore_index=True)


def _step(tmp_path: Path, ensemble: list[str]) -> StepModelling:
    """A StepModelling with just enough state for the diagnostics hook (no DB, no training)."""
    step = StepModelling.__new__(StepModelling)          # bypass __init__ (needs a DB context)
    step._context = types.SimpleNamespace(
        save=True, paths={"OUTPUT_DIR": tmp_path}, log=None)
    step._log = types.SimpleNamespace(
        info=lambda *a, **k: None, warning=lambda *a, **k: warnings.append(a))
    step._config = {"model": {"diagnostics": {"enabled": True, "top_n_features": 3,
                                              "shap_sample": 300, "pdp_grid": 8}},
                    "train": {"start_date": "2018-01-01", "end_date": "2019-01-01"}}
    # OmegaConf-like attribute access over the plain dicts above
    step._config = _Attr(step._config)
    step.model_types = list(ensemble)
    step.target_type = "rank"
    step.label_column = "y"
    step._full_history = False
    step._diag_summaries = {}
    step._run_stamp = "RUNSTAMP"
    return step


class _Attr(dict):
    """Minimal attribute-access dict so the step can read `config.model.diagnostics`."""
    def __getattr__(self, k):
        v = self[k]
        return _Attr(v) if isinstance(v, dict) else v

    def get(self, k, default=None):
        v = super().get(k, default)
        return _Attr(v) if isinstance(v, dict) else v


warnings: list = []


def test_booster_members_resolved_by_type_not_name():
    """THE regression: a member named `lgbm` (or `lightgbm`, or `random_forest`) is found;
    linear members are ignored."""
    panel, feats = _panel(n_days=30, n_tickers=20)
    booster = train_ranker(panel, feats, "y", num_boost_round=10)
    linear = object()                                   # stands in for the elasticnet member

    step = StepModelling.__new__(StepModelling)
    for names in (["elasticnet", "lgbm", "random_forest"],      # current config
                  ["elasticnet", "lightgbm"],                   # legacy name
                  ["elasticnet"]):                              # linear-only -> nothing
        step.models = {30: {n: (linear if n in ("elasticnet", "ridge") else booster)
                            for n in names}}
        got = set(step._booster_members(30))
        assert got == {n for n in names if n not in ("elasticnet", "ridge")}, (names, got)
        assert "elasticnet" not in got

    print("\n=== SANITY CHECK: booster member resolution ===")
    print("  ensemble [elasticnet, lgbm, random_forest] -> boosters {lgbm, random_forest}")
    print("  ensemble [elasticnet, lightgbm]            -> boosters {lightgbm}   (legacy name)")
    print("  ensemble [elasticnet]                      -> boosters {}          (warns, no crash)")
    print("  Resolution is by isinstance(lgb.Booster), so renaming a member in "
          "model.ensemble can no longer silently disable diagnostics. Validated.")


def test_run_writes_shap_values_and_kpis_per_horizon(tmp_path):
    """A training run writes, per horizon: raw SHAP values, SHAP/gain importance, PDPs, the
    IC curve and kpis.json -- for EVERY booster member -- plus the run-level kpis.csv."""
    panel, feats = _panel()
    lgbm = train_ranker(panel, feats, "y", num_boost_round=30)
    rf = train_ranker(panel, feats, "y", num_boost_round=30,
                      params={"objective": "regression", "boosting": "rf",
                              "bagging_fraction": 0.7, "bagging_freq": 1,
                              "feature_fraction": 0.7, "verbosity": -1})
    oos = _oos(panel, feats)

    step = _step(tmp_path, ["elasticnet", "lgbm", "random_forest"])
    step.models = {30: {"elasticnet": object(), "lgbm": lgbm, "random_forest": rf}}
    step.horizon_ic = {30: {"mean_ic": 0.031, "ic_ir": 1.42}}
    step.member_ic = {30: {"lgbm": {"mean_ic": 0.028, "ic_ir": 1.20},
                           "random_forest": {"mean_ic": 0.026, "ic_ir": 1.05},
                           "elasticnet": {"mean_ic": 0.019, "ic_ir": 0.90}}}
    step.oos_predictions = {30: oos}
    step._lgb_feats = lambda h=None: feats
    step.horizon_weights = {30: 1.0}

    step._horizon_diagnostics(30, panel, "RUNSTAMP")
    step._save_run_diagnostics_kpis()

    hdir = tmp_path / "diagnostics" / "RUNSTAMP" / "h30"
    assert hdir.is_dir(), f"no horizon folder written: {sorted(tmp_path.rglob('*'))}"

    # ensemble-level artifacts, once per horizon
    assert (hdir / "ic_over_time.png").exists() and (hdir / "ic_over_time.csv").exists()
    assert (hdir / "kpis.json").exists()

    # per booster member: PDPs + SHAP values + SHAP importance + gain table
    per_member = {}
    for member in ("lgbm", "random_forest"):
        mdir = hdir / member
        assert mdir.is_dir(), f"no member folder for {member}"
        pdps = sorted((mdir / "pdp").glob("pdp_*.png"))
        assert pdps, f"{member}: no PDP written"
        assert (mdir / "shap_values.parquet").exists(), f"{member}: raw SHAP values missing"
        assert (mdir / "shap_importance.csv").exists() and (mdir / "shap_importance.png").exists()
        assert ((mdir / "feature_importance.xlsx").exists()
                or (mdir / "feature_importance.csv").exists())
        sv = pd.read_parquet(mdir / "shap_values.parquet")
        # the RAW matrix: keyed by (date, ticker), one column per model feature
        assert list(sv.columns[:2]) == ["date", "ticker"], sv.columns[:4].tolist()
        assert set(feats).issubset(sv.columns)
        assert len(sv) == min(300, len(panel))
        assert np.isfinite(sv[feats].to_numpy()).all()
        per_member[member] = (len(pdps), len(sv))
    assert not (hdir / "elasticnet").exists(), "linear member must not get a booster folder"

    # per-horizon KPIs carry the CV numbers the writer cannot compute itself
    hk = json.loads((hdir / "kpis.json").read_text())
    assert hk["cv_mean_ic"] == 0.031 and hk["cv_ic_ir"] == 1.42
    assert hk["members"]["lgbm"]["cv_ic_ir"] == 1.20
    assert hk["n_rows"] == len(panel) and hk["n_days"] == panel["date"].nunique()
    assert hk["oos_ic_days"] > 0

    # run-level flat table: one row per (horizon, member) + the ENSEMBLE row
    kcsv = pd.read_csv(tmp_path / "diagnostics" / "RUNSTAMP" / "kpis.csv")
    assert set(kcsv["member"]) == {"ENSEMBLE", "lgbm", "random_forest", "elasticnet"}
    ens = kcsv[kcsv["member"] == "ENSEMBLE"].iloc[0]
    assert ens["cv_ic_ir"] == 1.42 and ens["blend_weight"] == 1.0

    print("\n=== SANITY CHECK: per-run diagnostics written per horizon ===")
    print(f"  {hdir.relative_to(tmp_path)}/  kpis.json, ic_over_time.png+csv "
          f"({hk['oos_ic_days']} OOS IC days, mean {hk['oos_ic_mean']:+.4f})")
    for member, (n_pdp, n_shap) in per_member.items():
        print(f"    {member}/  {n_pdp} PDP PNGs, shap_values.parquet "
              f"({n_shap} rows x {len(feats)} features), shap_importance.png+csv, gain table")
    print("    elasticnet/ -> absent (linear member, no SHAP/PDP)")
    print(f"  run kpis.csv: {len(kcsv)} rows "
          f"(ENSEMBLE IC_IR {ens['cv_ic_ir']:+.2f}, blend weight {ens['blend_weight']:.2f})")
    print("  CONCLUSION: training now saves the raw SHAP values + PDPs per booster member and "
          "the key KPIs per horizon, under one run-stamp folder. Validated.")


def test_shap_values_match_importance_and_are_signed(tmp_path):
    """The persisted matrix IS the SHAP values: signed per-row contributions whose mean|.|
    reproduces shap_importance.csv exactly (one computation, two artifacts)."""
    panel, feats = _panel(n_days=40, n_tickers=25, seed=3)
    booster = train_ranker(panel, feats, "y", num_boost_round=25)
    x = panel[feats].to_numpy("float32")

    got = diagnostics.shap_row_values(booster, x, feats, sample=200)
    if got is None:
        import pytest
        pytest.skip("shap not installed in this environment")
    values, idx = got
    imp = diagnostics.shap_importance_from_values(values, feats)

    diagnostics.save_shap_values(values, idx, feats, panel, tmp_path / "shap_values.parquet")
    sv = pd.read_parquet(tmp_path / "shap_values.parquet")

    recomputed = sv[feats].abs().mean().sort_values(ascending=False)
    pd.testing.assert_series_equal(recomputed, imp.astype("float32"),
                                  check_names=False, rtol=1e-5)
    assert (values < 0).any() and (values > 0).any(), "SHAP values must be SIGNED"
    # rows are keyed back to the panel they came from
    assert (sv["ticker"].to_numpy() == panel.iloc[idx]["ticker"].to_numpy()).all()

    print("\n=== SANITY CHECK: raw SHAP values <-> importance ===")
    print(f"  {len(sv)} rows x {len(feats)} features; signed (min {values.min():+.4f}, "
          f"max {values.max():+.4f})")
    print(f"  mean|SHAP| recomputed from the saved matrix == shap_importance ranking "
          f"(top: {list(imp.head(3).index)})")
    print("  rows join back to (date, ticker), so a single name/day attribution is "
          "recoverable after the run. Validated.")


def test_design_matrix_matches_training_encoding_for_categoricals():
    """SHAP/PDP must see the encoding the model was FITTED on.

    The diagnostics matrix used a bare `panel[feats].to_numpy("float32")`, which raises on a
    categorical that arrives as TEXT (`sector='Energy'`) even though training coerces it to a
    numeric code. Both sides now go through `model.coerce_categoricals`, so the encoding is
    identical and a text categorical can never crash the diagnostics."""
    from src.modelling.long_short.utils import model as ml

    panel, feats = _panel(n_days=12, n_tickers=15, seed=7)
    panel["sector"] = np.where(panel["ticker"] < "T007", "Energy", "Utilities")  # TEXT
    panel["industry_group"] = np.where(panel["ticker"] < "T004", 3, 8)           # already numeric
    cats = ["sector", "industry_group"]
    all_feats = feats + cats

    x = ml.design_matrix(panel, all_feats, categorical_features=cats)
    assert x.shape == (len(panel), len(all_feats)) and x.dtype == np.dtype("float32")
    sector_col = x[:, all_feats.index("sector")]
    # unparseable text -> the SAME missing code training uses, not an exception
    assert set(np.unique(sector_col)) == {float(ml.CATEGORICAL_NA_CODE)}
    assert set(np.unique(x[:, all_feats.index("industry_group")])) == {3.0, 8.0}

    coerced = ml.coerce_categoricals(panel[all_feats], cats)
    assert (coerced["industry_group"].to_numpy() == panel["industry_group"].to_numpy()).all()

    # and the whole pipeline runs on it end to end (train -> SHAP) without raising
    booster = train_ranker(panel, all_feats, "y", num_boost_round=10,
                           categorical_features=cats)
    got = diagnostics.shap_row_values(booster, x, all_feats, sample=50)
    n_shap = "shap absent" if got is None else f"{got[0].shape[0]}x{got[0].shape[1]}"

    print("\n=== SANITY CHECK: diagnostics encoding == training encoding ===")
    print(f"  sector (TEXT 'Energy'/'Utilities') -> code {ml.CATEGORICAL_NA_CODE} "
          "(training's missing code), no exception where the old to_numpy('float32') raised")
    print("  industry_group (already numeric 3/8) -> passed through unchanged: {3.0, 8.0}")
    print(f"  train(categoricals) -> SHAP on the same matrix: {n_shap}")
    print("  ONE coercion rule (model.coerce_categoricals) serves the LightGBM Dataset and the "
          "SHAP/PDP matrix. Validated.")


if __name__ == "__main__":
    import sys

    import pytest
    sys.exit(pytest.main([__file__, "-v", "-s"]))
