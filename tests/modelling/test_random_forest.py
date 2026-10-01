"""`RandomForestModel` (src/modelling/transformers/random_forest.py): the column fallback chain,
a real save -> load roundtrip inside a 3-member ensemble, and the fixed tree count."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from src.modelling.transformers import LinearRegression, RandomForestModel, model_class
from src.modelling.utils.artifacts import load_ensemble, member_path
from src.modelling.utils.cv import temporal_valid_split
from src.modelling.utils.ensemble import ensemble_predict
from tests.modelling.model_fixtures import ctx, make_config, signal_panel


def test_column_chain_rf_override_then_rf_then_lgbm() -> None:
    cfg = OmegaConf.create(
        {
            "lgbm": {"columns": ["l1", "l2"], "columns_by_horizon": {90: ["l90"]}},
            "random_forest": {"columns": ["r1"], "columns_by_horizon": {30: ["r30"]}},
        }
    )
    assert RandomForestModel.configured_columns(cfg, 30) == ["r30"]
    assert RandomForestModel.configured_columns(cfg, 60) == ["r1"]
    cfg_no_rf = OmegaConf.create({"lgbm": cfg.lgbm, "random_forest": {"columns_by_horizon": {30: ["r30"]}}})
    assert RandomForestModel.configured_columns(cfg_no_rf, 60) == ["l1", "l2"]
    assert RandomForestModel.configured_columns(cfg_no_rf, 90) == ["l90"]
    assert RandomForestModel.configured_categoricals(OmegaConf.create({"lgbm": {"categoricals": ["sector"]}, "random_forest": {}})) == ["sector"]
    print("\n=== SANITY CHECK: RF column chain ===")
    print("  rf.columns_by_horizon -> rf.columns -> lgbm.columns_by_horizon -> lgbm.columns; categoricals from lgbm. Validated.")


def test_random_forest_member_saves_and_reloads_into_the_ensemble(tmp_path: Path) -> None:
    cfg = make_config()
    panel = signal_panel()
    dates = np.sort(panel["date"].unique())
    train, test = panel[panel["date"] < dates[120]], panel[panel["date"] >= dates[125]]
    sub_tr, sub_val = temporal_valid_split(train)
    members = {fam: model_class(fam)(ctx(), cfg, fam, 60).fit(sub_tr, sub_val) for fam in cfg.model.ensemble}
    rf = members["random_forest"]
    assert isinstance(rf, RandomForestModel) and rf.model.num_trees() == 40, "fixed tree count, no early stopping"
    assert isinstance(members["elasticnet"], LinearRegression) and "sector" not in members["elasticnet"].features

    for fam, m in members.items():
        m.save(member_path(tmp_path, 60, fam))
    reloaded = load_ensemble(tmp_path, [60], list(cfg.model.ensemble))[60]
    assert list(reloaded) == list(cfg.model.ensemble)
    before, _ = ensemble_predict(members, test)
    after, after_members = ensemble_predict(reloaded, test)
    assert np.array_equal(before.to_numpy(), after.to_numpy(), equal_nan=True)
    assert np.isfinite(after.to_numpy()).all() and after.std() > 0
    print("\n=== SANITY CHECK: RF member persistence ===")
    print(f"  RF = {rf.model.num_trees()} bagged trees on {rf.features}; 3 members pickled + reloaded in order {list(reloaded)};")
    print(
        f"  ensemble prediction identical after reload; per-member std { ({k: round(float(v.std()), 3) for k, v in after_members.items()}) }. Validated."
    )
