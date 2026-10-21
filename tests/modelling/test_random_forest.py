"""`RandomForestModel` (src/modelling/transformers/random_forest.py): the column fallback chain,
a real save -> load roundtrip inside a 3-member ensemble, and the fixed tree count."""

from __future__ import annotations

import io
import pickle
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

from src.modelling.transformers import LinearRegression, RandomForestModel, model_class
from src.modelling.utils.artifacts import load_ensemble, member_path, save_member
from src.modelling.utils.ensemble import ensemble_predict
from src.modelling.utils.features import design_matrix
from tests.modelling.model_fixtures import ctx, make_config, signal_panel


@pytest.mark.parametrize("family", ["lgbm", "random_forest"])
def test_numeric_text_prediction_and_explanation_parity(family: str, tmp_path: Path) -> None:
    panel = signal_panel(n_days=40, n_tickers=20)
    panel.loc[panel.index[::13], "f0"] = np.nan
    member = model_class(family)(ctx(), make_config(), family, 60).fit(panel)
    text_panel = panel.copy()
    text_panel["f0"] = panel["f0"].astype("str")
    assert np.array_equal(member.predict(panel), member.predict(text_panel))
    assert np.array_equal(
        design_matrix(panel, member.features, member.categoricals),
        design_matrix(text_panel, member.features, member.categoricals),
        equal_nan=True,
    )
    loaded = type(member).load(save_member(member, tmp_path / "m.pkl"))
    assert np.array_equal(loaded.predict(text_panel), member.predict(panel))
    bad = text_panel.copy()
    bad.loc[bad.index[0], "f0"] = "unparseable"
    with pytest.raises(ValueError, match="f0"):
        loaded.predict(bad)
    assert text_panel["f0"].isna().iloc[0]
    print(f"SANITY: {family} numeric/text predictions, SHAP inputs and reloaded predictions agree; missingness preserved.")


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
    assert RandomForestModel.configured_categoricals(
        OmegaConf.create({"lgbm": {"columns": ["sector"], "categoricals": ["sector"]}, "random_forest": {}})
    ) == ["sector"]
    print("\n=== SANITY CHECK: RF column chain ===")
    print("  rf.columns_by_horizon -> rf.columns -> lgbm.columns_by_horizon -> lgbm.columns; categoricals from lgbm. Validated.")


def test_explicit_empty_family_horizon_lists_do_not_fall_back() -> None:
    cfg = make_config(random_forest={"columns_by_horizon": {60: []}})
    assert RandomForestModel.configured_columns(cfg, 60) == []
    assert RandomForestModel.configured_categoricals(cfg, 60) == []
    assert RandomForestModel.configured_categoricals(cfg, 30) == ["sector"]
    print("SANITY: explicit empty h60 pools stay empty; missing overrides use the default feature list and shared category registry.")


def test_random_forest_member_saves_and_reloads_into_the_ensemble(tmp_path: Path) -> None:
    cfg = make_config()
    panel = signal_panel()
    dates = np.sort(panel["date"].unique())
    train, test = panel[panel["date"] < dates[120]], panel[panel["date"] >= dates[125]]
    sub_tr, sub_val = train, test
    members = {fam: model_class(fam)(ctx(), cfg, fam, 60).fit(sub_tr, sub_val) for fam in cfg.model.ensemble}
    rf = members["random_forest"]
    assert isinstance(rf, RandomForestModel) and rf.model.num_trees() == 40, "fixed tree count, no early stopping"
    assert isinstance(members["elasticnet"], LinearRegression) and "sector" not in members["elasticnet"].features

    for fam, m in members.items():
        save_member(m, member_path(tmp_path, 60, fam))
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


def test_member_pickles_hold_no_library_objects() -> None:
    """A member pickle names only its own class and plain Python values (the booster as its LightGBM
    text model, the linear arrays as floats), so it loads under another lightgbm / numpy version."""
    cfg = make_config()
    panel = signal_panel()
    dates = np.sort(panel["date"].unique())
    train, test = panel[panel["date"] < dates[120]], panel[panel["date"] >= dates[125]]
    sub_tr, sub_val = train, test
    referenced = {}
    for fam in cfg.model.ensemble:
        member = model_class(fam)(ctx(), cfg, fam, 60).fit(sub_tr, sub_val)
        blob = pickle.dumps(member, protocol=pickle.HIGHEST_PROTOCOL)
        unpickler = _ModuleRecorder(io.BytesIO(blob))
        reloaded = unpickler.load()
        referenced[fam] = sorted(unpickler.modules)
        assert unpickler.modules == {type(member).__module__}, f"{fam}: pickle resolves {unpickler.modules}"
        assert type(reloaded) is type(member) and reloaded.features == member.features
        assert np.array_equal(reloaded.predict(test).to_numpy(), member.predict(test).to_numpy())
    print("\n=== SANITY CHECK: portable member pickles ===")
    print(f"  modules each pickle resolves: {referenced} (no numpy / lightgbm / pandas); predictions identical after reload. Validated.")


class _ModuleRecorder(pickle.Unpickler):
    """Unpickler that records every module a pickle asks to import."""

    def __init__(self, file: io.BytesIO) -> None:
        super().__init__(file)
        self.modules: set[str] = set()

    def find_class(self, module: str, name: str) -> type:
        self.modules.add(module)
        return super().find_class(module, name)
