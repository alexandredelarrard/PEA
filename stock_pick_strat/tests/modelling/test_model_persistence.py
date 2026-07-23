"""Round-trip test: one pickle file per horizon."""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest

from src.data_aggregate.utils.cube import panel_from_cube
from src.modelling.long_short.utils import model as ml


@pytest.fixture(scope="module")
def sample_panel():
    from src.context import get_config_context

    config, ctx = get_config_context("./configs", use_cache=False, save=False)
    cube = ctx.store.load("cube")
    if cube.empty:
        pytest.skip("cube table is empty")

    label = config.model.label_column
    h = config.build_cube.targets.primary_horizon
    feats = [c for c in config.inputs.columns if c in cube.columns][:12]
    panel = panel_from_cube(cube, h, label, feats).head(8000)
    if panel.empty:
        pytest.skip("empty modelling panel")
    return panel, feats, label, h


def test_one_pickle_per_horizon_roundtrip(sample_panel, tmp_path):
    panel, feats, label, h = sample_panel
    sub_tr, sub_va = ml.temporal_valid_split(panel)
    booster = ml.train_ranker(sub_tr, feats, label, valid_panel=sub_va, num_boost_round=30)
    models = {h: booster}

    meta = {
        "label_column": label,
        "feature_cols": feats,
        "primary_horizon": h,
        "horizon_weights": {str(h): 1.0},
        "horizon_ic": {str(h): {"mean_ic": 0.01, "ic_ir": 0.5}},
        "panel_date_min": str(panel["date"].min().date()),
        "panel_date_max": str(panel["date"].max().date()),
    }
    ml.save_models(tmp_path, models, meta)

    pkl = ml.model_pickle_path(tmp_path, h)
    assert pkl.exists()

    with pkl.open("rb") as f:
        bundle = pickle.load(f)
    assert bundle["horizon"] == h
    assert bundle["feature_cols"] == feats
    assert "model" in bundle

    loaded_all, loaded_meta = ml.load_models(tmp_path)
    assert set(loaded_all.keys()) == {h}

    holdout = panel.tail(500)
    orig = ml.predict(booster, holdout, feats).to_numpy()
    reloaded = ml.predict(loaded_all[h], holdout, feats).to_numpy()
    assert np.allclose(orig, reloaded, rtol=1e-5, atol=1e-5)
    assert loaded_meta["feature_cols"] == feats

    print("\n=== SANITY CHECK: one pickle per horizon ===")
    print(f"  {pkl.name}: horizon={h}, {len(feats)} features, "
          f"max |pred diff|={np.abs(orig - reloaded).max():.2e}. Validated.")
