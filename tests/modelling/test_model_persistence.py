"""Round-trip test: one pickle file per horizon."""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest

from src.data_aggregate.utils.assemble.cube import panel_from_cube
from src.modelling.long_short.utils import model as ml


_PANEL_ROWS = 8_000
_WINDOW_DAYS = 400          # ~one trading year back from the latest labelled date


@pytest.fixture(scope="module")
def sample_panel():
    """One horizon's modelling panel, read scoped to a RECENT labelled window.

    Was `store.load("cube")` -- the whole 5.6M x 574 table, ~7GB -- to keep 8000 rows. Both
    narrowings `panel_from_cube` performs (one horizon, non-null label) are pushed into SQL.

    The window is anchored on the LATEST labelled date, not the earliest: the oldest rows have
    no `mom_12_1` yet (252-day warm-up), so an earliest-first slice returns an all-NULL column
    that arrives as dtype `object` and LightGBM refuses it. The date predicate also lets the
    read use the date index instead of sorting the whole table.
    """
    from src.context import get_config_context

    config, ctx = get_config_context("./configs", use_cache=False, save=False)
    label = config.model.label_column
    h = int(config.build_cube.targets.primary_horizon)
    cube_cols = ctx.store.columns("cube")
    if not cube_cols:
        pytest.skip("cube table does not exist")

    # cfg.lgbm.columns, not the long-gone cfg.inputs.columns: this trains a LightGBM ranker
    feats = [c for c in config.lgbm.columns if c in cube_cols][:12]
    target_col = "target_rank" if "target_rank" in cube_cols else "target"
    scope = {"target_horizon": h, target_col: ctx.store.NOT_NULL}

    latest = ctx.store.distinct("cube", "date", where=scope, order="desc", limit=1)
    if not latest:
        pytest.skip("no labelled cube rows for the primary horizon")
    since = pd.Timestamp(latest[0]) - pd.Timedelta(days=_WINDOW_DAYS)

    cube = ctx.store.load("cube",
                          columns=["date", "ticker", "target_horizon", target_col] + feats,
                          where=scope, since=since, optional=True)
    if cube is None:
        pytest.skip("no labelled cube rows in the recent window")

    panel = panel_from_cube(cube, h, label, feats).tail(_PANEL_ROWS)
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
