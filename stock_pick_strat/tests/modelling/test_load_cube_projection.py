"""
StepModelling.load_cube now pulls ONLY the columns the model needs (index + target
+ the modelling.yml allow-list & categoricals that exist in the cube) and ONLY rows
where the target is available. These tests pin the pure column/target resolution
(the SQL projection + labelled-row filter are exercised end-to-end by the smoke run).
"""
from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from src.modelling.long_short.step_train import StepModelling


def _fake(target_type="rank", columns=None, cats=None):
    """Minimal stand-in exposing just what the two pure helpers read."""
    # A real but UNINITIALISED StepModelling, not a SimpleNamespace: `_select_load_columns`
    # resolves its feature set through a chain of per-member config helpers
    # (`_union_all_columns` -> `_lgbm_categoricals` -> ...) that grows with the config
    # schema, and a namespace stub has to be re-patched every time one is added — which is
    # exactly how this test broke. Stubbing only the config keeps every real resolver live.
    ns = object.__new__(StepModelling)
    ns.target_type = target_type
    ns._log = logging.getLogger("load_cube_projection_test")
    ns._config = OmegaConf.create({"inputs": {"columns": columns or [],
                                              "categoricals": cats or []}})
    return ns


def test_target_column_resolution_and_fallback():
    # configured target present
    assert StepModelling._target_column(_fake("rank"),
                                        {"target_rank", "target_zscore", "date"}) == "target_rank"
    assert StepModelling._target_column(_fake("zscore"),
                                        {"target_zscore", "date"}) == "target_zscore"
    # legacy single-target cube -> fall back to `target`
    assert StepModelling._target_column(_fake("rank"), {"target", "date"}) == "target"
    # neither present -> explicit error (don't silently load nothing)
    with pytest.raises(KeyError):
        StepModelling._target_column(_fake("rank"), {"date", "ticker"})
    print("\n=== SANITY CHECK: target-column resolution ===")
    print("  target_<type> when present; legacy 'target' fallback; KeyError when absent. Validated.")


def test_select_load_columns_projects_present_and_reports_absent():
    cube_cols = {"date", "ticker", "target_horizon", "target_rank",
                 "mom_12_1", "f_ebitda_to_ev_xs", "sector"}
    f = _fake("rank",
              columns=["mom_12_1", "f_ebitda_to_ev_xs", "f_pegy_xs", "f_asset_growth_xs"],
              cats=["sector", "industry_group"])
    load_cols, dropped = StepModelling._select_load_columns(f, cube_cols, "target_rank")

    # index+target first, then only the present features/categoricals
    assert load_cols == ["date", "ticker", "target_horizon", "target_rank",
                         "mom_12_1", "f_ebitda_to_ev_xs", "sector"]
    assert set(dropped) == {"f_pegy_xs", "f_asset_growth_xs", "industry_group"}

    # a configured name that collides with a meta column is not double-listed
    f2 = _fake("rank", columns=["date", "mom_12_1"], cats=[])
    lc2, _ = StepModelling._select_load_columns(f2, cube_cols, "target_rank")
    assert lc2.count("date") == 1 and lc2 == ["date", "ticker", "target_horizon",
                                              "target_rank", "mom_12_1"]

    # no inputs configured -> just the index + target
    f3 = _fake("rank", columns=[], cats=[])
    lc3, drop3 = StepModelling._select_load_columns(f3, cube_cols, "target_rank")
    assert lc3 == ["date", "ticker", "target_horizon", "target_rank"] and drop3 == []
    print("\n=== SANITY CHECK: load-column projection ===")
    print("  loads index+target + present allow-list/categoricals; absent ones reported, "
          "not queried; meta never duplicated. Validated.")


if __name__ == "__main__":
    test_target_column_resolution_and_fallback()
    test_select_load_columns_projects_present_and_reports_absent()
