"""Horizon-specific feature selection for the modelling step.

The RF (and any member) reads `columns_by_horizon[<horizon>]` when the config declares one
for the horizon being trained, else falls back to the member's default `columns`. LightGBM has
no per-horizon override, so it keeps its own default set for every horizon.
"""

from __future__ import annotations

import numpy as np
from omegaconf import OmegaConf

from src.modelling.long_short.step_train import StepModelling


def _step(cfg):
    """A StepModelling with only `_config` set (bypasses __init__ -> no Context / DB)."""
    s = StepModelling.__new__(StepModelling)
    s._config = cfg
    return s


def test_columns_by_horizon_resolution_and_fallback():
    cfg = OmegaConf.create(
        {
            "linear": {"columns": ["a", "b"]},
            "lgbm": {"columns": ["l1", "l2", "l3"]},  # default only, no override
            "random_forest": {
                "columns": ["d1", "d2", "d3"],  # DEFAULT (h60-style)
                "columns_by_horizon": {30: ["s1", "s2"]},  # h30 OVERRIDE
            },
        }
    )
    step = _step(cfg)

    # static resolver: override for 30, None (fall back) for others / when absent
    assert StepModelling._by_horizon(cfg.random_forest, 30) == ["s1", "s2"]
    assert StepModelling._by_horizon(cfg.random_forest, 60) is None
    assert StepModelling._by_horizon(cfg.random_forest, None) is None
    assert StepModelling._by_horizon(cfg.lgbm, 30) is None  # no columns_by_horizon

    # RF: horizon-specific for 30 (incl. numpy-int key), DEFAULT for 60 / 90 / None
    assert step._rf_columns(30) == ["s1", "s2"]
    assert step._rf_columns(np.int64(30)) == ["s1", "s2"]
    assert step._rf_columns(60) == ["d1", "d2", "d3"]
    assert step._rf_columns(90) == ["d1", "d2", "d3"]
    assert step._rf_columns(None) == ["d1", "d2", "d3"]
    assert step._rf_columns() == ["d1", "d2", "d3"]

    # lgbm has no override -> its own default set for EVERY horizon
    for h in (30, 60, 90, None):
        assert step._lgbm_columns(h) == ["l1", "l2", "l3"]

    # cube-projection superset = defaults + every override + linear
    u = set(step._union_all_columns())
    assert {"a", "b", "l1", "l2", "l3", "d1", "d2", "d3", "s1", "s2"} <= u

    # a string-typed YAML key ("30") still matches an int horizon (30)
    cfg_str = OmegaConf.create({"random_forest": {"columns": ["d1"], "columns_by_horizon": {"30": ["s1", "s2"]}}})
    assert _step(cfg_str)._rf_columns(30) == ["s1", "s2"]

    print(
        "\nSANITY: RF h30 -> ['s1','s2'] (override); h60/h90/None -> default "
        "['d1','d2','d3']; lgbm -> its default for all horizons; union superset covers "
        "default+override; str YAML key matches int horizon. Horizon-with-fallback works."
    )


def test_real_config_resolves_h30_vs_default():
    """Against the ACTUAL configs: RF resolves its h30 override at h=30 and its default at
    h=60/90; lgbm keeps its own default at every horizon.

    ⚠ ASSERTS NO DUPLICATES rather than a hardcoded list length, because the hardcoded 50 is
    what hid the defect. `columns_by_horizon: 30` was written as a "50-feature set" and
    listed `f_ic_inst_net_options_ratio` twice, so it was 49 distinct features in a 50-long
    list -- and the length assertion passed on the repeat while the default's honest 49
    failed. A duplicate feeds the same column to the model twice; the count it inflates is
    the only thing that ever noticed.
    """
    from src.utils.config import read_config

    step = _step(read_config(path="./configs"))
    rf30, rf60, rf90 = step._rf_columns(30), step._rf_columns(60), step._rf_columns(90)
    lg30, lg60 = step._lgbm_columns(30), step._lgbm_columns(60)
    for label, cols in (("RF h30", rf30), ("RF default", rf60), ("lgbm default", lg60)):
        dupes = sorted({c for c in cols if list(cols).count(c) > 1})
        assert not dupes, f"{label} lists {dupes} more than once -- a feature fed twice"
    assert len(rf30) == len(rf60), (
        f"the h30 override ({len(rf30)}) and the default ({len(rf60)}) are different sizes; "
        f"they are two selections of the same budget, so a mismatch is an editing slip"
    )
    assert rf30 != rf60  # genuinely different horizon set
    assert rf90 == rf60  # 90 has no override -> default
    assert lg30 == lg60 and len(lg60) >= 60  # lgbm default for all horizons
    print(
        f"\nSANITY(real cfg): RF h30={len(rf30)} feats (override) != RF default h60={len(rf60)}; "
        f"RF h90={len(rf90)} == default; lgbm={len(lg60)} default and equal across h30/h60; "
        f"RF overlap h30-and-h60={len(set(rf30) & set(rf60))}/{len(rf60)}; no duplicates in "
        f"any list. Config + fallback wired correctly."
    )
