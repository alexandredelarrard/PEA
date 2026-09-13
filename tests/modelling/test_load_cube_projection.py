"""
`StepModelling` pulls ONLY the columns the model needs and ONLY rows where that horizon's
label is available. Against the WIDE cube that splits in two:

  * `_select_load_columns`  -> the horizon-INDEPENDENT projection (index + the modelling.yml
                               allow-list & categoricals that exist in the cube). It carries
                               NO target column: one row now serves every horizon.
  * `_load_cols_for(tcol)`  -> that projection plus the ONE label column a horizon trains on.
  * `_distinct_horizons`    -> which horizons exist, read from the SCHEMA (`target_<type>_h*`)
                               rather than `SELECT DISTINCT target_horizon`.

These pin the pure column/horizon resolution; the SQL projection and the labelled-row filter
are exercised end-to-end by the smoke run.
"""
from __future__ import annotations

import logging

import pytest
from omegaconf import OmegaConf

from src.data_aggregate.utils.assemble.cube import target_column
from src.modelling.long_short.step_train import StepModelling


# a realistic wide cube schema: 2 labels x 3 horizons, plus features and the meta columns
_CUBE_COLS = {
    "date", "ticker", "peers", "beta_m",
    "target_rank_h30", "target_rank_h60", "target_rank_h90",
    "target_zscore_h30", "target_zscore_h60", "target_zscore_h90",
    "mom_12_1", "f_ebitda_to_ev_xs", "sector",
}


def _fake(target_type="rank", columns=None, cats=None):
    """Minimal stand-in exposing just what the pure helpers read."""
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


def test_distinct_horizons_reads_the_schema_for_this_label_only():
    """Horizons come from the COLUMN NAMES of the configured label family. A sibling label's
    columns must not contribute a horizon the model then fails to find a column for."""
    assert StepModelling._distinct_horizons(_fake("rank"), _CUBE_COLS) == [30, 60, 90]
    assert StepModelling._distinct_horizons(_fake("zscore"), _CUBE_COLS) == [30, 60, 90]
    # a label the cube does not carry -> no horizons, which `_setup` turns into a clear raise
    assert StepModelling._distinct_horizons(_fake("epsilon"), _CUBE_COLS) == []
    # and it is per-label, not "any target column"
    partial = {"date", "ticker", "target_rank_h30", "target_zscore_h60", "target_zscore_h90"}
    assert StepModelling._distinct_horizons(_fake("rank"), partial) == [30]
    assert StepModelling._distinct_horizons(_fake("zscore"), partial) == [60, 90]

    print("\n=== SANITY CHECK: horizon discovery from the schema ===")
    print(f"  rank -> {StepModelling._distinct_horizons(_fake('rank'), _CUBE_COLS)}, "
          f"epsilon (not built) -> [] ; on a mixed schema rank->[30] while zscore->[60, 90]. "
          f"Schema only, no data scan. Validated.")


def test_select_load_columns_projects_present_and_reports_absent():
    f = _fake("rank",
              columns=["mom_12_1", "f_ebitda_to_ev_xs", "f_pegy_xs", "f_asset_growth_xs"],
              cats=["sector", "industry_group"])
    load_cols, dropped = StepModelling._select_load_columns(f, _CUBE_COLS)

    # index first, then only the present features/categoricals — and NO target column
    assert load_cols == ["date", "ticker", "mom_12_1", "f_ebitda_to_ev_xs", "sector"]
    assert set(dropped) == {"f_pegy_xs", "f_asset_growth_xs", "industry_group"}

    # a configured name that collides with a meta column is not double-listed
    f2 = _fake("rank", columns=["date", "mom_12_1"], cats=[])
    lc2, _ = StepModelling._select_load_columns(f2, _CUBE_COLS)
    assert lc2.count("date") == 1 and lc2 == ["date", "ticker", "mom_12_1"]

    # no inputs configured -> just the index
    f3 = _fake("rank", columns=[], cats=[])
    lc3, drop3 = StepModelling._select_load_columns(f3, _CUBE_COLS)
    assert lc3 == ["date", "ticker"] and drop3 == []

    print("\n=== SANITY CHECK: load-column projection ===")
    print(f"  {load_cols} — index + present allow-list/categoricals only; "
          f"absent {sorted(dropped)} reported, not queried; meta never duplicated")
    print("  the projection is horizon-INDEPENDENT and carries no label column. Validated.")


def test_a_target_column_can_never_be_loaded_as_a_feature():
    """THE wide-cube trap. Every horizon's label now sits in the cube as an ordinary column,
    so a configured allow-list naming one would feed the model a label as a feature — and
    `target_rank_h60` leaking into an h30 model is a near-perfect leak. `is_meta_column`
    matches the `target_<label>_h<horizon>` PATTERN, so this holds for horizons and labels
    that did not exist when the filter was written."""
    f = _fake("rank",
              columns=["mom_12_1", "target_rank_h30", "target_rank_h60", "target_zscore_h90"],
              cats=["peers"])
    load_cols, dropped = StepModelling._select_load_columns(f, _CUBE_COLS)

    assert load_cols == ["date", "ticker", "mom_12_1"]
    assert not [c for c in load_cols if c.startswith("target")]
    assert "peers" not in load_cols                       # meta, not a feature
    assert dropped == [], "these exist in the cube — they are excluded, not missing"

    print("\n=== SANITY CHECK: no target column can enter as a feature ===")
    print("  allow-list naming target_rank_h30 / _h60 / target_zscore_h90 + peers -> "
          f"projection is {load_cols}: all four excluded by PATTERN, so a horizon or label "
          "added to the config later cannot leak either. Validated.")


def test_load_cols_for_appends_exactly_one_label_column():
    f = _fake("rank", columns=["mom_12_1", "f_ebitda_to_ev_xs"], cats=["sector"])
    f._load_cols, _ = StepModelling._select_load_columns(f, _CUBE_COLS)

    per_h = {h: StepModelling._load_cols_for(f, target_column("rank", h)) for h in (30, 60, 90)}

    for h, cols in per_h.items():
        assert cols == f._load_cols + [f"target_rank_h{h}"]
        assert len([c for c in cols if c.startswith("target")]) == 1, \
            f"h{h} would load more than its own label: {cols}"
    # the shared projection is not mutated by appending to it
    assert f._load_cols == ["date", "ticker", "mom_12_1", "f_ebitda_to_ev_xs", "sector"]
    # asking twice for the same horizon does not duplicate the label
    assert StepModelling._load_cols_for(f, "target_rank_h30") == per_h[30]

    print("\n=== SANITY CHECK: one label column per horizon ===")
    print(f"  shared projection {f._load_cols}")
    for h, cols in per_h.items():
        print(f"    h{h} -> +{cols[-1]} ({len(cols)} columns, exactly 1 target)")
    print("  each horizon reads its own label and no other; the shared list is untouched. "
          "Validated.")


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))
