"""The former split panels are one unified raw/history earnings-call contract."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from src.constants.constants import EARNINGS_CALL_FEATURES
from src.data_aggregate.utils.text import earnings_call_features as ec


def test_embedding_kpis_are_part_of_the_unified_contract() -> None:
    retained_embedding_kpis = {"ec_qa_coherence_mean", "ec_qa_qq_distance", "ec_prep_qq_distance"}
    assert retained_embedding_kpis <= set(EARNINGS_CALL_FEATURES)
    assert len(EARNINGS_CALL_FEATURES) == 10
    calls = pd.DataFrame({"ticker": ["A"], "quarter": ["2024Q1"], "as_of": ["2024-05-01"], "ec_prep_tone": [0.1]})
    panel = ec.build_earnings_call_feature_panel(calls, pd.bdate_range("2024-05-01", "2024-08-31"), availability_lag_days=2)
    assert set(panel.columns) == {"date", "ticker", *[f"f_{name}" for name in EARNINGS_CALL_FEATURES]}
    assert not any(name.endswith(("_xs", "_vs_peers")) for name in EARNINGS_CALL_FEATURES)
    print("\n=== SANITY CHECK: unified earnings-call contract ===")
    print("  embedding KPIs are part of the exact ten fields; no separate peer/xs panel remains. Validated.")


def test_ten_fields_agree_with_models_bounds_and_fresh_schema() -> None:
    expected = {f"f_{name}" for name in EARNINGS_CALL_FEATURES}
    for file, family in (("lgbm_modelling.yml", "lgbm"), ("random_forest_modelling.yml", "random_forest")):
        config = OmegaConf.load(Path("configs/models") / file)
        assert isinstance(config, DictConfig)
        settings = config[family]
        actual = [name for name in settings.columns if name.startswith("f_ec_")]
        assert len(actual) == 10 and set(actual) == expected
    bounds = OmegaConf.load("configs/validate.yml").validate.tables.cube_part_text.bounds
    assert set(bounds) == expected
    for name, limits in bounds.items():
        assert list(limits) == ([0.0, 2.0] if name.endswith("distance") else [-2.0, 2.0] if name.endswith("delta") else [-1.0, 1.0])
    ddl = Path("sql/schema.sql").read_text(encoding="utf-8")
    assert set(re.findall(r'"(f_ec_[a-z0-9_]+)"', ddl)) == expected
    assert "uncertainty_ratio" not in ddl
    print("SANITY: both tree-model inputs, all ten validator bounds and fresh SQL columns agree; retired feature/dictionary fields are absent.")
