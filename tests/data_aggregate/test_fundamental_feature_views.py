"""Phase-8 contract for the deliberately compact fundamentals feature schema."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from scripts.cube_feature_catalogue import CATALOGUE, split
from src.data_aggregate.utils.common.panel import build_peer_relative_panel
from src.data_aggregate.utils.fundamentals.feature_views import (
    EXCLUDED_FEATURES,
    FEATURE_COLUMNS,
    HISTORY_FEATURES,
    RAW_FEATURES,
    build_fundamental_views,
)

EXPECTED_HISTORY_FEATURES = frozenset(
    {
        "accruals",
        "accruals_ratio",
        "acquisition_intensity",
        "affo_dividend_coverage",
        "affo_margin",
        "aoci_to_equity",
        "asset_turnover",
        "bank_roa",
        "buyback_intensity",
        "capex_intensity",
        "capex_to_dep",
        "capex_to_rate_base",
        "cash_conversion_cycle",
        "cash_to_debt",
        "da_to_capex",
        "ddna_intensity",
        "debtToEquity",
        "deferred_rev_intensity",
        "dio",
        "dividend_coverage",
        "dividend_payout_ratio",
        "dividend_yield",
        "dpo",
        "dpo_change",
        "dso",
        "dso_change",
        "earningsGrowth",
        "earnings_quality",
        "ebit_interest_coverage",
        "ebitda_margin",
        "ebitda_to_ev",
        "effective_tax_rate",
        "fcf_growth",
        "fcf_margin",
        "fcf_to_ev",
        "fcf_yield",
        "ffo_margin",
        "ffo_payout",
        "ffo_yield",
        "fixed_cost_coverage_margin",
        "forward_earnings_yield",
        "fwd_eps_yield",
        "gmroi",
        "grossMargins",
        "gross_profitability",
        "intangibles_roic_drag",
        "intangibles_to_assets",
        "intangibles_to_equity",
        "interest_coverage",
        "intrinsic_yield",
        "loss_intensity",
        "nci_income_share",
        "net_debt_to_ebitda",
        "nwc_elasticity",
        "operatingMargins",
        "operating_leverage_elasticity",
        "option_overhang",
        "payout_ratio",
        "pegy",
        "pension_funded_ratio",
        "pension_overhang_leverage",
        "pension_retirement_liability",
        "profitMargins",
        "rd_capitalized_roic",
        "rd_intensity",
        "refinancing_risk",
        "reinvestment_rate",
        "returnOnEquity",
        "revenue_per_employee",
        "roic_ex_intangibles",
        "roic_incl_intangibles",
        "sales_yield",
        "sbc_intensity",
        "sbc_to_buyback",
        "sbc_to_ocf",
        "sga_elasticity",
        "sga_intensity",
        "shareholder_yield",
        "y_margin_vs_ttm",
    }
)


def test_approved_feature_view_contract_is_exactly_200_columns() -> None:
    assert len(RAW_FEATURES) == 121
    assert HISTORY_FEATURES == EXPECTED_HISTORY_FEATURES
    assert len(HISTORY_FEATURES) == 79
    assert HISTORY_FEATURES < RAW_FEATURES
    assert EXCLUDED_FEATURES == frozenset({"q_rev_growth", "revenueGrowth"})
    assert set(CATALOGUE) == set(RAW_FEATURES)
    assert len(FEATURE_COLUMNS) == 200
    assert len(set(FEATURE_COLUMNS)) == len(FEATURE_COLUMNS)
    assert not any(column.endswith(("_vs_peers", "_xs")) for column in FEATURE_COLUMNS)

    print("\n=== SANITY CHECK: approved fundamentals feature views ===")
    print("  121 raw + 79 self-history = 200 unique columns; peer and XS legs are absent.")
    print("  q_rev_growth and the redundant revenueGrowth alias are explicitly excluded.")


def test_fundamental_view_builder_emits_raw_and_selected_history_only() -> None:
    dates = pd.bdate_range("2024-01-01", periods=6)
    fields = {
        "dividend_yield": pd.DataFrame({"AAA": np.arange(1.0, 7.0)}, index=dates),
        "asset_growth": pd.DataFrame({"AAA": np.arange(2.0, 8.0)}, index=dates),
        "q_rev_growth": pd.DataFrame({"AAA": 0.1}, index=dates),
        "revenueGrowth": pd.DataFrame({"AAA": 0.2}, index=dates),
    }

    panel = build_fundamental_views(
        fields,
        {},
        history_window=3,
        history_min_periods=2,
    )

    expected = {
        "date",
        "ticker",
        "f_dividend_yield",
        "f_dividend_yield_vs_hist",
        "f_asset_growth",
    }
    assert set(panel) == expected
    assert np.isfinite(panel["f_dividend_yield_vs_hist"]).any()
    assert all(panel[column].dtype == np.float32 for column in expected - {"date", "ticker"})

    with pytest.raises(KeyError, match="not in the approved fundamentals feature contract"):
        build_fundamental_views({"unreviewed_feature": fields["asset_growth"]}, {})

    print("\n=== SANITY CHECK: fundamentals view emission ===")
    print("  raw is universal, history is allow-listed, killed aliases stay internal, and drift fails loudly.")


def test_seeded_incremental_raw_history_matches_full_tail() -> None:
    dates = pd.bdate_range("2020-01-01", periods=40)
    current_dates = dates[-10:]
    output_since = dates[-5]
    full_fields = {
        "dividend_yield": pd.DataFrame(
            {
                "AAA": np.linspace(0.01, 0.04, len(dates)),
                "BBB": np.linspace(0.04, 0.01, len(dates)),
            },
            index=dates,
        ),
        "asset_growth": pd.DataFrame(
            {
                "AAA": np.linspace(-0.2, 0.3, len(dates)),
                "BBB": np.nan,
            },
            index=dates,
        ),
    }
    emission = {"dividend_yield": "raw+hist", "asset_growth": "raw"}

    full = build_peer_relative_panel(
        full_fields,
        {},
        emission=emission,
        history_window=20,
        history_min_periods=4,
    )
    incremental = build_peer_relative_panel(
        {name: frame.loc[current_dates] for name, frame in full_fields.items()},
        {},
        emission=emission,
        history_window=20,
        history_min_periods=4,
        history_fields={name: frame.loc[dates[:-10]] for name, frame in full_fields.items()},
        output_since=output_since,
    )

    expected = full.loc[full["date"] >= output_since].reset_index(drop=True)
    pd.testing.assert_frame_equal(incremental, expected, check_exact=True)

    print("\n=== SANITY CHECK: seeded incremental fundamentals views ===")
    print("  stored raw history plus the recomputed overlap reproduces the exact full-build tail.")


def _strings(value: object) -> set[str]:
    if isinstance(value, dict):
        return {str(key) for key in value} | {item for child in value.values() for item in _strings(child)}
    if isinstance(value, list):
        return {item for child in value for item in _strings(child)}
    return {str(value)} if isinstance(value, str) else set()


def _assert_unique(label: str, values: list[str]) -> None:
    duplicates = sorted({value for value in values if values.count(value) > 1})
    assert not duplicates, f"{label} has duplicate columns after migration: {duplicates}"


def test_active_model_configs_use_only_surviving_fundamental_views() -> None:
    approved = set(FEATURE_COLUMNS)
    known_characteristics = set(RAW_FEATURES) | set(EXCLUDED_FEATURES)
    paths = [
        Path("configs/models/lgbm_modelling.yml"),
        Path("configs/models/linear_modelling.yml"),
        Path("configs/models/random_forest_modelling.yml"),
    ]

    for path in paths:
        values = _strings(OmegaConf.to_container(OmegaConf.load(path), resolve=True))
        stale = sorted(
            column for column in values if column.startswith("f_") and split(column)[0] in known_characteristics and column not in approved
        )
        assert not stale, f"{path} still selects removed fundamentals views: {stale}"

    lgbm = OmegaConf.to_container(OmegaConf.load(paths[0]), resolve=True)["lgbm"]
    linear = OmegaConf.to_container(OmegaConf.load(paths[1]), resolve=True)["linear"]
    forest = OmegaConf.to_container(OmegaConf.load(paths[2]), resolve=True)["random_forest"]
    _assert_unique("lgbm.columns", lgbm["columns"])
    _assert_unique("linear.columns", linear["columns"])
    _assert_unique("random_forest.columns", forest["columns"])
    for horizon, columns in forest["columns_by_horizon"].items():
        _assert_unique(f"random_forest.columns_by_horizon.{horizon}", columns)

    print("\n=== SANITY CHECK: model-config compatibility ===")
    print("  three active configs contain no removed fundamentals views and no duplicate selections.")
