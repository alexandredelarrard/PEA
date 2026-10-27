"""Phase-8 contract for the deliberately compact fundamentals feature schema."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from scripts.cube_feature_catalogue import CATALOGUE, split
from src.context import define_global_paths
from src.data_aggregate.utils.common.frames import ratio
from src.data_aggregate.utils.common.panel import build_peer_relative_panel
from src.data_aggregate.utils.common.peers_io import load_peers
from src.data_aggregate.utils.common.pit import daily_market_cap, fundamentals_to_daily
from src.data_aggregate.utils.fundamentals.feature_views import (
    EXCLUDED_FEATURES,
    FEATURE_COLUMNS,
    HISTORY_FEATURES,
    PEER_FEATURES,
    RAW_FEATURES,
    build_fundamental_views,
)
from src.data_store.schema import Tables
from src.utils.config import read_config
from tests.conftest import _store

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
        "y_earnings_growth_signed",
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
        "reported_expected_ttm_yield",
        "reported_expected_eps_yield",
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


def test_approved_feature_view_contract_is_exactly_218_columns() -> None:
    assert len(RAW_FEATURES) == 137
    assert HISTORY_FEATURES == EXPECTED_HISTORY_FEATURES
    assert len(HISTORY_FEATURES) == 79
    assert HISTORY_FEATURES < RAW_FEATURES
    assert EXCLUDED_FEATURES == frozenset({"q_rev_growth", "revenueGrowth", "earningsGrowth", "y_earnings_growth"})
    assert set(CATALOGUE) == set(RAW_FEATURES)
    assert len(FEATURE_COLUMNS) == 218
    assert len(set(FEATURE_COLUMNS)) == len(FEATURE_COLUMNS)
    assert {column for column in FEATURE_COLUMNS if column.endswith("_vs_peers")} == {"f_fcf_to_ev_vs_peers", "f_roic_incl_intangibles_vs_peers"}
    assert PEER_FEATURES == frozenset({"fcf_to_ev", "roic_incl_intangibles"})
    assert not any(column.endswith("_xs") for column in FEATURE_COLUMNS)

    print("\n=== SANITY CHECK: approved fundamentals feature views ===")
    print("  137 raw + 79 self-history + 2 existing-peer comparisons = 218 unique columns.")
    print("  Obsolete annual earnings aliases and redundant revenue growth views are excluded.")


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


def test_selected_peer_views_reuse_policy_and_preserve_raw_history() -> None:
    idx = pd.bdate_range("2024-01-01", periods=6)
    fields = {name: pd.DataFrame(np.arange(30).reshape(6, 5) + 1.0, index=idx, columns=list("ABCDE")) for name in PEER_FEATURES}
    peers = {name: {peer: 1.0 for peer in "ABCDE" if peer != name} for name in "ABCDE"}
    panel = build_fundamental_views(fields, peers, history_window=3, history_min_periods=2).set_index(["date", "ticker"])
    reference = build_peer_relative_panel(fields, peers, emission={name: "raw+peers" for name in fields}).set_index(["date", "ticker"])
    for name in fields:
        for suffix in ("", "_vs_peers"):
            column = f"f_{name}{suffix}"
            pd.testing.assert_series_equal(panel[column], reference[column])
        assert panel[f"f_{name}_vs_hist"].notna().any()
    print("\nSANITY: two peer views use the shared support, dispersion and clipping policy while preserving raw and own-history views.")


def test_new_valuation_views_real_yearly_cross_sections() -> None:
    context = SimpleNamespace(store=_store(), paths=define_global_paths(read_config("./configs")))
    peers = load_peers(context)
    if not peers:
        pytest.skip("existing peer cache is unavailable; this check never reconstructs it")
    metadata = context.store.load(Tables.sp500_tickers, columns=["ticker", "sector"])
    names = metadata.ticker.astype(str).to_list()
    calendar = pd.DatetimeIndex(pd.to_datetime(context.store.distinct(Tables.cube_part_fundamentals, "date")))
    recent = pd.Series(calendar[calendar >= "2022-01-01"])
    dates = pd.DatetimeIndex(recent.groupby(recent.dt.year).max().to_list())
    if dates.empty:
        pytest.skip("no persisted 2022+ fundamentals snapshots")
    raw = context.store.load(
        Tables.cube_part_fundamentals,
        columns=["date", "ticker", "f_fcf_to_ev", "f_roic_incl_intangibles"],
        where={"ticker": names, "date": dates.to_list()},
    )
    raw["date"] = pd.to_datetime(raw.date)
    fields = {name: raw.pivot(index="date", columns="ticker", values=f"f_{name}").reindex(index=dates, columns=names) for name in PEER_FEATURES}
    part = build_fundamental_views(fields, peers)
    prices = context.store.load(
        Tables.cube_part_prices,
        columns=["date", "ticker", "close_split", "level_factor"],
        where={"ticker": names, "date": dates.to_list()},
    )
    prices["date"] = pd.to_datetime(prices.date)
    close = prices.pivot(index="date", columns="ticker", values="close_split").reindex(index=dates, columns=names)
    factor = prices.pivot(index="date", columns="ticker", values="level_factor").reindex_like(close)
    history = context.store.load(
        Tables.fundamentals_history,
        columns=["ticker", "as_of", "sharesOutstanding", "operatingCashFlow"],
        where={"ticker": names},
        since=dates.min() - pd.Timedelta(days=370),
        until=dates.max(),
    )
    market_cap = daily_market_cap(history, close, level_factor=factor, max_age_days=185)
    ocf = fundamentals_to_daily(history, "operatingCashFlow", dates, max_age_days=185)
    output = {"f_ocf_yield": ratio(ocf, market_cap, positive_den=True).reindex(index=dates, columns=names)}
    for name in PEER_FEATURES:
        column = f"f_{name}_vs_peers"
        output[column] = part.pivot(index="date", columns="ticker", values=column).reindex(index=dates, columns=names)
    print(f"\nSANITY: {len(names)} names across {len(dates)} annual/current valuation snapshots; existing peer cache reused:")
    for name, frame in output.items():
        values = frame.stack().dropna()
        assert len(values) > 0
        assert np.isfinite(values).all()
        if name.endswith("_vs_peers"):
            assert values.abs().le(8).all()
        tails = values.quantile([0, 0.001, 0.01, 0.5, 0.99, 0.999, 1]).round(6).to_list()
        yearly = frame.notna().mean(axis=1).mul(100).round(1).to_dict()
        sectors = {sector: round(frame.loc[:, rows.ticker].notna().to_numpy().mean() * 100, 1) for sector, rows in metadata.groupby("sector")}
        print(f"  {name}: coverage={len(values) / frame.size:.1%}, min/p0.1/p1/p50/p99/p99.9/max={tails}; by date %={yearly}; by sector %={sectors}")
        for date, ticker in [values.idxmin(), values.idxmax()]:
            if name == "f_ocf_yield":
                legs = {"operatingCashFlow": ocf.loc[date, ticker], "market_cap": market_cap.loc[date, ticker]}
            else:
                raw_name = split(name)[0]
                legs = {
                    "own_raw": fields[raw_name].loc[date, ticker],
                    "observed_peers": fields[raw_name].loc[date].reindex(list(peers.get(ticker, {}))).notna().sum(),
                }
            print(f"    tail {ticker} {date.date()}: {values.loc[(date, ticker)]:.6f}, source legs={legs}")
    print("  Cash-flow signs and finite denominators hold; peer support and clips follow the existing shared policy. No profitability claim is made.")


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
    for horizon, columns in (forest.get("columns_by_horizon") or {}).items():
        _assert_unique(f"random_forest.columns_by_horizon.{horizon}", columns)

    print("\n=== SANITY CHECK: model-config compatibility ===")
    print("  three active configs contain no removed fundamentals views and no duplicate selections.")
