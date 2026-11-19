"""The simplified policy has one owner and explicit additional input history."""

from __future__ import annotations

from copy import deepcopy

import pandas as pd
import pytest
from omegaconf import OmegaConf

from src.data_store.schema import Tables
from src.utils.config import read_config
from src.utils.data_policy import input_read_window, publication_policy, source_read_window, validate_data_policy, window_policy
from src.utils.window import Offset, Window


def test_policy_inventory_and_price_derived_cube_dates() -> None:
    config = read_config("configs")
    assert len(config.data.tables) == 47
    assert "universe_tickers" not in config.data.tables
    assert all("start_date" not in item for name, item in config.data.tables.items() if name == "cube" or name.startswith("cube_part_"))
    calendar = pd.DatetimeIndex(["2020-01-02", "2026-11-13", "2026-11-16"])
    policy = window_policy(config, Tables.cube_part_momentum, pd.Timestamp("2026-11-13"), calendar)
    assert policy is not None and policy.requested == Window(pd.Timestamp("2020-01-02"), pd.Timestamp("2026-11-13"))
    assert policy.overlap == Offset(30, "calendar_days")
    print("SANITY:47 applicable policies; cube bounds come from observed prices, with one calendar-day overlap.")


@pytest.mark.parametrize("field", ["status", "end_date", "source_start", "coverage_start", "freshness", "reprocess", "lookahead"])
def test_obsolete_source_fields_are_rejected(field: str) -> None:
    config = read_config("configs")
    config.data.tables.prices[field] = None
    with pytest.raises(ValueError, match="Invalid policy fields"):
        validate_data_policy(config)
    print(f"SANITY: obsolete source field {field} fails before I/O.")


@pytest.mark.parametrize("field", ["planning", "freshness", "derived_feature_starts"])
def test_obsolete_global_controls_are_rejected(field: str) -> None:
    config = read_config("configs")
    config.data[field] = {}
    with pytest.raises(ValueError):
        validate_data_policy(config)
    print(f"SANITY: deleted global control {field} cannot become a second policy owner.")


@pytest.mark.parametrize("value", [-1, True, 1.5, "full"])
def test_invalid_warmup_is_rejected(value: object) -> None:
    config = read_config("configs")
    config.data.tables.cube_part_momentum.warmup.cube_part_prices = value
    with pytest.raises(ValueError, match="calendar-day"):
        validate_data_policy(config)
    print(f"SANITY: invalid warmup {value!r} refuses instead of guessing history.")


def test_missing_null_and_zero_warmup_are_distinct() -> None:
    config = read_config("configs")
    write = Window(pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-31"))
    run = pd.Timestamp("2026-11-16")
    amount = config.data.tables.cube_part_momentum.warmup.cube_part_prices
    assert input_read_window(config, Tables.cube_part_momentum, Tables.cube_part_prices, write, run) == Window(
        write.start - pd.Timedelta(days=amount), run
    )
    config.data.tables.cube_part_momentum.warmup.cube_part_prices = 0
    assert input_read_window(config, Tables.cube_part_momentum, Tables.cube_part_prices, write, run) == Window(write.start, run)
    config.data.tables.cube_part_momentum.warmup.cube_part_prices = None
    assert input_read_window(config, Tables.cube_part_momentum, Tables.cube_part_prices, write, run) is None
    del config.data.tables.cube_part_momentum.warmup.cube_part_prices
    with pytest.raises(ValueError, match="missing warmup"):
        input_read_window(config, Tables.cube_part_momentum, Tables.cube_part_prices, write, run)
    print("SANITY: stale rewrite anchors finite reads;0 keeps its inclusive start;null is full;missing is an error.")


def test_approved_ftd_basis_dependencies_are_explicit_full_reads() -> None:
    config = read_config("configs")
    mapping = config.data.tables.cube_part_institutionals.warmup
    for table in (Tables.prices_splits, Tables.sharadar_actions, Tables.sharadar_fundamentals_facts):
        assert table.name in mapping and mapping[table.name] is None
    assert mapping.prices == 750
    print("SANITY: approved FTD share anchors retain complete evidence; secondary-class tape retains750CD.")


def test_unknown_tables_columns_and_cube_start_are_contract_errors() -> None:
    config = read_config("configs")
    cases = [
        ("data.tables.prices.pit.column", "unknown"),
        ("data.tables.prices.overlap.column", "unknown"),
        ("data.tables.cube_part_prices.start_date", "2020-01-01"),
        ("data.tables.cube_part_momentum.warmup.unknown", 0),
        ("data.tables.prices.overlap.method", "tail"),
    ]
    for path, value in cases:
        changed = deepcopy(config)
        OmegaConf.update(changed, path, value)
        with pytest.raises(ValueError):
            validate_data_policy(changed)
    print("SANITY: unknown identities/clocks, old methods and cube start dates fail validation.")


def test_sources_retain_fixed_extraction_bounds_and_explicit_publication() -> None:
    config = read_config("configs")
    before = source_read_window(config, Tables.prices, pd.Timestamp("1990-01-01"))
    assert before is None
    early = source_read_window(config, Tables.prices, pd.Timestamp("2020-01-02"))
    late = source_read_window(config, Tables.prices, pd.Timestamp("2026-11-16"))
    assert early is not None and late is not None and early.start == late.start == pd.Timestamp("1995-10-09")
    assert publication_policy(config, Tables.notes_num).lag_days == 12
    assert publication_policy(config, Tables.notes_text).column == "period"
    with pytest.raises(ValueError, match="no source publication"):
        publication_policy(config, Tables.sharadar_actions)
    print("SANITY: extraction starts stay fixed; notes use explicit period+12; structural snapshots have no invented lag.")
