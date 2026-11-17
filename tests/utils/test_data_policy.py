"""Approved table-policy values and fail-fast configuration contracts."""

from __future__ import annotations

from copy import deepcopy
from functools import cache

import pandas as pd
import pytest
from omegaconf import DictConfig, OmegaConf

from src.data_store.schema import ALL, Table, Tables
from src.utils.config import read_config
from src.utils.data_policy import publication_policy, validate_data_policy, window_policy
from src.utils.window import Offset, WindowRun, resolve_window


@cache
def _config() -> DictConfig:
    return read_config("configs")


def test_all_registry_tables_have_an_explicit_policy() -> None:
    config = _config()
    validate_data_policy(config)
    assert set(config.data.tables) == {table.name for table in ALL}
    assert config.data.tables.universe_tickers.status == "snapshot"
    assert config.data.tables.notes_embedding.status == "inactive"
    assert config.data.tables.predictions_latest.status == "model_owned"
    print("SANITY: all63 table identities have explicit active, snapshot, inherited or inactive policies.")


@pytest.mark.parametrize(
    ("table", "value", "unit"),
    [
        (Tables.prices, 7, "calendar_days"),
        (Tables.dividends, 7, "calendar_days"),
        (Tables.prices_macro, 7, "calendar_days"),
        (Tables.sharadar_fundamentals_facts, 100, "calendar_days"),
        (Tables.sec_short_volume_security, 30, "calendar_days"),
        (Tables.cube_part_prices, 30, "calendar_days"),
        (Tables.cube_part_betas, 30, "calendar_days"),
        (Tables.cube_part_targets, 90, "trading_sessions"),
        (Tables.cube_part_fundamentals, 100, "calendar_days"),
        (Tables.cube_part_momentum, 30, "calendar_days"),
        (Tables.cube_part_text, 7, "calendar_days"),
        (Tables.cube_part_institutionals, 30, "calendar_days"),
        (Tables.cube_part_sec_thirteen, 100, "calendar_days"),
        (Tables.cube_part_governance, 380, "calendar_days"),
    ],
)
def test_approved_numeric_overlaps_are_operative(table: Table, value: int, unit: str) -> None:
    policy = window_policy(_config(), table, pd.Timestamp("2026-11-16"))
    assert policy is not None
    assert policy.overlap == Offset(value, unit)
    print(f"SANITY: {table.name} resolves {value} {unit} from its sole configured owner.")


def test_nonrolling_work_has_no_artificial_overlap() -> None:
    config = _config()
    for table in (Tables.sec_13d, Tables.sec_fundamentals_facts, Tables.notes_num, Tables.sec13f_hr):
        policy = window_policy(config, table, pd.Timestamp("2026-11-16"))
        assert policy is not None
        assert policy.overlap is None
    assert config.data.tables.earnings_call_sections.reprocess.lookback == {"value": 180, "unit": "calendar_days"}
    print("SANITY: documents/archives have no numeric replay tail; calls have an explicit180-day source check.")


def test_fixed_dates_and_configured_overlap_reach_the_shared_resolver() -> None:
    config = deepcopy(_config())
    item = config.data.tables.cube_part_governance
    item.start_date, item.end_date = "2025-01-01", "2026-11-13"
    item.reprocess.overlap = {"value": 30, "unit": "calendar_days"}
    calendar = pd.bdate_range("2020-01-01", "2026-11-16")
    policy = window_policy(config, Tables.cube_part_governance, pd.Timestamp("2026-11-16"))
    assert policy is not None
    plan = resolve_window(
        policy,
        WindowRun("incremental", "configured policy counterfactual", pd.Timestamp("2026-11-16")),
        pd.Timestamp("2026-11-13"),
        calendar,
        (("ticker", ("AAPL",)),),
    )
    assert plan.requested.start == pd.Timestamp("2025-01-01")
    assert plan.write.start == pd.Timestamp("2026-10-14")
    assert plan.write.end == pd.Timestamp("2026-11-13")
    assert plan.read.start < plan.write.start
    print("SANITY: changed YAML dates/overlap reach the shared bounded plan; warm-up only expands reads.")


@pytest.mark.parametrize(
    ("table", "expected"),
    [(Tables.sec13f_hr, 2), (Tables.sec_13d, 1), (Tables.dividends, 1), (Tables.earnings_call_sections, 3), (Tables.earnings_surprises, 1)],
)
def test_explicit_publication_buffers(table: Table, expected: int) -> None:
    policy = publication_policy(_config(), table)
    assert policy.lag == {"value": expected, "unit": "calendar_days"}
    assert policy.roll == "market_session"
    print(f"SANITY: {table.name} declares +{expected} calendar days and actual-market-session roll.")


@pytest.mark.parametrize(
    "mutation",
    ["unknown_table", "missing_table", "invalid_date", "invalid_unit", "unknown_clock", "inheritance_cycle"],
)
def test_invalid_data_policy_is_rejected_before_io(mutation: str) -> None:
    config = deepcopy(_config())
    if mutation == "unknown_table":
        config.data.tables.typo = OmegaConf.create({"status": "inactive"})
    elif mutation == "missing_table":
        del config.data.tables.prices
    elif mutation == "invalid_date":
        config.data.tables.prices.start_date = "2026-02-30"
    elif mutation == "invalid_unit":
        config.data.tables.prices.reprocess.overlap.unit = "business_days"
    elif mutation == "unknown_clock":
        config.data.tables.prices.pit.clock = "unknown_publication_clock"
    else:
        config.data.tables.insider_transactions.pit = {"mode": "inherit", "source": "insider_footnotes"}
    with pytest.raises(ValueError):
        validate_data_policy(config)
    print(f"SANITY: {mutation} fails policy validation before any source or store call.")


def test_candidate_work_is_bounded_from_the_shared_run_anchor() -> None:
    config = _config()
    day = pd.Timestamp("2026-11-16")
    policy = window_policy(config, Tables.earnings_call_sections, day)
    assert policy is not None
    run = WindowRun("incremental", "routine source candidate check", day)
    plan = resolve_window(policy, run, None, pd.bdate_range("2005-01-03", day), (("ticker", ("AAPL",)),))
    assert plan.write.start == pd.Timestamp("2026-05-20")
    assert plan.requested.start == pd.Timestamp("2005-01-01")
    assert plan.write.end == day
    print("SANITY: routine calls check180 calendar days from the run anchor, separate from hard source history.")


def test_assembly_inherits_actual_component_start_and_its_own_catchup() -> None:
    config = _config()
    day = pd.Timestamp("2026-11-16")
    policy = window_policy(config, Tables.cube, day)
    assert policy is not None
    run = WindowRun("incremental", "selected component assembly", day)
    calendar = pd.bdate_range("1995-01-03", day)
    with pytest.raises(ValueError, match="component"):
        resolve_window(policy, run, pd.Timestamp("2025-09-15"), calendar, (("ticker", ("AAPL",)),))
    plan = resolve_window(policy, run, pd.Timestamp("2025-09-15"), calendar, (("ticker", ("AAPL",)),), inherited_start=pd.Timestamp("2025-10-21"))
    assert plan.write.start == pd.Timestamp("2025-09-15")
    print("SANITY: assembly requires selected component boundaries and also catches up its own earlier missing dates.")


def test_source_before_its_configured_history_has_no_work_window() -> None:
    config = _config()
    assert window_policy(config, Tables.earnings_call_sections, pd.Timestamp("2000-01-03")) is None
    assert window_policy(config, Tables.universe_tickers, pd.Timestamp("2026-11-16")) is None
    print("SANITY: unavailable early source history and undated snapshots have no invented execution interval.")
