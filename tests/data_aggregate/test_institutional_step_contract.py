"""Separate source ownership, single payload reads and daily persistence contracts."""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import pandas as pd
import pytest
from omegaconf import OmegaConf

from src.constants.constants import DEFAULT_CONFIG_DIR
from src.context import Context
from src.data_aggregate.transformers import step_cube_institutionals as institutional_step
from src.data_aggregate.transformers.step_cube_institutionals import StepCubeInstitutionals
from src.data_aggregate.transformers.step_cube_sec_thirteen import StepCubeSecThirteen
from src.data_aggregate.utils.common import inputs as capital_inputs
from src.data_aggregate.utils.common.incremental import PartWindow
from src.data_aggregate.utils.common.price_frames import PriceFrames
from src.data_aggregate.utils.institutionals.inputs import QuantityBasis
from src.data_store.schema import Tables
from src.utils.price_adjustments import resolve_share_events
from tests.data_aggregate.insider_pit_reference import row


def context(store: Any, path: Path) -> Context:
    config = OmegaConf.create({"build_cube": {}, "local": {"paths": {"fails_deliver": "ftd"}}})
    return cast(
        Context,
        SimpleNamespace(store=store, log=logging.getLogger(__name__), config=config, config_dir=Path(DEFAULT_CONFIG_DIR), paths={"DATA_STORE": path}),
    )


def seed_prices(store: Any, periods: int = 40) -> pd.DatetimeIndex:
    days = pd.bdate_range("2024-01-02", periods=periods, name="date")
    store.replace(
        Tables.cube_part_prices,
        pd.DataFrame({"date": days, "ticker": "AAA", "close_split": 100.0, "close_total": 100.0, "volume": 1000.0, "level_factor": 1.0}),
    )
    return days


def test_institutionals_emits_only_three_source_panels(sqlite_store: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    days = seed_prices(sqlite_store)
    ctx = context(sqlite_store, tmp_path)
    step = StepCubeInstitutionals(ctx, ctx.config)
    monkeypatch.setattr(capital_inputs, "load_peers_or_raise", lambda ctx: {})
    calls: list[str] = []

    def panel(name: str) -> pd.DataFrame:
        calls.append(name)
        return pd.DataFrame({"date": days, "ticker": "AAA", "f_" + name: 0.0})

    monkeypatch.setattr(step, "_insider_panel", lambda *args, **kwargs: panel("ic_insider_test"))
    monkeypatch.setattr(step, "_short_flow_panel", lambda *args, **kwargs: panel("ic_shortvol_test"))
    monkeypatch.setattr(step, "_ownership_panel", lambda *args, **kwargs: panel("ic_act_test"))
    result = step.build_panel()
    assert step._store is sqlite_store
    assert calls == ["ic_insider_test", "ic_shortvol_test", "ic_act_test"]
    assert set(result.columns) == {"date", "ticker", *("f_" + name for name in calls)}
    assert not result.duplicated(["date", "ticker"]).any()
    print("SANITY: institutionals merges exactly insider, short-flow and ownership on price keys.")


@pytest.mark.parametrize(
    "cls, table", [(StepCubeInstitutionals, Tables.cube_part_institutionals), (StepCubeSecThirteen, Tables.cube_part_sec_thirteen)]
)
def test_each_daily_part_rewrites_its_own_inclusive_seven_session_tail(
    sqlite_store: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    cls: type,
    table: Any,
) -> None:
    days = seed_prices(sqlite_store)
    ctx = context(sqlite_store, tmp_path)
    step = cls(ctx, ctx.config)
    expected = pd.DataFrame({"date": days, "ticker": "AAA", "f_test": pd.Series(range(len(days)), dtype="float32")})
    initial = expected.iloc[:31].copy()
    cutoff = days[30 - 7]
    initial.loc[initial["date"] >= cutoff, "f_test"] = -1.0
    sqlite_store.replace(table, initial)
    if cls is StepCubeSecThirteen:
        monkeypatch.setattr(step, "build_panel", lambda window=None, *, existing_columns=None: expected.copy())
    else:
        monkeypatch.setattr(step, "build_panel", lambda full=False: expected.copy())
    step.run()
    first = sqlite_store.load(table).sort_values(["date", "ticker"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(first, expected, check_dtype=False)
    step.run()
    second = sqlite_store.load(table).sort_values(["date", "ticker"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(first, second)
    assert len(initial.loc[initial["date"] >= cutoff]) == 8
    other = Tables.cube_part_sec_thirteen if table == Tables.cube_part_institutionals else Tables.cube_part_institutionals
    assert not sqlite_store.exists(other)
    print(f"SANITY: {table.name} rewrites last-7 inclusively (8 stored dates), appends and reruns idempotently.")


@pytest.mark.parametrize("cls", [StepCubeInstitutionals, StepCubeSecThirteen])
def test_builder_payload_reads_are_unique_with_optional_sources_absent(
    sqlite_store: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    cls: type,
) -> None:
    seed_prices(sqlite_store)
    ctx = context(sqlite_store, tmp_path)
    step = cls(ctx, ctx.config)
    monkeypatch.setattr(capital_inputs, "load_peers_or_raise", lambda ctx: {})
    reads: Counter[str] = Counter()
    for method in ("load", "iter_load"):
        original = getattr(sqlite_store, method)

        def capture(table: Any, *args: Any, _original: Any = original, **kwargs: Any) -> Any:
            reads[table.name] += 1
            return _original(table, *args, **kwargs)

        monkeypatch.setattr(sqlite_store, method, capture)
    result = step.build_panel()
    assert set(result.columns) == {"date", "ticker"}
    assert reads and max(reads.values()) == 1, reads
    print(f"SANITY: {cls.__name__} reads each payload once, including absent optional sources.")


def test_ftd_step_retains_global_zip_dates_from_one_stream(sqlite_store: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    idx = pd.bdate_range("2023-12-01", "2024-02-14", name="date")
    sqlite_store.save(
        Tables.sec_fails_to_deliver,
        pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-02", "2024-01-12"]),
                "ticker": ["AAA", "OTHER"],
                "fails_quantity": [100.0, 50.0],
                "period": ["202401a", "202401a"],
            }
        ),
    )
    ctx = context(sqlite_store, tmp_path)
    step = StepCubeInstitutionals(ctx, ctx.config)
    frames = PriceFrames(trading_index=idx, universe=("AAA",), peers={}, volume=pd.DataFrame({"AAA": 1000.0}, index=idx))
    streams = []
    original = sqlite_store.iter_load

    def capture(table: Any, **kwargs: Any) -> Any:
        streams.append(table)
        return original(table, **kwargs)

    monkeypatch.setattr(sqlite_store, "iter_load", capture)
    result = step._short_flow_panel(frames, None, None, pd.DataFrame(), QuantityBasis.from_context(resolve_share_events(None, None, {}), idx))
    assert result is not None
    ratio = result.set_index(["date", "ticker"])["f_ic_ftd_to_adv20"]
    assert ratio.loc[(pd.Timestamp("2024-01-30"), "AAA")] == 0
    assert streams.count(Tables.sec_fails_to_deliver) == 1
    assert set(result["ticker"]) == {"AAA"}
    print("SANITY: actual step uses OTHER's observed ZIP date to publish AAA's zero with one FTD stream.")


@pytest.mark.parametrize("source", [None, pd.DataFrame(), pd.DataFrame({"ticker": ["AAA"], "shares": [10.0]})], ids=["absent", "empty", "malformed"])
def test_public_step_preserves_unusable_insider_source_boundary(
    sqlite_store: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, source: pd.DataFrame | None
) -> None:
    seed_prices(sqlite_store)
    ctx = context(sqlite_store, tmp_path)
    step = StepCubeInstitutionals(ctx, ctx.config)
    monkeypatch.setattr(capital_inputs, "load_peers_or_raise", lambda ctx: {})
    monkeypatch.setattr("src.data_aggregate.utils.institutionals.inputs.load_insider_transactions", lambda *args: source)

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("unusable source reached quantity reconstruction")

    monkeypatch.setattr("src.data_aggregate.transformers.step_cube_institutionals.position_units", forbidden)
    result = step.build_panel()
    assert set(result.columns) == {"date", "ticker"}
    print(
        "SANITY: absent, empty or malformed optional insider payload never enters share reconstruction and keeps the public step's empty-family behavior."
    )


def test_short_stage_uses_the_owner_publication_clock(sqlite_store: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    idx = pd.bdate_range("2023-12-01", "2024-02-14", name="date")
    sqlite_store.save(
        Tables.sec_fails_to_deliver,
        pd.DataFrame({"date": pd.to_datetime(["2024-01-15"]), "ticker": ["AAA"], "fails_quantity": [100.0], "period": ["202401a"]}),
    )
    ctx = context(sqlite_store, tmp_path)
    step = StepCubeInstitutionals(ctx, ctx.config)
    frames = PriceFrames(trading_index=idx, universe=("AAA",), peers={}, volume=pd.DataFrame({"AAA": 1000.0}, index=idx))
    calls: list[object] = []

    def fixed_clock(coverage: pd.DataFrame, cache_dir: Path | None, **kwargs: Any) -> dict[str, pd.Timestamp]:
        calls.append(coverage)
        return {"202401a": pd.Timestamp("2024-02-05")}

    monkeypatch.setattr("src.data_aggregate.utils.institutionals.short_flow_features._ftd_available_dates", fixed_clock)
    events = resolve_share_events(None, pd.DataFrame(columns=["ticker", "date", "ratio"]), {})
    result = step._short_flow_panel(frames, None, None, pd.DataFrame(), QuantityBasis.from_context(events, idx))
    assert result is not None and len(calls) == 1
    values = result.set_index(["date", "ticker"])["f_ic_ftd_to_adv20"]
    assert pd.isna(values.loc[(pd.Timestamp("2024-02-02"), "AAA")])
    assert values.loc[(pd.Timestamp("2024-02-05"), "AAA")] == pytest.approx(0.1)
    print("SANITY: the staged builder resolves the owning module's patched clock exactly once and publishes no FTD value before its fixed session.")


def test_populated_insider_stage_reads_before_clean_rebase_compute(sqlite_store: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    days = seed_prices(sqlite_store)
    df_source = pd.DataFrame([row(ticker="AAA", transaction_date="2024-01-03", filing_date="2024-01-04", lineage_role="canonical_current")])
    sqlite_store.save(Tables.insider_transactions, df_source)
    ctx = context(sqlite_store, tmp_path)
    step = StepCubeInstitutionals(ctx, ctx.config)
    monkeypatch.setattr(capital_inputs, "load_peers_or_raise", lambda ctx: {})
    stages: list[str] = []
    reads: Counter[str] = Counter()
    computing = False
    for method in ("load", "iter_load"):
        original = getattr(sqlite_store, method)

        def capture(table: Any, *args: Any, _original: Any = original, **kwargs: Any) -> Any:
            assert not computing, "clean/rebase/compute performed a hidden table payload read"
            reads[table.name] += 1
            return _original(table, *args, **kwargs)

        monkeypatch.setattr(sqlite_store, method, capture)
    clean = institutional_step.clean_transactions
    rebase = institutional_step.position_units
    compute = institutional_step.build_insider_feature_panel

    def clean_stage(*args: Any, **kwargs: Any) -> Any:
        nonlocal computing
        computing = True
        stages.append("clean")
        return clean(*args, **kwargs)

    def rebase_stage(*args: Any, **kwargs: Any) -> Any:
        stages.append("rebase")
        return rebase(*args, **kwargs)

    def compute_stage(*args: Any, **kwargs: Any) -> Any:
        nonlocal computing
        stages.append("compute")
        result = compute(*args, **kwargs)
        computing = False
        return result

    monkeypatch.setattr(institutional_step, "clean_transactions", clean_stage)
    monkeypatch.setattr(institutional_step, "position_units", rebase_stage)
    monkeypatch.setattr(institutional_step, "build_insider_feature_panel", compute_stage)
    result = step.build_panel()
    assert stages == ["clean", "rebase", "compute"]
    assert reads[Tables.insider_transactions.name] == 1 and max(reads.values()) == 1
    assert len(result) == len(days)
    print(
        "SANITY: the populated public step loads each payload once, then executes clean→rebase→compute with a store-read tripwire active throughout the numerical stages."
    )


@pytest.mark.parametrize(
    "cls, table", [(StepCubeInstitutionals, Tables.cube_part_institutionals), (StepCubeSecThirteen, Tables.cube_part_sec_thirteen)]
)
def test_changed_columns_trigger_full_owned_part_replacement(
    sqlite_store: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cls: type, table: Any
) -> None:
    days = seed_prices(sqlite_store)
    ctx = context(sqlite_store, tmp_path)
    step = cls(ctx, ctx.config)
    sqlite_store.replace(table, pd.DataFrame({"date": days[:3], "ticker": "AAA", "f_old": 9.0}))
    expected = pd.DataFrame({"date": days, "ticker": "AAA", "f_current": 1.0})
    modes: list[bool] = []

    def build(full: bool = False) -> pd.DataFrame:
        modes.append(full)
        return expected.copy()

    def sec_build(window: PartWindow | None = None, *, existing_columns: Sequence[str] | None = None) -> pd.DataFrame:
        assert window is not None and existing_columns is not None
        modes.append(window.is_full)
        return expected.copy()

    monkeypatch.setattr(step, "build_panel", sec_build if cls is StepCubeSecThirteen else build)
    replace = Mock(wraps=sqlite_store.replace)
    monkeypatch.setattr(sqlite_store, "replace", replace)
    step.run()
    assert modes == [False]
    replace.assert_called_once()
    assert set(sqlite_store.columns(table)) == set(expected.columns)
    assert sqlite_store.row_count(table) == len(expected)
    saved = sqlite_store.load(table).sort_values(["date", "ticker"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(saved, expected, check_dtype=False)
    print(f"SANITY: {table.name} builds once and replaces its obsolete schema once; all {len(expected)} saved rows match.")
