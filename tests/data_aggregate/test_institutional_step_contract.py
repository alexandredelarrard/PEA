"""Separate source ownership, single payload reads and daily persistence contracts."""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pandas as pd
import pytest
from omegaconf import OmegaConf

from src.constants.constants import DEFAULT_CONFIG_DIR
from src.context import Context
from src.data_aggregate.transformers.step_cube_institutionals import StepCubeInstitutionals
from src.data_aggregate.transformers.step_cube_sec_thirteen import StepCubeSecThirteen
from src.data_aggregate.utils.common import inputs as capital_inputs
from src.data_aggregate.utils.common.incremental import PartWindow
from src.data_aggregate.utils.common.price_frames import PriceFrames
from src.data_store.schema import Tables
from src.utils.price_adjustments import resolve_share_events


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
    result = step._short_flow_panel(frames, None, None, None, pd.DataFrame(), share_basis=resolve_share_events(None, None, {}))
    assert result is not None
    ratio = result.set_index(["date", "ticker"])["f_ic_ftd_to_adv20"]
    assert ratio.loc[(pd.Timestamp("2024-01-30"), "AAA")] == 0
    assert streams.count(Tables.sec_fails_to_deliver) == 1
    assert set(result["ticker"]) == {"AAA"}
    print("SANITY: actual step uses OTHER's observed ZIP date to publish AAA's zero with one FTD stream.")


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
    step.run()
    expected_modes = [False] if cls is StepCubeSecThirteen else [False, True]
    assert modes == expected_modes
    assert set(sqlite_store.columns(table)) == set(expected.columns)
    assert sqlite_store.row_count(table) == len(expected)
    saved = sqlite_store.load(table).sort_values(["date", "ticker"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(saved, expected, check_dtype=False)
    strategy = "the already-computed full-history panel" if cls is StepCubeSecThirteen else "a full rerun"
    print(f"SANITY: {table.name} replaces its obsolete schema using {strategy}; all {len(expected)} saved rows match.")
