"""Exact owner/window mathematics and disposable SQLite persistence contracts."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from src.data_aggregate.transformers.step_cube_sec_thirteen import StepCubeSecThirteen
from src.data_aggregate.utils.common import inputs as capital_inputs
from src.data_aggregate.utils.common.incremental import PartWindow
from src.data_store.schema import Tables
from tests.data_aggregate.test_institutional_step_contract import context, seed_prices


def _owner(store: Any, monkeypatch: pytest.MonkeyPatch, path: Path) -> tuple[StepCubeSecThirteen, pd.DatetimeIndex]:
    days = seed_prices(store, periods=120)
    ctx = context(store, path)
    ctx.config.build_cube = OmegaConf.create({"incremental": {"warmup_trading_days": 0}, "institutionals": {"decay_halflife": {"super": 20}}})
    monkeypatch.setattr(capital_inputs, "load_peers_or_raise", lambda ctx: {})
    return StepCubeSecThirteen(ctx, ctx.config), days


def _facts(days: pd.DatetimeIndex, offsets: Sequence[int], **fields: Sequence[float]) -> pd.DataFrame:
    count = len(offsets)
    return pd.DataFrame(
        {
            "ticker": "AAA",
            "period": pd.Timestamp("2023-09-30"),
            "as_of": days[list(offsets)],
            "inst_shares": [20.0] * count,
            "inst_value": [2000.0] * count,
            **fields,
        }
    )


def _elite(days: pd.DatetimeIndex, offsets: Sequence[int], **fields: Sequence[float]) -> dict[str, pd.DataFrame]:
    return {name: pd.DataFrame({"AAA": values}, index=days[list(offsets)]) for name, values in fields.items()}


def _prepare(
    step: StepCubeSecThirteen,
    monkeypatch: pytest.MonkeyPatch,
    facts: pd.DataFrame | None,
    levels: dict[str, pd.DataFrame] | None,
) -> tuple[Mock, Mock]:
    broad, elite = Mock(return_value=facts), Mock(return_value=levels)
    monkeypatch.setattr(step, "_institutional_features", broad)
    monkeypatch.setattr(step, "_superinvestor_levels", elite)
    return broad, elite


def _projections(step: StepCubeSecThirteen, monkeypatch: pytest.MonkeyPatch) -> tuple[Mock, Mock]:
    broad, elite = Mock(wraps=step._project_institutional_panel), Mock(wraps=step._project_superinvestor_panel)
    monkeypatch.setattr(step, "_project_institutional_panel", broad)
    monkeypatch.setattr(step, "_project_superinvestor_panel", elite)
    return broad, elite


def _ordered(panel: pd.DataFrame) -> pd.DataFrame:
    return panel.sort_values(["date", "ticker"]).reset_index(drop=True)


def _saved(panel: pd.DataFrame) -> pd.DataFrame:
    features = [column for column in panel if column not in {"date", "ticker"}]
    return _ordered(panel.dropna(how="all", subset=features))


def _load(store: Any) -> pd.DataFrame:
    panel = store.load(Tables.cube_part_sec_thirteen)
    assert panel is not None
    return _ordered(panel)


def _shares(store: Any, days: pd.DatetimeIndex, offsets: Sequence[int], amounts: Sequence[float]) -> None:
    store.replace(
        Tables.fundamentals_history,
        pd.DataFrame({"ticker": "AAA", "as_of": days[list(offsets)], "sharesOutstanding": amounts, "sharesOutstandingPit": amounts}),
    )


def _bounded(projections: tuple[Mock, Mock], days: pd.DatetimeIndex, maximum: int) -> None:
    for family, projection in enumerate(projections):
        for call in projection.call_args_list:
            frames = call.args[1]
            assert 0 < len(frames.trading_index) <= maximum
            pd.testing.assert_index_equal(frames.close_split.index, frames.trading_index)
            pd.testing.assert_index_equal(frames.level_factor.index, frames.trading_index)
            assert frames.universe == ("AAA",)
            clock = call.args[4] if family == 0 else pd.DatetimeIndex(call.args[2].index)
            pd.testing.assert_index_equal(clock, days)


def test_tail_retains_full_dependency_clock_with_bounded_dense_projection(sqlite_store: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    step, days = _owner(sqlite_store, monkeypatch, tmp_path)
    prepared = _prepare(
        step,
        monkeypatch,
        _facts(days, (5, 90), ic_inst_concentration=[0.25, 0.5]),
        _elite(days, (8,), ic_super_holders=[0.5], ic_super_new_top10=[1.0]),
    )
    full = step.build_panel()
    for source in prepared:
        source.reset_mock()
    projections = _projections(step, monkeypatch)
    window = PartWindow(days[-1], days[-8], days[-8])
    tail = step.build_panel(window, existing_columns=list(full.columns))

    pd.testing.assert_frame_equal(_ordered(tail), _ordered(full.loc[full.date >= window.since]))
    assert [source.call_count for source in prepared] == [1, 1]
    assert [projection.call_count for projection in projections] == [1, 1]
    _bounded(projections, days, 8)
    event = tail.set_index(["date", "ticker"]).loc[(days[-8], "AAA"), "f_ic_super_new_top10"]
    assert event == pytest.approx(0.5 ** ((112 - 8) / 20))
    assert prepared[0].call_args.args[0].trading_index.equals(days)
    pd.testing.assert_index_equal(prepared[1].call_args.kwargs["session_positions"].index, days)
    print(
        "SANITY: an eight-session tail equals full history, preserves 104-session event age and prepares each source once; dense projections stay inside the tail."
    )


def test_early_window_uses_the_original_publication_floor(sqlite_store: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    step, _ = _owner(sqlite_store, monkeypatch, tmp_path)
    days = pd.bdate_range("2013-08-01", periods=120, name="date")
    sqlite_store.replace(Tables.cube_part_prices, pd.DataFrame({"date": days, "ticker": "AAA", "close_split": 100.0, "level_factor": 1.0}))
    facts = _facts(days, (0,), ic_inst_concentration=[0.25]).assign(period=pd.Timestamp("2013-06-30"))
    _prepare(step, monkeypatch, facts, {})
    full = step.build_panel()
    tail = step.build_panel(PartWindow(days[-1], pd.Timestamp("2013-08-15")), existing_columns=list(full.columns))
    pd.testing.assert_frame_equal(_ordered(tail), _ordered(full.loc[full.date >= "2013-08-15"]))
    values = tail.set_index("date").f_ic_inst_concentration
    assert values.loc[["2013-08-15", "2013-08-16"]].isna().all()
    assert values.loc["2013-08-19"] == 0.25
    print("SANITY: the three-session publication floor stays tied to the full clock when output starts inside that floor.")


@pytest.mark.parametrize("new_sessions", [0, 5])
def test_run_refreshes_same_day_or_new_dates_without_changing_prefix(
    sqlite_store: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, new_sessions: int
) -> None:
    step, days = _owner(sqlite_store, monkeypatch, tmp_path)
    prepared = _prepare(step, monkeypatch, _facts(days, (5, 95), ic_inst_concentration=[0.25, 0.5]), _elite(days, (5,), ic_super_holders=[0.5]))
    expected = _saved(step.build_panel())
    last = len(days) - 1 - new_sessions
    cutoff = days[last - 7]
    initial = expected.loc[expected.date <= days[last]].copy()
    initial.loc[initial.date >= cutoff, "f_ic_inst_concentration"] = -9.0
    sqlite_store.replace(Tables.cube_part_sec_thirteen, initial)
    prefix = _ordered(initial.loc[initial.date < cutoff])
    for source in prepared:
        source.reset_mock()
    projections = _projections(step, monkeypatch)

    step.run()
    first = _load(sqlite_store)
    pd.testing.assert_frame_equal(first, expected, check_dtype=False)
    pd.testing.assert_frame_equal(first.loc[first.date < cutoff].reset_index(drop=True), prefix, check_dtype=False)
    assert [source.call_count for source in prepared] == [1, 1]
    step.run()
    pd.testing.assert_frame_equal(_load(sqlite_store), first)
    assert [source.call_count for source in prepared] == [2, 2]
    _bounded(projections, days, 8 + new_sessions)
    print(
        f"SANITY: SQLite refresh repairs the inclusive eight-date tail plus {new_sessions} new dates, preserves the older prefix and is idempotent."
    )


def test_unknown_tail_clears_stale_rows_and_preserves_historical_schema(sqlite_store: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    step, days = _owner(sqlite_store, monkeypatch, tmp_path)
    broad, elite = _prepare(step, monkeypatch, _facts(days, (5,), ic_inst_concentration=[0.25]), _elite(days, (5,), ic_super_holders=[0.5]))
    initial = _saved(step.build_panel())
    sqlite_store.replace(Tables.cube_part_sec_thirteen, initial)
    cutoff = days[-8]
    broad.return_value = _facts(days, (5, 112), ic_inst_concentration=[0.25, np.nan], inst_shares=[20, np.nan], inst_value=[2000, np.nan])
    elite.return_value = _elite(days, (5, 112), ic_super_holders=[0.5, np.nan])
    broad.reset_mock()
    elite.reset_mock()
    projections = _projections(step, monkeypatch)

    step.run()
    expected = initial.loc[initial.date < cutoff].reset_index(drop=True)
    pd.testing.assert_frame_equal(_load(sqlite_store), expected, check_dtype=False)
    assert set(sqlite_store.columns(Tables.cube_part_sec_thirteen)) == set(initial.columns)
    assert broad.call_count == elite.call_count == 1
    _bounded(projections, days, 8)
    step.run()
    pd.testing.assert_frame_equal(_load(sqlite_store), expected, check_dtype=False)
    print("SANITY: an explicitly unknown tail deletes stale SQLite values while retaining supported historical columns and the unchanged prefix.")


@pytest.mark.parametrize("family", ["broad", "elite"])
@pytest.mark.parametrize("full", [False, True])
def test_failed_saved_family_refuses_writes(sqlite_store: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, family: str, full: bool) -> None:
    step, days = _owner(sqlite_store, monkeypatch, tmp_path)
    initial = pd.DataFrame({"date": days, "ticker": "AAA", "f_ic_inst_concentration": 0.25, "f_ic_super_holders": 0.5})
    sqlite_store.replace(Tables.cube_part_sec_thirteen, initial)
    _prepare(step, monkeypatch, None if family == "broad" else _facts(days, (5,), ic_inst_concentration=[0.25]), None if family == "elite" else {})
    replace, append = Mock(wraps=sqlite_store.replace), Mock(wraps=sqlite_store.append_tail)
    monkeypatch.setattr(sqlite_store, "replace", replace)
    monkeypatch.setattr(sqlite_store, "append_tail", append)
    with pytest.raises(RuntimeError, match="source preparation failed"):
        step.run(full=full)
    assert replace.call_count == append.call_count == 0
    pd.testing.assert_frame_equal(_load(sqlite_store), initial, check_dtype=False)
    print(f"SANITY: failed {family} preparation with full={full} preserves all saved cells and makes zero write calls.")


@pytest.mark.parametrize("full", [False, True])
def test_shortened_price_calendar_refuses_even_a_full_rebuild(sqlite_store: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, full: bool) -> None:
    step, days = _owner(sqlite_store, monkeypatch, tmp_path)
    initial = pd.DataFrame({"date": [days[-1] + pd.offsets.BDay(1)], "ticker": "AAA", "f_ic_inst_concentration": 0.25})
    sqlite_store.replace(Tables.cube_part_sec_thirteen, initial)
    prepared = _prepare(step, monkeypatch, pd.DataFrame(), {})
    replace, append = Mock(wraps=sqlite_store.replace), Mock(wraps=sqlite_store.append_tail)
    monkeypatch.setattr(sqlite_store, "replace", replace)
    monkeypatch.setattr(sqlite_store, "append_tail", append)
    with pytest.raises(RuntimeError, match="calendar ends before"):
        step.run(full=full)
    assert replace.call_count == append.call_count == 0
    assert [source.call_count for source in prepared] == [0, 0]
    pd.testing.assert_frame_equal(_load(sqlite_store), initial, check_dtype=False)
    print(f"SANITY: a saved frontier beyond prices blocks full={full} before source preparation or writes.")


@pytest.mark.parametrize("hole", ["saved_frontier", "rewrite_interior"])
@pytest.mark.parametrize("full", [False, True])
def test_missing_saved_rewrite_date_refuses_without_writes(
    sqlite_store: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, hole: str, full: bool
) -> None:
    step, days = _owner(sqlite_store, monkeypatch, tmp_path)
    initial = pd.DataFrame({"date": days[:-1], "ticker": "AAA", "f_ic_inst_concentration": 0.25})
    sqlite_store.replace(Tables.cube_part_sec_thirteen, initial)
    missing = days[-2] if hole == "saved_frontier" else days[-3]
    sqlite_store.delete(Tables.cube_part_prices, where={"date": missing})
    prepared = _prepare(step, monkeypatch, _facts(days, (5,), ic_inst_concentration=[0.25]), {})
    replace, append = Mock(wraps=sqlite_store.replace), Mock(wraps=sqlite_store.append_tail)
    distinct = Mock(wraps=sqlite_store.distinct)
    monkeypatch.setattr(sqlite_store, "replace", replace)
    monkeypatch.setattr(sqlite_store, "append_tail", append)
    monkeypatch.setattr(sqlite_store, "distinct", distinct)

    with pytest.raises(RuntimeError, match="calendar.*saved"):
        step.run(full=full)

    assert replace.call_count == append.call_count == 0
    assert [source.call_count for source in prepared] == [0, 0]
    pd.testing.assert_frame_equal(_load(sqlite_store), initial, check_dtype=False)
    saved_reads = [call for call in distinct.call_args_list if call.args[0] == Tables.cube_part_sec_thirteen]
    assert len(saved_reads) == 1 and saved_reads[0].args[1] == "date"
    print(f"SANITY: missing {hole} with full={full} refuses before preparation or writes even though prices extend past the saved frontier.")


def test_schema_promotion_refuses_missing_saved_prefix_date(sqlite_store: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    step, days = _owner(sqlite_store, monkeypatch, tmp_path)
    initial = pd.DataFrame({"date": days, "ticker": "AAA", "f_ic_inst_concentration": 0.25})
    sqlite_store.replace(Tables.cube_part_sec_thirteen, initial)
    sqlite_store.delete(Tables.cube_part_prices, where={"date": days[10]})
    prepared = _prepare(step, monkeypatch, _facts(days, (5,), ic_inst_concentration=[0.25], ic_inst_holders=[0.5]), {})
    replace, append = Mock(wraps=sqlite_store.replace), Mock(wraps=sqlite_store.append_tail)
    distinct = Mock(wraps=sqlite_store.distinct)
    monkeypatch.setattr(sqlite_store, "replace", replace)
    monkeypatch.setattr(sqlite_store, "append_tail", append)
    monkeypatch.setattr(sqlite_store, "distinct", distinct)

    with pytest.raises(RuntimeError, match="calendar.*saved"):
        step.run()

    assert replace.call_count == append.call_count == 0
    assert [source.call_count for source in prepared] == [1, 1]
    pd.testing.assert_frame_equal(_load(sqlite_store), initial, check_dtype=False)
    saved_reads = [call for call in distinct.call_args_list if call.args[0] == Tables.cube_part_sec_thirteen]
    assert len(saved_reads) == 1 and saved_reads[0].args[1] == "date"
    print("SANITY: genuine column promotion checks the whole saved calendar and preserves the missing prefix row with zero writes and one date read.")


def test_missing_saved_prefix_date_allows_one_incremental_write(sqlite_store: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    step, days = _owner(sqlite_store, monkeypatch, tmp_path)
    initial = pd.DataFrame({"date": days, "ticker": "AAA", "f_ic_inst_concentration": -9.0})
    sqlite_store.replace(Tables.cube_part_sec_thirteen, initial)
    sqlite_store.delete(Tables.cube_part_prices, where={"date": days[10]})
    prepared = _prepare(step, monkeypatch, _facts(days, (5,), ic_inst_concentration=[0.25]), {})
    replace, append = Mock(wraps=sqlite_store.replace), Mock(wraps=sqlite_store.append_tail)
    distinct = Mock(wraps=sqlite_store.distinct)
    monkeypatch.setattr(sqlite_store, "replace", replace)
    monkeypatch.setattr(sqlite_store, "append_tail", append)
    monkeypatch.setattr(sqlite_store, "distinct", distinct)

    step.run()

    expected = initial.copy()
    expected.loc[expected.date >= days[-8], "f_ic_inst_concentration"] = 0.25
    pd.testing.assert_frame_equal(_load(sqlite_store), expected, check_dtype=False)
    assert replace.call_count == 0 and append.call_count == 1
    assert [source.call_count for source in prepared] == [1, 1]
    saved_reads = [call for call in distinct.call_args_list if call.args[0] == Tables.cube_part_sec_thirteen]
    assert len(saved_reads) == 1 and saved_reads[0].args[1] == "date"
    print("SANITY: a calendar hole before the incremental deletion footprint preserves its saved row and allows exactly one safe tail write.")


@pytest.mark.parametrize("change", ["addition", "removal", "historical_addition"])
def test_schema_change_reprojects_full_once_from_prepared_facts(
    sqlite_store: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, change: str
) -> None:
    step, days = _owner(sqlite_store, monkeypatch, tmp_path)
    old = _facts(days, (5,), ic_inst_concentration=[0.25])
    if change == "removal":
        old["ic_inst_holders"] = 0.5
    broad, elite = _prepare(step, monkeypatch, old, _elite(days, (5,), ic_super_holders=[0.5]))
    sqlite_store.replace(Tables.cube_part_sec_thirteen, _saved(step.build_panel()))
    broad.return_value = _facts(days, (5, 112), ic_inst_concentration=[0.25, 0.5])
    if change != "removal":
        broad.return_value["ic_inst_holders"] = [0.5, np.nan if change == "historical_addition" else 0.75]
    expected = _saved(step.build_panel())
    broad.reset_mock()
    elite.reset_mock()
    projections = _projections(step, monkeypatch)
    replace, append = Mock(wraps=sqlite_store.replace), Mock(wraps=sqlite_store.append_tail)
    distinct = Mock(wraps=sqlite_store.distinct)
    monkeypatch.setattr(sqlite_store, "replace", replace)
    monkeypatch.setattr(sqlite_store, "append_tail", append)
    monkeypatch.setattr(sqlite_store, "distinct", distinct)

    step.run()
    pd.testing.assert_frame_equal(_load(sqlite_store), expected, check_dtype=False)
    assert broad.call_count == elite.call_count == 1
    assert replace.call_count == 1 and append.call_count == 0
    saved_reads = [call for call in distinct.call_args_list if call.args[0] == Tables.cube_part_sec_thirteen]
    assert len(saved_reads) == 1 and saved_reads[0].args[1] == "date"
    assert ("f_ic_inst_holders" in expected) == (change != "removal")
    for projection in projections:
        assert sum(len(call.args[1].trading_index) == len(days) for call in projection.call_args_list) == 1
        assert all(len(call.args[1].trading_index) in {8, len(days)} for call in projection.call_args_list)
    print(
        f"SANITY: {change} changes the physical SQLite schema with one full projection per family and one source preparation, including support seen only before the tail."
    )


def test_historical_only_saved_column_uses_bounded_probe_and_float32_padding(
    sqlite_store: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    step, days = _owner(sqlite_store, monkeypatch, tmp_path)
    prepared = _prepare(step, monkeypatch, _facts(days, (5, 110), ic_inst_concentration=[0.25, 0.5], ic_inst_holders=[0.5, np.nan]), {})
    full = step.build_panel()
    for source in prepared:
        source.reset_mock()
    projections = _projections(step, monkeypatch)
    tail = step.build_panel(PartWindow(days[-1], days[-8], days[-8]), existing_columns=list(full.columns))

    pd.testing.assert_frame_equal(_ordered(tail), _ordered(full.loc[full.date >= days[-8]]))
    assert tail.f_ic_inst_holders.isna().all() and tail.f_ic_inst_holders.dtype == np.dtype("float32")
    assert [source.call_count for source in prepared] == [1, 1]
    assert [projection.call_count for projection in projections] == [2, 1]
    _bounded(projections, days, 8)
    print(
        "SANITY: a historical-only saved feature stays float32/unknown in the tail after one eight-session backward witness, with no full dense projection."
    )


@pytest.mark.parametrize("denominator", ["globally_absent", "temporarily_unknown"])
def test_denominator_support_distinguishes_schema_removal_from_unknown_tail(
    sqlite_store: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, denominator: str
) -> None:
    step, days = _owner(sqlite_store, monkeypatch, tmp_path)
    broad, elite = _prepare(step, monkeypatch, _facts(days, (5,), ic_inst_concentration=[0.25]), {})
    if denominator == "temporarily_unknown":
        _shares(sqlite_store, days, (0, 112), (100.0, 0.0))
    current = step.build_panel()
    initial = _saved(current).copy()
    initial["f_ic_inst_ownership_pct"] = 0.2
    initial["f_ic_inst_value_to_mcap"] = 0.2
    sqlite_store.replace(Tables.cube_part_sec_thirteen, initial)
    broad.reset_mock()
    elite.reset_mock()
    projections = _projections(step, monkeypatch)

    step.run()
    result = _load(sqlite_store)
    derived = {"f_ic_inst_ownership_pct", "f_ic_inst_value_to_mcap"}
    if denominator == "globally_absent":
        assert derived.isdisjoint(result.columns)
        assert sum(len(call.args[1].trading_index) == len(days) for call in projections[0].call_args_list) == 1
        pd.testing.assert_frame_equal(result, _saved(current), check_dtype=False)
    else:
        assert derived <= set(result.columns)
        assert result.loc[result.date >= days[-8], list(derived)].isna().all().all()
        _bounded(projections, days, 8)
    assert broad.call_count == elite.call_count == 1
    print(f"SANITY: {denominator} has its own schema outcome; unavailable tail denominators never erase valid historical support.")


def test_observation_before_first_price_remains_supported_after_listing(sqlite_store: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    step, days = _owner(sqlite_store, monkeypatch, tmp_path)
    sqlite_store.replace(
        Tables.cube_part_prices,
        pd.concat(
            [
                pd.DataFrame({"date": days, "ticker": "OTHER", "close_split": 100.0, "level_factor": 1.0}),
                pd.DataFrame({"date": days[100:], "ticker": "AAA", "close_split": 100.0, "level_factor": 1.0}),
            ],
            ignore_index=True,
        ),
    )
    _prepare(step, monkeypatch, _facts(days, (5,), ic_inst_concentration=[0.25]), _elite(days, (5,), ic_super_holders=[0.5]))
    full = step.build_panel()
    tail = step.build_panel(PartWindow(days[-1], days[-8], days[-8]), existing_columns=list(full.columns))
    pd.testing.assert_frame_equal(_ordered(tail), _ordered(full.loc[full.date >= days[-8]]))
    listed = tail.loc[tail.ticker.eq("AAA")]
    assert len(listed) == 8 and listed.f_ic_inst_concentration.eq(0.25).all() and listed.f_ic_super_holders.eq(0.5).all()
    assert full.loc[full.ticker.eq("AAA"), "date"].min() == days[100]
    print("SANITY: old public observations carry into a ticker's first priced dates while unpriced keys stay outside the panel.")


def test_rejected_ownership_ratios_do_not_create_conditional_schema(sqlite_store: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    step, days = _owner(sqlite_store, monkeypatch, tmp_path)
    _shares(sqlite_store, days, (0,), (100.0,))
    broad, elite = _prepare(step, monkeypatch, _facts(days, (5,), ic_inst_concentration=[0.25], inst_shares=[300.0]), {})
    expected = _saved(step.build_panel())
    assert "f_ic_inst_ownership_pct" not in expected and "f_ic_inst_value_to_mcap" in expected
    sqlite_store.replace(Tables.cube_part_sec_thirteen, expected.assign(f_ic_inst_ownership_pct=0.5))
    broad.reset_mock()
    elite.reset_mock()
    projections = _projections(step, monkeypatch)
    step.run()
    pd.testing.assert_frame_equal(_load(sqlite_store), expected, check_dtype=False)
    assert broad.call_count == elite.call_count == 1
    assert sum(len(call.args[1].trading_index) == len(days) for call in projections[0].call_args_list) == 1
    assert all(len(call.args[1].trading_index) <= 8 or len(call.args[1].trading_index) == len(days) for call in projections[0].call_args_list)
    print(
        "SANITY: 300 reported shares / 100 outstanding exceeds the unchanged 2x ceiling everywhere; bounded probes find no valid support and the obsolete column is removed once."
    )


def test_valid_empty_preparation_is_not_a_source_failure(sqlite_store: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    step, days = _owner(sqlite_store, monkeypatch, tmp_path)
    prepared = _prepare(step, monkeypatch, pd.DataFrame(), {})
    panel = step.build_panel(PartWindow(days[-1], days[-8], days[-8]), existing_columns=["date", "ticker", "f_ic_inst_holders", "f_ic_super_holders"])
    assert set(panel.columns) == {"date", "ticker"} and panel.empty
    assert [source.call_count for source in prepared] == [1, 1]
    print("SANITY: verified empty source preparations produce an empty feature schema rather than a failed-source exception.")
