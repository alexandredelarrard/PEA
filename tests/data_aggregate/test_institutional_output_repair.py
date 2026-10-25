"""Exact output persistence controls; these fixtures do not model feature economics."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import event

from src.data_aggregate.transformers import step_cube_institutionals as institutional_step
from src.data_aggregate.transformers.step_cube_institutionals import StepCubeInstitutionals
from src.data_store.schema import Table, Tables
from src.data_store.store import DataStore
from tests.data_aggregate.test_institutional_step_contract import context


def _panel() -> pd.DataFrame:
    keys = pd.MultiIndex.from_product([pd.bdate_range("2024-01-02", periods=40), ["AAA", "BBB"]], names=["date", "ticker"])
    fields = [*institutional_step.INSIDER_EMISSION, *institutional_step.short_flow.EMISSION, *institutional_step.OWNERSHIP_EMISSION]
    rows = keys.to_frame(index=False)
    for offset, field in enumerate(fields):
        rows[f"f_{field}"] = np.arange(len(rows), dtype="float64") + offset + 1
    rows.loc[::5, f"f_{fields[-1]}"] = np.nan
    return rows


def _step(store: DataStore, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, panel: pd.DataFrame) -> tuple[StepCubeInstitutionals, Mock]:
    store.replace(
        Tables.cube_part_prices,
        _panel()[["date", "ticker"]].assign(close_split=100.0, close_total=100.0, volume=1000.0, level_factor=1.0),
    )
    ctx = context(store, tmp_path)
    step = StepCubeInstitutionals(ctx, ctx.config)
    build = Mock(return_value=panel.copy())
    monkeypatch.setattr(step, "build_panel", build)
    return step, build


def _assert_saved(store: DataStore, expected: pd.DataFrame) -> None:
    actual = store.load(Tables.cube_part_institutionals, optional=True)
    fields = [column for column in expected if column not in ("date", "ticker")]
    expected = expected.loc[expected[fields].notna().any(axis=1)] if fields else expected.iloc[:0]
    assert set(store.columns(Tables.cube_part_institutionals)) == set(expected.columns)
    if expected.empty:
        assert actual is None
    else:
        assert actual is not None
        pd.testing.assert_frame_equal(
            actual.sort_values(["date", "ticker"]).reset_index(drop=True)[expected.columns],
            expected.sort_values(["date", "ticker"]).reset_index(drop=True),
            check_exact=True,
            check_dtype=False,
        )


@pytest.mark.parametrize("change", ["float64", "null", "missing", "stale", "equal-count-key", "all-null-row", "empty", "schema", "unchanged"])
def test_run_repairs_exact_historical_output_once(sqlite_store: DataStore, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, change: str) -> None:
    expected = _panel()
    stored = expected.copy()
    field = str(expected.columns[2])
    if change == "float64":
        value = stored[field].to_numpy(dtype="float64")[4]
        next_value = np.nextafter(value, np.inf)
        expected.loc[4, field] = next_value
        assert np.float32(next_value) == np.float32(value)
    elif change == "null":
        expected.loc[4, field] = np.nan
    elif change == "missing":
        stored = stored.drop(index=4)
    elif change == "stale":
        stored = pd.concat([stored, stored.iloc[[4]].assign(ticker="STALE")], ignore_index=True)
    elif change == "equal-count-key":
        stored.loc[4, "ticker"] = "STALE"
        assert len(stored) == len(expected)
    elif change == "all-null-row":
        expected.loc[4, expected.columns[2:]] = np.nan
    elif change == "empty":
        expected.loc[:, expected.columns[2:]] = np.nan
    elif change == "schema":
        stored = stored.drop(columns=[str(stored.columns[-1])]).assign(f_retired=99.0)
    sqlite_store.replace(Tables.cube_part_institutionals, stored)
    step, build = _step(sqlite_store, monkeypatch, tmp_path, expected)
    stream = sqlite_store.iter_load
    calls: list[dict[str, Any]] = []
    closed: list[bool] = []

    def capture(table: Table | str, **kwargs: Any) -> Iterator[pd.DataFrame]:
        if table != Tables.cube_part_institutionals:
            yield from stream(table, **kwargs)
            return
        calls.append(kwargs.copy())
        try:
            yield from stream(table, **{**kwargs, "chunksize": 7})
        finally:
            closed.append(True)

    monkeypatch.setattr(sqlite_store, "iter_load", capture)
    step.run()
    _assert_saved(sqlite_store, expected)
    build.assert_called_once()
    assert len(calls) == 1 and closed == [True]
    assert calls[0].get("columns") and set(calls[0]["columns"]).issubset(stored.columns)
    assert list(calls[0]["order_by"]) == ["date", "ticker"]
    build.reset_mock()
    calls.clear()
    closed.clear()
    step.run()
    _assert_saved(sqlite_store, expected)
    build.assert_called_once()
    assert len(calls) == 1 and closed == [True]
    print(f"SANITY: {change} repairs exact old keys/float64 values/NULLs/schema with one build and one closed stored stream; retry is identical.")


@pytest.mark.parametrize(
    "invalid", ["candidate-empty", "candidate-duplicate", "candidate-null-key", "stored-duplicate", "stored-null-key", "stored-chunk-schema"]
)
def test_invalid_output_keys_or_stream_refuse_before_writing(
    sqlite_store: DataStore, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, invalid: str
) -> None:
    original = _panel()
    sqlite_store.replace(Tables.cube_part_institutionals, original)
    before = sqlite_store.load(Tables.cube_part_institutionals)
    assert before is not None
    candidate = original.copy()
    if invalid == "candidate-empty":
        candidate = candidate.iloc[:0]
    elif invalid == "candidate-duplicate":
        candidate = pd.concat([candidate, candidate.iloc[[0]]], ignore_index=True)
    elif invalid == "candidate-null-key":
        candidate.loc[0, "ticker"] = None
    step, _ = _step(sqlite_store, monkeypatch, tmp_path, candidate)
    stream = sqlite_store.iter_load

    def corrupt(table: Table | str, **kwargs: Any) -> Iterator[pd.DataFrame]:
        if table != Tables.cube_part_institutionals or not invalid.startswith("stored-"):
            yield from stream(table, **kwargs)
            return
        for index, chunk in enumerate(stream(table, **{**kwargs, "chunksize": 1})):
            if index == 0 and invalid == "stored-null-key":
                chunk = chunk.assign(ticker=None)
            elif index == 1 and invalid == "stored-chunk-schema":
                chunk = chunk.drop(columns=[str(chunk.columns[-1])])
            yield chunk
            if index == 0 and invalid == "stored-duplicate":
                yield chunk.copy()

    monkeypatch.setattr(sqlite_store, "iter_load", corrupt)
    with pytest.raises(ValueError):
        step.run()
    actual = sqlite_store.load(Tables.cube_part_institutionals)
    assert actual is not None
    pd.testing.assert_frame_equal(actual, before, check_exact=True)
    print(f"SANITY: {invalid} is refused before changing any saved key, value, NULL or column.")


@pytest.mark.parametrize("failure", ["after-delete", "after-drop", "after-create", "after-insert"])
def test_output_repair_failure_rolls_back_schema_rows_and_retries(
    sqlite_store: DataStore, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failure: str
) -> None:
    expected = _panel()
    stored = expected.copy()
    if failure in ("after-drop", "after-create"):
        stored = stored.assign(f_retired=99.0)
    else:
        stored.loc[4, str(stored.columns[2])] = -99.0
    sqlite_store.replace(Tables.cube_part_institutionals, stored)
    before = sqlite_store.load(Tables.cube_part_institutionals)
    assert before is not None
    schema = sqlite_store.columns(Tables.cube_part_institutionals)
    other = pd.DataFrame({"date": [expected["date"].iloc[0]], "ticker": ["OTHER"], "f_unrelated": [123.0]})
    sqlite_store.replace(Tables.cube_part_sec_thirteen, other)
    step, _ = _step(sqlite_store, monkeypatch, tmp_path, expected)
    reached: list[str] = []
    verbs = {"after-delete": "DELETE FROM", "after-drop": "DROP TABLE", "after-create": "CREATE TABLE", "after-insert": "INSERT INTO"}

    def fail_after_statement(conn: Any, cursor: Any, statement: str, parameters: Any, execution_context: Any, executemany: bool) -> None:
        normalized = " ".join(statement.replace('"', "").split())
        if normalized.startswith(verbs[failure]) and Tables.cube_part_institutionals.name in normalized:
            reached.append(failure)
            raise RuntimeError("controlled output repair failure")

    event.listen(sqlite_store.engine, "after_cursor_execute", fail_after_statement)
    try:
        with pytest.raises(RuntimeError, match="controlled output repair"):
            step.run()
    finally:
        event.remove(sqlite_store.engine, "after_cursor_execute", fail_after_statement)
    assert reached == [failure]
    assert sqlite_store.columns(Tables.cube_part_institutionals) == schema
    actual = sqlite_store.load(Tables.cube_part_institutionals)
    assert actual is not None
    pd.testing.assert_frame_equal(actual, before, check_exact=True)
    unrelated = sqlite_store.load(Tables.cube_part_sec_thirteen)
    assert unrelated is not None
    pd.testing.assert_frame_equal(unrelated, other, check_exact=True)
    step.run()
    _assert_saved(sqlite_store, expected)
    step.run()
    _assert_saved(sqlite_store, expected)
    print(
        f"SANITY: {failure} rolls back actual {sqlite_store.engine.dialect.name} DML/DDL, keys, values and NULLs; another part survives and two retries match the full panel."
    )
