"""Exact output persistence controls; these fixtures do not model feature economics."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import event

from src.data_aggregate.transformers import step_cube_institutionals as institutional_step
from src.data_aggregate.transformers.step_cube_institutionals import StepCubeInstitutionals
from src.data_store.schema import Tables
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


def test_output_writer_keeps_package_boundary_and_one_transaction(sqlite_store: DataStore, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    module = ast.parse(Path(institutional_step.__file__).read_text(encoding="utf-8"))
    imports = [node.module for node in ast.walk(module) if isinstance(node, ast.ImportFrom) and node.module]
    assert not any(name.startswith("src.data_extract") for name in imports), imports
    step, _ = _step(sqlite_store, monkeypatch, tmp_path, _panel())
    transaction = Mock(wraps=sqlite_store.transaction)
    monkeypatch.setattr(sqlite_store, "transaction", transaction)
    step.run()
    transaction.assert_called_once_with()
    print("SANITY: the aggregate imports no extractor owner and writes inside one store transaction.")


@pytest.mark.parametrize("change", ["float64", "null", "missing", "stale", "all-null-row", "schema", "old-history", "unchanged"])
def test_run_rewrites_the_refresh_tail_and_keeps_old_history(
    sqlite_store: DataStore, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, change: str
) -> None:
    """D-13: a consumer rewrites only its refresh tail (or the whole part on a schema change); saved old rows stay."""
    expected = _panel()
    row = len(expected) - 1 if change != "old-history" else 4
    field = str(expected.columns[2])
    stored = expected.copy()
    if change == "float64":
        expected.loc[row, field] = np.nextafter(stored[field].to_numpy(dtype="float64")[row], np.inf)
    elif change in ("null", "old-history"):
        expected.loc[row, field] = np.nan
    elif change == "missing":
        stored = stored.drop(index=row)
    elif change == "stale":
        stored = pd.concat([stored, stored.iloc[[row]].assign(ticker="STALE")], ignore_index=True)
    elif change == "all-null-row":
        expected.loc[row, expected.columns[2:]] = np.nan
    elif change == "schema":
        stored = stored.drop(columns=[str(stored.columns[-1])]).assign(f_retired=99.0)
    sqlite_store.replace(Tables.cube_part_institutionals, stored)
    step, build = _step(sqlite_store, monkeypatch, tmp_path, expected)
    step.run()
    want = stored if change == "old-history" else expected
    _assert_saved(sqlite_store, want)
    build.assert_called_once()
    step.run()
    _assert_saved(sqlite_store, want)
    print(f"SANITY: {change}: the tail (or schema) is rewritten from one build, an old-history change is kept as saved; a retry is identical.")


@pytest.mark.parametrize("invalid", ["candidate-duplicate", "candidate-null-key"])
def test_invalid_output_keys_or_stream_refuse_before_writing(
    sqlite_store: DataStore, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, invalid: str
) -> None:
    original = _panel()
    sqlite_store.replace(Tables.cube_part_institutionals, original)
    before = sqlite_store.load(Tables.cube_part_institutionals)
    assert before is not None
    candidate = original.copy()
    if invalid == "candidate-duplicate":
        candidate = pd.concat([candidate, candidate.iloc[[0]]], ignore_index=True)
    elif invalid == "candidate-null-key":
        candidate.loc[0, "ticker"] = None
    step, _ = _step(sqlite_store, monkeypatch, tmp_path, candidate)
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
        stored.loc[len(stored) - 1, str(stored.columns[2])] = -99.0
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
