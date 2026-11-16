"""Persist a successful empty refresh without retaining stale feature rows."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.data_aggregate.utils.common.incremental import write_part
from src.data_store.schema import Tables
from src.data_store.store import DataStore
from src.utils.window import Window, WindowPlan, WindowRun


def _window(dates: pd.DatetimeIndex, *, full: bool = False) -> WindowPlan:
    requested = Window(dates[0], dates[-1])
    write = requested if full else Window(dates[1], dates[-1])
    return WindowPlan(
        requested, requested, write, WindowRun("full" if full else "incremental", "known-truth replacement"), dates[-1], (("ticker", ("AAPL",)),)
    )


def test_all_null_refresh_clears_inclusive_tail_and_preserves_prefix(sqlite_store: DataStore) -> None:
    dates = pd.date_range("2025-01-06", periods=3)
    rows = pd.DataFrame({"date": dates, "ticker": "AAPL", "feature": [1.0, 2.0, 3.0]})
    sqlite_store.replace(Tables.cube_part_sec_thirteen, rows)
    saved = sqlite_store.load(Tables.cube_part_sec_thirteen)
    refresh = rows.iloc[1:].assign(feature=np.nan)
    window = _window(dates)

    assert write_part(sqlite_store, Tables.cube_part_sec_thirteen, refresh, window, drop_empty=True) == 0
    assert write_part(sqlite_store, Tables.cube_part_sec_thirteen, refresh, window, drop_empty=True) == 0

    actual = sqlite_store.load(Tables.cube_part_sec_thirteen)
    assert saved is not None and actual is not None
    pd.testing.assert_frame_equal(actual, saved.iloc[:1].reset_index(drop=True), check_exact=True)
    print("Sanity: two all-null refreshes cleared the inclusive tail and kept the stored prefix exactly.")


def test_unavailable_input_preserves_rows_but_valid_empty_clears_scope(sqlite_store: DataStore) -> None:
    dates = pd.date_range("2025-01-06", periods=3)
    rows = pd.DataFrame({"date": dates, "ticker": "AAPL", "feature": [1.0, 2.0, 3.0]})
    sqlite_store.replace(Tables.cube_part_sec_thirteen, rows)
    saved = sqlite_store.load(Tables.cube_part_sec_thirteen)
    window = _window(dates)

    with patch.object(sqlite_store, "replace", wraps=sqlite_store.replace) as replace:
        with patch.object(sqlite_store, "append_tail", wraps=sqlite_store.append_tail) as append:
            assert write_part(sqlite_store, Tables.cube_part_sec_thirteen, None, window, drop_empty=True) == 0
            unchanged = sqlite_store.load(Tables.cube_part_sec_thirteen)
            assert saved is not None and unchanged is not None
            pd.testing.assert_frame_equal(saved, unchanged)
            assert write_part(sqlite_store, Tables.cube_part_sec_thirteen, rows.iloc[:0], window, drop_empty=True) == 0
            replace.assert_not_called()
            append.assert_not_called()

    actual = sqlite_store.load(Tables.cube_part_sec_thirteen)
    assert saved is not None and actual is not None
    pd.testing.assert_frame_equal(actual, saved.iloc[:1].reset_index(drop=True), check_exact=True)
    assert write_part(sqlite_store, Tables.cube_part_sec_thirteen, rows.assign(feature=np.nan), _window(dates, full=True), drop_empty=True) == 0
    assert sqlite_store.row_count(Tables.cube_part_sec_thirteen) == 0
    print("Sanity: unavailable input preserved every row; valid empty and all-null corrections cleared only their authorized scope.")


def test_write_part_reuses_supplied_column_metadata(sqlite_store: DataStore) -> None:
    dates = pd.date_range("2025-01-06", periods=3)
    rows = pd.DataFrame({"date": dates, "ticker": "AAPL", "feature": [1.0, 2.0, 3.0]})
    sqlite_store.replace(Tables.cube_part_sec_thirteen, rows)
    columns = sqlite_store.columns(Tables.cube_part_sec_thirteen)
    window = _window(dates)

    with patch.object(sqlite_store, "columns", side_effect=AssertionError("saved columns were already loaded")):
        assert write_part(sqlite_store, Tables.cube_part_sec_thirteen, rows, window, existing_columns=columns) == 2
        assert write_part(sqlite_store, Tables.cube_part_sec_thirteen, rows, window, existing_columns=[]) == 2
        with pytest.raises(ValueError, match="explicit retained-schema rebuild"):
            write_part(sqlite_store, Tables.cube_part_sec_thirteen, rows, window, existing_columns=["date", "ticker", "old_feature"])

    with patch.object(sqlite_store, "columns", wraps=sqlite_store.columns) as lookup:
        assert write_part(sqlite_store, Tables.cube_part_sec_thirteen, rows, window) == 2
        lookup.assert_not_called()
    print("Sanity: provided column metadata avoided another read; absent metadata was read once and drift still required a full build.")
