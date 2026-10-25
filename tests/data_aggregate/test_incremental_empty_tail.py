"""Persist a successful empty refresh without retaining stale feature rows."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd

from src.data_aggregate.utils.common.incremental import COLUMNS_CHANGED, PartWindow, write_part
from src.data_store.schema import Tables
from src.data_store.store import DataStore


def test_all_null_refresh_clears_inclusive_tail_and_preserves_prefix(sqlite_store: DataStore) -> None:
    dates = pd.date_range("2025-01-06", periods=3)
    rows = pd.DataFrame({"date": dates, "ticker": "AAPL", "feature": [1.0, 2.0, 3.0]})
    sqlite_store.replace(Tables.cube_part_sec_thirteen, rows)
    saved = sqlite_store.load(Tables.cube_part_sec_thirteen)
    refresh = rows.iloc[1:].assign(feature=np.nan)
    window = PartWindow(dates[-1], dates[0], dates[1])

    assert write_part(sqlite_store, Tables.cube_part_sec_thirteen, refresh, window, drop_empty=True) == 0
    assert write_part(sqlite_store, Tables.cube_part_sec_thirteen, refresh, window, drop_empty=True) == 0

    actual = sqlite_store.load(Tables.cube_part_sec_thirteen)
    assert saved is not None and actual is not None
    pd.testing.assert_frame_equal(actual, saved.iloc[:1].reset_index(drop=True), check_exact=True)
    print("Sanity: two all-null refreshes cleared the inclusive tail and kept the stored prefix exactly.")


def test_full_all_null_and_genuinely_empty_input_do_not_write(sqlite_store: DataStore) -> None:
    dates = pd.date_range("2025-01-06", periods=3)
    rows = pd.DataFrame({"date": dates, "ticker": "AAPL", "feature": [1.0, 2.0, 3.0]})
    sqlite_store.replace(Tables.cube_part_sec_thirteen, rows)
    saved = sqlite_store.load(Tables.cube_part_sec_thirteen)
    window = PartWindow(dates[-1], dates[0], dates[1])

    with patch.object(sqlite_store, "replace", wraps=sqlite_store.replace) as replace:
        with patch.object(sqlite_store, "append_tail", wraps=sqlite_store.append_tail) as append:
            assert write_part(sqlite_store, Tables.cube_part_sec_thirteen, rows.assign(feature=np.nan), PartWindow(None, None), drop_empty=True) == 0
            assert write_part(sqlite_store, Tables.cube_part_sec_thirteen, rows.iloc[:0], window, drop_empty=True) == 0
            replace.assert_not_called()
            append.assert_not_called()

    actual = sqlite_store.load(Tables.cube_part_sec_thirteen)
    assert saved is not None and actual is not None
    pd.testing.assert_frame_equal(actual, saved, check_exact=True)
    print("Sanity: full all-null output and genuinely empty input left the stored part untouched.")


def test_write_part_reuses_supplied_column_metadata(sqlite_store: DataStore) -> None:
    dates = pd.date_range("2025-01-06", periods=3)
    rows = pd.DataFrame({"date": dates, "ticker": "AAPL", "feature": [1.0, 2.0, 3.0]})
    sqlite_store.replace(Tables.cube_part_sec_thirteen, rows)
    columns = sqlite_store.columns(Tables.cube_part_sec_thirteen)
    window = PartWindow(dates[-1], dates[0], dates[1])

    with patch.object(sqlite_store, "columns", side_effect=AssertionError("saved columns were already loaded")):
        assert write_part(sqlite_store, Tables.cube_part_sec_thirteen, rows, window, existing_columns=columns) == 2
        assert write_part(sqlite_store, Tables.cube_part_sec_thirteen, rows, window, existing_columns=[]) == 2
        assert (
            write_part(sqlite_store, Tables.cube_part_sec_thirteen, rows, window, existing_columns=["date", "ticker", "old_feature"])
            == COLUMNS_CHANGED
        )

    with patch.object(sqlite_store, "columns", wraps=sqlite_store.columns) as lookup:
        assert write_part(sqlite_store, Tables.cube_part_sec_thirteen, rows, window) == 2
        lookup.assert_called_once_with(Tables.cube_part_sec_thirteen)
    print("Sanity: provided column metadata avoided another read; absent metadata was read once and drift still required a full build.")
