"""Physical bounds, known-empty results and rollback through the actual store facade."""

from __future__ import annotations

from typing import Any, cast

import pandas as pd
import pytest
from sqlalchemy import Connection, event
from sqlalchemy.exc import IntegrityError

from src.data_store.schema import Table, Tables
from src.data_store.store import DataStore
from src.utils.window import Window


def _rows() -> pd.DataFrame:
    dates = pd.to_datetime(["2026-11-06", "2026-11-09", "2026-11-13", "2026-11-16"])
    return pd.DataFrame({"date": dates.repeat(2), "ticker": ["AAPL", "MSFT"] * len(dates), "feature": [float(value) for value in range(8)]})


def _saved(store: DataStore, table: Table = Tables.cube_part_text) -> pd.DataFrame:
    df_rows = store.load(table)
    assert df_rows is not None
    return df_rows.sort_values(["date", "ticker"]).reset_index(drop=True)


def _reject_insert(_conn: Connection, _cursor: Any, statement: str, _parameters: Any, _context: Any, _many: bool) -> None:
    if statement.lstrip().upper().startswith("INSERT"):
        raise RuntimeError("controlled database insertion failure")


def test_bounded_replace_preserves_both_ends_and_other_keys(sqlite_store: DataStore) -> None:
    # Known storage truth: update one ticker's two authorized dates, including both endpoints.
    df_rows = _rows()
    sqlite_store.replace(Tables.cube_part_text, df_rows)
    window = Window(pd.Timestamp("2026-11-09"), pd.Timestamp("2026-11-13"))
    selected = df_rows.ticker.eq("AAPL") & window.mask(df_rows.date)
    df_write = df_rows.loc[selected].assign(feature=10.0)
    effect = sqlite_store.replace_window(Tables.cube_part_text, df_write, window, {"ticker": ["AAPL"]})
    df_expected = df_rows.copy()
    df_expected.loc[selected, "feature"] = 10.0
    pd.testing.assert_frame_equal(_saved(sqlite_store), df_expected, check_dtype=False)
    assert (effect.deleted, effect.inserted) == (2, 2)
    repeated = sqlite_store.replace_window(Tables.cube_part_text, df_write, window, {"ticker": ["AAPL"]})
    assert (repeated.deleted, repeated.inserted) == (2, 2)
    pd.testing.assert_frame_equal(_saved(sqlite_store), df_expected, check_dtype=False)
    print("SANITY: bounded replacement is inclusive and idempotent; prefix, finite-end suffix and all MSFT rows remain exact.")


def test_known_empty_clears_only_the_authorized_interval(sqlite_store: DataStore) -> None:
    df_rows = _rows()
    sqlite_store.replace(Tables.cube_part_text, df_rows)
    window = Window(pd.Timestamp("2026-11-09"), pd.Timestamp("2026-11-13"))
    effect = sqlite_store.replace_window(Tables.cube_part_text, df_rows.iloc[:0], window, {"ticker": ["AAPL"]})
    df_expected = df_rows.loc[~(df_rows.ticker.eq("AAPL") & window.mask(df_rows.date))].reset_index(drop=True)
    assert (effect.deleted, effect.inserted) == (2, 0)
    pd.testing.assert_frame_equal(_saved(sqlite_store), df_expected, check_dtype=False)
    print("SANITY: a successful empty correction clears exactly the authorized ticker/dates and preserves every other row.")


def test_failed_input_cannot_clear_saved_rows(sqlite_store: DataStore) -> None:
    df_rows = _rows()
    sqlite_store.replace(Tables.cube_part_text, df_rows)
    window = Window(pd.Timestamp("2026-11-09"), pd.Timestamp("2026-11-13"))
    with pytest.raises(TypeError, match="DataFrame"):
        sqlite_store.replace_window(Tables.cube_part_text, cast(pd.DataFrame, None), window, {"ticker": ["AAPL"]})
    pd.testing.assert_frame_equal(_saved(sqlite_store), df_rows, check_dtype=False)
    print("SANITY: unavailable input is refused before mutation; it cannot masquerade as a known-empty correction.")


@pytest.mark.parametrize("outside", ["date", "ticker"])
def test_outside_input_is_refused_before_deleting(sqlite_store: DataStore, outside: str) -> None:
    df_rows = _rows()
    sqlite_store.replace(Tables.cube_part_text, df_rows)
    window = Window(pd.Timestamp("2026-11-09"), pd.Timestamp("2026-11-13"))
    df_write = df_rows.loc[df_rows.ticker.eq("AAPL") & window.mask(df_rows.date)].copy()
    if outside == "date":
        df_write.loc[df_write.index[0], "date"] = pd.Timestamp("2026-11-16")
    else:
        df_write.loc[df_write.index[0], "ticker"] = "MSFT"
    with pytest.raises(ValueError, match="scope|Window|window"):
        sqlite_store.replace_window(Tables.cube_part_text, df_write, window, {"ticker": ["AAPL"]})
    pd.testing.assert_frame_equal(_saved(sqlite_store), df_rows, check_dtype=False)
    print(f"SANITY: out-of-{outside} insert rows cannot cause a partial destructive refresh.")


def test_insert_failure_rolls_back_the_actual_delete(sqlite_store: DataStore) -> None:
    df_rows = _rows()
    sqlite_store.replace(Tables.cube_part_text, df_rows)
    window = Window(pd.Timestamp("2026-11-09"), pd.Timestamp("2026-11-13"))
    df_write = df_rows.loc[df_rows.ticker.eq("AAPL") & window.mask(df_rows.date)].assign(feature=10.0)
    event.listen(sqlite_store.engine, "before_cursor_execute", _reject_insert)
    try:
        with pytest.raises(RuntimeError, match="controlled database insertion failure"):
            sqlite_store.replace_window(Tables.cube_part_text, df_write, window, {"ticker": ["AAPL"]})
    finally:
        event.remove(sqlite_store.engine, "before_cursor_execute", _reject_insert)
    pd.testing.assert_frame_equal(_saved(sqlite_store), df_rows, check_dtype=False)
    print("SANITY: an insertion error after the real SQL delete restores the complete original table.")


@pytest.mark.parametrize("table", [Tables.cube_part_text, Tables.cube], ids=lambda table: table.name)
def test_schema_rebuild_refuses_a_partial_retained_scope(sqlite_store: DataStore, table: Table) -> None:
    df_rows = _rows()
    sqlite_store.replace(table, df_rows)
    window = Window(pd.Timestamp("2026-11-09"), pd.Timestamp("2026-11-13"))
    df_write = df_rows.loc[df_rows.ticker.eq("AAPL") & window.mask(df_rows.date)].rename(columns={"feature": "new_feature"})
    with pytest.raises(ValueError, match="schema|Schema|columns"):
        sqlite_store.replace_window(table, df_write, window, {"ticker": ["AAPL"]}, schema_rebuild=True)
    pd.testing.assert_frame_equal(_saved(sqlite_store, table), df_rows, check_dtype=False)
    assert sqlite_store.columns(table) == list(df_rows.columns)
    print("SANITY: even explicit full mode cannot drop columns while retained dates or other tickers remain outside its scope.")


@pytest.mark.parametrize("table", [Tables.cube_part_text, Tables.cube], ids=lambda table: table.name)
def test_explicit_whole_retained_rebuild_can_replace_unmanaged_columns(sqlite_store: DataStore, table: Table) -> None:
    df_rows = _rows()
    sqlite_store.replace(table, df_rows)
    window = Window(pd.Timestamp("2026-11-06"), pd.Timestamp("2026-11-16"))
    df_write = df_rows.rename(columns={"feature": "new_feature"})
    effect = sqlite_store.replace_window(table, df_write, window, {"ticker": ["AAPL", "MSFT"]}, schema_rebuild=True)
    assert (effect.deleted, effect.inserted) == (8, 8)
    assert sqlite_store.columns(table) == list(df_write.columns)
    pd.testing.assert_frame_equal(_saved(sqlite_store, table), df_write, check_dtype=False)
    print("SANITY: an explicit full scope covering every retained row can atomically replace unmanaged feature columns.")


@pytest.mark.parametrize("table", [Tables.cube_part_text, Tables.cube], ids=lambda table: table.name)
def test_schema_failure_rolls_back_ddl_and_rows(sqlite_store: DataStore, table: Table) -> None:
    df_rows = _rows()
    sqlite_store.replace(table, df_rows)
    window = Window(df_rows.date.min(), df_rows.date.max())
    event.listen(sqlite_store.engine, "before_cursor_execute", _reject_insert)
    try:
        with pytest.raises(RuntimeError, match="controlled database insertion failure"):
            sqlite_store.replace_window(
                table, df_rows.rename(columns={"feature": "changed"}), window, {"ticker": ["AAPL", "MSFT"]}, schema_rebuild=True
            )
    finally:
        event.remove(sqlite_store.engine, "before_cursor_execute", _reject_insert)
    pd.testing.assert_frame_equal(_saved(sqlite_store, table), df_rows, check_dtype=False)
    assert sqlite_store.columns(table) == list(df_rows.columns)
    print("SANITY: failure after actual schema replacement restores original physical columns and every original row.")


def test_managed_source_schema_is_never_rebuilt_from_a_feature_frame(sqlite_store: DataStore) -> None:
    df_rows = _rows()
    sqlite_store.replace(Tables.prices, df_rows)
    with pytest.raises(ValueError, match="explicit retained-schema rebuild"):
        sqlite_store.replace_window(
            Tables.prices,
            df_rows.rename(columns={"feature": "changed"}),
            Window(df_rows.date.min(), df_rows.date.max()),
            {"ticker": ["AAPL", "MSFT"]},
            schema_rebuild=True,
        )
    pd.testing.assert_frame_equal(_saved(sqlite_store, Tables.prices), df_rows, check_dtype=False)
    print("SANITY: whole-scope rebuild authority for derived layouts cannot change a managed source schema.")


def test_same_primary_key_outside_window_cannot_be_upserted(sqlite_store: DataStore) -> None:
    table = Tables.earnings_call_sections
    df_rows = pd.DataFrame(
        {"ticker": ["AAPL"], "quarter": ["2026Q3"], "paragraph": [1], "as_of": pd.to_datetime(["2026-11-06"]), "text": ["protected source"]}
    )
    sqlite_store.replace(table, df_rows)
    before = sqlite_store.load(table)
    replacement = df_rows.assign(as_of=pd.Timestamp("2026-11-09"), text="new source")
    with pytest.raises(IntegrityError):
        sqlite_store.replace_window(table, replacement, Window(pd.Timestamp("2026-11-09"), pd.Timestamp("2026-11-13")), {"ticker": ["AAPL"]})
    actual = sqlite_store.load(table)
    pd.testing.assert_frame_equal(actual, before, check_exact=True)
    print("SANITY: a primary-key collision outside the authorized clock window fails atomically instead of overwriting the protected source.")
