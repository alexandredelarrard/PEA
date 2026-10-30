"""Generic transaction, scope and summary contracts used by source owners."""

from __future__ import annotations

import pandas as pd
import pytest
from sqlalchemy import event
from sqlalchemy.exc import IntegrityError

from src.data_store.schema import Tables
from src.data_store.store import DataStore


def _prices(store: DataStore) -> pd.DataFrame:
    df_prices = pd.DataFrame(
        {"ticker": ["A", "A", "B", "B"], "date": pd.to_datetime(["2024-01-01", "2024-01-02"] * 2), "close_split": [1.0, 2.0, 3.0, None]}
    )
    store.save(Tables.prices, df_prices)
    return df_prices


def test_exact_keys_and_flat_or_scopes(sqlite_store: DataStore) -> None:
    df_prices = _prices(sqlite_store)
    df_keys = df_prices.iloc[[0, 3]][list(Tables.prices.pk)]
    df_actual = sqlite_store.load(Tables.prices, where=df_keys, order_by=["ticker", "date"])
    assert df_actual is not None and df_actual.close_split.isna().tolist() == [False, True]
    assert df_actual.ticker.tolist() == ["A", "B"] and len(df_actual) == 2
    df_actual = sqlite_store.load(Tables.prices, where=[{"ticker": "A"}, df_keys], order_by=["ticker", "date"])
    assert df_actual is not None and len(df_actual) == 3
    assert sqlite_store.load(Tables.prices, where=[], optional=True) is None
    assert sqlite_store.load(Tables.prices, where=df_keys.iloc[:0], optional=True) is None
    with pytest.raises(ValueError, match="non-empty"):
        sqlite_store.load(Tables.prices, where=[{}])
    with pytest.raises(ValueError, match="primary key"):
        sqlite_store.load(Tables.prices, where=df_keys[["ticker"]])
    with pytest.raises(ValueError, match="primary key"):
        sqlite_store.load(Tables.prices, where=df_keys.assign(ticker=None))
    print("SANITY: complete tuple keys reject cross-pairs, OR scopes deduplicate, and empty scopes match nothing.")


def test_bound_changes_and_tail_replacement_rollback(sqlite_store: DataStore) -> None:
    df_prices = _prices(sqlite_store)
    with pytest.raises(RuntimeError, match="injected"):
        with sqlite_store.transaction() as store:
            assert store.engine is sqlite_store.engine
            store.update(Tables.prices, {"close_split": 10.0}, {"ticker": "A"})
            store.delete(Tables.prices, df_prices.iloc[[3]][list(Tables.prices.pk)])
            store.insert(Tables.prices, df_prices.iloc[[0]].assign(ticker="C"))
            assert store.row_count(Tables.prices) == 4
            raise RuntimeError("injected after all write types")
    df_actual = sqlite_store.load(Tables.prices, order_by=["ticker", "date"])
    assert df_actual is not None and df_actual.ticker.tolist() == ["A", "A", "B", "B"]
    assert df_actual.close_split.iloc[0] == 1.0 and df_actual.close_split.isna().sum() == 1
    with pytest.raises(IntegrityError):
        with sqlite_store.transaction() as store:
            store.append_tail(Tables.prices, df_prices.iloc[[0]], pd.Timestamp("2024-01-02"), inclusive=True)
    assert sqlite_store.row_count(Tables.prices) == 4
    print("SANITY: update/delete/strict insert share one transaction; failed tail insertion restores the deleted rows.")


def test_bound_save_preserves_fields_and_explicit_null(sqlite_store: DataStore) -> None:
    df_prices = _prices(sqlite_store)
    with sqlite_store.transaction() as store:
        assert store.save(Tables.prices, df_prices.iloc[[0]][list(Tables.prices.pk)], ignore_all_conflicts=True) == 1
        store.save(Tables.prices, df_prices.iloc[[1]].assign(close_split=None))
        assert store.update(Tables.prices, {"close_split": 99.0}, []) == 0
        assert store.delete(Tables.prices, df_prices.iloc[:0][list(Tables.prices.pk)]) == 0
    df_actual = sqlite_store.load(Tables.prices, order_by=["ticker", "date"])
    assert df_actual is not None and df_actual.close_split.iloc[0] == 1.0 and pd.isna(df_actual.close_split.iloc[1])
    for scope in [None, {}]:
        with pytest.raises(ValueError):
            sqlite_store.update(Tables.prices, {"close_split": None}, scope)  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            sqlite_store.delete(Tables.prices, scope)  # type: ignore[arg-type]
    print("SANITY: bound upserts keep omitted values, explicit NULL overwrites, and unscoped mutations are refused.")


def test_filtered_summary_and_complete_cohort(sqlite_store: DataStore) -> None:
    df_prices = _prices(sqlite_store)
    df_summary = sqlite_store.summarize(
        Tables.prices,
        "ticker",
        {"n": ("count", None), "missing": ("count", None), "first": ("min", "date")},
        count_where={"missing": {"close_split": None}},
    ).set_index("ticker")
    assert df_summary["n"].to_dict() == {"A": 2, "B": 2}
    assert df_summary["missing"].to_dict() == {"A": 0, "B": 1}
    df_cohort = sqlite_store.load_cohort(Tables.prices, "ticker", df_prices.iloc[[0]][list(Tables.prices.pk)], columns=["ticker", "date"])
    assert len(df_cohort) == 2 and df_cohort.ticker.eq("A").all()
    print("SANITY: filtered counts stay server-side and one touched member selects its complete group.")


def test_sql_date_null_comparison_and_explicit_staging(sqlite_store: DataStore) -> None:
    df_prices = _prices(sqlite_store)
    df_actual = sqlite_store.load(Tables.prices, order_by=["ticker", "date"])
    assert df_actual is not None
    assert sqlite_store.same_rows(Tables.prices, df_actual, df_prices)
    assert not sqlite_store.same_rows(Tables.prices, df_actual, df_prices.assign(date=df_prices.date + pd.Timedelta(hours=12)))
    assert not sqlite_store.same_rows(Tables.prices, df_actual, df_prices.assign(close_split=0.0))
    with DataStore.temporary([Tables.insider_transactions, Tables.insider_footnotes, Tables.sec_fails_to_deliver]) as store:
        assert "inventory_checked_through" in store.columns(Tables.insider_transactions)
        assert not store.exists(Tables.prices)
        df_fails = df_prices.rename(columns={"close_split": "fails_quantity"})
        store.save(Tables.sec_fails_to_deliver, df_fails)
        df_stored = store.load(Tables.sec_fails_to_deliver, columns=list(df_fails.columns))
        assert df_stored is not None
        assert store.same_rows(Tables.sec_fails_to_deliver, df_stored, df_fails.assign(date=df_fails.date + pd.Timedelta(hours=12)))
    print("SANITY: equality follows SQL DATE/NULL binding and staging creates only explicitly selected full schemas.")


def test_generic_metadata_reuse_and_explicit_schema_invalidation(sqlite_store: DataStore) -> None:
    _prices(sqlite_store)
    statements = []

    def record(conn, cursor, statement, parameters, context, executemany):
        if statement.startswith("PRAGMA"):
            statements.append(statement)

    event.listen(sqlite_store.engine, "before_cursor_execute", record)
    sqlite_store.load(Tables.prices)
    cold = len(statements)
    with sqlite_store.transaction() as bound:
        bound.load(Tables.prices)
        bound.insert(Tables.prices, pd.DataFrame({"ticker": ["C"], "date": [pd.Timestamp("2024-01-01")], "close_split": [1.0]}))
    assert len(statements) == cold, "ordinary and bound calls must reuse the same generic schema metadata"
    sqlite_store.ensure_columns(Tables.prices, pd.DataFrame({"cache_note": ["explicit"]}))
    assert "cache_note" in sqlite_store.columns(Tables.prices)
    sqlite_store.drop(Tables.prices)
    assert not sqlite_store.exists(Tables.prices)
    _prices(sqlite_store)
    assert "cache_note" not in sqlite_store.columns(Tables.prices)
    print("SANITY: generic metadata is shared with transactions and refreshed by explicit alter/drop/recreate.")
