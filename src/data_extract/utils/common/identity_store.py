"""Commit identity changes and invalidate dependent source evidence atomically."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import pandas as pd

from src.data_store.schema import Table, Tables
from src.data_store.store import DataStore, Where

SOURCE_LOCK = "institutional-source-proof"


def load_source_identity(store: DataStore) -> dict[Table, pd.DataFrame]:
    """Load the closed four-table identity projection for source classification and guarded writes."""
    cohort: dict[Table, pd.DataFrame] = {}
    for table in (Tables.entity_lineage, Tables.symbol_tenure, Tables.sp500_tickers, Tables.security_master):
        df_rows = (
            store.load(table, columns=["ticker", "cik"], optional=True)
            if table == Tables.sp500_tickers
            else store.load(table, project=True, optional=True)
        )
        cohort[table] = df_rows if df_rows is not None else pd.DataFrame()
    return cohort


def assert_source_identity(store: DataStore, actual: dict[Table, pd.DataFrame], expected: dict[Table, pd.DataFrame]) -> None:
    """Refuse classification based on a changed or incomplete identity cohort."""
    if set(expected) != set(actual):
        raise ValueError("Expected identity must contain the closed institutional source cohort")
    for table, df_expected in expected.items():
        if not store.same_rows(table, actual[table], df_expected):
            raise ValueError(f"Institutional identity cohort changed before commit: {table}")


def invalidate_source_proof(store: DataStore, tickers: set[str], *, identity_change: bool = False) -> None:
    """Clear retained proof and checked quiet quantities in the caller's active transaction."""
    if not tickers:
        return
    for table, stamp in (
        (Tables.insider_transactions, "inventory_checked_through"),
        (Tables.short_interest, "source_checked_at"),
        (Tables.sec_fails_to_deliver, "source_checked_at"),
    ):
        if not identity_change and table != Tables.insider_transactions:
            continue
        columns = set(store.columns(table))
        if stamp not in columns:
            continue
        scope = {"ticker": sorted(tickers)}
        if table == Tables.short_interest and {"short_volume", "total_volume"} <= columns:
            store.update(table, {"short_volume": None}, {**scope, stamp: store.NOT_NULL, "short_volume": 0, "total_volume": None})
        elif table == Tables.sec_fails_to_deliver and "fails_quantity" in columns:
            values = {column: None for column in ("fails_quantity", "fails_value") if column in columns}
            store.update(table, values, {**scope, stamp: store.NOT_NULL, "fails_quantity": 0})
        store.update(table, {stamp: None}, scope)


def _identity_values(df_rows: pd.DataFrame, columns: Sequence[str], *, cik: bool = False) -> set[str]:
    """Non-null identity strings used to scope retained-owner discovery."""
    values = {str(value) for column in columns if column in df_rows for value in df_rows[column].dropna()}
    return {value.zfill(10) for value in values} if cik else values


def _retained_owners(store: DataStore, df_rows: pd.DataFrame) -> pd.DataFrame:
    """Read compact retained owner tuples before identity comparisons and writes."""
    table = Tables.insider_transactions
    columns = store.columns(table)
    ciks = _identity_values(df_rows, ("cik", "issuer_cik"), cik=True)
    symbols = _identity_values(df_rows, ("symbol", "source_symbol", "market_symbol", "ticker"))
    scope: Where = [{column: sorted(values)} for column, values in (("issuer_cik", ciks), ("source_symbol", symbols)) if column in columns and values]
    if "ticker" not in columns or not scope:
        return pd.DataFrame()
    groups = [column for column in ("ticker", "issuer_cik", "source_symbol") if column in columns]
    return store.summarize(table, groups, {"n": ("count", None)}, where=scope)


def _affected_tickers(df_changed: pd.DataFrame, cohorts: dict[Table, pd.DataFrame], df_retained: pd.DataFrame) -> set[str]:
    """Resolve old and new identity-bearing CIKs/symbols to current and retained proof owners."""
    ciks = _identity_values(df_changed, ("cik", "issuer_cik"), cik=True)
    symbols = _identity_values(df_changed, ("symbol", "source_symbol", "market_symbol", "ticker"))
    tickers = _identity_values(df_changed, ("canonical_ticker", "canonical_company", "ticker"))
    for df_rows in cohorts.values():
        matched = pd.Series(False, index=df_rows.index)
        for column in ("cik", "issuer_cik"):
            if column in df_rows:
                matched |= df_rows[column].astype("string").str.zfill(10).isin(ciks)
        for column in ("symbol", "source_symbol", "market_symbol", "ticker"):
            if column in df_rows:
                matched |= df_rows[column].isin(symbols)
        tickers.update(_identity_values(df_rows.loc[matched], ("ticker", "canonical_ticker", "canonical_company")))
    if not df_retained.empty:
        matched = pd.Series(False, index=df_retained.index)
        if "issuer_cik" in df_retained:
            matched |= df_retained["issuer_cik"].isin(ciks)
        if "source_symbol" in df_retained:
            matched |= df_retained["source_symbol"].isin(symbols)
        tickers.update(df_retained.loc[matched, "ticker"].dropna().astype(str))
    return tickers


def _scope_rows(df_rows: pd.DataFrame, where: dict[str, object] | None) -> pd.DataFrame:
    """Select the already-loaded identity partition without another table read."""
    selected = pd.Series(True, index=df_rows.index)
    for column, value in (where or {}).items():
        if column not in df_rows:
            return df_rows.iloc[:0]
        if isinstance(value, list | tuple | set | frozenset | pd.Series | pd.Index):
            selected &= df_rows[column].isin(value)
        else:
            selected &= df_rows[column].isna() if value is None else df_rows[column].eq(cast(Any, value))
    return df_rows.loc[selected]


def _identity_columns(table: Table, available: set[str]) -> list[str]:
    """Semantic identity fields and owner stamps, excluding observation/evidence churn."""
    columns = [*table.pk, "valid_to"]
    if table == Tables.sp500_tickers:
        columns = ["ticker", "cik"]
    elif table == Tables.entity_lineage:
        columns += ["entity_id", "canonical_ticker", "cik", "scope_changed_at"]
    elif table == Tables.security_master:
        columns += [
            "canonical_company",
            "issuer_cik",
            "market_symbol",
            "cusip",
            "lineage_role",
            "security_class",
            "conversion_ratio",
            "scope_changed_at",
        ]
    return [column for column in dict.fromkeys(columns) if column in available]


def save_identity(store: DataStore, table: Table, df_rows: pd.DataFrame, where: dict[str, object] | None = None) -> int:
    """Save one roster/identity scope with physical dependent-proof invalidation."""
    if table not in (Tables.sp500_tickers, Tables.entity_lineage, Tables.security_master, Tables.symbol_tenure):
        raise ValueError("Only the roster, lineage, master and tenure writers use the identity commit")
    if where == {} or table != Tables.symbol_tenure and where is not None:
        raise ValueError("Only a symbol-tenure partition can carry a replacement scope")
    if table != Tables.sp500_tickers and (not set(table.pk).issubset(df_rows) or df_rows.duplicated(list(table.pk)).any()):
        raise ValueError("Identity replacement needs unique declared keys")

    # Load the closed cohort and compact retained owners under the shared guard.
    with store.transaction(SOURCE_LOCK) as bound:
        cohorts = load_source_identity(bound)
        scope: dict[str, object] | None = {"ticker": df_rows["ticker"].tolist()} if table == Tables.sp500_tickers else where
        df_old = _scope_rows(cohorts[table], scope)
        df_retained = _retained_owners(bound, pd.concat([df_old, df_rows], ignore_index=True))

        # Compare semantic fields using canonical date precision, then invalidate and save.
        columns = _identity_columns(table, set(df_old.columns) | set(df_rows.columns))
        df_compared = pd.concat([df_old.reindex(columns=columns), df_rows.reindex(columns=columns)])
        for column in set(columns) & {*table.date_type_cols, "scope_changed_at"}:
            values = pd.to_datetime(df_compared[column], errors="coerce")
            df_compared[column] = values.dt.normalize() if column in table.date_type_cols else values
        df_changed = df_compared.drop_duplicates(keep=False)
        if not df_changed.empty:
            invalidate_source_proof(bound, _affected_tickers(df_changed, cohorts, df_retained), identity_change=True)
        if table == Tables.sp500_tickers:
            return bound.save(table, df_rows, ignore_all_conflicts=True)
        if where is None:
            return bound.replace(table, df_rows)
        bound.delete(table, where)
        return bound.save(table, df_rows, ignore_all_conflicts=True)
