"""
DDL generation for every MANAGED table in `schema.py`. Two consumers, one code path:
`store.ensure_table` (create on first write, from the frame's dtypes) and
`scripts/generate_schema_sql` -> `sql/schema.sql`, which Postgres runs on first init.

Changes from the old `schema_sql.py`:
  * No flat-file path -- the parquet migrator is gone, so regenerating schema.sql now
    REQUIRES a reachable database.
  * Fixes a real bug: reflecting a Postgres `DOUBLE PRECISION[]` gives a type whose `str()`
    is the bare word `ARRAY`, which Postgres rejects as DDL. The committed schema.sql carried
    that at `earning_calls_embedding."embedding"`, so it could not initialise a fresh DB. A
    `vector_col` now always emits the array type.
  * Imports no `store` -- reflection is the caller's job, passed in as a
    `{table: [(column, sql_type)]}` map. That removes the back-edge that made the data_store
    imports circular.
"""
from __future__ import annotations

import re

import pandas as pd
from pandas.api import types as ptypes

from src.data_store.schema import MANAGED, Table

# Columns that are zero-padded string identifiers, never numeric -- forcing them to TEXT
# preserves leading zeros (SEC CIK "0000320193" would lose them as BIGINT).
_TEXT_IDENTIFIER_COLS = {"cik"}

# The SQL type a `vector_col` always gets, whether its columns were inferred from a frame
# or reflected from the live DB. See item 2 in the module docstring.
VECTOR_SQL_TYPE = "DOUBLE PRECISION[]"

# What a reflected array column's `str(type)` degrades to.
_REFLECTED_ARRAY_TYPES = {"ARRAY", "ARRAY[]"}


def sql_type(col: str, dtype, spec: Table | None = None) -> str:
    """The SQL type for one column, from its pandas dtype."""
    if spec is not None and col in spec.date_type_cols:
        return "DATE"
    if col.lower() in _TEXT_IDENTIFIER_COLS:
        return "TEXT"
    if ptypes.is_datetime64_any_dtype(dtype):
        return "TIMESTAMP"
    if ptypes.is_bool_dtype(dtype):
        return "BOOLEAN"
    if ptypes.is_integer_dtype(dtype):
        return "BIGINT"
    if ptypes.is_float_dtype(dtype):
        return "DOUBLE PRECISION"
    return "TEXT"


def quote(ident: str) -> str:
    """Quote an identifier so mixed-case / slash column names survive (e.g. beta_USD/EUR).
    Postgres folds unquoted names to lowercase, which would break round-tripping those."""
    return '"' + ident.replace('"', '""') + '"'


def columns_from_frame(spec: Table, df: pd.DataFrame) -> list[tuple[str, str]]:
    """Ordered (column, SQL type) from a frame's dtypes, collapsing a scalar vector family
    (e0..eN) into one array column. `vector_col` itself is excluded from `other` because
    producers may already supply it pre-collapsed -- else it would be emitted twice."""
    if spec.vector_col and spec.vector_prefix:
        prefix = spec.vector_prefix
        vec_cols = [c for c in df.columns
                    if c.startswith(prefix) and c[len(prefix):].isdigit()]
        other = [c for c in df.columns if c not in vec_cols and c != spec.vector_col]
        cols = [(c, sql_type(c, df[c].dtype, spec)) for c in other]
        cols.append((spec.vector_col, VECTOR_SQL_TYPE))
        return cols
    return [(c, sql_type(c, df[c].dtype, spec)) for c in df.columns]


def _repair_vector_type(spec: Table, cols: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Restore the element type on a reflected array column (see docstring item 2)."""
    if not spec.vector_col:
        return cols
    return [(name, VECTOR_SQL_TYPE)
            if name == spec.vector_col and str(sqltype).upper() in _REFLECTED_ARRAY_TYPES
            else (name, sqltype)
            for name, sqltype in cols]


def table_ddl(spec: Table, cols: list[tuple[str, str]]) -> str:
    """`CREATE TABLE` + the ticker / date indexes, from an ordered column list."""
    cols = _repair_vector_type(spec, cols)
    pk = set(spec.pk)
    lines = [f"    {quote(name)} {sqltype}{' NOT NULL' if name in pk else ''}"
             for name, sqltype in cols]
    lines.append(f"    PRIMARY KEY ({', '.join(quote(c) for c in spec.pk)})")
    body = ",\n".join(lines)
    ddl = [f"CREATE TABLE IF NOT EXISTS {quote(spec.name)} (\n{body}\n);"]
    if spec.ticker_col and spec.ticker_col not in pk:
        ddl.append(f"CREATE INDEX IF NOT EXISTS ix_{spec.name}_{spec.ticker_col} "
                   f"ON {quote(spec.name)} ({quote(spec.ticker_col)});")
    if spec.date_col and spec.date_col not in pk:
        ddl.append(f"CREATE INDEX IF NOT EXISTS ix_{spec.name}_{spec.date_col} "
                   f"ON {quote(spec.name)} ({quote(spec.date_col)});")
    return "\n".join(ddl)


def table_ddl_from_frame(spec: Table, df: pd.DataFrame) -> str:
    """DDL inferred from a frame -- the `ensure_table` path (and the only path available for
    an unmanaged `cube_part_*` table, which has no stored DDL to reflect)."""
    return table_ddl(spec, columns_from_frame(spec, df))


_BLOCK_RE = re.compile(
    r'CREATE TABLE IF NOT EXISTS "(?P<name>[a-z0-9_]+)".*?\n\);'
    r'(?:\nCREATE INDEX[^\n]*\n?)*', re.S)


def existing_blocks(schema_sql: str) -> dict[str, str]:
    """`{table: its CREATE TABLE (+ CREATE INDEX) text}` parsed out of a schema.sql."""
    return {m.group("name"): m.group(0).rstrip() for m in _BLOCK_RE.finditer(schema_sql)}


def generate_schema_sql(reflected: dict[str, list[tuple[str, str]]],
                        previous: str | None = None) -> str:
    """sql/schema.sql text for every MANAGED table (`cube_part_*` own their own DDL).

    `reflected` is `{table: [(column, sql_type)]}` from the live DB, produced by the caller.

    NON-DESTRUCTIVE: a table missing from `reflected` keeps its block from `previous` rather
    than being dropped -- otherwise regenerating against a DB that lacks a table would delete
    the only record of its schema and stop a fresh init from creating it. DDL is never
    invented from nothing.
    """
    prior = existing_blocks(previous) if previous else {}
    blocks = [
        "-- AUTO-GENERATED by src/data_store/ddl.py — do not edit by hand.\n"
        "-- Regenerate: python -m scripts.generate_schema_sql\n"
        "-- Source of truth: src/data_store/schema.py (Tables).\n"
        "-- Every statement is idempotent (CREATE TABLE/INDEX IF NOT EXISTS).\n"
        "-- The cube_part_* tables are deliberately absent: they are private plumbing\n"
        "-- between the cube sub-steps and are created by their owning step."
    ]
    for spec in MANAGED:
        cols = reflected.get(spec.name)
        if cols:
            blocks.append(f"-- [{spec.kind}] {spec.name}  (pk: {', '.join(spec.pk)})")
            blocks.append(table_ddl(spec, cols))
        elif spec.name in prior:
            blocks.append(f"-- [{spec.kind}] {spec.name}  (pk: {', '.join(spec.pk)}) "
                          f"-- CARRIED OVER: not present in the database that generated "
                          f"this file, so its previous DDL is preserved verbatim.")
            blocks.append(prior[spec.name])
        else:
            blocks.append(f"-- SKIPPED (no live schema and no previous DDL): {spec.name}")
    return "\n\n".join(blocks) + "\n"
