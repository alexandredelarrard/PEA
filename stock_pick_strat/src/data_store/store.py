"""
`DataStore` -- the only place in the repo that issues SQL. Engine-level helpers below it,
the facade at the bottom.

Dialect-aware: Postgres and SQLite both use native INSERT .. ON CONFLICT DO UPDATE, any
other dialect falls back to delete-by-PK-then-insert, so the same code path serves the
production container and an in-memory SQLite test DB.
"""
from __future__ import annotations

import csv
import datetime as dt
import io
import logging
from typing import Iterator, Sequence

import numpy as np
import pandas as pd
from sqlalchemy import Engine, MetaData, Table, false, func, inspect, select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy import tuple_
from sqlalchemy import types as sqltypes

from src.data_store import ddl
from src.data_store.errors import TableEmptyError, TableMissingError
from src.data_store.schema import (
    MANAGED, Table as SchemaTable, name_of, projection_report, resolve,
)

_CHUNK = 10_000
logger = logging.getLogger(__name__)


def table_exists(engine: Engine, name: str) -> bool:
    return inspect(engine).has_table(name)


def _reflect(engine: Engine, name: str) -> Table:
    return Table(name, MetaData(), autoload_with=engine)


def ensure_table(engine: Engine, name: str, df: pd.DataFrame) -> None:
    """Create the table (with its registry PK) if it does not exist yet, using the
    DataFrame's dtypes for column types. Needed for tables with no DDL in sql/schema.sql
    yet (e.g. def14a_llm) so the first fetcher write succeeds -- and for every unmanaged
    `cube_part_*` table, whose DDL only ever comes from the frame being written."""
    if table_exists(engine, name):
        return
    spec = resolve(name)
    with engine.begin() as conn:
        for stmt in ddl.table_ddl_from_frame(spec, df).split(";"):
            if stmt.strip():
                conn.execute(text(stmt))


def reflect_columns(engine: Engine, name: str) -> list[tuple[str, str]]:
    """Ordered (column, SQL type) from the LIVE table's real schema.

    Used instead of sampled dtypes so a column that is all-NaN in an early sample (gold
    before 2000) keeps its true DOUBLE PRECISION type rather than being inferred as TEXT.
    """
    return [(c.name, str(c.type)) for c in _reflect(engine, name).c]


def reflect_all(engine: Engine) -> dict[str, list[tuple[str, str]]]:
    """`{table: [(column, sql_type)]}` for every MANAGED table that exists.

    This is the map `ddl.generate_schema_sql` consumes. Doing the reflection HERE rather
    than inside `ddl` is what removes the `ddl -> store` back-edge that made the data_store
    imports circular (and forced four deferred imports inside this module).
    """
    return {t.name: reflect_columns(engine, t.name)
            for t in MANAGED if table_exists(engine, t.name)}


def ensure_columns(engine: Engine, name: str, df: pd.DataFrame) -> list[str]:
    """Schema evolution: ADD COLUMN for any DataFrame column missing from the
    table, so newly-extracted fundamentals / features persist instead of being
    silently dropped by the column-filter in upsert/copy. Returns the columns
    added. No-op if the table doesn't exist yet (caller runs ensure_table first)
    or on non-Postgres engines (SQLite is dynamically typed)."""
    if engine.dialect.name != "postgresql" or not table_exists(engine, name):
        return []
    existing = set(_reflect(engine, name).c.keys())
    missing = [c for c in df.columns if c not in existing]
    if not missing:
        return []
    with engine.begin() as conn:
        for c in missing:
            sqltype = ddl.sql_type(c, df[c].dtype, spec=None)
            conn.execute(text(f'ALTER TABLE "{name}" '
                              f'ADD COLUMN IF NOT EXISTS "{c}" {sqltype}'))
    return missing


def copy_load(engine: Engine, df: pd.DataFrame, name: str) -> int:
    """Fast bulk insert via Postgres COPY (CSV). For the one-time seed of empty
    tables; no conflict handling. Falls back to upsert on non-Postgres engines
    or for frames carrying array/list columns (COPY can't format those)."""
    if df is None or df.empty:
        return 0
    # Only OBJECT columns can hold a list/tuple, so restrict the per-cell scan to those.
    # The old version ran `.apply(isinstance)` over EVERY column -- 574 of them per 200k-row
    # cube chunk, all but a handful numeric and incapable of holding a list. Compared on
    # `dtype == object` rather than via select_dtypes, which folds pandas-3 `str` columns in.
    object_cols = [c for c, dt in df.dtypes.items() if dt == object]
    has_list = any(df[c].apply(lambda v: isinstance(v, (list, tuple))).any()
                   for c in object_cols)
    if engine.dialect.name != "postgresql" or has_list:
        return upsert_dataframe(engine, df, name, list(resolve(name).pk))

    tbl = _reflect(engine, name)
    df = df[[c for c in df.columns if c in tbl.c]]
    buf = io.StringIO()
    df.to_csv(buf, index=False, header=False, quoting=csv.QUOTE_MINIMAL)
    buf.seek(0)
    cols = ", ".join(f'"{c}"' for c in df.columns)
    raw = engine.raw_connection()
    try:
        with raw.cursor() as cur:
            cur.copy_expert(
                f'COPY "{name}" ({cols}) FROM STDIN WITH (FORMAT csv)', buf)
        raw.commit()
    finally:
        raw.close()
    return len(df)


class _NotNull:
    """`where={"col": NOT_NULL}` -> `col IS NOT NULL`. An object, not a magic string, so it
    cannot collide with a stored value (`None` already means IS NULL)."""

    __slots__ = ()

    def __repr__(self) -> str:                                # pragma: no cover
        return "NOT_NULL"


NOT_NULL = _NotNull()

# Iterables that mean "IN (...)". `str`/`bytes` are iterable but mean a single value, so they
# must never reach this branch.
_IN_TYPES = (list, tuple, set, frozenset, pd.Series, pd.Index, np.ndarray)
_DATE_TYPES = (pd.Timestamp, dt.datetime, dt.date)


def _is_temporal_column(column) -> bool:
    return isinstance(column.type, (sqltypes.Date, sqltypes.DateTime))


def _bind_date(column, value):
    """Bind a date the way the COLUMN stores it: a real datetime/date for TIMESTAMP/DATE, an
    ISO string for TEXT (SQLite). Reading it off the reflected type is what lets one `since=`
    work on both dialects."""
    ts = pd.Timestamp(value)
    if isinstance(column.type, sqltypes.DateTime):
        return ts.to_pydatetime()
    if isinstance(column.type, sqltypes.Date):
        return ts.date()
    return ts.strftime("%Y-%m-%d")


def _predicate(column, value):
    """One WHERE clause. Normalizes the shapes already in use (`str`, `list`, `set`, `tuple`)
    plus None/NOT_NULL, sorting collections so the SQL is deterministic."""
    if isinstance(value, _NotNull):
        return column.isnot(None)
    if value is None:
        return column.is_(None)
    if isinstance(value, (str, bytes)):
        return column == value
    if isinstance(value, _IN_TYPES):
        values = sorted({v for v in value})
        if not values:
            # An empty IN list is a dialect trap (and a silent full-table read if dropped).
            # "match nothing" is the only honest reading of "in this empty set".
            return false()
        if _is_temporal_column(column):
            values = [_bind_date(column, v) for v in values]
        return column.in_(values)
    if isinstance(value, _DATE_TYPES):
        return column == _bind_date(column, value)
    return column == value


def build_select(tbl, columns=None, where=None, since=None, until=None,
                 date_col: str | None = None, order_by=None,
                 descending: bool = False, limit: int | None = None,
                 distinct_on: str | None = None):
    """The one query builder every read goes through. `where`, `since` and `until` are ANDed.

    Built from reflected Columns and bound parameters, never string interpolation, so a value
    cannot inject SQL and an unknown column raises KeyError before reaching the DB.
    """
    if distinct_on is not None:
        stmt = select(tbl.c[distinct_on]).distinct()
    else:
        stmt = select(*( [tbl.c[c] for c in columns] if columns else [tbl] ))

    for col, value in (where or {}).items():
        stmt = stmt.where(_predicate(tbl.c[col], value))

    if since is not None or until is not None:
        if date_col is None:
            raise ValueError("since/until need a date column: the table declares no "
                             "`date_col`, so pass date_col= explicitly")
        if where and date_col in where:
            # `date = a AND date >= b` is almost never what the caller meant, and silently
            # emitting it would hide the mistake behind an empty result.
            raise ValueError(f"{date_col!r} is constrained by both where= and since/until")
        column = tbl.c[date_col]
        if since is not None:
            stmt = stmt.where(column >= _bind_date(column, since))
        if until is not None:
            # `< until + 1 day`, not `<= until`: on a text-typed column
            # "2024-03-22 00:00:00.000000" <= "2024-03-22" is lexically FALSE, which would
            # drop the very day the caller asked to include. This form is also right for a
            # TIMESTAMP column carrying intraday times.
            end = pd.Timestamp(until).normalize() + pd.Timedelta(days=1)
            stmt = stmt.where(column < _bind_date(column, end))

    if order_by is not None:
        keys = [order_by] if isinstance(order_by, str) else list(order_by)
        stmt = stmt.order_by(*(tbl.c[k].desc() if descending else tbl.c[k].asc()
                               for k in keys))
    if limit is not None:
        stmt = stmt.limit(limit)
    return stmt


def read_table(engine: Engine, name: str, columns: list[str] | None = None,
               limit: int | None = None,
               where: dict[str, object] | None = None, **kwargs) -> pd.DataFrame:
    """Execute a built SELECT and return the frame. See `build_select` for the filters."""
    tbl = _reflect(engine, name)
    stmt = build_select(tbl, columns=columns, where=where, limit=limit, **kwargs)
    with engine.connect() as conn:
        return pd.read_sql(stmt, conn)


def row_count(engine: Engine, name: str) -> int:
    tbl = _reflect(engine, name)
    with engine.connect() as conn:
        return int(conn.execute(select(func.count()).select_from(tbl)).scalar_one())


def _records(df: pd.DataFrame) -> list[dict]:
    """DataFrame -> list of dicts with pandas NaN/NaT coerced to None so the
    driver binds SQL NULL rather than a float nan."""
    return df.astype(object).where(pd.notna(df), None).to_dict("records")


def _coerce_temporal(df: pd.DataFrame, tbl: Table) -> pd.DataFrame:
    """Bind an ISO-string date as a real date/datetime for DATE/TIMESTAMP columns.

    psycopg2 casts the string itself, SQLite's driver refuses it -- so a frame that writes
    fine in production raised `SQLite Date type only accepts Python date objects` on the
    in-memory test DB. Driven off the REFLECTED column type, mirroring `_bind_date` on the
    read side, so one frame writes to both dialects.
    """
    fixed = {}
    for name in df.columns:
        col = tbl.c.get(name)
        if col is None or not _is_temporal_column(col):
            continue
        ts = pd.to_datetime(df[name], errors="coerce")
        # A DATE column needs `datetime.date` even from a datetime64 series, so this is not
        # gated on dtype -- a string dtype under pandas 3 would slip past such a check anyway.
        fixed[name] = ts.dt.date if isinstance(col.type, sqltypes.Date) else ts
    return df.assign(**fixed) if fixed else df


def upsert_dataframe(engine: Engine, df: pd.DataFrame, name: str,
                     pk: list[str], chunksize: int = _CHUNK) -> int:
    """INSERT the frame, updating non-PK columns on PK conflict. Returns the
    number of rows sent. No-op for an empty frame."""
    if df is None or df.empty:
        return 0
    tbl = _reflect(engine, name)
    df = df[[c for c in df.columns if c in tbl.c]]     # only real columns
    records = _records(_coerce_temporal(df, tbl))
    dialect = engine.dialect.name
    n = len(records)

    with engine.begin() as conn:
        for i in range(0, n, chunksize):
            chunk = records[i:i + chunksize]
            if dialect in ("postgresql", "sqlite"):
                ins = pg_insert(tbl) if dialect == "postgresql" else sqlite_insert(tbl)
                update_cols = {c.name: ins.excluded[c.name]
                               for c in tbl.c if c.name not in pk}
                stmt = (ins.on_conflict_do_update(index_elements=pk, set_=update_cols)
                        if update_cols else ins.on_conflict_do_nothing(index_elements=pk))
                conn.execute(stmt, chunk)
            else:                                       # generic fallback
                _delete_then_insert(conn, tbl, chunk, pk)
    return n


def _delete_then_insert(conn, tbl: Table, chunk: list[dict], pk: list[str]) -> None:
    
    keys = [tuple(r[k] for k in pk) for r in chunk]
    conn.execute(tbl.delete().where(tuple_(*[tbl.c[k] for k in pk]).in_(keys)))
    conn.execute(tbl.insert(), chunk)


TableRef = "SchemaTable | str"


class DataStore:
    """THE database access layer -- nothing outside `src/data_store/` issues SQL.

    One instance lives on `context.store`. Every method takes a `Table` from `schema.py` or
    its name, so a call site that has the table also has its pk, date column and projection.

    Absorbs `PartStore` (a second store implementation that existed only because the
    `cube_part_*` tables were unregistered) and the capabilities eleven modules hand-rolled
    with raw `pd.read_sql`: `date >= x`, `IS NOT NULL`, chunked reads, MAX/MIN, DISTINCT and
    column introspection.

    `load` RAISES when there is nothing to read (see `errors.py`); `optional=True` returns
    None for the few genuinely optional reads.
    """

    def __init__(self, engine: Engine):
        self.engine = engine

    # so callers write `store.NOT_NULL` without importing it
    NOT_NULL = NOT_NULL

    # -- introspection ----------------------------------------------------- #
    def exists(self, table) -> bool:
        return table_exists(self.engine, name_of(table))

    def columns(self, table) -> list[str]:
        """Column names, `[]` if absent. Reflection, not `information_schema` -- that view
        does not exist in SQLite, so the callers querying it could not be tested offline."""
        name = name_of(table)
        if not self.exists(name):
            return []
        return list(_reflect(self.engine, name).c.keys())

    def row_count(self, table) -> int:
        name = name_of(table)
        return row_count(self.engine, name) if self.exists(name) else 0

    def bounds(self, table, column: str | None = None) -> tuple:
        """RAW `(MIN, MAX)` of `column` (defaults to the table's `date_col`). Raw because the
        useful bounds are not always temporal -- `fetch_hf_transcripts` reads a `quarter`
        string. `(None, None)` when absent or empty."""
        name = name_of(table)
        col = column or self._date_col(table)
        if col is None:
            raise ValueError(f"{name}: no column given and the table declares no date_col")
        if not self.exists(name):
            return (None, None)
        tbl = _reflect(self.engine, name)
        with self.engine.connect() as conn:
            row = conn.execute(select(func.min(tbl.c[col]), func.max(tbl.c[col]))).one()
        return (row[0], row[1])

    def max_date(self, table, column: str | None = None) -> pd.Timestamp | None:
        """Latest stored date at midnight, `None` if absent/empty.

        None-on-absent is load-bearing: `plan_window` reads it as "no stored data -> full
        rebuild", so this must not raise for an unbuilt part."""
        name = name_of(table)
        col = column or self._date_col(table)
        if col is None or not self.exists(name):
            return None
        tbl = _reflect(self.engine, name)
        if col not in tbl.c:
            return None
        with self.engine.connect() as conn:
            value = conn.execute(select(func.max(tbl.c[col]))).scalar()
        if value is None:
            return None
        try:
            return pd.Timestamp(value).normalize()
        except (TypeError, ValueError):
            return None

    def distinct(self, table, column: str, *, where: dict | None = None,
                 order: str | None = None, limit: int | None = None,
                 dropna: bool = True) -> list:
        """`SELECT DISTINCT col [WHERE ...] [ORDER BY col asc|desc] [LIMIT n]`. Replaces the
        DISTINCT queries in `sec_utils`, `bulk_cache`, the earnings-call streamers and
        `step_train`."""
        name = name_of(table)
        if not self.exists(name):
            return []
        tbl = _reflect(self.engine, name)
        stmt = build_select(tbl, where=where, distinct_on=column,
                            order_by=column if order else None,
                            descending=(order == "desc"), limit=limit)
        with self.engine.connect() as conn:
            values = [r[0] for r in conn.execute(stmt).all()]
        return [v for v in values if v is not None] if dropna else values

    # -- reads ------------------------------------------------------------- #
    def load(self, table, columns: Sequence[str] | None = None,
             limit: int | None = None,
             where: dict[str, object] | None = None, *,
             project: bool = False,
             since: object = None, until: object = None,
             date_col: str | None = None,
             order_by=None, descending: bool = False,
             optional: bool = False) -> pd.DataFrame | None:
        """Read `table`, filtered server-side.

        `where` is equality / IN / IS NULL / `NOT_NULL`; `since`/`until` bound the date
        column; `project=True` uses the table's declared `read_columns`.

        RAISES on a missing or empty result -- an empty read is nearly always a real fault and
        should stop the run here, not surface later as an empty feature panel. Never returns a
        fabricated frame. `optional=True` returns None instead, for reads that are genuinely
        allowed to have no data yet (a fetcher's resume check on a cold DB).
        """
        name = name_of(table)
        cols = self._resolve_columns(table, columns, project)
        if not self.exists(name):
            if optional:
                return None
            raise TableMissingError(name)
        df = read_table(self.engine, name, columns=list(cols) if cols else None,
                        limit=limit, where=where, since=since, until=until,
                        date_col=date_col or self._date_col(table),
                        order_by=order_by, descending=descending)
        if df.empty:
            if optional:
                return None
            raise TableEmptyError(name, where)
        return df

    def iter_load(self, table, *, chunksize: int = 200_000,
                  columns: Sequence[str] | None = None, project: bool = False,
                  where: dict[str, object] | None = None,
                  since: object = None, until: object = None,
                  date_col: str | None = None) -> Iterator[pd.DataFrame]:
        """Stream `table` in row chunks so a multi-GB table never fully materializes.

        Both parts are required: a PROJECTION (mandatory -- `cube` is 574 columns, its readers
        need ~50) and `stream_results=True`, without which psycopg2 buffers the whole result
        client-side before pandas chunks it, making `chunksize` useless for peak memory.

        Callers MUST exhaust or close the iterator -- it holds a pooled connection for its
        lifetime, so breaking out early leaks it.
        """
        name = name_of(table)
        cols = self._resolve_columns(table, columns, project)
        if not cols:
            raise ValueError(
                f"iter_load({name}) needs an explicit `columns=` or `project=True`: "
                "streaming an unprojected wide table defeats the purpose (cube is 574 cols)")
        if not self.exists(name):
            return
        tbl = _reflect(self.engine, name)
        stmt = build_select(tbl, columns=list(cols), where=where, since=since, until=until,
                            date_col=date_col or self._date_col(table))
        with self.engine.connect() as conn:
            conn = conn.execution_options(stream_results=True, yield_per=chunksize,
                                          max_row_buffer=chunksize)
            for chunk in pd.read_sql(stmt, conn, chunksize=chunksize):
                yield chunk

    # -- helpers ----------------------------------------------------------- #
    @staticmethod
    def _date_col(table) -> str | None:
        """The table's declared incremental date column, or None for a bare name."""
        if isinstance(table, SchemaTable):
            return table.date_col
        try:
            return resolve(table).date_col
        except KeyError:
            return None

    def _resolve_columns(self, table, columns, project: bool) -> Sequence[str] | None:
        if columns is not None and project:
            raise ValueError("pass either columns= or project=True, not both")
        if columns is not None:
            return list(columns)
        if not project:
            return None
        cols, required_missing, optional_missing = projection_report(
            table, self.columns(table) or None)
        if required_missing:
            logger.warning("%s is missing REQUIRED column(s) %s -> the features that need "
                           "them will be empty", name_of(table), required_missing)
        elif optional_missing:
            logger.info("%s has no %s (optional) -> those features are skipped",
                        name_of(table), optional_missing)
        return cols

    # -- writes ------------------------------------------------------------ #
    def save(self, table, df: pd.DataFrame, pk: list[str] | None = None) -> int:
        """Upsert `df` into `table` on its PK, creating the table if needed."""
        if df is None or df.empty:
            return 0
        name = name_of(table)
        if pk is None:
            pk = list(resolve(table).pk)
        ensure_table(self.engine, name, df)
        ensure_columns(self.engine, name, df)     # evolve schema for new columns
        return upsert_dataframe(self.engine, df, name, pk)

    def bulk_seed(self, table, df: pd.DataFrame) -> int:
        """Fast COPY append onto a schema `replace` already established (the cube's streaming
        writer). `copy_load` filters to existing columns and nothing here evolves the schema,
        so an unknown column would be dropped silently -- raise instead."""
        name = name_of(table)
        if df is not None and not df.empty and self.exists(name):
            unknown = [c for c in df.columns if c not in set(self.columns(name))]
            if unknown:
                raise ValueError(
                    f"bulk_seed({name}): frame has {len(unknown)} column(s) the table does "
                    f"not have and COPY would silently drop: {unknown[:8]}. Use save() or "
                    f"replace(), which evolve the schema.")
        return copy_load(self.engine, df, name)

    def append_tail(self, table, df: pd.DataFrame, cutoff, *,
                    inclusive: bool = False, date_col: str | None = None) -> int:
        """DELETE rows after `cutoff` (>= if `inclusive`), then append `df`. Idempotent, so
        re-running the same day never duplicates. `inclusive` is what the forward-looking
        target part needs: a label that was NaN last run matures into a value."""
        name = name_of(table)
        col = date_col or self._date_col(table)
        if col is None:
            raise ValueError(f"{name}: append_tail needs a date column")
        tbl = _reflect(self.engine, name)
        # Always a `>=` against a DAY BOUNDARY, never `> cutoff`. A text-typed date column
        # (SQLite stores "2024-03-08 00:00:00.000000") compares LEXICALLY, so `> '2024-03-08'`
        # is true for that very day -- it deleted the cutoff day and the append never brought
        # it back, silently losing one day per incremental run. Postgres hid this because its
        # column is a real TIMESTAMP.
        boundary = pd.Timestamp(cutoff).normalize()
        if not inclusive:
            boundary += pd.Timedelta(days=1)
        with self.engine.begin() as conn:
            conn.execute(
                text(f'DELETE FROM "{name}" WHERE "{col}" >= :cut'),
                {"cut": _bind_date(tbl.c[col], boundary)})
        if df is None or df.empty:
            return 0
        df.to_sql(name, self.engine, if_exists="append", index=False)
        return len(df)

    def delete(self, table, where: dict[str, object]) -> int:
        """DELETE the rows matching `where` (same predicate shapes as `load`); returns rows deleted.

        `where` is REQUIRED and may not be empty: the alternative to a targeted delete is
        load-the-whole-table-and-`replace`, which is how `_drop_stale_turns` used to reconcile a
        few orphaned rows by pulling every 1536-dim vector in the cache into RAM.
        """
        if not where:
            raise ValueError("delete() needs a where= filter; use replace() to rewrite a table")
        name = name_of(table)
        if not self.exists(name):
            return 0
        tbl = _reflect(self.engine, name)
        stmt = tbl.delete()
        for col, value in where.items():
            stmt = stmt.where(_predicate(tbl.c[col], value))
        with self.engine.begin() as conn:
            return int(conn.execute(stmt).rowcount)

    def replace(self, table, df: pd.DataFrame, chunksize: int = 200_000) -> int:
        """Full rebuild: empty the table, fast-COPY the frame back in chunks.

        Managed tables are DELETEd (keeping their sql/schema.sql DDL); unmanaged
        `cube_part_*` tables are DROPPED and recreated, because a rebuild may legitimately
        REMOVE a column and `ensure_columns` only ever adds. With a DELETE the stale column
        would persist, `write_part`'s drift check would fire on every later run, and each
        incremental build would silently become a full 15-year rebuild.
        """
        if df is None or df.empty:
            return 0
        name = name_of(table)
        if resolve(table).is_unmanaged:
            self.drop(name)
        ensure_table(self.engine, name, df)
        ensure_columns(self.engine, name, df)     # evolve schema for new columns
        with self.engine.begin() as conn:
            conn.execute(text(f'DELETE FROM "{name}"'))
        for i in range(0, len(df), chunksize):
            copy_load(self.engine, df.iloc[i:i + chunksize], name)
        return len(df)

    def ensure_columns(self, table, df: pd.DataFrame) -> list[str]:
        """ADD COLUMN for any frame column the table lacks; returns the columns added."""
        return ensure_columns(self.engine, name_of(table), df)

    def drop(self, table) -> None:
        """Drop a table if it exists (unmanaged-part rebuilds, superseded-table cleanup).

        Takes a bare name as well as a `Table`, since the thing being dropped is sometimes a
        table that has already been removed from the registry.
        """
        with self.engine.begin() as conn:
            conn.execute(text(f'DROP TABLE IF EXISTS "{name_of(table)}"'))
