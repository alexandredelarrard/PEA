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
import hashlib
import io
import json
import logging
import tempfile
import threading
from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from pandas.core.dtypes.cast import maybe_box_native
from sqlalchemy import Connection, Engine, MetaData, Select, Table, case, create_engine, false, func, inspect, or_, select, text, tuple_
from sqlalchemy import types as sqltypes
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.sql.elements import ColumnElement

from src.constants.constants import SEC_13F_SOURCE_ACCOUNTED
from src.data_store import ddl
from src.data_store.errors import TableEmptyError, TableMissingError
from src.data_store.schema import (
    MANAGED,
    Tables,
    name_of,
    projection_report,
    resolve,
)
from src.data_store.schema import (
    Table as SchemaTable,
)
from src.utils.institutional_vintages import reconcile_13f_source

_CHUNK = 10_000
_CREATE_LOCK = threading.Lock()
logger = logging.getLogger(__name__)


def table_exists(engine: Engine, name: str) -> bool:
    return inspect(engine).has_table(name)


def _reflect(engine: Engine | Connection, name: str) -> Table:
    return Table(name, MetaData(), autoload_with=engine)


def ensure_table(engine: Engine, name: str, df: pd.DataFrame) -> None:
    """Create the table (registry PK, column types from `df`'s dtypes) if it does not exist.

    The CREATE runs under a process-wide lock with a re-check inside it, so concurrent writers
    to a cold table create it exactly once."""
    if table_exists(engine, name):
        return
    statements = [stmt for stmt in ddl.table_ddl_from_frame(resolve(name), df).split(";") if stmt.strip()]
    with _CREATE_LOCK:
        if table_exists(engine, name):
            return
        with engine.begin() as conn:
            for stmt in statements:
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
    return {t.name: reflect_columns(engine, t.name) for t in MANAGED if table_exists(engine, t.name)}


def ensure_columns(engine: Engine, name: str, df: pd.DataFrame) -> list[str]:
    """Schema evolution: ADD COLUMN for any DataFrame column missing from the
    table, so newly-extracted fundamentals / features persist instead of being
    silently dropped by the column-filter in upsert/copy. Returns the columns
    added. No-op if the table doesn't exist yet (caller runs ensure_table first)
    or on engines other than PostgreSQL and SQLite."""
    if engine.dialect.name not in {"postgresql", "sqlite"} or not table_exists(engine, name):
        return []
    existing = set(_reflect(engine, name).c.keys())
    missing = [c for c in df.columns if c not in existing]
    if not missing:
        return []
    spec = resolve(name)
    with engine.begin() as conn:
        for c in missing:
            sqltype = ddl.sql_type(c, df[c].dtype, spec=spec)
            guard = "IF NOT EXISTS " if engine.dialect.name == "postgresql" else ""
            conn.execute(text(f'ALTER TABLE "{name}" ADD COLUMN {guard}"{c}" {sqltype}'))
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
    object_cols = [c for c, dt in df.dtypes.items() if isinstance(dt, object)]
    has_list = any(df[c].apply(lambda v: isinstance(v, list | tuple)).any() for c in object_cols)
    if engine.dialect.name != "postgresql" or has_list:
        return upsert_dataframe(engine, df, name, list(resolve(name).pk))

    tbl = _reflect(engine, name)
    df = cast(pd.DataFrame, df[[c for c in df.columns if c in tbl.c]])
    buf = io.StringIO()
    df.to_csv(buf, index=False, header=False, quoting=csv.QUOTE_MINIMAL)
    buf.seek(0)
    cols = ", ".join(f'"{c}"' for c in df.columns)
    # CSV COPY reads an unquoted empty field as NULL; a NOT NULL text column keeps '' instead.
    keep_empty = [f'"{c}"' for c in df.columns if not tbl.c[c].nullable and isinstance(tbl.c[c].type, sqltypes.String)]
    options = "FORMAT csv" + (f", FORCE_NOT_NULL ({', '.join(keep_empty)})" if keep_empty else "")
    raw = engine.raw_connection()
    try:
        cur = raw.cursor()
        try:
            cur.copy_expert(f'COPY "{name}" ({cols}) FROM STDIN WITH ({options})', buf)
        finally:
            cur.close()
        raw.commit()
    finally:
        raw.close()
    return len(df)


class _NotNull:
    """`where={"col": NOT_NULL}` -> `col IS NOT NULL`. An object, not a magic string, so it
    cannot collide with a stored value (`None` already means IS NULL)."""

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover
        return "NOT_NULL"


NOT_NULL = _NotNull()

# Iterables that mean "IN (...)". `str`/`bytes` are iterable but mean a single value, so they
# must never reach this branch.
_IN_TYPES = (list, tuple, set, frozenset, pd.Series, pd.Index, np.ndarray)
_DATE_TYPES = (pd.Timestamp, dt.datetime, dt.date)


def _is_temporal_column(column) -> bool:
    return isinstance(column.type, sqltypes.Date | sqltypes.DateTime)


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
    if isinstance(value, str | bytes):
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


def build_select(
    tbl: Table,
    columns: Sequence[str] | None = None,
    where: dict[str, Any] | None = None,
    since: Any = None,
    until: Any = None,
    date_col: str | None = None,
    order_by: str | Sequence[str] | None = None,
    descending: bool = False,
    limit: int | None = None,
    distinct_on: str | None = None,
    exclude: tuple[str, object] | None = None,
) -> Select[Any]:
    """The one query builder every read goes through. `where`, `since` and `until` are ANDed.

    `exclude` = (column, value) drops the rows holding that value (NULLs kept); a table without
    the column has no such rows, so it is skipped there.

    Built from reflected Columns and bound parameters, never string interpolation, so a value
    cannot inject SQL and an unknown column raises KeyError before reaching the DB.
    """
    if distinct_on is not None:
        stmt = select(tbl.c[distinct_on]).distinct()
    else:
        stmt = select(*([tbl.c[c] for c in columns] if columns else [tbl]))

    for col, value in (where or {}).items():
        stmt = stmt.where(_predicate(tbl.c[col], value))
    if exclude is not None and exclude[0] in tbl.c:
        stmt = stmt.where(tbl.c[exclude[0]].is_distinct_from(exclude[1]))

    if since is not None or until is not None:
        if date_col is None:
            raise ValueError("since/until need a date column: the table declares no `date_col`, so pass date_col= explicitly")
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
            end = cast(pd.Timestamp, pd.Timestamp(until)).normalize() + pd.Timedelta(days=1)
            stmt = stmt.where(column < _bind_date(column, end))

    if order_by is not None:
        keys = [order_by] if isinstance(order_by, str) else list(order_by)
        stmt = stmt.order_by(*(tbl.c[k].desc() if descending else tbl.c[k].asc() for k in keys))
    if limit is not None:
        stmt = stmt.limit(limit)
    return stmt


def read_table(
    engine: Engine,
    name: str,
    columns: list[str] | None = None,
    limit: int | None = None,
    where: dict[str, object] | None = None,
    exclude: tuple[str, object] | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Execute a built SELECT and return the frame. See `build_select` for the filters."""
    tbl = _reflect(engine, name)
    stmt = build_select(tbl, columns=columns, where=where, limit=limit, exclude=exclude, **kwargs)
    with engine.connect() as conn:
        return pd.read_sql(stmt, conn)


def row_count(engine: Engine, name: str) -> int:
    tbl = _reflect(engine, name)
    with engine.connect() as conn:
        return int(conn.execute(select(func.count()).select_from(tbl)).scalar_one())


def _records(df: pd.DataFrame) -> list[dict]:
    """DataFrame -> list of dicts with pandas NaN/NaT coerced to None so the
    driver binds SQL NULL rather than a float nan.

    One object array for the whole frame: a numpy or string column already yields native
    scalars, so only the other columns are boxed element-wise as `to_dict` would box them."""
    names = list(df.columns)
    if not names:
        return []
    values = df.astype(object).to_numpy(copy=True)
    values[df.isna().to_numpy()] = None
    rows = values.tolist()
    boxed = [
        i for i, dtype in enumerate(df.dtypes) if not (isinstance(dtype, pd.StringDtype) or (isinstance(dtype, np.dtype) and dtype.kind in "biufcmM"))
    ]
    for row in rows if boxed else ():
        for i in boxed:
            row[i] = maybe_box_native(row[i])
    return [dict(zip(names, row, strict=True)) for row in rows]


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


def upsert_dataframe(engine: Engine, df: pd.DataFrame, name: str, pk: list[str], chunksize: int = _CHUNK) -> int:
    """INSERT the frame, updating the columns THE FRAME CARRIES on PK conflict. Returns the
    number of rows sent. No-op for an empty frame.

    A partial frame MERGES: a column the frame does not carry keeps its stored value. The
    update list is therefore built from the frame's columns and not from the table's, which
    is not a nicety -- `excluded.<col>` for a column absent from the INSERT resolves to that
    column's DEFAULT, so building the list from `tbl.c` sets every omitted column to NULL.
    Measured: a 4-column gender patch over `def14a_llm` nulled 392 of 409 rows including
    `def14a_json`, destroying the LLM answers they were the only copy of.

    Callers that mean to blank a column must send it explicitly as null; a caller that means
    to rewrite a whole table wants `replace`.
    """
    if df is None or df.empty:
        return 0
    tbl = _reflect(engine, name)
    df = cast(pd.DataFrame, df[[c for c in df.columns if c in tbl.c]])  # only real columns
    records = _records(_coerce_temporal(df, tbl))
    dialect = engine.dialect.name
    n = len(records)
    carried = set(df.columns)

    with engine.begin() as conn:
        for i in range(0, n, chunksize):
            chunk = records[i : i + chunksize]
            if dialect in ("postgresql", "sqlite"):
                ins = pg_insert(tbl) if dialect == "postgresql" else sqlite_insert(tbl)
                update_cols = {c.name: ins.excluded[c.name] for c in tbl.c if c.name not in pk and c.name in carried}
                stmt = (
                    ins.on_conflict_do_update(index_elements=pk, set_=update_cols) if update_cols else ins.on_conflict_do_nothing(index_elements=pk)
                )
                conn.execute(stmt, chunk)
            else:  # generic fallback
                _delete_then_insert(conn, tbl, chunk, pk)
    return n


def _delete_then_insert(conn, tbl: Table, chunk: list[dict], pk: list[str]) -> None:
    """DELETE the conflicting keys, then INSERT. Only reached on a dialect with no
    `ON CONFLICT` (neither Postgres nor SQLite, i.e. nothing this repo runs on).

    It does NOT merge: a column the chunk omits comes back as that column's default, because
    the old row is gone before the new one lands. Reproducing the merge here means reading
    the conflicting rows first, so the divergence is recorded rather than hidden.
    """
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

    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        self._13f_tables: dict[str, Table] = {}
        self._13f_lock = threading.RLock()

    @staticmethod
    @contextmanager
    def temporary_insider_source() -> Generator[DataStore]:
        """An isolated actual-payload spool using only the existing insider source pair."""
        with tempfile.TemporaryDirectory(prefix="pea-insider-") as directory:
            engine = create_engine("sqlite:///" + (Path(directory) / "source.sqlite").as_posix())
            try:
                yield DataStore(engine)
            finally:
                engine.dispose()

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
            return cast(pd.Timestamp, pd.Timestamp(value)).normalize()
        except (TypeError, ValueError):
            return None

    def max_date_by(self, table, key_col: str, date_col: str | None = None) -> dict[str, pd.Timestamp]:
        """Per-key latest stored date -- `SELECT key, MAX(date) ... GROUP BY key`.

        The grouped counterpart of `max_date`, for the fetchers that resume per entity.
        One aggregate query, so the resume check never loads the table: `prices` is
        ~1.8M rows, and reading it just to `groupby("ticker")["date"].max()` in pandas
        cost seconds and hundreds of MB per run.

        Empty dict when the table or either column is absent -- callers read that as
        "nothing stored yet", the same contract as `max_date`'s None."""
        name = name_of(table)
        col = date_col or self._date_col(table)
        if col is None or not self.exists(name):
            return {}
        tbl = _reflect(self.engine, name)
        if col not in tbl.c or key_col not in tbl.c:
            return {}
        with self.engine.connect() as conn:
            rows = conn.execute(select(tbl.c[key_col], func.max(tbl.c[col])).group_by(tbl.c[key_col])).all()
        # a DATE column round-trips as `datetime.date`, so normalise every value
        out: dict[str, pd.Timestamp] = {}
        for key, value in rows:
            if key is None or value is None:
                continue
            try:
                out[str(key)] = cast(pd.Timestamp, pd.Timestamp(value)).normalize()
            except (TypeError, ValueError):
                continue
        return out

    def key_stats(self, table: SchemaTable | str, key_col: str, date_col: str | None = None, where: dict[str, object] | None = None) -> pd.DataFrame:
        """Per-key `first`, `last` and row count `n` of `date_col`, in one `GROUP BY` read.

        Returns columns `key, first, last, n` with the dates as midnight Timestamps. Marker rows
        are counted. An empty frame when the table or either column is absent."""
        name = name_of(table)
        col = date_col or self._date_col(table)
        rows: list = []
        if col is not None and self.exists(name):
            tbl = _reflect(self.engine, name)
            if col in tbl.c and key_col in tbl.c:
                key, day = tbl.c[key_col], tbl.c[col]
                stmt = select(key, func.min(day), func.max(day), func.count()).where(key.isnot(None))
                for where_col, value in (where or {}).items():
                    stmt = stmt.where(_predicate(tbl.c[where_col], value))
                with self.engine.connect() as conn:
                    rows = list(conn.execute(stmt.group_by(key)).all())
        df = pd.DataFrame([tuple(r) for r in rows], columns=["key", "first", "last", "n"])
        for edge in ("first", "last"):
            df[edge] = pd.to_datetime(df[edge], errors="coerce").dt.normalize()
        return df.astype({"key": str, "n": "int64"})

    def distinct(
        self, table, column: str, *, where: dict | None = None, order: str | None = None, limit: int | None = None, dropna: bool = True
    ) -> list:
        """`SELECT DISTINCT col [WHERE ...] [ORDER BY col asc|desc] [LIMIT n]`. Replaces the
        DISTINCT queries in `sec_utils`, `bulk_cache`, the earnings-call streamers and
        `step_train`."""
        name = name_of(table)
        if not self.exists(name):
            return []
        tbl = _reflect(self.engine, name)
        stmt = build_select(tbl, where=where, distinct_on=column, order_by=column if order else None, descending=(order == "desc"), limit=limit)
        with self.engine.connect() as conn:
            values = [r[0] for r in conn.execute(stmt).all()]
        return [v for v in values if v is not None] if dropna else values

    def _migrate_13f_table(self, spec: SchemaTable) -> bool:
        """Verify one source key or migrate its existing legacy key atomically."""
        with self.engine.begin() as conn:
            if self.engine.dialect.name == "postgresql":
                conn.execute(text("SELECT pg_advisory_xact_lock(hashtext(:scope))"), {"scope": spec.name})
            old = inspect(conn).get_pk_constraint(spec.name)
            if tuple(old["constrained_columns"]) == spec.pk:
                return False
            expected = tuple(key for key in spec.pk if key != "accession_number")
            if tuple(old["constrained_columns"]) != expected:
                raise ValueError(f"Unexpected 13F primary key: {spec.name}: {old}")
            if self.engine.dialect.name != "postgresql":
                raise ValueError("Legacy 13F primary-key migration requires PostgreSQL")
            conn.execute(text(f"ALTER TABLE {ddl.quote(spec.name)} ADD COLUMN IF NOT EXISTS accession_number TEXT NOT NULL DEFAULT '_legacy'"))
            conn.execute(text(f"ALTER TABLE {ddl.quote(spec.name)} DROP CONSTRAINT {ddl.quote(old['name'])}"))
            conn.execute(text(f"ALTER TABLE {ddl.quote(spec.name)} ADD PRIMARY KEY ({', '.join(ddl.quote(key) for key in spec.pk)})"))
        return True

    def migrate_13f_tables(self) -> list[str]:
        """Rekey existing PostgreSQL tables, retaining every unverified legacy row."""
        changed: list[str] = []
        with self._13f_lock:
            for spec in (Tables.sec13f_hr, Tables.sec13f_manager_holdings):
                prepared = self._13f_tables.get(spec.name)
                if prepared is not None and tuple(prepared.primary_key.columns.keys()) == spec.pk:
                    continue
                if not self.exists(spec):
                    continue
                if self._migrate_13f_table(spec):
                    self._13f_tables.pop(spec.name, None)
                    changed.append(spec.name)
                    logger.warning("%s: accession primary key installed; legacy rows retained as unverified _legacy.", spec.name)
        return changed

    def _prepare_13f_table(self, table: SchemaTable, columns: list[str], build: Callable[[], pd.DataFrame]) -> Table:
        """Reuse verified schema until incoming columns or a store schema operation changes it;
        `build` makes the typed frame for DDL only when the schema must be prepared."""
        with self._13f_lock:
            prepared = self._13f_tables.get(table.name)
            if prepared is not None and tuple(prepared.primary_key.columns.keys()) == table.pk and set(columns) <= set(prepared.c.keys()):
                return prepared
            frame = build()
            ensure_table(self.engine, table.name, frame)
            self.migrate_13f_tables()
            ensure_columns(self.engine, table.name, frame)
            prepared = _reflect(self.engine, table.name)
            self._13f_tables[table.name] = prepared
            return prepared

    def save_13f_filing(
        self,
        manifest: pd.DataFrame,
        holdings: pd.DataFrame,
        *,
        table: SchemaTable = Tables.sec13f_manager_holdings,
    ) -> int:
        """Atomically preserve a full accession and its completion marker in an existing table."""
        if table not in (Tables.sec13f_hr, Tables.sec13f_manager_holdings):
            raise ValueError("13F filings belong in the two existing holdings tables")
        if len(manifest) != 1 or manifest.iloc[0]["parse_status"] != "complete":
            raise ValueError("13F save requires one complete filing manifest")
        row = manifest.iloc[0]
        if int(row["parsed_row_count"]) != len(holdings) or int(row["source_row_count"]) < len(holdings):
            raise ValueError("13F source row count does not reconcile")
        reconcile_13f_source(
            int(row["source_row_count"]),
            row["declared_row_count"],
            float(row["source_value_total"]),
            row["declared_value_total"],
            source_accounted=str(row.get("value_unit_evidence", "")).startswith(SEC_13F_SOURCE_ACCOUNTED),
        )
        if holdings.duplicated(["cik", "accession_number", "cusip"]).any() or holdings.cusip.eq(table.empty_marker[1]).any():
            raise ValueError("Duplicate or reserved CUSIP within an immutable filing")
        for key in ("cik", "accession_number", "period", "filing_date"):
            if key not in holdings or (
                not holdings.empty
                and not (
                    pd.to_datetime(holdings[key]).eq(pd.Timestamp(row[key])).all()
                    if key in {"period", "filing_date"}
                    else holdings[key].eq(row[key]).all()
                )
            ):
                raise ValueError(f"13F manifest/holdings mismatch: {key}")
        raw_columns = ["raw_value_usd", "raw_call_value", "raw_put_value", "raw_debt_value", "raw_other_value"]
        if any(column not in holdings for column in raw_columns):
            raise ValueError("13F immutable holdings require raw typed value legs")
        raw_values = holdings[raw_columns].to_numpy(dtype=float)
        if not np.isfinite(raw_values).all() or (raw_values < 0).any():
            raise ValueError("13F raw holdings values must be finite and nonnegative")
        if not np.isclose(float(raw_values.sum()), float(row["source_value_total"]), rtol=0.005, atol=1):
            raise ValueError("13F holdings raw value total does not reconcile to source")
        # Byte layout and clocks differ between ZIP and EDGAR; compare economic source content.
        keys = (
            "cik",
            "accession_number",
            "period",
            "filing_date",
            "amendment_type",
            "amendment_number",
            "report_type",
            "restricted",
            "source_row_count",
            "source_value_total",
        )
        canonical = {key: str(row[key]) for key in keys}
        payload_cols = ["cusip", "shares", "call_shares", "put_shares", "debt_prn", *raw_columns, "class_evidence"]
        payload = holdings[payload_cols].sort_values("cusip").fillna("").astype(str)
        canonical["positions"] = [
            dict(zip(payload_cols, values, strict=True)) for values in zip(*(payload[c].tolist() for c in payload_cols), strict=True)
        ]
        checksum = hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest()
        marker = manifest.assign(cusip=table.empty_marker[1], content_checksum=checksum)
        if table == Tables.sec13f_hr:
            if "ticker" not in holdings or holdings.ticker.isna().any():
                raise ValueError("13F broad holdings require ticker or the unmapped sentinel")
            marker["ticker"] = table.empty_marker[1]
        positions = holdings.assign(parse_status="position")
        columns = list(dict.fromkeys([*positions.columns, *marker.columns]))

        def typed_frame() -> pd.DataFrame:
            frame = pd.concat([positions, marker], ignore_index=True)
            for column in manifest:
                if column not in holdings:
                    dtype = manifest[column].dtype
                    dtype = "Int64" if pd.api.types.is_integer_dtype(dtype) else ("boolean" if pd.api.types.is_bool_dtype(dtype) else dtype)
                    frame[column] = frame[column].astype(dtype)
            return frame

        books = self._prepare_13f_table(table, columns, typed_frame)
        with self.engine.begin() as conn:
            if self.engine.dialect.name == "postgresql":
                conn.execute(text("SELECT pg_advisory_xact_lock(hashtext(:scope))"), {"scope": f"{table.name}:{row['accession_number']}"})
            existing = (
                conn.execute(
                    select(books.c.content_checksum, books.c.declared_row_count, books.c.declared_value_total).where(
                        books.c.cik == row["cik"],
                        books.c.period == _bind_date(books.c.period, pd.Timestamp(row["period"])),
                        books.c.accession_number == row["accession_number"],
                        books.c.cusip == table.empty_marker[1],
                    )
                )
                .mappings()
                .one_or_none()
            )
            if existing is not None:
                declared_count = None if pd.isna(row["declared_row_count"]) else int(row["declared_row_count"])
                declared_total = None if pd.isna(row["declared_value_total"]) else float(row["declared_value_total"])
                if (
                    existing["content_checksum"] != checksum
                    or existing["declared_row_count"] != declared_count
                    or existing["declared_value_total"] != declared_total
                ):
                    raise ValueError(f"13F checksum conflict for {row['accession_number']}")
                return 0
            records = _records(_coerce_temporal(positions.reindex(columns=columns), books))
            for offset in range(0, len(records), _CHUNK):
                conn.execute(books.insert(), records[offset : offset + _CHUNK])
            conn.execute(books.insert(), _records(_coerce_temporal(marker.reindex(columns=columns), books)))
        return len(holdings)

    def load(
        self,
        table: SchemaTable | str,
        columns: Sequence[str] | None = None,
        limit: int | None = None,
        where: dict[str, Any] | None = None,
        *,
        project: bool = False,
        since: object = None,
        until: object = None,
        date_col: str | None = None,
        order_by: str | Sequence[str] | None = None,
        descending: bool = False,
        optional: bool = False,
        markers: bool = False,
    ) -> pd.DataFrame | None:
        """Read `table`, filtered server-side.

        `where` is equality / IN / IS NULL / `NOT_NULL`; `since`/`until` bound the date column;
        `project=True` uses the table's declared `read_columns`. Empty-filing marker rows
        (`Table.empty_marker`) are dropped unless `markers=True`.

        RAISES on a missing or empty result, since an empty read is nearly always a real fault.
        `optional=True` returns None instead, for reads allowed to have no data yet: branch on `is None`.
        """
        name = name_of(table)
        cols = self._resolve_columns(table, columns, project)
        if not self.exists(name):
            if optional:
                return None
            raise TableMissingError(name)

        df = read_table(
            self.engine,
            name,
            columns=list(cols) if cols else None,
            limit=limit,
            where=where,
            exclude=self._marker(table, markers),
            since=since,
            until=until,
            date_col=date_col or self._date_col(table),
            order_by=order_by,
            descending=descending,
        )

        if df.empty:
            if optional:
                return None
            raise TableEmptyError(name, where)

        return df

    def iter_load(
        self,
        table: SchemaTable | str,
        *,
        chunksize: int = 200_000,
        columns: Sequence[str] | None = None,
        project: bool = False,
        where: dict[str, object] | None = None,
        since: object = None,
        until: object = None,
        date_col: str | None = None,
        order_by: str | Sequence[str] | None = None,
        markers: bool = False,
    ) -> Generator[pd.DataFrame]:
        """Stream `table` in row chunks so a multi-GB table never fully materializes.

        Needs a projection (`columns=` or `project=True`) and reads with `stream_results=True`, without
        which psycopg2 buffers the whole result client-side. Marker rows are dropped unless
        `markers=True`, as in `load`. `order_by` orders the stream before chunking.

        Callers MUST exhaust or close the iterator: it holds a pooled connection for its lifetime.
        """
        name = name_of(table)
        cols = self._resolve_columns(table, columns, project)
        if not cols:
            raise ValueError(
                f"iter_load({name}) needs an explicit `columns=` or `project=True`: "
                "streaming an unprojected wide table defeats the purpose (cube is 574 cols)"
            )
        if not self.exists(name):
            return
        tbl = _reflect(self.engine, name)
        stmt = build_select(
            tbl,
            columns=list(cols),
            where=where,
            exclude=self._marker(table, markers),
            since=since,
            until=until,
            date_col=date_col or self._date_col(table),
            order_by=order_by,
        )
        with self.engine.connect() as conn:
            conn = conn.execution_options(stream_results=True, yield_per=chunksize, max_row_buffer=chunksize)
            yield from pd.read_sql(stmt, conn, chunksize=chunksize)

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

    @staticmethod
    def _marker(table: SchemaTable | str, markers: bool) -> tuple[str, object] | None:
        """The (column, sentinel) a consumer read excludes; None when `markers` or undeclared."""
        if markers:
            return None
        if isinstance(table, SchemaTable):
            return table.empty_marker
        try:
            return resolve(table).empty_marker
        except KeyError:
            return None

    def _resolve_columns(self, table, columns, project: bool) -> Sequence[str] | None:
        if columns is not None and project:
            raise ValueError("pass either columns= or project=True, not both")
        if columns is not None:
            return list(columns)
        if not project:
            return None
        cols, required_missing, optional_missing = projection_report(table, self.columns(table) or None)
        if required_missing:
            logger.warning("%s is missing REQUIRED column(s) %s -> the features that need them will be empty", name_of(table), required_missing)
        elif optional_missing:
            logger.info("%s has no %s (optional) -> those features are skipped", name_of(table), optional_missing)
        return cols

    @contextmanager
    def _institutional_transaction(self) -> Generator[Connection]:
        """Serialize source, identity and certification commits without permanent metadata."""
        with self.engine.begin() as conn:
            if conn.dialect.name == "postgresql":
                conn.execute(text("SELECT pg_advisory_xact_lock(hashtext(:scope))"), {"scope": "institutional-source-proof"})
            elif conn.dialect.name == "sqlite":
                conn.exec_driver_sql("BEGIN IMMEDIATE")
            else:
                raise ValueError("Institutional proof transactions require PostgreSQL or SQLite")
            yield conn

    @staticmethod
    def _institutional_select(conn: Connection, spec: SchemaTable, where: dict[str, object] | None = None) -> pd.DataFrame:
        """Read one exact mutation/certification cohort on the guarded connection."""
        if not inspect(conn).has_table(spec.name):
            return pd.DataFrame()
        tbl = _reflect(conn, spec.name)
        statement = select(tbl)
        for column, value in (where or {}).items():
            statement = statement.where(_predicate(tbl.c[column], value))
        return pd.read_sql(statement, conn)

    @staticmethod
    def _institutional_insert(conn: Connection, spec: SchemaTable, frame: pd.DataFrame) -> int:
        """Upsert a prepared bounded frame using this transaction, preserving omitted fields."""
        if frame.empty:
            return 0
        tbl = _reflect(conn, spec.name)
        records = _records(_coerce_temporal(frame, tbl))
        ins = pg_insert(tbl) if conn.dialect.name == "postgresql" else sqlite_insert(tbl)
        updates = {str(column): ins.excluded[str(column)] for column in frame if column not in spec.pk}
        statement = ins.on_conflict_do_update(index_elements=list(spec.pk), set_=updates) if updates else ins.on_conflict_do_nothing()
        for offset in range(0, len(records), _CHUNK):
            conn.execute(statement, records[offset : offset + _CHUNK])
        return len(records)

    @staticmethod
    def _invalidate_checked_quiet(conn: Connection, table: Table, spec: SchemaTable, tickers: set[str]) -> None:
        """An invalidated absence must not become an apparently observed legacy zero."""
        if spec == Tables.short_interest and {"short_volume", "total_volume"}.issubset(table.c.keys()):
            quiet = (table.c.short_volume == 0) & table.c.total_volume.is_(None)
            values = {"short_volume": None}
        elif spec == Tables.sec_fails_to_deliver and "fails_quantity" in table.c:
            quiet = table.c.fails_quantity == 0
            values = {column: None for column in ("fails_quantity", "fails_value") if column in table.c}
        else:
            return
        predicate = table.c.ticker.in_(sorted(tickers)) & table.c.source_checked_at.is_not(None) & quiet
        conn.execute(table.update().where(predicate).values(values))

    @staticmethod
    def _invalidate_institutional_proof(conn: Connection, tickers: set[str], identity_change: bool) -> None:
        """A committed mutation clears retained proof, including the last deleted accession."""
        for spec, column in (
            (Tables.insider_transactions, "inventory_checked_through"),
            (Tables.short_interest, "source_checked_at"),
            (Tables.sec_fails_to_deliver, "source_checked_at"),
        ):
            if not identity_change and spec != Tables.insider_transactions:
                continue
            if not tickers or not inspect(conn).has_table(spec.name):
                continue
            tbl = _reflect(conn, spec.name)
            if column in tbl.c:
                if identity_change and spec != Tables.insider_transactions:
                    DataStore._invalidate_checked_quiet(conn, tbl, spec, tickers)
                conn.execute(tbl.update().where(tbl.c.ticker.in_(sorted(tickers))).values({column: None}))

    def replace_institutional_source(
        self,
        frames: dict[SchemaTable, pd.DataFrame],
        scopes: dict[SchemaTable, dict[str, object] | None],
        expected_identity: dict[SchemaTable, pd.DataFrame] | None = None,
        expected_source: dict[SchemaTable, pd.DataFrame] | None = None,
    ) -> dict[SchemaTable, int]:
        """Atomically replace one source pair; retained EDGAR authority is excluded from new ZIP row counts."""
        pairs = (
            {Tables.sec_short_volume_security, Tables.short_interest},
            {Tables.sec_fails_to_deliver_security, Tables.sec_fails_to_deliver},
            {Tables.insider_transactions, Tables.insider_footnotes},
        )
        if set(frames) not in pairs or set(scopes) != set(frames) or any(scope == {} for scope in scopes.values()):
            raise ValueError("An institutional source write needs one exact source pair and explicit scopes")
        prepared = dict(frames)
        if Tables.insider_transactions in prepared:
            transactions = prepared[Tables.insider_transactions]
            prepared[Tables.insider_transactions] = transactions.assign(
                inventory_checked_through=pd.Series(pd.NaT, index=transactions.index, dtype="datetime64[ns]")
            )
        for spec, frame in prepared.items():
            if not set(spec.pk).issubset(frame) or frame.duplicated(list(spec.pk)).any():
                raise ValueError(f"{spec}: missing or repeated source keys")
            ensure_table(self.engine, spec.name, frame)
            ensure_columns(self.engine, spec.name, frame)
        with self._institutional_transaction() as conn:
            if expected_identity is not None:
                self._assert_institutional_identity(conn, expected_identity)
            self._assert_institutional_source_keys(conn, prepared, scopes)
            plans = {
                spec: self._institutional_source_predicates(conn, spec, scopes[spec], expected_identity, prepared.get(Tables.sec_fails_to_deliver))
                for spec in prepared
            }
            old = {spec: pd.read_sql(select(plans[spec][0]).where(*plans[spec][1]), conn) for spec in prepared}
            for spec, expected in (expected_source or {}).items():
                if spec not in old or not self._institutional_equal(conn, spec, old[spec], expected):
                    raise ValueError("Institutional source cohort changed before atomic restamping")
            if Tables.insider_transactions in prepared:
                transactions, notes = self._institutional_insider_payload(
                    old[Tables.insider_transactions],
                    old[Tables.insider_footnotes],
                    prepared[Tables.insider_transactions],
                    prepared[Tables.insider_footnotes],
                )
                prepared = {
                    Tables.insider_transactions: transactions.assign(
                        inventory_checked_through=pd.Series(pd.NaT, index=transactions.index, dtype="datetime64[ns]")
                    ),
                    Tables.insider_footnotes: notes,
                }
                transactions = [old[Tables.insider_transactions], prepared[Tables.insider_transactions]]
                tickers = {str(value) for frame in transactions for value in frame.get("ticker", pd.Series(dtype=str)).dropna()}
                self._invalidate_institutional_proof(conn, tickers, False)
            for spec in scopes:
                tbl, predicates = plans[spec]
                statement = tbl.delete().where(*predicates)
                conn.execute(statement)
            counts = {spec: self._institutional_insert(conn, spec, frame) for spec, frame in prepared.items()}
            supplied = frames.get(Tables.insider_transactions)
            if supplied is not None and "source" in supplied and supplied["source"].eq("zip").all():
                committed = prepared[Tables.insider_transactions]
                keys = pd.MultiIndex.from_frame(supplied[list(Tables.insider_transactions.pk)])
                incoming = pd.MultiIndex.from_frame(committed[list(Tables.insider_transactions.pk)]).isin(keys)
                counts[Tables.insider_transactions] = int((incoming & committed["source"].eq("zip")).sum())
            return counts

    @staticmethod
    def _institutional_source_predicates(
        conn: Connection,
        spec: SchemaTable,
        scope: dict[str, object] | None,
        identity: dict[SchemaTable, pd.DataFrame] | None,
        grain: pd.DataFrame | None,
    ) -> tuple[Table, list[ColumnElement[bool]]]:
        """A closed FTD raw cohort also owns dated, unambiguously placed NULL-owner source keys."""
        tbl = _reflect(conn, spec.name)
        ordinary: list[ColumnElement[bool]] = [_predicate(tbl.c[column], value) for column, value in (scope or {}).items()]
        if spec != Tables.sec_fails_to_deliver_security or identity is None or scope is None or "ticker" not in scope:
            return tbl, ordinary
        return DataStore._ftd_source_predicates(conn, scope, grain)

    @staticmethod
    def _ftd_source_predicates(conn: Connection, scope: dict[str, object], grain: pd.DataFrame | None) -> tuple[Table, list[ColumnElement[bool]]]:
        """Select and delete with the same dated owner predicate; ambiguity cannot certify absence."""
        raw = _reflect(conn, Tables.sec_fails_to_deliver_security.name)
        master = _reflect(conn, Tables.security_master.name).alias("ftd_scope_master")
        targets = scope["ticker"] if isinstance(scope["ticker"], _IN_TYPES) else [scope["ticker"]]
        base = [_predicate(raw.c[column], value) for column, value in scope.items() if column != "ticker"]
        covers = (master.c.valid_from <= raw.c.trade_date) & (master.c.valid_to.is_(None) | (raw.c.trade_date < master.c.valid_to))
        exact = select(1).where(master.c.cusip == raw.c.cusip, master.c.source_symbol == raw.c.source_symbol, covers).correlate(raw).exists()
        same = (master.c.cusip == raw.c.cusip) & covers & ((master.c.source_symbol == raw.c.source_symbol) | ~exact)
        owner_count = select(func.count(func.distinct(master.c.canonical_company))).where(same).correlate(raw).scalar_subquery()
        target = select(1).where(same, master.c.canonical_company.in_(targets)).correlate(raw).exists()
        unowned = select(1).where(same, master.c.canonical_company.is_(None)).correlate(raw).exists()
        proved = raw.c.ticker.is_(None) & target & (owner_count == 1) & ~unowned
        uncertain: set[tuple[str, pd.Timestamp]] = set()
        for table, company, symbol in (
            (Tables.security_master, "canonical_company", "source_symbol"),
            (Tables.entity_lineage, "canonical_ticker", "symbol"),
        ):
            if not inspect(conn).has_table(table.name):
                continue
            evidence = _reflect(conn, table.name)
            if not {company, symbol}.issubset(evidence.c.keys()):
                continue
            match = evidence.c[symbol] == raw.c.source_symbol
            if "cusip" in evidence.c:
                match |= evidence.c.cusip == raw.c.cusip
            statement = select(evidence.c[company], raw.c.date).select_from(raw.join(evidence, match))
            statement = statement.where(*base, raw.c.ticker.is_(None), ~proved, evidence.c[company].in_(targets))
            uncertain.update((str(company_value), pd.Timestamp(day)) for company_value, day in conn.execute(statement))
        if grain is not None and "source_checked_at" in grain:
            checked = grain.loc[grain["source_checked_at"].notna()]
            if uncertain & set(zip(checked["ticker"], pd.to_datetime(checked["date"]), strict=True)):
                raise ValueError("Ambiguous retained FTD raw ownership cannot certify a complete company observation")
        return raw, [*base, raw.c.ticker.in_(targets) | proved]

    @staticmethod
    def _assert_institutional_source_keys(
        conn: Connection, frames: dict[SchemaTable, pd.DataFrame], scopes: dict[SchemaTable, dict[str, object] | None]
    ) -> None:
        """Scoped writes cannot steal retained owners, and FTD periods cannot collide at unchanged keys."""
        for spec in (Tables.sec_short_volume_security, Tables.sec_fails_to_deliver_security, Tables.sec_fails_to_deliver):
            incoming = frames.get(spec)
            if incoming is None or incoming.empty:
                continue
            tbl = _reflect(conn, spec.name)
            statement = select(tbl).where(*(tbl.c[key].in_(incoming[key].drop_duplicates().tolist()) for key in spec.pk))
            actual = pd.read_sql(statement, conn)
            if actual.empty:
                continue
            both = _coerce_temporal(incoming, tbl).merge(_coerce_temporal(actual, tbl), on=list(spec.pk), suffixes=("", "_stored"))
            if spec in (Tables.sec_fails_to_deliver_security, Tables.sec_fails_to_deliver) and "period_stored" in both:
                if both["period_stored"].notna().any() and both["period"].ne(both["period_stored"]).any():
                    raise ValueError("Conflicting FTD source periods at an unchanged source key")
            scope = scopes[spec] or {}
            owners = scope.get("ticker")
            if spec != Tables.sec_fails_to_deliver and owners is not None and "ticker_stored" in both:
                owners = owners if isinstance(owners, _IN_TYPES) else [owners]
                if (both["ticker_stored"].notna() & ~both["ticker_stored"].isin(owners)).any():
                    raise ValueError("Scoped institutional source write would mutate another retained issuer")

    @staticmethod
    def _institutional_insider_payload(
        old_transactions: pd.DataFrame, old_notes: pd.DataFrame, transactions: pd.DataFrame, notes: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Preserve current EDGAR authority and another issuer's actual owner-only filing."""
        if old_transactions.empty or transactions.empty:
            return transactions, notes
        if "quarter" in old_transactions:
            quarter_of = old_transactions.groupby("accession_number")["quarter"].first()
            quarter = transactions.get("quarter", pd.Series(index=transactions.index, dtype=str))
            transactions = transactions.assign(quarter=quarter.fillna(transactions["accession_number"].map(quarter_of)))
        markers = transactions["security_type"].eq("_empty")
        marker_only = set(transactions.loc[markers, "accession_number"]) - set(transactions.loc[~markers, "accession_number"])
        incoming_keys = pd.MultiIndex.from_frame(transactions.loc[markers, ["accession_number", "ticker"]])
        stored_keys = pd.MultiIndex.from_frame(old_transactions[["accession_number", "ticker"]])
        retained = old_transactions.loc[old_transactions["accession_number"].isin(marker_only) & ~stored_keys.isin(incoming_keys)]
        source = transactions.get("source", pd.Series(index=transactions.index, dtype=str))
        zipped = transactions.loc[source.eq("zip")]
        if not zipped.empty and "source" in old_transactions:
            edgar = old_transactions.loc[
                old_transactions["source"].eq("edgar") & old_transactions["accession_number"].isin(zipped["accession_number"])
            ]
            if not edgar.empty:
                quarter_of = zipped.groupby("accession_number")["quarter"].first()
                edgar = edgar.assign(quarter=edgar["accession_number"].map(quarter_of))
                transactions = transactions.loc[~transactions["accession_number"].isin(edgar["accession_number"])]
                retained = pd.concat([retained, edgar], ignore_index=True)
        if retained.empty:
            return transactions, notes
        preserved = set(retained["accession_number"])
        payload = pd.concat([transactions, retained], ignore_index=True).drop_duplicates(list(Tables.insider_transactions.pk), keep="last")
        notes = pd.concat(
            [notes.loc[~notes["accession_number"].isin(preserved)], old_notes.loc[old_notes["accession_number"].isin(preserved)]], ignore_index=True
        )
        return payload, notes

    def patch_insider_source(
        self, patch: pd.DataFrame, removed: Sequence[str] | pd.DataFrame, expected_identity: dict[SchemaTable, pd.DataFrame] | None = None
    ) -> int:
        """Apply actual quarter/lineage patches and accession deletions under the source-pair guard."""
        allowed = {*Tables.insider_transactions.pk, "quarter", "economic_date", "lineage_role"}
        if set(patch.columns) - allowed:
            raise ValueError("An insider patch carries only existing quarter/lineage fields")
        dropped = removed.get("accession_number", pd.Series(dtype=str)).astype(str) if isinstance(removed, pd.DataFrame) else removed
        accessions = sorted(set(patch.get("accession_number", pd.Series(dtype=str)).astype(str)) | set(dropped))
        if not accessions or not self.exists(Tables.insider_transactions):
            return 0
        if not patch.empty:
            ensure_columns(self.engine, Tables.insider_transactions.name, patch)
        with self._institutional_transaction() as conn:
            if expected_identity is not None:
                self._assert_institutional_identity(conn, expected_identity)
            stored = self._institutional_select(conn, Tables.insider_transactions, {"accession_number": accessions})
            notes = self._institutional_select(conn, Tables.insider_footnotes, {"accession_number": accessions})
            payload = stored.set_index(list(Tables.insider_transactions.pk))
            if not patch.empty:
                changes = patch.set_index(list(Tables.insider_transactions.pk))
                if not changes.index.isin(payload.index).all():
                    raise ValueError("An insider patch cannot invent an accession row")
                payload.loc[changes.index, changes.columns] = changes
            payload = payload.reset_index()
            payload = payload.loc[~self._insider_removal_mask(payload, removed)]
            tickers = set(stored["ticker"].dropna().astype(str)) | set(payload["ticker"].dropna().astype(str))
            self._invalidate_institutional_proof(conn, tickers, False)
            if "inventory_checked_through" in payload:
                payload["inventory_checked_through"] = pd.NaT
            transactions = _reflect(conn, Tables.insider_transactions.name)
            conn.execute(transactions.delete().where(transactions.c.accession_number.in_(accessions)))
            orphaned = set(accessions) - set(payload["accession_number"])
            if not notes.empty and orphaned:
                footnotes = _reflect(conn, Tables.insider_footnotes.name)
                conn.execute(footnotes.delete().where(footnotes.c.accession_number.in_(sorted(orphaned))))
            self._institutional_insert(conn, Tables.insider_transactions, payload)
            return len(stored) - len(payload) if len(removed) else len(patch)

    @staticmethod
    def _insider_removal_mask(payload: pd.DataFrame, removed: Sequence[str] | pd.DataFrame) -> pd.Series:
        """Issuer-scoped keyed deletions cannot erase another retained owner of the accession."""
        if not isinstance(removed, pd.DataFrame):
            return payload["accession_number"].isin(removed)
        if removed.empty:
            return pd.Series(False, index=payload.index)
        allowed = {*Tables.insider_transactions.pk, "ticker", "issuer_cik"}
        if not set(Tables.insider_transactions.pk).issubset(removed) or set(removed) - allowed:
            raise ValueError("Insider deletion requires declared row keys and optional actual issuer identity")
        columns = list(removed.columns)
        return pd.Series(pd.MultiIndex.from_frame(payload[columns]).isin(pd.MultiIndex.from_frame(removed[columns])), index=payload.index)

    @staticmethod
    def _institutional_equal(conn: Connection, spec: SchemaTable, actual: pd.DataFrame, expected: pd.DataFrame) -> bool:
        """Compare exact payloads with SQL NULL/date normalization, ignoring only the proof being stamped."""
        columns = [column for column in actual if column != "inventory_checked_through"]
        if set(expected.columns) - {"inventory_checked_through"} != set(columns):
            return False
        if actual.empty and expected.empty:
            return True
        tbl = _reflect(conn, spec.name)
        left = _records(_coerce_temporal(actual[columns].sort_values(list(spec.pk)), tbl))
        right = _records(_coerce_temporal(expected[columns].sort_values(list(spec.pk)), tbl))
        return left == right

    def certify_insider_inventory(
        self,
        ticker: str,
        through: pd.Timestamp,
        expected_transactions: pd.DataFrame,
        expected_footnotes: pd.DataFrame,
        expected_identity: dict[SchemaTable, pd.DataFrame],
    ) -> int:
        """Stamp only the exact reconciled payload/notes/identity cohort under the mutation guard."""
        identities = {Tables.sp500_tickers, Tables.entity_lineage, Tables.symbol_tenure, Tables.security_master}
        if set(expected_identity) != identities or expected_transactions.empty:
            raise ValueError("Certification requires an actual accession cohort and the closed identity cohort")
        with self._institutional_transaction() as conn:
            actual = self._institutional_select(conn, Tables.insider_transactions, {"ticker": ticker})
            if "inventory_checked_through" not in actual or not self._institutional_equal(
                conn, Tables.insider_transactions, actual, expected_transactions
            ):
                raise ValueError("Insider source cohort changed before certification")
            accessions = sorted(set(actual["accession_number"].astype(str)))
            notes = self._institutional_select(conn, Tables.insider_footnotes, {"accession_number": accessions})
            if not self._institutional_equal(conn, Tables.insider_footnotes, notes, expected_footnotes):
                raise ValueError("Insider footnote cohort changed before certification")
            self._assert_institutional_identity(conn, expected_identity)
            fetched = pd.to_datetime(actual.get("fetched_at", pd.Series(pd.NaT, index=actual.index)), errors="coerce")
            changes = self._insider_scope_changes(ticker, actual, expected_identity)
            if fetched.isna().any() or (changes.notna().any() and fetched.lt(changes.max()).any()):
                raise ValueError("Every insider accession must be freshly reconciled for its dated identity scope")
            tbl = _reflect(conn, Tables.insider_transactions.name)
            return int(
                conn.execute(
                    tbl.update().where(tbl.c.ticker == ticker).values(inventory_checked_through=_bind_date(tbl.c.inventory_checked_through, through))
                ).rowcount
            )

    def _assert_institutional_identity(self, conn: Connection, expected_identity: dict[SchemaTable, pd.DataFrame]) -> None:
        """The mapping used outside the transaction must still be the exact guarded identity cohort."""
        if set(expected_identity) != {Tables.sp500_tickers, Tables.entity_lineage, Tables.symbol_tenure, Tables.security_master}:
            raise ValueError("Expected identity must contain the closed institutional source cohort")
        for spec, expected in expected_identity.items():
            identity = self._institutional_select(conn, spec)
            columns = ["ticker", "cik"] if spec == Tables.sp500_tickers else list(spec.read_columns)
            columns = [column for column in columns if column in identity]
            if not self._institutional_equal(conn, spec, identity[columns], expected):
                raise ValueError(f"Institutional identity cohort changed before commit: {spec}")

    def insider_inventory_snapshot(self, ticker: str, expected_accessions: Sequence[str] | None) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Capture a prior joint cohort under its guard; an inventory change clears retained proof."""
        with self._institutional_transaction() as conn:
            transactions = self._institutional_select(conn, Tables.insider_transactions, {"ticker": ticker})
            accessions = sorted(set(transactions.get("accession_number", pd.Series(dtype=str)).astype(str)))
            notes = self._institutional_select(conn, Tables.insider_footnotes, {"accession_number": accessions})
            if expected_accessions is not None and set(accessions) != set(expected_accessions):
                self._invalidate_institutional_proof(conn, {ticker}, False)
            return transactions, notes

    def insider_inventory_boundaries(self, tickers: Sequence[str]) -> pd.Series:
        """Compact requested-issuer proof summary, including markers and every actual row."""
        result = pd.Series(pd.NaT, index=pd.Index(sorted(set(tickers)), name="ticker"), dtype="datetime64[ns]")
        columns = self.columns(Tables.insider_transactions)
        if not {"inventory_checked_through", "fetched_at", "ticker"}.issubset(columns):
            return result
        tbl = _reflect(self.engine, Tables.insider_transactions.name)
        checked = tbl.c.inventory_checked_through
        statement = select(tbl.c.ticker, func.min(checked), func.max(checked), func.count(checked), func.count(tbl.c.fetched_at), func.count())
        statement = statement.where(tbl.c.ticker.in_(list(result.index))).group_by(tbl.c.ticker)
        with self.engine.connect() as conn:
            rows = conn.execute(statement).all()
        for ticker, first, last, verified, fetched, total in rows:
            if first is not None and first == last and verified == fetched == total:
                result.loc[ticker] = pd.Timestamp(first)
        return result

    def short_volume_source_cohort(self, companies: Sequence[str], symbols: Sequence[str]) -> pd.DataFrame:
        """Read complete touched-symbol cohorts once, including retained other-owner dates."""
        if not self.exists(Tables.sec_short_volume_security):
            return pd.DataFrame()
        raw = _reflect(self.engine, Tables.sec_short_volume_security.name)
        owned_symbols = select(raw.c.source_symbol).where(raw.c.ticker.in_(list(companies))).distinct()
        statement = select(raw).where(raw.c.source_symbol.in_(owned_symbols) | raw.c.source_symbol.in_(list(symbols)))
        with self.engine.connect() as conn:
            return pd.read_sql(statement, conn)

    def fails_source_periods(self) -> tuple[set[str], set[str]]:
        """Compact raw inventory and period keys whose raw or derived payload needs reconciliation."""
        pending = set(map(str, self.distinct(Tables.sec_fails_to_deliver, "period", where={"fails_quantity": None})))
        if not self.exists(Tables.sec_fails_to_deliver_security):
            return set(), pending
        raw = _reflect(self.engine, Tables.sec_fails_to_deliver_security.name)
        statement = select(raw.c.period, func.count(), func.count(raw.c.security_id)).group_by(raw.c.period)
        with self.engine.connect() as conn:
            rows = conn.execute(statement).all()
        stored = {str(period) for period, _, _ in rows if period is not None}
        pending |= {str(period) for period, total, stamped in rows if period is not None and stamped < total}
        return stored, pending

    def short_source_days(self, tickers: Sequence[str]) -> tuple[set[pd.Timestamp], set[pd.Timestamp]]:
        """One grouped date read retains global day presence and scoped unknown-quantity retries."""
        if not self.exists(Tables.short_interest):
            return set(), set()
        table = _reflect(self.engine, Tables.short_interest.name)
        unknown = table.c.short_volume.is_(None) if "short_volume" in table.c else table.c.date.is_not(None)
        retry = func.max(case((table.c.ticker.in_(tickers) & unknown, 1), else_=0))
        statement = select(table.c.date, retry).where(table.c.date.is_not(None)).group_by(table.c.date)
        with self.engine.connect() as conn:
            rows = conn.execute(statement).all()
        stored = {pd.Timestamp(day).normalize() for day, _ in rows}
        pending = {pd.Timestamp(day).normalize() for day, needs_retry in rows if needs_retry}
        return stored, pending

    @staticmethod
    def _insider_scope_changes(ticker: str, transactions: pd.DataFrame, identity: dict[SchemaTable, pd.DataFrame]) -> pd.Series:
        changes = []
        ciks = set(transactions.get("issuer_cik", pd.Series(dtype=str)).dropna().astype(str).str.zfill(10))
        for spec, company in ((Tables.entity_lineage, "canonical_ticker"), (Tables.security_master, "canonical_company")):
            frame = identity[spec]
            if "scope_changed_at" not in frame:
                continue
            applicable = frame[company].eq(ticker) if company in frame else pd.Series(False, index=frame.index)
            cik_column = "cik" if spec == Tables.entity_lineage else "issuer_cik"
            if cik_column in frame:
                applicable |= frame[cik_column].astype("string").str.zfill(10).isin(ciks)
            changes.append(pd.to_datetime(frame.loc[applicable, "scope_changed_at"], errors="coerce"))
        return pd.concat(changes, ignore_index=True) if changes else pd.Series(dtype="datetime64[ns]")

    @staticmethod
    def _institutional_subset(frame: pd.DataFrame, where: dict[str, object] | None) -> pd.DataFrame:
        selected = pd.Series(True, index=frame.index)
        for column, value in (where or {}).items():
            if column not in frame:
                return frame.iloc[:0]
            if isinstance(value, _IN_TYPES):
                selected &= frame[column].isin(value)
            else:
                selected &= frame[column].isna() if value is None else (frame[column] == value)
        return frame.loc[selected]

    @staticmethod
    def _institutional_identity_tickers(conn: Connection, changed: pd.DataFrame, cohorts: dict[SchemaTable, pd.DataFrame]) -> set[str]:
        """Resolve old and new identity-bearing CIK/symbols to retained proof owners."""
        ciks = {str(value).zfill(10) for column in ("cik", "issuer_cik") if column in changed for value in changed[column].dropna()}
        symbols = {
            str(value) for column in ("symbol", "source_symbol", "market_symbol", "ticker") if column in changed for value in changed[column].dropna()
        }
        tickers = {
            str(value) for column in ("canonical_ticker", "canonical_company", "ticker") if column in changed for value in changed[column].dropna()
        }
        for frame in cohorts.values():
            matched = pd.Series(False, index=frame.index)
            for column in ("cik", "issuer_cik"):
                if column in frame:
                    matched |= frame[column].astype("string").str.zfill(10).isin(ciks)
            for column in ("symbol", "source_symbol", "market_symbol", "ticker"):
                if column in frame:
                    matched |= frame[column].isin(symbols)
            for column in ("ticker", "canonical_ticker", "canonical_company"):
                if column in frame:
                    tickers.update(str(value) for value in frame.loc[matched, column].dropna())
        if inspect(conn).has_table(Tables.insider_transactions.name):
            tbl = _reflect(conn, Tables.insider_transactions.name)
            predicates = []
            if "issuer_cik" in tbl.c and ciks:
                predicates.append(tbl.c.issuer_cik.in_(sorted(ciks)))
            if "source_symbol" in tbl.c and symbols:
                predicates.append(tbl.c.source_symbol.in_(sorted(symbols)))
            if predicates:
                tickers.update(str(row[0]) for row in conn.execute(select(tbl.c.ticker).where(or_(*predicates)).distinct()) if row[0] is not None)
        return tickers

    def replace_institutional_identity(self, table: SchemaTable, rows: pd.DataFrame, where: dict[str, object] | None = None) -> int:
        """Commit an existing identity replacement/partition with physical proof invalidation."""
        if table not in (Tables.entity_lineage, Tables.security_master, Tables.symbol_tenure):
            raise ValueError("Only the institutional lineage, master and tenure writers use this seam")
        if where == {} or (table != Tables.symbol_tenure and where is not None):
            raise ValueError("Only a symbol-tenure partition can carry a replacement scope")
        if not set(table.pk).issubset(rows) or rows.duplicated(list(table.pk)).any():
            raise ValueError("Identity replacement needs unique declared keys")
        ensure_table(self.engine, table.name, rows)
        ensure_columns(self.engine, table.name, rows)
        with self._institutional_transaction() as conn:
            cohorts = {
                spec: self._institutional_select(conn, spec)
                for spec in (Tables.sp500_tickers, Tables.entity_lineage, Tables.symbol_tenure, Tables.security_master)
            }
            old = self._institutional_subset(cohorts[table], where)
            tbl = _reflect(conn, table.name)
            columns = self._institutional_identity_columns(table, set(old.columns) | set(rows.columns))
            compared = pd.concat([_coerce_temporal(old.reindex(columns=columns), tbl), _coerce_temporal(rows.reindex(columns=columns), tbl)])
            changed = compared.drop_duplicates(keep=False)
            if not changed.empty:
                tickers = self._institutional_identity_tickers(conn, changed, cohorts)
                self._invalidate_institutional_proof(conn, tickers, True)
            statement = tbl.delete()
            for column, value in (where or {}).items():
                statement = statement.where(_predicate(tbl.c[column], value))
            conn.execute(statement)
            return self._institutional_insert(conn, table, rows)

    @staticmethod
    def _institutional_identity_columns(table: SchemaTable, available: set[str]) -> list[str]:
        """Semantic scope and existing owner stamps; observation/evidence churn is not new identity."""
        columns = [*table.pk, "valid_to"]
        if table == Tables.entity_lineage:
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

    def _save_institutional_roster(self, frame: pd.DataFrame) -> int:
        """Preserve roster partial-upsert semantics while guarding actual ticker/CIK changes."""
        spec = Tables.sp500_tickers
        ensure_table(self.engine, spec.name, frame)
        ensure_columns(self.engine, spec.name, frame)
        with self._institutional_transaction() as conn:
            cohorts = {
                table: self._institutional_select(conn, table)
                for table in (spec, Tables.entity_lineage, Tables.symbol_tenure, Tables.security_master)
            }
            old = self._institutional_subset(cohorts[spec], {"ticker": frame["ticker"].tolist()})
            changed = pd.concat([old.reindex(columns=["ticker", "cik"]), frame[["ticker", "cik"]]]).drop_duplicates(keep=False)
            if not changed.empty:
                self._invalidate_institutional_proof(conn, self._institutional_identity_tickers(conn, changed, cohorts), True)
            return self._institutional_insert(conn, spec, frame)

    # -- writes ------------------------------------------------------------ #
    def save(self, table: SchemaTable | str, df: pd.DataFrame, pk: list[str] | None = None) -> int:
        """Upsert `df` into `table` on its PK, creating the table if needed."""
        name = name_of(table)
        if df is None or df.empty:
            logger.warning(f"save() skipped: table '{name}' is empty")
            return 0
        if name == Tables.sp500_tickers.name and "cik" in df:
            return self._save_institutional_roster(df)
        if pk is None:
            pk = list(resolve(table).pk)
        with self._13f_lock if name in (Tables.sec13f_hr.name, Tables.sec13f_manager_holdings.name) else nullcontext():
            self._13f_tables.pop(name, None)
            if name in (Tables.sec13f_hr.name, Tables.sec13f_manager_holdings.name):
                if "accession_number" not in df:
                    df = df.assign(accession_number="_legacy")
                self.migrate_13f_tables()
            ensure_table(self.engine, name, df)
            ensure_columns(self.engine, name, df)  # evolve schema for new columns
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
                    f"replace(), which evolve the schema."
                )
        return copy_load(self.engine, df, name)

    def replace_institutional_output(self, rows: pd.DataFrame, *, since: pd.Timestamp | None = None) -> int:
        """Atomically replace the owned output schema or its inclusive corrected range."""
        spec = Tables.cube_part_institutionals
        keys = list(spec.pk)
        if not set(keys).issubset(rows) or rows[keys].isna().any().any() or rows.duplicated(keys).any() or rows.columns.duplicated().any():
            raise ValueError("Institutional output requires complete unique keys and columns")
        if since is not None and (pd.to_datetime(rows["date"]) < since).any():
            raise ValueError("Institutional replacement rows precede their corrected range")
        with self._institutional_transaction() as conn:
            if since is None:
                conn.execute(text(f'DROP TABLE IF EXISTS "{spec.name}"'))
                for statement in ddl.table_ddl_from_frame(spec, rows).split(";"):
                    if statement.strip():
                        conn.execute(text(statement))
            tbl = _reflect(conn, spec.name)
            if set(tbl.c.keys()) != set(rows.columns):
                raise ValueError("Institutional corrected range must retain the stored schema")
            if since is not None:
                conn.execute(tbl.delete().where(tbl.c.date >= _bind_date(tbl.c.date, pd.Timestamp(since).normalize())))
            for offset in range(0, len(rows), _CHUNK):
                chunk = rows.iloc[offset : offset + _CHUNK]
                conn.execute(tbl.insert(), _records(_coerce_temporal(chunk, tbl)))
        return len(rows)

    def append_tail(self, table, df: pd.DataFrame, cutoff, *, inclusive: bool = False, date_col: str | None = None) -> int:
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
        boundary = cast(pd.Timestamp, pd.Timestamp(cutoff)).normalize()
        if not inclusive:
            boundary += pd.Timedelta(days=1)
        with self.engine.begin() as conn:
            conn.execute(text(f'DELETE FROM "{name}" WHERE "{col}" >= :cut'), {"cut": _bind_date(tbl.c[col], boundary)})
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

    def replace(self, table: SchemaTable | str, df: pd.DataFrame, chunksize: int = 200_000) -> int:
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
        with self._13f_lock if name in (Tables.sec13f_hr.name, Tables.sec13f_manager_holdings.name) else nullcontext():
            self._13f_tables.pop(name, None)
            if resolve(table).is_unmanaged:
                self.drop(name)
            if name in (Tables.sec13f_hr.name, Tables.sec13f_manager_holdings.name):
                if "accession_number" not in df:
                    df = df.assign(accession_number="_legacy")
                self.migrate_13f_tables()
            ensure_table(self.engine, name, df)
            ensure_columns(self.engine, name, df)  # evolve schema for new columns
            with self.engine.begin() as conn:
                conn.execute(text(f'DELETE FROM "{name}"'))
            for i in range(0, len(df), chunksize):
                copy_load(self.engine, df.iloc[i : i + chunksize], name)
            return len(df)

    def ensure_columns(self, table: SchemaTable | str, df: pd.DataFrame) -> list[str]:
        """ADD COLUMN for missing frame columns and invalidate prepared 13F metadata."""
        name = name_of(table)
        with self._13f_lock if name in (Tables.sec13f_hr.name, Tables.sec13f_manager_holdings.name) else nullcontext():
            self._13f_tables.pop(name, None)
            return ensure_columns(self.engine, name, df)

    def drop(self, table: SchemaTable | str) -> None:
        """Drop a table if it exists (unmanaged-part rebuilds, superseded-table cleanup).

        Takes a bare name as well as a `Table`, since the thing being dropped is sometimes a
        table that has already been removed from the registry.
        """
        name = name_of(table)
        with self._13f_lock if name in (Tables.sec13f_hr.name, Tables.sec13f_manager_holdings.name) else nullcontext():
            self._13f_tables.pop(name, None)
            with self.engine.begin() as conn:
                conn.execute(text(f'DROP TABLE IF EXISTS "{name}"'))
