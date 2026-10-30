"""The sole SQL boundary: typed PostgreSQL/SQLite reads, writes and schema preparation."""

from __future__ import annotations

import csv
import datetime as dt
import io
import logging
import re
import tempfile
import threading
from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from sqlalchemy import Connection, Engine, MetaData, Select, Table, and_, create_engine, false, func, inspect, or_, select, text, true, tuple_
from sqlalchemy import types as sqltypes
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.sql.elements import ColumnElement

from src.data_store import ddl
from src.data_store.errors import TableEmptyError, TableMissingError
from src.data_store.schema import (
    MANAGED,
    name_of,
    projection_report,
    resolve,
)
from src.data_store.schema import (
    Table as SchemaTable,
)

_CHUNK = 10_000
_CREATE_LOCK = threading.Lock()
logger = logging.getLogger(__name__)


type Where = dict[str, Any] | pd.DataFrame | list[dict[str, Any] | pd.DataFrame]


@contextmanager
def _connection(engine: Engine | Connection, *, write: bool = False) -> Generator[Connection]:
    """Reuse a caller-owned connection without closing or committing it."""
    if isinstance(engine, Connection):
        yield engine
        return
    with engine.begin() if write else engine.connect() as conn:
        yield conn


def _where(tbl: Table, where: Where | None) -> ColumnElement[bool]:
    """Compile dictionary AND, exact declared-key tuples, or a flat OR of those scopes."""
    if where is None:
        return true()
    if isinstance(where, dict):
        return and_(true(), *(_predicate(tbl.c[column], value) for column, value in where.items()))
    if isinstance(where, pd.DataFrame):
        keys = list(resolve(tbl.name).pk)
        if set(where.columns) != set(keys) or where.columns.duplicated().any() or where[keys].isna().any().any():
            raise ValueError("Key scopes require exactly the complete non-null declared primary key")
        df_keys = _coerce_temporal(where[keys].drop_duplicates(), tbl)
        return tuple_(*(tbl.c[key] for key in keys)).in_(list(df_keys.itertuples(index=False, name=None)))
    if isinstance(where, list):
        if any(not isinstance(branch, dict | pd.DataFrame) or isinstance(branch, dict) and not branch for branch in where):
            raise ValueError("OR scopes require non-empty dictionaries or complete-key frames; nesting is not supported")
        return or_(false(), *(_where(tbl, branch) for branch in where))
    raise TypeError("where must be a dictionary, complete-key DataFrame or flat OR list")


def _create_canonical(engine: Engine, tables: Sequence[SchemaTable]) -> None:
    """Initialize selected fresh staging tables from the canonical managed SQL."""
    blocks = ddl.existing_blocks((Path(__file__).resolve().parents[2] / "sql" / "schema.sql").read_text(encoding="utf-8"))
    statements = [statement for table in tables for statement in re.sub(r"--[^\n]*", "", blocks[table.name]).split(";") if statement.strip()]
    with _connection(engine, write=True) as conn:
        for statement in statements:
            conn.execute(text(statement))


def table_exists(engine: Engine | Connection, name: str) -> bool:
    return inspect(engine).has_table(name)


def _reflect(engine: Engine | Connection, name: str, metadata: MetaData | None = None) -> Table:
    return Table(name, metadata if metadata is not None else MetaData(), autoload_with=engine)


def ensure_table(engine: Engine | Connection, name: str, df: pd.DataFrame) -> None:
    """Create missing registry DDL from frame dtypes; serialize and recheck cold-table creation."""
    if table_exists(engine, name):
        return
    statements = [stmt for stmt in ddl.table_ddl_from_frame(resolve(name), df).split(";") if stmt.strip()]
    with _CREATE_LOCK:
        if table_exists(engine, name):
            return
        with _connection(engine, write=True) as conn:
            for stmt in statements:
                conn.execute(text(stmt))


def reflect_columns(engine: Engine | Connection, name: str) -> list[tuple[str, str]]:
    """Ordered live column names/types; never infer SQL types from a nullable sample."""
    return [(c.name, str(c.type)) for c in _reflect(engine, name).c]


def reflect_all(engine: Engine | Connection) -> dict[str, list[tuple[str, str]]]:
    """Live schemas for existing managed tables, consumed by ddl.generate_schema_sql."""
    return {t.name: reflect_columns(engine, t.name) for t in MANAGED if table_exists(engine, t.name)}


def ensure_columns(engine: Engine | Connection, name: str, df: pd.DataFrame) -> list[str]:
    """Add newly observed columns on PostgreSQL/SQLite and return their names; never drop fields."""
    if engine.dialect.name not in {"postgresql", "sqlite"} or not table_exists(engine, name):
        return []
    existing = set(_reflect(engine, name).c.keys())
    missing = [c for c in df.columns if c not in existing]
    if not missing:
        return []
    spec = resolve(name)
    with _connection(engine, write=True) as conn:
        for c in missing:
            sqltype = ddl.sql_type(c, df[c].dtype, spec=spec)
            guard = "IF NOT EXISTS " if engine.dialect.name == "postgresql" else ""
            conn.execute(text(f'ALTER TABLE "{name}" ADD COLUMN {guard}"{c}" {sqltype}'))
    return missing


def copy_load(engine: Engine | Connection, df: pd.DataFrame, name: str) -> int:
    """Append using COPY or the existing array/SQLite upsert fallback on one connection."""
    if df is None or df.empty:
        return 0
    if engine.dialect.name != "postgresql" or any(df[c].apply(lambda value: isinstance(value, list | tuple)).any() for c in df):
        return upsert_dataframe(engine, df, name, list(resolve(name).pk))
    tbl = _reflect(engine, name)
    df = cast(pd.DataFrame, df[[c for c in df.columns if c in tbl.c]])
    buf = io.StringIO()
    df.to_csv(buf, index=False, header=False, quoting=csv.QUOTE_MINIMAL)
    buf.seek(0)
    cols = ", ".join(f'"{c}"' for c in df.columns)
    keep_empty = [f'"{c}"' for c in df.columns if not tbl.c[c].nullable and isinstance(tbl.c[c].type, sqltypes.String)]
    options = "FORMAT csv" + (f", FORCE_NOT_NULL ({', '.join(keep_empty)})" if keep_empty else "")
    with _connection(engine, write=True) as conn:
        cursor = conn.connection.cursor()
        try:
            cursor.copy_expert(f'COPY "{name}" ({cols}) FROM STDIN WITH ({options})', buf)
        finally:
            cursor.close()
    return len(df)


class _NotNull:
    """Unambiguous IS NOT NULL filter sentinel; None already means IS NULL."""

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
    """Bind dates according to the reflected DATE, TIMESTAMP or ISO-text column type."""
    ts = pd.Timestamp(value)
    if isinstance(column.type, sqltypes.DateTime):
        return ts.to_pydatetime()
    if isinstance(column.type, sqltypes.Date):
        return ts.date()
    return ts.strftime("%Y-%m-%d")


def _predicate(column, value):
    """Equality, deterministic IN, NULL and NOT_NULL predicates; an empty collection matches nothing."""
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
    where: Where | None = None,
    since: Any = None,
    until: Any = None,
    date_col: str | None = None,
    order_by: str | Sequence[str] | None = None,
    descending: bool = False,
    limit: int | None = None,
    distinct_on: str | None = None,
    exclude: tuple[str, object] | None = None,
) -> Select[Any]:
    """Build a bound SELECT; filters and inclusive day bounds are ANDed.

    Exclusion keeps NULLs and ignores absent marker columns. Unknown columns raise KeyError."""
    if distinct_on is not None:
        stmt = select(tbl.c[distinct_on]).distinct()
    else:
        stmt = select(*([tbl.c[c] for c in columns] if columns else [tbl]))

    stmt = stmt.where(_where(tbl, where))
    if exclude is not None and exclude[0] in tbl.c:
        stmt = stmt.where(tbl.c[exclude[0]].is_distinct_from(exclude[1]))

    if since is not None or until is not None:
        if date_col is None:
            raise ValueError("since/until need a date column: the table declares no `date_col`, so pass date_col= explicitly")
        branches = where if isinstance(where, list) else [where]
        if any(branch is not None and date_col in branch for branch in branches):
            # Refuse an ambiguous equality/window combination instead of returning a misleading empty result.
            raise ValueError(f"{date_col!r} is constrained by both where= and since/until")
        column = tbl.c[date_col]
        if since is not None:
            stmt = stmt.where(column >= _bind_date(column, since))
        if until is not None:
            # Half-open next-day bounds include intraday timestamps and text-typed SQLite dates.
            end = cast(pd.Timestamp, pd.Timestamp(until)).normalize() + pd.Timedelta(days=1)
            stmt = stmt.where(column < _bind_date(column, end))

    if order_by is not None:
        keys = [order_by] if isinstance(order_by, str) else list(order_by)
        stmt = stmt.order_by(*(tbl.c[k].desc() if descending else tbl.c[k].asc() for k in keys))
    if limit is not None:
        stmt = stmt.limit(limit)
    return stmt


def read_table(
    engine: Engine | Connection,
    name: str,
    columns: list[str] | None = None,
    limit: int | None = None,
    where: Where | None = None,
    exclude: tuple[str, object] | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Execute a built SELECT and return the frame. See `build_select` for the filters."""
    tbl = _reflect(engine, name)
    stmt = build_select(tbl, columns=columns, where=where, limit=limit, exclude=exclude, **kwargs)
    with _connection(engine) as conn:
        return pd.read_sql(stmt, conn)


def row_count(engine: Engine | Connection, name: str) -> int:
    tbl = _reflect(engine, name)
    with _connection(engine) as conn:
        return int(conn.execute(select(func.count()).select_from(tbl)).scalar_one())


def _records(df: pd.DataFrame) -> list[dict]:
    """Bind SQL NULLs, using public pandas boxing only for object/extension columns."""
    names = list(df.columns)
    if not names:
        return []
    values = df.astype(object).to_numpy(copy=True)
    values[df.isna().to_numpy()] = None
    rows: list[dict] = [dict(zip(names, row, strict=True)) for row in values.tolist()]
    boxed = [
        column
        for column, dtype in df.dtypes.items()
        if not (isinstance(dtype, pd.StringDtype) or (isinstance(dtype, np.dtype) and dtype.kind in "biufcmM"))
    ]
    if boxed:
        subset = df[boxed].astype(object)
        for row, native in zip(rows, subset.where(subset.notna(), None).to_dict("records"), strict=True):
            row.update(native)
    return rows


def _coerce_temporal(df: pd.DataFrame, tbl: Table) -> pd.DataFrame:
    """Coerce frame date columns to reflected SQL temporal types, including Python dates for SQLite DATE."""
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


def upsert_dataframe(engine: Engine | Connection, df: pd.DataFrame, name: str, pk: list[str], chunksize: int = _CHUNK) -> int:
    """Upsert carried fields on PK conflict, preserving omitted columns; explicit nulls overwrite.

    Returns rows sent. Empty frames do nothing; whole-table rewrites use replace()."""
    if df is None or df.empty:
        return 0
    tbl = _reflect(engine, name)
    if set(df.columns) - set(tbl.c.keys()):
        raise ValueError(f"{name}: upsert contains unknown columns")
    records = _records(_coerce_temporal(df, tbl))
    with _connection(engine, write=True) as conn:
        return _upsert_records(conn, tbl, records, pk, chunksize)


def _upsert_records(
    conn: Connection, tbl: Table, records: list[dict], pk: list[str], chunksize: int = _CHUNK, *, ignore_all_conflicts: bool = False
) -> int:
    """Execute carried-column upserts without taking ownership of the caller's transaction."""
    offsets = range(0, len(records), chunksize)
    if not records:
        return 0
    dialect = conn.dialect.name
    if dialect in {"postgresql", "sqlite"}:
        ins = pg_insert(tbl) if dialect == "postgresql" else sqlite_insert(tbl)
        updates = {column: ins.excluded[column] for column in records[0] if column not in pk}
        statement = (
            ins.on_conflict_do_update(index_elements=pk, set_=updates)
            if updates
            else ins.on_conflict_do_nothing(index_elements=None if ignore_all_conflicts else pk)
        )
        for offset in offsets:
            conn.execute(statement, records[offset : offset + chunksize])
    else:
        for offset in offsets:
            _delete_then_insert(conn, tbl, records[offset : offset + chunksize], pk)
    return len(records)


def _delete_then_insert(conn, tbl: Table, chunk: list[dict], pk: list[str]) -> None:
    """Generic-dialect fallback: replace conflicting rows, so omitted fields revert to defaults rather than merge."""
    keys = [tuple(r[k] for k in pk) for r in chunk]
    conn.execute(tbl.delete().where(tuple_(*[tbl.c[k] for k in pk]).in_(keys)))
    conn.execute(tbl.insert(), chunk)


class DataStore:
    """Registry-backed database facade. Missing/empty reads raise unless optional=True."""

    def __init__(self, engine: Engine | Connection) -> None:
        self.engine = engine.engine if isinstance(engine, Connection) else engine
        self._database: Engine | Connection = engine
        self._metadata = MetaData()

    # so callers write `store.NOT_NULL` without importing it
    NOT_NULL = NOT_NULL

    @contextmanager
    def transaction(self) -> Generator[DataStore]:
        """Bind generic operations to one transaction."""
        if isinstance(self._database, Connection):
            raise ValueError("Nested store transactions are not supported; reuse the bound store")
        with self.engine.begin() as conn:
            if conn.dialect.name == "sqlite":
                conn.exec_driver_sql("BEGIN IMMEDIATE")
            bound = DataStore(conn)
            bound._metadata = self._metadata
            try:
                yield bound
            except BaseException:
                self._metadata.clear()
                raise

    @staticmethod
    @contextmanager
    def temporary(tables: Sequence[SchemaTable]) -> Generator[DataStore]:
        """Create an isolated disposable SQLite store with the selected canonical schemas."""
        with tempfile.TemporaryDirectory(prefix="pea-store-") as directory:
            engine = create_engine("sqlite:///" + (Path(directory) / "source.sqlite").as_posix())
            try:
                _create_canonical(engine, tables)
                yield DataStore(engine)
            finally:
                engine.dispose()

    def insert(self, table: SchemaTable | str, df: pd.DataFrame) -> int:
        """Strictly append prepared rows to an existing table; conflicts fail and roll back."""
        if df.empty:
            return 0
        tbl = self._table(table)
        if set(df.columns) - set(tbl.c.keys()):
            raise ValueError(f"{name_of(table)}: insert contains unknown columns")
        records = _records(_coerce_temporal(df, tbl))
        with _connection(self._database, write=True) as conn:
            for offset in range(0, len(records), _CHUNK):
                conn.execute(tbl.insert(), records[offset : offset + _CHUNK])
        return len(records)

    def update(self, table: SchemaTable | str, values: Mapping[str, object], where: Where) -> int:
        """Update carried values on an explicit scope; empty tuple/OR scopes change no rows."""
        if where is None or isinstance(where, dict) and not where or not values:
            raise ValueError("update() needs values and an explicit where scope")
        if not self.exists(table):
            return 0
        tbl = self._table(table)
        if set(values) - set(tbl.c.keys()):
            raise ValueError(f"{name_of(table)}: update contains unknown columns")
        bound = _records(_coerce_temporal(pd.DataFrame([values]), tbl))[0]
        with _connection(self._database, write=True) as conn:
            return int(conn.execute(tbl.update().where(_where(tbl, where)).values(bound)).rowcount)

    def summarize(
        self,
        table: SchemaTable | str,
        by: str | Sequence[str],
        aggregations: dict[str, tuple[str, str | None]],
        *,
        count_where: dict[str, Where] | None = None,
        where: Where | None = None,
    ) -> pd.DataFrame:
        """Return grouped count/min/max values, with optional filters on named counts."""
        groups = [by] if isinstance(by, str) else list(by)
        if not aggregations or set(groups) & set(aggregations) or set(count_where or {}) - set(aggregations):
            raise ValueError("Summary groups, aliases and filtered counts must be unambiguous")
        if not self.exists(table):
            return pd.DataFrame(columns=[*groups, *aggregations])
        tbl = self._table(table)
        values = []
        for alias, (operation, column) in aggregations.items():
            if operation not in {"count", "min", "max"} or column is None and operation != "count":
                raise ValueError("Summaries support count/min/max; min/max need a column")
            aggregate = getattr(func, operation)(tbl.c[column]) if column is not None else func.count()
            if alias in (count_where or {}):
                if operation != "count":
                    raise ValueError("Only counts accept count_where")
                aggregate = aggregate.filter(_where(tbl, cast(dict[str, Where], count_where)[alias]))
            values.append(aggregate.label(alias))
        stmt = select(*(tbl.c[column] for column in groups), *values).select_from(tbl).where(_where(tbl, where))
        if groups:
            stmt = stmt.group_by(*(tbl.c[column] for column in groups))
        with _connection(self._database) as conn:
            return pd.read_sql(stmt, conn)

    def load_cohort(
        self,
        table: SchemaTable | str,
        group_column: str,
        where: Where,
        columns: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Read every member of groups touched by the seed scope, using one same-table subquery."""
        if not self.exists(table):
            return pd.DataFrame(columns=list(columns or ()))
        tbl = self._table(table)
        group = tbl.c[group_column]
        seeds = select(group).where(_where(tbl, where))
        stmt = select(*(tbl.c[column] for column in columns)) if columns else select(tbl)
        with _connection(self._database) as conn:
            return pd.read_sql(stmt.where(group.in_(seeds)), conn)

    def same_rows(self, table: SchemaTable | str, actual: pd.DataFrame, expected: pd.DataFrame) -> bool:
        """Compare caller-selected columns and keys using the actual SQL DATE/NULL binding rules."""
        columns = list(actual.columns)
        if set(columns) != set(expected.columns):
            return False
        if actual.empty and expected.empty:
            return True
        tbl = self._table(table)
        keys = list(resolve(table).pk)
        left = _records(_coerce_temporal(actual[columns].sort_values(keys), tbl))
        right = _records(_coerce_temporal(expected[columns].sort_values(keys), tbl))
        return left == right

    # -- introspection ----------------------------------------------------- #
    def _table(self, table: SchemaTable | str) -> Table:
        """Reuse SQLAlchemy schema metadata across ordinary and transaction-bound operations."""
        name = name_of(table)
        with _CREATE_LOCK:
            if name not in self._metadata.tables:
                _reflect(self._database, name, self._metadata)
            return self._metadata.tables[name]

    def exists(self, table) -> bool:
        return name_of(table) in self._metadata.tables or table_exists(self._database, name_of(table))

    def columns(self, table) -> list[str]:
        """Ordered column names, or an empty list for an absent table."""
        name = name_of(table)
        if name in self._metadata.tables:
            return list(self._metadata.tables[name].c.keys())
        inspector = inspect(self._database)
        return [column["name"] for column in inspector.get_columns(name)] if inspector.has_table(name) else []

    def row_count(self, table) -> int:
        name = name_of(table)
        return row_count(self._database, name) if self.exists(name) else 0

    def bounds(self, table, column: str | None = None) -> tuple:
        """Raw MIN/MAX of the requested or declared date column; (None, None) when missing/empty."""
        name = name_of(table)
        col = column or self._date_col(table)
        if col is None:
            raise ValueError(f"{name}: no column given and the table declares no date_col")
        if not self.exists(name):
            return (None, None)
        tbl = self._table(name)
        with _connection(self._database) as conn:
            row = conn.execute(select(func.min(tbl.c[col]), func.max(tbl.c[col]))).one()
        return (row[0], row[1])

    def max_date(self, table, column: str | None = None) -> pd.Timestamp | None:
        """Latest stored day at midnight, or None when missing/empty/unparseable so callers can rebuild."""
        name = name_of(table)
        col = column or self._date_col(table)
        if col is None or not self.exists(name):
            return None
        tbl = self._table(name)
        if col not in tbl.c:
            return None
        with _connection(self._database) as conn:
            value = conn.execute(select(func.max(tbl.c[col]))).scalar()
        if value is None:
            return None
        try:
            return cast(pd.Timestamp, pd.Timestamp(value)).normalize()
        except (TypeError, ValueError):
            return None

    def max_date_by(self, table, key_col: str, date_col: str | None = None) -> dict[str, pd.Timestamp]:
        """Per-key latest stored day in one grouped query; omit NULL keys/dates and unparseable dates."""
        name = name_of(table)
        col = date_col or self._date_col(table)
        if col is None or not self.exists(name):
            return {}
        tbl = self._table(name)
        if col not in tbl.c or key_col not in tbl.c:
            return {}
        with _connection(self._database) as conn:
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
        """One grouped key/first/last/n read, including markers; missing tables/columns return an empty typed frame."""
        name = name_of(table)
        col = date_col or self._date_col(table)
        rows: list = []
        if col is not None and self.exists(name):
            tbl = self._table(name)
            if col in tbl.c and key_col in tbl.c:
                key, day = tbl.c[key_col], tbl.c[col]
                stmt = select(key, func.min(day), func.max(day), func.count()).where(key.isnot(None))
                for where_col, value in (where or {}).items():
                    stmt = stmt.where(_predicate(tbl.c[where_col], value))
                with _connection(self._database) as conn:
                    rows = list(conn.execute(stmt.group_by(key)).all())
        df = pd.DataFrame([tuple(r) for r in rows], columns=["key", "first", "last", "n"])
        for edge in ("first", "last"):
            df[edge] = pd.to_datetime(df[edge], errors="coerce").dt.normalize()
        return df.astype({"key": str, "n": "int64"})

    def distinct(
        self, table, column: str, *, where: dict | None = None, order: str | None = None, limit: int | None = None, dropna: bool = True
    ) -> list:
        """Distinct values with optional filtering, ordering, limit and NULL removal."""
        name = name_of(table)
        if not self.exists(name):
            return []
        tbl = self._table(name)
        stmt = build_select(tbl, where=where, distinct_on=column, order_by=column if order else None, descending=(order == "desc"), limit=limit)
        with _connection(self._database) as conn:
            values = [r[0] for r in conn.execute(stmt).all()]
        return [v for v in values if v is not None] if dropna else values

    def load(
        self,
        table: SchemaTable | str,
        columns: Sequence[str] | None = None,
        limit: int | None = None,
        where: Where | None = None,
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
        """Read a server-filtered projection, excluding empty-filing markers unless requested.

        Missing/empty results raise; optional reads return None. Date bounds are inclusive.
        """
        name = name_of(table)
        tbl, cols = self._read_setup(table, columns, project)
        if tbl is None:
            if optional:
                return None
            raise TableMissingError(name)
        stmt = build_select(
            tbl,
            columns=cols,
            limit=limit,
            where=where,
            exclude=self._marker(table, markers),
            since=since,
            until=until,
            date_col=date_col or self._date_col(table),
            order_by=order_by,
            descending=descending,
        )
        with _connection(self._database) as conn:
            df = pd.read_sql(stmt, conn)

        if df.empty and not optional:
            raise TableEmptyError(name, where)
        return None if df.empty else df

    def iter_load(
        self,
        table: SchemaTable | str,
        *,
        chunksize: int = 400_000,
        columns: Sequence[str] | None = None,
        project: bool = False,
        where: Where | None = None,
        since: object = None,
        until: object = None,
        date_col: str | None = None,
        order_by: str | Sequence[str] | None = None,
        markers: bool = False,
    ) -> Generator[pd.DataFrame]:
        """Stream an ordered projection with a server-side cursor and normal marker rules.

        Exhaust or close the iterator to release its held connection.
        """
        name = name_of(table)
        projection_error = (
            f"iter_load({name}) needs an explicit `columns=` or `project=True`: "
            "streaming an unprojected wide table defeats the purpose (cube is 574 cols)"
        )
        if not project:
            columns = list(columns) if columns is not None else None
            if not columns:
                raise ValueError(projection_error)
        tbl, cols = self._read_setup(table, columns, project)
        if not cols:
            raise ValueError(projection_error)
        if tbl is None:
            return
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
        with _connection(self._database) as conn:
            options = conn.get_execution_options()
            conn.execution_options(stream_results=True, yield_per=chunksize, max_row_buffer=chunksize)
            try:
                yield from pd.read_sql(stmt, conn, chunksize=chunksize)
            finally:
                conn.execution_options(
                    stream_results=options.get("stream_results", False),
                    yield_per=options.get("yield_per"),
                    max_row_buffer=options.get("max_row_buffer", 1000),
                )

    # -- helpers ----------------------------------------------------------- #
    @staticmethod
    def _date_col(table) -> str | None:
        """The table's declared incremental date column, or None for a bare name."""
        try:
            return resolve(table).date_col
        except KeyError:
            return None

    @staticmethod
    def _marker(table: SchemaTable | str, markers: bool) -> tuple[str, object] | None:
        """The (column, sentinel) a consumer read excludes; None when `markers` or undeclared."""
        if markers:
            return None
        try:
            return resolve(table).empty_marker
        except KeyError:
            return None

    def _read_setup(self, table: SchemaTable | str, columns: Sequence[str] | None, project: bool) -> tuple[Table | None, Sequence[str] | None]:
        """Resolve projection and selection from the same verified schema metadata."""
        if columns is not None and project:
            raise ValueError("pass either columns= or project=True, not both")
        columns = list(columns) if columns is not None else None
        tbl = self._table(table) if self.exists(table) else None
        if not project:
            return tbl, columns
        cols, required_missing, optional_missing = projection_report(table, (list(tbl.c.keys()) or None) if tbl is not None else None)
        if required_missing:
            logger.warning("%s is missing REQUIRED column(s) %s -> the features that need them will be empty", name_of(table), required_missing)
        elif optional_missing:
            logger.info("%s has no %s (optional) -> those features are skipped", name_of(table), optional_missing)
        return tbl, cols

    def save(self, table: SchemaTable | str, df: pd.DataFrame, pk: list[str] | None = None, *, ignore_all_conflicts: bool = False) -> int:
        """Upsert carried fields; omitted fields remain unchanged and explicit NULLs overwrite."""
        name = name_of(table)
        if df is None or df.empty:
            if not isinstance(self._database, Connection):
                logger.warning(f"save() skipped: table '{name}' is empty")
            return 0
        if not isinstance(self._database, Connection):
            self._prepare_write(name, df)
        tbl = self._table(table)
        if set(df.columns) - set(tbl.c.keys()):
            raise ValueError(f"{name}: save contains unknown columns")
        with _connection(self._database, write=True) as conn:
            return _upsert_records(
                conn, tbl, _records(_coerce_temporal(df, tbl)), pk or list(resolve(table).pk), ignore_all_conflicts=ignore_all_conflicts
            )

    def _prepare_write(self, name: str, df: pd.DataFrame) -> None:
        """Keep frame-created/dynamic layouts generic; existing managed schemas never evolve implicitly."""
        ensure_table(self._database, name, df)
        if any(table.name == name for table in MANAGED):
            unknown = set(df.columns) - set(self.columns(name))
            if unknown:
                raise ValueError(f"{name}: managed schema lacks columns {sorted(unknown)}")
        elif ensure_columns(self._database, name, df):
            self._metadata.clear()

    def bulk_seed(self, table, df: pd.DataFrame) -> int:
        """Append to an established schema via COPY/fallback; reject unknown columns instead of silently dropping them."""
        name = name_of(table)
        if df is not None and not df.empty and self.exists(name):
            existing = set(self.columns(name))
            unknown = [c for c in df.columns if c not in existing]
            if unknown:
                raise ValueError(
                    f"bulk_seed({name}): frame has {len(unknown)} column(s) the table does "
                    f"not have and COPY would silently drop: {unknown[:8]}. Use an explicit "
                    f"ensure_columns() after checking the declared schema."
                )
        return copy_load(self._database, df, name)

    def append_tail(self, table, df: pd.DataFrame, cutoff, *, inclusive: bool = False, date_col: str | None = None) -> int:
        """Delete the tail after cutoff (inclusive when requested), then append; empty input still deletes the tail."""
        name = name_of(table)
        col = date_col or self._date_col(table)
        if col is None:
            raise ValueError(f"{name}: append_tail needs a date column")
        tbl = self._table(name)
        # A day boundary also preserves the cutoff day on text-typed SQLite dates.
        boundary = cast(pd.Timestamp, pd.Timestamp(cutoff)).normalize()
        if not inclusive:
            boundary += pd.Timedelta(days=1)
        with _connection(self._database, write=True) as conn:
            conn.execute(text(f'DELETE FROM "{name}" WHERE "{col}" >= :cut'), {"cut": _bind_date(tbl.c[col], boundary)})
        if df is None or df.empty:
            return 0
        if isinstance(self._database, Connection):
            return self.insert(table, df)
        df.to_sql(name, self._database, if_exists="append", index=False)
        return len(df)

    def delete(self, table: SchemaTable | str, where: Where) -> int:
        """Delete an explicit row scope; empty tuple/OR scopes match nothing."""
        if where is None or isinstance(where, dict) and not where:
            raise ValueError("delete() needs a where filter; use replace() to rewrite a table")
        if not self.exists(table):
            return 0
        tbl = self._table(table)
        with _connection(self._database, write=True) as conn:
            return int(conn.execute(tbl.delete().where(_where(tbl, where))).rowcount)

    def replace(self, table: SchemaTable | str, df: pd.DataFrame, chunksize: int = 200_000) -> int:
        """Rebuild with chunked COPY: DELETE managed tables, DROP/recreate unmanaged tables to remove stale columns.

        DROP, preparation, DELETE and COPY chunks retain their separate commits; failures can leave a partial rebuild."""

        if isinstance(self._database, Connection):
            if resolve(table).is_unmanaged:
                self.drop(table)
                ensure_table(self._database, name_of(table), df)
            tbl = self._table(table)
            self._database.execute(tbl.delete())
            for offset in range(0, len(df), chunksize):
                self.bulk_seed(table, df.iloc[offset : offset + chunksize])
            return len(df)
        if df is None or df.empty:
            return 0
        name = name_of(table)
        if resolve(table).is_unmanaged:
            self.drop(name)
        self._prepare_write(name, df)
        with _connection(self._database, write=True) as conn:
            conn.execute(text(f'DELETE FROM "{name}"'))
        for i in range(0, len(df), chunksize):
            copy_load(self._database, df.iloc[i : i + chunksize], name)
        return len(df)

    def ensure_columns(self, table: SchemaTable | str, df: pd.DataFrame) -> list[str]:
        """Add newly observed columns on PostgreSQL/SQLite and return their names; never drop fields."""
        name = name_of(table)
        try:
            return ensure_columns(self._database, name, df)
        finally:
            self._metadata.clear()

    def drop(self, table: SchemaTable | str) -> None:
        """Drop a registered or bare table name and invalidate its prepared metadata."""
        name = name_of(table)
        self._metadata.clear()
        with _connection(self._database, write=True) as conn:
            conn.execute(text(f'DROP TABLE IF EXISTS "{name}"'))
