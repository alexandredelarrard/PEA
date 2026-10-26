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
from sqlalchemy import Connection, Engine, MetaData, Select, Table, and_, case, create_engine, false, func, inspect, or_, select, text, true, tuple_
from sqlalchemy import types as sqltypes
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.sql.elements import ColumnElement

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

_CHUNK = 10_000
_CREATE_LOCK = threading.Lock()
logger = logging.getLogger(__name__)
_IDENTITY_TABLES = (Tables.sp500_tickers, Tables.entity_lineage, Tables.symbol_tenure, Tables.security_master)


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
    df = cast(pd.DataFrame, df[[c for c in df.columns if c in tbl.c]])  # only real columns
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

    @contextmanager
    def transaction(self, lock_key: str | None = None) -> Generator[DataStore]:
        """Bind generic operations to one transaction, optionally serializing a caller-owned scope."""
        if isinstance(self._database, Connection):
            raise ValueError("Nested store transactions are not supported; reuse the bound store")
        with self.engine.begin() as conn:
            if conn.dialect.name == "sqlite":
                conn.exec_driver_sql("BEGIN IMMEDIATE")
            elif lock_key is not None:
                if conn.dialect.name != "postgresql":
                    raise ValueError("Scoped transaction locks require PostgreSQL or SQLite")
                conn.execute(text("SELECT pg_advisory_xact_lock(hashtext(:scope))"), {"scope": lock_key})
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
        chunksize: int = 200_000,
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
        """Resolve projection and selection from the same operation-local schema."""
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

    @contextmanager
    def _institutional_transaction(self) -> Generator[Connection]:
        """Serialize source, identity and certification commits without permanent metadata."""
        with _connection(self.engine, write=True) as conn:
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
        return pd.read_sql(build_select(tbl, where=where), conn)

    @staticmethod
    def _institutional_insert(conn: Connection, spec: SchemaTable, frame: pd.DataFrame) -> int:
        """Upsert a prepared bounded frame using this transaction, preserving omitted fields."""
        if frame.empty:
            return 0
        tbl = _reflect(conn, spec.name)
        records = _records(_coerce_temporal(frame, tbl))
        return _upsert_records(conn, tbl, records, list(spec.pk), ignore_all_conflicts=True)

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
        # Establish the proof column before schema preparation; retained rows are reset below.
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

    def _institutional_identity(self, conn: Connection) -> dict[SchemaTable, pd.DataFrame]:
        """Capture the closed identity cohort on the caller's guarded connection."""
        return {spec: self._institutional_select(conn, spec) for spec in _IDENTITY_TABLES}

    def _assert_institutional_identity(self, conn: Connection, expected_identity: dict[SchemaTable, pd.DataFrame]) -> None:
        """The mapping used outside the transaction must still be the exact guarded identity cohort."""
        if set(expected_identity) != set(_IDENTITY_TABLES):
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
        tbl, _ = self._read_setup(Tables.insider_transactions, None, False)
        if tbl is None or not {"inventory_checked_through", "fetched_at", "ticker"}.issubset(tbl.c.keys()):
            return result
        checked = tbl.c.inventory_checked_through
        statement = select(tbl.c.ticker, func.min(checked)).where(tbl.c.ticker.in_(list(result.index))).group_by(tbl.c.ticker)
        statement = statement.having(
            func.min(checked) == func.max(checked), func.count(checked) == func.count(), func.count(tbl.c.fetched_at) == func.count()
        )
        with _connection(self.engine) as conn:
            for ticker, boundary in conn.execute(statement):
                result.loc[ticker] = pd.Timestamp(boundary)
        return result

    def short_volume_source_cohort(self, companies: Sequence[str], symbols: Sequence[str]) -> pd.DataFrame:
        """Read complete touched-symbol cohorts once, including retained other-owner dates."""
        if not self.exists(Tables.sec_short_volume_security):
            return pd.DataFrame()
        raw = _reflect(self.engine, Tables.sec_short_volume_security.name)
        owned_symbols = select(raw.c.source_symbol).where(raw.c.ticker.in_(list(companies)))
        statement = select(raw).where(raw.c.source_symbol.in_(owned_symbols) | raw.c.source_symbol.in_(list(symbols)))
        with _connection(self.engine) as conn:
            return pd.read_sql(statement, conn)

    def fails_source_periods(self) -> tuple[set[str], set[str]]:
        """Compact raw inventory and period keys whose raw or derived payload needs reconciliation."""
        pending = set(map(str, self.distinct(Tables.sec_fails_to_deliver, "period", where={"fails_quantity": None})))
        if not self.exists(Tables.sec_fails_to_deliver_security):
            return set(), pending
        raw = _reflect(self.engine, Tables.sec_fails_to_deliver_security.name)
        statement = select(raw.c.period, func.count(), func.count(raw.c.security_id)).group_by(raw.c.period)
        with _connection(self.engine) as conn:
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
        with _connection(self.engine) as conn:
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
            cohorts = self._institutional_identity(conn)
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
            cohorts = self._institutional_identity(conn)
            old = self._institutional_subset(cohorts[spec], {"ticker": frame["ticker"].tolist()})
            changed = pd.concat([old.reindex(columns=["ticker", "cik"]), frame[["ticker", "cik"]]]).drop_duplicates(keep=False)
            if not changed.empty:
                self._invalidate_institutional_proof(conn, self._institutional_identity_tickers(conn, changed, cohorts), True)
            return self._institutional_insert(conn, spec, frame)

    # -- writes ------------------------------------------------------------ #
    def save(self, table: SchemaTable | str, df: pd.DataFrame, pk: list[str] | None = None, *, ignore_all_conflicts: bool = False) -> int:
        """Upsert `df` into `table` on its PK, creating the table if needed."""
        name = name_of(table)
        if df is None or df.empty:
            if not isinstance(self._database, Connection):
                logger.warning(f"save() skipped: table '{name}' is empty")
            return 0

        if isinstance(self._database, Connection):
            tbl = self._table(name)
            if set(df.columns) - set(tbl.c.keys()):
                raise ValueError(f"{name}: save contains unknown columns")
            return _upsert_records(
                self._database, tbl, _records(_coerce_temporal(df, tbl)), pk or list(resolve(table).pk), ignore_all_conflicts=ignore_all_conflicts
            )
        if name == Tables.sp500_tickers.name and "cik" in df:
            return self._save_institutional_roster(df)
        if pk is None:
            pk = list(resolve(table).pk)
        df = self._prepare_write(name, df)
        return upsert_dataframe(self._database, df, name, pk)

    def _prepare_write(self, name: str, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare ordinary save/rebuild layouts while the caller holds any 13F lock."""
        ensure_table(self._database, name, df)
        if ensure_columns(self._database, name, df):
            self._metadata.clear()
        return df

    def bulk_seed(self, table, df: pd.DataFrame) -> int:
        """Append to an established schema via COPY/fallback; reject unknown columns instead of silently dropping them."""
        name = name_of(table)
        if df is not None and not df.empty and self.exists(name):
            existing = set(self.columns(name))
            unknown = [c for c in df.columns if c not in existing]
            if unknown:
                raise ValueError(
                    f"bulk_seed({name}): frame has {len(unknown)} column(s) the table does "
                    f"not have and COPY would silently drop: {unknown[:8]}. Use save() or "
                    f"replace(), which evolve the schema."
                )
        return copy_load(self._database, df, name)

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
        df = self._prepare_write(name, df)
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
