"""
store.py  (src/data_store/store.py)
-----------------------------------
Generic read / upsert / incremental helpers over the SQLAlchemy engine. Used by
the parquet->DB migrator now and (Phase 2) by the fetchers for dual-write and
incremental "only fetch missing dates" logic.

  * table_exists(engine, name)                 -> bool
  * read_table(engine, name, ...)              -> DataFrame
  * upsert_dataframe(engine, df, name, pk)     -> rows written (INSERT .. ON CONFLICT)
  * existing_dates(engine, name, date_col, ..) -> {ticker: max_date} (incremental cutoff)

The upsert is dialect-aware: Postgres and SQLite both use native
INSERT .. ON CONFLICT DO UPDATE; any other dialect falls back to
delete-by-PK-then-insert. This lets the same code path run against the
production Postgres container and a throwaway SQLite DB for offline checks.
"""
from __future__ import annotations

import csv
import io

import pandas as pd
from sqlalchemy import Engine, MetaData, Table, func, inspect, select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy import and_, tuple_

_CHUNK = 5_000


def table_exists(engine: Engine, name: str) -> bool:
    return inspect(engine).has_table(name)


def _reflect(engine: Engine, name: str) -> Table:
    return Table(name, MetaData(), autoload_with=engine)


def ensure_table(engine: Engine, name: str, df: pd.DataFrame) -> None:
    """Create the table (with its registry PK) if it does not exist yet, using
    the DataFrame's dtypes for column types. Needed for tables with no seed file
    yet (e.g. def14a_llm) so the first fetcher write succeeds."""
    if table_exists(engine, name):
        return
    from src.data_store.schema_registry import BY_NAME
    from src.data_store.schema_sql import _table_ddl
    spec = BY_NAME[name]
    with engine.begin() as conn:
        for stmt in _table_ddl(spec, df).split(";"):
            if stmt.strip():
                conn.execute(text(stmt))


def copy_load(engine: Engine, df: pd.DataFrame, name: str) -> int:
    """Fast bulk insert via Postgres COPY (CSV). For the one-time seed of empty
    tables; no conflict handling. Falls back to upsert on non-Postgres engines
    or for frames carrying array/list columns (COPY can't format those)."""
    if df is None or df.empty:
        return 0
    has_list = any(df[c].apply(lambda v: isinstance(v, (list, tuple))).any()
                   for c in df.columns)
    if engine.dialect.name != "postgresql" or has_list:
        from src.data_store.schema_registry import BY_NAME
        return upsert_dataframe(engine, df, name, list(BY_NAME[name].pk))

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


def read_table(engine: Engine, name: str, columns: list[str] | None = None,
               limit: int | None = None) -> pd.DataFrame:
    tbl = _reflect(engine, name)
    cols = [tbl.c[c] for c in columns] if columns else [tbl]
    stmt = select(*cols)
    if limit is not None:
        stmt = stmt.limit(limit)
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


def upsert_dataframe(engine: Engine, df: pd.DataFrame, name: str,
                     pk: list[str], chunksize: int = _CHUNK) -> int:
    """INSERT the frame, updating non-PK columns on PK conflict. Returns the
    number of rows sent. No-op for an empty frame."""
    if df is None or df.empty:
        return 0
    tbl = _reflect(engine, name)
    df = df[[c for c in df.columns if c in tbl.c]]     # only real columns
    records = _records(df)
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


def existing_dates(engine: Engine, name: str, date_col: str,
                   ticker_col: str | None = "ticker") -> dict:
    """Incremental cutoff helper: the max `date_col` already stored.

    With a ticker column -> {ticker: max_date} so each name is refreshed only
    after its own latest stored date. Without one (e.g. macro) -> {"__all__":
    max_date}. Returns {} when the table does not exist yet (fetch everything).
    """
    if not table_exists(engine, name):
        return {}
    tbl = _reflect(engine, name)
    with engine.connect() as conn:
        if ticker_col and ticker_col in tbl.c:
            rows = conn.execute(
                select(tbl.c[ticker_col], func.max(tbl.c[date_col]))
                .group_by(tbl.c[ticker_col])
            ).all()
            return {t: d for t, d in rows}
        mx = conn.execute(select(func.max(tbl.c[date_col]))).scalar()
        return {"__all__": mx} if mx is not None else {}


class DataStore:
    """DB-backed data access facade used throughout the pipeline in place of
    parquet files. One instance lives on the Context (`context.store`).

    Table names are the logical names in the schema registry. `save` upserts on
    the registry PK (idempotent, so re-running a fetcher merges rather than
    duplicating); `load` returns an empty frame when the table is absent/empty
    so callers can treat "no data yet" uniformly.
    """

    def __init__(self, engine: Engine):
        self.engine = engine

    # -- reads ------------------------------------------------------------- #
    def exists(self, name: str) -> bool:
        return table_exists(self.engine, name)

    def load(self, name: str, columns: list[str] | None = None,
             limit: int | None = None) -> pd.DataFrame:
        if not self.exists(name):
            return pd.DataFrame(columns=columns or [])
        return read_table(self.engine, name, columns=columns, limit=limit)

    def row_count(self, name: str) -> int:
        return row_count(self.engine, name) if self.exists(name) else 0

    def existing_dates(self, name: str, date_col: str,
                       ticker_col: str | None = "ticker") -> dict:
        return existing_dates(self.engine, name, date_col, ticker_col)

    # -- writes ------------------------------------------------------------ #
    def save(self, name: str, df: pd.DataFrame, pk: list[str] | None = None) -> int:
        """Upsert `df` into `name` on its PK, creating the table if needed."""
        if df is None or df.empty:
            return 0
        if pk is None:
            from src.data_store.schema_registry import BY_NAME
            pk = list(BY_NAME[name].pk)
        ensure_table(self.engine, name, df)
        return upsert_dataframe(self.engine, df, name, pk)

    def bulk_seed(self, name: str, df: pd.DataFrame) -> int:
        """Fast COPY insert for the initial seed of empty tables."""
        return copy_load(self.engine, df, name)

    def replace(self, name: str, df: pd.DataFrame, chunksize: int = 200_000) -> int:
        """Full-rebuild semantics for aggregate tables (cube, predictions): empty
        the table, then fast-COPY the frame back in chunks (bounded memory)."""
        if df is None or df.empty:
            return 0
        ensure_table(self.engine, name, df)
        with self.engine.begin() as conn:
            conn.execute(text(f'DELETE FROM "{name}"'))
        for i in range(0, len(df), chunksize):
            copy_load(self.engine, df.iloc[i:i + chunksize], name)
        return len(df)
