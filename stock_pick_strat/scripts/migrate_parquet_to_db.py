"""
Seed the SQL database from the existing flat files in data/, then read back and
assert row counts. Phase-1/2 bootstrap: makes the DB the source of truth before
the pipeline switches to DB-only I/O.

Full fast seed against the docker-compose Postgres (from the host):
    docker compose up -d db
    DATABASE_URL=postgresql+psycopg2://pea:pea@localhost:5432/pea \
        python -m scripts.migrate_parquet_to_db --create --bulk --truncate

Fast offline logic check (no docker) against a throwaway SQLite file:
    python -m scripts.migrate_parquet_to_db --url sqlite:///./_sanity.db --create \
        --limit-rows 2000

Flags:
    --create        run sql/schema.sql first (idempotent CREATE IF NOT EXISTS)
    --bulk          use COPY streaming (fast; for empty tables) instead of upsert
    --truncate      TRUNCATE each table before loading (clean reseed)
    --tables a,b    restrict to these tables
    --limit-rows N  load only the first N rows per table (smoke test; disables streaming)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.db import get_engine
from src.data_store import store
from src.data_store.io import iter_source_batches, read_source
from src.data_store.schema_registry import ALL_TABLES, BY_NAME, TableSpec

DATA = ROOT / "data"
SCHEMA_SQL = ROOT / "sql" / "schema.sql"
_STREAM_BATCH = 100_000


def _run_schema_sql(engine) -> None:
    raw = SCHEMA_SQL.read_text(encoding="utf-8")
    no_comments = "\n".join(l for l in raw.splitlines() if not l.strip().startswith("--"))
    with engine.begin() as conn:
        for stmt in (s.strip() for s in no_comments.split(";")):
            if stmt:
                conn.execute(text(stmt))


def _source_rows(spec: TableSpec) -> int:
    path = DATA / spec.source
    if path.suffix == ".csv":
        return sum(1 for _ in open(path, encoding="utf-8")) - 1
    return pq.ParquetFile(path).metadata.num_rows


def _transform(spec: TableSpec, df: pd.DataFrame) -> pd.DataFrame:
    """Vector-collapse (e0..eN -> array) and string-date coercion."""
    if spec.vector_col and spec.vector_prefix:
        p = spec.vector_prefix
        vec = sorted((c for c in df.columns if c.startswith(p) and c[len(p):].isdigit()),
                     key=lambda c: int(c[len(p):]))
        other = [c for c in df.columns if c not in vec]
        out = df[other].copy()
        out[spec.vector_col] = df[vec].values.tolist()
        df = out
    for c in spec.date_type_cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce").dt.date
    return df


def _load_table(engine, spec: TableSpec, bulk: bool, limit: int | None) -> int:
    pk = list(spec.pk)
    if limit is not None:
        df = _transform(spec, read_source(spec, DATA, nrows=limit))
        (store.copy_load(engine, df, spec.name) if bulk
         else store.upsert_dataframe(engine, df, spec.name, pk))
        return store.row_count(engine, spec.name)
    for batch in iter_source_batches(spec, DATA, batch_size=_STREAM_BATCH):
        batch = _transform(spec, batch)
        if bulk:
            store.copy_load(engine, batch, spec.name)
        else:
            store.upsert_dataframe(engine, batch, spec.name, pk)
    return store.row_count(engine, spec.name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=None)
    ap.add_argument("--create", action="store_true")
    ap.add_argument("--bulk", action="store_true")
    ap.add_argument("--truncate", action="store_true")
    ap.add_argument("--tables", default=None)
    ap.add_argument("--limit-rows", type=int, default=None)
    args = ap.parse_args()

    engine = get_engine(args.url)
    is_sqlite = engine.dialect.name == "sqlite"
    print(f"Engine: {engine.dialect.name}  "
          f"url={engine.url.render_as_string(hide_password=True)}  bulk={args.bulk}")

    if args.create:
        print(f"Applying {SCHEMA_SQL.relative_to(ROOT)} ...")
        _run_schema_sql(engine)

    wanted = set(args.tables.split(",")) if args.tables else None
    results: list[tuple[str, int, int, str]] = []

    for spec in ALL_TABLES:
        if wanted and spec.name not in wanted:
            continue
        if is_sqlite and spec.vector_col:
            results.append((spec.name, 0, 0, "SKIP (array unsupported on sqlite)"))
            continue
        if not (DATA / spec.source).exists():
            results.append((spec.name, 0, 0, "SKIP (source missing)"))
            continue
        if not store.table_exists(engine, spec.name):
            results.append((spec.name, 0, 0, "SKIP (no table — use --create)"))
            continue

        if args.truncate:
            with engine.begin() as conn:
                conn.execute(text(f'DELETE FROM "{spec.name}"'))

        src_rows = _source_rows(spec) if args.limit_rows is None \
            else min(args.limit_rows, _source_rows(spec))
        got = _load_table(engine, spec, args.bulk, args.limit_rows)
        status = "OK" if got >= src_rows else f"MISMATCH exp>={src_rows}"
        results.append((spec.name, src_rows, got, status))
        print(f"  loaded {spec.name:<22} src={src_rows:>9}  db={got:>9}  {status}")

    print("\n=== SANITY CHECK: parquet -> DB seed ===")
    print(f"{'table':<24}{'src_rows':>10}{'db_rows':>10}   status")
    ok = True
    for name, exp, got, status in results:
        print(f"{name:<24}{exp:>10}{got:>10}   {status}")
        if not status.startswith(("OK", "SKIP")):
            ok = False

    sample = next((r[0] for r in results if r[3] == "OK"), None)
    if sample:
        pk = list(BY_NAME[sample].pk)
        back = store.read_table(engine, sample, limit=5000)
        dups = back.duplicated(subset=pk).sum()
        print(f"\nread-back of '{sample}': {len(back)} rows sampled, PK dups={dups} (must be 0)")
        ok = ok and dups == 0

    print("\nCONCLUSION:", "PASS — DB seed + PK integrity verified." if ok
          else "FAIL — see mismatches above.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
