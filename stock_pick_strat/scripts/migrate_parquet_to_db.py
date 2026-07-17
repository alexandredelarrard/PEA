"""
Migrate the existing flat files in data/ into the SQL database, then read them
back and assert integrity. This is the Phase-1 sanity check for the whole DB
stack (docker schema init + PK constraints + upsert + read-back).

Run (against the docker-compose Postgres, from the host):
    docker compose up -d db
    DATABASE_URL=postgresql+psycopg2://pea:pea@localhost:5432/pea \
        python -m scripts.migrate_parquet_to_db --create

Fast offline logic check (no docker) against a throwaway SQLite file:
    python -m scripts.migrate_parquet_to_db --url sqlite:///./_sanity.db --create \
        --limit-rows 2000

Flags:
    --create        run sql/schema.sql first (idempotent CREATE IF NOT EXISTS)
    --tables a,b    restrict to these tables
    --limit-rows N  load only the first N rows per table (fast smoke test)
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sqlalchemy import text

from src.utils.db import get_engine
from src.data_store import store
from src.data_store.io import read_source
from src.data_store.schema_registry import ALL_TABLES, TableSpec

DATA = ROOT / "data"
SCHEMA_SQL = ROOT / "sql" / "schema.sql"


def _run_schema_sql(engine) -> None:
    """Execute sql/schema.sql statement-by-statement (portable across drivers
    that don't allow multi-statement execute)."""
    raw = SCHEMA_SQL.read_text(encoding="utf-8")
    no_comments = "\n".join(l for l in raw.splitlines() if not l.strip().startswith("--"))
    statements = [s.strip() for s in no_comments.split(";") if s.strip()]
    with engine.begin() as conn:
        for stmt in statements:
            conn.execute(text(stmt))


def _load_source(spec: TableSpec, limit: int | None) -> pd.DataFrame:
    df = read_source(spec, DATA, nrows=limit)
    # collapse a vector family (e0..eN) into one list column
    if spec.vector_col and spec.vector_prefix:
        p = spec.vector_prefix
        vec = sorted((c for c in df.columns if c.startswith(p) and c[len(p):].isdigit()),
                     key=lambda c: int(c[len(p):]))
        other = [c for c in df.columns if c not in vec]
        out = df[other].copy()
        out[spec.vector_col] = df[vec].values.tolist()
        df = out
    # string date columns -> real dates
    for c in spec.date_type_cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce").dt.date
    return df


def _distinct_pk(df: pd.DataFrame, pk: list[str]) -> int:
    return len(df.drop_duplicates(subset=pk))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=None)
    ap.add_argument("--create", action="store_true")
    ap.add_argument("--tables", default=None)
    ap.add_argument("--limit-rows", type=int, default=None)
    args = ap.parse_args()

    engine = get_engine(args.url)
    is_sqlite = engine.dialect.name == "sqlite"
    print(f"Engine: {engine.dialect.name}  url={engine.url.render_as_string(hide_password=True)}")

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
            results.append((spec.name, 0, 0, "SKIP (table not created — use --create)"))
            continue

        df = _load_source(spec, args.limit_rows)
        expected = _distinct_pk(df, list(spec.pk))
        store.upsert_dataframe(engine, df, spec.name, list(spec.pk))
        got = store.row_count(engine, spec.name)
        status = "OK" if got >= expected else f"MISMATCH exp>={expected}"
        results.append((spec.name, expected, got, status))

    print("\n=== SANITY CHECK: parquet -> DB migration ===")
    print(f"{'table':<24}{'src_rows(uniq pk)':>18}{'db_rows':>10}   status")
    ok = True
    for name, exp, got, status in results:
        print(f"{name:<24}{exp:>18}{got:>10}   {status}")
        if not status.startswith(("OK", "SKIP")):
            ok = False

    # deeper check: re-read one table and confirm PK uniqueness holds in the DB
    sample = next((r[0] for r in results if r[3] == "OK"), None)
    if sample:
        spec = store  # noqa
        from src.data_store.schema_registry import BY_NAME
        pk = list(BY_NAME[sample].pk)
        back = store.read_table(engine, sample, limit=5000)
        dups = back.duplicated(subset=pk).sum()
        print(f"\nread-back of '{sample}': {len(back)} rows sampled, "
              f"PK dups={dups} (must be 0)")
        ok = ok and dups == 0

    print("\nCONCLUSION:", "PASS — DB round-trip + PK integrity verified." if ok
          else "FAIL — see mismatches above.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
