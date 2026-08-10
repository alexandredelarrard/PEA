"""
Regenerate sql/schema.sql from `src/data_store/schema.py` + the LIVE database schema.

REQUIRES THE POSTGRES CONTAINER UP. The old generator preferred flat files under data/ and
used the DB only as a fallback; the parquet->DB migration is finished and its migrator is
deleted, so the DB is now the only source of column types. A table that does not exist in
the database yet is emitted as a `-- SKIPPED` comment rather than guessed at.

Run:
    python -m scripts.generate_schema_sql
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_store.ddl import generate_schema_sql
from src.data_store.store import reflect_all


def main() -> None:
    from src.context import get_config_context

    _, context = get_config_context("./configs", use_cache=False, save=False)
    reflected = reflect_all(context.store.engine)
    out = ROOT / "sql" / "schema.sql"
    out.parent.mkdir(parents=True, exist_ok=True)
    # Pass the current file in so a table this database does not have keeps its existing
    # DDL instead of being dropped from the schema (see ddl.generate_schema_sql).
    previous = out.read_text(encoding="utf-8") if out.exists() else None
    out.write_text(generate_schema_sql(reflected, previous=previous), encoding="utf-8")
    print(f"Wrote schema DDL for {len(reflected)} live tables -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
