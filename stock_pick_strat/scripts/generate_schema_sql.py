"""
Regenerate sql/schema.sql from the current flat files in data/, falling back to the
LIVE DB for DB-only tables that have no flat source (e.g. macro_asset_prices) so their
real DDL is emitted. Needs the Postgres container up for the DB-introspection fallback.

Run:
    python -m scripts.generate_schema_sql
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_store.schema_sql import write_schema_sql


def main() -> None:
    engine = None
    try:                                      # DB fallback for source-missing tables
        from src.context import get_config_context
        _, context = get_config_context("./configs", use_cache=False, save=False)
        engine = context.store.engine
    except Exception as exc:                  # noqa: BLE001 - DB optional; flat files still work
        print(f"(no DB engine: {exc}); generating from flat files only")
    out = write_schema_sql(ROOT / "data", ROOT / "sql" / "schema.sql", engine=engine)
    print(f"Wrote schema DDL -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
