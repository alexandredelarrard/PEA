"""
recreate_fundamentals_tables.py  (scripts/)
--------------------------------------------------------------------------------------------
DROP and re-CREATE the four fundamentals tables from `sql/schema.sql`.

Needed because `sql/schema.sql` is applied only when Postgres INITIALISES a volume. On a
live volume `store.save` creates a missing table from the FIRST frame it is handed, via
`ensure_table`'s dtype inference -- which is how an all-None column once became TEXT and every
later ticker's number was stored as a string. A schema change to any of these tables therefore
has to be applied deliberately, and this is the deliberate application.

DESTRUCTIVE. Every row in the named tables is deleted; they are rebuilt by
`fundamentals-facts -F` (network) and `fundamentals-history-sec` (local). Nothing else is touched.

    "$PY" scripts/recreate_fundamentals_tables.py --dry-run
    "$PY" scripts/recreate_fundamentals_tables.py --yes
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.context import get_config_context
from src.data_store.ddl import existing_blocks

#: Facts first in the DROP, history last in the CREATE -- there are no FKs between them, but
#: the order is the dependency order a reader expects, and a partial failure then leaves the
#: upstream table missing rather than a downstream one silently empty.
TABLES = ("fundamentals_facts", "fundamentals_history", "fundamentals_reason_codes",
          "fundamentals_employees")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yes", action="store_true", help="actually drop and recreate")
    parser.add_argument("--dry-run", action="store_true", help="print the SQL and the counts")
    args = parser.parse_args()

    blocks = existing_blocks(Path("sql/schema.sql").read_text(encoding="utf-8"))
    missing = [t for t in TABLES if t not in blocks]
    assert not missing, f"sql/schema.sql has no CREATE TABLE for {missing}"

    _, context = get_config_context("./configs", use_cache=False, save=False)
    engine = context.store.engine
    with engine.begin() as conn:
        for table in TABLES:
            try:
                n = conn.execute(text(f'SELECT count(*) FROM "{table}"')).scalar()
            except Exception:                                    # table not there yet
                n = None
            print(f"  {table:32s} {'absent' if n is None else f'{n:,} rows'}")

    if args.dry_run or not args.yes:
        print("\n--- SQL that WOULD run ---")
        for table in TABLES:
            print(f'DROP TABLE IF EXISTS "{table}";')
        for table in TABLES:
            print(blocks[table])
        print("\nNothing executed. Pass --yes to apply.")
        return

    with engine.begin() as conn:
        for table in TABLES:
            conn.execute(text(f'DROP TABLE IF EXISTS "{table}" CASCADE'))
            print(f"  DROPPED {table}")
        for table in TABLES:
            for statement in blocks[table].split(";"):
                if statement.strip():
                    conn.execute(text(statement))
            print(f"  CREATED {table}")
    print("\nDone. Now: `fundamentals-facts -F` per chunk, then `fundamentals-history-sec`.")


if __name__ == "__main__":
    main()
