"""
Regenerate sql/schema.sql from the current flat files in data/.

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
    out = write_schema_sql(ROOT / "data", ROOT / "sql" / "schema.sql")
    print(f"Wrote schema DDL -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
