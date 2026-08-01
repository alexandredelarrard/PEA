"""
One-time DB migration: rename two existing tables to match the form-dispatch
registry's logical names (src/data_extract/utils/common/form_registry.py):

    institutional_holdings -> sec13f_hr
    sec_8k_items            -> sec_8k

`ALTER TABLE ... RENAME TO ...` is a metadata-only operation in Postgres (no data
rewrite, no downtime, fully reversible by renaming back) -- this does NOT touch
rows, indexes, or column definitions. Idempotent: skips a rename whose source
table is already absent (e.g. already renamed, or a fresh DB that was seeded
directly with the new names).

Run once, alongside the code deploy that changes schema_registry.py's table
names for these two tables:

    python -m scripts.rename_form_dispatch_tables
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sqlalchemy import inspect, text

RENAMES = [
    ("institutional_holdings", "sec13f_hr"),
    ("sec_8k_items", "sec_8k"),
]


def main() -> None:
    from src.context import get_config_context
    _, context = get_config_context("./configs", use_cache=False, save=False)
    engine = context.store.engine
    existing = set(inspect(engine).get_table_names())

    for old, new in RENAMES:
        if old not in existing:
            print(f"skip: '{old}' not present (already renamed or never created)")
            continue
        if new in existing:
            print(f"skip: '{new}' already exists -- resolve manually before re-running")
            continue
        with engine.begin() as conn:
            conn.execute(text(f'ALTER TABLE "{old}" RENAME TO "{new}"'))
        print(f"renamed: '{old}' -> '{new}'")


if __name__ == "__main__":
    main()
