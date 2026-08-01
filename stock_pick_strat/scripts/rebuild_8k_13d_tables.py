"""
One-time DB migration: DROP the existing `sec_8k` / `sec_13d` tables so they can
be rebuilt from scratch by the edgartools-based fetchers (fetch_8k_edgar.py /
fetch_13d_edgar.py), which replace the old submissions-JSON-only extraction
(fetch_8k_items.py / fetch_13d.py) with a richer schema -- `sec_8k` gains
is_amendment/has_earnings/has_press_release; `sec_13d`'s grain changes to one row
PER REPORTING PERSON (ticker, accession_number, rp_seq), which cannot be expressed
as an incremental ALTER TABLE on the old (ticker, accession_number) PK.

DESTRUCTIVE and NOT reversible from within this script (unlike the metadata-only
rename in scripts/rename_form_dispatch_tables.py) -- confirmed with the user
before running given the live tables held 3,219 (sec_13d) / 97,383 (sec_8k) rows
in the old schema. `ensure_table`/`ensure_columns` (called by the fetchers' first
`context.store.save(...)`) recreate the table fresh, matching the new schema.

Run once:
    python -m scripts.rebuild_8k_13d_tables
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sqlalchemy import inspect, text

TABLES = ["sec_8k", "sec_13d"]


def main() -> None:
    from src.context import get_config_context
    _, context = get_config_context("./configs", use_cache=False, save=False)
    engine = context.store.engine
    existing = set(inspect(engine).get_table_names())

    for table in TABLES:
        if table not in existing:
            print(f"skip: '{table}' not present")
            continue
        with engine.begin() as conn:
            n = conn.execute(text(f'SELECT COUNT(*) FROM "{table}"')).scalar()
            conn.execute(text(f'DROP TABLE "{table}"'))
        print(f"dropped: '{table}' ({n} rows removed)")


if __name__ == "__main__":
    main()
