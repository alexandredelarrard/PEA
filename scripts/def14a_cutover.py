"""
def14a_cutover.py  (scripts/)
--------------------------------------------------------------------------------------------
Phase-6 Step 4: the ONLY script in this plan that changes live data.

It runs the four DDL/DML statements the plan specifies, in the order the plan specifies, and
nothing else. It does NOT run the backfill — that is the CLI's job afterwards, so a cutover
and a 15,000-call refetch can never be one irreversible action.

Why truncate-and-rebuild rather than a migration: `existing_filings` dedups on accession, so
an incremental run after a parser change skips every filing it already has. There is no way to
re-extract in place.

    "$PY" scripts/def14a_cutover.py [-c ./configs] --confirm        # do it
    "$PY" scripts/def14a_cutover.py [-c ./configs]                  # dry run, prints the plan

`--confirm` is required because every statement below is irreversible. A pre-cutover snapshot
(`def14a_baseline.py --tag pre-cutover`) is the only rollback that exists, and the plan makes
taking it a prerequisite rather than an option.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.context import get_config_context

#: The retired HTML-parsed child tables. edgartools' proxy parser returns values that are
#: silently WRONG rather than absent (a fabricated 0.5 for the "<1%" footnote, a missed
#: "(in thousands)" fee header, the Total column duplicated into a component slot), and the
#: defects are ticker-persistent, so they do not average out. The LLM path's own child tables
#: replaced them on every measured axis.
DROP_TABLES = ("sec_def14a_executive_comp", "sec_def14a_director_comp",
               "sec_def14a_ownership", "sec_def14a_votes")

#: `sec_def14a` is DROPPED, not truncated: Phase 4 took it from 46 columns to 25, and
#: `CREATE TABLE IF NOT EXISTS` cannot retire a column on a table that already exists, so a
#: truncate would leave 21 dead columns behind. `def14a_llm` keeps its shape, so a truncate is
#: enough there -- but it must happen, because the accession dedup would otherwise skip all
#: 8,667 stored filings and the new extraction would never run.
DROP_FOR_RESHAPE = ("sec_def14a",)
TRUNCATE_TABLES = ("def14a_llm",)

#: Removed from `sql/schema.sql` in Phase 3, but that only fixes a FRESH bootstrap. On a live
#: table the columns survive a truncate, and three permanently-NULL columns read downstream as
#: "this company discloses no technology directors" rather than "we stopped extracting an
#: opinion". They were an opinion: mean |delta| of 1.06 directors between consecutive filings
#: of the same company, only 38.8% unchanged, and wrong by 7x on HUBB 2022 whose own skills
#: matrix states "Cybersecurity and Technology 78%" of 9 directors.
DROP_COLUMNS = ("n_technology_directors", "pct_technology_directors", "technology_committee")

#: Clearing these makes the next run do a FULL rescan instead of a manifest-narrowed window.
#: Without it the fetcher lists only filings since the last run date and the truncated table
#: never refills.
MANIFEST_TABLES = ("def14a_llm", "sec_def14a")


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase-6 Step 4 cutover (irreversible).")
    ap.add_argument("-c", "--config", default="./configs")
    ap.add_argument("--confirm", action="store_true", help="actually execute")
    args = ap.parse_args()

    _, context = get_config_context(args.config, use_cache=False, save=False)
    store = context.store
    engine = store.engine

    stmts: list[tuple[str, str]] = []
    for t in DROP_TABLES:
        stmts.append((f'DROP TABLE IF EXISTS "{t}"', f"retire {t}"))
    for t in DROP_FOR_RESHAPE:
        stmts.append((f'DROP TABLE IF EXISTS "{t}"', f"drop {t} (column set changed 46 -> 25)"))
    for t in TRUNCATE_TABLES:
        if store.exists(t):
            stmts.append((f'TRUNCATE TABLE "{t}"', f"empty {t} so the accession dedup re-reads"))
    if store.exists("def14a_llm"):
        live = set(store.columns("def14a_llm"))
        present = [c for c in DROP_COLUMNS if c in live]
        if present:
            cols = ", ".join(f'DROP COLUMN IF EXISTS "{c}"' for c in present)
            stmts.append((f'ALTER TABLE "def14a_llm" {cols}',
                          f"retire {len(present)} technology column(s)"))
    if store.exists("extraction_run"):
        names = ", ".join(f"'{t}'" for t in MANIFEST_TABLES)
        stmts.append((f'DELETE FROM "extraction_run" WHERE table_name IN ({names})',
                      "clear the run manifest so the next run does a FULL rescan"))

    print("=" * 78)
    print("PHASE-6 CUTOVER PLAN" + ("" if args.confirm else "   (DRY RUN — nothing executed)"))
    print("=" * 78)
    print("\nRow counts before:")
    for t in DROP_TABLES + DROP_FOR_RESHAPE + TRUNCATE_TABLES:
        print(f"  {t:30s} {store.row_count(t) if store.exists(t) else '(absent)'}")

    print(f"\n{len(stmts)} statement(s):")
    for sql, why in stmts:
        print(f"  {why}\n    {sql}")

    if not args.confirm:
        print("\nRe-run with --confirm to execute. Take the pre-cutover snapshot FIRST:")
        print('  "$PY" scripts/def14a_baseline.py -c ./configs --tag pre-cutover')
        return

    with engine.begin() as conn:
        for sql, why in stmts:
            conn.execute(text(sql))
            print(f"  OK  {why}")

    print("\nRow counts after:")
    for t in DROP_TABLES + DROP_FOR_RESHAPE + TRUNCATE_TABLES:
        print(f"  {t:30s} {store.row_count(t) if store.exists(t) else '(absent)'}")
    if store.exists("def14a_llm"):
        left = [c for c in DROP_COLUMNS if c in set(store.columns("def14a_llm"))]
        print(f"\n  def14a_llm columns: {len(store.columns('def14a_llm'))}, "
              f"technology columns left: {left or 'NONE'}")
    print("\nCUTOVER DONE. The tables are now empty/absent — the backfill is the next step:")
    print('  "$PY" -m src data_extract def14a       -c ./configs -t AAPL   # warm the tables')
    print('  "$PY" -m src data_extract def14a       -c ./configs')
    print('  "$PY" -m src data_extract def14a-edgar -c ./configs')
    print('  "$PY" -m src data_extract sec-8k-votes -c ./configs')


if __name__ == "__main__":
    main()
