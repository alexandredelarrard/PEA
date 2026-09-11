"""
gpt_refactor_smoke_prep.py  (scripts/)
--------------------------------------------------------------------------------------------
Snapshot `def14a_llm`, then reshape it to the 42-column contract and clear the smoke
tickers so the refactored extraction path actually re-runs on them.

The ONLY script in this plan that changes live data. It runs three statements and nothing
else; it does NOT run the extraction, so a reshape and a ~300-call paid run can never be one
irreversible action.

Why each statement:

  1. `ALTER TABLE def14a_llm DROP COLUMN` x3 -- `n_technology_directors`,
     `pct_technology_directors` and `technology_committee` were an opinion, not an extraction
     (mean |delta| of 1.06 directors between consecutive filings of the same company, only
     38.8% unchanged). No code writes them any more, and `CREATE TABLE IF NOT EXISTS` cannot
     retire a column on a table that already exists, so a live table needs the explicit ALTER.
     45 columns -> 42.

  2. `DELETE FROM def14a_llm WHERE ticker IN (<smoke set>)` -- the fetcher skips any accession
     already stored (`existing_filings`), so leaving the smoke tickers' rows in place would
     make the run cost zero calls and produce zero new rows. The rows are not the comparison
     set once they are overwritten anyway: `save(pk=[ticker, accession_number])` is an upsert,
     so a re-extracted accession REPLACES its old row. The parquet snapshot in step 0 is
     therefore the comparison set, and the delete is what makes the run happen at all.

  3. Everything else is left alone. The other ~482 tickers keep their rows; there is no
     TRUNCATE. `extraction_run` does not exist on this deployment, so the manifest needs no
     clearing -- an absent entry already means "full rescan".

The snapshot is written on BOTH the dry run and the confirmed run, because it is the only
rollback that exists for the row data and it costs nothing to have taken it twice.

    "$PY" scripts/gpt_refactor_smoke_prep.py [-c ./configs]             # dry run + snapshot
    "$PY" scripts/gpt_refactor_smoke_prep.py [-c ./configs] --confirm   # do it
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.context import get_config_context
from src.data_store.schema import Tables

#: The 15 smoke tickers, chosen for filing GEOMETRY rather than familiarity: BA/NKE co-PEO
#: years, SBUX's all-zero PVP matrix, GOOGL/BRK-B dual-class ownership, A's pre-2001 filings
#: with an empty `primaryDocument`, TDG's tally-free 5.07(d), XOM's transposed vote layout.
SMOKE_TICKERS = ("AAPL", "JPM", "BA", "NKE", "SBUX", "GOOGL", "BRK-B", "XOM", "PG",
                 "CAT", "PFE", "A", "AMAT", "TDG", "GE")

DROP_COLUMNS = ("n_technology_directors", "pct_technology_directors", "technology_committee")

#: The column count of the LIVE table once the three technology columns are gone (45 - 3),
#: asserted immediately after the ALTER. It is NOT the flatten's column count: the refactored
#: flatten produces 54, which is what `sql/schema.sql` declares for a fresh bootstrap. The
#: extra 12 arrive on the first write -- `store.save` runs `ADD COLUMN IF NOT EXISTS` for any
#: frame column the table lacks -- so the smoke run takes this table 42 -> 54 and leaves the
#: 12 new columns NULL on the ~8,375 rows that were not re-extracted.
EXPECTED_COLUMNS = 42

OUT_DIR = ROOT / "reports/planning/active-tasks/2026-09-02-gpt-extract-refactor/baseline"


def snapshot(store, out_dir: Path) -> dict:
    """Freeze the WHOLE of `def14a_llm` to parquet -- every row, every column, `def14a_json`
    included.

    Deliberately unprojected, which is the one case AGENTS.md's projection rule does not
    cover: a rollback artefact that omits a column cannot roll anything back. It is also
    what makes Phase 7's before/after comparison possible after the upsert has overwritten
    the smoke tickers' live rows.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    df = store.load(Tables.def14a_llm)
    df.to_parquet(out_dir / "def14a_llm.parquet", index=False)

    smoke = df[df["ticker"].isin(SMOKE_TICKERS)]
    info = {
        "snapshot_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "tickers": int(df["ticker"].nunique()),
        "smoke_tickers": list(SMOKE_TICKERS),
        "smoke_rows": int(len(smoke)),
        "smoke_accessions": int(smoke["accession_number"].nunique()),
        "per_ticker": {t: int(n) for t, n in smoke["ticker"].value_counts().sort_index().items()},
    }
    (out_dir / "snapshot_manifest.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase-7 steps 1-2 (irreversible with --confirm).")
    ap.add_argument("-c", "--config", default="./configs")
    ap.add_argument("--confirm", action="store_true", help="actually execute the statements")
    args = ap.parse_args()

    _, context = get_config_context(args.config, use_cache=False, save=False)
    store = context.store

    if not store.exists(Tables.def14a_llm):
        print("def14a_llm does not exist — nothing to reshape.")
        return

    print("=" * 78)
    print("PHASE-7 SMOKE PREP" + ("" if args.confirm else "   (DRY RUN — no DB change)"))
    print("=" * 78)

    info = snapshot(store, OUT_DIR)
    print(f"\nSnapshot -> {OUT_DIR}")
    print(f"  def14a_llm  {info['rows']} rows x {info['columns']} cols, "
          f"{info['tickers']} tickers")
    print(f"  of which the 15 smoke tickers: {info['smoke_rows']} rows, "
          f"{info['smoke_accessions']} accessions")
    print("  " + "  ".join(f"{t}={n}" for t, n in sorted(info["per_ticker"].items())))

    live = list(store.columns(Tables.def14a_llm))
    present = [c for c in DROP_COLUMNS if c in live]
    stmts: list[tuple[str, str]] = []
    if present:
        cols = ", ".join(f'DROP COLUMN IF EXISTS "{c}"' for c in present)
        stmts.append((f'ALTER TABLE "def14a_llm" {cols}',
                      f"retire {len(present)} technology column(s): {len(live)} -> "
                      f"{len(live) - len(present)}"))
    names = ", ".join(f"'{t}'" for t in SMOKE_TICKERS)
    stmts.append((f'DELETE FROM "def14a_llm" WHERE "ticker" IN ({names})',
                  f"clear {info['smoke_rows']} smoke row(s) so the accession dedup re-extracts "
                  f"them ({info['rows']} -> {info['rows'] - info['smoke_rows']})"))

    print(f"\n{len(stmts)} statement(s):")
    for sql, why in stmts:
        print(f"  {why}\n    {sql}")

    if not args.confirm:
        print("\nRe-run with --confirm to execute. The snapshot above is the only rollback.")
        return

    with context.store.engine.begin() as conn:
        for sql, why in stmts:
            conn.execute(text(sql))
            print(f"  OK  {why}")

    after = list(store.columns(Tables.def14a_llm))
    left = [c for c in DROP_COLUMNS if c in after]
    rows_after = store.row_count(Tables.def14a_llm)
    remaining = store.load(Tables.def14a_llm, columns=["ticker"])
    still_smoke = sorted(set(remaining["ticker"]) & set(SMOKE_TICKERS))

    print(f"\nAfter:")
    print(f"  columns              {len(after)} (expected {EXPECTED_COLUMNS})")
    print(f"  technology columns   {left or 'NONE'}")
    print(f"  rows                 {rows_after}")
    print(f"  smoke tickers left   {still_smoke or 'NONE'}")
    assert not left, f"technology columns survived the ALTER: {left}"
    assert len(after) == EXPECTED_COLUMNS, \
        f"def14a_llm has {len(after)} columns, expected {EXPECTED_COLUMNS}"
    assert not still_smoke, f"smoke tickers still have rows: {still_smoke}"

    print("\nREADY. The paid smoke run is the next step, one ticker first so the four child "
          "tables are created warm:")
    print('  "$PY" -m src data_extract def14a -c ./configs -t AAPL')
    print('  "$PY" -m src data_extract def14a -c ./configs -t ' + ",".join(SMOKE_TICKERS[1:]))


if __name__ == "__main__":
    main()
