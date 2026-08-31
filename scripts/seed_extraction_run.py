"""
seed_extraction_run.py  (scripts/)
--------------------------------------------------------------------------------------------
ONE-TIME cutover: `data/extraction_manifest.json` -> `extraction_run`, run once after
`run_manifest.py` has been switched to read/write the DB ledger (Phase 5.3).

PRECONDITION -- do not skip: no fetcher may run between the code cutover and this script.
`manifest_window` / `record_run` are reached only from the fetch path, so the window between
"code reads the DB" and "the DB is seeded" is safe as long as nothing fetches in it. Also do
not run this while an OLDER, JSON-writing process is still walking (check with
`Get-CimInstance Win32_Process -Filter "Name='python.exe'"` or equivalent) -- it will keep
appending to the JSON after this script has read it, and those entries would be lost.

Idempotent: `run_id` is the constant "seed", so re-running upserts on `(table_name, run_id)`
rather than duplicating.

    "$PY" scripts/seed_extraction_run.py --dry-run
    "$PY" scripts/seed_extraction_run.py --yes
    "$PY" scripts/seed_extraction_run.py --verify          # after --yes, before deleting the JSON
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.context import get_config_context
from src.data_store.schema import Tables

_MANIFEST_PATH = Path("data/extraction_manifest.json")


def _seed_frame(manifest: dict) -> pd.DataFrame:
    """One `extraction_run` row per JSON entry. Pure -- unit-testable without the DB."""
    rows = []
    for table_name, entry in manifest.items():
        last_run = entry.get("last_run_date")
        last_full = entry.get("last_full_rescan_date")
        rows.append({
            "table_name": table_name,
            "run_id": "seed",
            "scope_hash": None,
            "run_date": last_run,
            "last_full_rescan_date": last_full,
            # The JSON's ticker_count meant four different things across fetchers (Phase 5.3
            # research); seeded as-is under the new name because it is what `manifest_window`
            # actually compared against on the live run.
            "tickers_requested": entry.get("ticker_count"),
            "tickers_written": None,       # unknown, and honestly so -- the JSON never had it
            "tickers_failed": None,
            "rows_added": entry.get("rows_added"),
            "is_full_rescan": bool(last_run) and last_run == last_full,
            "started_at": entry.get("updated_at"),
            "finished_at": entry.get("updated_at"),
            "status": "ok",
        })
    return pd.DataFrame(rows)


def _old_manifest_window(entry: dict | None, ticker_count: int, full_rescan_days: int):
    """The JSON-era `manifest_window` fallback logic, INLINED (not imported): by the time this
    script runs, `run_manifest.py` has already been cut over to the DB and no longer carries
    this branch. Kept here only so `--verify` has something to compare the new path against."""
    if not entry or entry.get("ticker_count") != ticker_count:
        return None, True
    last_full = entry.get("last_full_rescan_date")
    if not last_full:
        return None, True
    age_days = (pd.Timestamp.today().normalize() - pd.Timestamp(last_full).normalize()).days
    if age_days >= full_rescan_days:
        return None, True
    last_run = entry.get("last_run_date")
    if not last_run:
        return None, True
    return pd.Timestamp(last_run).normalize(), False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--yes", action="store_true", help="write the seed rows")
    parser.add_argument("--dry-run", action="store_true", help="print what would be written")
    parser.add_argument("--verify", action="store_true",
                        help="assert manifest_window agrees, JSON vs DB, for every table "
                             "(run after --yes)")
    parser.add_argument("--manifest", default=str(_MANIFEST_PATH),
                        help=f"path to the JSON manifest (default: {_MANIFEST_PATH})")
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    frame = _seed_frame(manifest)
    print(f"{len(frame)} table(s) in {args.manifest}:")
    print(frame[["table_name", "run_date", "last_full_rescan_date",
                "tickers_requested", "rows_added"]].to_string(index=False))

    if args.dry_run or not (args.yes or args.verify):
        print("\nNothing written. Pass --yes to seed, --verify to check parity.")
        return

    _, context = get_config_context("./configs", use_cache=False, save=False)

    if args.yes:
        written = context.store.save(Tables.extraction_run, frame)
        print(f"\nWrote {written} row(s) to {Tables.extraction_run} (run_id='seed').")

    if args.verify:
        from src.data_extract.utils.common.run_manifest import manifest_window

        full_rescan_days = int(context.config.data_extract.manifest_full_rescan_days)
        mismatches = []
        for table_name, entry in manifest.items():
            ticker_count = int(entry.get("ticker_count") or 0)
            old_since, old_full = _old_manifest_window(entry, ticker_count, full_rescan_days)
            new_since, new_full = manifest_window(
                context, table_name, ticker_count,
                fallback_since=pd.Timestamp("1900-01-01"), full_rescan_days=full_rescan_days)
            # A fallback (`is_full_rescan=True`) makes `since` a caller-supplied constant on
            # BOTH paths, so only the flag is comparable there; a non-fallback window compares
            # the resolved date too.
            ok = (old_full == new_full) and (old_full or old_since == new_since)
            print(f"  {table_name:28s} old=({old_since}, {old_full})  "
                 f"new=({new_since}, {new_full})  {'OK' if ok else 'MISMATCH'}")
            if not ok:
                mismatches.append(table_name)

        if mismatches:
            print(f"\nFAILED: {len(mismatches)} table(s) disagree -- do NOT delete the JSON: "
                 f"{mismatches}")
            sys.exit(1)
        print(f"\nPASS: manifest_window agrees for all {len(manifest)} table(s). "
             "The JSON may now be deleted.")


if __name__ == "__main__":
    main()
