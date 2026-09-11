"""
def14a_legacy_purge.py  (scripts/)
--------------------------------------------------------------------------------------------
Snapshot every DEF 14A table to parquet, then delete the LEGACY-vintage rows of `def14a_llm`
so the extraction re-runs on them. It does NOT run the extraction: a snapshot, a delete and a
~$200 paid run can never be one irreversible action.

WHY the rows have to go rather than be re-fetched in place. `sec_utils.existing_filings` skips
any accession already stored, so a legacy row is invisible to every future run -- it can never
be refreshed by re-running the fetcher, only by removing it first.

WHAT "legacy" means, stated as two independent predicates that are measured, not assumed:

    A = the accession owns no row in `def14a_directors`
    B = the accession carries none of the 12 columns the refactored flatten added

Measured 2026-09-04: |A| = 8,443, |B| = 8,396, and B is a strict SUBSET of A (B \\ A = 0). The
47 rows in A \\ B were extracted by the NEW prompt and still produced no directors -- filings
that genuinely have no board roster (a DEF 14C, a special-meeting proxy). Re-running those
costs money and changes nothing, so the delete set is the INTERSECTION, not either predicate
alone. The script re-derives both at run time and aborts if the subset relation has broken,
because that would mean the two predicates no longer describe the same cohort.

The delete is surgical by construction: the target accessions own ZERO rows in all four child
tables (that is what predicate A says), so nothing is orphaned and no child cleanup is needed.
Asserted at run time rather than trusted.

The manifest entry MUST be cleared as part of the same operation. `manifest_window` returns a
narrow `since` cutoff whenever the ticker count is unchanged and the last full rescan is under
`manifest_full_rescan_days` old -- so after a full-universe run stamps today's date, the next
run would list only filings after today, find nothing, and the deleted rows would stay gone
without being refetched. Deleting rows and leaving the manifest is the one combination that
loses data for real.

    "$PY" scripts/def14a_legacy_purge.py [-c ./configs]             # dry run + snapshot
    "$PY" scripts/def14a_legacy_purge.py [-c ./configs] --confirm   # snapshot, then delete
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.context import get_config_context
from src.data_extract.utils.common.run_manifest import _manifest_path
from src.data_store.schema import Tables

#: The 12 columns the refactored flatten added. A row carrying ANY of them was written by the
#: new prompt; a row carrying none of them predates it. Hard-coded rather than diffed against
#: the live table because the point is to name the contract, not to discover it.
NEW_SCHEMA_COLUMNS = ("sct_years", "auditor_name", "auditor_since_year", "audit_fees_audit",
                      "audit_fees_audit_related", "audit_fees_tax", "audit_fees_other",
                      "auditor_fees_prior", "n_director_comp_rows", "n_ownership_rows",
                      "pct_gender_stated", "n_women_directors_vs_inferred")

CHILD_TABLES = (("def14a_directors", Tables.def14a_directors),
                ("def14a_executive_comp", Tables.def14a_executive_comp),
                ("def14a_director_comp", Tables.def14a_director_comp),
                ("def14a_ownership", Tables.def14a_ownership))

#: Postgres has a bind-parameter ceiling per statement; the delete runs in chunks well under it.
DELETE_CHUNK = 500

OUT_DIR = ROOT / "reports/2026-09-04/def14a-legacy-purge"


def snapshot(store, out_dir: Path) -> dict:
    """Freeze all five DEF 14A tables to parquet -- every row, every column, `def14a_json`
    included -- plus a byte copy of the run manifest.

    Deliberately unprojected, the one case AGENTS.md's projection rule does not cover: a
    rollback artefact that omits a column cannot roll anything back. The child tables are in
    the snapshot even though the delete cannot touch them, because "the delete cannot touch
    them" is a claim this script makes and a snapshot is what makes the claim falsifiable.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    info: dict = {"snapshot_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                  "tables": {}}

    for name, table in (("def14a_llm", Tables.def14a_llm),) + CHILD_TABLES:
        df = store.load(table)
        df.to_parquet(out_dir / f"{name}.parquet", index=False)
        info["tables"][name] = {"rows": int(len(df)), "columns": int(df.shape[1]),
                                "tickers": int(df["ticker"].nunique()),
                                "accessions": int(df["accession_number"].nunique())}
        print(f"  snapshot {name:24s} {len(df):>7,} rows x {df.shape[1]:>2} cols")

    return info


def legacy_accessions(store) -> tuple[set[str], pd.DataFrame, dict]:
    """The delete set (A & B), the parent rows it selects, and the evidence for both."""
    llm = store.load(Tables.def14a_llm,
                     columns=["ticker", "as_of", "accession_number", *NEW_SCHEMA_COLUMNS])
    llm["as_of"] = pd.to_datetime(llm["as_of"])

    with_directors = set(store.load(Tables.def14a_directors,
                                    columns=["accession_number"])["accession_number"])
    a = set(llm.loc[~llm["accession_number"].isin(with_directors), "accession_number"])
    b = set(llm.loc[~llm[list(NEW_SCHEMA_COLUMNS)].notna().any(axis=1), "accession_number"])

    if b - a:
        raise SystemExit(f"ABORT: {len(b - a)} accessions carry no new-schema column yet own "
                         "director rows. The two predicates no longer agree; re-measure before "
                         "deleting anything.")

    shared = llm.groupby("accession_number")["ticker"].nunique()
    if int((shared > 1).sum()):
        raise SystemExit("ABORT: some accession maps to more than one ticker, so deleting by "
                         "accession alone is not surgical.")

    target = a & b
    rows = llm[llm["accession_number"].isin(target)]
    evidence = {"parent_rows": int(len(llm)),
                "A_no_director_rows": len(a), "B_no_new_schema_column": len(b),
                "delete_set": len(target), "kept_A_minus_B": len(a - b),
                "tickers_affected": int(rows["ticker"].nunique()),
                "span": [str(rows["as_of"].min().date()), str(rows["as_of"].max().date())],
                "rows_per_year": {int(y): int(n) for y, n
                                  in rows.groupby(rows["as_of"].dt.year).size().items()}}
    return target, rows, evidence


def assert_no_child_rows(store, target: set[str]) -> None:
    """Predicate A says the target owns no child rows. Verify it against all four tables
    rather than trusting the one (`def14a_directors`) the predicate was derived from."""
    for name, table in CHILD_TABLES:
        acc = set(store.load(table, columns=["accession_number"])["accession_number"])
        hit = target & acc
        if hit:
            raise SystemExit(f"ABORT: {len(hit)} target accessions own rows in {name}; the "
                             "delete would orphan them. Example: " + str(sorted(hit)[:3]))
        print(f"  {name:24s} 0 of {len(target):,} target accessions own rows here")


def clear_manifest_entry(context, out_dir: Path) -> None:
    """Drop `def14a_llm` from the run manifest so the next run does a FULL relist.

    An absent entry makes `manifest_window` return `(fallback_since, True)` on its first
    branch -- the same state a first-ever run sees. The file is copied into the snapshot
    directory first; it is the only part of this operation that touches `data/`.
    """
    path = _manifest_path(context)
    if not path.exists():
        print("  manifest absent -- nothing to clear (already means 'full rescan')")
        return
    shutil.copy2(path, out_dir / "extraction_manifest.json.before")
    data = json.loads(path.read_text(encoding="utf-8"))
    removed = data.pop("def14a_llm", None)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    print(f"  cleared manifest entry def14a_llm: {removed}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-c", "--config", default="./configs")
    ap.add_argument("--confirm", action="store_true",
                    help="actually delete; without it the script only snapshots and reports")
    args = ap.parse_args()

    _cfg, context = get_config_context(args.config, use_cache=False, save=False)
    store = context.store

    print("\n=== snapshot ===")
    info = snapshot(store, OUT_DIR)

    print("\n=== delete set ===")
    target, rows, evidence = legacy_accessions(store)
    for k, v in evidence.items():
        if k != "rows_per_year":
            print(f"  {k:26s} {v}")
    print("  rows_per_year " + " ".join(f"{y}:{n}" for y, n
                                        in sorted(evidence["rows_per_year"].items())))

    print("\n=== child-table safety ===")
    assert_no_child_rows(store, target)

    info["delete_set"] = evidence
    rows[["ticker", "as_of", "accession_number"]].to_parquet(
        OUT_DIR / "deleted_accessions.parquet", index=False)
    (OUT_DIR / "purge_manifest.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"\nsnapshot + delete list written to {OUT_DIR}")

    if not args.confirm:
        print("\nDRY RUN -- nothing deleted. Re-run with --confirm to delete "
              f"{len(target):,} rows.")
        return

    print("\n=== deleting ===")
    ordered = sorted(target)
    deleted = 0
    for i in range(0, len(ordered), DELETE_CHUNK):
        chunk = ordered[i:i + DELETE_CHUNK]
        deleted += store.delete(Tables.def14a_llm, where={"accession_number": chunk})
        print(f"  {deleted:>6,} / {len(ordered):,}", end="\r")
    print(f"  deleted {deleted:,} rows" + " " * 20)
    if deleted != len(target):
        raise SystemExit(f"ABORT: deleted {deleted} but targeted {len(target)}; investigate "
                         f"against {OUT_DIR / 'def14a_llm.parquet'} before re-running anything.")

    print("\n=== manifest ===")
    clear_manifest_entry(context, OUT_DIR)

    print("\n=== after ===")
    after = store.load(Tables.def14a_llm, columns=["ticker", "accession_number"])
    print(f"  def14a_llm {len(after):,} rows ({info['tables']['def14a_llm']['rows']:,} before), "
          f"{after['ticker'].nunique()} tickers")
    left = target & set(after["accession_number"])
    print(f"  target accessions still present: {len(left)}")
    for name, table in CHILD_TABLES:
        n = len(store.load(table, columns=["ticker"]))
        print(f"  {name:24s} {n:>7,} rows ({info['tables'][name]['rows']:,} before)")


if __name__ == "__main__":
    main()
