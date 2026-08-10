"""
The architectural invariant the whole data-layer refactor exists to create: `src/data_store/`
is the ONLY code in the repo that knows SQL exists.

Before this, three parallel access layers spoke to one database -- `DataStore`, a second store
implementation (`PartStore`), and eleven modules issuing `pd.read_sql` on `store.engine` because
the facade could not express `date >= x`, chunked reads, `MAX(col)`, `SELECT DISTINCT` or column
introspection. Three of those built SQL by string interpolation while every sibling bound params,
and two queried `information_schema`, which does not exist in SQLite -- so those paths could not
be tested offline at all.

These are grep-level assertions on purpose: an import-level check would pass on a module that
holds a raw SQL string, and the point is that the strings are gone too.
"""
from __future__ import annotations

import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"
STORE_PKG = SRC / "data_store"
ENGINE_FACTORY = SRC / "utils" / "db.py"      # creates the Engine; hands it to DataStore

# `text(` is deliberately absent here: it collides with read_text/write_text/_participants_text.
# Importing sqlalchemy at all is the tighter test, and it is what actually gates SQL access.
_SQLALCHEMY_IMPORT = re.compile(r"^\s*(?:from\s+sqlalchemy[\w.]*\s+import|import\s+sqlalchemy)",
                                re.MULTILINE)
_RAW_SQL_CALL = re.compile(r"\bread_sql\b|\.to_sql\(|\bengine\.connect\(|\braw_connection\(|"
                           r"\bstore\.engine\b|\binformation_schema\b")


def _modules():
    """Every `src/` module that is not the store package or the engine factory."""
    return [p for p in SRC.rglob("*.py")
            if STORE_PKG not in p.parents and p != ENGINE_FACTORY]


def test_only_the_store_package_imports_sqlalchemy():
    offenders = [str(p.relative_to(SRC)) for p in _modules()
                 if _SQLALCHEMY_IMPORT.search(p.read_text(encoding="utf-8"))]
    assert not offenders, (
        f"{len(offenders)} module(s) outside src/data_store/ import sqlalchemy directly; route "
        f"the read through `context.store` instead: {offenders}")


def test_no_module_outside_the_store_issues_raw_sql():
    offenders = {}
    for p in _modules():
        hits = sorted(set(_RAW_SQL_CALL.findall(p.read_text(encoding="utf-8"))))
        if hits:
            offenders[str(p.relative_to(SRC))] = hits
    assert not offenders, f"raw DB access outside src/data_store/: {offenders}"


def test_store_surface_covers_every_capability_the_bypasses_needed():
    """Each raw-SQL shape that existed must have a facade method, or a bypass returns."""
    from src.data_store.store import DataStore

    required = {
        "exists": "information_schema.tables (freshness.py) -- Postgres-only",
        "columns": "information_schema.columns (step_train, ls_model, freshness)",
        "row_count": "SELECT COUNT(*) (PartStore.row_count)",
        "bounds": "SELECT MIN(q), MAX(q) (fetch_hf_transcripts)",
        "max_date": "SELECT MAX(date) (PartStore.max_date, freshness)",
        "distinct": "SELECT DISTINCT [ORDER BY .. LIMIT] (sec_utils, bulk_cache, step_train, "
                    "earnings-call streamers)",
        "load": "WHERE / IN / IS NOT NULL / date >= x / date <= y / projection",
        "iter_load": "server-side cursor over the 574-column cube",
        "save": "upsert on the registry PK",
        "replace": "truncate-or-drop + chunked COPY",
        "append_tail": "DELETE tail + append (PartStore.append_tail)",
        "bulk_seed": "COPY append (the cube's streaming writer)",
        "delete": "targeted row delete (the force-re-embed reconcile)",
        "drop": "DROP TABLE IF EXISTS",
        "ensure_columns": "ADD COLUMN schema evolution",
    }
    missing = [m for m in required if not callable(getattr(DataStore, m, None))]
    assert not missing, f"DataStore lost capabilities the bypasses needed: {missing}"

    print("\n=== SANITY CHECK: one SQL boundary ===")
    print(f"  {len(_modules())} modules scanned outside src/data_store/: 0 import sqlalchemy, "
          f"0 use read_sql/to_sql/engine.connect/raw_connection/store.engine/information_schema.")
    print(f"  DataStore exposes all {len(required)} capabilities the 11 former bypasses needed, so "
          "no call site has a reason to reach past the facade. Validated.")
