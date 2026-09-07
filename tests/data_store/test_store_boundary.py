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

import ast
import io
import re
import tokenize
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


def code_only(text: str) -> str:
    """`text` with COMMENTS and DOCSTRINGS blanked, other string literals kept.

    ⚠ Why this exists, and why it does NOT strip every string. The scan is grep-level on
    purpose -- an import-level check would pass a module that holds a raw SQL string, and the
    point is that the strings are gone too. But prose is not code: `capital.py` documents why
    a `pensionDeficit` column does not exist by saying an `information_schema` match on
    `%pension%` returns nothing, and flagging that docstring reported a store-boundary
    violation in a module that issues no SQL at all. That is a false positive AND a bad
    incentive -- the cheapest way to make it green is to delete the explanation.

    So: docstrings and comments are removed (they cannot execute a query), every other string
    literal is retained (a raw `"SELECT ... FROM information_schema"` assigned or passed
    anywhere is still caught). `test_the_scan_still_catches_real_raw_sql` pins both halves.
    """
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return text

    lines = text.splitlines()
    blank: set[int] = set()                      # 1-indexed source lines to drop

    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                 ast.AsyncFunctionDef)):
            continue
        body = getattr(node, "body", None)
        if not body:
            continue
        first = body[0]
        if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)):
            blank.update(range(first.lineno, (first.end_lineno or first.lineno) + 1))

    # ⚠ TRUNCATE at the comment column, never blank the whole line: `x = read_sql(q)  # why`
    # is a violation with a comment on it, and dropping the line would hide it.
    cut: dict[int, int] = {}
    try:
        for tok in tokenize.generate_tokens(io.StringIO(text).readline):
            if tok.type == tokenize.COMMENT:
                row, col = tok.start
                cut[row] = min(col, cut.get(row, col))
    except (tokenize.TokenError, IndentationError):
        pass

    out = []
    for i, line in enumerate(lines, start=1):
        if i in blank:
            out.append("")
        elif i in cut:
            out.append(line[:cut[i]])
        else:
            out.append(line)
    return "\n".join(out)


def test_only_the_store_package_imports_sqlalchemy():
    offenders = [str(p.relative_to(SRC)) for p in _modules()
                 if _SQLALCHEMY_IMPORT.search(p.read_text(encoding="utf-8"))]
    assert not offenders, (
        f"{len(offenders)} module(s) outside src/data_store/ import sqlalchemy directly; route "
        f"the read through `context.store` instead: {offenders}")


def test_no_module_outside_the_store_issues_raw_sql():
    offenders = {}
    for p in _modules():
        hits = sorted(set(_RAW_SQL_CALL.findall(code_only(p.read_text(encoding="utf-8")))))
        if hits:
            offenders[str(p.relative_to(SRC))] = hits
    assert not offenders, f"raw DB access outside src/data_store/: {offenders}"


def test_the_scan_still_catches_real_raw_sql():
    """Guard the guard. `code_only` narrowed the scan from "the file mentions it" to "the code
    does it", and a narrowing that went too far would silently un-gate the boundary -- so pin
    both directions on synthetic sources."""
    caught = 'import pandas as pd\ndf = pd.read_sql("SELECT 1", store.engine)\n'
    inline = 'df = pd.read_sql(q, con)  # a comment must not hide this\n'
    literal = 'QUERY = "SELECT column_name FROM information_schema.columns"\n'
    prose_only = ('"""An information_schema match on %pension% returns nothing."""\n'
                  "VALUE = 1\n")
    commented = "# df = pd.read_sql(q, con)\nVALUE = 1\n"

    for label, src in (("read_sql call", caught), ("call with inline comment", inline),
                       ("raw SQL string literal", literal)):
        assert _RAW_SQL_CALL.search(code_only(src)), f"{label} must still be caught"
    for label, src in (("docstring prose", prose_only), ("commented-out code", commented)):
        assert not _RAW_SQL_CALL.search(code_only(src)), f"{label} must NOT be flagged"

    print("\n=== SANITY CHECK: the boundary scan reads code, not prose ===")
    print("  still caught: a read_sql call, one with an inline comment after it, and a raw")
    print("  SQL string literal assigned to a constant.")
    print("  no longer flagged: the same words inside a docstring or a commented-out line --")
    print("  `capital.py` documents why a pensionDeficit column does not exist by naming")
    print("  information_schema, and that is documentation, not a query. Validated.")


def test_store_surface_covers_every_capability_the_bypasses_needed():
    """Each raw-SQL shape that existed must have a facade method, or a bypass returns."""
    from src.data_store.store import DataStore

    required = {
        "exists": "information_schema.tables -- Postgres-only",
        "columns": "information_schema.columns (step_train, ls_model)",
        "row_count": "SELECT COUNT(*) (PartStore.row_count)",
        "bounds": "SELECT MIN(q), MAX(q) (fetch_hf_transcripts)",
        "max_date": "SELECT MAX(date) (PartStore.max_date, fetch_short_interest)",
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
