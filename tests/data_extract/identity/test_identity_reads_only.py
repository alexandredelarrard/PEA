"""AC-011 / REQ-010: only the identity readers read the identity tables.

A static scan (AST) of `src/data_extract`, `src/data_aggregate` and `src/utils`: every store read (`load`, `iter_load`,
`columns`, `distinct`, `key_stats`, `summarize`, ...) of `universe_cik_symbol`, `universe_cusip`, `symbol_tenure`,
`cusip_ticker_map`, `sec_company_tickers`, `sec_cik_profile` or `universe_tickers` must sit in an owner module:
the identity build, the `Identity` resolver, the one table reader `src/utils/identity_tables.py`, the universe
package (`universe_tickers`, `sec_company_tickers`) or the `cusip_ticker_map` builder. `build_cusip_ticker_map`
is called only by the 13F fetchers (the market-wide map stays: superinvestor books need non-universe holdings).
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
SCANNED = ("src/data_extract", "src/data_aggregate", "src/utils")

#: The identity and identity-evidence tables.
IDENTITY_TABLES = {
    "universe_cik_symbol",
    "universe_cusip",
    "symbol_tenure",
    "cusip_ticker_map",
    "sec_company_tickers",
    "sec_cik_profile",
    "universe_tickers",
}
#: Store methods that read a table.
READS = {"load", "iter_load", "columns", "distinct", "key_stats", "summarize", "max_date", "bounds", "row_count", "exists", "load_where"}

#: Owner modules allowed to read (prefix match on the repo-relative path).
OWNERS = (
    "src/data_extract/utils/identity/",  # the identity build
    "src/data_extract/transformers/step_extract_identity.py",
    "src/data_extract/utils/common/identity.py",  # the `Identity` resolver
    "src/data_extract/utils/common/identity_store.py",  # the build's store and the resolver's source cohort
    "src/utils/identity_tables.py",  # the one table reader of every other module
    "src/utils/tickers_universe/",  # `universe_tickers` (Universe, sync) and `sec_company_tickers` (sync)
    "src/data_extract/utils/institutionals/fetch_cusip_map.py",  # `cusip_ticker_map` builder (OpenFIGI cache)
)
#: The only callers of `build_cusip_ticker_map` besides its module.
MAP_BUILDERS = ("src/data_extract/utils/institutionals/fetch_13f.py", "src/data_extract/utils/institutionals/fetch_13f_backfill.py")


def _identity_table(node: ast.AST) -> str | None:
    """`name` for a `Tables.<name>` node naming an identity table."""
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "Tables" and node.attr in IDENTITY_TABLES:
        return node.attr
    return None


def _violations() -> tuple[list[str], list[str], int]:
    """(direct reads outside the owners, `build_cusip_ticker_map` calls outside the 13F fetchers, files scanned)."""
    reads, builders, scanned = [], [], 0
    for root in SCANNED:
        for path in sorted((REPO / root).rglob("*.py")):
            rel = path.relative_to(REPO).as_posix()
            scanned += 1
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                name = node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id if isinstance(node.func, ast.Name) else ""
                tables = [t for arg in [*node.args, *(k.value for k in node.keywords)] if (t := _identity_table(arg))]
                if name in READS and tables and not rel.startswith(OWNERS):
                    reads.append(f"{rel}:{node.lineno} {name}({tables[0]})")
                if name == "build_cusip_ticker_map" and rel not in MAP_BUILDERS and not rel.endswith("fetch_cusip_map.py"):
                    builders.append(f"{rel}:{node.lineno}")
    return reads, builders, scanned


def test_only_the_identity_readers_read_the_identity_tables() -> None:
    reads, builders, scanned = _violations()
    print("\n=== SANITY CHECK: identity tables are read through the identity readers only ===")
    print(f"  {scanned} module(s) scanned under {', '.join(SCANNED)}; owners: {len(OWNERS)} module path(s)")
    print(f"  direct identity-table reads outside the owners: {len(reads)} {reads[:10]}")
    print(f"  build_cusip_ticker_map calls outside the 13F fetchers: {len(builders)} {builders}")
    assert not reads, f"read the identity tables through src/utils/identity_tables.py or Identity: {reads}"
    assert not builders, f"build_cusip_ticker_map outside the 13F fetchers: {builders}"
    print("  Validated: every identity read goes through the build, Identity or identity_tables.read_identity.")


#: U-70: inside `src/data_extract`, only the identity build and the `Identity` module read the published tables; every
#: other step takes them from `Identity` (`load_identity`, `identity.cusip_lines`, `identity.source_cohort`).
DATA_EXTRACT_OWNERS = OWNERS[:4]
#: The shared table readers a data_extract step must not call directly.
SHARED_READERS = {"read_identity", "identity_columns", "load_cusip_lines", "load_source_identity"}


def _shared_reader_calls() -> list[str]:
    """Calls of a shared identity-table reader in `src/data_extract` outside the identity build and `Identity`."""
    calls = []
    for path in sorted((REPO / "src/data_extract").rglob("*.py")):
        rel = path.relative_to(REPO).as_posix()
        if rel.startswith(DATA_EXTRACT_OWNERS):
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Call):
                name = node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id if isinstance(node.func, ast.Name) else ""
                if name in SHARED_READERS:
                    calls.append(f"{rel}:{node.lineno} {name}")
    return calls


def test_data_extract_steps_read_identity_through_identity_only() -> None:
    calls = _shared_reader_calls()
    print("\n=== SANITY CHECK: data_extract steps read identity through Identity only (U-70) ===")
    print(f"  owners: {', '.join(DATA_EXTRACT_OWNERS)}")
    print(f"  shared identity-table reader calls outside the owners: {len(calls)} {calls[:10]}")
    assert not calls, f"take the identity rows from Identity (load_identity): {calls}"
    print("  Validated: no data_extract step reads the published identity tables except through Identity.")
