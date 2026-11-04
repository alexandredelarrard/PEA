"""AC-012: the published identity tables carry their new names and `company_id`, and no old name is left as a table reference.

`tickers_universe` -> `universe_tickers`, `entity_lineage` -> `universe_cik_symbol`, `security_master` -> `universe_cusip`;
the lineage column `entity_id` -> `company_id`. Reference check only (PD-12): the `src.utils.tickers_universe` package path,
the `entity_lineage.py` / `security_master.py` module names and the `*_manual.json` file names keep their names, and the
comments of the ask-first files (`context.py`, `utils/step.py`, `constants/`) are not scanned.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

from src.data_store import ddl
from src.data_store.schema import Tables

REPO = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())

#: old name -> new name.
RENAMES = {"tickers_universe": "universe_tickers", "entity_lineage": "universe_cik_symbol", "security_master": "universe_cusip"}

#: Where a table reference can live; the Airflow DAGs are under `src/dags`.
SCANNED = ("src", "configs", "sql")
SUFFIXES = {".py", ".json", ".yml", ".yaml", ".sql"}

#: Ask-first files (AGENTS.md): only their code is scanned, not their comments or docstrings.
ASK_FIRST = {"src/context.py", "src/utils/step.py"}
ASK_FIRST_DIRS = ("src/constants/",)

#: Old spellings that are not table references: the package path, the module names and their imports.
NOT_A_TABLE = re.compile(
    r"utils[./]tickers_universe\b"
    r"|identity[./](?:entity_lineage|security_master)\b"
    r"|import (?:entity_lineage|security_master)\b"
    r"|\b(?:entity_lineage|security_master)\.py\b"
)
OLD_WORD = re.compile(r"\b(?:tickers_universe|entity_lineage|security_master|entity_id|identity_history)\b")


def _hits(rel: str, text: str) -> list[str]:
    """`rel:line: name` for every old name left in `text` once the non-table spellings are masked out."""
    masked = NOT_A_TABLE.sub(lambda m: "#" * len(m.group(0)), text)
    return [f"{rel}:{text.count(chr(10), 0, m.start()) + 1}: {m.group(0)}" for m in OLD_WORD.finditer(masked)]


def _code_only(text: str) -> str:
    """The names and non-docstring string constants of a Python file, one per line (comments and docstrings dropped)."""
    tree = ast.parse(text)
    docstrings = {
        id(body[0].value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef)
        and (body := node.body)
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
    }
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            out.append(node.attr)
        elif isinstance(node, ast.Name):
            out.append(node.id)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings:
            out.append(node.value)
    return "\n".join(out)


def test_registry_exposes_the_new_identity_tables_only() -> None:
    for old, new in RENAMES.items():
        assert hasattr(Tables, new), f"Tables.{new} missing"
        assert getattr(Tables, new).name == new
        assert not hasattr(Tables, old), f"Tables.{old} still registered"
    assert not hasattr(Tables, "identity_history")
    columns = Tables.universe_cik_symbol.read_columns
    assert "company_id" in columns and "entity_id" not in columns, columns

    print("\n=== SANITY CHECK: identity table registry ===")
    print(f"  Tables exposes {sorted(RENAMES.values())}, none of {sorted(RENAMES)}, no identity_history;")
    print("  universe_cik_symbol carries company_id, not entity_id. Validated.")


def test_schema_sql_blocks_and_indexes_use_the_new_names() -> None:
    schema_sql = (REPO / "sql/schema.sql").read_text(encoding="utf-8")
    blocks = ddl.existing_blocks(schema_sql)
    assert set(RENAMES.values()) <= set(blocks), sorted(set(RENAMES.values()) - set(blocks))
    assert not set(RENAMES) & set(blocks), sorted(set(RENAMES) & set(blocks))
    indexes = re.findall(r"CREATE (?:UNIQUE )?INDEX IF NOT EXISTS (\w+)", schema_sql)
    stale = [name for name in indexes if any(old in name for old in (*RENAMES, "entity_id"))]
    assert stale == [], f"index names still carrying an old table or column name: {stale}"
    assert "ix_universe_cik_symbol_company_id" in indexes
    assert '"company_id" TEXT' in blocks["universe_cik_symbol"]

    print("\n=== SANITY CHECK: sql/schema.sql identity blocks ===")
    print(f"  {len(blocks)} blocks, the three identity blocks under their new names; {len(indexes)} index names, none carries an old name.")
    print("  ix_universe_cik_symbol_company_id cannot collide with the old index still in Postgres. Validated.")


def test_no_old_identity_name_is_referenced() -> None:
    hits: list[str] = []
    scanned = 0
    for top in SCANNED:
        for path in sorted((REPO / top).rglob("*")):
            if path.suffix not in SUFFIXES or not path.is_file():
                continue
            rel = path.relative_to(REPO).as_posix()
            text = path.read_text(encoding="utf-8")
            if rel in ASK_FIRST or rel.startswith(ASK_FIRST_DIRS):
                text = _code_only(text)
            hits += _hits(rel, text)
            scanned += 1
    assert hits == [], f"{len(hits)} old identity name(s) still referenced:\n" + "\n".join(hits)

    print("\n=== SANITY CHECK: AC-012 rename sweep ===")
    print(f"  {scanned} files under {'/, '.join(SCANNED)}/ scanned; no tickers_universe, entity_lineage, security_master,")
    print("  entity_id or identity_history table reference left (package path, module and JSON file names excepted). Validated.")
