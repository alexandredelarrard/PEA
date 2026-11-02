"""Create selected tables in a test database from the canonical hand-maintained `sql/schema.sql`."""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

from sqlalchemy import Connection, Engine, text

from src.data_store import ddl
from src.data_store.schema import Table

_SCHEMA_SQL = Path(__file__).resolve().parents[1] / "sql" / "schema.sql"


def create_canonical(engine: Engine | Connection, tables: Sequence[Table]) -> None:
    """Run each table's `sql/schema.sql` block (comments stripped) on `engine`, in one transaction."""
    blocks = ddl.existing_blocks(_SCHEMA_SQL.read_text(encoding="utf-8"))
    statements = [statement for table in tables for statement in re.sub(r"--[^\n]*", "", blocks[table.name]).split(";") if statement.strip()]
    if isinstance(engine, Connection):
        for statement in statements:
            engine.execute(text(statement))
        return
    with engine.begin() as conn:
        for statement in statements:
            conn.execute(text(statement))
