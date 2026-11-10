"""Create selected tables in a test database from the canonical hand-maintained `sql/schema.sql`."""

from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy import Connection, Engine, text

from src.data_store import ddl
from src.data_store.schema import Table


def create_canonical(engine: Engine | Connection, tables: Sequence[Table]) -> None:
    """Run each table's `sql/schema.sql` block (comments stripped) on `engine`, in one transaction."""
    statements = [statement for table in tables for statement in ddl.declared_statements(table.name)]
    if isinstance(engine, Connection):
        for statement in statements:
            engine.execute(text(statement))
        return
    with engine.begin() as conn:
        for statement in statements:
            conn.execute(text(statement))
