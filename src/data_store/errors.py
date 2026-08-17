"""
Typed store failures, so "no data yet" stops being indistinguishable from "the read is
broken". `load` used to raise a bare `Exception` for both, so callers wrapped it in
`except Exception` and swallowed mistyped columns and dead connections too.
"""
from __future__ import annotations


class StoreError(Exception):
    """Base class for every store failure."""


class TableMissingError(StoreError, LookupError):
    def __init__(self, table: str) -> None:
        super().__init__(f"table {table!r} does not exist")
        self.table = table


class TableEmptyError(StoreError, LookupError):
    def __init__(self, table: str, where: object = None) -> None:
        detail = f" (where={where!r})" if where else ""
        super().__init__(f"table {table!r} returned no rows{detail}")
        self.table = table
        self.where = where
