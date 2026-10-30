"""Shared assembly probe: is a registered cube part physically populated?"""

from __future__ import annotations

from src.data_store.schema import Table
from src.data_store.store import DataStore


def part_has_rows(store: DataStore, table: Table) -> bool:
    """Probe physical presence before validating requested optional payload columns."""
    if not store.exists(table):
        return False
    probe = store.load(table, columns=store.columns(table)[:1], limit=1, optional=True)
    return probe is not None and not probe.empty
