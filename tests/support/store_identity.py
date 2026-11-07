"""A store-backed `Identity` for fetcher tests that seed `universe_cusip` but publish no lineage.

A fetcher reads the published identity through `load_identity(context)` (U-70), which returns the resolver cached on
the context. `store_identity(store)` is such a resolver with no CIK or symbol rows; its `source_cohort` re-reads the
three identity tables on every access, so a test may seed or replace `universe_cusip` after building its context.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.data_extract.utils.common.identity import Identity, load_source_identity
from src.data_store.schema import Table
from src.data_store.store import DataStore


class _StoreCohort(dict):
    """`load_source_identity(store)`, read again on each lookup."""

    def __init__(self, store: DataStore) -> None:
        super().__init__()
        self._store = store

    def __getitem__(self, table: Table) -> pd.DataFrame:
        return load_source_identity(self._store)[table]

    def get(self, table: Table, default: Any = None) -> Any:
        return load_source_identity(self._store).get(table, default)

    def items(self) -> Any:
        return load_source_identity(self._store).items()


def store_identity(store: DataStore) -> Identity:
    """An `Identity` whose published rows are the store's current `universe_cusip` (and lineage, tenure) tables."""
    return Identity(
        entity_by_cik={},
        roster_cik={},
        ticker_by_entity={},
        tenure_by_symbol={},
        manual_tenure_by_symbol={},
        redundant_symbols=frozenset(),
        source_cohort=_StoreCohort(store),
    )
