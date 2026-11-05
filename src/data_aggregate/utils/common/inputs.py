"""Shared projected source loading for cube builders."""

from __future__ import annotations

import logging
from collections.abc import Sequence

import pandas as pd

from src.data_store.schema import Table
from src.data_store.store import DataStore

logger = logging.getLogger(__name__)


def load_source(store: DataStore, table: Table, universe: Sequence[str] | None = None) -> pd.DataFrame | None:
    """Stream one projected source payload in its economic scope."""
    filters = {table.ticker_col: sorted(set(universe))} if universe is not None and table.ticker_col else None
    chunks = list(store.iter_load(table, project=True, where=filters))
    df = pd.concat(chunks, ignore_index=True) if chunks else None
    if df is None or df.empty:
        logger.warning("%s is absent or empty -> its features are skipped.", table.name)
        return None
    logger.info("Loaded %s: %s rows x %s cols", table.name, len(df), len(df.columns))
    return df
