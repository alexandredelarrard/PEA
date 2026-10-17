"""
panel.py  (src/modelling/utils/panel.py)
----------------------------------------
THE projected loader every modelling read goes through (`context.store.load`, never SQL).
Training reads are downcast to float32; prediction reads stay float64, because the linear
member scores in float64 and a float32 round-trip would move its output.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.data_store.schema import Table
from src.modelling.utils.features import downcast_f32, prepare_features


def _prepare_loaded_frame(
    frame: pd.DataFrame, feature_cols: list[str] | None, categorical_features: list[str] | None, downcast: bool
) -> pd.DataFrame:
    if feature_cols:
        columns = [column for column in dict.fromkeys(feature_cols) if column in frame.columns]
        frame[columns] = prepare_features(frame[columns], categorical_features, downcast=downcast)
    return downcast_f32(frame) if downcast else frame


def load_frame(
    store: Any,
    table: Table,
    *,
    columns: list[str] | None = None,
    where: dict | None = None,
    since: object = None,
    until: object = None,
    downcast: bool = True,
    optional: bool = True,
    stream: bool = False,
    feature_cols: list[str] | None = None,
    categorical_features: list[str] | None = None,
    chunksize: int | None = None,
) -> pd.DataFrame | None:
    """Projected, row-scoped `store.load`; float64 -> float32 when `downcast`. None for an absent
    or empty table when `optional` (otherwise the store raises)."""
    if stream:
        batch = {"chunksize": chunksize} if chunksize is not None else {}
        chunks = [
            _prepare_loaded_frame(chunk, feature_cols, categorical_features, downcast)
            for chunk in store.iter_load(table, columns=columns, where=where, since=since, until=until, **batch)
        ]
        frame = pd.concat(chunks, ignore_index=True) if chunks else None
    else:
        frame = store.load(table, columns=columns, where=where, since=since, until=until, optional=optional)
    if frame is None:
        return None
    return _prepare_loaded_frame(frame, feature_cols, categorical_features, downcast) if not stream else frame
