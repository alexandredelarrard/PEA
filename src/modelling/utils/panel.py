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
from src.modelling.utils.features import downcast_f32


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
) -> pd.DataFrame | None:
    """Projected, row-scoped `store.load`; float64 -> float32 when `downcast`. None for an absent
    or empty table when `optional` (otherwise the store raises)."""
    frame = store.load(table, columns=columns, where=where, since=since, until=until, optional=optional)
    if frame is None:
        return None
    return downcast_f32(frame) if downcast else frame
