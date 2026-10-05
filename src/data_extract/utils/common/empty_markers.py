"""Empty-filing marker rows (`Table.empty_marker`) and the rule that a marker never replaces data.

`marker_frame` builds one marker: the given key and stamp values, the declared sentinel, and a
typed NULL in every other column. A marker whose sentinel column is not part of the primary key
has the same key as the real row of that filing, so upserting it would overwrite stored data.
`drop_markers_over_data` removes such a marker before any save when a real row with its key is
stored or is in the same frame.
"""

from __future__ import annotations

import logging
from typing import cast

import pandas as pd

from src.data_extract.utils.common.frame_sanitize import pin_dtypes
from src.data_store.schema import Table
from src.data_store.store import DataStore

logger = logging.getLogger(__name__)


def marker_mask(table: Table, df: pd.DataFrame) -> pd.Series:
    """True on the rows of `df` that carry `table`'s empty-filing sentinel."""
    if table.empty_marker is None or table.empty_marker[0] not in df.columns:
        return pd.Series(False, index=df.index)
    column, sentinel = table.empty_marker
    return df[column].eq(cast("str | float | int", sentinel)).fillna(False).astype(bool)


def marker_frame(table: Table, values: dict[str, object], key_fill: dict[str, object] | None = None) -> pd.DataFrame:
    """One marker row of `table` from `values` (key, accession, frontier date and any stamp columns).

    The sentinel is set; a primary-key column not in `values` takes `key_fill`'s value, else the
    frontier date for a date column, else the sentinel. Absent columns are stored as NULL.
    """
    if table.resume is None or table.empty_marker is None or table.resume.frontier_col is None:
        raise ValueError(f"{table.name}: a marker needs a resume contract with a frontier column and an empty_marker")
    column, sentinel = table.empty_marker
    frontier = table.resume.frontier_col
    row = {**values, column: sentinel}
    for key_col in table.pk:
        if key_col not in row:
            row[key_col] = (key_fill or {}).get(key_col, row[frontier] if key_col in table.date_type_cols else sentinel)
    dates = [c for c in row if c in table.date_type_cols or c == frontier]
    return pin_dtypes(pd.DataFrame([row]), dates=dates)


def marker_shares_data_key(table: Table) -> bool:
    """True when a marker row has the same primary key as the real row of its filing."""
    return table.empty_marker is not None and table.empty_marker[0] not in table.pk


def _key_frame(table: Table, df: pd.DataFrame) -> pd.DataFrame:
    """`df`'s primary-key columns as comparable strings (dates normalised to ISO days)."""
    df_keys = pd.DataFrame(index=df.index)
    for column in table.pk:
        values = df[column]
        if column in table.date_type_cols:
            values = pd.to_datetime(values, errors="coerce").dt.strftime("%Y-%m-%d")
        df_keys[column] = values.astype(str)
    return df_keys


def drop_markers_over_data(store: DataStore, table: Table, df: pd.DataFrame, log: logging.Logger | None = None) -> pd.DataFrame:
    """`df` without the marker rows whose primary key already holds a real row, stored or in `df`.

    Only tables whose marker shares the real row's key can collide; any other frame is returned
    unchanged without a read. The stored check is one read projected on the key and scoped to the
    markers' own key values (markers excluded).
    """
    if df is None or df.empty or not marker_shares_data_key(table) or not set(table.pk) <= set(df.columns):
        return df
    is_marker = marker_mask(table, df)
    if not is_marker.any():
        return df
    df_keys = _key_frame(table, df)
    df_markers = df[is_marker]
    scope = {c: sorted({str(v) for v in df_markers[c].dropna()}) for c in table.pk if c not in table.date_type_cols}
    df_stored = store.load(table, columns=list(table.pk), where=scope, optional=True)
    real = [df_keys[~is_marker]] + ([_key_frame(table, df_stored)] if df_stored is not None else [])
    taken = set(pd.concat(real, ignore_index=True).itertuples(index=False, name=None))
    over_data = is_marker & pd.Series([key in taken for key in df_keys.itertuples(index=False, name=None)], index=df.index)
    if not over_data.any():
        return df
    (log or logger).info(
        "%s: %d empty-filing marker(s) not written: their key already holds data (e.g. %s)",
        table.name,
        int(over_data.sum()),
        ", ".join(df.loc[over_data, "accession_number"].astype(str).head(5)) if "accession_number" in df.columns else "",
    )
    return df[~over_data]
