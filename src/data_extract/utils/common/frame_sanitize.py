"""
frame_sanitize.py  (src/data_extract/utils/common/frame_sanitize.py)
---------------------------------------------------------------------
Make a freshly-built DataFrame safe to hand to Postgres.

`finalise_frame` turns builder rows into an upsert-ready frame: declared columns, one row per
primary key, numeric columns coerced.

`strip_nul`: a Postgres TEXT column cannot store a NUL byte, and psycopg2 rejects the whole
INSERT when one cell holds one. Filing-derived text (DEF 14A, 8-K) occasionally carries a stray
`\x00`.

`pin_dtypes`: one dtype per column whatever the rows held (dates, floats, nullable text), so a
frame's dtypes never depend on whether a column happened to be all-null.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from pandas.api.types import infer_dtype, is_string_dtype

from src.data_store.schema import Table

#: `infer_dtype` kinds of a column that can hold a `str` cell (a pure-bytes, numeric or
#: all-null object column cannot).
_TEXT_KINDS = frozenset({"string", "mixed", "mixed-integer"})


def strip_nul(df: pd.DataFrame) -> pd.DataFrame:
    """Remove NUL (`\x00`) characters from every string cell, in place, returning `df`.

    Only object / string columns that hold a NUL are rewritten; a rewritten column is re-inferred
    the way pandas infers a mapped column (an all-text object column becomes `str`).
    """
    for column in df.columns[[is_string_dtype(dtype) for dtype in df.dtypes]]:
        values = df[column]
        if infer_dtype(values, skipna=True) not in _TEXT_KINDS:
            continue
        has_nul = np.asarray(values.str.contains("\x00", regex=False, na=False), dtype=bool)
        if has_nul.any():
            cells = values.to_numpy(dtype=object, copy=True)
            cells[has_nul] = values[has_nul].str.replace("\x00", "", regex=False).to_numpy(dtype=object)
            df[column] = pd.Series(cells, index=df.index)
    return df


def pin_dtypes(df: pd.DataFrame, *, dates: Sequence[str] = (), floats: Sequence[str] = (), texts: Sequence[str] = ()) -> pd.DataFrame:
    """`df` with `dates` as `datetime64[ns]`, `floats` as `float64` (both coerced, NaT/NaN on
    failure) and `texts` as `object` holding None for every missing cell."""
    pinned = {column: pd.to_datetime(df[column], errors="coerce").astype("datetime64[ns]") for column in dates}
    pinned |= {column: pd.to_numeric(df[column], errors="coerce").astype(float) for column in floats}
    pinned |= {column: df[column].astype(object).where(df[column].notna(), None) for column in texts}
    return df.assign(**pinned)


def finalise_frame(table: Table, rows: list[dict], *, columns: Sequence[str], numeric: Sequence[str] = ()) -> pd.DataFrame:
    """`rows` as a frame with exactly `columns`, de-duplicated on `table.pk` (last row wins), then
    `numeric` columns coerced with `pd.to_numeric(errors="coerce")`.

    An upsert touching one primary key twice is an error in Postgres, so the dedup is required;
    coercion runs after it so no work is spent on dropped rows.
    """
    df = pd.DataFrame(rows, columns=list(columns)).drop_duplicates(subset=list(table.pk), keep="last")
    for column in numeric:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df
