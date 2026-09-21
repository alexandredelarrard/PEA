"""
frame.py  (src/validate/frame.py)
--------------------------------------------------------------------------------------------
The three things every check needs before it can measure anything: which columns are the
grain, which are the numeric legs under test, and how to compare two dates without lying.

`as_ts` is not defensive tidying. Postgres DATE round-trips through psycopg2 as
`datetime.date` while TIMESTAMP arrives as `pd.Timestamp`, and `datetime.date(2024, 1, 1) <
pd.Timestamp("2024-01-01")` raises rather than comparing -- so a PIT check that reads
`sec_fails_to_deliver.date` (DATE) against `cube_part_*.date` (TIMESTAMP) crashes, and one
that reads two DATE columns silently compares day-resolution values against nanosecond ones.
Both sides go through here.
"""
from __future__ import annotations

from typing import Any, Sequence

import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from src.context import Context
from src.data_store.schema import Table, resolve

#: Rows pulled to learn a table's dtypes. Wide enough that a leg is not judged on one row,
#: small enough that it costs nothing on a 249-column part.
_DTYPE_SAMPLE_ROWS = 200


def key_columns(table: Table | str) -> tuple[str, ...]:
    """The DECLARED grain. Never guessed: cube parts key on `(date, ticker)` and
    `fundamentals_history` on `(ticker, as_of)`, and a check that assumes the first is a
    check that reports a clean grain on the second by measuring the wrong thing."""
    return resolve(table).pk


def feature_columns(context: Context, table: Table | str, *,
                    frame: pd.DataFrame | None = None) -> list[str]:
    """The numeric columns under test: everything that is not part of the grain.

    Dtypes come from a `LIMIT` sample rather than from reflection, because the store facade
    exposes column NAMES but not their SQL types -- and reaching past it for
    `information_schema` is exactly what `tests/data_store/test_store_boundary.py` forbids.
    Booleans are excluded: a flag has no percentile and no correlation worth the name."""
    keys = set(key_columns(table))
    if frame is None:
        frame = context.store.load(table, limit=_DTYPE_SAMPLE_ROWS)
    return [c for c in frame.columns
            if c not in keys and is_numeric_dtype(frame[c]) and not is_bool_dtype(frame[c])]


def as_ts(values: Any) -> Any:
    """A scalar, Series or Index of dates as nanosecond `pd.Timestamp`s. See the module
    docstring for the DATE-vs-TIMESTAMP trap this closes."""
    if values is None:
        return None
    if isinstance(values, pd.Series):
        return pd.to_datetime(values, errors="coerce").astype("datetime64[ns]")
    if isinstance(values, pd.Index):
        return pd.DatetimeIndex(pd.to_datetime(values, errors="coerce")).as_unit("ns")
    if isinstance(values, (list, tuple)):
        return [as_ts(v) for v in values]
    try:
        stamp = pd.Timestamp(values)
    except (TypeError, ValueError):
        return pd.NaT
    return stamp.as_unit("ns") if stamp is not pd.NaT else pd.NaT


def column_groups(columns: Sequence[str], size: int) -> list[list[str]]:
    """`columns` in blocks of `size`. The memory control behind `profile`: a full-table read
    of `cube_part_fundamentals` is 3,315,035 x 249 x 8B = 6.6 GB, while one group of 8 is
    210 MB."""
    if size < 1:
        raise ValueError(f"column group size must be >= 1, got {size}")
    return [list(columns[i:i + size]) for i in range(0, len(columns), size)]
