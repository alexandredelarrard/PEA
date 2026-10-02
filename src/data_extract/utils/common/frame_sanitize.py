"""
frame_sanitize.py  (src/data_extract/utils/common/frame_sanitize.py)
---------------------------------------------------------------------
Make a freshly-built DataFrame safe to hand to Postgres.

`finalise_frame` turns builder rows into an upsert-ready frame: declared columns, one row per
primary key, numeric columns coerced.

`strip_nul`: a Postgres TEXT column cannot store a NUL byte
(psycopg2 raises "a string literal cannot contain NUL characters" and takes the whole
INSERT down with it, not just the offending row). Filing-derived text is where NULs come
from — DEF 14A and 8-K documents are HTML- or PDF-derived, so an extracted string
occasionally carries a stray `\\x00`.

Shared rather than duplicated: this is a correctness invariant for every LLM-extract
fetcher that writes filing text, and two copies of it are how one copy later drifts.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from src.data_store.schema import Table


def strip_nul(df: pd.DataFrame) -> pd.DataFrame:
    """Remove NUL (`\\x00`) characters from every string cell, in place, returning `df`.

    Dtype-agnostic (pandas 2 `object` AND pandas 3 `str` string columns): only the columns
    that actually hold a NUL-bearing string are rewritten, so numeric / datetime columns keep
    their dtype instead of being round-tripped through `object`.
    """
    for c in df.columns:
        col = df[c]
        if col.map(lambda v: isinstance(v, str) and "\x00" in v).any():
            df[c] = col.map(lambda v: v.replace("\x00", "") if isinstance(v, str) else v)
    return df


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
