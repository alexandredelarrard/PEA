"""
frame_sanitize.py  (src/data_extract/utils/common/frame_sanitize.py)
---------------------------------------------------------------------
Make a freshly-built DataFrame safe to hand to Postgres.

One job, one invariant: a Postgres TEXT column cannot store a NUL byte
(psycopg2 raises "a string literal cannot contain NUL characters" and takes the whole
INSERT down with it, not just the offending row). Filing-derived text is where NULs come
from — DEF 14A and 8-K documents are HTML- or PDF-derived, so an extracted string
occasionally carries a stray `\\x00`.

Shared rather than duplicated: this is a correctness invariant for every LLM-extract
fetcher that writes filing text, and two copies of it are how one copy later drifts.
"""
from __future__ import annotations

import pandas as pd


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
