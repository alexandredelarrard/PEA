"""
part_io.py  (src/data_aggregate/utils/common/part_io.py)
------------------------------------------------------
Lifecycle of the AD-HOC `cube_part_*` tables: existence, latest date, column set, full
replace, and the idempotent tail append the incremental rebuild needs.

WHY THIS OWNS RAW SQL. These tables are deliberately absent from
`src/data_store/schema_registry.py` -- they are private plumbing between cube sub-steps,
rebuilt by their owning step, with no PK and no place in `sql/schema.sql`. That has one
hard consequence: `DataStore.replace` -> `ensure_table` -> `BY_NAME[name]` raises
`KeyError` for any name not in the registry. So `replace` PRE-CREATES the table from the
frame's own dtypes via `to_sql(head(0))` and only then hands it to the fast COPY path.
That pre-create is load-bearing, not cosmetic.

`read_since` exists because `DataStore.load`'s `where=` is equality/IN only, while the
incremental path needs `date >= :since` pushed into SQL -- otherwise every step would load
15 years and throw ~90% of it away, which is exactly what the old `_trim_window` did.

USED ONLY BY THE PART-BUILDING STEPS. The assemble step must keep using the plain `store`
facade (exists/load/replace/bulk_seed) so it stays testable against a store with no
engine; `tests/data_aggregate/test_assemble_cube.py` enforces that by making `engine`
raise.
"""
from __future__ import annotations

import logging
from typing import Sequence

import pandas as pd
from sqlalchemy import text

from src.data_store.store import DataStore


class PartStore:
    """Read/write helpers for the ad-hoc `cube_part_*` tables."""

    def __init__(self, store: DataStore, log: logging.Logger | None = None) -> None:
        self._store = store
        self._log = log or logging.getLogger(__name__)

    # ---- introspection ---- #
    def exists(self, part: str) -> bool:
        return bool(self._store.exists(part))

    def max_date(self, part: str) -> pd.Timestamp | None:
        """Latest `date` already stored (None if the table is absent/empty/oddly shaped)."""
        if not self.exists(part):
            return None
        try:
            with self._store.engine.connect() as c:
                v = c.execute(text(f'SELECT MAX(date) FROM "{part}"')).scalar()
        except Exception:                          # noqa: BLE001 (table shape unexpected)
            return None
        return None if v is None else pd.Timestamp(v).normalize()

    def columns(self, part: str) -> list[str] | None:
        """Column names of an existing part (None if absent) -- used to detect a
        feature-set change that would break an append."""
        if not self.exists(part):
            return None
        try:
            with self._store.engine.connect() as c:
                return list(c.execute(text(f'SELECT * FROM "{part}" LIMIT 0')).keys())
        except Exception:                          # noqa: BLE001
            return None

    def row_count(self, part: str) -> int:
        if not self.exists(part):
            return 0
        with self._store.engine.connect() as c:
            return int(c.execute(text(f'SELECT COUNT(*) FROM "{part}"')).scalar())

    # ---- reads ---- #
    def read(self, part: str, columns: Sequence[str] | None = None,
             since: pd.Timestamp | None = None) -> pd.DataFrame:
        """Read a part, optionally projected to `columns` and trimmed to `date >= since`.

        The `since` filter is pushed into SQL: the incremental steps read a few hundred
        trading days instead of fifteen years."""
        cols = list(columns) if columns else None
        if since is None:
            return self._store.load(part, columns=cols)
        select = ", ".join(f'"{c}"' for c in cols) if cols else "*"
        sql = text(f'SELECT {select} FROM "{part}" WHERE date >= :since')
        with self._store.engine.connect() as c:
            return pd.read_sql(sql, c, params={"since": pd.Timestamp(since).strftime("%Y-%m-%d")})

    # ---- writes ---- #
    def replace(self, part: str, df: pd.DataFrame) -> int:
        """Full rebuild. Pre-creates the table from the frame's OWN dtypes so
        `store.replace`'s registry-PK lookup is skipped (these tables are intentionally
        not in schema_registry.py and `store.replace` would raise KeyError), then lets
        the store do the fast COPY load."""
        if df is None or df.empty:
            return 0
        df.head(0).to_sql(part, self._store.engine, if_exists="replace", index=False)
        return int(self._store.replace(part, df))

    def append_tail(self, part: str, df: pd.DataFrame, cutoff: pd.Timestamp,
                    inclusive: bool = False) -> int:
        """Idempotently replace the tail: DELETE rows with date > cutoff (>= if
        `inclusive`), then append `df`. Idempotent so re-running the same day never
        duplicates rows."""
        op = ">=" if inclusive else ">"
        cut = pd.Timestamp(cutoff).strftime("%Y-%m-%d")
        with self._store.engine.begin() as c:
            c.execute(text(f'DELETE FROM "{part}" WHERE date {op} :d'), {"d": cut})
        if df is None or df.empty:
            return 0
        df.to_sql(part, self._store.engine, if_exists="append", index=False)
        return len(df)

    def drop(self, part: str) -> None:
        """Drop a superseded part table (used by the migration off the old per-group parts)."""
        with self._store.engine.begin() as c:
            c.execute(text(f'DROP TABLE IF EXISTS "{part}"'))


def normalize_date_col(df: pd.DataFrame | None) -> pd.DataFrame | None:
    """Normalize a part's `date` column to midnight so merges across parts align."""
    if df is not None and not df.empty and "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df


def downcast_float32(df: pd.DataFrame | None) -> pd.DataFrame | None:
    """Downcast float64 -> float32 (feature z-scores / ranks / returns need no float64
    precision) -> halves the wide panel and every horizon slice built from it. Keys, ints
    and object columns untouched."""
    if df is None or df.empty:
        return df
    f64 = df.select_dtypes(include=["float64"]).columns
    if len(f64):
        df[f64] = df[f64].astype("float32")
    return df
