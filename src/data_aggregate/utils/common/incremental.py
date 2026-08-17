"""
incremental.py  (src/data_aggregate/utils/common/incremental.py)
--------------------------------------------------------------
The full-vs-incremental decision, in ONE place.

Every cube sub-step follows the same rule: read the part's latest stored date, recompute
only a warm-up-padded trailing window, and append the rows after that date -- instead of
truncating and reloading fifteen years. This is only correct because the feature builders
are backward-looking (window <= warm-up) and the cross-sectional standardization is
per-day, so a trailing recompute reproduces the full build's tail exactly. That
equivalence is proved on the price builder by
`tests/data_aggregate/test_cube_incremental.py`.

Two shapes of write:
  * BACKWARD-looking parts (features, betas) append dates strictly after the stored max.
  * FORWARD-looking targets must ALSO refresh the trailing `max_horizon` window, because
    a label that was NaN last run (no future price yet) MATURES into a value between runs.

The old code loaded the whole history and then called `_trim_window` to throw ~90% of it
away. Here the window is decided FIRST, from the market part's dates alone (~15k rows),
and the trim is pushed into SQL.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import pandas as pd

from src.constants.constants import PANEL_KEYS
from src.data_store.store import DataStore

# `write_part` returns this instead of a row count when the part's feature set changed, to
# tell the caller "your column set no longer matches the stored table -- re-run full".
COLUMNS_CHANGED = -1

logger = logging.getLogger(__name__)

@dataclass(frozen=True, slots=True)
class PartWindow:
    """`last` is the part's stored max date (None -> full rebuild); `since` is the first
    date to READ and compute from (warm-up padded)."""
    last: pd.Timestamp | None
    since: pd.Timestamp | None

    @property
    def is_full(self) -> bool:
        return self.last is None


def window_start(trading_index: pd.DatetimeIndex, last: pd.Timestamp,
                 n_back: int) -> pd.Timestamp:
    """The date `n_back` trading days BEFORE `last` on the (untrimmed) price calendar."""
    pos = int(trading_index.searchsorted(pd.Timestamp(last).normalize()))
    return trading_index[max(0, pos - n_back)]


def plan_window(store: DataStore, part: str, *, warmup: int, full: bool,
                trading_index: pd.DatetimeIndex | None = None,
                extra_back: int = 0) -> PartWindow:
    """Decide what to rebuild.

    `full=True`, a missing part, or no usable calendar -> a full rebuild. Otherwise the
    window reaches `warmup + extra_back` trading days before the stored max date;
    `extra_back` is the target step's forward horizon (so maturing labels are recomputed).
    """
    if full:
        return PartWindow(None, None)
    last = store.max_date(part)
    if last is None or trading_index is None or len(trading_index) == 0:
        return PartWindow(None, None)
    return PartWindow(last, window_start(trading_index, last, warmup + extra_back))


def drop_empty_feature_rows(rows: pd.DataFrame, keys: Sequence[str],
                            part: str) -> pd.DataFrame:
    """Drop (date, ticker) rows where EVERY feature is NaN.

    The merge-based builders left-join onto the full universe grid, so a name with no
    coverage at all — before its IPO, outside a sector gate, or simply absent from a source
    — still gets a row. Persisting those would store the whole 1.85M-cell grid per part
    regardless of how sparse the features are, and then carry it through the assemble merge.
    """
    fcols = [c for c in rows.columns if c not in set(keys)]
    if not fcols:
        return rows.iloc[0:0]
    keep = rows[fcols].notna().any(axis=1)
    dropped = int((~keep).sum())
    if dropped:
        logger.info("%s: dropped %s all-NaN grid rows (%.1f%% of %s)", part, dropped,
                 100 * dropped / len(rows), len(rows))
    return rows[keep]


def write_part(store: DataStore, part: str, rows: pd.DataFrame, window: PartWindow,
               *, keys: Sequence[str] = tuple(PANEL_KEYS),
               refresh_from: pd.Timestamp | None = None,
               drop_empty: bool = False) -> int:
    """Persist a part according to `window`.

    FULL -> replace. INCREMENTAL -> compare the stored column set against `rows` and
    return `COLUMNS_CHANGED` when they differ (the caller must re-run with full=True,
    since an append into a changed schema would silently misalign); otherwise append the
    tail after `refresh_from or window.last`, inclusive when `refresh_from` is given
    (that is the maturing-label overwrite).

    `drop_empty` (feature parts) removes rows carrying no feature values at all.
    """
    if rows is not None and not rows.empty and drop_empty:
        rows = drop_empty_feature_rows(rows, keys, part)
    if rows is None or rows.empty:
        logger.warning("%s produced no rows -> nothing persisted.", part)
        return 0

    if window.is_full:
        n = store.replace(part, rows)
        logger.info("Persisted %s (FULL): %s rows x %s cols.", part, n,
                 len([c for c in rows.columns if c not in keys]))
        return n

    # `columns` returns [] for a table that does not exist -- "no stored column set to
    # compare against", not "a stored set that differs". Treating [] as a difference would
    # report COLUMNS_CHANGED for an absent part.
    existing = store.columns(part)
    if existing and set(existing) != set(rows.columns):
        logger.warning("%s column set changed (%s stored vs %s built) -> full rebuild needed.",
                    part, len(existing), len(rows.columns))
        return COLUMNS_CHANGED

    cutoff = refresh_from if refresh_from is not None else window.last
    inclusive = refresh_from is not None
    tail = rows[rows["date"] >= cutoff] if inclusive else rows[rows["date"] > cutoff]
    n = store.append_tail(part, tail, cutoff, inclusive=inclusive)
    logger.info("Appended %s (INCREMENTAL): +%s rows %s %s.", part, n,
             ">=" if inclusive else ">", pd.Timestamp(cutoff).date())
    return n
