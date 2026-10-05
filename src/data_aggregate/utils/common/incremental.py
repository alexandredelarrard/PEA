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
  * BACKWARD-looking parts (features, betas) rewrite their trailing `refresh` window
    INCLUSIVELY and append everything after it. Strictly-after used to be enough on the
    theory that a stored date is final once written -- it is not. A part's last stored date
    is the one most likely to be WRONG, because it is the date whose price inputs were newest
    and least settled: a truncated extract run left `cube_part_momentum` stopping ON a date
    ranked over 45 of 491 tickers, and because the append started strictly after it, no
    incremental run could ever have replaced that row. Only a `--full` rebuild would.
  * FORWARD-looking targets must ALSO refresh the trailing `max_horizon` window, because
    a label that was NaN last run (no future price yet) MATURES into a value between runs.
    That is the same mechanism with a much wider window, and it takes precedence.

The old code loaded the whole history and then called `_trim_window` to throw ~90% of it
away. Here the window is decided FIRST, from the market part's dates alone (~15k rows),
and the trim is pushed into SQL.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from src.constants.constants import PANEL_KEYS
from src.data_store.schema import Table
from src.data_store.store import DataStore

# `write_part` returns this instead of a row count when the part's feature set changed, to
# tell the caller "your column set no longer matches the stored table -- re-run full".
COLUMNS_CHANGED = -1

#: How many trading days of its own tail every incremental part run recomputes and REWRITES.
#:
#: Seven sessions cover the price fetcher's 7-day re-pull overlap, and let a filing recovered
#: within a week replace the features first built without it. Fundamentals (45) and targets
#: (their maturing-label horizon) rewrite wider tails. `store.append_tail(inclusive=True)`
#: deletes `>=` the cutoff before appending, so a same-day re-run is idempotent.
PART_REFRESH_TRADING_DAYS = 7

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PartWindow:
    """`last` is the part's stored max date (None -> full rebuild); `since` is the first
    date to READ and compute from (warm-up padded); `refresh_from` is the first date to
    REWRITE (None -> append strictly after `last`, the pre-refresh behaviour)."""

    last: pd.Timestamp | None
    since: pd.Timestamp | None
    refresh_from: pd.Timestamp | None = None

    @property
    def is_full(self) -> bool:
        return self.last is None


def window_start(trading_index: pd.DatetimeIndex, last: pd.Timestamp, n_back: int) -> pd.Timestamp:
    """The date `n_back` trading days BEFORE `last` on the (untrimmed) price calendar."""
    pos = int(trading_index.searchsorted(pd.Timestamp(last).normalize()))
    return trading_index[max(0, pos - n_back)]


def plan_window(
    store: DataStore, part: Table, *, warmup: int, full: bool, trading_index: pd.DatetimeIndex | None = None, extra_back: int = 0, refresh: int = 0
) -> PartWindow:
    """Decide what to rebuild.

    `full=True`, a missing part, or no usable calendar -> a full rebuild. Otherwise the
    window reaches `warmup + extra_back + refresh` trading days before the stored max date:
    `extra_back` is the target step's forward horizon (so maturing labels are recomputed) and
    `refresh` is how far back the part rewrites its own tail. The warm-up counts from the
    earliest rewritten date, so every rewritten row gets the look-back a full rebuild gives it.
    """
    if full:
        return PartWindow(None, None)
    last = store.max_date(part)
    if last is None or trading_index is None or len(trading_index) == 0:
        return PartWindow(None, None)
    return PartWindow(
        last, window_start(trading_index, last, warmup + extra_back + refresh), window_start(trading_index, last, refresh) if refresh else None
    )


def drop_empty_feature_rows(rows: pd.DataFrame, keys: Sequence[str], part: Table) -> pd.DataFrame:
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
        logger.info("%s: dropped %s all-NaN grid rows (%.1f%% of %s)", part, dropped, 100 * dropped / len(rows), len(rows))
    return rows[keep]


def write_part(
    store: DataStore,
    part: Table,
    rows: pd.DataFrame,
    window: PartWindow,
    *,
    keys: Sequence[str] = tuple(PANEL_KEYS),
    refresh_from: pd.Timestamp | None = None,
    drop_empty: bool = False,
) -> int:
    """Persist a part according to `window`.

    FULL -> replace. INCREMENTAL -> `COLUMNS_CHANGED` when the stored column set differs
    from `rows` (the caller re-runs with full=True); otherwise append the tail from the first
    cutoff given: an explicit `refresh_from` (the target step's maturing-label window), then
    `window.refresh_from` (the part's own trailing rewrite), both inclusive because
    `append_tail` deletes `>=` the cutoff first, so a same-day re-run is idempotent; else
    strictly after `window.last`. `drop_empty` removes rows with no feature value at all.
    """
    if rows is not None and not rows.empty and drop_empty:
        rows = drop_empty_feature_rows(rows, keys, part)

    if rows is None or rows.empty:
        logger.warning("%s produced no rows -> nothing persisted.", part)
        return 0

    if window.is_full:
        n = store.replace(part, rows)
        logger.info("Persisted %s (FULL): %s rows x %s cols.", part, n, len([c for c in rows.columns if c not in keys]))
        return n

    # `columns` returns [] for a table that does not exist -- "no stored column set to
    # compare against", not "a stored set that differs". Treating [] as a difference would
    # report COLUMNS_CHANGED for an absent part.
    existing = store.columns(part)
    if existing and set(existing) != set(rows.columns):
        logger.warning("%s column set changed (%s stored vs %s built) -> full rebuild needed.", part, len(existing), len(rows.columns))
        return COLUMNS_CHANGED

    cutoff = refresh_from if refresh_from is not None else window.refresh_from
    inclusive = cutoff is not None
    if cutoff is None:
        cutoff = window.last
    assert cutoff is not None
    tail = rows[rows["date"] >= cutoff] if inclusive else rows[rows["date"] > cutoff]
    n = store.append_tail(part, tail, cutoff, inclusive=inclusive)
    logger.info("Appended %s (INCREMENTAL): +%s rows %s %s.", part, n, ">=" if inclusive else ">", pd.Timestamp(cutoff).date())
    return n
