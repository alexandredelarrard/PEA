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
from dataclasses import dataclass
from typing import Sequence

import pandas as pd

from src.constants.constants import PANEL_KEYS
from src.data_store.store import DataStore

# `write_part` returns this instead of a row count when the part's feature set changed, to
# tell the caller "your column set no longer matches the stored table -- re-run full".
COLUMNS_CHANGED = -1

#: How many trading days of a backward-looking part's own tail every incremental run
#: recomputes and REWRITES, rather than only appending after.
#:
#: One trading week. The bound that matters is the price fetcher's own re-pull floor
#: (`PRICE_REFRESH_TRADING_DAYS = 7` business days): a part must rewrite at least as far back
#: as its inputs can still change underneath it, or a corrected price would sit in `prices`
#: with the stale feature built from its predecessor left in the part forever. 5 trading days
#: covers 7 business days of calendar (they are the same span; the fetcher counts BDays from
#: the settled close, the part counts sessions on the trading index).
#:
#: The cost is bounded and small: 5 dates x ~491 tickers re-computed and re-written per part
#: per run, against parts of ~3.3M rows. `store.append_tail(inclusive=True)` DELETEs `>=` the
#: cutoff before appending, so re-running the same day is idempotent -- it never duplicates
#: and never leaves a stale row behind.
PART_REFRESH_TRADING_DAYS = 5

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


def window_start(trading_index: pd.DatetimeIndex, last: pd.Timestamp,
                 n_back: int) -> pd.Timestamp:
    """The date `n_back` trading days BEFORE `last` on the (untrimmed) price calendar."""
    pos = int(trading_index.searchsorted(pd.Timestamp(last).normalize()))
    return trading_index[max(0, pos - n_back)]


def plan_window(store: DataStore, part: str, *, warmup: int, full: bool,
                trading_index: pd.DatetimeIndex | None = None,
                extra_back: int = 0, refresh: int = 0) -> PartWindow:
    """Decide what to rebuild.

    `full=True`, a missing part, or no usable calendar -> a full rebuild. Otherwise the
    window reaches `warmup + extra_back + refresh` trading days before the stored max date;
    `extra_back` is the target step's forward horizon (so maturing labels are recomputed) and
    `refresh` is how far back the part REWRITES its own tail.

    ⚠ `warmup + extra_back + refresh`, not `warmup + extra_back`. The warm-up has to be
    measured from the earliest REWRITTEN date, not from `last`, or the oldest refreshed date
    gets only `warmup - refresh` days of look-back context and is computed differently from
    the way a full rebuild would compute it. For momentum that would be 1,320 - 5 = 1,315
    days against a binding look-back of 1,260: it happens to survive today purely on margin,
    and would break silently the moment either number moved. Adding `refresh` removes the
    coupling instead of relying on the slack.
    """
    if full:
        return PartWindow(None, None)
    last = store.max_date(part)
    if last is None or trading_index is None or len(trading_index) == 0:
        return PartWindow(None, None)
    return PartWindow(
        last,
        window_start(trading_index, last, warmup + extra_back + refresh),
        window_start(trading_index, last, refresh) if refresh else None)


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
    since an append into a changed schema would silently misalign); otherwise write the
    tail from the widest cutoff on offer:

      1. an explicit `refresh_from` -- the target step's maturing-label window (~90 trading
         days), which is always the widest and so takes precedence;
      2. `window.refresh_from` -- the backward-looking part's own trailing rewrite;
      3. `window.last`, strictly after -- the pre-refresh behaviour, kept for the parts that
         opt out (fundamentals / text / extras, all driven by filing-space sources rather
         than the daily price grid).

    Cases 1 and 2 write INCLUSIVELY, so the cutoff date's own row is replaced rather than
    skipped. `store.append_tail(inclusive=True)` DELETEs `>=` the cutoff first, which makes
    a same-day re-run idempotent.

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

    cutoff = refresh_from if refresh_from is not None else window.refresh_from
    inclusive = cutoff is not None
    if cutoff is None:
        cutoff = window.last
    tail = rows[rows["date"] >= cutoff] if inclusive else rows[rows["date"] > cutoff]
    n = store.append_tail(part, tail, cutoff, inclusive=inclusive)
    logger.info("Appended %s (INCREMENTAL): +%s rows %s %s.", part, n,
             ">=" if inclusive else ">", pd.Timestamp(cutoff).date())
    return n
