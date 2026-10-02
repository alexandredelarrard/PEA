"""
parallel_fetch.py (src/data_extract/utils/common/parallel_fetch.py)
---------------------------------------------------------------------
Bounded thread pool for the per-entity EDGAR walks. Each walk is network I/O spaced by
edgartools' shared, thread-safe rate limiter (~9 req/sec globally), so several entities in
flight keep every request under SEC's cap while actually using it; a sequential walk has one
request in flight and is latency-bound. Writers call `context.store.save` directly: the store
serializes the CREATE of a cold table.

Does NOT apply to `fetch_def14a_llm.py`, which is bound by OpenAI's rate limits and cost.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import pandas as pd
from tqdm import tqdm

DEFAULT_WORKERS = 8  # network-bound; edgartools' own client caps ~9 req/sec globally

#: Exception classes that mean this pipeline is broken, not the source record. They are
#: re-raised wherever a per-entity or per-filing handler would otherwise swallow them, so a
#: repo defect fails the run instead of being logged once per ticker. `KeyError` is included:
#: on these paths it means a frame's column contract broke. A narrow `except` around a
#: library parse (`filing.xbrl()`) still absorbs everything.
PROGRAMMING_ERRORS = (NameError, AttributeError, TypeError, KeyError, ImportError)


def run_per_ticker[R](
    df_scope: pd.DataFrame,
    worker: Callable[..., R],
    desc: str,
    *,
    log: logging.Logger,
    max_workers: int | None = None,
    key_cols: Sequence[str] = ("ticker", "cik"),
) -> list[R | None]:
    """Call `worker(*row[key_cols])` for every row of `df_scope` on a bounded pool.

    Returns results in `df_scope` row order. `PROGRAMMING_ERRORS` abort the pool; any other
    exception is logged as a warning keyed on the row's first key column and yields None."""

    def _guarded(key: Any, *args: Any) -> R | None:
        try:
            return worker(key, *args)
        except PROGRAMMING_ERRORS:
            raise
        except Exception as exc:  # noqa: BLE001 -- one entity must not abort the walk
            log.warning("%s: %s failed (%s)", desc, key, exc)
            return None

    rows = list(df_scope[list(key_cols)].itertuples(index=False, name=None))
    with ThreadPoolExecutor(max_workers=DEFAULT_WORKERS if max_workers is None else max_workers) as pool:
        futures = [pool.submit(_guarded, *row) for row in rows]
        for future in tqdm(as_completed(futures), total=len(futures), desc=desc):
            future.result()
    return [future.result() for future in futures]
