"""Bounded thread pool for the per-entity EDGAR walks.

Requests are spaced by edgartools' shared, thread-safe per-process rate limiter, which keeps one
process under SEC's 10 req/s cap; run one EDGAR walk (process) at a time. Workers save through
`context.store.save`, which serializes the CREATE of a cold table.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import pandas as pd
from tqdm import tqdm

DEFAULT_WORKERS = 8  # network-bound; the shared rate limiter caps throughput

#: Exceptions meaning the pipeline is broken, not the source record; per-entity and per-filing
#: handlers re-raise them instead of logging (`KeyError` = a broken frame column contract).
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
