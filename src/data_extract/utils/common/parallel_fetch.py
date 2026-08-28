"""
parallel_fetch.py (src/data_extract/utils/common/parallel_fetch.py)
---------------------------------------------------------------------
Thread-pool driver for the per-ticker EDGAR fetchers (8-K, 13D, DEF 14A,
filing text). Each ticker's `Company(ticker).get_filings(...)` walk plus
per-filing `.obj()` / `.text()` / attachment calls is pure network I/O bound
by SEC's request rate, not CPU -- and `edgartools` already serializes/spaces
*request starts* through a single shared, thread-safe rate limiter
(`httpxthrottlecache`, ~9 req/sec globally, see edgar.httpclient), letting
transfers overlap. A single-threaded sequential ticker walk never actually
saturates that limit -- it only ever has ONE request in flight, so it is
latency-bound, not rate-limit-bound. Running several tickers concurrently on
a bounded thread pool keeps every request under SEC's cap while actually
using it, which is what turns a ~10h from-scratch pull into ~1-2h.

Does NOT apply to `fetch_def14a_llm.py` -- that fetcher is bound by OpenAI's
rate limits/cost, a different domain, and is deliberately serialized
per-ticker today for crash-safety on expensive LLM calls.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, TypeVar

import pandas as pd
from tqdm import tqdm

R = TypeVar("R")

DEFAULT_WORKERS = 8   # network-bound; edgartools' own client caps ~9 req/sec globally


def run_per_ticker(cik_map: pd.DataFrame, worker: Callable[[str, str], R],
                   desc: str, max_workers: int = DEFAULT_WORKERS) -> list[R]:
    """Call `worker(ticker, cik)` for every row of `cik_map` on a bounded thread
    pool (I/O-bound EDGAR walk -- see module docstring), driving one shared
    tqdm bar. Returns results in COMPLETION order, not `cik_map`'s row order --
    every caller aggregates by summing counts / saving to the DB, so ticker
    order does not matter.

    `worker` must catch its own per-ticker exceptions (matching every
    fetcher's "one bad ticker can't abort the batch" convention): an uncaught
    exception here still aborts the whole pool once `.result()` re-raises it.

    `edgar_driver._worker` leaves ONE class uncaught on purpose --
    `edgar_driver.PROGRAMMING_ERRORS` -- and relies on exactly that abort: a defect in
    this repo will hit every remaining ticker too, so failing the run beats logging 490
    warnings and reporting success.
    """
    rows = list(cik_map[["ticker", "cik"]].itertuples(index=False, name=None))
    results: list[R] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(worker, ticker, cik) for ticker, cik in rows]
        for future in tqdm(as_completed(futures), total=len(futures), desc=desc):
            results.append(future.result())
    return results
