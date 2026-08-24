"""
edgar_driver.py (src/data_extract/utils/common/edgar_driver.py)
-----------------------------------------------------------------
Shared driver for the per-ticker edgartools fetchers (8-K, 13D, DEF 14A, filing
text): resolve the listing window, dedup by accession, walk tickers on a thread
pool, upsert each ticker's frames and record the run. Each fetcher supplies only
its forms and its row builder.
"""

from __future__ import annotations

import os
import threading
from typing import Protocol

import pandas as pd
from edgar import Company, set_identity

from src.context import Context
from src.data_store.schema import Table
from src.data_extract.utils.common.parallel_fetch import run_per_ticker
from src.data_extract.utils.common.run_manifest import manifest_window, record_run
from src.data_extract.utils.common.sec_utils import existing_filings, load_cik_mapping


class BuildFn(Protocol):
    def __call__(self, ticker: str, cik: str, *, since: pd.Timestamp | None,
                 done_accessions: frozenset[str]) -> dict[Table, pd.DataFrame]: ...


def configure_identity() -> None:
    """SEC EDGAR blocks requests without a descriptive User-Agent -- fail loudly
    if unset, matching `sec_utils._sec_headers`."""
    ua = os.getenv("SEC_USER_AGENT", "").strip()
    if not ua:
        raise RuntimeError(
            "SEC_USER_AGENT is not set. SEC EDGAR blocks requests without a "
            "descriptive User-Agent (name + email). Add it to your .env file, e.g.\n"
            '  SEC_USER_AGENT="Your Name your.email@example.com"'
        )
    set_identity(ua)


def new_filings(ticker: str, forms: list[str], since: pd.Timestamp | None,
                done_accessions: frozenset[str]) -> list:
    """`ticker`'s filings of `forms`, oldest first, already stripped of accessions
    that are stored and of anything filed before `since`. Dedup and the date filter
    run BEFORE the sort so a routine run orders the handful of new filings rather
    than the ticker's full multi-year history."""
    kept = [f for f in Company(ticker).get_filings(form=list(forms))
            if f.accession_number not in done_accessions]
    dated = [(pd.Timestamp(f.filing_date), f) for f in kept]
    if since is not None:
        dated = [pair for pair in dated if pair[0] >= since]
    dated.sort(key=lambda pair: pair[0])
    return [f for _, f in dated]


def run_edgar_fetch(context: Context, tickers: list[str], years_history: int, *,
                    tables: tuple[Table, ...], build: BuildFn, desc: str,
                    max_workers: int | None = None) -> None:
    """Fetch `tables` for `tickers` using `build(ticker, cik, since, done_accessions)
    -> {table: frame}`.

    `max_workers` overrides the shared pool width for fetchers that need their own
    (fundamentals: its from-scratch backfill is the only one measured in hours).

    `tables[0]` is the primary: it keys the manifest window and the accession dedup
    set. Every declared table gets a `record_run` entry even when no ticker produced
    rows for it, so a table that is legitimately empty this run does not read as
    "never run" and force a full rescan forever.
    """
    configure_identity()
    cik_map = load_cik_mapping(context, tickers)
    since, is_full_rescan = manifest_window(
        context, tables[0], len(cik_map),
        fallback_since=pd.Timestamp.today() - pd.DateOffset(years=years_history),
        full_rescan_days=int(context.config.data_extract.manifest_full_rescan_days))
    done = existing_filings(context, tables[0])
    declared = set(tables)

    # `store.ensure_table` is a check-then-create with no locking, so on a cold table
    # several workers can each see it missing and race the CREATE; the losers raise and
    # would lose their ticker's rows. Serialize writes to a table until it is known to
    # exist -- afterwards `save` is a plain concurrent upsert.
    create_lock = threading.Lock()
    created: set[str] = set()

    def _save(table: Table, df: pd.DataFrame) -> None:
        if table.name in created:
            context.store.save(table, df)
            return
        with create_lock:
            context.store.save(table, df)
            created.add(table.name)

    def _worker(ticker: str, cik: str) -> dict[Table, int] | None:
        try:
            frames = build(ticker, cik, since=since, done_accessions=done)
        except Exception as e:                                   # noqa: BLE001
            context.log.warning("%s: %s failed (%s)", desc, ticker, e)
            return None
        counts: dict[Table, int] = {}
        for table, df in frames.items():
            if df is None or df.empty:
                continue
            if table not in declared:
                context.log.warning("%s: %s built undeclared table '%s'", desc, ticker, table)
                continue
            # Saving INSIDE the try: `run_per_ticker` re-raises whatever escapes a
            # worker, so an uncaught DB error here would abort the whole pool.
            try:
                _save(table, df)
            except Exception as e:                               # noqa: BLE001
                context.log.warning("%s: %s save to '%s' failed (%s)", desc, ticker, table, e)
                continue
            counts[table] = len(df)
        return counts

    results = run_per_ticker(cik_map, _worker, desc=desc,
                             **({} if max_workers is None else {"max_workers": max_workers}))
    failed = sum(1 for r in results if r is None)
    totals = {table: 0 for table in tables}
    for result in results:
        for table, n in (result or {}).items():
            totals[table] += n

    context.log.info("%s: %d/%d ticker(s) ok, %d failed -> %s", desc,
                     len(results) - failed, len(cik_map), failed,
                     ", ".join(f"+{n} '{t}'" for t, n in totals.items()))
    for table in tables:
        record_run(context, table, len(cik_map), totals[table], is_full_rescan=is_full_rescan)
