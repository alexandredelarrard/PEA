"""
edgar_driver.py (src/data_extract/utils/common/edgar_driver.py)
-----------------------------------------------------------------
Shared driver for the per-ticker edgartools fetchers (8-K, 13D, DEF 14A, filing
text): resolve the listing window, dedup by accession, walk tickers on a thread
pool, upsert each ticker's frames and record the run. Each fetcher supplies only
its forms and its row builder.
"""

from __future__ import annotations

import threading
from typing import Protocol

import pandas as pd
from edgar import Company

from src.context import Context
from src.data_store.schema import Table
from src.data_extract.utils.common.parallel_fetch import run_per_ticker
from src.data_extract.utils.common.run_manifest import manifest_window, record_run
from src.data_extract.utils.common.sec_utils import existing_filings, load_cik_mapping


#: Exception classes that mean THIS pipeline is broken, not the filing. They are re-raised
#: wherever a per-ticker or per-filing handler would otherwise swallow them, because a
#: programming error and a malformed filing are indistinguishable once both are logged as a
#: warning -- and the walk that started 2026-08-27 00:06 proved the cost: a one-word
#: `NameError` in `xbrl_linkbase.statement_arcs` cost NEM, MO and AIZ every fact they had
#: while the run reported success for 10 hours.
#:
#: `KeyError` is on the list deliberately: on these paths it means a frame's column contract
#: broke, which is ours. The narrow `except` around a LIBRARY parse (`filing.xbrl()`) keeps
#: swallowing everything, since malformed XBRL is exactly what it exists to absorb.
PROGRAMMING_ERRORS = (NameError, AttributeError, TypeError, KeyError, ImportError)


class BuildFn(Protocol):
    def __call__(self, ticker: str, cik: str, *, since: pd.Timestamp | None,
                 done_accessions: frozenset[str]) -> dict[Table, pd.DataFrame]: ...


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
                    max_workers: int | None = None, full: bool = False,
                    cik_map: pd.DataFrame | None = None) -> None:
    """Fetch `tables` for `tickers` using `build(ticker, cik, since, done_accessions)
    -> {table: frame}`.

    `max_workers` overrides the shared pool width for fetchers that need their own
    (fundamentals: its from-scratch backfill is the only one measured in hours).

    `cik_map` is `load_cik_mapping`'s frame, accepted from a caller that already needed it
    -- the fundamentals fetch reads the same three GICS levels off it to route regimes --
    so the universe is read ONCE per run instead of once here and once there.

    `tables[0]` is the primary: it keys the manifest window and the accession dedup
    set. Every declared table gets a `record_run` entry even when no ticker produced
    rows for it, so a table that is legitimately empty this run does not read as
    "never run" and force a full rescan forever.
    """
    context.ensure_edgar_identity()
    if cik_map is None:
        cik_map = load_cik_mapping(context, tickers)
    fallback_since = pd.Timestamp.today() - pd.DateOffset(years=years_history)
    if full:
        # `-F/--full`: take the whole years-history window and do not consult the manifest.
        #
        # Needed for a CHUNKED from-scratch backfill, which the manifest cannot express. Its
        # incremental test is "did the ticker universe change size since the last run?", so
        # running `-t A,B,C,D,E,F` twice in a row -- two different chunks, six tickers each --
        # looks like a repeat of the same run and the second chunk gets `since = last run`,
        # i.e. nothing. Measured the hard way: chunk 1 wrote 31,540 rows and chunks 2-9 wrote
        # 0. Chunking is not optional here (edgartools never releases its per-filing caches,
        # and an all-52 single process reached 14.7 GB RSS), so the flag is the fix.
        since, is_full_rescan = fallback_since, True
    else:
        since, is_full_rescan = manifest_window(
            context, tables[0], len(cik_map), fallback_since=fallback_since,
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
        except PROGRAMMING_ERRORS:
            # Our bug, not this ticker's data: let it escape the pool and fail the run.
            # `run_per_ticker` re-raises whatever escapes a worker, which is the point --
            # every remaining ticker would hit the same defect, and each already-saved
            # ticker's rows are upserted and keep.
            raise
        except Exception as e:                                   # noqa: BLE001 -- one ticker
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
