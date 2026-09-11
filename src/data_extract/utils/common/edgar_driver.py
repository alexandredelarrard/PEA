"""
edgar_driver.py (src/data_extract/utils/common/edgar_driver.py)
-----------------------------------------------------------------
Shared driver for the per-ticker edgartools fetchers (8-K, 13D, DEF 14A, filing
text): resolve the listing window, dedup by accession, walk tickers on a thread
pool, upsert each ticker's frames and record the run. Each fetcher supplies only
its forms and its row builder.
"""

from __future__ import annotations

import logging
import threading
from typing import Protocol

import pandas as pd

from src.context import Context
from src.data_store.schema import Table
from src.data_extract.utils.common.parallel_fetch import run_per_ticker
from src.data_extract.utils.common.registrant import resolve_registrant_filings
from src.data_extract.utils.common.run_manifest import manifest_window, record_run
from src.data_extract.utils.common.sec_utils import existing_filings, load_cik_mapping

logger = logging.getLogger(__name__)


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


def filed_by(filing, roster_cik: str) -> str:
    """The CIK that ACTUALLY FILED this document, falling back to the roster's only when the
    filing exposes none.

    ⚠ THE `cik` COLUMN WAS A STAMP, NOT AN OBSERVATION, AND THAT CONCEALED THIS WHOLE DEFECT
    CLASS. Every tier-A fetcher resolved by TICKER and then wrote the roster's CIK onto the
    row, so `sec_8k` held 0 of 491 tickers with more than one distinct CIK and 0 rows where
    `sec_8k.cik <> sp500_tickers.cik` -- measured 2026-09-09. XOM's 526 8-K rows going back to
    1996 all carried CIK 2115436, an entity whose entire archive is 29 filings beginning
    2026-07-01. The one column that would have made a registrant boundary self-announcing
    instead reported the roster back to itself.

    ⚠ THIS FIXES OBSERVABILITY, NOT RESOLUTION. You can only read a CIK off filings you
    already have, and a wrong roster CIK yields none to read -- that is what the register is
    for. What it buys is that the NEXT reorganisation shows up immediately as two CIKs either
    side of a date, instead of hiding behind a uniformly stamped column for a year.
    """
    return str(getattr(filing, "cik", None) or roster_cik).zfill(10)


def period_of_report(filing):
    """The filing's `period_of_report`, or None when EDGAR's own metadata cannot yield it.

    ⚠ `getattr(filing, "period_of_report", None)` DOES NOT GUARD THIS. edgartools implements it
    as a `@property` that falls back to `filing.homepage.period_of_report` ->
    `attachments.get_filing_dates()`, and on some older submissions that returns None, so the
    unpack `_, _, period = ...` raises `TypeError` from INSIDE the property. `getattr`'s default
    only ever answers `AttributeError`, so the exception passes straight through it.

    That mattered the moment the register started walking predecessor archives. `TypeError` is
    in `PROGRAMMING_ERRORS`, which `_worker` re-raises on purpose -- our bug should fail the
    run, not be logged per ticker -- so ONE unparseable 2004 filing aborted a whole 16-ticker
    8-K walk after BKR (237 predecessor filings) and VTRS (292) had already resolved. The
    classification was right and the read was wrong: this is a property of the FILING, not of
    our code, so it belongs behind a guard rather than behind a widened exception policy.

    `period_of_report` is optional metadata on an 8-K and every consumer already tolerates NaT,
    so returning None is the honest answer and losing the whole walk was not.
    """
    try:
        return filing.period_of_report
    except Exception:                                   # noqa: BLE001 -- EDGAR metadata defect
        return None

def num_or_null(value, trust_value: bool) -> float:
    """A beneficial-ownership numeric (13D or 13G) is only meaningful once the caller has
    established the value is real rather than a class default -- usually 0, which a schedule
    parser emits for every field it could not find. `trust_value` is the caller's AND of
    every reason to disbelieve it: the filing carried no structured data at all, or it did
    but this reporting person deferred its numbers to a narrative item.

    Returns NaN (never None/Python-null) so the column stays float dtype even when every row
    in a batch is unknown -- an all-None object column gets inferred as SQL TEXT by
    `ensure_table`'s dtype mapping, which would corrupt a genuinely numeric field the first
    time a real value needs to share that column.

    Lives here rather than in either fetcher because both schedules need exactly this rule and
    two copies would drift: 13D nulls on `has_structured_data` AND its placeholder test, 13G on
    `has_structured_data` alone, and the difference must be visible at the CALL site."""
    if not trust_value or value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


class BuildFn(Protocol):
    def __call__(self, ticker: str, cik: str, *, since: pd.Timestamp | None,
                 done_accessions: frozenset[str]) -> dict[Table, pd.DataFrame]: ...


def new_filings(ticker: str, forms: list[str], since: pd.Timestamp | None,
                done_accessions: frozenset[str]) -> list:
    """`ticker`'s filings of `forms`, oldest first, stripped of stored accessions and of
    anything filed before `since`.

    A thin wrapper over `registrant.resolve_registrant_filings`, kept because five call sites
    use this name. THE REASONING MOVED WITH THE IMPLEMENTATION -- why a union here and a dated
    split for consolidating forms, and the measurements behind both -- and now lives on
    `registrant.FORM_POLICY`, where every pipeline can see it rather than only these five.

    The one behavioural change: a form with no declared policy now RAISES instead of being
    unioned by default. That is deliberate. `Company(ticker)` resolving exactly one registrant,
    silently, is how this defect class stayed invisible for a year, and a new form family
    quietly inheriting the wrong rule would be the same failure wearing different clothes.
    """
    return resolve_registrant_filings(ticker, forms, since=since,
                                      done_accessions=done_accessions)


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
