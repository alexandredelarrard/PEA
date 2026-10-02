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
from dataclasses import dataclass
from typing import Protocol

import pandas as pd

from src.context import Context
from src.data_extract.utils.common.identity import Identity, load_identity
from src.data_extract.utils.common.parallel_fetch import run_per_ticker
from src.data_extract.utils.common.registrant import (
    Registrant,
    identity_scope_fingerprint,
    load_registrants,
    resolve_registrant_filings,
    resolve_schedule_subject_filings,
)
from src.data_extract.utils.common.run_manifest import changed_scope_tickers, get_entry, manifest_window, record_run
from src.data_extract.utils.common.sec_utils import existing_filings, load_cik_mapping
from src.data_store.schema import Table
from src.utils.string import pad_cik

logger = logging.getLogger(__name__)


class IncompleteEdgarRunError(RuntimeError):
    """A completeness-sensitive EDGAR walk had one or more failed tickers."""


@dataclass(frozen=True)
class EdgarScope:
    """What every per-ticker EDGAR walk of one run resolves filings against: the identity layer
    (None when the fetch is not identity-aware) and the run's registrant register."""

    identity: Identity | None
    registrants: dict[str, Registrant]


def filed_by(filing: object, roster_cik: str) -> str:
    """The padded CIK that actually filed this document, falling back to the roster CIK only
    when the filing exposes none; a stored filer CIK makes a registrant boundary visible."""
    return pad_cik(getattr(filing, "cik", None) or roster_cik)


def period_of_report(filing):
    """The filing's `period_of_report`, or None when EDGAR's own metadata cannot yield it.

    ⚠ `getattr(filing, "period_of_report", None)` DOES NOT GUARD THIS. edgartools implements it
    as a `@property` that falls back to `filing.homepage.period_of_report` ->
    `attachments.get_filing_dates()`, and on some older submissions that returns None, so the
    unpack `_, _, period = ...` raises `TypeError` from INSIDE the property. `getattr`'s default
    only ever answers `AttributeError`, so the exception passes straight through it.

    That mattered the moment the register started walking predecessor archives. `TypeError` is
    in `PROGRAMMING_ERRORS`, which `run_per_ticker` re-raises on purpose -- our bug should fail the
    run, not be logged per ticker -- so ONE unparseable 2004 filing aborted a whole 16-ticker
    8-K walk after BKR (237 predecessor filings) and VTRS (292) had already resolved. The
    classification was right and the read was wrong: this is a property of the FILING, not of
    our code, so it belongs behind a guard rather than behind a widened exception policy.

    `period_of_report` is optional metadata on an 8-K and every consumer already tolerates NaT,
    so returning None is the honest answer and losing the whole walk was not.
    """
    try:
        return filing.period_of_report
    except Exception:  # noqa: BLE001 -- EDGAR metadata defect
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
    def __call__(
        self,
        ticker: str,
        cik: str,
        *,
        since: pd.Timestamp | None,
        done_accessions: frozenset[str],
        scope: EdgarScope,
    ) -> dict[Table, pd.DataFrame]: ...


def new_filings(
    ticker: str,
    forms: list[str],
    since: pd.Timestamp | None,
    done_accessions: frozenset[str],
    scope: EdgarScope,
) -> list:
    """`ticker`'s filings of `forms` resolved against `scope`, oldest first, stripped of stored
    accessions and of anything filed before `since` (see `registrant.resolve_registrant_filings`)."""
    return resolve_registrant_filings(
        ticker,
        forms,
        since=since,
        done_accessions=done_accessions,
        registrants=scope.registrants,
        identity=scope.identity,
    )


def new_schedule_filings(
    ticker: str,
    subject_ciks: frozenset[str],
    forms: list[str],
    since: pd.Timestamp | None,
    done_accessions: frozenset[str],
) -> list:
    """Issuer-side schedule discovery, including filings submitted under holder CIKs."""
    return resolve_schedule_subject_filings(
        ticker,
        subject_ciks,
        forms,
        since=since,
        done_accessions=done_accessions,
    )


def load_edgar_scope(
    context: Context,
    cik_map: pd.DataFrame,
    entry: dict | None,
    *,
    identity_aware: bool,
) -> tuple[EdgarScope, dict[str, str] | None, frozenset[str]]:
    """The run's `EdgarScope` from `context.config_dir`, plus (identity-aware only) each ticker's
    identity-scope fingerprint and the tickers whose fingerprint changed since manifest `entry`."""
    registrants = load_registrants(str(context.config_dir))
    if not identity_aware:
        return EdgarScope(None, registrants), None, frozenset()
    identity = load_identity(context)
    fingerprints: dict[str, str] = {}
    for ticker in cik_map["ticker"].astype(str):
        filing_scope = identity.filing_scope(ticker)
        fingerprints[ticker] = identity_scope_fingerprint(filing_scope, registrants.get(filing_scope.ticker))
    return EdgarScope(identity, registrants), fingerprints, changed_scope_tickers(entry, fingerprints)


def run_edgar_fetch(
    context: Context,
    tickers: list[str],
    years_history: int,
    *,
    tables: tuple[Table, ...],
    build: BuildFn,
    desc: str,
    max_workers: int | None = None,
    full: bool = False,
    cik_map: pd.DataFrame | None = None,
    minimum_since: pd.Timestamp | None = None,
    completion_table: Table | None = None,
    require_complete: bool = False,
    identity_aware: bool = False,
) -> None:
    """Fetch `tables` for `tickers` using `build(ticker, cik, since=, done_accessions=, scope=)
    -> {table: frame}`; `scope` comes from `load_edgar_scope`.

    `tables[0]` keys the manifest window and the accession dedup set; every declared table gets
    a `record_run` entry, even with zero rows. `cik_map` lets a caller that already loaded the
    universe pass it in. `completion_table` is saved last and only when every earlier frame
    saved. `require_complete` makes build and save failures fatal to the run manifest (saved
    rows stay as idempotent progress).
    """
    context.ensure_edgar_identity()
    if cik_map is None:
        cik_map = load_cik_mapping(context, tickers)
    entry = get_entry(context, tables[0])
    scope, scope_fingerprints, changed_scopes = load_edgar_scope(context, cik_map, entry, identity_aware=identity_aware)
    if changed_scopes:
        context.log.info(
            "%s: %d ticker identity scope(s) changed -> full-window relist: %s",
            desc,
            len(changed_scopes),
            ", ".join(sorted(changed_scopes)),
        )
    fallback_since = pd.Timestamp.today() - pd.DateOffset(years=years_history)
    if minimum_since is not None:
        fallback_since = max(fallback_since, pd.Timestamp(minimum_since).normalize())
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
    elif require_complete and not (entry or {}).get("coverage_complete"):
        # A legacy manifest only proves that the old discovery code finished. It cannot prove
        # issuer-side Schedule coverage because that code silently skipped large filer books.
        # The first run under the completeness contract must therefore walk the full configured
        # history before it is allowed to mint a trustworthy frontier.
        since, is_full_rescan = fallback_since, True
    else:
        since, is_full_rescan = manifest_window(
            context,
            tables[0],
            len(cik_map),
            fallback_since=fallback_since,
            full_rescan_days=int(context.config.data_extract.manifest_full_rescan_days),
            tickers=cik_map["ticker"],
        )
    done = existing_filings(context, tables[0])
    declared = set(tables)

    def _worker(ticker: str, cik: str) -> dict[Table, int] | None:
        frames = build(ticker, cik, since=fallback_since if ticker in changed_scopes else since, done_accessions=done, scope=scope)
        counts: dict[Table, int] = {}
        failed_save = False
        ordered_frames = [(table, df) for table, df in frames.items() if table != completion_table]
        if completion_table is not None and completion_table in frames:
            ordered_frames.append((completion_table, frames[completion_table]))
        for table, df in ordered_frames:
            if df is None or df.empty:
                continue
            if table == completion_table and failed_save:
                context.log.warning(
                    "%s: %s coverage not advanced because an earlier save failed",
                    desc,
                    ticker,
                )
                continue
            if table not in declared:
                context.log.warning("%s: %s built undeclared table '%s'", desc, ticker, table)
                failed_save = True
                continue
            # A failed save is per table: it marks `failed_save` and the other tables still save.
            try:
                context.store.save(table, df)
            except Exception as e:  # noqa: BLE001
                context.log.warning("%s: %s save to '%s' failed (%s)", desc, ticker, table, e)
                failed_save = True
                continue
            counts[table] = len(df)
        if require_complete and failed_save:
            return None
        return counts

    results = run_per_ticker(cik_map, _worker, desc=desc, log=context.log, max_workers=max_workers)
    failed = sum(1 for r in results if r is None)
    totals = {table: 0 for table in tables}
    for result in results:
        for table, n in (result or {}).items():
            totals[table] += n

    context.log.info(
        "%s: %d/%d ticker(s) ok, %d failed -> %s",
        desc,
        len(results) - failed,
        len(cik_map),
        failed,
        ", ".join(f"+{n} '{t}'" for t, n in totals.items()),
    )
    if require_complete and failed:
        raise IncompleteEdgarRunError(
            f"{desc}: {failed}/{len(cik_map)} ticker(s) failed; rows already saved remain "
            "idempotent, but no run manifest was advanced because coverage is incomplete"
        )
    for table in tables:
        record_run(
            context,
            table,
            len(cik_map),
            totals[table],
            is_full_rescan=is_full_rescan,
            coverage_complete=require_complete,
            identity_scope_fingerprints=scope_fingerprints,
            tickers=cik_map["ticker"],
        )
