"""Shared driver for the per-ticker edgartools fetchers.

Resolves the listing window, dedups by accession, walks tickers on a thread pool, upserts each
ticker's frames and records the run. Each fetcher declares one `EdgarFetch`; single-table filing
fetchers build rows with `build_filing_rows` and supply only a per-filing row function.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from functools import cached_property, partial
from typing import Any, Protocol

import pandas as pd

from src.context import Context
from src.data_extract.utils.common.edgar_fillings import archive_url
from src.data_extract.utils.common.frame_sanitize import finalise_frame
from src.data_extract.utils.common.identity import FilingScope, Identity, load_identity
from src.data_extract.utils.common.incremental import stored_values
from src.data_extract.utils.common.parallel_fetch import run_per_ticker
from src.data_extract.utils.common.registrant import resolve_registrant_filings
from src.data_extract.utils.common.run_manifest import get_entry, manifest_window, record_run, scope_changed_tickers
from src.data_extract.utils.common.sec_utils import load_cik_mapping
from src.data_store.schema import Table
from src.utils.string import pad_cik

logger = logging.getLogger(__name__)


class IncompleteEdgarRunError(RuntimeError):
    """A completeness-sensitive EDGAR walk had one or more failed tickers."""


@dataclass
class GuardTally:
    """Thread-safe count of filings the scope guard skipped during one run."""

    skipped: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def add(self, n: int) -> None:
        if n:
            with self._lock:
                self.skipped += n


@dataclass(frozen=True)
class EdgarScope:
    """What every per-ticker EDGAR walk of one run lists filings against.

    `identity` gives each ticker's `FilingScope`; without it a ticker's scope is its roster CIK alone.
    `guard` counts the filings skipped as outside a scope.
    """

    identity: Identity | None = None
    guard: GuardTally = field(default_factory=GuardTally)

    def filing_scope(self, ticker: str, cik: str) -> FilingScope:
        """`ticker`'s filing scope from the identity layer, else its roster CIK alone."""
        return self.identity.filing_scope(ticker) if self.identity is not None else FilingScope.roster_only(ticker, cik)

    def list_filings(
        self,
        ticker: str,
        cik: str,
        forms: Sequence[str],
        *,
        since: pd.Timestamp | None,
        done_accessions: frozenset[str],
        stats: dict[str, int] | None = None,
    ) -> list:
        """`resolve_registrant_filings` over `ticker`'s scope; guard skips are added to `guard`."""
        counts: dict[str, int] = {} if stats is None else stats
        filings = resolve_registrant_filings(self.filing_scope(ticker, cik), forms, since=since, done_accessions=done_accessions, stats=counts)
        self.guard.add(counts.get("foreign_skipped", 0))
        return filings


@dataclass(frozen=True)
class FilingStamp:
    """The filing-level values every EDGAR fetcher stamps on its rows, read once per filing.

    `cik` is the padded CIK that actually filed the document, falling back to the roster CIK only
    when the filing exposes none. `filed` is the raw filing-date Timestamp (callers normalise when
    their table stores a date). `period_of_report` and `doc_url` are read lazily, at most once.
    """

    accession_number: str
    form: str
    cik: str
    filed: pd.Timestamp
    is_amendment: bool
    primary_document: str | None
    filing: Any = field(repr=False, compare=False)

    @classmethod
    def of(cls, filing: Any, roster_cik: str) -> FilingStamp:
        return cls(
            accession_number=filing.accession_number,
            form=filing.form,
            cik=pad_cik(getattr(filing, "cik", None) or roster_cik),
            filed=pd.Timestamp(filing.filing_date),
            is_amendment=str(filing.form).upper().endswith("/A"),
            primary_document=getattr(filing, "primary_document", None),
            filing=filing,
        )

    @cached_property
    def period_of_report(self) -> Any:
        """The filing's raw `period_of_report`, or None when EDGAR's metadata cannot yield it.

        Guarded because the edgartools property can raise `TypeError`, which `run_per_ticker`
        re-raises as a `PROGRAMMING_ERRORS` member; consumers tolerate a null.
        """
        try:
            return self.filing.period_of_report
        except Exception:  # noqa: BLE001 -- EDGAR metadata defect
            return None

    @cached_property
    def doc_url(self) -> str | None:
        """The primary document's URL: the attachment's own `url`, else the archives path, else None."""
        document = getattr(self.filing, "document", None)
        url = getattr(document, "url", None) if document is not None else None
        if not url and self.accession_number and self.primary_document and self.cik:
            url = archive_url(self.cik, str(self.accession_number), self.primary_document)
        return str(url) if url else None


def num_or_null(value, trust_value: bool) -> float:
    """A 13D/13G beneficial-ownership numeric as float, or NaN when `trust_value` is False or it is unparseable.

    `trust_value` is the caller's judgement that the value is real rather than a parser default.
    Returns NaN, never None, so an all-unknown batch stays float dtype rather than SQL TEXT."""
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


@dataclass(frozen=True)
class EdgarFetch:
    """One per-ticker EDGAR fetch, declared once and walked by `run_edgar_fetch`.

    `tables[0]` keys the manifest window and the accession dedup set; every table gets a
    `record_run` entry. `build(ticker, cik, since=, done_accessions=, scope=)` returns
    `{table: frame}`. A failed ticker is fatal to the run manifest; a ticker whose lineage scope
    changed since the last run is relisted over the full window; `minimum_since` floors the listing
    window; `completion_table` is saved last and only when every earlier frame saved.
    """

    desc: str
    tables: tuple[Table, ...]
    build: BuildFn
    minimum_since: pd.Timestamp | None = None
    completion_table: Table | None = None


def build_filing_rows(
    ticker: str,
    cik: str,
    *,
    since: pd.Timestamp | None = None,
    done_accessions: frozenset[str] = frozenset(),
    scope: EdgarScope,
    forms: Sequence[str],
    table: Table,
    columns: Sequence[str],
    row_fn: Callable[[str, FilingStamp], list[dict]],
    numeric: Sequence[str] = (),
) -> dict[Table, pd.DataFrame]:
    """`ticker`'s new filings of `forms` resolved against `scope` (oldest first, stored accessions
    and pre-`since` filings dropped), `row_fn(ticker, stamp)` rows per filing, one `table` frame
    finalised by `finalise_frame`."""
    filings = scope.list_filings(ticker, cik, forms, since=since, done_accessions=done_accessions)
    rows = [row for filing in filings for row in row_fn(ticker, FilingStamp.of(filing, cik))]
    return {table: finalise_frame(table, rows, columns=columns, numeric=numeric)}


def load_edgar_scope(context: Context) -> EdgarScope:
    """The run's `EdgarScope` over the identity layer of `context`."""
    return EdgarScope(load_identity(context))


def relist_tickers(scope: EdgarScope, cik_map: pd.DataFrame, entry: dict | None) -> frozenset[str]:
    """Tickers whose lineage `scope_changed_at` is at or after the table's last run (manifest `entry`)."""
    if scope.identity is None:
        return frozenset()
    stamps = {
        ticker: scope.filing_scope(ticker, cik).scope_changed_at for ticker, cik in zip(cik_map["ticker"].astype(str), cik_map["cik"], strict=True)
    }
    return scope_changed_tickers(entry, stamps)


@dataclass(frozen=True)
class RunWindow:
    """One run's listing window: `since` for unchanged tickers, `fallback_since` (the whole configured
    history) for a ticker whose lineage scope changed, and whether the run counts as a full rescan."""

    since: pd.Timestamp
    fallback_since: pd.Timestamp
    is_full_rescan: bool


def _resolve_window(
    context: Context,
    fetch: EdgarFetch,
    cik_map: pd.DataFrame,
    entry: dict | None,
    years_history: int,
    full: bool,
) -> RunWindow:
    """`fetch`'s window from manifest `entry`: the whole `years_history` window (floored at
    `fetch.minimum_since`) under `full` or a not-yet-complete manifest, else `manifest_window`."""
    fallback_since = pd.Timestamp.today() - pd.DateOffset(years=years_history)
    if fetch.minimum_since is not None:
        fallback_since = max(fallback_since, pd.Timestamp(fetch.minimum_since).normalize())
    # `-F/--full` serves chunked backfills, whose universe-size change the manifest cannot see.
    if full:
        return RunWindow(fallback_since, fallback_since, True)
    # A manifest without `coverage_complete` cannot prove coverage, so walk the full history.
    if not (entry or {}).get("coverage_complete"):
        return RunWindow(fallback_since, fallback_since, True)
    since, is_full_rescan = manifest_window(
        context,
        fetch.tables[0],
        cik_map["ticker"].astype(str).tolist(),
        fallback_since=fallback_since,
        full_rescan_days=int(context.config.data_extract.manifest_full_rescan_days),
    )
    return RunWindow(since, fallback_since, is_full_rescan)


def _build_ticker(
    fetch: EdgarFetch,
    scope: EdgarScope,
    window: RunWindow,
    changed: frozenset[str],
    done: frozenset[str],
    ticker: str,
    cik: str,
) -> dict[Table, pd.DataFrame]:
    """`fetch.build`'s frames for one ticker; a ticker in `changed` relists from `window.fallback_since`."""
    since = window.fallback_since if ticker in changed else window.since
    return fetch.build(ticker, cik, since=since, done_accessions=done, scope=scope)


def _save_frames(context: Context, fetch: EdgarFetch, ticker: str, frames: dict[Table, pd.DataFrame]) -> tuple[dict[Table, int], bool]:
    """Upsert one ticker's non-empty `frames`, `fetch.completion_table` last. Returns `(rows saved per
    table, failed_save)`: an undeclared table or a failed save sets `failed_save` without stopping
    the other tables, and then the completion table is not saved."""
    completion_table = fetch.completion_table
    ordered_frames = [(table, df) for table, df in frames.items() if table != completion_table]
    if completion_table is not None and completion_table in frames:
        ordered_frames.append((completion_table, frames[completion_table]))
    counts: dict[Table, int] = {}
    failed_save = False
    for table, df in ordered_frames:
        if df is None or df.empty:
            continue
        if table == completion_table and failed_save:
            context.log.warning("%s: %s coverage not advanced because an earlier save failed", fetch.desc, ticker)
            continue
        if table not in fetch.tables:
            context.log.warning("%s: %s built undeclared table '%s'", fetch.desc, ticker, table)
            failed_save = True
            continue
        try:
            context.store.save(table, df)
        except Exception as e:  # noqa: BLE001 -- a failed save is per table; the others still save
            context.log.warning("%s: %s save to '%s' failed (%s)", fetch.desc, ticker, table, e)
            failed_save = True
            continue
        counts[table] = len(df)
    return counts, failed_save


def _walk_ticker(
    context: Context,
    fetch: EdgarFetch,
    scope: EdgarScope,
    window: RunWindow,
    changed: frozenset[str],
    done: frozenset[str],
    ticker: str,
    cik: str,
) -> dict[Table, int] | None:
    """The pool worker: build then save one ticker. None (a failed ticker) when a save failed,
    else the rows saved per table."""
    frames = _build_ticker(fetch, scope, window, changed, done, ticker, cik)
    counts, failed_save = _save_frames(context, fetch, ticker, frames)
    return None if failed_save else counts


def _tally(results: list[dict[Table, int] | None], tables: tuple[Table, ...]) -> tuple[dict[Table, int], int]:
    """`(rows saved per table, failed ticker count)`; a None result is a failed ticker."""
    totals = {table: 0 for table in tables}
    for result in results:
        for table, n in (result or {}).items():
            totals[table] += n
    return totals, sum(1 for result in results if result is None)


def _record_tables(
    context: Context,
    fetch: EdgarFetch,
    cik_map: pd.DataFrame,
    totals: dict[Table, int],
    window: RunWindow,
) -> None:
    """One `record_run` entry per table of `fetch`, zero-row tables included."""
    for table in fetch.tables:
        record_run(
            context,
            table,
            len(cik_map),
            totals[table],
            is_full_rescan=window.is_full_rescan,
            coverage_complete=True,
            tickers=cik_map["ticker"],
        )


def run_edgar_fetch(
    context: Context,
    tickers: list[str],
    years_history: int,
    fetch: EdgarFetch,
    *,
    full: bool = False,
    cik_map: pd.DataFrame | None = None,
    max_workers: int | None = None,
) -> None:
    """Run `fetch` for `tickers` (or a preloaded `cik_map`): upsert every ticker's frames and record
    each of `fetch.tables`, even with zero rows. `full` takes the whole `years_history` window;
    a failed ticker raises before any manifest entry advances. The summary counts guard skips."""
    context.ensure_edgar_identity()
    if cik_map is None:
        cik_map = load_cik_mapping(context, tickers)
    entry = get_entry(context, fetch.tables[0])
    scope = load_edgar_scope(context)
    changed = relist_tickers(scope, cik_map, entry)
    if changed:
        context.log.info("%s: %d ticker lineage scope(s) changed -> full-window relist: %s", fetch.desc, len(changed), ", ".join(sorted(changed)))
    window = _resolve_window(context, fetch, cik_map, entry, years_history, full)
    done = stored_values(context, fetch.tables[0], "accession_number")
    worker = partial(_walk_ticker, context, fetch, scope, window, changed, done)
    results = run_per_ticker(cik_map, worker, desc=fetch.desc, log=context.log, max_workers=max_workers)
    totals, failed = _tally(results, fetch.tables)
    summary = ", ".join(f"+{n} '{t}'" for t, n in totals.items())
    context.log.info(
        "%s: %d/%d ticker(s) ok, %d failed -> %s; guard skipped %d filing(s) outside a filing scope for '%s'",
        fetch.desc,
        len(results) - failed,
        len(cik_map),
        failed,
        summary,
        scope.guard.skipped,
        fetch.tables[0],
    )
    if failed:
        raise IncompleteEdgarRunError(
            f"{fetch.desc}: {failed}/{len(cik_map)} ticker(s) failed; rows already saved remain "
            "idempotent, but no run manifest was advanced because coverage is incomplete"
        )
    _record_tables(context, fetch, cik_map, totals, window)
