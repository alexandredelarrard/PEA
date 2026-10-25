"""Shared driver for the per-filing EDGAR document fetchers.

`run_edgar_fetch` refreshes the local EDGAR index, plans each key's work list with
`resume.document_worklist` (index rows not yet stored), reads every listed filing on a pool of keys
and saves what it read. A filing that holds no data for the key (not its subject, nothing parsed,
or a deterministic parse failure) is saved as one empty-filing marker in the done table. A
transient SEC failure saves nothing for that filing; a key reading below the configured rate gets
in-task retry rounds, and whatever is still missing is listed again on the next run. Each key's
tables are saved after its pass, the done table last. Only a programming error fails the run.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from functools import cached_property, partial
from operator import attrgetter
from typing import Any, Protocol, cast

import pandas as pd

from src.context import Context
from src.data_extract.utils.common import edgar_index
from src.data_extract.utils.common.edgar_fillings import archive_url
from src.data_extract.utils.common.empty_markers import drop_markers_over_data, marker_frame
from src.data_extract.utils.common.frame_sanitize import finalise_frame
from src.data_extract.utils.common.identity import FilingScope, Identity, load_identity
from src.data_extract.utils.common.parallel_fetch import PROGRAMMING_ERRORS, run_per_ticker
from src.data_extract.utils.common.registrant import resolve_registrant_filings
from src.data_extract.utils.common.resume import DONE_TABLE, DocumentWork, document_worklist
from src.data_extract.utils.common.sec_io import ParseFailureError, TransientReadError, configure, sec_call
from src.data_extract.utils.common.sec_utils import load_cik_mapping
from src.data_extract.utils.institutionals.insider_store import bind_insider_identity, certify_insider_source, save_insider_source
from src.data_store.schema import Resume, Table, Tables
from src.utils.string import pad_cik, run_date

logger = logging.getLogger(__name__)

#: Accessions named per key in the coverage log.
_MISSING_SHOWN = 20
_UNIT_COLUMNS = ["cik", "company", "form", "filed", "accession"]


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

    `identity` gives each ticker's `FilingScope` (event CIKs and seam-widened CIK windows); without it
    a ticker's scope is its roster CIK alone. `guard` counts the `Company` filings skipped as outside a scope.
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
        """`resolve_registrant_filings` over `ticker`'s scope (live `Company` listings); guard skips are added to `guard`."""
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
        re-raises as a `PROGRAMMING_ERRORS` member; consumers tolerate a null. Read under the
        `sec_io` retry policy; a transient SEC failure raises rather than storing a null.
        """
        try:
            return sec_call(attrgetter("period_of_report"), self.filing, label=f"{self.accession_number} period_of_report")
        except TransientReadError:
            raise
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


class ParseFn(Protocol):
    def __call__(self, ticker: str, cik: str, stamp: FilingStamp, scope: EdgarScope) -> dict[Table, pd.DataFrame]: ...


class SubjectFn(Protocol):
    def __call__(self, ticker: str, cik: str, stamp: FilingStamp, scope: EdgarScope) -> bool: ...


class FilingsFn(Protocol):
    def __call__(self, ticker: str, df_units: pd.DataFrame) -> dict[str, Any]: ...


@dataclass(frozen=True)
class EdgarFetch:
    """One per-filing EDGAR fetch, declared once and walked by `run_edgar_fetch`.

    `parse(ticker, cik, stamp, scope)` returns `{table: frame}` for one filing; `is_subject` (same
    arguments) rejects a filing on which the key is only a filer or owner. `done_table` (default
    `tables[0]`) holds the stored-accession set and is saved last; `done_scope` reads that set per
    key or table-wide, on the rows matching `done_where` (all rows when None). `filings(ticker,
    df_units)` may supply richer `Filing` objects by accession. `runtime_floor(context, run_date)`
    floors the listing at run time. `run_scope` optionally prepares a run-local descriptor and
    finalizes it on exit; a planning declaration may omit `parse` when that context supplies it.
    The done table owns the description, forms and marker stamps.
    """

    tables: tuple[Table, ...]
    parse: ParseFn | None = None
    done_table: Table | None = None
    is_subject: SubjectFn | None = None
    identity_aware: bool = True
    done_scope: str = DONE_TABLE
    filings: FilingsFn | None = None
    done_where: dict[str, object] | None = None
    runtime_floor: Callable[[Context, pd.Timestamp], pd.Timestamp | None] | None = None
    run_scope: Callable[[Context, Sequence[str], pd.Timestamp], AbstractContextManager[EdgarFetch]] | None = None

    @property
    def done(self) -> Table:
        return self.done_table if self.done_table is not None else self.tables[0]


@dataclass(frozen=True)
class RetryRounds:
    """In-task retry rounds for keys reading below `threshold` with at least `min_failed` failed
    documents; `waits[i]` seconds before round i+1. Defaults mirror `data_extract.retry_rounds`."""

    rounds: int = 3
    waits: tuple[float, ...] = (60.0, 120.0, 240.0)
    threshold: float = 0.985
    min_failed: int = 2

    @classmethod
    def from_config(cls, config: Any) -> RetryRounds:
        section = getattr(getattr(config, "data_extract", None), "retry_rounds", None)
        if section is None:
            return cls()
        return cls(int(section.rounds), tuple(float(w) for w in section.waits), float(section.threshold), int(section.min_failed))

    def due(self, listed: int, failed: int) -> bool:
        return listed > 0 and failed >= self.min_failed and (listed - failed) / listed < self.threshold


@dataclass
class KeyOutcome:
    """One key's pass: documents listed, those still failed (index rows), markers and rows saved."""

    listed: int
    failed: pd.DataFrame
    markers: int = 0
    saved: dict[Table, int] = field(default_factory=dict)

    @property
    def read(self) -> int:
        return self.listed - len(self.failed)


@dataclass
class FetchSummary:
    """What one `run_edgar_fetch` call did, per key, after its retry rounds."""

    work: DocumentWork
    outcomes: dict[str, KeyOutcome]

    @property
    def missing(self) -> dict[str, list[str]]:
        return {key: [str(a) for a in o.failed["accession"]] for key, o in self.outcomes.items() if not o.failed.empty}


#: The sleeper between retry rounds; tests replace it.
_sleep: Callable[[float], None] = time.sleep


def parse_filing_rows(
    ticker: str,
    cik: str,
    stamp: FilingStamp,
    scope: EdgarScope,
    *,
    table: Table,
    row_fn: Callable[[str, FilingStamp], list[dict]],
    numeric: tuple[str, ...] = (),
) -> dict[Table, pd.DataFrame]:
    """A single-table fetch's `parse`: `row_fn(ticker, stamp)` rows as one `table` frame finalised by `finalise_frame`."""
    return {table: finalise_frame(table, row_fn(ticker, stamp), numeric=numeric)}


def marker_row(table: Table, ticker: str, stamp: FilingStamp) -> pd.DataFrame:
    """One empty-filing marker for `table`: key, accession, filing date, CIK, form, the table's
    constant marker columns and the declared sentinel; every other column is left to its typed NULL."""
    if table.resume is None or table.resume.frontier_col is None:
        raise ValueError(f"{table.name}: a marker needs a resume contract with a frontier column and an empty_marker")
    row: dict[str, object] = {
        table.resume.key or "ticker": ticker,
        "accession_number": stamp.accession_number,
        table.resume.frontier_col: stamp.filed.normalize(),
        table.marker_form_col: str(stamp.form),
    }
    if table.marker_cik_col is not None:
        row[table.marker_cik_col] = stamp.cik
    row |= dict(table.marker_extra)
    if table == Tables.insider_transactions:
        row["fetched_at"] = pd.Timestamp.now(tz="UTC").tz_localize(None)
    return marker_frame(table, row, dict(table.marker_key_fill))


def _has_key_rows(df: pd.DataFrame | None, table: Table, ticker: str) -> bool:
    key_col = table.resume.key if table.resume is not None and table.resume.key else "ticker"
    return df is not None and not df.empty and key_col in df.columns and bool(df[key_col].astype(str).eq(ticker).any())


def _read_unit(fetch: EdgarFetch, scope: EdgarScope, ticker: str, cik: str, stamp: FilingStamp) -> tuple[dict[Table, pd.DataFrame] | None, bool]:
    """`(frames, is_marker)` for one filing; frames are None when it failed and must be listed again."""
    assert fetch.parse is not None
    try:
        if fetch.is_subject is not None and not fetch.is_subject(ticker, cik, stamp, scope):
            return {fetch.done: marker_row(fetch.done, ticker, stamp)}, True
        frames = fetch.parse(ticker, cik, stamp, scope)
    except TransientReadError as exc:
        logger.info("%s: %s %s not read (%s)", fetch.done.desc, ticker, stamp.accession_number, exc)
        return None, False
    except ParseFailureError as exc:
        if fetch.done == Tables.insider_transactions:
            logger.warning("%s: %s %s could not be parsed; listed again (%s)", fetch.done.desc, ticker, stamp.accession_number, exc)
            return None, False
        logger.warning("%s: %s %s could not be parsed -> empty-filing marker (%s)", fetch.done.desc, ticker, stamp.accession_number, exc)
        return {fetch.done: marker_row(fetch.done, ticker, stamp)}, True
    except PROGRAMMING_ERRORS:
        raise
    except Exception as exc:  # noqa: BLE001 -- an unclassified source failure is retried, never marked
        logger.warning("%s: %s %s failed (%s: %s); listed again next run", fetch.done.desc, ticker, stamp.accession_number, type(exc).__name__, exc)
        return None, False
    if _has_key_rows(frames.get(fetch.done), fetch.done, ticker):
        return frames, False
    done_frames = [df for df in (frames.get(fetch.done),) if df is not None and not df.empty]
    return {**frames, fetch.done: pd.concat([*done_frames, marker_row(fetch.done, ticker, stamp)], ignore_index=True)}, True


def _unit_filings(fetch: EdgarFetch, ticker: str, df_units: pd.DataFrame) -> dict[str, Any] | None:
    """The fetch's own `Filing` objects by accession, `{}` when it supplies none, None when its listing read failed."""
    if fetch.filings is None or df_units.empty:
        return {}
    try:
        return fetch.filings(ticker, df_units)
    except TransientReadError as exc:
        logger.warning("%s: %s filing metadata not read (%s); its %d document(s) are listed again", fetch.done.desc, ticker, exc, len(df_units))
        return None


def _read_key(
    fetch: EdgarFetch, scope: EdgarScope, ticker: str, cik: str, df_units: pd.DataFrame
) -> tuple[dict[Table, list[pd.DataFrame]], pd.DataFrame, int]:
    """Read every listed filing of one key: `(frames per table, failed index rows, markers)`."""
    filings = _unit_filings(fetch, ticker, df_units)
    if filings is None:
        return {}, df_units, 0
    frames: dict[Table, list[pd.DataFrame]] = {}
    failed: list[int] = []
    markers = 0
    for position, unit in enumerate(df_units.itertuples(index=False)):
        filing = filings.get(str(unit.accession)) or edgar_index.index_filing(unit.cik, unit.company, unit.form, unit.filed, unit.accession)
        outcome, is_marker = _read_unit(fetch, scope, ticker, cik, FilingStamp.of(filing, cik))
        if outcome is None:
            failed.append(position)
            continue
        markers += int(is_marker)
        _add_frames(frames, outcome)
    return frames, df_units.iloc[failed].reset_index(drop=True), markers


def _add_frames(frames: dict[Table, list[pd.DataFrame]], outcome: dict[Table, pd.DataFrame]) -> None:
    """Append each non-empty frame of one filing's `outcome` to its table's list in `frames`."""
    for table, df in outcome.items():
        if df is not None and not df.empty:
            frames.setdefault(table, []).append(df)


def _save_key(context: Context, fetch: EdgarFetch, ticker: str, frames: dict[Table, list[pd.DataFrame]]) -> tuple[dict[Table, int], bool]:
    """Upsert one key's frames, the done table last. Returns `(rows per table, failed_save)`; after a
    failed or undeclared save the done table is withheld, so the key's documents stay listed."""
    if fetch.done == Tables.insider_transactions:
        raise ValueError("The insider source writer requires its actual parser resolver")
    order = [t for t in frames if t != fetch.done] + ([fetch.done] if fetch.done in frames else [])
    counts: dict[Table, int] = {}
    failed_save = False
    for table in order:
        if table == fetch.done and failed_save:
            context.log.warning("%s: %s done table not saved because an earlier save failed", fetch.done.desc, ticker)
            continue
        if table not in fetch.tables:
            context.log.warning("%s: %s built undeclared table '%s'", fetch.done.desc, ticker, table)
            failed_save = True
            continue
        df = drop_markers_over_data(context.store, table, pd.concat(frames[table], ignore_index=True), log=context.log)
        df = df.drop_duplicates(subset=[c for c in table.pk if c in df.columns], keep="last")
        if df.empty:
            counts[table] = 0
            continue
        try:
            context.store.save(table, df)
        except Exception as exc:  # noqa: BLE001 -- a failed save is per table; the others still save
            context.log.warning("%s: %s save to '%s' failed (%s)", fetch.done.desc, ticker, table, exc)
            failed_save = True
            continue
        counts[table] = len(df)
    return counts, failed_save


def _save_insider_key(context: Context, frames: dict[Table, list[pd.DataFrame]], identity: Identity | None) -> tuple[dict[Table, int], bool]:
    transaction_frames = frames.get(Tables.insider_transactions, [])
    if not transaction_frames:
        return {}, False
    if identity is None:
        context.log.warning("Insider source pair not saved without its actual parser resolver")
        return {}, True
    transactions = pd.concat(transaction_frames, ignore_index=True).drop_duplicates(list(Tables.insider_transactions.pk), keep="last")
    note_frames = frames.get(Tables.insider_footnotes, [])
    notes = pd.concat(note_frames, ignore_index=True) if note_frames else pd.DataFrame(columns=list(Tables.insider_footnotes.retrieval_columns))
    try:
        return save_insider_source(context, transactions, notes, identity.source_cohort), False
    except Exception as exc:
        context.log.warning("Insider source pair not saved; all accessions remain retryable (%s)", exc)
        return {}, True


def _walk_key(context: Context, fetch: EdgarFetch, scope: EdgarScope, units: dict[str, pd.DataFrame], ticker: str, cik: str) -> KeyOutcome:
    """The pool worker: read then save one key's listed documents."""
    df_units = units.get(ticker, pd.DataFrame(columns=_UNIT_COLUMNS))
    frames, failed, markers = _read_key(fetch, scope, ticker, cik, df_units)
    saved, failed_save = (
        _save_insider_key(context, frames, scope.identity) if fetch.done == Tables.insider_transactions else _save_key(context, fetch, ticker, frames)
    )
    if failed_save:
        return KeyOutcome(listed=len(df_units), failed=df_units, saved=saved)
    return KeyOutcome(listed=len(df_units), failed=failed, markers=markers, saved=saved)


def _run_pass(
    context: Context,
    fetch: EdgarFetch,
    scope: EdgarScope,
    cik_map: pd.DataFrame,
    units: dict[str, pd.DataFrame],
    max_workers: int | None,
) -> dict[str, KeyOutcome]:
    """One pool pass over `cik_map`'s keys; a key whose worker failed outright keeps all its documents failed."""
    worker = partial(_walk_key, context, fetch, scope, units)
    results = run_per_ticker(cik_map, worker, desc=fetch.done.desc, log=context.log, max_workers=max_workers)
    outcomes: dict[str, KeyOutcome] = {}
    for ticker, result in zip(cik_map["ticker"].astype(str), results, strict=True):
        df_units = units.get(ticker, pd.DataFrame(columns=_UNIT_COLUMNS))
        outcomes[ticker] = result if result is not None else KeyOutcome(listed=len(df_units), failed=df_units)
    return outcomes


def _retry_rounds(
    context: Context,
    fetch: EdgarFetch,
    scope: EdgarScope,
    cik_map: pd.DataFrame,
    outcomes: dict[str, KeyOutcome],
    max_workers: int | None,
) -> None:
    """Re-read only the failed documents of keys below the read-rate threshold, up to the configured rounds, in place."""
    policy = RetryRounds.from_config(getattr(context, "config", None))
    for round_no in range(1, policy.rounds + 1):
        due = [t for t, o in outcomes.items() if policy.due(o.listed, len(o.failed))]
        if not due:
            return
        wait = policy.waits[min(round_no - 1, len(policy.waits) - 1)] if policy.waits else 0.0
        retry = {t: outcomes[t].failed for t in due}
        n_documents = sum(len(df) for df in retry.values())
        context.log.info(
            "%s: retry round %d/%d for %d key(s), %d document(s), after %.0fs", fetch.done.desc, round_no, policy.rounds, len(due), n_documents, wait
        )
        _sleep(wait)
        again = _run_pass(context, fetch, scope, cik_map[cik_map["ticker"].astype(str).isin(due)], retry, max_workers)
        for ticker, outcome in again.items():
            previous = outcomes[ticker]
            saved = {t: previous.saved.get(t, 0) + outcome.saved.get(t, 0) for t in {*previous.saved, *outcome.saved}}
            outcomes[ticker] = KeyOutcome(listed=previous.listed, failed=outcome.failed, markers=previous.markers + outcome.markers, saved=saved)


def _log_coverage(context: Context, fetch: EdgarFetch, work: DocumentWork, outcomes: dict[str, KeyOutcome]) -> None:
    """One line per key still missing documents, then the run summary."""
    for ticker, outcome in sorted(outcomes.items()):
        if outcome.failed.empty:
            continue
        missing = [str(a) for a in outcome.failed["accession"]]
        more = f" (+{len(missing) - _MISSING_SHOWN} more)" if len(missing) > _MISSING_SHOWN else ""
        context.log.warning(
            "%s: %s read %d/%d, missing: %s%s", fetch.done.desc, ticker, outcome.read, outcome.listed, ", ".join(missing[:_MISSING_SHOWN]), more
        )
    listed = sum(o.listed for o in outcomes.values())
    read = sum(o.read for o in outcomes.values())
    totals: dict[str, int] = {}
    for outcome in outcomes.values():
        for table, n in outcome.saved.items():
            totals[str(table)] = totals.get(str(table), 0) + n
    context.log.info(
        "%s: %d/%d document(s) read for %d key(s) (%d empty-filing marker(s)); %d still missing, listed again next run; rows saved %s; work %s",
        fetch.done.desc,
        read,
        listed,
        len(outcomes),
        sum(o.markers for o in outcomes.values()),
        listed - read,
        totals,
        work.counts,
    )


def load_edgar_scope(context: Context, *, identity_aware: bool = True) -> EdgarScope:
    """The run's `EdgarScope` over the identity layer of `context` (roster CIKs only when not identity-aware)."""
    return EdgarScope(load_identity(context) if identity_aware else None)


def _document_cap(context: Context, no_cap: bool) -> int | None:
    value = getattr(getattr(getattr(context, "config", None), "data_extract", None), "max_documents_per_run", None)
    return None if no_cap or value is None else int(value)


def plan_fetch(
    context: Context,
    fetch: EdgarFetch,
    cik_map: pd.DataFrame,
    scope: EdgarScope,
    run_date: pd.Timestamp,
    years_history: int,
    *,
    full: bool = False,
    no_cap: bool = False,
) -> DocumentWork:
    """`fetch`'s work list for `cik_map`'s keys on `run_date` (read-only; the index cache as it stands)."""
    return document_worklist(
        context,
        fetch.done,
        cik_map[["ticker", "cik"]],
        run_date,
        forms=cast(Resume, fetch.done.resume).forms,
        identity=scope.identity,
        years_history=years_history,
        runtime_floor=fetch.runtime_floor(context, run_date) if fetch.runtime_floor is not None else None,
        done_scope=fetch.done_scope,
        done_where=fetch.done_where,
        full=full,
        cap=_document_cap(context, no_cap),
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
    as_of: pd.Timestamp | None = None,
    no_cap: bool = False,
    refresh_index: bool = True,
) -> FetchSummary:
    """Fetch every document `fetch.done` still lacks for `tickers` and save what was read.

    `full` re-reads the whole listed history (stored accessions and markers included); `as_of` is
    the run date (default today); `no_cap` lifts `max_documents_per_run`. Never raises for a
    source failure: what could not be read is logged and listed again on the next run.
    """
    start_date = run_date(as_of)
    with fetch.run_scope(context, tickers, start_date) if fetch.run_scope is not None else nullcontext(fetch) as prepared:
        if prepared.parse is None:
            raise ValueError(f"{prepared.done.desc}: run scope did not supply a parser")
        context.ensure_edgar_identity()
        configure(context)
        if cik_map is None:
            cik_map = load_cik_mapping(context, tickers)
        scope = load_edgar_scope(context, identity_aware=prepared.identity_aware)
        if prepared.done == Tables.insider_transactions and scope.identity is not None:
            bind_insider_identity(context, scope.identity.source_cohort)
        if refresh_index:
            index_years = (
                max(years_history, start_date.year - pd.Timestamp(cast(str, cast(Resume, prepared.done.resume).source_start)).year + 1)
                if prepared.done == Tables.insider_transactions
                else years_history
            )
            edgar_index.refresh(context, start_date, index_years)
        work = plan_fetch(context, prepared, cik_map, scope, start_date, years_history, full=full, no_cap=no_cap)
        context.log.info("%s: %d document(s) to read for %d key(s) %s", prepared.done.desc, work.size, len(work.units), work.counts)
        keys = cik_map[cik_map["ticker"].astype(str).isin(work.units)]
        outcomes = _run_pass(context, prepared, scope, keys, work.units, max_workers)
        _retry_rounds(context, prepared, scope, keys, outcomes, max_workers)
        _log_coverage(context, prepared, work, outcomes)
        if prepared.done == Tables.insider_transactions:
            _certify_insider_work(context, work, start_date)
        return FetchSummary(work=work, outcomes=outcomes)


def _certify_insider_work(context: Context, work: DocumentWork, as_of: pd.Timestamp) -> None:
    """Only a full validated source index and the exact reconciled source cohort establish quiet support."""
    through = edgar_index.inventory_complete_through(context, as_of, cast(str, cast(Resume, Tables.insider_transactions.resume).source_start))
    if through is None:
        context.log.warning("insider inventory: incomplete or changed original index; no boundary advanced")
        return
    certified = 0
    for ticker, listed in work.listed.items():
        accessions = sorted(set(listed["accession"].astype(str)))
        certified += int(certify_insider_source(context, ticker, through, accessions))
    context.log.info("insider inventory: %d/%d actual issuer cohorts jointly certified through %s", certified, len(work.listed), through.date())
