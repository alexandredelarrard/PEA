"""
fetch_13f_managers.py (src/data_extract/utils/institutionals/fetch_13f_managers.py)
------------------------------------------------------------------------------------
Per-CIK catch-up of `sec13f_manager_holdings` (complete CUSIP books, no universe filter) for every CIK
ever on `superinvestor_roster`, with `fetch_13f`'s parser and save. A listed filing is done when the
stored book shows it, or a later filing of the same period, by `(period, filing_date)`; every other
filing is read, so a failed read or an inline-written newer filing never hides older gaps. A transient
read failure holds its quarter back for the next run; a deterministic one skips only that filing.
`fetch_13f` writes the same rows inline during its walk.
"""

from __future__ import annotations

import logging
from functools import partial

import pandas as pd

from src.constants.constants import SEC_13F_FORMS
from src.context import Context
from src.data_extract.utils.common.edgar_driver import FilingStamp
from src.data_extract.utils.common.parallel_fetch import run_per_ticker
from src.data_extract.utils.common.sec_io import company, company_filings, configure
from src.data_extract.utils.institutionals.fetch_13f import _IMPLIED_PRICE_BAND, ReadFailure, _latest_per_key, _read_filing, _save_book
from src.data_store.schema import Tables
from src.utils.superinvestor_roster import roster_cik_union, roster_map_as_of

logger = logging.getLogger(__name__)


class SuperinvestorRosterEmptyError(RuntimeError):
    """`superinvestor_roster` holds no CIK; the roster is the walk's entire input, so the run stops."""


def _listed_filings(cik: str, since: pd.Timestamp) -> list[tuple[FilingStamp, pd.Timestamp]]:
    """`(stamp, period)` for the CIK's 13F-HR filings whose period is on/after `since`, oldest
    first by (filed, amendment last, accession) -- edgartools lists newest first; a null or
    unparseable period is skipped."""
    listing = company_filings(company(cik), SEC_13F_FORMS) or []
    stamps = sorted((FilingStamp.of(f, cik) for f in listing), key=lambda s: (s.filed, s.is_amendment, s.accession_number))
    periods = [pd.to_datetime(s.period_of_report, errors="coerce") for s in stamps]
    return [(s, period.normalize()) for s, period in zip(stamps, periods, strict=True) if pd.notna(period) and period >= since]


def _stored_filings(context: Context, cik: str) -> set[tuple[pd.Timestamp, pd.Timestamp]]:
    """Distinct normalised `(period, filing_date)` pairs of the CIK's stored book, from one
    projected, CIK-scoped read (DATE reads back as `date`)."""
    df_stored = context.store.load(Tables.sec13f_manager_holdings, columns=["period", "filing_date"], where={"cik": cik}, optional=True)
    if df_stored is None:
        return set()
    df_pairs = df_stored.drop_duplicates()
    return {(pd.Timestamp(p).normalize(), pd.Timestamp(f).normalize()) for p, f in zip(df_pairs["period"], df_pairs["filing_date"], strict=True)}


def _pending_filings(context: Context, cik: str, listed: list[tuple[FilingStamp, pd.Timestamp]]) -> list[tuple[FilingStamp, pd.Timestamp]]:
    """The listed filings the stored book does not show yet: a filing is done when the book holds
    rows with its own `(period, filing_date)` pair, and so is every filing of that period filed on
    or before it (a restatement overwrites its original's rows). No side state."""
    stored = _stored_filings(context, cik)
    newest_done: dict[pd.Timestamp, pd.Timestamp] = {}
    for stamp, period in listed:
        if (period, stamp.filed.normalize()) in stored:
            newest_done[period] = stamp.filed.normalize()
    return [(s, p) for s, p in listed if p not in newest_done or s.filed.normalize() > newest_done[p]]


def _log_read_failure(context: Context, cik: str, stamp: FilingStamp, period: pd.Timestamp, failure: ReadFailure) -> None:
    """WARNING for a transient failure (its quarter is held back), ERROR for a deterministic one."""
    if failure.transient:
        context.log.warning(
            "13F managers: CIK %s %s (period %s) failed transiently (%s); the quarter is held back and retried next run",
            cik,
            stamp.accession_number,
            period.date(),
            failure.reason,
        )
        return
    context.log.error(
        "13F managers: CIK %s %s (period %s) is unreadable (%s); the filing is skipped and the quarter's readable filings are saved",
        cik,
        stamp.accession_number,
        period.date(),
        failure.reason,
    )


def _catch_up_cik(label: str, cik: str, *, context: Context, since: pd.Timestamp) -> tuple[int, int, int]:
    """Read and save one roster CIK's pending filings, the last filed winning per (cik, period,
    cusip). A transient read failure keeps its whole period out of this save, so the period stays
    pending (no stored rows) and is retried next run; a deterministic failure skips only that
    filing. `label` is the log key `run_per_ticker` passes first. Returns (rows saved,
    suspect-price rows, transient failures)."""
    frames: list[pd.DataFrame] = []
    held_periods: list[pd.Timestamp] = []
    for stamp, period in _pending_filings(context, cik, _listed_filings(cik, since)):
        rows = _read_filing(stamp)
        if isinstance(rows, ReadFailure):
            _log_read_failure(context, cik, stamp, period, rows)
        elif not rows.empty:
            frames.append(rows)
        if isinstance(rows, ReadFailure) and rows.transient:
            held_periods.append(period)
    if not frames:
        return 0, 0, len(held_periods)
    book = pd.concat(frames, ignore_index=True)
    saved, suspect = _save_book(context, _latest_per_key(book[~book["period"].isin(held_periods)]))
    return saved, suspect, len(held_periods)


def _warn_empty_books(context: Context, empty: list[str], n_ciks: int) -> None:
    """Warn on book-less roster CIKs that produced no rows, naming those with `sec13f_hr`
    history: those were throttled or failed transiently and must be re-run; the rest never filed."""
    if not empty:
        return
    known_filers = set(context.store.distinct(Tables.sec13f_hr, "cik", where={"cik": [c.lstrip("0") for c in empty] + empty}))
    known = {c for c in empty if c in known_filers or c.lstrip("0") in known_filers}
    context.log.warning(
        "13F managers: %d/%d roster CIK(s) with no stored book produced NO rows; %d of them have "
        "history in %s and must be re-run (throttling or a transient listing failure, not an "
        "absent filer): %s",
        len(empty),
        n_ciks,
        len(known),
        Tables.sec13f_hr,
        ", ".join(sorted(known)) or "none",
    )


def fetch_13f_managers(context: Context, years_history: int = 15) -> int:
    """Catch every roster CIK's book up with every listed filing it does not show yet; returns rows
    saved. `years_history` bounds by PERIOD. A failed CIK counts as zero rows. Raises
    `SuperinvestorRosterEmptyError` when the roster holds no CIK."""
    context.ensure_edgar_identity()
    configure(context)
    ciks = sorted(roster_cik_union(context))
    if not ciks:
        raise SuperinvestorRosterEmptyError(
            "superinvestor_roster is empty -- run `data_extract superinvestors -F` first. "
            "The roster is this walk's entire scope; there is nothing to fetch without it."
        )
    names = roster_map_as_of(context)  # latest snapshot; only used for log lines
    with_book = set(context.store.distinct(Tables.sec13f_manager_holdings, "cik"))
    since = pd.Timestamp.today().normalize() - pd.DateOffset(years=years_history)
    logger.info("13F managers: %d roster CIK(s), %d with a stored book, periods from %s", len(ciks), len(with_book & set(ciks)), since.date())

    worker = partial(_catch_up_cik, context=context, since=since)
    df_scope = pd.DataFrame({"cik_label": [names.get(c, c) for c in ciks], "cik": ciks})
    guarded = run_per_ticker(df_scope, worker, desc="13F manager books", log=context.log, key_cols=("cik_label", "cik"))
    results = [result or (0, 0, 0) for result in guarded]
    saved = sum(n for n, _, _ in results)
    suspect = sum(s for _, s, _ in results)
    failed = [(c, f) for c, (_, _, f) in zip(ciks, results, strict=True) if f]

    _warn_empty_books(context, [c for c, (n, _, _) in zip(ciks, results, strict=True) if n == 0 and c not in with_book], len(ciks))
    if failed:
        logger.warning(
            "13F managers: %d filing read(s) failed transiently across %d CIK(s); their quarters were held back and are retried next run: %s",
            sum(f for _, f in failed),
            len(failed),
            ", ".join(c for c, _ in failed),
        )
    if suspect:
        logger.warning(
            "13F managers: %d/%d saved rows imply a share price outside %s -- check "
            "edgartools' per-filing $thousands detection before trusting value_usd",
            suspect,
            saved,
            _IMPLIED_PRICE_BAND,
        )
    logger.info("13F managers: saved %d row(s) across %d manager(s) -> %s", saved, len(ciks), Tables.sec13f_manager_holdings)
    return saved
