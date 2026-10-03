"""
fetch_13f_managers.py (src/data_extract/utils/institutionals/fetch_13f_managers.py)
------------------------------------------------------------------------------------
Per-CIK catch-up of `sec13f_manager_holdings` (complete CUSIP books, no universe filter) for every CIK
ever on `superinvestor_roster`, with `fetch_13f`'s parser and save. A listed filing is done when the
stored book shows it, or a later filing of the same period, by `(period, filing_date)`; every other
filing is read, so a failed read or an inline-written newer filing never hides older gaps.
`fetch_13f` writes the same rows inline during its walk.
"""

from __future__ import annotations

import logging
from functools import partial

import pandas as pd
from edgar import Company

from src.constants.constants import SEC_13F_FORMS
from src.context import Context
from src.data_extract.utils.common.edgar_driver import FilingStamp
from src.data_extract.utils.common.parallel_fetch import run_per_ticker
from src.data_extract.utils.common.run_manifest import record_run
from src.data_extract.utils.institutionals.fetch_13f import _IMPLIED_PRICE_BAND, _latest_per_key, _read_filing, _save_book
from src.data_store.schema import Tables
from src.utils.superinvestor_roster import roster_cik_union, roster_map_as_of

logger = logging.getLogger(__name__)


class SuperinvestorRosterEmptyError(RuntimeError):
    """`superinvestor_roster` holds no CIK; the roster is the walk's entire input, so the run stops."""


def _listed_filings(cik: str, since: pd.Timestamp) -> list[tuple[FilingStamp, pd.Timestamp]]:
    """`(stamp, period)` for the CIK's 13F-HR filings whose period is on/after `since`, oldest
    first by (filed, accession) -- edgartools lists newest first; a null or unparseable period is
    skipped."""
    listing = Company(cik).get_filings(form=SEC_13F_FORMS) or []
    stamps = sorted((FilingStamp.of(f, cik) for f in listing), key=lambda s: (s.filed, s.accession_number))
    periods = [pd.to_datetime(s.period_of_report, errors="coerce") for s in stamps]
    return [(s, period.normalize()) for s, period in zip(stamps, periods, strict=True) if pd.notna(period) and period >= since]


def _stored_dates(context: Context, cik: str, column: str) -> set[pd.Timestamp]:
    """Distinct normalised `column` dates of the CIK's stored book (DATE reads back as `date`)."""
    values = context.store.distinct(Tables.sec13f_manager_holdings, column, where={"cik": cik})
    return {pd.Timestamp(v).normalize() for v in values}


def _pending_filings(context: Context, cik: str, listed: list[tuple[FilingStamp, pd.Timestamp]]) -> list[tuple[FilingStamp, pd.Timestamp]]:
    """The listed filings the stored book does not show yet: a filing is done when its period and
    filing date are both stored, and so is every filing of that period filed on or before it (a
    restatement overwrites its original's rows). Two scoped DISTINCT reads, no side state."""
    periods, filed = _stored_dates(context, cik, "period"), _stored_dates(context, cik, "filing_date")
    newest_done: dict[pd.Timestamp, pd.Timestamp] = {}
    for stamp, period in listed:
        if period in periods and stamp.filed.normalize() in filed:
            newest_done[period] = stamp.filed.normalize()
    return [(s, p) for s, p in listed if p not in newest_done or s.filed.normalize() > newest_done[p]]


def _catch_up_cik(label: str, cik: str, *, context: Context, since: pd.Timestamp) -> tuple[int, int, int]:
    """Read and save one roster CIK's pending filings, the last filed winning per (cik, period,
    cusip). A failed read keeps its whole period out of this save, so the period stays pending and
    is retried next run; other periods still save. `label` is the log key `run_per_ticker` passes
    first. Returns (rows saved, suspect-price rows, failed reads)."""
    frames: list[pd.DataFrame] = []
    failed_periods: list[pd.Timestamp] = []
    for stamp, period in _pending_filings(context, cik, _listed_filings(cik, since)):
        rows = _read_filing(stamp)
        if rows is None:
            failed_periods.append(period)
        elif not rows.empty:
            frames.append(rows)
    if not frames:
        return 0, 0, len(failed_periods)
    book = pd.concat(frames, ignore_index=True)
    saved, suspect = _save_book(context, _latest_per_key(book[~book["period"].isin(failed_periods)]))
    return saved, suspect, len(failed_periods)


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
    ciks = sorted(roster_cik_union(context))
    if not ciks:
        raise SuperinvestorRosterEmptyError(
            "superinvestor_roster is empty -- run `data_extract superinvestors --seed` first. "
            "The roster is this walk's entire scope; there is nothing to fetch without it."
        )
    names = roster_map_as_of(context)  # latest snapshot; only used for log lines
    with_book = set(context.store.distinct(Tables.sec13f_manager_holdings, "cik"))
    since = pd.Timestamp.today().normalize() - pd.DateOffset(years=years_history)
    logger.info("13F managers: %d roster CIK(s), %d with a stored book, periods from %s", len(ciks), len(with_book & set(ciks)), since.date())

    worker = partial(_catch_up_cik, context=context, since=since)
    scope = pd.DataFrame({"cik_label": [names.get(c, c) for c in ciks], "cik": ciks})
    guarded = run_per_ticker(scope, worker, desc="13F manager books", log=context.log, key_cols=("cik_label", "cik"))
    results = [result or (0, 0, 0) for result in guarded]
    saved = sum(n for n, _, _ in results)
    suspect = sum(s for _, s, _ in results)
    failed = [(c, f) for c, (_, _, f) in zip(ciks, results, strict=True) if f]

    _warn_empty_books(context, [c for c, (n, _, _) in zip(ciks, results, strict=True) if n == 0 and c not in with_book], len(ciks))
    if failed:
        logger.warning(
            "13F managers: %d filing read(s) failed across %d CIK(s); their periods were not saved and are retried next run: %s",
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
    record_run(context, Tables.sec13f_manager_holdings, len(ciks), saved)
    return saved
