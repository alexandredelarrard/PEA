"""
fetch_13f_managers.py (src/data_extract/utils/institutionals/fetch_13f_managers.py)
------------------------------------------------------------------------------------
Per-CIK catch-up of `sec13f_manager_holdings` (complete CUSIP books, no universe filter) for every CIK
ever on `superinvestor_roster`: each CIK re-reads only filings filed on/after its stored `filing_date`
frontier, with `fetch_13f`'s parser and save. `fetch_13f` writes the same rows inline during its walk.
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
from src.data_extract.utils.institutionals.fetch_13f import _IMPLIED_PRICE_BAND, _read_filing, _save_book
from src.data_store.schema import Tables
from src.utils.superinvestor_roster import roster_cik_union, roster_map_as_of

logger = logging.getLogger(__name__)


class SuperinvestorRosterEmptyError(RuntimeError):
    """`superinvestor_roster` holds no CIK. The roster IS this walk's entire input, so an empty
    one must stop the run rather than let it report success over zero managers -- the failure
    mode a warning would produce is a table that silently stops growing."""


def _filings_to_read(cik: str, floor: pd.Timestamp | None, since: pd.Timestamp) -> list[FilingStamp]:
    """The CIK's 13F-HR filings filed on/after `floor` (all when None) whose period is on/after
    `since`; a null or unparseable period is skipped."""
    stamps = [FilingStamp.of(f, cik) for f in Company(cik).get_filings(form=SEC_13F_FORMS) or []]
    if floor is not None:
        stamps = [s for s in stamps if s.filed.normalize() >= floor]
    periods = [pd.to_datetime(s.period_of_report, errors="coerce") for s in stamps]
    return [s for s, period in zip(stamps, periods, strict=True) if pd.notna(period) and period >= since]


def _catch_up_cik(label: str, cik: str, *, context: Context, frontier: dict[str, pd.Timestamp], since: pd.Timestamp) -> tuple[int, int]:
    """Read and save one roster CIK's filings newer than its frontier. `label` is the log key
    `run_per_ticker` passes first. Returns (rows saved, suspect-price rows)."""
    frames = [rows for stamp in _filings_to_read(cik, frontier.get(cik), since) if not (rows := _read_filing(stamp)).empty]
    if not frames:
        return 0, 0
    return _save_book(context, pd.concat(frames, ignore_index=True))


def _warn_empty_books(context: Context, empty: list[str], n_ciks: int) -> None:
    """Warn on frontier-less roster CIKs that produced no rows, naming those with `sec13f_hr`
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
    """Catch every roster CIK's book up from its stored `filing_date` frontier; returns rows saved.
    `years_history` bounds by PERIOD. A failed CIK counts as zero rows. Raises
    `SuperinvestorRosterEmptyError` when the roster holds no CIK."""
    context.ensure_edgar_identity()
    ciks = sorted(roster_cik_union(context))
    if not ciks:
        raise SuperinvestorRosterEmptyError(
            "superinvestor_roster is empty -- run `data_extract superinvestors --seed` first. "
            "The roster is this walk's entire scope; there is nothing to fetch without it."
        )
    names = roster_map_as_of(context)  # latest snapshot; only used for log lines
    frontier = context.store.max_date_by(Tables.sec13f_manager_holdings, "cik", "filing_date")
    since = pd.Timestamp.today().normalize() - pd.DateOffset(years=years_history)
    logger.info("13F managers: %d roster CIK(s), %d with a stored frontier, periods from %s", len(ciks), len(frontier), since.date())

    worker = partial(_catch_up_cik, context=context, frontier=frontier, since=since)
    scope = pd.DataFrame({"cik_label": [names.get(c, c) for c in ciks], "cik": ciks})
    guarded = run_per_ticker(scope, worker, desc="13F manager books", log=context.log, key_cols=("cik_label", "cik"))
    results = [result or (0, 0) for result in guarded]
    saved = sum(n for n, _ in results)
    suspect = sum(s for _, s in results)

    _warn_empty_books(context, [c for c, (n, _) in zip(ciks, results, strict=True) if n == 0 and c not in frontier], len(ciks))
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
