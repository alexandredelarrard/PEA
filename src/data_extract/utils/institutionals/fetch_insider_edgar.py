"""Daily EDGAR ownership filings for the open quarterly-bulk gap.

The live table is provisional. Quarterly ZIP rows remain canonical whenever an accession is
present in both sources; the aggregation loader performs that accession-level overlay.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from functools import partial
from typing import Any, cast
from xml.etree import ElementTree

import pandas as pd
from edgar import Filing

from src.constants.constants import SEC_INSIDER_FORMS
from src.context import Context
from src.data_extract.utils.common.edgar_driver import EdgarFetch, EdgarScope, run_edgar_fetch
from src.data_extract.utils.common.identity import Identity, load_identity
from src.data_extract.utils.common.registrant import resolve_registrant_filings
from src.data_extract.utils.common.sec_atom import (
    SEC_INSIDER_FORM_FAMILIES,
    SEC_INSIDER_OWNER_ATOM_PAGE_SIZE,
    atom_filing,
    atom_page_url,
    fetch_atom_entries,
    parse_atom_entry,
)
from src.data_extract.utils.institutionals.fetch_insider_transactions import (
    _filter_universe,
    _repair_transaction_dates,
    _to_quarantine,
)
from src.data_extract.utils.institutionals.insider_edgar_parser import (
    FOOTNOTE_COLUMNS,
    TRANSACTION_COLUMNS,
    parse_ownership_xml,
)
from src.data_store.schema import Table, Tables
from src.utils.string import pad_cik

_LIVE_COLUMNS = (
    "accession_number",
    *TRANSACTION_COLUMNS,
    "filing_date",
    "acceptance_datetime",
    "fetched_at",
)
_LOG = logging.getLogger(__name__)


def ownership_filings(
    ticker: str,
    cik: str,
    *,
    since: pd.Timestamp | None,
    through: pd.Timestamp,
    done_accessions: frozenset[str],
) -> list[Filing]:
    """List issuer ownership filings, including forms submitted under an owner's CIK."""
    start_date = pd.Timestamp(since).normalize() if since is not None else None
    end_date = pd.Timestamp(through).normalize()
    filings: dict[str, Filing] = {}
    for family in SEC_INSIDER_FORM_FAMILIES:
        filings.update(_family_filings(ticker, cik, family, start_date, end_date, done_accessions))
    return sorted(filings.values(), key=lambda filing: filing.filing_date)


def _family_filings(
    ticker: str,
    cik: str,
    family: str,
    start_date: pd.Timestamp | None,
    end_date: pd.Timestamp,
    done_accessions: frozenset[str],
) -> dict[str, Filing]:
    """Page one form family newest-first until a short page or an entry older than `start_date`.

    A failed page logs a warning and ends this family with the pages already read.
    """
    filings: dict[str, Filing] = {}
    start = 0
    while True:
        url = atom_page_url(pad_cik(cik), family, start_date, end_date, start)
        try:
            entries = fetch_atom_entries(url, f"{ticker} {family} offset {start}", retry=False)
        except Exception as exc:  # noqa: BLE001 -- preserve other discovery channels
            _LOG.warning("ownership filing search failed for %s form %s at offset %d: %r", ticker, family, start, exc)
            break
        if not entries:
            break
        page, oldest = _page_filings(entries, ticker, cik, start_date, end_date, done_accessions)
        filings.update(page)
        if len(entries) < SEC_INSIDER_OWNER_ATOM_PAGE_SIZE or (start_date is not None and oldest < start_date):
            break
        start += SEC_INSIDER_OWNER_ATOM_PAGE_SIZE
    return filings


def _page_filings(
    entries: list[ElementTree.Element],
    ticker: str,
    cik: str,
    start_date: pd.Timestamp | None,
    end_date: pd.Timestamp,
    done_accessions: frozenset[str],
) -> tuple[dict[str, Filing], pd.Timestamp]:
    """Kept insider filings on one page by accession, and the page's oldest dated entry (capped at `end_date`)."""
    target_forms = frozenset(SEC_INSIDER_FORMS)
    filings: dict[str, Filing] = {}
    oldest = end_date
    for raw in entries:
        entry = parse_atom_entry(raw)
        if entry is None:
            continue
        oldest = min(oldest, entry.filing_date)
        if (
            entry.form not in target_forms
            or entry.accession is None
            or entry.accession in done_accessions
            or entry.filing_date > end_date
            or (start_date is not None and entry.filing_date < start_date)
        ):
            continue
        filings[entry.accession] = atom_filing(entry, cik=cik, company=ticker)
    return filings, oldest


def insider_filings(
    ticker: str,
    cik: str,
    *,
    since: pd.Timestamp | None,
    through: pd.Timestamp,
    done_accessions: frozenset[str],
    scope: EdgarScope,
) -> list[Any]:
    """Union issuer submissions with the owner-inclusive issuer search."""
    issuer_filings = resolve_registrant_filings(
        ticker,
        SEC_INSIDER_FORMS,
        since=since,
        done_accessions=done_accessions,
        registrants=scope.registrants,
        identity=scope.identity,
    )
    discovered: dict[str, Any] = {str(filing.accession_number): filing for filing in issuer_filings}
    for filing in ownership_filings(
        ticker,
        cik,
        since=since,
        through=through,
        done_accessions=done_accessions,
    ):
        discovered.setdefault(str(filing.accession_number), filing)
    return sorted(
        discovered.values(),
        key=lambda filing: pd.Timestamp(filing.filing_date),
    )


def _acceptance_datetime(filing: object) -> pd.Timestamp:
    filing_obj = cast(Any, filing)
    try:
        raw = getattr(filing_obj.header, "acceptance_datetime", None)
    except Exception:  # noqa: BLE001 -- optional EDGAR header metadata
        raw = None
    value = pd.to_datetime(cast(Any, raw), errors="coerce")
    if pd.notna(value) and value.tz is not None:
        value = value.tz_localize(None)
    return value


def _filing_frames(
    filing: object,
    *,
    universe: Sequence[str],
    identity: Identity,
    fetched_at: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """One ownership filing -> kept transactions, footnotes, and rejected transactions."""
    filing_obj = cast(Any, filing)
    xml = filing_obj.xml()
    if not xml:
        raise ValueError(f"{getattr(filing_obj, 'accession_number', '?')}: no ownership XML")
    transactions, footnotes = parse_ownership_xml(xml)
    if transactions.empty:
        return transactions, pd.DataFrame(columns=("accession_number", *FOOTNOTE_COLUMNS)), pd.DataFrame()

    accession = str(filing_obj.accession_number)
    transactions = transactions.assign(
        accession_number=accession,
        filing_date=pd.Timestamp(filing_obj.filing_date).normalize(),
        acceptance_datetime=_acceptance_datetime(filing_obj),
        fetched_at=fetched_at,
    )
    transactions = _repair_transaction_dates(transactions)
    kept, rejected = _filter_universe(transactions, universe, identity)

    kept_notes = pd.DataFrame(columns=("accession_number", *FOOTNOTE_COLUMNS))
    if not kept.empty and not footnotes.empty:
        kept_notes = footnotes.assign(accession_number=accession)[["accession_number", *FOOTNOTE_COLUMNS]]

    quarantine = pd.DataFrame()
    if not rejected.empty:
        rejected = rejected.assign(
            transaction_sk=("edgar:" + rejected["security_type"].astype(str) + ":" + rejected["source_row_sequence"].astype(str))
        )
        quarantine = _to_quarantine(rejected)
    return kept, kept_notes, quarantine


def build_ticker_insider_edgar(
    ticker: str,
    cik: str,
    *,
    since: pd.Timestamp | None = None,
    done_accessions: frozenset[str] = frozenset(),
    universe: Sequence[str],
    identity: Identity,
    scan_through: pd.Timestamp,
    scope: EdgarScope,
    rescan_stored: bool = False,
) -> dict[Table, pd.DataFrame]:
    """Build live rows for one ticker and a coverage row even when no filing was found.

    `rescan_stored` (a `--full` run) ignores `done_accessions` and re-reads stored filings.
    """
    fetched_at = pd.Timestamp.now(tz="UTC").tz_localize(None)
    transaction_frames: list[pd.DataFrame] = []
    footnote_frames: list[pd.DataFrame] = []
    quarantine_frames: list[pd.DataFrame] = []
    for filing in insider_filings(
        ticker,
        cik,
        since=since,
        through=scan_through,
        done_accessions=frozenset() if rescan_stored else done_accessions,
        scope=scope,
    ):
        transactions, footnotes, quarantine = _filing_frames(
            filing,
            universe=universe,
            identity=identity,
            fetched_at=fetched_at,
        )
        if not transactions.empty:
            transaction_frames.append(transactions)
        if not footnotes.empty:
            footnote_frames.append(footnotes)
        if not quarantine.empty:
            quarantine_frames.append(quarantine)

    live = pd.concat(transaction_frames, ignore_index=True) if transaction_frames else pd.DataFrame(columns=_LIVE_COLUMNS)
    if not live.empty:
        live = live[[column for column in _LIVE_COLUMNS if column in live.columns]]
        live = live.drop_duplicates(subset=list(Tables.insider_transactions_live.pk), keep="last")
    notes = pd.concat(footnote_frames, ignore_index=True) if footnote_frames else pd.DataFrame(columns=("accession_number", *FOOTNOTE_COLUMNS))
    rejected = pd.concat(quarantine_frames, ignore_index=True) if quarantine_frames else pd.DataFrame()
    coverage = pd.DataFrame([{"ticker": ticker, "complete_through": scan_through, "updated_at": fetched_at}])
    return {
        Tables.insider_transactions_live: live,
        Tables.insider_footnotes: notes,
        Tables.insider_transactions_quarantine: rejected,
        Tables.insider_transactions_live_coverage: coverage,
    }


def _bulk_frontier(context: Context) -> pd.Timestamp | None:
    _, latest_quarter = context.store.bounds(Tables.insider_transactions, "quarter")
    if latest_quarter is None:
        return None
    try:
        return pd.Period(str(latest_quarter).upper(), freq="Q").end_time.normalize()
    except (TypeError, ValueError):
        return None


def fetch_insider_edgar(
    context: Context,
    tickers: list[str],
    years_history: int,
    *,
    full: bool = False,
) -> None:
    """Fill the open-quarter tail and advance coverage only for successful tickers."""
    identity = load_identity(context)
    scan_through = pd.Timestamp.today().normalize()
    bulk_frontier = _bulk_frontier(context)
    minimum_since = bulk_frontier + pd.Timedelta(days=1) if bulk_frontier is not None else None

    fetch = EdgarFetch(
        desc="insider Forms 3/4/5 (EDGAR live)",
        tables=(
            Tables.insider_transactions_live,
            Tables.insider_footnotes,
            Tables.insider_transactions_quarantine,
            Tables.insider_transactions_live_coverage,
        ),
        build=partial(build_ticker_insider_edgar, universe=tickers, identity=identity, scan_through=scan_through, rescan_stored=full),
        identity_aware=False,
        minimum_since=minimum_since,
        completion_table=Tables.insider_transactions_live_coverage,
    )
    run_edgar_fetch(context, tickers, years_history, fetch, full=full)
