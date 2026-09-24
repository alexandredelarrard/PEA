"""Daily EDGAR ownership filings for the open quarterly-bulk gap.

The live table is provisional. Quarterly ZIP rows remain canonical whenever an accession is
present in both sources; the aggregation loader performs that accession-level overlay.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from xml.etree import ElementTree

import pandas as pd
from edgar import Filing
from edgar.httprequests import download_text

from src.constants.constants import (
    SEC_INSIDER_FORM_FAMILIES,
    SEC_INSIDER_FORMS,
    SEC_INSIDER_OWNER_ATOM_PAGE_SIZE,
    SEC_INSIDER_OWNER_ATOM_URL,
)
from src.context import Context
from src.data_extract.utils.common.edgar_driver import new_filings, run_edgar_fetch
from src.data_extract.utils.common.identity import Identity, load_identity
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

_LIVE_COLUMNS = (
    "accession_number",
    *TRANSACTION_COLUMNS,
    "filing_date",
    "acceptance_datetime",
    "fetched_at",
)
_ATOM_NAMESPACE = {"atom": "http://www.w3.org/2005/Atom"}
_LOG = logging.getLogger(__name__)


def _atom_text(entry: ElementTree.Element, name: str) -> str | None:
    value = entry.findtext(f"atom:content/atom:{name}", namespaces=_ATOM_NAMESPACE)
    return value.strip() if value and value.strip() else None


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
    target_forms = frozenset(SEC_INSIDER_FORMS)
    filings: dict[str, Filing] = {}
    for family in SEC_INSIDER_FORM_FAMILIES:
        start = 0
        while True:
            url = SEC_INSIDER_OWNER_ATOM_URL.format(
                cik=str(cik).zfill(10),
                form=family,
                date_from=start_date.strftime("%Y%m%d") if start_date is not None else "",
                date_to=end_date.strftime("%Y%m%d"),
                start=start,
                count=SEC_INSIDER_OWNER_ATOM_PAGE_SIZE,
            )
            try:
                root = ElementTree.fromstring(download_text(url))
            except Exception as exc:  # noqa: BLE001 -- preserve other discovery channels
                _LOG.warning(
                    "ownership filing search failed for %s form %s at offset %d: %r",
                    ticker,
                    family,
                    start,
                    exc,
                )
                break
            entries = root.findall("atom:entry", _ATOM_NAMESPACE)
            if not entries:
                break
            oldest = end_date
            for entry in entries:
                form = _atom_text(entry, "filing-type")
                accession = _atom_text(entry, "accession-number")
                filing_date = pd.to_datetime(_atom_text(entry, "filing-date"), errors="coerce")
                if pd.notna(filing_date):
                    oldest = min(oldest, filing_date.normalize())
                if (
                    form not in target_forms
                    or accession is None
                    or accession in done_accessions
                    or pd.isna(filing_date)
                    or filing_date.normalize() > end_date
                    or (start_date is not None and filing_date.normalize() < start_date)
                ):
                    continue
                filings[accession] = Filing(
                    cik=int(cik),
                    company=ticker,
                    form=form,
                    filing_date=filing_date.strftime("%Y-%m-%d"),
                    accession_no=accession,
                )
            if len(entries) < SEC_INSIDER_OWNER_ATOM_PAGE_SIZE or (start_date is not None and oldest < start_date):
                break
            start += SEC_INSIDER_OWNER_ATOM_PAGE_SIZE
    return sorted(filings.values(), key=lambda filing: filing.filing_date)


def insider_filings(
    ticker: str,
    cik: str,
    *,
    since: pd.Timestamp | None,
    through: pd.Timestamp,
    done_accessions: frozenset[str],
) -> list[object]:
    """Union issuer submissions with the owner-inclusive issuer search."""
    discovered = {str(filing.accession_number): filing for filing in new_filings(ticker, SEC_INSIDER_FORMS, since, done_accessions)}
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
    try:
        raw = getattr(filing.header, "acceptance_datetime", None)
    except Exception:  # noqa: BLE001 -- optional EDGAR header metadata
        raw = None
    value = pd.to_datetime(raw, errors="coerce")
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
    xml = filing.xml()
    if not xml:
        raise ValueError(f"{getattr(filing, 'accession_number', '?')}: no ownership XML")
    transactions, footnotes = parse_ownership_xml(xml)
    if transactions.empty:
        return transactions, pd.DataFrame(columns=("accession_number", *FOOTNOTE_COLUMNS)), pd.DataFrame()

    accession = str(filing.accession_number)
    transactions = transactions.assign(
        accession_number=accession,
        filing_date=pd.Timestamp(filing.filing_date).normalize(),
        acceptance_datetime=_acceptance_datetime(filing),
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
) -> dict[Table, pd.DataFrame]:
    """Build live rows for one ticker and a coverage row even when no filing was found."""
    fetched_at = pd.Timestamp.now(tz="UTC").tz_localize(None)
    transaction_frames: list[pd.DataFrame] = []
    footnote_frames: list[pd.DataFrame] = []
    quarantine_frames: list[pd.DataFrame] = []
    for filing in insider_filings(
        ticker,
        cik,
        since=since,
        through=scan_through,
        done_accessions=done_accessions,
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

    def build(
        ticker: str,
        cik: str,
        *,
        since: pd.Timestamp | None,
        done_accessions: frozenset[str],
    ) -> dict[Table, pd.DataFrame]:
        return build_ticker_insider_edgar(
            ticker,
            cik,
            since=since,
            done_accessions=(frozenset() if full else done_accessions),
            universe=tickers,
            identity=identity,
            scan_through=scan_through,
        )

    run_edgar_fetch(
        context,
        tickers,
        years_history,
        tables=(
            Tables.insider_transactions_live,
            Tables.insider_footnotes,
            Tables.insider_transactions_quarantine,
            Tables.insider_transactions_live_coverage,
        ),
        build=build,
        desc="insider Forms 3/4/5 (EDGAR live)",
        minimum_since=minimum_since,
        completion_table=Tables.insider_transactions_live_coverage,
        full=full,
    )
