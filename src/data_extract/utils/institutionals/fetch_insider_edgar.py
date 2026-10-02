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
from src.data_extract.utils.institutionals.insider_common import (
    INSIDER_COLUMNS,
    LIVE_DATE_FORMATS,
    build_insider_frame,
    empty_footnotes,
    filter_footnotes,
    screen_insider_rows,
)
from src.data_extract.utils.institutionals.insider_edgar_parser import extract_xml_strings
from src.data_store.schema import Table, Tables
from src.utils.string import pad_cik

_LIVE_COLUMNS = (
    "accession_number",
    "source_row_sequence",
    *(column for column in INSIDER_COLUMNS if column not in ("accession_number", "transaction_sk", "filing_date", "quarter")),
    "footnote_ids",
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


def _ticker_strings(filings: Sequence[Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Ownership filings -> (canonical string transactions, footnotes, filing metadata), keyed on accession."""
    transaction_frames: list[pd.DataFrame] = []
    footnote_frames: list[pd.DataFrame] = []
    metadata: list[dict[str, object]] = []
    for filing in filings:
        xml = filing.xml()
        if not xml:
            raise ValueError(f"{getattr(filing, 'accession_number', '?')}: no ownership XML")
        df_str, df_notes = extract_xml_strings(xml)
        if df_str.empty:
            continue
        accession = str(filing.accession_number)
        transaction_frames.append(df_str.assign(accession_number=accession))
        if not df_notes.empty:
            footnote_frames.append(df_notes.assign(accession_number=accession))
        metadata.append(
            {
                "accession_number": accession,
                "filing_date": pd.Timestamp(filing.filing_date).normalize(),
                "acceptance_datetime": _acceptance_datetime(filing),
            }
        )
    df_str = pd.concat(transaction_frames, ignore_index=True) if transaction_frames else pd.DataFrame()
    df_notes = pd.concat(footnote_frames, ignore_index=True) if footnote_frames else empty_footnotes()
    return df_str, df_notes, pd.DataFrame(metadata)


def _screen_live_rows(df_str: pd.DataFrame, df_meta: pd.DataFrame, universe: Sequence[str], identity: Identity) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Type the string rows, attach filing metadata and an `edgar:` row key, then screen into (kept, quarantine)."""
    df_built = build_insider_frame(df_str, value_rule="stated_total_first", numeric_rule="strip_currency_float", date_formats=LIVE_DATE_FORMATS)
    df_built = df_built.merge(df_meta.drop_duplicates("accession_number", keep="last"), on="accession_number", how="left")
    df_built["transaction_sk"] = "edgar:" + df_built["security_type"].astype(str) + ":" + df_built["source_row_sequence"].astype(str)
    return screen_insider_rows(df_built, universe, identity)


def live_insider_frames(filings: Sequence[Any], *, universe: Sequence[str], identity: Identity) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Ownership filings -> (kept transactions, footnotes of kept accessions, quarantine rows)."""
    df_str, df_notes, df_meta = _ticker_strings(filings)
    if df_str.empty:
        return pd.DataFrame(), empty_footnotes(), pd.DataFrame()
    df_kept, df_quarantine = _screen_live_rows(df_str, df_meta, universe, identity)
    df_kept_notes = filter_footnotes(df_notes, set(df_kept["accession_number"])) if not df_kept.empty else empty_footnotes()
    return df_kept, df_kept_notes, df_quarantine


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
    filings = insider_filings(
        ticker, cik, since=since, through=scan_through, done_accessions=frozenset() if rescan_stored else done_accessions, scope=scope
    )
    df_kept, df_notes, df_quarantine = live_insider_frames(filings, universe=universe, identity=identity)

    live = pd.DataFrame(columns=_LIVE_COLUMNS)
    if not df_kept.empty:
        df_live = df_kept.assign(fetched_at=fetched_at)
        live = df_live[[column for column in _LIVE_COLUMNS if column in df_live.columns]].drop_duplicates(
            subset=list(Tables.insider_transactions_live.pk), keep="last"
        )
    coverage = pd.DataFrame([{"ticker": ticker, "complete_through": scan_through, "updated_at": fetched_at}])
    return {
        Tables.insider_transactions_live: live,
        Tables.insider_footnotes: df_notes,
        Tables.insider_transactions_quarantine: df_quarantine if not df_quarantine.empty else pd.DataFrame(),
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
