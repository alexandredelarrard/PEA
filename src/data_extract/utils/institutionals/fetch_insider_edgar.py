"""Daily EDGAR ownership filings for the open quarterly-bulk gap.

The live table is provisional. Quarterly ZIP rows remain canonical whenever an accession is
present in both sources; the aggregation loader performs that accession-level overlay. The local
EDGAR index lists a Form 3/4/5 under its issuer and under every reporting owner, so a filing whose
XML issuer is not the key's company (the key is only an owner) becomes an empty-filing marker, as
does a holdings-only filing with no transaction row.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from functools import partial
from typing import Any, cast

import pandas as pd

from src.constants.constants import SEC_INSIDER_FORMS
from src.context import Context
from src.data_extract.utils.common import edgar_index
from src.data_extract.utils.common.edgar_driver import EdgarFetch, EdgarScope, FilingStamp, run_edgar_fetch
from src.data_extract.utils.common.identity import Identity
from src.data_extract.utils.common.registrant import issuer_ciks, listing_ciks, resolve_registrant_entries
from src.data_extract.utils.common.resume import DONE_PER_KEY
from src.data_extract.utils.common.sec_io import ParseFailureError, TransientReadError, filing_header, filing_xml
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


def insider_filings(
    context: Context,
    ticker: str,
    cik: str,
    *,
    since: pd.Timestamp | None,
    through: pd.Timestamp,
    scope: EdgarScope,
) -> list[Any]:
    """`ticker`'s Forms 3/4/5 in the local EDGAR index filed in `[since, through]`, oldest first (issuer and owner roles)."""
    df_index = edgar_index.entries(context, set(listing_ciks(ticker, cik, scope.registrants, scope.identity)), SEC_INSIDER_FORMS, since=since)
    df_rows = resolve_registrant_entries(ticker, cik, df_index, SEC_INSIDER_FORMS, registrants=scope.registrants, identity=scope.identity)
    df_rows = df_rows[df_rows["filed"] <= pd.Timestamp(through).normalize()]
    return [edgar_index.index_filing(r.cik, r.company, r.form, r.filed, r.accession) for r in df_rows.itertuples(index=False)]


def _acceptance_datetime(filing: object) -> pd.Timestamp:
    """The header's acceptance time (NaT when the header has none); a transient read raises."""
    try:
        raw = getattr(filing_header(filing), "acceptance_datetime", None)
    except TransientReadError:
        raise
    except Exception:  # noqa: BLE001 -- optional EDGAR header metadata
        raw = None
    value = pd.to_datetime(cast(Any, raw), errors="coerce")
    if pd.notna(value) and value.tz is not None:
        value = value.tz_localize(None)
    return value


def _filing_strings(filing: Any) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One ownership filing -> (canonical string transactions, footnotes), keyed on its accession.

    A filing without ownership XML raises `ParseFailureError`."""
    xml = filing_xml(filing)
    if not xml:
        raise ParseFailureError(f"{getattr(filing, 'accession_number', '?')}: no ownership XML")
    df_str, df_notes = extract_xml_strings(xml)
    accession = str(filing.accession_number)
    return df_str.assign(accession_number=accession), df_notes.assign(accession_number=accession)


def _filing_metadata(filing: Any) -> dict[str, object]:
    return {
        "accession_number": str(filing.accession_number),
        "filing_date": pd.Timestamp(filing.filing_date).normalize(),
        "acceptance_datetime": _acceptance_datetime(filing),
    }


def _ticker_strings(filings: Sequence[Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Ownership filings -> (canonical string transactions, footnotes, filing metadata), keyed on accession."""
    transaction_frames: list[pd.DataFrame] = []
    footnote_frames: list[pd.DataFrame] = []
    metadata: list[dict[str, object]] = []
    for filing in filings:
        df_str, df_notes = _filing_strings(filing)
        if df_str.empty:
            continue
        transaction_frames.append(df_str)
        if not df_notes.empty:
            footnote_frames.append(df_notes)
        metadata.append(_filing_metadata(filing))
    df_str = pd.concat(transaction_frames, ignore_index=True) if transaction_frames else pd.DataFrame()
    df_notes = pd.concat(footnote_frames, ignore_index=True) if footnote_frames else empty_footnotes()
    return df_str, df_notes, pd.DataFrame(metadata)


def _screen_live_rows(df_str: pd.DataFrame, df_meta: pd.DataFrame, universe: Sequence[str], identity: Identity) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Type the string rows, attach filing metadata and an `edgar:` row key, then screen into (kept, quarantine)."""
    df_built = build_insider_frame(df_str, value_rule="stated_total_first", numeric_rule="strip_currency_float", date_formats=LIVE_DATE_FORMATS)
    df_built = df_built.merge(df_meta.drop_duplicates("accession_number", keep="last"), on="accession_number", how="left")
    df_built["transaction_sk"] = "edgar:" + df_built["security_type"].astype(str) + ":" + df_built["source_row_sequence"].astype(str)
    return screen_insider_rows(df_built, universe, identity)


def _screened_frames(
    df_str: pd.DataFrame, df_notes: pd.DataFrame, df_meta: pd.DataFrame, universe: Sequence[str], identity: Identity
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """String rows -> (kept transactions, footnotes of kept accessions, quarantine rows)."""
    df_kept, df_quarantine = _screen_live_rows(df_str, df_meta, universe, identity)
    df_kept_notes = filter_footnotes(df_notes, set(df_kept["accession_number"])) if not df_kept.empty else empty_footnotes()
    return df_kept, df_kept_notes, df_quarantine


def live_insider_frames(filings: Sequence[Any], *, universe: Sequence[str], identity: Identity) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Ownership filings -> (kept transactions, footnotes of kept accessions, quarantine rows)."""
    df_str, df_notes, df_meta = _ticker_strings(filings)
    if df_str.empty:
        return pd.DataFrame(), empty_footnotes(), pd.DataFrame()
    return _screened_frames(df_str, df_notes, df_meta, universe, identity)


def _owner_role(df_str: pd.DataFrame, ticker: str, cik: str, scope: EdgarScope) -> bool:
    """True when the XML names an issuer CIK outside the key's lineage (the key is only a reporting owner)."""
    issuer = pad_cik(df_str["issuer_cik"].dropna().iloc[0]) if "issuer_cik" in df_str and df_str["issuer_cik"].notna().any() else ""
    lineage = issuer_ciks(ticker, cik, scope.registrants, scope.identity)
    return bool(issuer and lineage and issuer not in lineage)


def parse_insider(ticker: str, cik: str, stamp: FilingStamp, scope: EdgarScope, *, universe: Sequence[str]) -> dict[Table, pd.DataFrame]:
    """One ownership filing -> live rows (kept by the identity screen), their footnotes and the
    quarantine; nothing for a holdings-only filing or one where the key is only an owner."""
    if scope.identity is None:
        raise ValueError("insider live fetch needs an identity-aware scope")
    df_str, df_notes = _filing_strings(stamp.filing)
    if df_str.empty or _owner_role(df_str, ticker, cik, scope):
        return {}
    df_meta = pd.DataFrame([_filing_metadata(stamp.filing)])
    df_kept, df_notes, df_quarantine = _screened_frames(df_str, df_notes, df_meta, universe, scope.identity)
    live = pd.DataFrame(columns=_LIVE_COLUMNS)
    if not df_kept.empty:
        df_live = df_kept.assign(fetched_at=pd.Timestamp.now(tz="UTC").tz_localize(None))
        live = df_live[[c for c in _LIVE_COLUMNS if c in df_live.columns]].drop_duplicates(
            subset=list(Tables.insider_transactions_live.pk), keep="last"
        )
    return {
        Tables.insider_footnotes: df_notes,
        Tables.insider_transactions_quarantine: df_quarantine,
        Tables.insider_transactions_live: live,
    }


def insider_coverage(ticker: str, as_of: pd.Timestamp, failed_dates: list[pd.Timestamp]) -> pd.DataFrame:
    """The key's coverage row: complete through the day before its oldest unread filing, else through `as_of`."""
    through = min(failed_dates) - pd.Timedelta(days=1) if failed_dates else as_of
    return pd.DataFrame(
        [{"ticker": ticker, "complete_through": pd.Timestamp(through).normalize(), "updated_at": pd.Timestamp.now(tz="UTC").tz_localize(None)}]
    )


def bulk_frontier_floor(context: Context) -> pd.Timestamp | None:
    """The day after the last quarter of the bulk data set (the live table covers only what follows)."""
    _, latest_quarter = context.store.bounds(Tables.insider_transactions, "quarter")
    if latest_quarter is None:
        return None
    try:
        return pd.Period(str(latest_quarter).upper(), freq="Q").end_time.normalize() + pd.Timedelta(days=1)
    except (TypeError, ValueError):
        return None


def insider_fetch(tickers: Sequence[str]) -> EdgarFetch:
    """The live Forms 3/4/5 fetch for universe `tickers`; done per key (an owner-role marker under one
    company never hides the filing from its issuer)."""
    return EdgarFetch(
        desc="insider Forms 3/4/5 (EDGAR live)",
        tables=(Tables.insider_footnotes, Tables.insider_transactions_quarantine, Tables.insider_transactions_live),
        forms=tuple(SEC_INSIDER_FORMS),
        parse=partial(parse_insider, universe=list(tickers)),
        done_table=Tables.insider_transactions_live,
        done_scope=DONE_PER_KEY,
        coverage=insider_coverage,
        coverage_table=Tables.insider_transactions_live_coverage,
        runtime_floor=bulk_frontier_floor,
    )


def fetch_insider_edgar(
    context: Context,
    tickers: list[str],
    years_history: int,
    *,
    full: bool = False,
    as_of: pd.Timestamp | None = None,
    no_cap: bool = False,
) -> None:
    """Fill the open-quarter tail from the local index; each key's coverage row advances to its oldest unread filing."""
    run_edgar_fetch(context, tickers, years_history, insider_fetch(tickers), full=full, as_of=as_of, no_cap=no_cap)
