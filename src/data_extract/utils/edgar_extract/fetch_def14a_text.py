"""DEF 14A text: each proxy's carved section payloads, untruncated, for the DEF 14A LLM step.

Writes `sec_def14a_text`, one row per non-empty section (`executive_pay`, `director_pay`, `audit`, `ownership`,
`governance`) with its `payload`, `n_chars`, the submissions `primary_document` and the filing stamp. The filer's submissions listing is read once per CIK holding pending proxies; the document is its
primary document, else the `<accession>.txt` full submission, read with `sec_get`. A walked proxy with no payload
in any section is one visible `section = "_none"` row. A multi-window ticker's proxy whose SGML subject names none
of its entity CIKs becomes an empty-filing marker; a transient SEC failure or an unreadable listing stores nothing
and the filing is listed again.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from functools import partial
from typing import cast

import pandas as pd
from edgar import Filing

from src.context import Context
from src.data_extract.utils.common.registrant import header_subject_ciks
from src.data_extract.utils.common.resume import DONE_PER_KEY
from src.data_extract.utils.common.sec_io import TransientReadError, sec_get
from src.data_extract.utils.edgar_extract.edgar_driver import EdgarFetch, EdgarScope, FilingStamp, parse_filing_rows
from src.data_extract.utils.edgar_extract.edgar_fillings import archive_url, list_filings
from src.data_extract.utils.edgar_extract.html_text import html_to_text
from src.data_extract.utils.edgar_extract.parallel_fetch import PROGRAMMING_ERRORS
from src.data_extract.utils.structure.def14a.carve import prepare_def14a_payloads
from src.data_store.schema import Resume, Tables

logger = logging.getLogger(__name__)

#: The `section` of the one row a walked proxy with no payload in any section gets.
NO_PAYLOAD = "_none"


@dataclass(frozen=True)
class ProxyFiling:
    """A listed proxy as the walk reads it: what `FilingStamp.of` reads plus the submissions `primaryDocument`
    ('' when SEC names none or the listing does not hold the filing)."""

    accession_number: str
    cik: str
    form: str
    filing_date: str
    company: str
    primary_document: str

    @property
    def doc_url(self) -> str:
        """The primary document, else the `<accession>.txt` full submission."""
        return archive_url(self.cik, self.accession_number, self.primary_document or f"{self.accession_number}.txt")

    @property
    def txt_url(self) -> str:
        return archive_url(self.cik, self.accession_number, f"{self.accession_number}.txt")


def subject_accepted(ticker: str, filer_cik: str, accession: str, subjects: frozenset[str], accepted: frozenset[str]) -> bool:
    """Reject only a known subject that is outside the accepted registrant entity."""
    if not subjects or not subjects.isdisjoint(accepted):
        return True
    logger.warning(
        "%s: rejecting DEF 14A accession %s before extraction; filer CIK %s; subject CIK(s) %s; accepted entity CIK(s) %s; "
        "reason=subject_cik_disjoint",
        ticker,
        accession,
        filer_cik,
        ",".join(sorted(subjects)),
        ",".join(sorted(accepted)),
    )
    return False


def proxy_is_subject(ticker: str, cik: str, stamp: FilingStamp, scope: EdgarScope) -> bool:
    """The subject guard of a multi-window ticker: False (a marker) when the proxy's SGML subject names none of its
    event CIKs. A transient header failure raises (listed again); any other unreadable header accepts the proxy."""
    # read: the accepted subjects (a multi-window ticker's event CIKs), then the proxy's SGML subject CIKs
    filing_scope = scope.filing_scope(ticker, cik)
    accepted = frozenset(filing_scope.event_ciks) if len(filing_scope.windows) > 1 else frozenset()
    if not accepted:
        return True
    company = str(getattr(stamp.filing, "company", ""))
    try:
        filed = pd.Timestamp(stamp.filed).date().isoformat()
        header = Filing(cik=int(stamp.cik), company=company, form=str(stamp.form), filing_date=filed, accession_no=stamp.accession_number)
        subjects = header_subject_ciks(header)
    except TransientReadError:
        raise
    except Exception as exc:  # noqa: BLE001 -- an unknown header follows the existing path
        logger.info("%s: DEF 14A accession %s subject header unavailable (%s); continuing", ticker, stamp.accession_number, exc)
        return True
    return subject_accepted(ticker, stamp.cik, stamp.accession_number, subjects, accepted)


def proxy_filings(context: Context, ticker: str, df_units: pd.DataFrame) -> dict[str, ProxyFiling]:
    """The pending proxies by accession, with their submissions primary document: one listing per
    filer CIK, from 31 days before its oldest pending index date (SEC's `filingDate` can precede it). A listing that
    cannot be read raises `TransientReadError`, so the key's proxies are listed again; a programming error propagates."""
    forms = cast(Resume, Tables.sec_def14a_text.resume).forms
    listed: list[pd.DataFrame] = []
    for cik, units in df_units.groupby(df_units["cik"].astype(str), sort=False):
        first = pd.Timestamp(units["filed"].min()).normalize()
        try:
            listed.append(list_filings(context, cik, forms, first - pd.Timedelta(days=32)))
        except (TransientReadError, *PROGRAMMING_ERRORS):
            raise
        except Exception as exc:  # noqa: BLE001 -- an unreadable listing is retried, never stored
            raise TransientReadError(f"{ticker} CIK {cik} submissions not listed ({type(exc).__name__}: {exc})") from exc
    return _by_accession(df_units, listed)


def _by_accession(df_units: pd.DataFrame, listed: Sequence[pd.DataFrame]) -> dict[str, ProxyFiling]:
    """One `ProxyFiling` per pending unit: the listing's primary document, else none."""
    found = {str(row["accession_number"]): row for part in listed for row in part.to_dict("records")}
    out: dict[str, ProxyFiling] = {}
    for unit in df_units.itertuples(index=False):
        row = found.get(str(unit.accession), {})
        primary = str(row.get("primary_document") or "")
        out[str(unit.accession)] = ProxyFiling(str(unit.accession), str(unit.cik), str(unit.form), str(unit.filed), str(unit.company), primary)
    if missing := sorted(out.keys() - found.keys()):
        logger.warning("DEF 14A text: %d indexed proxy(ies) absent from the submissions listing: %s", len(missing), ", ".join(missing[:20]))
    return out


def _section_rows(ticker: str, stamped: Mapping[str, object], payloads: Mapping[str, str]) -> list[dict]:
    """One row per non-empty section payload, untruncated; one `_none` row when every section is empty."""
    rows = [{"ticker": ticker, **stamped, "section": key, "payload": text, "n_chars": len(text)} for key, text in payloads.items() if text.strip()]
    return rows or [{"ticker": ticker, **stamped, "section": NO_PAYLOAD, "payload": "", "n_chars": 0}]


def _def14a_rows(context: Context, ticker: str, stamp: FilingStamp) -> list[dict]:
    """The proxy's section rows: its document carved into the five section payloads."""
    # read: the listed document, else the full submission when the primary is not served (a transient failure raises)
    filing = cast(ProxyFiling, stamp.filing)
    try:
        html = sec_get(context, filing.doc_url).text
    except TransientReadError:
        raise
    except Exception:
        if filing.txt_url == filing.doc_url:
            raise
        logger.info("%s: primary document unavailable, falling back to the full submission", stamp.accession_number)
        html = sec_get(context, filing.txt_url).text

    # compute: the carved payloads, stamped with the filing
    payloads = prepare_def14a_payloads(html, html_to_text(html))
    stamped = {
        "cik": stamp.cik,
        "accession_number": stamp.accession_number,
        "form": str(stamp.form),
        "filing_date": stamp.filed.normalize(),
        "primary_document": filing.primary_document,
    }
    return _section_rows(ticker, stamped, payloads)


@contextmanager
def _def14a_run_scope(context: Context, tickers: Sequence[str], as_of: pd.Timestamp) -> Iterator[EdgarFetch]:
    """Bind the run's `Context`: the submissions listing and the document reads go through `sec_get(context, ...)`."""
    del tickers, as_of
    yield replace(
        DEF14A_TEXT_FETCH,
        parse=partial(parse_filing_rows, table=Tables.sec_def14a_text, row_fn=partial(_def14a_rows, context)),
        filings=partial(proxy_filings, context),
    )


DEF14A_TEXT_FETCH = EdgarFetch(
    tables=(Tables.sec_def14a_text,),
    is_subject=proxy_is_subject,
    done_scope=DONE_PER_KEY,
    run_scope=_def14a_run_scope,
)
