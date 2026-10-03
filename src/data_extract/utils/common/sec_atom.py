"""SEC owner-inclusive company-browse Atom feed: page URL, page fetch, entry parse, `Filing`.

The feed lists filings where the queried CIK is the issuer OR a reporting person. This module
builds, fetches, decodes and pages the feed and holds the shared keep filter; callers own their
failure policy and early stop.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from functools import partial
from typing import Any, TypeIs, cast
from urllib.parse import quote_plus
from xml.etree import ElementTree

import edgar
import edgar.httprequests
import pandas as pd

from src.data_extract.utils.common.rate_limit import call_with_retries

ATOM_NAMESPACE = {"atom": "http://www.w3.org/2005/Atom"}
SEC_INSIDER_FORM_FAMILIES = ("3", "4", "5")
SEC_INSIDER_OWNER_ATOM_URL = (
    "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}"
    "&type={form}&datea={date_from}&dateb={date_to}&owner=include"
    "&start={start}&count={count}&output=atom"
)
SEC_INSIDER_OWNER_ATOM_PAGE_SIZE = 100


class AtomPageError(RuntimeError):
    """One feed page failed to load or parse; `offset` is its `start`, the cause is chained."""

    def __init__(self, offset: int) -> None:
        super().__init__(f"SEC Atom page at offset {offset} failed")
        self.offset = offset


@dataclass(frozen=True)
class AtomEntry:
    """One feed entry; `filing_date` is normalised to midnight."""

    form: str | None
    accession: str | None
    filing_date: pd.Timestamp
    file_number: str | None


def atom_page_url(cik: str, family: str, date_from: pd.Timestamp | None, date_to: pd.Timestamp, start: int) -> str:
    """One page of `family` filings for `cik` between two inclusive dates; no lower bound when `date_from` is None."""
    return SEC_INSIDER_OWNER_ATOM_URL.format(
        cik=cik,
        form=quote_plus(family),
        date_from=date_from.strftime("%Y%m%d") if date_from is not None else "",
        date_to=date_to.strftime("%Y%m%d"),
        start=start,
        count=SEC_INSIDER_OWNER_ATOM_PAGE_SIZE,
    )


def fetch_atom_entries(url: str, label: str, *, retry: bool) -> list[ElementTree.Element]:
    """The `<entry>` elements of one page. Raises on a failed request, an empty payload or bad XML.

    `retry` routes the request through `call_with_retries` (labelled `label`).
    """
    download = partial(edgar.httprequests.download_text, url)
    payload = call_with_retries(download, label=label) if retry else download()
    if payload is None:
        raise ValueError("SEC Atom response was empty")
    return ElementTree.fromstring(payload).findall("atom:entry", ATOM_NAMESPACE)


def _atom_text(entry: ElementTree.Element, name: str) -> str | None:
    value = entry.findtext(f"atom:content/atom:{name}", namespaces=ATOM_NAMESPACE)
    return value.strip() if value and value.strip() else None


def parse_atom_entry(entry: ElementTree.Element) -> AtomEntry | None:
    """Decode one entry; None when its filing date is missing or unparseable."""
    filing_date = pd.to_datetime(cast(Any, _atom_text(entry, "filing-date")), errors="coerce")
    if pd.isna(filing_date):
        return None
    return AtomEntry(
        form=_atom_text(entry, "filing-type"),
        accession=_atom_text(entry, "accession-number"),
        filing_date=filing_date.normalize(),
        file_number=_atom_text(entry, "file-number"),
    )


def iter_atom_pages(
    cik: str, family: str, date_from: pd.Timestamp | None, date_to: pd.Timestamp, label: str, *, retry: bool
) -> Iterator[tuple[int, list[AtomEntry | None]]]:
    """`(offset, decoded entries)` per page, ending after the first short (or empty) page. Lazy: a
    caller that stops iterating requests no further page. A failed page raises `AtomPageError`
    chained to its cause; `label` prefixes the retry log key."""
    start = 0
    while True:
        url = atom_page_url(cik, family, date_from, date_to, start)
        try:
            entries = fetch_atom_entries(url, f"{label} {family} offset {start}", retry=retry)
        except Exception as exc:  # noqa: BLE001 -- each caller owns its failure policy
            raise AtomPageError(start) from exc
        yield start, [parse_atom_entry(raw) for raw in entries]
        if len(entries) < SEC_INSIDER_OWNER_ATOM_PAGE_SIZE:
            return
        start += SEC_INSIDER_OWNER_ATOM_PAGE_SIZE


def keep_atom_entry(
    entry: AtomEntry | None, forms: frozenset[str], done: frozenset[str], date_from: pd.Timestamp | None, date_to: pd.Timestamp
) -> TypeIs[AtomEntry]:
    """True for a dated entry of one of `forms` whose accession is known and not in `done`, filed
    within `[date_from, date_to]` (no lower bound when `date_from` is None)."""
    return (
        entry is not None
        and entry.form in forms
        and entry.accession is not None
        and entry.accession not in done
        and entry.filing_date <= date_to
        and (date_from is None or entry.filing_date >= date_from)
    )


def atom_filing(entry: AtomEntry, *, cik: str, company: str) -> edgar.Filing:
    """An edgartools `Filing` for a kept entry, indexed under `cik`."""
    return edgar.Filing(
        cik=int(cik),
        company=company,
        form=cast(str, entry.form),
        filing_date=entry.filing_date.strftime("%Y-%m-%d"),
        accession_no=cast(str, entry.accession),
    )
