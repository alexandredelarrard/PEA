"""The document a 10-K family filing's workforce count is read from: the primary body, or its annual-report exhibit.

A primary document that states no workforce number, or points its workforce disclosure at a page of a document it
incorporates by reference, falls back to the first EX-13 / annual-report EX-99 that states one. Every SEC read runs
under the `sec_io` retry policy; a transient failure raises.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from functools import partial

from edgar import Filing

from src.data_extract.utils.common.sec_io import filing_attachments, sec_call
from src.data_extract.utils.common.workforce import workforce_number_spans
from src.data_extract.utils.edgar_extract.html_text import html_to_text

_BY_REFERENCE_REACH = 1_000  # max chars between "incorporated by reference" and a workforce topic

_IX_HEADER_RE = re.compile(r"<ix:header\b.*?</ix:header\s*>", re.I | re.S)
_BY_REFERENCE_RE = re.compile(r"\bincorporated\s+(?:herein\s+)?by\s+reference\b", re.I)
# A workforce topic pointed at a page of another document: "Human Capital—page 15".
_WORKFORCE_PAGE_RE = re.compile(r"\b(?:human\s+capital|employees|(?:persons|people)\s+employed)\b[^.]{0,80}?\bpages?\s+[a-z]?-?\d", re.I)


@dataclass(frozen=True)
class EmployeeText:
    """The document text the employee excerpt is cut from, which document it is ("primary" or an exhibit type),
    and its workforce-number spans."""

    text: str
    source_document: str
    spans: tuple[tuple[int, int], ...]


def filing_body_text(filing: Filing) -> str:
    """Primary-document text (visible HTML table cells included, no OCR), else the full submission; "" if none.

    edgartools raises AttributeError for a filing with no primary document; that is absorbed as a filing property.
    """
    readers: tuple[tuple[str, Callable[[str], str]], ...] = (
        ("html", _visible_text),
        ("text", str),
        ("full_text_submission", _visible_text),
    )
    for method, to_text in readers:
        try:
            raw = sec_call(getattr(filing, method), label=f"{getattr(filing, 'accession_number', '?')} {method}")
        except AttributeError:  # edgartools dereferences a missing primary document
            continue
        if raw and (text := to_text(raw)).strip():
            return text
    return ""


def _visible_text(raw: str) -> str:
    """`html_to_text` without the hidden inline-XBRL header, whose fact values are not filing prose."""
    return html_to_text(_IX_HEADER_RE.sub(" ", raw))


def is_annual_report_exhibit(document_type: str, description: str) -> bool:
    """An EX-13 (annual report to security holders) or an EX-99 its filer describes as an annual report."""
    kind = document_type.strip().upper()
    return kind.startswith("EX-13") or (kind.startswith("EX-99") and "annual report" in description.casefold())


def _incorporates_workforce(text: str) -> bool:
    """Whether the text points its workforce disclosure at a page of a document it incorporates by reference."""
    reach = _BY_REFERENCE_REACH
    return any(_WORKFORCE_PAGE_RE.search(text, max(0, match.start() - reach), match.end() + reach) for match in _BY_REFERENCE_RE.finditer(text))


def choose_employee_text(primary: str, exhibits: Iterable[tuple[str, Callable[[], str]]]) -> EmployeeText:
    """The primary document, unless it states no workforce number or incorporates its workforce disclosure
    by reference; then the first annual-report exhibit (read lazily, in order) that states one."""
    primary_spans = tuple(workforce_number_spans(primary))
    if primary_spans and not _incorporates_workforce(primary):
        return EmployeeText(primary, "primary", primary_spans)
    for document_type, read in exhibits:
        if spans := tuple(workforce_number_spans(text := read())):
            return EmployeeText(text, document_type, spans)
    return EmployeeText(primary, "primary", primary_spans)


def _annual_report_exhibits(filing: Filing) -> Iterator[tuple[str, Callable[[], str]]]:
    """The filing's annual-report exhibits as (document type, text reader); attachments are listed on first use."""
    for attachment in filing_attachments(filing) or []:
        document_type = str(attachment.document_type or "").strip().upper()
        if attachment.is_binary() or not is_annual_report_exhibit(document_type, str(attachment.description or "")):
            continue
        label = f"{getattr(filing, 'accession_number', '?')} {document_type}"
        yield document_type, partial(_attachment_text, attachment, label)


def _attachment_text(attachment: object, label: str) -> str:
    """The visible text of one attachment, bytes decoded as UTF-8 with replacement."""
    raw = sec_call(getattr, attachment, "content", label=label)
    return _visible_text(raw.decode("utf-8", "replace") if isinstance(raw, bytes) else str(raw or ""))


def employee_text(filing: Filing) -> EmployeeText:
    """The text the employee excerpt is cut from: the primary document, or its annual-report exhibit fallback."""
    return choose_employee_text(filing_body_text(filing), _annual_report_exhibits(filing))
