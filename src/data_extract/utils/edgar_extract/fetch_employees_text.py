"""10-K family employees text: the document a workforce count is read from, as candidate windows and their union.

Writes `sec_employees_text`, one row per filing: `source_document` ("primary" or an annual-report exhibit type),
`windows` (JSON `[start, end]` candidate windows in selection priority order), `pieces` (JSON `[offset, text]` of
the windows' merged union), `n_chars` (the chosen document's length) and the filing stamp. The employees LLM step
rebuilds its excerpt from the stored windows at any budget. A filing with no usable text becomes an empty-filing
marker; a transient SEC failure raises and the filing is listed again.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from functools import partial

from edgar import Filing

from src.data_extract.utils.common.sec_io import filing_attachments, forget_sgml, sec_call
from src.data_extract.utils.common.workforce import merge_spans, workforce_number_spans
from src.data_extract.utils.edgar_extract.edgar_driver import EdgarFetch, FilingStamp, parse_filing_rows
from src.data_extract.utils.edgar_extract.html_text import html_to_text
from src.data_store.schema import Tables

_OPENING = 8_000  # chars of the filing's opening text, a candidate after the workforce-number windows
_WINDOW = (550, 850)  # chars kept before and after a workforce mention
_BY_REFERENCE_REACH = 1_000  # max chars between "incorporated by reference" and a workforce topic

_IX_HEADER_RE = re.compile(r"<ix:header\b.*?</ix:header\s*>", re.I | re.S)
_CONTEXT_RE = re.compile(
    r"\b(?:employees?|workforce|associates?|team\s+members?|human\s+capital|"
    r"personnel|staff|colleagues?|full[- ]time|part[- ]time)\b",
    re.I,
)
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
    Each read runs under the `sec_io` retry policy; a transient failure raises.
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
    raw = sec_call(getattr, attachment, "content", label=label)
    return _visible_text(raw.decode("utf-8", "replace") if isinstance(raw, bytes) else str(raw or ""))


def employee_text(filing: Filing) -> EmployeeText:
    """The text the employee excerpt is cut from: the primary document, or its annual-report exhibit fallback."""
    return choose_employee_text(filing_body_text(filing), _annual_report_exhibits(filing))


def candidate_windows(text: str, spans: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    """Windows around the workforce-number `spans`, then the opening, then windows around other workforce mentions,
    in selection priority order, each clipped to the text."""
    before, after = _WINDOW
    windows = [(max(0, start - before), min(len(text), end + after)) for start, end in spans]
    windows.append((0, min(_OPENING, len(text))))
    windows += [(max(0, match.start() - before), min(len(text), match.end() + after)) for match in _CONTEXT_RE.finditer(text)]
    return windows


def _employee_rows(ticker: str, stamp: FilingStamp) -> list[dict]:
    """One row for the filing's chosen document; none when it has no usable text."""
    # read: the stamp (period of report first, as the listing did), then the document; the submission is dropped after
    stamped = stamp.columns(Tables.sec_employees_text)
    document = employee_text(stamp.filing)
    forget_sgml(stamp.filing)
    if not document.text.strip():
        return []

    # compute: candidate windows in priority order and the text of their merged union
    windows = candidate_windows(document.text, document.spans)
    pieces = [[start, document.text[start:end]] for start, end in merge_spans(windows)]
    return [
        {
            "ticker": ticker,
            **stamped,
            "source_document": document.source_document,
            "windows": json.dumps([list(window) for window in windows]),
            "pieces": json.dumps(pieces),
            "n_chars": len(document.text),
        }
    ]


EMPLOYEES_TEXT_FETCH = EdgarFetch(
    tables=(Tables.sec_employees_text,),
    parse=partial(parse_filing_rows, table=Tables.sec_employees_text, row_fn=_employee_rows),
)
