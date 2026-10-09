"""Source-backed SEC annual employee counts, dated by the filing's public date."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from functools import partial
from itertools import combinations
from typing import Literal, cast

import pandas as pd
from edgar import Filing
from omegaconf import DictConfig
from pydantic import BaseModel, Field

from src.context import Context
from src.data_extract.utils.common.edgar_driver import EdgarScope, FilingStamp
from src.data_extract.utils.common.edgar_extract import html_to_text
from src.data_extract.utils.common.identity import Identity, load_identity
from src.data_extract.utils.common.parallel_fetch import run_per_ticker
from src.data_extract.utils.common.sec_io import configure, filing_attachments, forget_sgml, sec_call
from src.data_extract.utils.common.sec_utils import load_cik_mapping
from src.data_store.schema import Tables
from src.gpt_extract.transformers.gpt_getter import LLMExtractor
from src.gpt_extract.transformers.step_gpt_extracter import with_gpt_overrides
from src.gpt_extract.utils.schemas_gpt import LlmResult, LlmTask

HEADCOUNT_FORMS = ("10-K", "10-K/A", "10-K405")
# The stored row: filer identity, the guarded components, their basis and status, and where they were read.
FRAME_COLUMNS = [
    "ticker",
    "as_of",
    "cik",
    "accession_number",
    "form",
    "employees_total",
    "employees_full_time",
    "employees_part_time",
    "basis",
    "status",
    "source_document",
    "source_quote",
    "measurement_period",
]
_COUNT_COLUMNS = ("employees_total", "employees_full_time", "employees_part_time")
_GAP = "[... filing gap ...]"
_FRAGMENT_GAP = 2_000  # max source chars between two `...`-joined quote fragments
_FAILED_SHOWN = 20  # failed tickers named in the coverage log
_OPENING = 8_000  # chars of the filing's opening text sent after the workforce-number windows
_WINDOW = (550, 850)  # chars kept before and after a workforce mention
_BY_REFERENCE_REACH = 1_000  # max chars between "incorporated by reference" and a workforce topic

_IX_HEADER_RE = re.compile(r"<ix:header\b.*?</ix:header\s*>", re.I | re.S)

_CONTEXT_RE = re.compile(
    r"\b(?:employees?|workforce|associates?|team\s+members?|human\s+capital|"
    r"personnel|staff|colleagues?|full[- ]time|part[- ]time)\b",
    re.I,
)
_NUMBER = r"(?P<n>(?<![\d,.])(?!(?:19|20)\d\d\b)(?:\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?\s*(?:thousand|million)\b|\d{2,}))"
_NOUN = r"(?:employees|people|persons|associates|team\s+members|workforce|headcount|full[\s-]*time|part[\s-]*time)"
# A workforce number: a number a few words from a workforce noun or `employ(s/ed)`, or after a headcount label.
_WORKFORCE_NUMBER_RES = tuple(
    re.compile(pattern, re.I)
    for pattern in (
        rf"{_NUMBER}(?:[\s-]+[^\s\d-]+){{0,4}}?[\s-]+(?:{_NOUN}|employ(?:s|ed)?)\b",
        rf"\b(?:{_NOUN}|employ(?:s|ed)?)(?:[\s:,-]+[^\s\d:,-]+){{0,4}}?[\s:,-]+{_NUMBER}",
        rf"\b(?:headcount|number\s+of\s+(?:[a-z]+\s+){{0,2}}?employees)\b[^.\d]{{0,60}}{_NUMBER}",
    )
)
_BY_REFERENCE_RE = re.compile(r"\bincorporated\s+(?:herein\s+)?by\s+reference\b", re.I)
# A workforce topic pointed at a page of another document: "Human Capital—page 15".
_WORKFORCE_PAGE_RE = re.compile(r"\b(?:human\s+capital|employees|(?:persons|people)\s+employed)\b[^.]{0,80}?\bpages?\s+[a-z]?-?\d", re.I)
_ELLIPSIS_RE = re.compile(r"\s*(?:\.\s*\.\s*\.|…)\s*")
_QUANTITY_RE = re.compile(r"(?<![\d,.])(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?(?:\s*(thousand|million)\b)?", re.I)
_SCALES = {"thousand": 1_000, "million": 1_000_000}
_IN_THOUSANDS_RE = re.compile(r"\(\s*(?:in\s+)?thousands\s*\)|\bin thousands\b", re.I)
_LOWER_BOUND_RE = re.compile(r"\b(?:over|more than|greater than|at least|in excess of|exceeding)\s*$", re.I)
_UPPER_BOUND_RE = re.compile(r"\b(?:nearly|almost|under|less than|fewer than|up to|at most)\s*$", re.I)
_RANGE_BEFORE_RE = re.compile(
    r"(?:\bbetween\s*|\bbetween\s+[\d,.]+\s*(?:thousand|million)?\s+and\s*|\d\s*(?:thousand\s*|million\s*)?(?:-|–|to)\s*)$", re.I
)
_RANGE_AFTER_RE = re.compile(r"^\s*(?:-|–|to)\s*\d", re.I)
# Prose names a number's shift in lower case after it ("11,040 full-time"); a table names it in a
# capitalised row label before the row's numbers ("Part-time Associates 97,913 41,184 13 139,110"),
# so case keeps the next row's label from tagging the number before it. A Total row has no shift.
_SHIFT_AFTER_RE = re.compile(r"^\W*(?:[a-z]+\s+){0,3}?(full|part)[\s-]*time")
_ROW_LABEL_RE = re.compile(
    r"\b(Total|TOTAL|Full|FULL|Part|PART)(?:[\s-]*(?:time|Time|TIME))?\b"
    r"(?:(?!\b(?:Total|TOTAL|Full|FULL|Part|PART)\b)[^\d]){0,40}(?:[\d,.]+(?:\s*%)?(?:\s+|$))*$"
)  # a row's earlier cells may be percentages: "Full-time 30,497 76 % 9,716 24 % 40,213"
# Prose may label the number before it, in its own phrase: "the number of full-time employees ... was 38,100". Only a
# "number of" label counts: in "N1 part-time employees in the US and N2 employees outside", N2 has no shift.
_SHIFT_BEFORE_RE = re.compile(r"\bnumber\s+of\s+(full|part)[\s-]*time\s+(?:employees|associates|staff|workers|team\s+members|people|persons)\b")
_FULL_AND_PART_RE = re.compile(r"full[\s-]*(?:time\s*)?(?:and|or|&|/)\s*part", re.I)
_FTE_AFTER_RE = re.compile(r"^\W*(?:[a-z]+\s+){0,2}?full[\s-]*time[\s-]*equivalent", re.I)
_MONTH_RE = re.compile(r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s*$", re.I)
# Non-employees folded into a number: prose after it is lower case (a capitalised word there is the next table
# row's label); a label or subject before it may be capitalised. "an additional" / "excluding" state them apart.
_CONTRACTOR_RE = re.compile(r"\b(?:contractors?|contingent|temporar(?:y|ies)|complementary)\b")
_CONTRACTOR_LABEL_RE = re.compile(_CONTRACTOR_RE.pattern, re.I)
_SEPARATELY_RE = re.compile(r"\b(?:exclud\w*|not\s+includ\w*|additional|in\s+addition|besides|apart\s+from|other\s+than)\b", re.I)
_PHRASE_STOP_RE = re.compile(r"\d|[.;]\s")  # a number's own phrase ends at the next number or sentence end

COMPONENTS = ("total", "full_time", "part_time")
_COMPONENT_SHIFT = {"total": None, "full_time": "full", "part_time": "part"}  # the shift label a component's number must carry
# Decided statuses. `found`: at least one component is supported. With every component NULL: `not_disclosed`
# (no count stated), `image_only` (only in an image), `incorporated` (in a document the filing incorporates by
# reference but does not include), `ambiguous` (contradictory, a range, or no count claimed), `unsupported`
# (a component was claimed and the guard rejected it).
DECIDED_STATUSES = ("found", "not_disclosed", "image_only", "incorporated", "ambiguous", "unsupported")
# The proxy formula a decided row supports: `total_incl_contractors` and `fte` qualify the total; `full_part` is
# full-time with part-time or with a stated total; `total` alone; `full_time_only`.
BASES = ("total", "full_part", "full_time_only", "fte", "total_incl_contractors")
_NULL_STATUS = {"not_disclosed": "not_disclosed", "image_only": "image_only", "incorporated_by_reference": "incorporated", "ambiguous": "ambiguous"}


_WHOLE = "the whole consolidated registrant (subject 'we', 'the Company', the registrant's name, or the registrant and its subsidiaries)"
_SUBSET = (
    "one subsidiary, operating company, segment, country, site, union or workforce group (such as shipboard or shoreside staff), even the largest"
)
_QUOTE = (
    "Shortest span of the supplied text that contains this number and its noun, copied character for character. If a page "
    "number or header interrupts the span, stop before it and continue after ` ... `; never delete, reorder or reword words. Null when the value is null."
)


class EmployeeAnswer(BaseModel):
    """One filing's company-wide employee components. Fields are generated in order, so the scope check in `reason` comes
    before any number. The JSON schema, descriptions included, is sent to the model as the structured-output format."""

    reason: str = Field(
        description=(
            "Write this first. Name the sentence or table row that states the workforce of " + _WHOLE + ", with its date. "
            "Name each count you reject because it covers only " + _SUBSET + ", a prior year, a range, or contractors. "
            "If you sum disjoint parts that together cover the whole registrant, list every part and the arithmetic."
        )
    )
    status: Literal["found", "not_disclosed", "image_only", "incorporated_by_reference", "ambiguous"] = Field(
        description=(
            "found: total, full_time or part_time is filled. not_disclosed: no company-wide count is stated (counts for subsets "
            "only included). image_only: the count is only in an image. incorporated_by_reference: the filing says the count is in "
            "another document not supplied. ambiguous: only a range is given, or company-wide counts for the same date truly contradict. The same count stated twice at different precision (a rounded narrative and an exact table) is not a contradiction: use the exact one."
        )
    )
    total: int | None = Field(
        description=(
            "Current-period number of employees of " + _WHOLE + ", part-time and seasonal staff included; full-time equivalents "
            "when is_fte. Never a count for " + _SUBSET + ". For 'more than N' or 'over N' return N. Null when no company-wide count is stated."
        )
    )
    total_quote: str | None = Field(description=_QUOTE)
    full_time: int | None = Field(
        description=(
            "Current-period full-time employees of " + _WHOLE + ", only when the company-wide count is labelled full-time. Null when "
            "the full-/part-time split covers only " + _SUBSET + "."
        )
    )
    full_time_quote: str | None = Field(description=_QUOTE)
    part_time: int | None = Field(
        description=(
            "Current-period part-time employees of " + _WHOLE + ", only when the company-wide count is labelled part-time. Null when "
            "the split covers only " + _SUBSET + "."
        )
    )
    part_time_quote: str | None = Field(description=_QUOTE)
    is_fte: bool = Field(description="True when the total is stated as full-time equivalents")
    includes_contractors: bool = Field(description="True when the only stated total folds in contractors, contingent or temporary workers")
    measurement_period: str | None = Field(description="Headcount date or period as stated, at original precision")
    qualifier: str | None = Field(description="For example approximate, over, more than, average or FTE, if stated")


@dataclass(frozen=True)
class EmployeeDecision:
    """The guarded components of one answer, their basis and status; `quotes` are the (component, quote) pairs
    kept, and `rejected` names each claimed component the guard rejected, as `component: reason`."""

    status: str
    employees_total: int | None = None
    employees_full_time: int | None = None
    employees_part_time: int | None = None
    basis: str | None = None
    measurement_period: str | None = None
    quotes: tuple[tuple[str, str], ...] = ()
    rejected: tuple[str, ...] = ()

    @property
    def counted(self) -> bool:
        return any(value is not None for value in (self.employees_total, self.employees_full_time, self.employees_part_time))


@dataclass(frozen=True)
class EmployeeTickerResult:
    """One `FRAME_COLUMNS` row per decided filing date (NULL components with a status when no count is
    supported), the per-filing decisions, and the accessions whose LLM call failed (listed again next run)."""

    frame: pd.DataFrame
    outcomes: list[dict]
    failed: tuple[str, ...] = ()


@dataclass(frozen=True)
class _Decision:
    stamp: FilingStamp
    source_document: str  # "primary" or an exhibit type such as "EX-13"
    decided: EmployeeDecision


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


def _workforce_number_spans(text: str) -> list[tuple[int, int]]:
    """Matches of a workforce number in document order; a day after a month name is not one."""
    spans = {
        (match.start(), match.end())
        for pattern in _WORKFORCE_NUMBER_RES
        for match in pattern.finditer(text)
        if not _MONTH_RE.search(text[max(0, match.start("n") - 12) : match.start("n")])
    }
    return sorted(spans)


def _merge(spans: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def employee_excerpt(text: str, limit: int, spans: Sequence[tuple[int, int]] | None = None) -> str:
    """Windows around workforce numbers, then the filing's opening, then windows around other workforce
    mentions, each kept while the gap-joined excerpt stays within `limit` characters. `spans` are the
    text's `_workforce_number_spans` when the caller already has them."""
    before, after = _WINDOW
    spans = _workforce_number_spans(text) if spans is None else spans
    windows = [(max(0, start - before), min(len(text), end + after)) for start, end in spans]
    windows.append((0, min(_OPENING, len(text))))
    windows += [(max(0, match.start() - before), min(len(text), match.end() + after)) for match in _CONTEXT_RE.finditer(text)]
    separator = f"\n\n{_GAP}\n\n"
    chosen: list[tuple[int, int]] = []
    for window in windows:
        trial = _merge([*chosen, window])
        if sum(end - start for start, end in trial) + len(separator) * (len(trial) - 1) <= limit:
            chosen = trial
    return separator.join(text[start:end] for start, end in chosen)


@dataclass(frozen=True)
class EmployeeText:
    """The document text the employee excerpt is cut from, which document it is ("primary" or an exhibit type),
    and its workforce-number spans when already scanned."""

    text: str
    source_document: str
    spans: tuple[tuple[int, int], ...] | None = None


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
    primary_spans = tuple(_workforce_number_spans(primary))
    if primary_spans and not _incorporates_workforce(primary):
        return EmployeeText(primary, "primary", primary_spans)
    for document_type, read in exhibits:
        if spans := tuple(_workforce_number_spans(text := read())):
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


@dataclass(frozen=True)
class _Quantity:
    """One number the source states inside a located quote."""

    value: int
    bound: int  # +1 for "over N", -1 for "nearly N", 0 for an exact or approximate N
    shift: str | None  # "full", "part" or "fte" when the source labels the number full-, part-time or FTE
    in_range: bool
    is_date: bool
    contractor: bool  # the number's own phrase folds in contractors, contingent or temporary workers


def _compact(text: str) -> tuple[str, list[int]]:
    """Letters and digits only, casefolded, with each kept character's index in `text`.

    Filing text breaks words and numbers in ways a quote does not reproduce (`part-\\ntime`,
    `2018 .`, `A s of`, mis-decoded apostrophes); comparing letters and digits ignores all of them.
    """
    kept = [(char, i) for i, char in enumerate(text.casefold()) if char.isalnum()]
    return "".join(char for char, _ in kept), [i for _, i in kept]


def _chain(fragments: list[str], compact: str, index: list[int], source: str, start: int) -> list[tuple[int, int]] | None:
    """Spans of every fragment from `start` on, each close after the previous with no date or gap between."""
    spans = [(index[start], index[start + len(fragments[0]) - 1] + 1)]
    position = start + len(fragments[0])
    for fragment in fragments[1:]:
        found = compact.find(fragment, position)
        if found < 0:
            return None
        gap = source[spans[-1][1] : index[found]].casefold()
        if len(gap) > _FRAGMENT_GAP or _GAP in gap or re.search(r"\bas of\b", gap) or gap.count("number of employees") > 1:
            return None
        spans.append((index[found], index[found + len(fragment) - 1] + 1))
        position = found + len(fragment)
    return spans


def _locate(quote: str, source: str) -> list[tuple[int, int]] | None:
    """Source spans of the quote's `...`-joined fragments, or None when the source does not contain it."""
    fragments = [fragment for fragment in (_compact(part)[0] for part in _ELLIPSIS_RE.split(quote)) if fragment]
    if not fragments:
        return None
    compact, index = _compact(source)
    start = compact.find(fragments[0])
    while start >= 0:
        if spans := _chain(fragments, compact, index, source, start):
            return spans
        start = compact.find(fragments[0], start + 1)
    return None


def _shift(before: str, after: str) -> str | None:
    row = _ROW_LABEL_RE.search(before)
    if row and row.group(1).casefold() == "total":
        return None
    if _FTE_AFTER_RE.match(after):
        return "fte"
    if _FULL_AND_PART_RE.search(after[:40]):
        return None
    if label := _SHIFT_AFTER_RE.match(after) or row:
        return label.group(1).casefold()
    phrase = _PHRASE_STOP_RE.split(before)[-1]
    if not _FULL_AND_PART_RE.search(phrase) and (label := _SHIFT_BEFORE_RE.search(phrase)):
        return label.group(1)
    return None


def _folds_in_contractors(source: str, start: int, end: int) -> bool:
    """Whether the phrase of the number at `source[start:end]` (to the neighbouring numbers or sentence ends)
    counts contractors, contingent or temporary workers with it, rather than stating them apart."""
    before = _PHRASE_STOP_RE.split(source[max(0, start - 120) : start])[-1]
    after = _PHRASE_STOP_RE.split(source[end : end + 120], maxsplit=1)[0]
    mentioned = _CONTRACTOR_RE.search(after) or _CONTRACTOR_LABEL_RE.search(before)
    return bool(mentioned) and not _SEPARATELY_RE.search(f"{before} {after}")


def _quantities(source: str, spans: list[tuple[int, int]]) -> list[_Quantity]:
    """Every number inside the spans, read with the source words just before and after it."""
    quantities = []
    for start, end in spans:
        in_thousands = bool(_IN_THOUSANDS_RE.search(source[max(0, start - 80) : end]))
        for match in _QUANTITY_RE.finditer(source, start, end):
            before = " ".join(source[max(0, match.start() - 120) : match.start()].split())
            after = " ".join(source[match.end() : match.end() + 60].split())
            digits, fraction, scale = match.groups()
            is_date = (not scale and not fraction and "," not in digits and 1900 <= int(digits) <= 2100) or bool(_MONTH_RE.search(before))
            multiplier = _SCALES[scale.casefold()] if scale else 1_000 if in_thousands and not is_date else 1
            quantities.append(
                _Quantity(
                    value=round(float(digits.replace(",", "") + (fraction or "")) * multiplier),
                    bound=1 if _LOWER_BOUND_RE.search(before) or after.startswith("+") else -1 if _UPPER_BOUND_RE.search(before) else 0,
                    shift=_shift(before, after),
                    in_range=bool(_RANGE_BEFORE_RE.search(before) or _RANGE_AFTER_RE.match(after)),
                    is_date=is_date,
                    contractor=_folds_in_contractors(source, match.start(), match.end()),
                )
            )
    return quantities


def _claimed(quantities: list[_Quantity], count: int) -> list[_Quantity] | None:
    """The stated number equal to the model's count, else the fewest stated components summing to it."""
    pool = [quantity for quantity in quantities if not quantity.is_date]
    for size in range(1, min(len(pool), 4) + 1):
        for combination in combinations(pool, size):
            if sum(quantity.value for quantity in combination) == count:
                return list(combination)
    return None


def _resolve_bound(quantity: _Quantity) -> int:
    """`over N` -> N + half a unit and `nearly N` -> N - half a unit, the unit being N's last
    significant digit at no coarser than two significant digits: over 13,000 -> 13,500,
    over 300,000 -> 305,000, over 6,250 -> 6,255, nearly 2.2 million -> 2,150,000."""
    text = str(quantity.value)
    unit = 10 ** min(len(text) - len(text.rstrip("0")), max(len(text) - 2, 0))
    return quantity.value + quantity.bound * (unit // 2)


def _support(component: str, value: int, quote: str | None, source_text: str) -> list[_Quantity] | str:
    """The stated numbers behind one claimed component, or why the source does not support it.

    The quote must be located in the source, and the value must be one number it states, or a sum of
    stated disjoint numbers, carrying the component's shift label (full- or part-time); a range is unclear.
    """
    if value <= 0 or not quote or _GAP in quote:
        return "no_quote"
    spans = _locate(quote, source_text)
    if not spans:
        return "not_located"
    shift = _COMPONENT_SHIFT[component]
    claimed = _claimed([quantity for quantity in _quantities(source_text, spans) if shift is None or quantity.shift == shift], value)
    if claimed is None:
        return "not_stated"
    return "range" if any(quantity.in_range for quantity in claimed) else claimed


def _settle_total(supported: dict[str, list[_Quantity]], quotes: dict[str, str], rejected: list[str], includes_contractors: bool) -> None:
    """Keep a supported total only when it is a total: one that folds in contractors (unflagged) or is only
    part-time is rejected; one stated only as full-time, or as full- plus part-time numbers, becomes those components."""
    total = supported.get("total")
    if total is None:
        return
    shifts = {quantity.shift for quantity in total}
    if not includes_contractors and any(quantity.contractor for quantity in total):
        rejected.append("total: contractors")
    elif shifts == {"part"}:
        rejected.append("total: part_time_only")
    elif shifts <= {"full", "part"}:
        for component in ("full_time", "part_time"):
            parts = [quantity for quantity in total if quantity.shift == _COMPONENT_SHIFT[component]]
            if parts and component not in supported:
                supported[component], quotes[component] = parts, quotes["total"]
    else:
        return
    del supported["total"], quotes["total"]


def _basis(values: dict[str, int], is_fte: bool, includes_contractors: bool) -> str | None:
    """The proxy formula the supported components allow (one of `BASES`), or None when none does."""
    if "total" in values and includes_contractors:
        return "total_incl_contractors"
    if "total" in values and is_fte:
        return "fte"
    if "full_time" in values and ("part_time" in values or "total" in values):
        return "full_part"
    if "total" in values:
        return "total"
    return "full_time_only" if "full_time" in values else None


def decide_employee_answer(answer: EmployeeAnswer, source_text: str) -> EmployeeDecision:
    """Guard each claimed component against the text sent, then derive the stored total, basis and status.

    The total is the stated total, else the sum of stated full- and part-time counts; it is never a total
    less a component. A part-time count with neither a full-time count nor a total supports no basis and
    is dropped. Open bounds resolve by `_resolve_bound`. Every status is final: the filing is not re-asked.
    """
    supported: dict[str, list[_Quantity]] = {}
    quotes: dict[str, str] = {}
    rejected: list[str] = []
    for component in COMPONENTS:
        value, quote = cast("int | None", getattr(answer, component)), cast("str | None", getattr(answer, f"{component}_quote"))
        if value is None:
            continue
        support = _support(component, value, quote, source_text)
        if isinstance(support, str):
            rejected.append(f"{component}: {support}")
        else:
            supported[component], quotes[component] = support, str(quote)
    _settle_total(supported, quotes, rejected, answer.includes_contractors)
    if "part_time" in supported and not {"full_time", "total"} & supported.keys():
        del supported["part_time"], quotes["part_time"]
        rejected.append("part_time: no_full_time_or_total")
    values = {component: sum(map(_resolve_bound, quantities)) for component, quantities in supported.items()}
    if not values:
        status = "unsupported" if rejected else _NULL_STATUS.get(answer.status, "ambiguous")
        return EmployeeDecision(status, measurement_period=answer.measurement_period, rejected=tuple(rejected))
    is_fte = answer.is_fte or {quantity.shift for quantity in supported.get("total", [])} == {"fte"}
    full_part = values["full_time"] + values["part_time"] if {"full_time", "part_time"} <= values.keys() else None
    return EmployeeDecision(
        "found",
        employees_total=values.get("total", full_part),
        employees_full_time=values.get("full_time"),
        employees_part_time=values.get("part_time"),
        basis=_basis(values, is_fte, answer.includes_contractors),
        measurement_period=answer.measurement_period,
        quotes=tuple((component, quotes[component]) for component in COMPONENTS if component in quotes),
        rejected=tuple(rejected),
    )


def _filing_key(stamp: FilingStamp) -> tuple[pd.Timestamp, int, str]:
    """Filing date, then original before amendment, so a same-day 10-K/A supersedes its original."""
    return (stamp.filed.normalize(), int(stamp.is_amendment), str(stamp.accession_number))


def _task_stamp(result: LlmResult) -> FilingStamp:
    return cast(FilingStamp, result.task.meta["stamp"])


def employee_task(
    sequence: int,
    ticker: str,
    document: EmployeeText,
    max_chars: int,
    *,
    filed: pd.Timestamp,
    accession: str,
    report_date: pd.Timestamp | None,
    meta: dict[str, object] | None = None,
) -> LlmTask:
    """One LLM task for a filing's document text, within `max_chars`; needs no Filing or network.

    `meta` carries the caller's keys plus `report_date`, `source_text` (the excerpt sent, which the guard
    checks quotes against) and `source_document`.
    """
    if not document.text.strip():
        raise ValueError(f"{ticker} {accession}: filing text unavailable")
    prefix = (
        f"Ticker: {ticker}\nFiscal period end: {report_date.date() if report_date is not None else 'unknown'}\n"
        f"SEC filing date: {filed.date()}\nAccession: {accession}\nSource document: {document.source_document}\n"
        "The following is an excerpt, not necessarily the complete 10-K:\n\n"
    )
    if max_chars <= len(prefix):
        raise ValueError("gpt.max_chars.employees is too small for filing metadata")
    source_text = employee_excerpt(document.text, max_chars - len(prefix), document.spans)
    return LlmTask(
        seq=sequence,
        payload=prefix + source_text,
        schema=EmployeeAnswer,
        meta={**(meta or {}), "report_date": report_date, "source_text": source_text, "source_document": document.source_document},
    )


def _employee_task(sequence: int, ticker: str, stamp: FilingStamp, max_chars: int) -> LlmTask:
    """Read one owned filing's employee text and package it as one LLM task carrying its stamp.

    The filing's cached submission is then dropped: the stamp outlives the read, and an edgartools
    Filing otherwise keeps every document of its submission in memory.
    """
    report = stamp.period_of_report
    document = employee_text(stamp.filing)
    forget_sgml(stamp.filing)
    return employee_task(
        sequence,
        ticker,
        document,
        max_chars,
        filed=stamp.filed,
        accession=str(stamp.accession_number),
        report_date=pd.Timestamp(report).normalize() if report is not None else None,
        meta={"stamp": stamp},
    )


def _extract_answers(context: Context, config: DictConfig, ticker: str, stamps: list[FilingStamp]) -> list[LlmResult]:
    """One LLM result per filing, in filing order; a failed call is a result with its error, not a raise.

    Filings are read as the calls run, so reading the next filing overlaps the call on the previous one.
    """
    extractor = LLMExtractor(context, config, action="employees", threads=1)
    max_chars = int(config.gpt.max_chars.employees)
    results = extractor.run(_employee_task(sequence, ticker, stamp, max_chars) for sequence, stamp in enumerate(stamps))
    if len(results) != len(stamps):
        raise RuntimeError(f"{ticker}: {len(results)} employee LLM result(s) for {len(stamps)} filing(s)")
    return results


def _llm_decisions(context: Context, ticker: str, stamps: list[FilingStamp]) -> tuple[list[_Decision], list[FilingStamp]]:
    """`(decisions, failed)`: one guarded decision per answered filing, and the filings whose call failed.

    A failed call (invalid JSON included) is not a property of the filing, so it gets no row and
    is listed again next run; only a parsed answer is decided.
    """
    config = with_gpt_overrides(context.config, "employees", provider="open_ai_cheap")
    decisions: list[_Decision] = []
    failed: list[FilingStamp] = []
    for result in _extract_answers(context, config, ticker, stamps):
        stamp = _task_stamp(result)
        if not result.ok:
            context.log.warning(
                "fundamentals employees: %s %s LLM call failed (%s); its filing date is not stored and is listed again next run",
                ticker,
                stamp.accession_number,
                result.error,
            )
            failed.append(stamp)
            continue
        answer = result.parsed
        if not isinstance(answer, EmployeeAnswer):
            raise TypeError(f"{ticker}: unexpected employee LLM result {type(answer).__name__}")
        meta = result.task.meta
        decided = decide_employee_answer(answer, str(meta["source_text"]))
        decisions.append(_Decision(stamp, str(meta["source_document"]), decided))
    return decisions, failed


def source_quote(quotes: tuple[tuple[str, str], ...]) -> str | None:
    """The kept quotes as a JSON object keyed by component (`total`, `full_time`, `part_time`); None when none is kept."""
    return json.dumps(dict(quotes), ensure_ascii=False) if quotes else None


def _decide_ticker(context: Context, ticker: str, decisions: list[_Decision]) -> EmployeeTickerResult:
    """One row per filing date: the last decision with a component in filing order, else the last decision
    (NULL components, its status); an earlier counted decision on that date is superseded."""
    outcomes: list[dict] = []
    chosen: dict[pd.Timestamp, int] = {}  # filing date -> index of the outcome stored as its row
    counted: set[pd.Timestamp] = set()
    for decision in sorted(decisions, key=lambda decision: _filing_key(decision.stamp)):
        stamp, decided = decision.stamp, decision.decided
        filed = stamp.filed.normalize()
        if decided.counted:
            if filed in counted:
                outcomes[chosen[filed]]["status"] = "superseded"
            chosen[filed] = len(outcomes)
            counted.add(filed)
        elif filed not in counted:
            chosen[filed] = len(outcomes)
        outcomes.append(
            {
                "ticker": ticker,
                "accession_number": str(stamp.accession_number),
                "cik": stamp.cik,
                "form": stamp.form,
                "filing_date": filed,
                "source_document": decision.source_document,
                "status": decided.status,
                "employees_total": decided.employees_total,
                "employees_full_time": decided.employees_full_time,
                "employees_part_time": decided.employees_part_time,
                "basis": decided.basis,
                "measurement_period": decided.measurement_period,
                "quotes": decided.quotes,
            }
        )
        context.log.info(
            "employees decision ticker=%s accession=%s cik=%s as_of=%s source_document=%s status=%s total=%s full_time=%s part_time=%s basis=%s rejected=%s",
            ticker,
            stamp.accession_number,
            stamp.cik,
            filed.date(),
            decision.source_document,
            decided.status,
            decided.employees_total,
            decided.employees_full_time,
            decided.employees_part_time,
            decided.basis,
            "; ".join(decided.rejected) or "-",
        )
    rows = [_row(ticker, filed, outcomes[chosen[filed]]) for filed in sorted(chosen)]
    frame = pd.DataFrame(rows, columns=FRAME_COLUMNS).astype(dict.fromkeys(_COUNT_COLUMNS, "Int64"))
    return EmployeeTickerResult(frame, outcomes)


def _row(ticker: str, filed: pd.Timestamp, outcome: dict) -> dict:
    """The stored row for one filing date, from its chosen outcome."""
    return {
        "ticker": ticker,
        "as_of": filed,
        "cik": outcome["cik"],
        "accession_number": outcome["accession_number"],
        "form": outcome["form"],
        **{column: outcome[column] for column in _COUNT_COLUMNS},
        "basis": outcome["basis"],
        "status": outcome["status"],
        "source_document": outcome["source_document"],
        "source_quote": source_quote(outcome["quotes"]),
        "measurement_period": outcome["measurement_period"],
    }


def build_ticker_employees(
    context: Context,
    ticker: str,
    cik: str,
    *,
    since: pd.Timestamp | None,
    done_dates: frozenset[pd.Timestamp],
    scope: EdgarScope,
) -> EmployeeTickerResult:
    """Decide every annual filing whose date has no row yet, through the LLM and the source guard.

    A listed filing whose filer the identity layer does not tie to `ticker` is skipped and counted in `scope.guard`.
    A filing whose LLM call failed withholds its whole filing date, so a stored same-day filing
    cannot mark that date done; the ticker's other dates are kept.
    """
    identity = scope.identity
    if identity is None:
        raise ValueError(f"{ticker}: employee extraction needs an identity-aware EdgarScope")
    listed = scope.list_filings(ticker, cik, HEADCOUNT_FORMS, since=since, done_accessions=frozenset())
    owned = [filing for filing in listed if _owned(context, identity, ticker, filing)]
    scope.guard.add(len(listed) - len(owned))
    stamps = sorted(
        (stamp for stamp in (FilingStamp.of(filing, cik) for filing in owned) if stamp.filed.normalize() not in done_dates),
        key=_filing_key,
    )
    by_llm, failed = _llm_decisions(context, ticker, stamps) if stamps else ([], [])
    withheld = {stamp.filed.normalize() for stamp in failed}
    decisions = [decision for decision in by_llm if decision.stamp.filed.normalize() not in withheld]
    result = _decide_ticker(context, ticker, decisions)
    return EmployeeTickerResult(result.frame, result.outcomes, tuple(str(stamp.accession_number) for stamp in failed))


def _owned(context: Context, identity: Identity, ticker: str, filing: object) -> bool:
    """Whether the filing's own CIK belongs to `ticker`'s entity; a filing without a CIK is skipped (warned)."""
    actual_cik = getattr(filing, "cik", None)
    if actual_cik and identity.owns(ticker, actual_cik):
        return True
    context.log.warning("%s %s: filing CIK %r is outside the issuer lineage; skipped", ticker, getattr(filing, "accession_number", "?"), actual_cik)
    return False


def _done_dates(context: Context, tickers: list[str], since: pd.Timestamp) -> dict[str, frozenset[pd.Timestamp]]:
    """Filing dates that already have a row, counted or NULL with a status: the table alone says what is decided."""
    stored = context.store.load(Tables.fundamentals_employees, columns=["ticker", "as_of"], where={"ticker": tickers}, since=since, optional=True)
    done: dict[str, set[pd.Timestamp]] = {}
    for row in [] if stored is None else stored.itertuples(index=False):
        done.setdefault(str(row.ticker), set()).add(pd.Timestamp(cast("str", row.as_of)).normalize())
    return {ticker: frozenset(dates) for ticker, dates in done.items()}


def _fetch_ticker_employees(
    ticker: str,
    cik: str,
    *,
    context: Context,
    since: pd.Timestamp,
    done: dict[str, frozenset[pd.Timestamp]],
    scope: EdgarScope,
) -> EmployeeTickerResult:
    """Decide one ticker's undecided filings and save its rows; the per-ticker worker."""
    result = build_ticker_employees(
        context,
        ticker,
        cik,
        since=since,
        done_dates=done.get(ticker, frozenset()),
        scope=scope,
    )
    if not result.frame.empty:
        context.store.save(Tables.fundamentals_employees, result.frame)
    return result


def fetch_fundamentals_employees(
    context: Context,
    tickers: list[str],
    years_history: int,
    *,
    full: bool = False,
) -> None:
    """Decide every annual filing in the window whose date has no row; `--full` re-decides them all.

    Each run lists the whole `years_history` window and skips filing dates already in
    `fundamentals_employees`. A filing with no supported count is stored as a row with NULL components
    and its status, so it is decided once and never sent to the LLM again. A failed LLM call fails only its own filing
    date, which gets no row and is retried on the next run; the ticker's other dates are saved. A
    ticker that fails otherwise saves nothing. Both are named in the coverage log; the run never raises.
    """
    context.ensure_edgar_identity()
    configure(context)
    cik_map = load_cik_mapping(context, tickers)
    missing = set(tickers) - set(cik_map["ticker"])
    if missing:
        raise ValueError(f"Employee extraction has no roster CIK for {', '.join(sorted(missing))}")
    since = pd.Timestamp.today().normalize() - pd.DateOffset(years=years_history)
    scope = EdgarScope(load_identity(context))
    done = {} if full else _done_dates(context, tickers, since)
    context.log.info("fundamentals employees: %d decided filing date(s) skipped", sum(map(len, done.values())))
    worker = partial(_fetch_ticker_employees, context=context, since=since, done=done, scope=scope)
    results = run_per_ticker(
        cik_map,
        worker,
        desc="fundamentals employees",
        log=context.log,
        max_workers=int(context.config.data_extract.fundamentals_workers),
    )
    successful = [result for result in results if result is not None]
    failed_tickers = [str(ticker) for ticker, result in zip(cik_map["ticker"], results, strict=True) if result is None]
    failed = len(failed_tickers)
    outcomes = [outcome for result in successful for outcome in result.outcomes]
    rows = sum(len(result.frame) for result in successful)
    counted = sum(int(result.frame["status"].eq("found").sum()) for result in successful)
    context.log.info(
        "fundamentals employees: %d/%d ticker(s) read, %d failed; %d filing(s) decided (%d ambiguous or unsupported) "
        "-> %d count row(s), %d NULL row(s); guard skipped %d filing(s) outside a filing scope for 'fundamentals_employees'",
        len(successful),
        len(cik_map),
        failed,
        len(outcomes),
        sum(outcome["status"] in {"ambiguous", "unsupported"} for outcome in outcomes),
        counted,
        rows - counted,
        scope.guard.skipped,
    )
    failed_filings = [
        f"{ticker} {accession}"
        for ticker, result in zip(cik_map["ticker"], results, strict=True)
        if result is not None
        for accession in result.failed
    ]
    if failed_filings:
        more = f" (+{len(failed_filings) - _FAILED_SHOWN} more)" if len(failed_filings) > _FAILED_SHOWN else ""
        context.log.warning(
            "fundamentals employees: %d filing(s) not decided (LLM call failed), listed again next run: %s%s",
            len(failed_filings),
            ", ".join(failed_filings[:_FAILED_SHOWN]),
            more,
        )
    if failed_tickers:
        more = f" (+{failed - _FAILED_SHOWN} more)" if failed > _FAILED_SHOWN else ""
        context.log.warning(
            "fundamentals employees: %d/%d ticker(s) not read, listed again next run: %s%s",
            failed,
            len(cik_map),
            ", ".join(failed_tickers[:_FAILED_SHOWN]),
            more,
        )
