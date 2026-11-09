"""Source-backed SEC annual employee counts, dated by the filing's public date, read from `sec_employees_text`."""

from __future__ import annotations

import json
import queue
import re
from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from itertools import combinations
from typing import Any, Literal, cast

import pandas as pd
from pydantic import BaseModel, Field

from src.context import Context
from src.data_extract.utils.common.identity import load_identity
from src.data_extract.utils.common.resume import DONE_PER_KEY, document_floor
from src.data_extract.utils.common.workforce import MONTH_RE, merge_spans, workforce_number_spans
from src.data_extract.utils.edgar_extract.document_resume import WorklistRequest, document_worklist
from src.data_extract.utils.edgar_extract.edgar_driver import FilingStamp, universe_rows
from src.data_extract.utils.edgar_extract.parallel_fetch import run_per_ticker
from src.data_store.schema import Resume, Tables
from src.gpt_extract.transformers.gpt_getter import LLMExtractor
from src.gpt_extract.utils.schemas_gpt import LlmResult, LlmTask
from src.utils.string import run_date

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
# The `sec_employees_text` columns the worklist reads; `windows` and `pieces` are read per ticker, for its undecided filings.
_TEXT_COLUMNS = ["ticker", "accession_number", "cik", "form", "filing_date", "period_of_report", "source_document"]
_GAP = "[... filing gap ...]"
_FRAGMENT_GAP = 2_000  # max source chars between two `...`-joined quote fragments
_FAILED_SHOWN = 20  # failed tickers named in the coverage log

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
_SHIFT_AFTER_RE = re.compile(r"^\W*(?:(?:[a-z]+\s+){0,3}?|of\s+(?:the\s+)?[\w’'�]+\s+employees\s+(?:are|were)\s+)(full|part)[\s-]*time")
_ROW_LABEL_RE = re.compile(
    r"\b(Total|TOTAL|Full|FULL|Part|PART)(?:[\s-]*(?:time|Time|TIME))?\b"
    r"(?:(?!\b(?:Total|TOTAL|Full|FULL|Part|PART)\b)[^\d]){0,40}(?:[\d,.]+(?:\s*%)?(?:\s+|$))*$"
)  # a row's earlier cells may be percentages: "Full-time 30,497 76 % 9,716 24 % 40,213"
# Prose may label the number before it, in its own phrase: "the number of full-time employees ... was 38,100". Only a
# "number of" label counts: in "N1 part-time employees in the US and N2 employees outside", N2 has no shift.
_SHIFT_BEFORE_RE = re.compile(r"\bnumber\s+of\s+(full|part)[\s-]*time\s+(?:employees|associates|staff|workers|team\s+members|people|persons)\b")
_FULL_AND_PART_RE = re.compile(r"full[\s-]*(?:time\s*)?(?:and|or|&|/)\s*part", re.I)
_FTE_AFTER_RE = re.compile(r"^\W*(?:[a-z]+\s+){0,2}?full[\s-]*time[\s-]*equivalent", re.I)
# Non-employees folded into a number: prose after it is lower case (a capitalised word there is the next table
# row's label); a label or subject before it may be capitalised. "an additional" / "excluding" state them apart.
_CONTRACTOR_RE = re.compile(r"\b(?:contractors?|contingent|temporar(?:y|ies)|complementary)\b")
_CONTRACTOR_LABEL_RE = re.compile(_CONTRACTOR_RE.pattern, re.I)
_SEPARATELY_RE = re.compile(r"\b(?:exclud\w*|not\s+includ\w*|additional|in\s+addition|besides|apart\s+from|other\s+than)\b", re.I)
_PHRASE_STOP_RE = re.compile(r"\d|[.;]\s")  # a number's own phrase ends at the next number or sentence end

#: The provider the employee extraction calls (a cheaper model than `config.gpt.default_api`).
EMPLOYEES_PROVIDER = "open_ai_cheap"

COMPONENTS = ("total", "full_time", "part_time")
_COMPONENT_SHIFT = {"total": None, "full_time": "full", "part_time": "part"}  # the shift label a component's number must carry
# Decided statuses. `found`: at least one component is supported. With every component NULL: `not_disclosed`
# (no count stated), `image_only` (only in an image), `incorporated` (in a document the filing incorporates by
# reference but does not include), `ambiguous` (contradictory, a range, or no count claimed), `unsupported`
# (a component was claimed and the guard rejected it).
# The proxy formula (`basis`) a decided row supports: `total_incl_contractors` and `fte` qualify the total; `full_part`
# is full-time with part-time or with a stated total; `total` alone; `full_time_only`.
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
    if label := _SHIFT_AFTER_RE.match(after):
        if label.group(1) == "full" and re.match(r"[\s-]*equivalent\b", after[label.end() :]):
            return "fte"
        return label.group(1).casefold()
    if row:
        return row.group(1).casefold()
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
            is_date = (not scale and not fraction and "," not in digits and 1900 <= int(digits) <= 2100) or bool(MONTH_RE.search(before))
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


def _recover_employee_quote(component: str, value: int, source_text: str) -> tuple[str, list[_Quantity]] | None:
    """Recover a verbatim workforce-number span that passes the component's existing numeric guard."""
    for start, end in workforce_number_spans(source_text):
        quote = source_text[start:end]
        if re.search(r";|\.(?!\d)", quote) or _GAP in quote:
            continue
        support = _support(component, value, quote, source_text)
        if not isinstance(support, str):
            return quote, support
    return None


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
    """The proxy formula (`basis`) the supported components allow, or None when none does."""
    if "total" in values and includes_contractors:
        return "total_incl_contractors"
    if "total" in values and is_fte:
        return "fte"
    if "full_time" in values and ("part_time" in values or "total" in values):
        return "full_part"
    if "total" in values:
        return "total"
    return "full_time_only" if "full_time" in values else None


def decide_employee_answer(answer: EmployeeAnswer, source_text: str, *, previous: dict[str, int] | None = None) -> EmployeeDecision:
    """Guard each claimed component against the text sent, then derive the stored total, basis and status.

    The total is the stated total, else the sum of stated full- and part-time counts; it is never a total
    less a component. A part-time count with neither a full-time count nor a total supports no basis and
    is dropped. A missing or malformed quote can be recovered from workforce text when the component
    is within 10% of its earlier accepted count. Open bounds resolve by `_resolve_bound`.
    """
    supported: dict[str, list[_Quantity]] = {}
    quotes: dict[str, str] = {}
    rejected: list[str] = []
    for component in COMPONENTS:
        value, quote = cast("int | None", getattr(answer, component)), cast("str | None", getattr(answer, f"{component}_quote"))
        if value is None:
            continue
        support = _support(component, value, quote, source_text)
        prior = (previous or {}).get(component)
        if isinstance(support, str) and support in {"no_quote", "not_located"} and prior is not None and prior > 0:
            if abs(value - prior) * 10 <= prior and (recovered := _recover_employee_quote(component, value, source_text)) is not None:
                quote, support = recovered
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


def employee_task(sequence: int, ticker: str, row: Any, budget: int) -> LlmTask:
    """One LLM task for a stored `sec_employees_text` row: the filing prefix, then the stored candidate windows kept in
    priority order while the gap-joined excerpt fits the rest of `budget`, each sliced from the stored piece holding it.
    `meta` carries the row's `stamp`, `report_date`, `source_text` (the excerpt the guard checks quotes against) and
    `source_document`."""
    # clean: the filing stamp and the prefix from the stored filing columns
    form = str(row.form)
    filed = pd.Timestamp(row.filing_date)
    report_date = None if pd.isna(row.period_of_report) else pd.Timestamp(row.period_of_report).normalize()
    accession = str(row.accession_number)
    stamp = FilingStamp(accession, form, str(row.cik), filed, form.upper().endswith("/A"), None, None)
    prefix = (
        f"Ticker: {ticker}\nFiscal period end: {report_date.date() if report_date is not None else 'unknown'}\n"
        f"SEC filing date: {filed.date()}\nAccession: {accession}\nSource document: {row.source_document}\n"
        "The following is an excerpt, not necessarily the complete 10-K:\n\n"
    )
    if budget <= len(prefix):
        raise ValueError("data_extract.llm_text_budget.employees is too small for filing metadata")

    # compute: the greedy choice of the stored windows within the excerpt limit
    windows, pieces, limit = json.loads(row.windows), json.loads(row.pieces), budget - len(prefix)
    separator = f"\n\n{_GAP}\n\n"
    chosen: list[tuple[int, int]] = []
    for start, end in windows:
        trial = merge_spans([*chosen, (start, end)])
        if sum(stop - begin for begin, stop in trial) + len(separator) * (len(trial) - 1) <= limit:
            chosen = trial

    # post-process: the text of each chosen span, from the last piece starting at or before it
    offsets = [int(offset) for offset, _ in pieces]
    texts = []
    for start, end in chosen:
        offset, text = pieces[bisect_right(offsets, start) - 1]
        texts.append(str(text)[start - int(offset) : end - int(offset)])
    source_text = separator.join(texts)
    return LlmTask(
        seq=sequence,
        payload=prefix + source_text,
        schema=EmployeeAnswer,
        meta={"stamp": stamp, "report_date": report_date, "source_text": source_text, "source_document": str(row.source_document)},
    )


def _llm_decisions(context: Context, ticker: str, results: list[LlmResult], history: pd.DataFrame) -> tuple[list[_Decision], list[FilingStamp]]:
    """`(decisions, failed)`: one guarded decision per answered filing, and the filings whose call failed (no row, listed
    again next run). `history` holds the ticker's stored counts (`as_of` and the count columns), extended with each new
    decision; quote recovery reads its earlier dates."""
    decisions: list[_Decision] = []
    failed: list[FilingStamp] = []
    for result in results:
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
        prior = history.loc[history["as_of"].lt(stamp.filed.normalize())].sort_values("as_of", kind="stable")
        previous: dict[str, int] = {}
        for component, column in zip(COMPONENTS, _COUNT_COLUMNS, strict=True):
            values = prior[column].dropna()
            if not values.empty:
                previous[component] = int(values.iloc[-1])
        decided = decide_employee_answer(answer, str(meta["source_text"]), previous=previous)
        decisions.append(_Decision(stamp, str(meta["source_document"]), decided))
        history.loc[len(history)] = [stamp.filed.normalize(), *(getattr(decided, column) for column in _COUNT_COLUMNS)]
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


def _undecided(df_text: pd.DataFrame | None, df_done: pd.DataFrame | None, pending: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, int]:
    """The text rows to decide: a filing date with no stored row (`df_done` None: every date) that precedes the ticker's
    earliest filing the text walk has not stored yet; plus the number of undecided (ticker, date) pairs held back."""
    # clean: the filing day and the amendment flag of each text row
    df = (pd.DataFrame(columns=_TEXT_COLUMNS) if df_text is None else df_text).copy()
    df["filed"] = pd.to_datetime(df["filing_date"]).dt.normalize()
    df["is_amendment"] = df["form"].astype(str).str.upper().str.endswith("/A")
    keys = pd.MultiIndex.from_arrays([df["ticker"].astype(str), df["filed"]])

    # compute: drop the decided dates, hold back every date from the ticker's first pending filing on
    if df_done is None or df_done.empty:
        decided = pd.Series(False, index=df.index)
    else:
        done = pd.MultiIndex.from_arrays([df_done["ticker"].astype(str), pd.to_datetime(df_done["as_of"]).dt.normalize()])
        decided = pd.Series(keys.isin(done), index=df.index)
    first_pending = {ticker: pd.Timestamp(df_units["filed"].min()).normalize() for ticker, df_units in pending.items() if not df_units.empty}
    held = df["filed"].ge(pd.to_datetime(df["ticker"].map(first_pending)))
    n_held = keys[(~decided & held).to_numpy()].nunique()
    return df[~decided & ~held].reset_index(drop=True), int(n_held)


def _ticker_employees(
    ticker: str, *, context: Context, extractors: queue.SimpleQueue[LLMExtractor], df_todo: pd.DataFrame, df_history: pd.DataFrame
) -> EmployeeTickerResult:
    """Decide one ticker's undecided filings through the LLM (on an extractor borrowed from `extractors`) and the source
    guard, and save its rows; the per-ticker worker. A filing whose LLM call failed withholds its whole filing date, so
    a stored same-day filing cannot mark that date done; the ticker's other dates are kept."""
    # load: the stored windows and pieces of the ticker's undecided filings
    df_mine = df_todo[df_todo["ticker"].eq(ticker)]
    df_text = context.store.load(
        Tables.sec_employees_text,
        columns=["accession_number", "windows", "pieces"],
        where={"ticker": ticker, "accession_number": list(df_mine["accession_number"])},
    )
    budget = int(context.config.data_extract.llm_text_budget.employees)

    # merge: one row per filing in filing order (date, original before amendment, accession)
    df_rows = df_mine.merge(df_text, on="accession_number", how="inner").sort_values(["filed", "is_amendment", "accession_number"], kind="stable")
    history = df_history.loc[df_history["ticker"].astype(str).eq(ticker), ["as_of", *_COUNT_COLUMNS]].reset_index(drop=True)
    history["as_of"] = pd.to_datetime(history["as_of"])

    # compute: the answers, then the guarded decisions; a failed call withholds its date
    tasks = [employee_task(sequence, ticker, row, budget) for sequence, row in enumerate(df_rows.itertuples(index=False))]
    extractor = extractors.get()
    try:
        results = extractor.run(tasks)
    finally:
        extractors.put(extractor)
    if len(results) != len(tasks):
        raise RuntimeError(f"{ticker}: {len(results)} employee LLM result(s) for {len(tasks)} filing(s)")
    by_llm, failed = _llm_decisions(context, ticker, results, history)
    withheld = {stamp.filed.normalize() for stamp in failed}
    result = _decide_ticker(context, ticker, [decision for decision in by_llm if decision.stamp.filed.normalize() not in withheld])

    # save: the decided dates
    if not result.frame.empty:
        context.store.save(Tables.fundamentals_employees, result.frame)
    return EmployeeTickerResult(result.frame, result.outcomes, tuple(str(stamp.accession_number) for stamp in failed))


def _log_run(context: Context, df_keys: pd.DataFrame, results: list[EmployeeTickerResult | None], n_held: int) -> None:
    """The run's coverage: decided filings and rows, held-back dates, failed filings and failed tickers."""
    # compute: the run's totals
    tickers = df_keys["ticker"].astype(str).tolist()
    successful = [result for result in results if result is not None]
    outcomes = [outcome for result in successful for outcome in result.outcomes]
    rows = sum(len(result.frame) for result in successful)
    counted = sum(int(result.frame["status"].eq("found").sum()) for result in successful)
    failed_filings = [
        f"{ticker} {accession}" for ticker, result in zip(tickers, results, strict=True) if result is not None for accession in result.failed
    ]
    failed_tickers = [ticker for ticker, result in zip(tickers, results, strict=True) if result is None]

    # log: the summary, then each list of what is listed again next run
    context.log.info(
        "fundamentals employees: %d/%d ticker(s) decided, %d failed; %d filing(s) decided (%d ambiguous or unsupported) "
        "-> %d count row(s), %d NULL row(s); %d date(s) held back behind a filing sec_employees_text has not stored yet",
        len(successful),
        len(tickers),
        len(failed_tickers),
        len(outcomes),
        sum(outcome["status"] in {"ambiguous", "unsupported"} for outcome in outcomes),
        counted,
        rows - counted,
        n_held,
    )
    for what, names in (("filing(s) not decided (LLM call failed)", failed_filings), ("ticker(s) not decided", failed_tickers)):
        if names:
            more = f" (+{len(names) - _FAILED_SHOWN} more)" if len(names) > _FAILED_SHOWN else ""
            context.log.warning(
                "fundamentals employees: %d %s, listed again next run: %s%s", len(names), what, ", ".join(names[:_FAILED_SHOWN]), more
            )


def fetch_employees_llm(context: Context, tickers: list[str], years_history: int, extractors: Sequence[LLMExtractor], *, full: bool = False) -> None:
    """Decide every stored employees text in the window whose filing date has no row (a NULL row with its status when no
    count is supported); `full` re-decides them all. Dates from a ticker's earliest filing the walk has not stored yet wait.
    Each worker reads its own ticker's `windows`/`pieces`, which bounds memory. `len(extractors)` tickers run at once, one
    extractor each; a failed call or ticker is logged and listed again next run."""
    # load: the window's text rows (no windows or pieces), the stored rows, the text walk's pending filings
    start_date = run_date(None)
    since = document_floor(Tables.sec_employees_text, start_date, years_history)
    cik_map = universe_rows(context, tickers)
    df_text = context.store.load(Tables.sec_employees_text, columns=_TEXT_COLUMNS, where={"ticker": tickers}, since=since, optional=True)
    stored = context.store.load(Tables.fundamentals_employees, columns=["ticker", "as_of", *_COUNT_COLUMNS], where={"ticker": tickers}, optional=True)
    forms = cast(Resume, Tables.sec_employees_text.resume).forms
    request = WorklistRequest(
        keys=cik_map[["ticker", "cik"]],
        as_of=start_date,
        forms=forms,
        identity=load_identity(context),
        years_history=years_history,
        done_scope=DONE_PER_KEY,
    )
    pending = document_worklist(context, Tables.sec_employees_text, request).units

    # compute: the undecided dates, then each ticker's decisions (saved per ticker)
    df_history = pd.DataFrame(columns=["ticker", "as_of", *_COUNT_COLUMNS]) if stored is None else stored
    df_todo, n_held = _undecided(df_text, None if full else df_history, pending)
    df_keys = cik_map[cik_map["ticker"].isin(df_todo["ticker"])].reset_index(drop=True)
    lent: queue.SimpleQueue[LLMExtractor] = queue.SimpleQueue()
    for extractor in extractors:
        lent.put(extractor)
    worker = partial(_ticker_employees, context=context, extractors=lent, df_todo=df_todo, df_history=df_history)
    results = run_per_ticker(df_keys, worker, desc="fundamentals employees", max_workers=len(extractors), key_cols=("ticker",))

    # log: the run's coverage
    _log_run(context, df_keys, results, n_held)
