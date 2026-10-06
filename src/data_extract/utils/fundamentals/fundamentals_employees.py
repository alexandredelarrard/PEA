"""Source-backed SEC annual employee counts, dated by the filing's public date."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from itertools import combinations
from pathlib import Path
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
from src.data_extract.utils.common.sec_io import configure, sec_call
from src.data_extract.utils.common.sec_utils import load_cik_mapping
from src.data_store.schema import Tables
from src.gpt_extract.transformers.gpt_getter import LLMExtractor
from src.gpt_extract.transformers.step_gpt_extracter import with_gpt_overrides
from src.gpt_extract.utils.schemas_gpt import LlmResult, LlmTask

HEADCOUNT_FORMS = ("10-K", "10-K/A", "10-K405")
FRAME_COLUMNS = ["ticker", "as_of", "employees"]
MANUAL_ROSTER = Path("sec") / "employees_manual_roster.json"
_GAP = "[... filing gap ...]"
_FRAGMENT_GAP = 2_000  # max source chars between two `...`-joined quote fragments
_FAILED_SHOWN = 20  # failed tickers named in the coverage log

_CONTEXT_RE = re.compile(
    r"\b(?:employees?|workforce|associates?|team\s+members?|human\s+capital|"
    r"personnel|staff|colleagues?|full[- ]time|part[- ]time)\b",
    re.I,
)
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
    r"(?:(?!\b(?:Total|TOTAL|Full|FULL|Part|PART)\b)[^\d]){0,40}(?:[\d,.]+(?:\s+|$))*$"
)
_FULL_AND_PART_RE = re.compile(r"full[\s-]*(?:time\s*)?(?:and|or|&|/)\s*part", re.I)
_MONTH_RE = re.compile(r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s*$", re.I)


class EmployeeAnswer(BaseModel):
    status: Literal["found", "not_disclosed", "image_only", "ambiguous"]
    count: int | None = Field(description="Issuer-wide current-period employees or FTE, or null")
    quote: str | None = Field(description="Short exact contiguous source quote, or null")
    measurement_period: str | None = Field(description="Headcount date or period as stated, at original precision")
    qualifier: str | None = Field(description="For example approximate or FTE, if stated")
    reason: str


@dataclass(frozen=True)
class EmployeeTickerResult:
    """One row per decided filing date (`employees` NaN when no count is supported) and the per-filing decisions."""

    frame: pd.DataFrame
    outcomes: list[dict]


@dataclass(frozen=True)
class _Decision:
    stamp: FilingStamp
    source: str  # "manual" (the roster file) or "llm"
    status: str
    count: int | None


def load_manual_roster(config_dir: str) -> dict[str, dict]:
    """Hand-kept employee decisions from `configs/sec/employees_manual_roster.json`, by accession.

    An entry's `employees` is stored as-is (null for a filing that states no usable count) and
    replaces the LLM for that filing; the pipeline only reads this file.
    """
    path = Path(config_dir) / MANUAL_ROSTER
    if not path.exists():
        return {}
    roster = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(entry["accession_number"]): {**entry, "ticker": ticker}
        for ticker, entries in roster.items()
        if not ticker.startswith("_")
        for entry in entries
    }


def filing_body_text(filing: Filing) -> str:
    """Primary-document text (visible HTML table cells included, no OCR), else the full submission; "" if none.

    edgartools raises AttributeError for a filing with no primary document; that is absorbed as a filing property.
    Each read runs under the `sec_io` retry policy; a transient failure raises.
    """
    readers: tuple[tuple[str, Callable[[str], str]], ...] = (
        ("html", html_to_text),
        ("text", str),
        ("full_text_submission", html_to_text),
    )
    for method, to_text in readers:
        try:
            raw = sec_call(getattr(filing, method), label=f"{getattr(filing, 'accession_number', '?')} {method}")
        except AttributeError:  # edgartools dereferences a missing primary document
            continue
        if raw and (text := to_text(raw)).strip():
            return text
    return ""


def employee_excerpt(text: str, limit: int) -> str:
    """The filing's opening text plus windows around the first workforce mentions, gap-marked and capped at `limit`."""
    spans = [(0, min(8_000, len(text)))]
    for match in _CONTEXT_RE.finditer(text):
        spans.append((max(0, match.start() - 550), min(len(text), match.end() + 850)))
        if len(spans) >= 121:
            break
    merged: list[list[int]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return f"\n\n{_GAP}\n\n".join(text[start:end] for start, end in merged)[:limit]


@dataclass(frozen=True)
class _Quantity:
    """One number the source states inside a located quote."""

    value: int
    bound: int  # +1 for "over N", -1 for "nearly N", 0 for an exact or approximate N
    shift: str | None  # "full" or "part" when the source labels the number full- or part-time
    in_range: bool
    is_date: bool


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
    if (row and row.group(1).casefold() == "total") or _FULL_AND_PART_RE.search(after[:40]):
        return None
    if label := _SHIFT_AFTER_RE.match(after) or row:
        return label.group(1).casefold()
    return None


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


def supported_employee_count(answer: EmployeeAnswer, source_text: str) -> int | None:
    """The source-backed headcount for the model's claim, or None when the source does not support it.

    The quote must be in the source and the model's count must be a number it states, or the sum of
    stated components. Only full-time is kept: part-time components are dropped, and a total the
    quoted passage splits into full- and part-time rows becomes its full-time row. Open bounds resolve
    by `_resolve_bound`, and a stated range such as `50,000 to 100,000` is unclear, so None.
    """
    if answer.count is None or answer.count <= 0 or not answer.quote or _GAP in answer.quote:
        return None
    spans = _locate(answer.quote, source_text)
    claimed = _claimed(_quantities(source_text, spans), answer.count) if spans else None
    if not claimed or any(quantity.in_range for quantity in claimed):
        return None
    if len(claimed) == 1 and claimed[0].shift is None and spans:
        passage = _quantities(source_text, [(spans[0][0], spans[-1][1])])
        part_values = {q.value for q in passage if q.shift == "part"}
        if split := next((q for q in passage if q.shift != "part" and claimed[0].value - q.value in part_values), None):
            claimed = [split]
    full_time = [quantity for quantity in claimed if quantity.shift != "part"]
    return sum(map(_resolve_bound, full_time)) if full_time else None


def _decide(answer: EmployeeAnswer, source_text: str) -> tuple[str, int | None]:
    """Every status is final: an `ambiguous` filing is not sent to the LLM again."""
    count = supported_employee_count(answer, source_text)
    if count is not None:
        return "saved", count
    if answer.status in {"not_disclosed", "image_only"} and answer.count is None:
        return "no_headcount", None
    return "ambiguous", None


def _filing_key(stamp: FilingStamp) -> tuple[pd.Timestamp, int, str]:
    """Filing date, then original before amendment, so a same-day 10-K/A supersedes its original."""
    return (stamp.filed.normalize(), int(stamp.is_amendment), str(stamp.accession_number))


def _task_stamp(result: LlmResult) -> FilingStamp:
    return cast(FilingStamp, result.task.meta["stamp"])


def _employee_task(sequence: int, ticker: str, stamp: FilingStamp, max_chars: int) -> LlmTask:
    """Package one owned filing's excerpt as one LLM task."""
    filing = stamp.filing
    report = stamp.period_of_report
    report_date = pd.Timestamp(report).normalize() if report is not None else None
    text = filing_body_text(filing)
    if not text.strip():
        raise ValueError(f"{ticker} {stamp.accession_number}: filing text unavailable")
    prefix = (
        f"Ticker: {ticker}\nFiscal period end: {report_date.date() if report_date is not None else 'unknown'}\n"
        f"SEC filing date: {stamp.filed.date()}\nAccession: {stamp.accession_number}\n"
        "The following is an excerpt, not necessarily the complete 10-K:\n\n"
    )
    if max_chars <= len(prefix):
        raise ValueError("gpt.max_chars.employees is too small for filing metadata")
    source_text = employee_excerpt(text, max_chars - len(prefix))
    return LlmTask(
        seq=sequence,
        payload=prefix + source_text,
        schema=EmployeeAnswer,
        meta={"stamp": stamp, "report_date": report_date, "source_text": source_text},
    )


def _extract_answers(context: Context, config: DictConfig, ticker: str, stamps: list[FilingStamp]) -> list[LlmResult]:
    """One LLM answer per filing, in filing order; any failed call fails the ticker."""
    extractor = LLMExtractor(context, config, action="employees", threads=1)
    max_chars = int(config.gpt.max_chars.employees)
    for sequence, stamp in enumerate(stamps):
        extractor.submit(_employee_task(sequence, ticker, stamp, max_chars))
    results = extractor.run()
    if len(results) != len(stamps) or any(not result.ok for result in results):
        errors = [f"{_task_stamp(result).accession_number}: {result.error}" for result in results if not result.ok]
        raise RuntimeError(f"{ticker}: employee LLM extraction incomplete: {errors}")
    return results


def _llm_decisions(context: Context, ticker: str, stamps: list[FilingStamp]) -> list[_Decision]:
    """One guarded LLM decision per filing; any failed call fails the ticker."""
    config = with_gpt_overrides(context.config, "employees", provider="open_ai_cheap")
    decisions = []
    for result in _extract_answers(context, config, ticker, stamps):
        answer = result.parsed
        if not isinstance(answer, EmployeeAnswer):
            raise TypeError(f"{ticker}: unexpected employee LLM result {type(answer).__name__}")
        status, count = _decide(answer, str(result.task.meta["source_text"]))
        decisions.append(_Decision(_task_stamp(result), "llm", status, count))
    return decisions


def _manual_decision(stamp: FilingStamp, entry: dict) -> _Decision:
    count = entry.get("employees")
    status = str(entry.get("status") or ("saved" if count is not None else "no_headcount"))
    return _Decision(stamp, "manual", status, None if count is None else int(count))


def _decide_ticker(context: Context, ticker: str, decisions: list[_Decision]) -> EmployeeTickerResult:
    """One row per filing date: the last supported count in filing order, else NaN; earlier counts are superseded."""
    outcomes: list[dict] = []
    chosen: dict[pd.Timestamp, int] = {}  # filing date -> index of the outcome whose count is kept
    for decision in sorted(decisions, key=lambda decision: _filing_key(decision.stamp)):
        stamp = decision.stamp
        filed = stamp.filed.normalize()
        if decision.count is not None:
            if filed in chosen:
                outcomes[chosen[filed]]["status"] = "superseded"
            chosen[filed] = len(outcomes)
        outcomes.append(
            {
                "ticker": ticker,
                "accession_number": str(stamp.accession_number),
                "filing_date": filed,
                "source": decision.source,
                "status": decision.status,
                "count": decision.count,
            }
        )
        context.log.info(
            "employees decision ticker=%s accession=%s cik=%s as_of=%s source=%s status=%s count=%s",
            ticker,
            stamp.accession_number,
            stamp.cik,
            filed.date(),
            decision.source,
            decision.status,
            decision.count,
        )
    rows = [
        {"ticker": ticker, "as_of": filed, "employees": float(outcomes[chosen[filed]]["count"]) if filed in chosen else float("nan")}
        for filed in sorted({outcome["filing_date"] for outcome in outcomes})
    ]
    return EmployeeTickerResult(pd.DataFrame(rows, columns=FRAME_COLUMNS), outcomes)


def build_ticker_employees(
    context: Context,
    ticker: str,
    cik: str,
    *,
    since: pd.Timestamp | None,
    done_dates: frozenset[pd.Timestamp],
    manual: dict[str, dict],
    scope: EdgarScope,
) -> EmployeeTickerResult:
    """Decide every annual filing whose date has no row yet: from the manual roster when listed, else the LLM.

    A listed filing whose filer the identity layer does not tie to `ticker` is skipped and counted in `scope.guard`.
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
    by_hand = [_manual_decision(stamp, manual[str(stamp.accession_number)]) for stamp in stamps if str(stamp.accession_number) in manual]
    to_read = [stamp for stamp in stamps if str(stamp.accession_number) not in manual]
    decisions = by_hand + (_llm_decisions(context, ticker, to_read) if to_read else [])
    return _decide_ticker(context, ticker, decisions)


def _owned(context: Context, identity: Identity, ticker: str, filing: object) -> bool:
    """Whether the filing's own CIK belongs to `ticker`'s entity; a filing without a CIK is skipped (warned)."""
    actual_cik = getattr(filing, "cik", None)
    if actual_cik and identity.owns(ticker, actual_cik):
        return True
    context.log.warning("%s %s: filing CIK %r is outside the issuer lineage; skipped", ticker, getattr(filing, "accession_number", "?"), actual_cik)
    return False


def _done_dates(context: Context, tickers: list[str], since: pd.Timestamp) -> dict[str, frozenset[pd.Timestamp]]:
    """Filing dates that already have a row, count or NULL: the table alone says what is decided."""
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
    manual: dict[str, dict],
    scope: EdgarScope,
) -> EmployeeTickerResult:
    """Decide one ticker's undecided filings and save its rows; the per-ticker worker."""
    result = build_ticker_employees(
        context,
        ticker,
        cik,
        since=since,
        done_dates=done.get(ticker, frozenset()),
        manual=manual,
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
    `fundamentals_employees`. A filing with no supported count is stored as a NULL row, so it is
    decided once and never sent to the LLM again. A filing listed in the manual roster takes its
    value from there instead of the LLM, on `--full` too. A ticker that fails saves nothing, is named
    in the coverage log and is retried on the next run; the run never raises for it.
    """
    context.ensure_edgar_identity()
    configure(context)
    cik_map = load_cik_mapping(context, tickers)
    missing = set(tickers) - set(cik_map["ticker"])
    if missing:
        raise ValueError(f"Employee extraction has no roster CIK for {', '.join(sorted(missing))}")
    since = pd.Timestamp.today().normalize() - pd.DateOffset(years=years_history)
    scope = EdgarScope(load_identity(context))
    manual = load_manual_roster(str(context.config_dir))
    done = {} if full else _done_dates(context, tickers, since)
    context.log.info(
        "fundamentals employees: %d decided filing date(s) skipped, %d manual roster entries",
        sum(map(len, done.values())),
        len(manual),
    )
    worker = partial(_fetch_ticker_employees, context=context, since=since, done=done, manual=manual, scope=scope)
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
    counted = sum(int(result.frame["employees"].notna().sum()) for result in successful)
    context.log.info(
        "fundamentals employees: %d/%d ticker(s) read, %d failed; %d filing(s) decided (%d from the manual roster, %d ambiguous) "
        "-> %d count row(s), %d NULL row(s); guard skipped %d filing(s) outside a filing scope for 'fundamentals_employees'",
        len(successful),
        len(cik_map),
        failed,
        len(outcomes),
        sum(outcome["source"] == "manual" for outcome in outcomes),
        sum(outcome["status"] == "ambiguous" for outcome in outcomes),
        counted,
        rows - counted,
        scope.guard.skipped,
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
