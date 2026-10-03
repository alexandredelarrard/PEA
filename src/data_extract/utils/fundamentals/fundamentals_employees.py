"""Source-backed SEC annual employee counts, dated by the filing's public date."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Literal, cast

import pandas as pd
from edgar import Filing
from omegaconf import DictConfig
from pydantic import BaseModel, Field

from src.context import Context
from src.data_extract.utils.common.edgar_driver import EdgarScope, FilingStamp, IncompleteEdgarRunError, load_edgar_scope
from src.data_extract.utils.common.edgar_extract import html_to_text
from src.data_extract.utils.common.identity import Identity
from src.data_extract.utils.common.parallel_fetch import run_per_ticker
from src.data_extract.utils.common.registrant import resolve_registrant_filings
from src.data_extract.utils.common.run_manifest import get_entry, manifest_window, record_filing_outcomes, record_run
from src.data_extract.utils.common.sec_utils import load_cik_mapping
from src.data_store.schema import Tables
from src.gpt_extract.transformers.gpt_getter import LLMExtractor
from src.gpt_extract.transformers.step_gpt_extracter import with_gpt_overrides
from src.gpt_extract.utils.schemas_gpt import LlmResult, LlmTask

HEADCOUNT_FORMS = ("10-K", "10-K/A", "10-K405")
TERMINAL_STATUSES = frozenset({"saved", "no_headcount", "superseded"})
FRAME_COLUMNS = ["ticker", "as_of", "employees"]
_GAP = "[... filing gap ...]"

# Locate filing context and check LLM evidence; none of these extracts a fallback count.
_CONTEXT_RE = re.compile(
    r"\b(?:employees?|workforce|associates?|team\s+members?|human\s+capital|"
    r"personnel|staff|colleagues?|full[- ]time|part[- ]time)\b",
    re.I,
)
_BOUND_RE = re.compile(r"\b(?:over|nearly|more than|less than|at least|at most|up to|fewer than|greater than)\s+[\d,]+", re.I)
_SPLIT_RE = re.compile(r"([\d,]+)\s+(full|part)[- ]time employees\s+and\s+([\d,]+)\s+(full|part)[- ]time employees", re.I)
_NUMBER_RE = re.compile(r"\b\d[\d,]*\b")


class EmployeeAnswer(BaseModel):
    status: Literal["found", "not_disclosed", "image_only", "ambiguous"]
    count: int | None = Field(description="Issuer-wide current-period employees or FTE, or null")
    quote: str | None = Field(description="Short exact contiguous source quote, or null")
    measurement_period: str | None = Field(description="Headcount date or period as stated, at original precision")
    qualifier: str | None = Field(description="For example approximate or FTE, if stated")
    reason: str


@dataclass(frozen=True)
class EmployeeTickerResult:
    frame: pd.DataFrame
    outcomes: list[dict]
    unavailable_dates: frozenset[pd.Timestamp]


@dataclass(frozen=True)
class _ResumePlan:
    """Listing window and existing accession/date decisions for one run."""

    since: pd.Timestamp
    is_full_rescan: bool
    done_accessions: frozenset[str]
    saved_dates: dict[str, frozenset[pd.Timestamp]]
    skip_dates: dict[str, frozenset[pd.Timestamp]]


def filing_body_text(filing: Filing) -> str:
    """Primary-document text (visible HTML table cells included, no OCR), else the full submission; "" if none.

    edgartools raises AttributeError for a filing with no primary document; that is absorbed as a filing property.
    """
    readers: tuple[tuple[str, Callable[[str], str]], ...] = (
        ("html", html_to_text),
        ("text", str),
        ("full_text_submission", html_to_text),
    )
    for method, to_text in readers:
        try:
            raw = getattr(filing, method)()
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


def _normalise(text: str) -> str:
    return re.sub(r"\s+([,;:])", r"\1", " ".join(text.split()).casefold())


def _anchored_table_spans(quote: str, source: str) -> list[tuple[int, int]]:
    """Spans for a `heading ... total` quote whose heading and total sit close together in the source."""
    if quote.count(" ... ") != 1:
        return []
    heading, total = quote.split(" ... ")
    if len(heading) < 40 or "employ" not in heading or not total.startswith("total ") or len(total) < 20:
        return []
    spans = []
    for start in re.finditer(re.escape(heading), source):
        end = source.find(total, start.end())
        gap = source[start.end() : end] if end >= 0 else ""
        if (
            0 <= end - start.end() <= 2_000
            and _GAP not in gap
            and not re.search(r"\bas of\b", gap)
            and len(re.findall(r"\bnumber of employees\b", gap)) <= 1
        ):
            spans.append((start.start(), end + len(total)))
    return spans


def supported_employee_count(answer: EmployeeAnswer, source_text: str) -> int | None:
    """Accept a source-backed claim; allow table ellipses only between real anchors."""
    if answer.status != "found" or answer.count is None or answer.count <= 0 or not answer.quote:
        return None
    quote, source = _normalise(answer.quote), _normalise(source_text)
    if _GAP in quote or _BOUND_RE.search(quote):
        return None
    spans = [(match.start(), match.end()) for match in re.finditer(re.escape(quote), source)] or _anchored_table_spans(quote, source)
    # A model can trim "over" off an otherwise literal source quote.
    if not spans or any(_BOUND_RE.search(source[max(0, start - 32) : end]) for start, end in spans):
        return None
    split = _SPLIT_RE.search(quote)
    if split and split.group(2).casefold() != split.group(4).casefold():
        first = int(split.group(1).replace(",", ""))
        second = int(split.group(3).replace(",", ""))
        return first + second if answer.count in {first, second, first + second} else None
    numbers = {int(token.replace(",", "")) for token in _NUMBER_RE.findall(quote)}
    return answer.count if answer.count in numbers else None


def _decide(answer: EmployeeAnswer, source_text: str) -> tuple[str, int | None]:
    count = supported_employee_count(answer, source_text)
    if count is not None:
        return "saved", count
    if answer.status in {"not_disclosed", "image_only"} and answer.count is None:
        return "no_headcount", None
    return "ambiguous", None


def _filing_key(stamp: FilingStamp) -> tuple[pd.Timestamp, int, str]:
    """Filing date, then original before amendment, so a same-day 10-K/A supersedes its original."""
    return (stamp.filed.normalize(), int(stamp.is_amendment), str(stamp.accession_number))


def _employee_task(sequence: int, ticker: str, stamp: FilingStamp, identity: Identity, max_chars: int) -> LlmTask:
    """Check the filer belongs to the issuer lineage and package its excerpt as one LLM task."""
    filing = stamp.filing
    actual_cik = getattr(filing, "cik", None)
    if not actual_cik or not identity.owns(ticker, actual_cik):
        raise ValueError(f"{ticker} {stamp.accession_number}: filing CIK {actual_cik!r} is outside the issuer lineage")
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


def _extract_answers(context: Context, config: DictConfig, ticker: str, stamps: list[FilingStamp], identity: Identity) -> list[LlmResult]:
    """One LLM answer per filing, in filing order; any failed call fails the ticker."""
    extractor = LLMExtractor(context, config, action="employees", threads=1)
    max_chars = int(config.gpt.max_chars.employees)
    for sequence, stamp in enumerate(stamps):
        extractor.submit(_employee_task(sequence, ticker, stamp, identity, max_chars))
    results = extractor.run()
    if len(results) != len(stamps) or any(not result.ok for result in results):
        errors = [f"{cast(FilingStamp, result.task.meta['stamp']).accession_number}: {result.error}" for result in results if not result.ok]
        raise RuntimeError(f"{ticker}: employee LLM extraction incomplete: {errors}")
    return results


def _outcome(ticker: str, result: LlmResult, answer: EmployeeAnswer, status: str, model_name: str) -> dict:
    """One filing's decision record; `cik` is the filer's own CIK from the task's stamp."""
    stamp = cast(FilingStamp, result.task.meta["stamp"])
    report_date = cast("pd.Timestamp | None", result.task.meta["report_date"])
    return {
        "ticker": ticker,
        "accession_number": str(stamp.accession_number),
        "cik": stamp.cik,
        "form": str(stamp.form),
        "filing_date": stamp.filed.strftime("%Y-%m-%d"),
        "report_date": report_date.strftime("%Y-%m-%d") if report_date is not None else None,
        "measurement_period": answer.measurement_period,
        "qualifier": answer.qualifier,
        "source_quote": answer.quote,
        "model": model_name,
        "source_status": answer.status,
        "reason": answer.reason,
        "ordering": result.seq,
        "status": status,
    }


def _decide_ticker(context: Context, ticker: str, results: list[LlmResult], model_name: str) -> EmployeeTickerResult:
    """Guard every answer; keep the last supported count per filing date and mark earlier ones superseded."""
    outcomes: list[dict] = []
    chosen: dict[pd.Timestamp, tuple[int, int]] = {}  # filing date -> (count, outcome index)
    all_dates: set[pd.Timestamp] = set()
    for result in results:
        answer = result.parsed
        if not isinstance(answer, EmployeeAnswer):
            raise TypeError(f"{ticker}: unexpected employee LLM result {type(answer).__name__}")
        status, count = _decide(answer, str(result.task.meta["source_text"]))
        outcome = _outcome(ticker, result, answer, status, model_name)
        filed = cast(FilingStamp, result.task.meta["stamp"]).filed.normalize()
        all_dates.add(filed)
        if count is not None:
            if filed in chosen:
                outcomes[chosen[filed][1]]["status"] = "superseded"
            chosen[filed] = (count, len(outcomes))
        outcomes.append(outcome)
        context.log.info(
            "employees decision ticker=%s accession=%s cik=%s as_of=%s status=%s count=%s",
            ticker,
            outcome["accession_number"],
            outcome["cik"],
            outcome["filing_date"],
            status,
            count,
        )
    frame = pd.DataFrame(
        [{"ticker": ticker, "as_of": filed, "employees": float(count)} for filed, (count, _) in sorted(chosen.items())],
        columns=FRAME_COLUMNS,
    )
    return EmployeeTickerResult(frame, outcomes, frozenset(all_dates - chosen.keys()))


def build_ticker_employees(
    context: Context,
    ticker: str,
    cik: str,
    *,
    since: pd.Timestamp | None,
    done_accessions: frozenset[str],
    skip_dates: frozenset[pd.Timestamp],
    scope: EdgarScope,
) -> EmployeeTickerResult:
    """Read undecided annual filings and validate their LLM answers."""
    identity = scope.identity
    if identity is None:
        raise ValueError(f"{ticker}: employee extraction needs an identity-aware EdgarScope")
    listed = resolve_registrant_filings(
        ticker,
        HEADCOUNT_FORMS,
        since=since,
        done_accessions=done_accessions,
        registrants=scope.registrants,
        identity=identity,
    )
    stamps = sorted(
        (stamp for stamp in (FilingStamp.of(filing, cik) for filing in listed) if stamp.filed.normalize() not in skip_dates),
        key=_filing_key,
    )
    if not stamps:
        return EmployeeTickerResult(pd.DataFrame(columns=FRAME_COLUMNS), [], frozenset())
    config = with_gpt_overrides(context.config, "employees", provider="open_ai_cheap")
    results = _extract_answers(context, config, ticker, stamps, identity)
    return _decide_ticker(context, ticker, results, str(config.gpt.llm_model.open_ai_cheap))


def _resume_plan(
    context: Context,
    entry: dict,
    tickers: list[str],
    fallback_since: pd.Timestamp,
    *,
    full: bool,
) -> _ResumePlan:
    """Replay all only on `--full`; routine runs retry unresolved filings and skip stored dates."""
    requested = set(tickers)
    relevant = [outcome for outcome in entry.get("filing_outcomes", []) if str(outcome.get("ticker")) in requested]
    pending = [outcome for outcome in relevant if outcome.get("status") not in TERMINAL_STATUSES]
    pending_dates = [pd.Timestamp(outcome["filing_date"]).normalize() for outcome in pending if outcome.get("filing_date")]
    if full or pending or not entry.get("coverage_complete"):
        since, is_full_rescan = min([fallback_since, *pending_dates]), True
    else:
        since, is_full_rescan = manifest_window(
            context,
            Tables.fundamentals_employees,
            tickers,
            fallback_since=fallback_since,
            full_rescan_days=int(context.config.data_extract.manifest_full_rescan_days),
        )
    if full:
        return _ResumePlan(since, is_full_rescan, frozenset(), {}, {})
    stored = context.store.load(
        Tables.fundamentals_employees,
        columns=["ticker", "as_of"],
        where={"ticker": tickers},
        since=min(fallback_since, since),
        optional=True,
    )
    saved_dates: dict[str, set[pd.Timestamp]] = {ticker: set() for ticker in requested}
    if stored is not None:
        for row in stored.itertuples(index=False):
            saved_dates[str(row.ticker)].add(pd.Timestamp(row.as_of).normalize())
    known_saved: dict[str, set[pd.Timestamp]] = {ticker: set() for ticker in requested}
    pending_by_ticker: dict[str, set[pd.Timestamp]] = {ticker: set() for ticker in requested}
    done: set[str] = set()
    for outcome in relevant:
        ticker = str(outcome["ticker"])
        status = outcome.get("status")
        filed = pd.Timestamp(outcome["filing_date"]).normalize() if outcome.get("filing_date") else None
        if status == "saved" and filed is not None:
            known_saved[ticker].add(filed)
        if status not in TERMINAL_STATUSES and filed is not None:
            pending_by_ticker[ticker].add(filed)
        if status == "no_headcount" or (status in {"saved", "superseded"} and filed in saved_dates[ticker]):
            done.add(str(outcome["accession_number"]))
    return _ResumePlan(
        since,
        is_full_rescan,
        frozenset(done),
        {ticker: frozenset(dates) for ticker, dates in saved_dates.items()},
        {ticker: frozenset(dates - known_saved[ticker] - pending_by_ticker[ticker]) for ticker, dates in saved_dates.items()},
    )


def _fetch_ticker_employees(
    ticker: str,
    cik: str,
    *,
    context: Context,
    plan: _ResumePlan,
    scope: EdgarScope,
    fallback_since: pd.Timestamp,
    changed_scopes: frozenset[str],
) -> EmployeeTickerResult:
    """Build one ticker's counts, save its rows and clear its now-unavailable dates; the per-ticker worker."""
    result = build_ticker_employees(
        context,
        ticker,
        cik,
        since=fallback_since if ticker in changed_scopes else plan.since,
        done_accessions=plan.done_accessions,
        skip_dates=plan.skip_dates.get(ticker, frozenset()),
        scope=scope,
    )
    if not result.frame.empty:
        context.store.save(Tables.fundamentals_employees, result.frame)
    # A now-null date is cleared unless a skipped (already decided) filing saved it.
    for filed in result.unavailable_dates - plan.saved_dates.get(ticker, frozenset()):
        context.store.delete(Tables.fundamentals_employees, where={"ticker": ticker, "as_of": filed})
    return result


def fetch_fundamentals_employees(
    context: Context,
    tickers: list[str],
    years_history: int,
    *,
    full: bool = False,
) -> None:
    """Fetch missing issuer-wide counts into `fundamentals_employees`, or recheck every filing on `--full`.

    Raises `IncompleteEdgarRunError` (outcomes saved, frontier not advanced) if any ticker fails or is ambiguous.
    """
    context.ensure_edgar_identity()
    cik_map = load_cik_mapping(context, tickers)
    missing = set(tickers) - set(cik_map["ticker"])
    if missing:
        raise ValueError(f"Employee extraction has no roster CIK for {', '.join(sorted(missing))}")
    fallback_since = pd.Timestamp.today().normalize() - pd.DateOffset(years=years_history)
    entry = get_entry(context, Tables.fundamentals_employees) or {}
    scope, scope_fingerprints, changed_scopes = load_edgar_scope(context, cik_map, entry, identity_aware=True)
    plan = _resume_plan(context, entry, tickers, fallback_since, full=full)
    context.log.info(
        "fundamentals employees resume: %d accession(s) and %d stored filing date(s) skipped",
        len(plan.done_accessions),
        sum(map(len, plan.skip_dates.values())),
    )

    worker = partial(
        _fetch_ticker_employees,
        context=context,
        plan=plan,
        scope=scope,
        fallback_since=fallback_since,
        changed_scopes=changed_scopes,
    )
    results = run_per_ticker(
        cik_map,
        worker,
        desc="fundamentals employees",
        log=context.log,
        max_workers=int(context.config.data_extract.fundamentals_workers),
    )
    successful = [result for result in results if result is not None]
    record_filing_outcomes(context, Tables.fundamentals_employees, [outcome for result in successful for outcome in result.outcomes])
    failed = len(results) - len(successful)
    ambiguous = sum(any(outcome["status"] == "ambiguous" for outcome in result.outcomes) for result in successful)
    rows_written = sum(len(result.frame) for result in successful)
    context.log.info(
        "fundamentals employees: %d/%d ticker(s) read, %d failed, %d ambiguous -> %d rows written",
        len(successful),
        len(cik_map),
        failed,
        ambiguous,
        rows_written,
    )
    if failed or ambiguous:
        raise IncompleteEdgarRunError(
            f"fundamentals employees: {failed} ticker(s) failed and {ambiguous} ambiguous; "
            "accession outcomes were saved, but no complete frontier was advanced"
        )
    record_run(
        context,
        Tables.fundamentals_employees,
        len(cik_map),
        rows_written,
        is_full_rescan=plan.is_full_rescan,
        coverage_complete=True,
        identity_scope_fingerprints=scope_fingerprints,
        tickers=tickers,
    )
