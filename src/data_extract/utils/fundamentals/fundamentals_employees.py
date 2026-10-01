"""Source-backed SEC annual employee counts, dated by the filing's public date."""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field

from src.context import Context
from src.data_extract.utils.common.edgar_driver import PROGRAMMING_ERRORS, IncompleteEdgarRunError, filed_by, period_of_report
from src.data_extract.utils.common.edgar_extract import html_to_text
from src.data_extract.utils.common.identity import Identity, load_identity
from src.data_extract.utils.common.parallel_fetch import run_per_ticker
from src.data_extract.utils.common.registrant import Registrant, identity_scope_fingerprint, load_registrants, resolve_registrant_filings
from src.data_extract.utils.common.run_manifest import changed_scope_tickers, get_entry, manifest_window, record_filing_outcomes, record_run
from src.data_extract.utils.common.sec_utils import load_cik_mapping
from src.data_store.schema import Tables
from src.gpt_extract.transformers.gpt_getter import LLMExtractor
from src.gpt_extract.transformers.step_gpt_extracter import with_gpt_overrides
from src.gpt_extract.utils.schemas_gpt import LlmTask

HEADCOUNT_FORMS = ("10-K", "10-K/A", "10-K405")
SAVED = "saved"
NO_HEADCOUNT = "no_headcount"
AMBIGUOUS = "ambiguous"
SUPERSEDED = "superseded"
TERMINAL_STATUSES = frozenset({SAVED, NO_HEADCOUNT, SUPERSEDED})

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


def filing_body_text(filing: object) -> str:
    """Use SEC filing text, including visible HTML table cells, without OCR."""
    raw = filing.html()
    return html_to_text(raw) if raw else filing.text() or ""


def employee_excerpt(text: str, limit: int) -> str:
    """Keep filing opening text and the first workforce contexts, as in the benchmark."""
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
    return "\n\n[... filing gap ...]\n\n".join(text[start:end] for start, end in merged)[:limit]


def supported_employee_count(answer: EmployeeAnswer, source_text: str) -> int | None:
    """Accept only a count supported by an exact text claim; abstain on bounds."""
    if answer.status != "found" or answer.count is None or answer.count <= 0 or not answer.quote:
        return None
    quote = " ".join(answer.quote.split()).casefold()
    source = " ".join(source_text.split()).casefold()
    if quote not in source or _BOUND_RE.search(quote):
        return None
    # A model can trim "over" off an otherwise literal source quote.
    for match in re.finditer(re.escape(quote), source):
        if _BOUND_RE.search(source[max(0, match.start() - 32) : match.end()]):
            return None
    split = _SPLIT_RE.search(quote)
    if split and split.group(2).casefold() != split.group(4).casefold():
        first = int(split.group(1).replace(",", ""))
        second = int(split.group(3).replace(",", ""))
        return first + second if answer.count in {first, second, first + second} else None
    numbers = {int(token.replace(",", "")) for token in _NUMBER_RE.findall(quote)}
    return answer.count if answer.count in numbers else None


def _filing_key(filing: object) -> tuple[pd.Timestamp, int, str]:
    return (pd.Timestamp(filing.filing_date).normalize(), 1 if str(filing.form).upper() == "10-K/A" else 0, str(filing.accession_number))


def build_ticker_employees(
    context: Context,
    ticker: str,
    cik: str,
    *,
    since: pd.Timestamp | None,
    done_accessions: frozenset[str],
    registrants: dict[str, Registrant],
    identity: Identity,
    symbol_tenure: pd.DataFrame,
) -> EmployeeTickerResult:
    """Read every eligible annual filing and validate its LLM answer before saving."""
    filings = sorted(
        resolve_registrant_filings(
            ticker,
            HEADCOUNT_FORMS,
            since=since,
            done_accessions=done_accessions,
            registrants=registrants,
            identity=identity,
            symbol_tenure=symbol_tenure,
            roster_cik=cik,
        ),
        key=_filing_key,
    )
    if not filings:
        return EmployeeTickerResult(pd.DataFrame(columns=["ticker", "as_of", "employees"]), [], frozenset())

    config = with_gpt_overrides(context.config, "employees", provider="open_ai_cheap")
    extractor = LLMExtractor(context, config, action="employees", threads=1)
    model_name = str(config.gpt.llm_model.open_ai_cheap)
    max_chars = int(config.gpt.max_chars.employees)
    for sequence, filing in enumerate(filings):
        actual_cik = getattr(filing, "cik", None)
        if not actual_cik or not identity.owns(ticker, actual_cik):
            raise ValueError(f"{ticker} {filing.accession_number}: filing CIK {actual_cik!r} is outside the issuer lineage")
        report = period_of_report(filing)
        text = filing_body_text(filing)
        if not text.strip():
            raise ValueError(f"{ticker} {filing.accession_number}: filing text unavailable")
        filed = pd.Timestamp(filing.filing_date).normalize()
        report_date = pd.Timestamp(report).normalize() if report is not None else None
        prefix = (
            f"Ticker: {ticker}\nFiscal period end: {report_date.date() if report_date is not None else 'unknown'}\n"
            f"SEC filing date: {filed.date()}\nAccession: {filing.accession_number}\n"
            "The following is an excerpt, not necessarily the complete 10-K:\n\n"
        )
        if max_chars <= len(prefix):
            raise ValueError("gpt.max_chars.employees is too small for filing metadata")
        source_text = employee_excerpt(text, max_chars - len(prefix))
        extractor.submit(
            LlmTask(
                seq=sequence,
                payload=prefix + source_text,
                schema=EmployeeAnswer,
                meta={"filing": filing, "report_date": report_date, "source_text": source_text},
            )
        )

    results = extractor.run()
    if len(results) != len(filings) or any(not result.ok for result in results):
        errors = [f"{result.task.meta['filing'].accession_number}: {result.error}" for result in results if not result.ok]
        raise RuntimeError(f"{ticker}: employee LLM extraction incomplete: {errors}")

    outcomes: list[dict] = []
    chosen: dict[pd.Timestamp, tuple[int, int]] = {}
    all_dates: set[pd.Timestamp] = set()
    for result in results:
        answer = result.parsed
        if not isinstance(answer, EmployeeAnswer):
            raise TypeError(f"{ticker}: unexpected employee LLM result {type(answer).__name__}")
        filing = result.task.meta["filing"]
        filed = pd.Timestamp(filing.filing_date).normalize()
        report_date = result.task.meta["report_date"]
        count = supported_employee_count(answer, str(result.task.meta["source_text"]))
        if count is not None:
            status = SAVED
        elif answer.status in {"not_disclosed", "image_only"} and answer.count is None:
            status = NO_HEADCOUNT
        else:
            status = AMBIGUOUS
        outcome = {
            "ticker": ticker,
            "accession_number": str(filing.accession_number),
            "cik": filed_by(filing, cik),
            "form": str(filing.form),
            "filing_date": filed.strftime("%Y-%m-%d"),
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
        all_dates.add(filed)
        if count is not None:
            prior = chosen.get(filed)
            if prior is not None:
                outcomes[prior[1]]["status"] = SUPERSEDED
            chosen[filed] = (count, len(outcomes))
        outcomes.append(outcome)
        context.log.info(
            "employees decision ticker=%s accession=%s cik=%s as_of=%s status=%s count=%s",
            ticker,
            filing.accession_number,
            outcome["cik"],
            outcome["filing_date"],
            status,
            count,
        )

    frame = pd.DataFrame(
        [{"ticker": ticker, "as_of": filed, "employees": float(count)} for filed, (count, _) in sorted(chosen.items())],
        columns=["ticker", "as_of", "employees"],
    )
    return EmployeeTickerResult(frame, outcomes, frozenset(all_dates - chosen.keys()))


def fetch_fundamentals_employees(
    context: Context,
    tickers: list[str],
    years_history: int,
    *,
    full: bool = False,
) -> None:
    """Fetch issuer-wide counts; recheck legacy decisions and clear unsupported rows."""
    context.ensure_edgar_identity()
    cik_map = load_cik_mapping(context, tickers)
    missing = set(tickers) - set(cik_map["ticker"])
    if missing:
        raise ValueError(f"Employee extraction has no roster CIK for {', '.join(sorted(missing))}")
    fallback_since = pd.Timestamp.today().normalize() - pd.DateOffset(years=years_history)
    entry = get_entry(context, Tables.fundamentals_employees) or {}
    identity = load_identity(context)
    symbol_tenure = pd.DataFrame(
        [{"symbol": symbol, "issuer_cik": cik} for symbol, ciks in identity.ciks_by_symbol.items() for cik in ciks],
        columns=["symbol", "issuer_cik"],
    )
    registrants = load_registrants(str(context.config_dir))
    scope_fingerprints = {
        str(row.ticker): identity_scope_fingerprint(str(row.ticker), str(row.cik), identity, symbol_tenure, registrants)
        for row in cik_map.itertuples()
    }
    changed_scopes = changed_scope_tickers(entry, scope_fingerprints)
    model_name = str(context.config.gpt.llm_model.open_ai_cheap)
    requested = set(tickers)
    relevant = [outcome for outcome in entry.get("filing_outcomes", []) if str(outcome.get("ticker")) in requested]
    needs_migration = any(outcome.get("model") != model_name for outcome in relevant)
    pending = [outcome for outcome in relevant if outcome.get("status") not in TERMINAL_STATUSES]
    pending_dates = [pd.Timestamp(outcome["filing_date"]).normalize() for outcome in pending if outcome.get("filing_date")]
    if full or needs_migration or pending or not entry.get("coverage_complete"):
        since, is_full_rescan = min([fallback_since, *pending_dates]), True
    else:
        since, is_full_rescan = manifest_window(
            context,
            Tables.fundamentals_employees,
            len(cik_map),
            fallback_since=fallback_since,
            full_rescan_days=int(context.config.data_extract.manifest_full_rescan_days),
        )
    done = (
        frozenset()
        if full
        else frozenset(
            str(outcome["accession_number"])
            for outcome in relevant
            if outcome.get("model") == model_name and outcome.get("status") in TERMINAL_STATUSES
        )
    )
    saved_dates = (
        {}
        if full
        else {
            ticker: frozenset(
                pd.Timestamp(outcome["filing_date"]).normalize()
                for outcome in relevant
                if outcome.get("ticker") == ticker and outcome.get("model") == model_name and outcome.get("status") == SAVED
            )
            for ticker in requested
        }
    )
    create_lock = threading.Lock()
    created = False

    def _save(frame: pd.DataFrame) -> None:
        nonlocal created
        if created:
            context.store.save(Tables.fundamentals_employees, frame)
            return
        with create_lock:
            context.store.save(Tables.fundamentals_employees, frame)
            created = True

    def _worker(ticker: str, cik: str) -> EmployeeTickerResult | None:
        try:
            result = build_ticker_employees(
                context,
                ticker,
                cik,
                since=fallback_since if ticker in changed_scopes else since,
                done_accessions=done,
                registrants=registrants,
                identity=identity,
                symbol_tenure=symbol_tenure,
            )
            if not result.frame.empty:
                _save(result.frame)
            for filed in result.unavailable_dates:
                if filed not in saved_dates.get(ticker, ()):
                    context.store.delete(Tables.fundamentals_employees, where={"ticker": ticker, "as_of": filed})
            return result
        except PROGRAMMING_ERRORS:
            raise
        except Exception as exc:  # noqa: BLE001 -- one ticker cannot hide incomplete coverage
            context.log.warning("fundamentals employees: %s failed (%s)", ticker, exc)
            return None

    results = run_per_ticker(
        cik_map,
        _worker,
        desc="fundamentals employees",
        max_workers=int(context.config.data_extract.fundamentals_workers),
    )
    successful = [result for result in results if result is not None]
    record_filing_outcomes(context, Tables.fundamentals_employees, [outcome for result in successful for outcome in result.outcomes])
    failed = len(results) - len(successful)
    ambiguous = sum(any(outcome["status"] == AMBIGUOUS for outcome in result.outcomes) for result in successful)
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
        is_full_rescan=is_full_rescan,
        coverage_complete=True,
        identity_scope_fingerprints=scope_fingerprints,
    )
