"""Source-backed SEC annual employee counts, dated by the filing's public date, read from `sec_employees_text`."""

from __future__ import annotations

import queue
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from typing import cast

import pandas as pd

from src.context import Context
from src.data_extract.utils.common.identity import load_identity
from src.data_extract.utils.common.resume import DONE_PER_KEY, document_floor
from src.data_extract.utils.edgar_extract.document_resume import WorklistRequest, document_worklist
from src.data_extract.utils.edgar_extract.edgar_driver import FilingStamp, universe_rows
from src.data_extract.utils.edgar_extract.parallel_fetch import run_per_ticker
from src.data_extract.utils.fundamentals.employee_guard import COUNT_COLUMNS, FilingDecision, employee_decisions, employee_task, ticker_rows
from src.data_store.schema import Resume, Tables
from src.gpt_extract.transformers.gpt_getter import LLMExtractor
from src.gpt_extract.utils.schemas_gpt import LlmResult
from src.utils.string import run_date

# The `sec_employees_text` columns the worklist reads; `windows` and `pieces` are read per ticker, for its undecided filings.
_TEXT_COLUMNS = ["ticker", "accession_number", "cik", "form", "filing_date", "period_of_report", "source_document"]
_FAILED_SHOWN = 20  # failed tickers named in the coverage log

#: The provider the employee extraction calls (a cheaper model than `config.gpt.default_api`).
EMPLOYEES_PROVIDER = "open_ai_cheap"


@dataclass(frozen=True)
class EmployeeTickerResult:
    """One `FRAME_COLUMNS` row per decided filing date (NULL components with a status when no count is
    supported), the per-filing decisions, and the accessions whose LLM call failed (listed again next run)."""

    frame: pd.DataFrame
    outcomes: list[dict]
    failed: tuple[str, ...] = ()


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
    history = df_history.loc[df_history["ticker"].astype(str).eq(ticker), ["as_of", *COUNT_COLUMNS]].reset_index(drop=True)
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
    by_llm, failed = employee_decisions(ticker, results, history)
    stamps = [cast(FilingStamp, result.task.meta["stamp"]) for result in failed]
    withheld = {stamp.filed.normalize() for stamp in stamps}
    decisions = [decision for decision in by_llm if decision.stamp.filed.normalize() not in withheld]
    frame, outcomes = ticker_rows(ticker, decisions)

    # save: the decided dates
    if not frame.empty:
        context.store.save(Tables.fundamentals_employees, frame)

    # log: the failed calls, then each filing's decision
    _log_ticker(context, ticker, failed, decisions)
    return EmployeeTickerResult(frame, outcomes, tuple(str(stamp.accession_number) for stamp in stamps))


def _log_ticker(context: Context, ticker: str, failed: list[LlmResult], decisions: list[FilingDecision]) -> None:
    """One warning per failed LLM call, then one line per decided filing in filing order."""
    # log: the failed calls (listed again next run)
    for result in failed:
        context.log.warning(
            "fundamentals employees: %s %s LLM call failed (%s); its filing date is not stored and is listed again next run",
            ticker,
            cast(FilingStamp, result.task.meta["stamp"]).accession_number,
            result.error,
        )

    # log: each decision with its guarded components and rejections
    for decision in decisions:
        stamp, decided = decision.stamp, decision.decided
        context.log.info(
            "employees decision ticker=%s accession=%s cik=%s as_of=%s source_document=%s status=%s total=%s full_time=%s part_time=%s basis=%s rejected=%s",
            ticker,
            stamp.accession_number,
            stamp.cik,
            stamp.filed.normalize().date(),
            decision.source_document,
            decided.status,
            decided.employees_total,
            decided.employees_full_time,
            decided.employees_part_time,
            decided.basis,
            "; ".join(decided.rejected) or "-",
        )


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
    stored = context.store.load(Tables.fundamentals_employees, columns=["ticker", "as_of", *COUNT_COLUMNS], where={"ticker": tickers}, optional=True)
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
    df_history = pd.DataFrame(columns=["ticker", "as_of", *COUNT_COLUMNS]) if stored is None else stored
    df_todo, n_held = _undecided(df_text, None if full else df_history, pending)
    df_keys = cik_map[cik_map["ticker"].isin(df_todo["ticker"])].reset_index(drop=True)
    lent: queue.SimpleQueue[LLMExtractor] = queue.SimpleQueue()
    for extractor in extractors:
        lent.put(extractor)
    worker = partial(_ticker_employees, context=context, extractors=lent, df_todo=df_todo, df_history=df_history)
    results = run_per_ticker(df_keys, worker, desc="fundamentals employees", max_workers=len(extractors), key_cols=("ticker",))

    # log: the run's coverage
    _log_run(context, df_keys, results, n_held)
