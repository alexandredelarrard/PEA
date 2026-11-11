"""EDGAR step: the per-filing EDGAR walks, one `EdgarFetch` each, through the shared driver.

Order: insider Forms 3/4/5 (EDGAR rows after the last stored zip quarter, then the zip reconcile; the insider run
enters its own source run), SC 13D, SC 13G, 8-K, 10-K/10-Q filing text, 10-K employees text, DEF 14A text, 10-K/10-Q
XBRL facts (the facts run binds its catalogue and GICS parser on entry). The window, run date, ticker scope and CIK
mapping are declared once and shared by every walk. Consumes the identity built and synced
upstream; the identity scope, document cap and retry policy are read once too. The local EDGAR index is refreshed
once per step: `run()` refreshes it over the widest window of the walks before them (`run_index`); a walk run
alone refreshes its own window, unless the step already did or today's refresh (the nightly `edgar-index` task,
over the same widest window) already covered it. The insider walk certifies from the quarter files' footers, so it
needs no refresh of its own. The DEF 14A LLM pass and the 8-K vote parser read what this step stores, so they run after it.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import pandas as pd
from omegaconf import DictConfig

from src.context import Context
from src.data_extract.utils.edgar_extract import edgar_index
from src.data_extract.utils.edgar_extract.edgar_driver import EdgarFetch, EdgarRun, FetchSummary, index_years, refresh_index, run_edgar_fetch
from src.data_extract.utils.edgar_extract.fetch_8k_edgar import SEC_8K_FETCH
from src.data_extract.utils.edgar_extract.fetch_13d_edgar import SEC_13D_FETCH
from src.data_extract.utils.edgar_extract.fetch_13g_edgar import SEC_13G_FETCH
from src.data_extract.utils.edgar_extract.fetch_def14a_text import DEF14A_TEXT_FETCH
from src.data_extract.utils.edgar_extract.fetch_employees_text import EMPLOYEES_TEXT_FETCH
from src.data_extract.utils.edgar_extract.fetch_filing_text import FILING_TEXT_FETCH
from src.data_extract.utils.edgar_extract.fetch_insider_edgar import INSIDER_EDGAR_FETCH
from src.data_extract.utils.fundamentals.fetch_fundamentals_sec import SEC_FUNDAMENTALS_FETCH
from src.data_store.schema import Tables
from src.utils.step import Step
from src.utils.string import run_date


@dataclass(frozen=True)
class EdgarOptions:
    """The EDGAR commands' options: the ticker subset (None: the universe), the window in years (None: the configured
    `years_history`), `full` re-read of stored filings, the run date and `no_cap` (lifts `max_documents_per_run`)."""

    tickers: list[str] | None = None
    years: int | None = None
    full: bool = False
    as_of: pd.Timestamp | None = None
    no_cap: bool = False


#: The walks, in `run()` order.
EDGAR_WALKS: tuple[EdgarFetch, ...] = (
    INSIDER_EDGAR_FETCH,
    SEC_13D_FETCH,
    SEC_13G_FETCH,
    SEC_8K_FETCH,
    FILING_TEXT_FETCH,
    EMPLOYEES_TEXT_FETCH,
    DEF14A_TEXT_FETCH,
    SEC_FUNDAMENTALS_FETCH,
)


def widest_index_years(years_history: int, run_date: pd.Timestamp) -> int:
    """The index window covering every walk on `run_date`: the source-history insider walk reaches its source start."""
    return max(index_years(fetch, years_history, run_date) for fetch in EDGAR_WALKS)


class StepExtractEdgar(Step):
    def __init__(self, context: Context, config: DictConfig, options: EdgarOptions | None = None) -> None:
        """The step over `options` (default: the universe, the configured window); reads nothing until a walk runs."""
        super().__init__(context=context, config=config)
        self._options = options or EdgarOptions()
        self._indexed_years: int | None = None

    @cached_property
    def _run(self) -> EdgarRun:
        """The run state (CIKs, identity, cap), read once on first use so constructing the step reads nothing."""
        o = self._options
        return EdgarRun.load(self._context, o.tickers or self._universe.tickers, o.years, as_of=o.as_of, full=o.full, no_cap=o.no_cap)

    def _run_fetch(self, fetch: EdgarFetch) -> FetchSummary:
        """Run one walk through the shared driver, on an index covering its window: refreshed unless this step,
        or today's refresh, already covered it."""
        # index: the walk's window, refreshed unless this step or today's nightly refresh already covers it
        years = index_years(fetch, self._run.years_history, self._run.run_date)
        covered = self._indexed_years is not None and self._indexed_years >= years
        refreshed = not covered and self._run.run_date == run_date(None) and edgar_index.refreshed_today(self._context, years)
        if refreshed:
            self._log.info("edgar index: already refreshed today over %d year(s); not refreshed again", years)
        if not covered and not refreshed:
            refresh_index(self._context, self._run, years)
            self._indexed_years = years

        # walk: the shared driver
        return run_edgar_fetch(self._context, fetch, self._run)

    def run_index(self) -> None:
        """Refresh the local EDGAR index once over the widest window of the walks."""
        years = widest_index_years(self._run.years_history, self._run.run_date)
        refresh_index(self._context, self._run, years)
        self._indexed_years = years

    def run_insider(self) -> FetchSummary:
        """Walk Forms 3/4/5 into the insider tables (EDGAR rows after the last zip quarter, then the zip reconcile)."""
        return self._run_fetch(INSIDER_EDGAR_FETCH)

    def run_13d(self) -> FetchSummary:
        """Walk the SC 13D filings into `sec_13d` and `sec_13d_transactions`."""
        return self._run_fetch(SEC_13D_FETCH)

    def run_13g(self) -> FetchSummary:
        """Walk the SC 13G filings into `sec_13g`."""
        return self._run_fetch(SEC_13G_FETCH)

    def run_8k(self) -> FetchSummary:
        """Walk the 8-K filings into `sec_8k`."""
        return self._run_fetch(SEC_8K_FETCH)

    def run_filing_text(self) -> FetchSummary:
        """Walk the 10-K/10-Q filings into `filing_risk_text`."""
        return self._run_fetch(FILING_TEXT_FETCH)

    def run_employees_text(self) -> FetchSummary:
        """Walk the 10-K family filings into `sec_employees_text` (candidate windows and their union), created from its
        committed DDL on a first run."""
        self._context.store.create_declared(Tables.sec_employees_text)
        return self._run_fetch(EMPLOYEES_TEXT_FETCH)

    def run_def14a_text(self) -> FetchSummary:
        """Walk the proxies into `sec_def14a_text` (untruncated section payloads, subject guard as markers), created from
        its committed DDL on a first run."""
        self._context.store.create_declared(Tables.sec_def14a_text)
        return self._run_fetch(DEF14A_TEXT_FETCH)

    def run_fundamentals_facts(self) -> FetchSummary:
        """Walk the 10-K/10-Q XBRL facts into `sec_fundamentals_facts`, created by its first write."""
        return self._run_fetch(SEC_FUNDAMENTALS_FETCH)

    def run(self) -> None:
        """Refresh the index once, then run every walk in order."""
        self.run_index()
        self.run_insider()
        self.run_13d()
        self.run_13g()
        self.run_8k()
        self.run_filing_text()
        self.run_employees_text()
        self.run_def14a_text()
        self.run_fundamentals_facts()
