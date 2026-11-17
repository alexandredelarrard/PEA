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

from omegaconf import DictConfig

from src.context import Context
from src.data_extract.utils.edgar_extract import edgar_index
from src.data_extract.utils.edgar_extract.edgar_driver import EdgarFetch, EdgarRun, FetchSummary, refresh_index, run_edgar_fetch
from src.data_extract.utils.edgar_extract.fetch_8k_edgar import SEC_8K_FETCH
from src.data_extract.utils.edgar_extract.fetch_13d_edgar import SEC_13D_FETCH
from src.data_extract.utils.edgar_extract.fetch_13g_edgar import SEC_13G_FETCH
from src.data_extract.utils.edgar_extract.fetch_def14a_text import DEF14A_TEXT_FETCH
from src.data_extract.utils.edgar_extract.fetch_employees_text import EMPLOYEES_TEXT_FETCH
from src.data_extract.utils.edgar_extract.fetch_filing_text import FILING_TEXT_FETCH
from src.data_extract.utils.edgar_extract.fetch_insider_edgar import INSIDER_EDGAR_FETCH
from src.data_extract.utils.fundamentals.fetch_fundamentals_sec import SEC_FUNDAMENTALS_FETCH
from src.data_store.schema import Tables
from src.utils.data_policy import window_policy
from src.utils.step import Step
from src.utils.window import Window


@dataclass(frozen=True)
class EdgarOptions:
    """The source-key subset and explicit removal of the configured document cap."""

    tickers: list[str] | None = None
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


def widest_index_window(context: Context) -> Window | None:
    """Union configured filing bounds for every active walk; source floors are compiled once."""
    windows = [policy.requested for fetch in EDGAR_WALKS if (policy := window_policy(context.config, fetch.done, context.run.as_of)) is not None]
    if not windows:
        return None
    return Window(min(window.start for window in windows), max(window.end for window in windows))


class StepExtractEdgar(Step):
    def __init__(self, context: Context, config: DictConfig, options: EdgarOptions | None = None) -> None:
        """The step over `options` (default: the universe, the configured window); reads nothing until a walk runs."""
        super().__init__(context=context, config=config)
        self._options = options or EdgarOptions()
        self._indexed_window: Window | None = None

    @cached_property
    def _run(self) -> EdgarRun:
        """The run state (CIKs, identity, cap), read once on first use so constructing the step reads nothing."""
        o = self._options
        return EdgarRun.load(self._context, o.tickers or self._universe.tickers, no_cap=o.no_cap)

    def _run_fetch(self, fetch: EdgarFetch) -> FetchSummary:
        """Run one walk through the shared driver, on an index covering its window: refreshed unless this step,
        or today's refresh, already covered it."""
        policy = window_policy(self._config, fetch.done, self._context.run.as_of)
        if policy is not None:
            window = policy.requested
            covered = self._indexed_window is not None and self._indexed_window.intersection(window) == window
            if not covered and not edgar_index.refreshed_today(self._context, window):
                if self._indexed_window is not None:
                    window = Window(min(window.start, self._indexed_window.start), max(window.end, self._indexed_window.end))
                refresh_index(self._context, window)
                self._indexed_window = window
        return run_edgar_fetch(self._context, fetch, self._run)

    def run_index(self) -> None:
        """Refresh the local EDGAR index once over all active configured filing windows."""
        window = widest_index_window(self._context)
        if window is not None:
            refresh_index(self._context, window)
            self._indexed_window = window

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
