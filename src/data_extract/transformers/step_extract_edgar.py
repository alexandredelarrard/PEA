"""EDGAR step: the per-filing EDGAR walks, one `EdgarFetch` each, through the shared driver.

Order: insider Forms 3/4/5 (EDGAR rows after the last stored zip quarter, then the zip reconcile; the insider run
enters its own source run), SC 13D, SC 13G, 8-K, 10-K/10-Q filing text, DEF 14A ECD XBRL. The window, run date,
ticker scope and CIK mapping are declared once and shared by every walk. Consumes the identity built and propagated
upstream; the identity scope, document cap and retry policy are read once too. The DEF 14A LLM pass and the 8-K vote parser read what this step stores, so they run after it.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from omegaconf import DictConfig

from src.context import Context
from src.data_extract.utils.edgar_extract.edgar_driver import EdgarFetch, EdgarRun, FetchSummary, run_edgar_fetch
from src.data_extract.utils.edgar_extract.fetch_8k_edgar import SEC_8K_FETCH
from src.data_extract.utils.edgar_extract.fetch_13d_edgar import SEC_13D_FETCH
from src.data_extract.utils.edgar_extract.fetch_13g_edgar import SEC_13G_FETCH
from src.data_extract.utils.edgar_extract.fetch_def14a_edgar import DEF14A_EDGAR_FETCH
from src.data_extract.utils.edgar_extract.fetch_filing_text import FILING_TEXT_FETCH
from src.data_extract.utils.edgar_extract.fetch_insider_edgar import INSIDER_EDGAR_FETCH
from src.utils.step import Step
from src.utils.universe import load_universe_tickers


@dataclass(frozen=True)
class EdgarOptions:
    """The EDGAR commands' options: the ticker subset (None: the universe), the window in years (None: the configured
    `years_history`), `full` re-read of stored filings, the run date and `no_cap` (lifts `max_documents_per_run`)."""

    tickers: list[str] | None = None
    years: int | None = None
    full: bool = False
    as_of: pd.Timestamp | None = None
    no_cap: bool = False


class StepExtractEdgar(Step):
    def __init__(self, context: Context, config: DictConfig, options: EdgarOptions | None = None) -> None:
        super().__init__(context=context, config=config)
        options = options or EdgarOptions()
        self._run = EdgarRun.load(
            context, options.tickers or load_universe_tickers(context), options.years, as_of=options.as_of, full=options.full, no_cap=options.no_cap
        )

    def _run_fetch(self, fetch: EdgarFetch) -> FetchSummary:
        return run_edgar_fetch(self._context, fetch, self._run)

    def run_insider(self) -> FetchSummary:
        return self._run_fetch(INSIDER_EDGAR_FETCH)

    def run_13d(self) -> FetchSummary:
        return self._run_fetch(SEC_13D_FETCH)

    def run_13g(self) -> FetchSummary:
        return self._run_fetch(SEC_13G_FETCH)

    def run_8k(self) -> FetchSummary:
        return self._run_fetch(SEC_8K_FETCH)

    def run_filing_text(self) -> FetchSummary:
        return self._run_fetch(FILING_TEXT_FETCH)

    def run_def14a_ecd(self) -> FetchSummary:
        return self._run_fetch(DEF14A_EDGAR_FETCH)

    def run(self) -> None:
        self.run_insider()
        self.run_13d()
        self.run_13g()
        self.run_8k()
        self.run_filing_text()
        self.run_def14a_ecd()
