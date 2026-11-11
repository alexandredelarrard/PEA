"""Zips step: the SEC bulk data sets (Form 3/4/5, Notes, fails-to-deliver).

The download phase only caches archives; the identity step reads those caches; the ingest phase then parses the cached archives into their tables, still downloading a
pending period not cached yet. The insider ingest runs without an insider source run: the EDGAR walk owns its own.
The Notes period list (one landing-page scrape) is shared by both phases.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from omegaconf import DictConfig

from src.context import Context
from src.data_extract.utils.common.bulk_cache import cache_dir
from src.data_extract.utils.common.resume import ZipScope
from src.data_extract.utils.zips.fetch_fails_to_deliver import download_fails_to_deliver, fetch_fails_to_deliver
from src.data_extract.utils.zips.fetch_financial_notes import (
    download_notes_periods,
    ingest_financial_notes,
    notes_periods,
    repair_notes_availability,
)
from src.data_extract.utils.zips.fetch_insider_transactions import download_insider_transactions, fetch_insider_transactions
from src.utils.step import Step
from src.utils.string import run_date


@dataclass(frozen=True)
class ZipsOptions:
    """The zip commands' options: the ingested ticker subset (None: the universe), `full` re-download/re-parse, the run
    date and the Notes metadata-only clock repair."""

    tickers: list[str] | None = None
    full: bool = False
    as_of: pd.Timestamp | None = None
    repair_availability: bool = False


class StepExtractZips(Step):
    def __init__(self, context: Context, config: DictConfig, options: ZipsOptions | None = None) -> None:
        super().__init__(context=context, config=config)
        self._options = options or ZipsOptions()
        paths = config.local.paths
        self._notes_cache = cache_dir(context, paths.financial_notes)
        self._years_history = int(config.data_extract.years_history)
        self._run_date = run_date(self._options.as_of)
        self._notes_periods: list[str] | None = None

    @property
    def _scope(self) -> ZipScope:
        """The ingest scope: the `-t` subset, else the universe (read on first use)."""
        return ZipScope(tickers=self._options.tickers or self._universe.tickers, run_date=self._run_date, full=self._options.full)

    def _notes_period_list(self) -> list[str]:
        """The Notes periods of the window on the run date, scraped once per step."""
        if self._notes_periods is None:
            self._notes_periods = notes_periods(self._context, self._years_history + 1, self._run_date)
        return self._notes_periods

    def run_download_insider(self) -> int:
        return len(download_insider_transactions(self._context))

    def run_download_financial_notes(self) -> int:
        return download_notes_periods(self._context, self._notes_cache, self._notes_period_list())

    def run_download_fails_to_deliver(self) -> int:
        return download_fails_to_deliver(self._context, self._years_history, self._run_date)

    def run_download(self) -> None:
        self.run_download_insider()
        self.run_download_financial_notes()
        self.run_download_fails_to_deliver()

    def run_ingest_insider(self) -> int:
        scope = self._scope
        return fetch_insider_transactions(self._context, scope.tickers, self._years_history, reparse=scope.full, as_of=scope.run_date)

    def run_ingest_financial_notes(self) -> int:
        if self._options.repair_availability:
            return repair_notes_availability(self._context, self._notes_cache)
        return ingest_financial_notes(self._context, self._notes_cache, self._scope, self._notes_period_list())

    def run_ingest_fails_to_deliver(self) -> int:
        return fetch_fails_to_deliver(self._context, self._scope.full)

    def run_ingest(self) -> None:
        self.run_ingest_insider()
        self.run_ingest_financial_notes()
        self.run_ingest_fails_to_deliver()

    def run(self) -> None:
        """Download then ingest; the identity step runs between the two phases when both are wanted (`StepExtractAllData`)."""
        self.run_download()
        self.run_ingest()
