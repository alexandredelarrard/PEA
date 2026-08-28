"""
step_extract_fundamentals.py  (src/data_extract/step_extract_fundamentals.py)
-----------------------------------------------------------------------------
Fundamentals / financials extraction:
  * company fundamentals (balance sheet, income, cash flow), in TWO layers: the
    linkbase-driven as-filed facts (`fetch_fundamentals_sec.py`, network-bound) and the
    publication-event history replay over them (`build_history.py`, no network). Headcount
    rides the facts walk and lands in `fundamentals_employees`. `FundamentalsValidator`
    follows at Phase 5b. See reports/planning/active-tasks/2026-08-23-fundamentals-rebuild-plan-v2.md.
  * earnings surprises
  * insider transactions
  * footnote (notes) pension detail + note text
"""

import time
from contextlib import contextmanager
from typing import Iterator

from omegaconf import DictConfig

from src.context import Context
from src.utils.step import Step
from src.data_extract.utils.fundamentals.fetch_earnings_surprises import fetch_earnings_surprises
from src.data_extract.utils.fundamentals.fetch_fundamentals_sec import fetch_fundamentals_sec
from src.data_extract.utils.fundamentals.build_history import build_fundamentals_history
from src.data_extract.utils.prices.fetch_insider_transactions import fetch_insider_transactions
from src.data_extract.utils.fundamentals.fetch_financial_notes import fetch_financial_notes


def _elapsed(seconds: float) -> str:
    """`3725.4` -> `"1h 02m 05s"`. Whole units only: the number that matters in a 10-hour
    walk's log is which stage took the hours, never its milliseconds."""
    hours, rest = divmod(int(seconds), 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    return f"{minutes}m {secs:02d}s" if minutes else f"{secs}s"


class StepExtractFundamentals(Step):

    def __init__(self, context: Context, config: DictConfig):
        super().__init__(context=context, config=config)

    @contextmanager
    def _stage(self, what: str, tickers: list[str]) -> Iterator[None]:
        """One INFO line entering a sub-fetcher and one leaving it with its wall clock.

        The live from-scratch walk logged **12 INFO lines in 10.6 h** and not one of them
        placed a stage in time, so a stall and a slow stage read identically and a failure
        could not be attributed to a fetcher at all. The `finally` is the point: a stage that
        RAISES still logs how long it ran before it did.
        """
        self._log.info("%s: starting, %d ticker(s)", what, len(tickers))
        start = time.perf_counter()
        try:
            yield
        finally:
            self._log.info("%s: done in %s", what, _elapsed(time.perf_counter() - start))

    def run(self, tickers: list[str], full : bool=False) -> None:

        # Per-filing SEC XBRL -> fundamentals_facts (+ fundamentals_employees from the same
        # 10-K prose), each KPI resolved from the filer's own calculation linkbase.
        # Append-only and accession-grain, so a nightly re-run is idempotent and an
        # amendment lands as its own row at its own filing date.
        with self._stage("fundamentals facts (SEC XBRL)", tickers):
            fetch_fundamentals_sec(
                self._context, tickers=tickers,
                years_history=int(self._config.data_extract.years_history),
                full=full)

        # ... then the publication-event replay over exactly those facts. Immediately after,
        # and never on its own schedule: the snapshot is only as fresh as the facts it reads,
        # and a nightly that fetched a 10-K without replaying it would leave the newest
        # filing invisible to every consumer. Refuses to overwrite an already-published row
        # (see `diff_against_stored`), so a resolution change surfaces here rather than in a
        # cube six months later.
        with self._stage("fundamentals history replay", tickers):
            build_fundamentals_history(self._context, tickers=tickers,
                                       rebuild_history=full)

        # earnings surprises
        with self._stage("earnings surprises", tickers):
            fetch_earnings_surprises(self._context, tickers=tickers)

        # Officer / director / 10%-owner transactions from the Insider Transactions sets:
        with self._stage("insider transactions", tickers):
            fetch_insider_transactions(self._context, tickers=tickers)

        # Footnote (notes) pension detail + note TEXT from the Financial Statement
        # AND Notes data sets -> notes_num / notes_text. Heavy (~26GB back-fill at
        # notes_years_history=15); own cache dir + config knob. (fetch_fails_to_deliver
        # lives in StepExtractPrices, alongside the other price/settlement signals.)
        with self._stage("financial notes (pension + note text)", tickers):
            fetch_financial_notes(self._context, tickers=tickers)
