"""
step_extract_fundamentals.py  (src/data_extract/step_extract_fundamentals.py)
-----------------------------------------------------------------------------
Fundamentals / financials extraction:
  * company fundamentals (balance sheet, income, cash flow), in TWO layers: the
    linkbase-driven as-filed facts (`fetch_fundamentals_sec.py`, network-bound) and the
    publication-event history replay over them (`build_history.py`, no network). Headcount
    rides the facts walk and lands in `fundamentals_employees`. `FundamentalsValidator`
    follows at Phase 5b. See reports/planning/active-tasks/2026-08-23-fundamentals-rebuild-plan-v2.md.
Both layers are dormant in `run()` (the `fundamentals` command runs them). Earnings surprises are
`StepExtractPrices.run_earnings_surprises`; the notes and financial-statement zips are `StepExtractZips`.
"""

import time
from collections.abc import Iterator
from contextlib import contextmanager

from omegaconf import DictConfig

from src.context import Context
from src.utils.step import Step


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

    def run(self, tickers: list[str], full: bool = False) -> None:

        # Both layers are dormant here; when re-enabled, the window is
        # `years_history = int(self._config.data_extract.years_history)` (`self._config`, the attribute
        # `Step.__init__` sets: `self.config` raised AttributeError on every real invocation of this step).

        # Per-filing SEC XBRL -> fundamentals_facts (+ fundamentals_employees from the same
        # 10-K prose), each KPI resolved from the filer's own calculation linkbase.
        # Append-only and accession-grain, so a nightly re-run is idempotent and an
        # amendment lands as its own row at its own filing date.
        # with self._stage("fundamentals facts (SEC XBRL)", tickers):
        #     fetch_fundamentals_sec(
        #         self._context, tickers=tickers,
        #         years_history=years_history,
        #         full=full)

        # ... then the publication-event replay over exactly those facts. Immediately after,
        # and never on its own schedule: the snapshot is only as fresh as the facts it reads,
        # and a nightly that fetched a 10-K without replaying it would leave the newest
        # filing invisible to every consumer. Refuses to overwrite an already-published row
        # (see `diff_against_stored`), so a resolution change surfaces here rather than in a
        # cube six months later.
        # with self._stage("fundamentals history replay", tickers):
        #     build_fundamentals_history(self._context, tickers=tickers,
        #                                rebuild_history=full)
        self._log.info("fundamentals: facts walk and history replay dormant in this step (%d ticker(s)); run `fundamentals`", len(tickers))
