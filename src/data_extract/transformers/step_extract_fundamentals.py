"""
step_extract_fundamentals.py  (src/data_extract/step_extract_fundamentals.py)
-----------------------------------------------------------------------------
Fundamentals / financials extraction:
  * company fundamentals (balance sheet, income, cash flow), in TWO layers: the
    linkbase-driven as-filed facts (`fetch_fundamentals_sec.py`, network-bound) and the
    publication-event history replay over them (`build_history.py`, no network). Headcount
    rides the facts walk and lands in `fundamentals_employees`. `FundamentalsValidator`
    follows at Phase 5b.
Both layers are dormant in `run()` (the `fundamentals` command runs them). Earnings surprises are
`StepExtractPrices.run_earnings_surprises`; the notes zips are `StepExtractZips`.
"""

from omegaconf import DictConfig

from src.context import Context
from src.utils.step import Step


class StepExtractFundamentals(Step):
    def __init__(self, context: Context, config: DictConfig):
        super().__init__(context=context, config=config)

    def run(self, tickers: list[str], full: bool = False) -> None:

        # Both layers are dormant here; when re-enabled, the window is
        # `years_history = int(self._config.data_extract.years_history)` (`self._config`, the attribute
        # `Step.__init__` sets: `self.config` raised AttributeError on every real invocation of this step).

        # Per-filing SEC XBRL -> fundamentals_facts (+ fundamentals_employees from the same
        # 10-K prose), each KPI resolved from the filer's own calculation linkbase.
        # Append-only and accession-grain, so a nightly re-run is idempotent and an
        # amendment lands as its own row at its own filing date.
        #     fetch_fundamentals_sec(self._context, tickers=tickers, years_history=years_history, full=full)

        # ... then the publication-event replay over exactly those facts. Immediately after,
        # and never on its own schedule: the snapshot is only as fresh as the facts it reads,
        # and a nightly that fetched a 10-K without replaying it would leave the newest
        # filing invisible to every consumer. Refuses to overwrite an already-published row
        # (see `diff_against_stored`), so a resolution change surfaces here rather than in a
        # cube six months later.
        # build_fundamentals_history(self._context, tickers=tickers, rebuild_history=full)
        self._log.info("fundamentals: facts walk and history replay dormant in this step (%d ticker(s)); run `fundamentals`", len(tickers))
