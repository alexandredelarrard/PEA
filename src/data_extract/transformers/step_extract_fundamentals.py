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

from omegaconf import DictConfig

from src.context import Context
from src.utils.step import Step
from src.data_extract.utils.fundamentals.fetch_earnings_surprises import fetch_earnings_surprises
from src.data_extract.utils.fundamentals.fetch_fundamentals_sec import fetch_fundamentals_sec
from src.data_extract.utils.fundamentals.build_history import build_fundamentals_history
from src.data_extract.utils.prices.fetch_insider_transactions import fetch_insider_transactions
from src.data_extract.utils.fundamentals.fetch_financial_notes import fetch_financial_notes


class StepExtractFundamentals(Step):

    def __init__(self, context: Context, config: DictConfig):
        super().__init__(context=context, config=config)

    def run(self, tickers: list[str], full : bool=False) -> None:

        # Per-filing SEC XBRL -> fundamentals_facts (+ fundamentals_employees from the same
        # 10-K prose), each KPI resolved from the filer's own calculation linkbase.
        # Append-only and accession-grain, so a nightly re-run is idempotent and an
        # amendment lands as its own row at its own filing date.
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
        build_fundamentals_history(self._context, tickers=tickers, rebuild_history=full)

        # earnings surprises
        fetch_earnings_surprises(self._context, tickers=tickers)

        # Officer / director / 10%-owner transactions from the Insider Transactions sets:
        fetch_insider_transactions(self._context, tickers=tickers)

        # Footnote (notes) pension detail + note TEXT from the Financial Statement
        # AND Notes data sets -> notes_num / notes_text. Heavy (~26GB back-fill at
        # notes_years_history=15); own cache dir + config knob. (fetch_fails_to_deliver
        # lives in StepExtractPrices, alongside the other price/settlement signals.)
        fetch_financial_notes(self._context, tickers=tickers)
