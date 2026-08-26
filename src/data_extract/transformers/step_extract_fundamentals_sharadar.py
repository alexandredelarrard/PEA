"""
step_extract_fundamentals_sharadar.py
  (src/data_extract/transformers/step_extract_fundamentals_sharadar.py)
-----------------------------------------------------------------------
Sharadar fundamentals extraction -> the four vendor-shaped tables.

A SIBLING of `StepExtractFundamentals` (the SEC layer) rather than a part of it, and it runs
BEFORE it: the two producers are independent, and the merged `fundamentals_history` that
consumes both is field-block precedence -- Sharadar owns a declared set of columns for all
history, SEC owns the rest, and no column ever switches source mid-series (D11/D14).

The four fetchers run in DEPENDENCY order, which is not negotiable: the fundamentals fetch
reads `currency` out of `sharadar_tickers` to enforce the USD assertion (D20) and raises if
that table is empty.
"""

from omegaconf import DictConfig

from src.context import Context
from src.utils.step import Step
from src.data_extract.utils.fundamentals_sharadar.fetch_sharadar import (
    fetch_sharadar_actions, fetch_sharadar_fundamentals, fetch_sharadar_sp500,
    fetch_sharadar_tickers,
)


class StepExtractFundamentalsSharadar(Step):

    def __init__(self, context: Context, config: DictConfig):
        super().__init__(context=context, config=config)

    def run(self, tickers: list[str]) -> None:

        years = int(self._config.data_extract.sharadar_years_history)

        # 1. The entity dimension FIRST -- `permaticker`, `currency`, `category`. A full
        #    refresh: `isdelisted` / `lastquarter` mutate, so an append-only view goes stale.
        fetch_sharadar_tickers(self._context)

        # 2. SF1, all 112 columns as delivered, on the three AS-REPORTED dimensions. Its own
        #    history knob (`sharadar_years_history`), separate from `years_history`, because
        #    the two sources are limited by different things -- the SEC walk by patience,
        #    Sharadar by subscription tier. A ticker outside the subscription returns 403,
        #    costs one request and is counted, never retried.
        fetch_sharadar_fundamentals(self._context, tickers=tickers, years_history=years)

        # 3. Corporate actions: dividends, splits, spinoffs, acquisitions, relations.
        fetch_sharadar_actions(self._context, years_history=years)

        # 4. S&P 500 membership events. Ingested only -- `src/utils/universe.py` still
        #    resolves the universe from `sp500_tickers`, and the survivorship-bias fix that
        #    would consume this table is a separate task (D27).
        fetch_sharadar_sp500(self._context)
