"""
step_extract_fundamentals_sharadar.py
  (src/data_extract/transformers/step_extract_fundamentals_sharadar.py)
-----------------------------------------------------------------------
Sharadar fundamentals extraction -> SF1, S&P 500 membership and `sharadar_fundamentals_history`.

A SIBLING of `StepExtractFundamentals` (the SEC layer) rather than a part of it: the two producers
are independent and neither reads the other's tables.

The stages run in DEPENDENCY order, which is not negotiable: the fundamentals fetch
reads `currency` out of `sharadar_tickers` to enforce the USD assertion (D20) and raises if
that table is empty. `sharadar_tickers` and `sharadar_actions` are refreshed by the universe step.
"""

from __future__ import annotations

from omegaconf import DictConfig

from src.context import Context
from src.data_extract.utils.fundamentals_sharadar.fetch_sharadar import (
    fetch_sharadar_fundamentals,
    fetch_sharadar_sp500,
    predecessor_vendor_tickers,
)
from src.data_extract.utils.fundamentals_sharadar.sharadar_history import build_sharadar_history
from src.utils.step import Step


class StepExtractFundamentalsSharadar(Step):
    def __init__(self, context: Context, config: DictConfig) -> None:
        super().__init__(context=context, config=config)

    def run(self, tickers: list[str]) -> None:
        """Fetch configured SF1/membership windows, then rebuild affected vendor histories."""
        predecessors = [ticker for ticker in predecessor_vendor_tickers(self._context, tickers) if ticker not in set(tickers)]
        if predecessors:
            self._log.info("Sharadar SF1: + %d predecessor vendor ticker(s): %s", len(predecessors), ", ".join(predecessors))

        # A missing predecessor gets initial history once, then its own normal frontier.
        fetch_sharadar_fundamentals(self._context, tickers=[*tickers, *predecessors])
        fetch_sharadar_sp500(self._context)
        build_sharadar_history(self._context, tickers=tickers)
