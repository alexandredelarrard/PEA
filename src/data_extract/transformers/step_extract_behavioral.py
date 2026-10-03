"""Behavioral alt-data extraction: earnings-call transcripts."""

from omegaconf import DictConfig

from src.context import Context
from src.data_extract.utils.behavioral.fetch_earnings_calls import fetch_earnings_calls
from src.utils.step import Step


class StepExtractBehavioral(Step):
    def __init__(self, context: Context, config: DictConfig):
        super().__init__(context=context, config=config)

    def run(self, tickers: list[str]) -> None:

        # Earnings-call transcripts -> earnings_call_sections.

        fetch_earnings_calls(self._context, tickers=tickers)
