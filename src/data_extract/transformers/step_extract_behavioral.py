"""Behavioral alt-data extraction: earnings-call transcripts."""

from omegaconf import DictConfig

from src.context import Context
from src.data_extract.utils.behavioral.fetch_earnings_call_transcripts import extract_earnings_calls
from src.utils.step import Step


class StepExtractBehavioral(Step):
    def __init__(self, context: Context, config: DictConfig):
        super().__init__(context=context, config=config)

    def run(self, tickers: list[str]) -> None:

        # Earnings-call transcripts -> earnings_call_sections; the split, FinBERT sentiment,
        # text KPIs and embeddings are built at aggregate time from these paragraphs.
        extract_earnings_calls(self._context, self._config.earnings_calls, tickers=tickers)
