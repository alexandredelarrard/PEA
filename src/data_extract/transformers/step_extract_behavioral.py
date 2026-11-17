"""Behavioral alt-data extraction: earnings-call transcripts."""

from src.data_extract.utils.behavioral.fetch_earnings_call_transcripts import extract_earnings_calls
from src.utils.step import Step


class StepExtractBehavioral(Step):
    def run(self, tickers: list[str]) -> None:
        """Extract scoped calls; aggregate time owns sentiment and embeddings."""
        extract_earnings_calls(self._context, tickers=tickers)
