"""Who owns and shorts each name: 13F books and RegSHO short volume (the superinvestor roster is the manual `superinvestors`).
The EDGAR walks (insiders, 13D/13G, 8-K) are `StepExtractEdgar`; the zip quarters and fails-to-deliver are
`StepExtractZips`. The window is resolved here and passed into every fetcher. Consumes the identity built and
synced upstream (`StepExtractAllData`, the DAG's identity stage); never rebuilds it.
"""

from omegaconf import DictConfig

from src.context import Context
from src.data_extract.utils.institutionals.fetch_13f import fetch_13f
from src.data_extract.utils.institutionals.fetch_short_interest import fetch_short_interest
from src.utils.step import Step


class StepExtractInstitutionals(Step):
    def __init__(self, context: Context, config: DictConfig):
        super().__init__(context=context, config=config)
        self.config = self._context.config

    def run(self, tickers: list[str]) -> None:
        years_history = int(self.config.data_extract.years_history)

        # One walk over every 13F-HR since the stored frontier: every filer's complete book, as filed.
        fetch_13f(self._context, tickers=tickers, years_history=years_history)

        # The short side: FINRA RegSHO short volume, stored as filed.
        fetch_short_interest(self._context, tickers=tickers)
