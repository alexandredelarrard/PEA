"""Who owns and shorts each name: 13F books and RegSHO short volume (the superinvestor roster is the manual `superinvestors`).
The EDGAR walks (insiders, 13D/13G, 8-K) are `StepExtractEdgar`; the zip quarters and fails-to-deliver are
`StepExtractZips`. The window is resolved here and passed into every fetcher. Consumes the identity built and
synced upstream (`StepExtractAllData`, the DAG's identity stage); never rebuilds it.
"""

from src.data_extract.utils.institutionals.fetch_13f import fetch_13f
from src.data_extract.utils.institutionals.fetch_short_interest import fetch_short_interest
from src.utils.step import Step


class StepExtractInstitutionals(Step):
    def run(self, tickers: list[str]) -> None:
        # One walk over every 13F-HR inside its configured filing bounds: every filer's complete book, as filed.
        fetch_13f(self._context)

        # The short side: FINRA RegSHO short volume, stored as filed.
        fetch_short_interest(self._context, tickers=tickers)
