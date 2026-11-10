"""Who owns and shorts each name: 13F, the superinvestor roster, the 13F managers' books and RegSHO short volume.
The EDGAR walks (insiders, 13D/13G, 8-K) are `StepExtractEdgar`; the zip quarters and fails-to-deliver are
`StepExtractZips`. The window is resolved here and passed into every fetcher. Consumes the identity built and
propagated upstream (`StepExtractAllData`, the DAG's identity stage); never rebuilds it.
"""

from omegaconf import DictConfig

from src.context import Context
from src.data_extract.utils.institutionals.fetch_13f import fetch_13f
from src.data_extract.utils.institutionals.fetch_13f_managers import fetch_13f_managers
from src.data_extract.utils.institutionals.fetch_short_interest import fetch_short_interest
from src.data_extract.utils.institutionals.fetch_superinvestors import upsert_roster_snapshot
from src.utils.step import Step


class StepExtractInstitutionals(Step):
    def __init__(self, context: Context, config: DictConfig):
        super().__init__(context=context, config=config)
        self.config = self._context.config

    def run(self, tickers: list[str]) -> None:
        years_history = int(self.config.data_extract.years_history)

        # One walk over every 13F-HR since the stored frontier: the S&P 500 slice and the roster managers' complete books.
        fetch_13f(self._context, tickers=tickers, years_history=years_history)

        # Point-in-time roster snapshot, written only on change; after the 13F pull, whose filings are the activity gate's evidence.
        upsert_roster_snapshot(self._context)

        # Per-CIK catch-up after the snapshot, so a manager added today gets its whole window; no universe filter by design.
        fetch_13f_managers(self._context, years_history=years_history)

        # The short side: FINRA RegSHO short volume, stored as filed.
        fetch_short_interest(self._context, tickers=tickers)
