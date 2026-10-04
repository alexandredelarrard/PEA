"""Who owns, trades and shorts each name: 13F, the superinvestor roster, insiders, 13D/13G, 8-K, RegSHO short
volume and fails-to-deliver. The 8-K fetch must precede `StepExtractStructure`, whose vote parser reads
`sec_8k` Item 5.07; the window is resolved here and passed into every fetcher.
"""

from omegaconf import DictConfig

from src.context import Context
from src.data_extract.utils.common.bulk_cache import cache_dir
from src.data_extract.utils.common.edgar_driver import run_edgar_fetch
from src.data_extract.utils.common.entity_lineage import build_entity_lineage
from src.data_extract.utils.common.identity import load_identity
from src.data_extract.utils.common.symbol_tenure import build_symbol_tenure, scan_form345_cache
from src.data_extract.utils.institutionals.fetch_8k_edgar import SEC_8K_FETCH
from src.data_extract.utils.institutionals.fetch_13d_edgar import SEC_13D_FETCH
from src.data_extract.utils.institutionals.fetch_13f import fetch_13f
from src.data_extract.utils.institutionals.fetch_13f_managers import fetch_13f_managers
from src.data_extract.utils.institutionals.fetch_13g_edgar import SEC_13G_FETCH
from src.data_extract.utils.institutionals.fetch_fails_to_deliver import fetch_fails_to_deliver
from src.data_extract.utils.institutionals.fetch_insider_edgar import fetch_insider_edgar
from src.data_extract.utils.institutionals.fetch_insider_transactions import fetch_insider_transactions
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

        # Point-in-time roster snapshot; after the 13F pull, which settles an ambiguous manager name on CIK row counts.
        upsert_roster_snapshot(self._context)

        # Per-CIK catch-up after the snapshot, so a manager added today gets its whole window; no universe filter by design.
        fetch_13f_managers(self._context, years_history=years_history)

        # Quarterly zips first (they add only the filings EDGAR lacks), then EDGAR reads the indexed filings after the last zip quarter.
        fetch_insider_transactions(self._context, tickers=tickers, years_history=years_history)
        fetch_insider_edgar(self._context, tickers=tickers, years_history=years_history)

        # Activist (SC 13D) then passive (SC 13G) stakes; same grain and column names, so they union.
        run_edgar_fetch(self._context, tickers=tickers, years_history=years_history, fetch=SEC_13D_FETCH)
        run_edgar_fetch(self._context, tickers=tickers, years_history=years_history, fetch=SEC_13G_FETCH)

        # Corporate events (8-K).
        run_edgar_fetch(self._context, tickers=tickers, years_history=years_history, fetch=SEC_8K_FETCH)

        # Rebuild the identity dimension from the Form 345 cache, then hand one frozen resolver to the symbol-only tapes.
        scan = scan_form345_cache(cache_dir(self._context, self.config.local.paths.insider_transactions))
        tenure = build_symbol_tenure(self._context, scan, self._context.config_dir)
        build_entity_lineage(self._context, tenure, scan.owner_pairs, str(self._context.config_dir))
        identity = load_identity(self._context, refresh=True)

        # The short side: FINRA RegSHO short volume, then SEC settlement fails (mid-2009 on).
        fetch_short_interest(self._context, tickers=tickers, years_history=years_history, identity=identity)
        fetch_fails_to_deliver(self._context, tickers=tickers, years_history=years_history, identity=identity)
