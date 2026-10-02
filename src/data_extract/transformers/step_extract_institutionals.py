"""
step_extract_institutionals.py (src/data_extract/transformers/step_extract_institutionals.py)
----------------------------------------------------------------------------------------------
Who owns, trades and shorts each name: 13F (one all-filer walk writing `sec13f_hr` and the roster
managers' complete books, then a per-CIK catch-up), the superinvestor roster, insiders, 13D/13G, 8-K,
RegSHO short volume and fails-to-deliver. The 8-K fetch must precede `StepExtractStructure`, whose
vote parser reads `sec_8k` Item 5.07; the window is resolved here and passed into every fetcher.
"""

from omegaconf import DictConfig

from src.context import Context
from src.data_extract.utils.common.bulk_cache import cache_dir
from src.data_extract.utils.common.edgar_driver import run_edgar_fetch
from src.data_extract.utils.common.entity_lineage import build_entity_lineage
from src.data_extract.utils.common.identity import load_identity
from src.data_extract.utils.common.symbol_tenure import build_symbol_tenure
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

        # 13F: one walk over every 13F-HR filed since max(filing_date) in sec13f_hr (CUSIP map
        # built inside), writing the S&P 500 slice AND the roster managers' complete books.
        fetch_13f(self._context, tickers=tickers, years_history=years_history)

        # Superinvestors roster: curated top managers (Dataroma) -> one dated snapshot row per
        # manager in `superinvestor_roster`, so membership stays point-in-time. AFTER the 13F
        # pull: resolution settles an ambiguous manager name on which candidate CIK actually
        # has rows in `sec13f_hr`.
        upsert_roster_snapshot(self._context)

        # Per-CIK catch-up of the managers' books from each stored frontier. AFTER the snapshot
        # above: a manager added today has no frontier and gets its whole window here. Takes the
        # window but NOT `tickers` -- having no universe filter is the entire point of it.
        fetch_13f_managers(self._context, years_history=years_history)

        # Insider history is quarterly bulk; the daily EDGAR pass immediately after it fills
        # only the open-quarter publication gap. Running bulk first moves that gap's floor.
        fetch_insider_transactions(self._context, tickers=tickers, years_history=years_history)
        fetch_insider_edgar(self._context, tickers=tickers, years_history=years_history)

        # activist stakes (SC 13D) then the passive ones (SC 13G), 13D first because it is
        # ~7x cheaper and a failure there is the cheaper one to discover. Same grain and column
        # names, so the escalation join across them is a plain union.
        run_edgar_fetch(self._context, tickers=tickers, years_history=years_history, fetch=SEC_13D_FETCH)
        run_edgar_fetch(self._context, tickers=tickers, years_history=years_history, fetch=SEC_13G_FETCH)

        # corporate events (8-K)
        run_edgar_fetch(self._context, tickers=tickers, years_history=years_history, fetch=SEC_8K_FETCH)

        # Refresh the two-axis identity dimension after the insider cache producer, then hand
        # one frozen resolver to the two symbol-only tapes at the end of the step.
        insider_cache = cache_dir(self._context, self.config.local.paths.insider_transactions)
        build_symbol_tenure(self._context, insider_cache, self._context.config_dir)
        build_entity_lineage(self._context, insider_cache, str(self._context.config_dir))
        identity = load_identity(self._context, refresh=True)

        # The short side: FINRA RegSHO short volume, then SEC settlement fails (mid-2009 on).
        fetch_short_interest(self._context, tickers=tickers, years_history=years_history, identity=identity)
        fetch_fails_to_deliver(self._context, tickers=tickers, years_history=years_history, identity=identity)
