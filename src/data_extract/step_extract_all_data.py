"""Super step orchestrating the data-extraction sub-steps over one universe from `sp500_tickers`.

The universe is seeded via the S&P 500 scraper only when the table is empty. Order: identity stage
(Form 3/4/5 and Notes zip downloads -> symbol_tenure + entity_lineage -> propagation) -> prices ->
institutionals -> fundamentals (Sharadar then SEC) -> structure -> behavioral. Every domain step only
reads the propagated lineage. Institutionals must run before structure: the vote parser reads the
`sec_8k` Item 5.07 narratives institutionals stores.
"""

from omegaconf import DictConfig

from src.context import Context
from src.data_extract.identity_propagate import propagate_identity
from src.data_extract.transformers.step_extract_behavioral import StepExtractBehavioral
from src.data_extract.transformers.step_extract_fundamentals import StepExtractFundamentals
from src.data_extract.transformers.step_extract_fundamentals_sharadar import (
    StepExtractFundamentalsSharadar,
)
from src.data_extract.transformers.step_extract_institutionals import (
    StepExtractInstitutionals,
)
from src.data_extract.transformers.step_extract_prices import StepExtractPrices
from src.data_extract.transformers.step_extract_structure import StepExtractStructure
from src.data_extract.utils.common.bulk_cache import cache_dir
from src.data_extract.utils.common.entity_lineage import build_entity_lineage
from src.data_extract.utils.common.identity import load_identity
from src.data_extract.utils.common.symbol_tenure import build_symbol_tenure, scan_form345_cache
from src.data_extract.utils.fundamentals.fetch_financial_notes import download_financial_notes
from src.data_extract.utils.institutionals.fetch_insider_transactions import download_insider_transactions
from src.data_extract.utils.prices.fetch_tickers import get_sp500_tickers
from src.data_store.schema import Tables
from src.utils.step import Step
from src.utils.universe import load_universe_tickers


class StepExtractAllData(Step):
    def __init__(self, context: Context, config: DictConfig):

        super().__init__(context=context, config=config)

        self._prices = StepExtractPrices(context=context, config=config)
        self._institutionals = StepExtractInstitutionals(context=context, config=config)
        self._fundamentals_sharadar = StepExtractFundamentalsSharadar(context=context, config=config)
        self._fundamentals = StepExtractFundamentals(context=context, config=config)
        self._structure = StepExtractStructure(context=context, config=config)
        self._behavioral = StepExtractBehavioral(context=context, config=config)

    def _resolve_tickers(self) -> list[str]:
        if self._context.store.row_count(Tables.sp500_tickers) == 0:
            get_sp500_tickers(self._context)
            self._log.info("Extracted ticker list")
        universe = load_universe_tickers(self._context)
        self._log.info(f"Equity universe: {len(universe)} tickers from {Tables.sp500_tickers}")
        return universe

    def _refresh_identity(self, tickers: list[str]) -> None:
        """Cache the identity evidence, rebuild `symbol_tenure` + `entity_lineage`, then propagate the changes."""
        download_insider_transactions(self._context)
        download_financial_notes(self._context, years_history=int(self._config.data_extract.years_history))
        scan = scan_form345_cache(cache_dir(self._context, self._config.local.paths.insider_transactions))
        tenure = build_symbol_tenure(self._context, scan, self._context.config_dir)
        build_entity_lineage(self._context, tenure, scan.owner_pairs, str(self._context.config_dir))
        propagate_identity(self._context, tickers, identity=load_identity(self._context, refresh=True))

    def run(self) -> None:
        tickers = self._resolve_tickers()

        self._refresh_identity(tickers)
        self._prices.run(tickers=tickers)
        self._institutionals.run(tickers=tickers)
        self._fundamentals_sharadar.run(tickers=tickers, full=False)
        self._fundamentals.run(tickers=tickers, full=False)
        self._structure.run(tickers=tickers)
        self._behavioral.run(tickers=tickers)
