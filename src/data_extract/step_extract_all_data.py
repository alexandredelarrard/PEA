"""Super step orchestrating the data-extraction sub-steps over one universe from `sp500_tickers`.

The universe is seeded via the S&P 500 scraper only when the table is empty. Order: zip downloads (Form 3/4/5,
Notes, FTD, SEC current tickers) -> `StepExtractIdentity` (Notes and FTD captures,
symbol_tenure + entity_lineage + security_master -> propagation) -> zip ingests (insider zip quarters, notes,
FTD) -> prices (+ earnings surprises) -> institutionals (13F, roster, managers, short interest) ->
Sharadar -> fundamentals -> EDGAR (insider, 13D, 13G, 8-K, filing text, DEF 14A ECD) -> LLM (DEF 14A, 8-K votes)
-> behavioral. Every domain step only reads the propagated lineage. The LLM step is built only once the EDGAR step
has run, so a missing API key fails from there on; employees stay on their own command (`fundamentals-employees`).
"""

from omegaconf import DictConfig

from src.context import Context
from src.data_extract.transformers.step_extract_behavioral import StepExtractBehavioral
from src.data_extract.transformers.step_extract_edgar import EdgarOptions, StepExtractEdgar
from src.data_extract.transformers.step_extract_fundamentals import StepExtractFundamentals
from src.data_extract.transformers.step_extract_fundamentals_sharadar import (
    StepExtractFundamentalsSharadar,
)
from src.data_extract.transformers.step_extract_identity import IdentityOptions, StepExtractIdentity
from src.data_extract.transformers.step_extract_institutionals import (
    StepExtractInstitutionals,
)
from src.data_extract.transformers.step_extract_llm import LlmOptions, StepExtractLlm
from src.data_extract.transformers.step_extract_prices import StepExtractPrices
from src.data_extract.transformers.step_extract_zips import StepExtractZips, ZipsOptions
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
        self._behavioral = StepExtractBehavioral(context=context, config=config)

    def _resolve_tickers(self) -> list[str]:
        if self._context.store.row_count(Tables.sp500_tickers) == 0:
            get_sp500_tickers(self._context)
            self._log.info("Extracted ticker list")
        universe = load_universe_tickers(self._context)
        self._log.info(f"Equity universe: {len(universe)} tickers from {Tables.sp500_tickers}")
        return universe

    def run(self) -> None:
        tickers = self._resolve_tickers()

        zips = StepExtractZips(self._context, self._config, ZipsOptions(tickers=tickers))
        zips.run_download()
        StepExtractIdentity(self._context, self._config, IdentityOptions(tickers=tickers)).run()
        zips.run_ingest()

        # run each extract big block
        self._prices.run(tickers=tickers)
        self._institutionals.run(tickers=tickers, all_tickers=tickers)
        self._fundamentals_sharadar.run(tickers=tickers, full=False)
        self._fundamentals.run(tickers=tickers, full=False)
        StepExtractEdgar(self._context, self._config, EdgarOptions(tickers=tickers)).run()
        # The LLM passes read what the EDGAR step stored (`sec_8k` Item 5.07); votes after DEF 14A. Built here, so a
        # missing API key fails only from this point on.
        llm = StepExtractLlm(self._context, self._config, LlmOptions(tickers=tickers))
        llm.run_def14a()
        llm.run_8k_votes()
        self._behavioral.run(tickers=tickers)
