"""Universe step: the SEC and Sharadar ticker lists, then `tickers_universe` built or updated from `configs/universe.yml`.

The two list refreshes are independent (the DAG runs them in parallel); the sync reads both, so it runs last.
"""

from __future__ import annotations

import pandas as pd
from omegaconf import DictConfig

from src.context import Context
from src.data_extract.utils.fundamentals_sharadar.fetch_sharadar import fetch_sharadar_tickers
from src.data_extract.utils.identity.sec_tickers import download_sec_company_tickers
from src.utils.step import Step
from src.utils.tickers_universe.sync import sync


class StepExtractUniverse(Step):
    def __init__(self, context: Context, config: DictConfig, as_of: pd.Timestamp):
        super().__init__(context=context, config=config)
        self._as_of = as_of

    def run_sec_tickers(self) -> int:
        return download_sec_company_tickers(self._context)

    def run_sharadar_tickers(self) -> None:
        fetch_sharadar_tickers(self._context)

    def run_sync(self) -> int:
        return sync(self._context.store, self._config, self._as_of)

    def run(self) -> None:
        self.run_sec_tickers()
        self.run_sharadar_tickers()
        self.run_sync()
