"""Universe step: the SEC and Sharadar ticker lists, the identity evidence (Sharadar actions, SEC submissions metadata),
then `universe_tickers` built or updated from `configs/universe.yml`.

The four refreshes are independent (the DAG runs them in parallel); the sync reads the ticker lists, so it runs last, and
the identity step reads all of them after it.
"""

from __future__ import annotations

import pandas as pd
from omegaconf import DictConfig

from src.context import Context
from src.data_extract.utils.common.bulk_cache import cache_dir
from src.data_extract.utils.fundamentals_sharadar.fetch_sharadar import fetch_sharadar_actions, fetch_sharadar_tickers
from src.data_extract.utils.identity.sec_profile import refresh_sec_cik_profile
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

    def run_sharadar_actions(self) -> int:
        """Every Sharadar corporate action of the configured window, replacing `sharadar_actions`."""
        return fetch_sharadar_actions(self._context, years_history=int(self._config.data_extract.sharadar_years_history), as_of=self._as_of)

    def run_sec_submissions(self) -> int:
        """The bulk SEC `submissions.zip` (once per run date) parsed into `sec_cik_profile`, one row per CIK."""
        return refresh_sec_cik_profile(self._context, cache_dir(self._context, self._config.local.paths.sec_submissions), self._as_of)

    def run_sync(self) -> int:
        return sync(self._context.store, self._config, self._as_of)

    def run(self) -> None:
        self.run_sec_tickers()
        self.run_sharadar_tickers()
        self.run_sharadar_actions()
        self.run_sec_submissions()
        self.run_sync()
