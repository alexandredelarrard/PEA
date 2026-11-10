"""
step_extract_fundamentals.py  (src/data_extract/transformers/step_extract_fundamentals.py)
-----------------------------------------------------------------------------------------
The SEC fundamentals history: `run_history` replays the stored `sec_fundamentals_facts` (written by the
`StepExtractEdgar` facts walk) into `sec_fundamentals_history`, offline -- nothing in this module's import chain
reaches the network. `run_check` logs where the SEC and Sharadar histories disagree (read-only). `run()` runs the
history, then the check. Headcount is `StepExtractLlm.run_employees`.
"""

from omegaconf import DictConfig

from src.context import Context
from src.data_extract.utils.fundamentals.build_history import build_fundamentals_history
from src.data_extract.utils.fundamentals.check_sources import check_fundamentals_sources
from src.utils.step import Step


class StepExtractFundamentals(Step):
    def __init__(self, context: Context, config: DictConfig):
        super().__init__(context=context, config=config)

    def run_history(self, tickers: list[str], full: bool = False, verify: bool = False) -> None:
        """`sec_fundamentals_facts` -> `sec_fundamentals_history`; `full` deletes and rebuilds, `verify` replays under the drift guard."""
        build_fundamentals_history(self._context, tickers, rebuild_history=full, verify_history=verify)

    def run_check(self, tickers: list[str]) -> None:
        """Log the SEC vs Sharadar check on `tickers` (thresholds: `data_extract.fundamentals_check`); reads only, writes nothing."""
        limits = self._config.data_extract.fundamentals_check
        check_fundamentals_sources(
            self._context, tickers, gap_pct=float(limits.gap_pct), field_share=float(limits.field_share), top_tickers=int(limits.top_tickers)
        )

    def run(self, tickers: list[str], full: bool = False) -> None:
        self.run_history(tickers, full=full)
        self.run_check(tickers)
