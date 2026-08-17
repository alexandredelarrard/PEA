"""
step_check_freshness.py  (src/data_extract/step_check_freshness.py)
-------------------------------------------------------------------
StepCheckFreshness — the data-drift / gap gate at the tail of the nightly extraction DAG. Reads the
latest observed date per source, compares it to the source's cadence expectation (daily .. yearly),
logs a grouped report and stores it on `self.report`. `run()` returns the report dict so the CLI can
emit it as JSON for the DAG to push to XCom (and colour the task RED when not up to date).
"""
from __future__ import annotations

from omegaconf import DictConfig

from src.context import Context
from src.utils.step import Step
from src.data_extract.utils.common.freshness import check_data_freshness


class StepCheckFreshness(Step):

    def __init__(self, context: Context, config: DictConfig):
        super().__init__(context=context, config=config)

    def run(self) -> dict:
        self.report = check_data_freshness(self._context, log=self._log)
        return self.report
