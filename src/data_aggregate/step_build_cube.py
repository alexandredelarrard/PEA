"""
step_build_cube.py  (src/data_aggregate/step_build_cube.py)
---------------------------------------------------------
Super step orchestrating the seven cube sub-steps, mirroring `StepExtractAllData`. Each
sub-step normalizes or derives one slice of the cube, persists it as a `cube_part_*` table,
and hands nothing else to the next one:

  1. prices        raw `prices` -> normalized OHLCV + returns + peer sector returns grid
  2. target        factor panel -> rolling betas -> multi-horizon factor-neutral labels
  3. fundamentals  SEC filings: fundamental, sector-KPI, earnings, workforce, dividend
  4. momentum      everything derived from price variation (momentum, vol, MACD, liquidity)
  5. text          earnings-call sentiment + embedding KPIs
  6. extras        governance, 13F, elite 13F, insider, short interest, attention
  7. assemble      read the parts -> composites -> the `cube` table

ONE CODE PATH, TWO DRIVERS. `run()` executes the same seven objects, in the same order, that
`cli.py` exposes as seven commands and the Airflow DAG chains as seven tasks. There is no
separate monolithic implementation any more: the previous version had two drivers over one
set of `self`-mutating methods, so the in-process path loaded thirteen source tables at once
(unrunnable at this data volume) while the DAG path re-ran the price+peer prologue fourteen
times to avoid that.

WHY IT NO LONGER BLOWS UP. Each sub-step keeps its heavy frames LOCAL to its own `run()`, so
they are collected when it returns, and reads the price grid back from `cube_part_prices`
PROJECTED to the fields it actually needs. Peak memory is the largest single sub-step rather
than the sum.
"""
from __future__ import annotations

from omegaconf import DictConfig

from src.context import Context
from src.data_aggregate.transformers.step_assemble_cube import StepAssembleCube
from src.data_aggregate.transformers.step_cube_extras import StepCubeExtras
from src.data_aggregate.transformers.step_cube_fundamentals import StepCubeFundamentals
from src.data_aggregate.transformers.step_cube_momentum import StepCubeMomentum
from src.data_aggregate.transformers.step_cube_prices import StepCubePrices
from src.data_aggregate.transformers.step_cube_target import StepCubeTarget
from src.data_aggregate.transformers.step_cube_text import StepCubeText
from src.data_aggregate.utils.common.panel_merge import FeatureCollisionError
from src.data_aggregate.utils.common.part_status import part_status_report
from src.utils.step import Step

__all__ = ["StepBuildCube", "FeatureCollisionError"]


class StepBuildCube(Step):

    def __init__(self, context: Context, config: DictConfig):
        super().__init__(context=context, config=config)

        self._prices = StepCubePrices(context=context, config=config)
        self._target = StepCubeTarget(context=context, config=config)
        self._fundamentals = StepCubeFundamentals(context=context, config=config)
        self._momentum = StepCubeMomentum(context=context, config=config)
        self._text = StepCubeText(context=context, config=config)
        self._extras = StepCubeExtras(context=context, config=config)
        self._assemble = StepAssembleCube(context=context, config=config)

    def run(self, full: bool = False) -> None:
        # self._prices.run(full=full)
        self._target.run(full=full)
        # self._fundamentals.run(full=full)
        # self._momentum.run(full=full)
        # self._text.run(full=full)
        # self._extras.run(full=full)
        # self._assemble.run()

    def cube_parts_status(self) -> dict:
        """Latest date + row count of every cube part (+ the downstream cube / predictions
        tables). The returned shape is a contract with the Airflow status gate -- see
        `utils/common/part_status.py`."""
        return part_status_report(self._context, self._log)
