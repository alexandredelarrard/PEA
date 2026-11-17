"""Build and publish adjusted prices and coherent historical security valuation inputs."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
from omegaconf import DictConfig

from src.context import Context
from src.data_aggregate.utils.common.incremental import changed_tickers, plan_window, write_part
from src.data_aggregate.utils.price.checks import check_candidate, describe
from src.data_aggregate.utils.price.clean import apply_null_ret, clean_prices
from src.data_aggregate.utils.price.features import as_traded_volume, pit_fields, resolve_price_basis
from src.data_aggregate.utils.price.frames import frames_to_long, universe_columns
from src.data_aggregate.utils.price.inputs import load_price_inputs
from src.data_aggregate.utils.price.returns import compute_sector_returns
from src.data_store.schema import Table, Tables
from src.utils.price.adjustments import load_bugfix
from src.utils.price.frames import daily_returns
from src.utils.price.levels import apply_level_bugfix, level_factor
from src.utils.step import Step
from src.utils.window import WindowPlan


class StepCubePrices(Step):
    def __init__(self, context: Context, config: DictConfig) -> None:
        super().__init__(context=context, config=config)
        self._store = context.store
        self._bugfix = load_bugfix(context.config_dir)

    def run(self, tickers: Sequence[str] = ()) -> dict[Table, WindowPlan]:
        # Explicit ticker recovery keeps the full cross-section for price/peer computations.
        rebuild = self._universe.subset(tickers) if tickers else []
        inputs = load_price_inputs(self._context, self._universe.tickers, self._bugfix)

        # Clean quote/total repairs and volume units on the authoritative calendar.
        wide, df_vendor = clean_prices(inputs, self._bugfix)

        # Merge canonical/predecessor events and validate repaired anchors once.
        names = universe_columns(self._universe.tickers, wide["close_split"])
        wide = {field: df_frame.reindex(columns=names) for field, df_frame in wide.items()}
        df_close = wide["close_split"]
        index = pd.DatetimeIndex(df_close.index)
        window = plan_window(self._context, Tables.cube_part_prices, index, rebuild or None)
        if window is None:
            return {}
        basis = resolve_price_basis(df_close, inputs, self._bugfix)

        # Compute one level history and the three valuation outputs on that same basis.
        df_factor = level_factor(index, names, basis.price, basis)
        df_factor = apply_level_bugfix(df_factor, self._bugfix, df_vendor, df_close).where(df_close.notna())
        quoted = pit_fields(df_close, df_factor, basis, inputs, self._bugfix, df_vendor)
        quoted["volume_pit"] = as_traded_volume(wide["volume"], quoted["security_basis"], basis)
        df_returns = daily_returns(wide["close_total"])

        # Null only registered returns, then calculate peer returns from the final frame.
        applied = apply_null_ret(df_returns, self._bugfix)
        candidate = {**wide, "ret": df_returns, "sector_ret": compute_sector_returns(df_returns, inputs.peers), "level_factor": df_factor, **quoted}
        df_rows = frames_to_long(candidate)

        # Check and report the complete candidate before touching the saved part.
        check_candidate(df_rows)
        self._log.info("%s; %d null-return repairs; %d candidate rows", describe(df_factor), applied, len(df_rows))

        # Publish only the resolved Window and explicitly selected ticker scope.
        return {Tables.cube_part_prices: self._write_candidate(df_rows, window)}

    def _write_candidate(self, df_rows: pd.DataFrame, window: WindowPlan) -> WindowPlan:
        """Require explicit root recovery before replacing older price history."""
        part = Tables.cube_part_prices
        columns = self._store.columns(part)
        if columns and set(columns) != set(df_rows.columns) and not window.is_full:
            raise ValueError("Price columns changed; an explicit full retained-schema rebuild is required")
        if not window.is_full:
            changed = changed_tickers(self._store, part, df_rows, window)
            if changed:
                raise ValueError(f"Historical price changes for {changed} require an explicit full cube rebuild; no price rows were changed")
        write_part(self._store, part, df_rows, window)
        return window
