"""Build and publish adjusted prices and coherent historical security valuation inputs."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import pandas as pd
from omegaconf import DictConfig

from src.context import Context
from src.data_aggregate.utils.common.incremental import PART_REFRESH_TRADING_DAYS, changed_tickers, plan_window, write_part
from src.data_aggregate.utils.common.parts import part_for
from src.data_aggregate.utils.price.checks import check_candidate, describe
from src.data_aggregate.utils.price.clean import apply_null_ret, clean_prices
from src.data_aggregate.utils.price.features import as_traded_volume, pit_fields, resolve_price_basis
from src.data_aggregate.utils.price.frames import frames_to_long, universe_columns
from src.data_aggregate.utils.price.inputs import load_price_inputs
from src.data_aggregate.utils.price.returns import compute_sector_returns
from src.data_store.schema import Tables
from src.utils.price.adjustments import load_bugfix
from src.utils.price.frames import daily_returns
from src.utils.price.levels import apply_level_bugfix, level_factor
from src.utils.step import Step
from src.utils.window import Offset, Window, WindowPlan, WindowPolicy, WindowRun


class StepCubePrices(Step):
    def __init__(self, context: Context, config: DictConfig) -> None:
        super().__init__(context=context, config=config)
        self._part = part_for(Tables.cube_part_prices)
        self._store = context.store
        self._bugfix = load_bugfix(context.config_dir)

    def run(self, full: bool = False, tickers: Sequence[str] = ()) -> None:
        # A ticker list rebuilds those tickers from scratch; `full` alone rebuilds the whole part.
        rebuild = self._universe.subset(tickers) if tickers else []
        inputs = load_price_inputs(self._context, self._universe.tickers, self._bugfix)

        # Clean quote/total repairs and volume units on the authoritative calendar.
        wide, df_vendor = clean_prices(inputs, self._bugfix)

        # Merge canonical/predecessor events and validate repaired anchors once.
        names = universe_columns(self._universe.tickers, wide["close_split"])
        wide = {field: df_frame.reindex(columns=names) for field, df_frame in wide.items()}
        df_close = wide["close_split"]
        index = pd.DatetimeIndex(df_close.index)
        window = self._plan_window(full and not rebuild, index)
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

        # Publish: whole part, or whole selected/changed tickers plus the others' tail.
        self._write_candidate(df_rows, window, rebuild)

    def _plan_window(self, full: bool, calendar: pd.DatetimeIndex) -> WindowPlan:
        """Resolve publication from the authoritative input calendar and selected frontier."""
        policy = WindowPolicy(Window(calendar[0], calendar[-1]), Offset(PART_REFRESH_TRADING_DAYS, "trading_sessions"), None)
        return plan_window(self._store, Tables.cube_part_prices, policy, calendar, full=full, scope=self._universe.tickers)

    def _write_candidate(self, df_rows: pd.DataFrame, window: WindowPlan, tickers: list[str]) -> None:
        """Publish the ordinary tail and separately scoped basis/explicit ticker recovery."""
        part = Tables.cube_part_prices
        if window.is_full:
            write_part(self._store, part, df_rows, window)
            return
        selected = set(dict(window.scope)["ticker"])
        rebuild = sorted(selected.intersection(tickers or changed_tickers(self._store, part, df_rows, window)))
        remaining = tuple(sorted(selected - set(rebuild)))
        with self._store.transaction() as store:
            if remaining and not tickers:
                write_part(store, part, df_rows, replace(window, scope=(("ticker", remaining),)))
            if rebuild:
                recovery = replace(
                    window,
                    write=window.requested,
                    run=WindowRun("recovery", "explicit ticker or changed historical price basis", window.run.as_of),
                    scope=(("ticker", tuple(rebuild)),),
                )
                write_part(store, part, df_rows, recovery)
        self._log.info("Price part: recovery for %s; ordinary tail from %s", rebuild, window.write.start)
