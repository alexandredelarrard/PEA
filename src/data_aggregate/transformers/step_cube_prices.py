"""Build and publish adjusted prices and coherent historical security valuation inputs."""

from __future__ import annotations

import pandas as pd
from omegaconf import DictConfig

from src.context import Context
from src.data_aggregate.utils.common.incremental import COLUMNS_CHANGED, PART_REFRESH_TRADING_DAYS, PartWindow, plan_window, write_part
from src.data_aggregate.utils.common.parts import part_for
from src.data_aggregate.utils.price.checks import check_candidate, describe, earliest_price_change
from src.data_aggregate.utils.price.clean import apply_null_ret, clean_prices
from src.data_aggregate.utils.price.features import pit_fields, resolve_price_basis
from src.data_aggregate.utils.price.frames import ALL_FIELDS, frames_to_long, universe_columns
from src.data_aggregate.utils.price.inputs import load_price_inputs
from src.data_aggregate.utils.price.returns import compute_sector_returns
from src.data_store.schema import Tables
from src.utils.price.adjustments import load_bugfix
from src.utils.price.frames import daily_returns
from src.utils.price.levels import apply_level_bugfix, level_factor
from src.utils.step import Step


class StepCubePrices(Step):
    def __init__(self, context: Context, config: DictConfig) -> None:
        super().__init__(context=context, config=config)
        self._part = part_for(Tables.cube_part_prices)
        self._store = context.store
        self._bugfix = load_bugfix(context.config_dir)

    def run(self, full: bool = False) -> None:
        # Load the publication plan before payloads, then read full required evidence once.
        window, existing = self._plan_window(full)
        inputs = load_price_inputs(self._context, self._universe.tickers, self._bugfix)

        # Clean quote/total repairs and volume units on the authoritative calendar.
        wide, df_vendor = clean_prices(inputs, self._bugfix)

        # Merge canonical/predecessor events and validate repaired anchors once.
        names = universe_columns(self._universe.tickers, wide["close_split"])
        wide = {field: df_frame.reindex(columns=names) for field, df_frame in wide.items()}
        df_close = wide["close_split"]
        index = pd.DatetimeIndex(df_close.index)
        basis = resolve_price_basis(df_close, inputs, self._bugfix)

        # Compute one level history and the three valuation outputs on that same basis.
        df_factor = level_factor(index, names, basis.price, basis)
        df_factor = apply_level_bugfix(df_factor, self._bugfix, df_vendor, df_close).where(df_close.notna())
        quoted = pit_fields(df_close, df_factor, basis, inputs, self._bugfix, df_vendor)
        df_returns = daily_returns(wide["close_total"])

        # Null only registered returns, then calculate peer returns from the final frame.
        applied = apply_null_ret(df_returns, self._bugfix)
        candidate = {**wide, "ret": df_returns, "sector_ret": compute_sector_returns(df_returns, inputs.peers), "level_factor": df_factor, **quoted}
        df_rows = frames_to_long(candidate)

        # Check and report the complete candidate before touching the saved part.
        check_candidate(df_rows)
        self._log.info("%s; %d null-return repairs; %d candidate rows", describe(df_factor), applied, len(df_rows))

        # Publish one coherent part, including every changed historical prefix.
        self._write_candidate(df_rows, window, existing)

    def _plan_window(self, full: bool) -> tuple[PartWindow, list[str]]:
        """Promote schema changes before payload reads; retain the ordinary refresh cutoff."""
        existing = self._store.columns(Tables.cube_part_prices)
        expected = {*Tables.cube_part_prices.pk, *ALL_FIELDS}
        promote = full or set(existing) != expected
        dates = self._store.distinct(Tables.cube_part_prices, "date") if existing and not promote else []
        index = pd.DatetimeIndex(sorted(pd.to_datetime(pd.Series(dates)).unique())) if dates else None
        window = plan_window(
            self._store,
            Tables.cube_part_prices,
            full=promote,
            warmup=self._part.warmup_trading_days,
            trading_index=index,
            refresh=PART_REFRESH_TRADING_DAYS,
        )
        self._log.info(
            "Price publication: %s; full historical evidence is required for basis invalidation",
            "full/schema promotion" if window.is_full else f"refresh from {window.refresh_from}",
        )
        return window, existing

    def _write_candidate(self, df_rows: pd.DataFrame, window: PartWindow, existing: list[str]) -> None:
        """Compare saved history once, then use the existing non-atomic part writer."""
        earliest = None if window.is_full else earliest_price_change(self._store, df_rows, existing)
        cutoff = window.refresh_from
        if earliest is not None:
            cutoff = earliest if cutoff is None else min(cutoff, earliest)
        replacement = PartWindow(window.last, None, cutoff)
        self._log.info("Price history: earliest change %s; refresh %s -> %s", earliest, window.refresh_from, cutoff)
        count = write_part(self._store, Tables.cube_part_prices, df_rows, replacement, existing_columns=existing)
        if count == COLUMNS_CHANGED:
            raise ValueError("Price schema changed during publication; rebuild without concurrent writers")
