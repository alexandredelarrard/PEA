"""Insider, beneficial ownership and short-flow features on the full price calendar."""

from __future__ import annotations

import pandas as pd
from omegaconf import DictConfig

from src.constants.constants import SECONDARY_CLASS
from src.context import Context
from src.data_aggregate.utils.common import inputs as capital_inputs
from src.data_aggregate.utils.common.availability import InstitutionalAvailability
from src.data_aggregate.utils.common.incremental import COLUMNS_CHANGED, PART_REFRESH_TRADING_DAYS, PartWindow, plan_window, write_part
from src.data_aggregate.utils.common.level_basis import load_bugfix
from src.data_aggregate.utils.common.panel_merge import PanelMerger
from src.data_aggregate.utils.common.parts import part_for
from src.data_aggregate.utils.common.price_frames import PriceFrames, load_trading_calendar
from src.data_aggregate.utils.institutionals import frontiers as institutional_frontiers
from src.data_aggregate.utils.institutionals import inputs as institutional_inputs
from src.data_aggregate.utils.institutionals.insider_capital import InsiderCapital
from src.data_aggregate.utils.institutionals.insider_features import build_insider_feature_panel
from src.data_aggregate.utils.institutionals.ownership_features import build_ownership_feature_panel
from src.data_aggregate.utils.institutionals.short_flow_features import build_short_flow_feature_panel, secondary_class_volume
from src.data_store.schema import Tables
from src.utils.price_adjustments import ShareBasis
from src.utils.step import Step


class StepCubeInstitutionals(Step):
    _FIELDS = ("close_split", "close_total", "volume", "level_factor")

    def __init__(self, context: Context, config: DictConfig) -> None:
        super().__init__(context=context, config=config)
        self._store = context.store
        self._cfg = config.build_cube
        self._part = part_for(Tables.cube_part_institutionals)
        self._bugfix = load_bugfix(context.config_dir)
        cfg = config.get("data_availability")
        self._availability = InstitutionalAvailability.from_config(cfg) if cfg is not None else None

    def run(self, full: bool = False) -> None:
        # Plan the saved tail before loading the full feature dependencies.
        trading_index = load_trading_calendar(self._store)
        window = plan_window(
            self._store,
            Tables.cube_part_institutionals,
            full=full,
            warmup=self._warmup(),
            trading_index=trading_index,
            refresh=PART_REFRESH_TRADING_DAYS,
        )

        # Compute once; history-dependent features need the complete calendar.
        df_panel = self.build_panel()

        # Replace an obsolete schema with the same completed panel.
        result = write_part(self._store, Tables.cube_part_institutionals, df_panel, window, drop_empty=True)
        if result == COLUMNS_CHANGED:
            write_part(self._store, Tables.cube_part_institutionals, df_panel, PartWindow(None, None), drop_empty=True)

    def _warmup(self) -> int:
        override = self._cfg.get("incremental", {}).get("warmup_trading_days")
        return int(override) if override is not None else self._part.warmup_trading_days

    def build_panel(self) -> pd.DataFrame:
        """Compute with full event history; run controls the independently rewritten tail."""
        # Load shared price, class and capital inputs once.
        frames = capital_inputs.load_full_price_frames(self._store, self._context, self._FIELDS)
        df_lines = self._store.load(
            Tables.security_master,
            columns=institutional_inputs.CLASS_LINE_COLUMNS,
            where={"lineage_role": SECONDARY_CLASS, "canonical_company": list(frames.universe)},
            optional=True,
        )
        df_shares, df_splits, basis, df_valuation_close, df_bars = capital_inputs.load_share_inputs(self._context, frames, self._bugfix, df_lines)

        # Build each family sequentially so raw sources do not overlap in memory.
        merger = PanelMerger(anchor=frames.skeleton())
        merger.add(self._insider_panel(frames, df_shares, share_basis=basis, valuation_close=df_valuation_close), "insider-trading")
        merger.add(self._short_flow_panel(frames, df_shares, df_splits, df_lines, df_bars, share_basis=basis), "short-flow")
        merger.add(self._ownership_panel(frames), "beneficial-ownership")

        # Merge once on observed price keys; the writer owns tail projection.
        return merger.to_long()

    def _insider_panel(
        self,
        frames: PriceFrames,
        shares: pd.DataFrame | None,
        *,
        share_basis: ShareBasis | None = None,
        valuation_close: pd.DataFrame | None = None,
    ) -> pd.DataFrame | None:
        last_session = pd.DatetimeIndex(frames.trading_index).max() if len(frames.trading_index) else None

        # Load transactions and their completeness/title evidence before cleaning.
        df_insider = institutional_inputs.load_insider_transactions(self._store, frames.universe)
        if df_insider is None or df_insider.empty:
            return None
        complete_through = institutional_frontiers.schedule_complete_through(self._store, Tables.insider_transactions, last_session)
        df_roles = institutional_inputs.load_insider_role_evidence(self._store, df_insider)

        # Clean, calculate and mask through the family's production builder.
        return build_insider_feature_panel(
            frames,
            df_insider,
            capital=InsiderCapital(shares, valuation_close, share_basis),
            availability=self._availability,
            complete_through=complete_through,
            role_evidence=df_roles,
        )

    def _short_flow_panel(
        self,
        frames: PriceFrames,
        shares: pd.DataFrame | None,
        splits: pd.DataFrame | None,
        lines: pd.DataFrame | None,
        bars: pd.DataFrame,
        *,
        share_basis: ShareBasis,
    ) -> pd.DataFrame | None:
        cache_dir = self._context.paths["DATA_STORE"] / self._context.config.local.paths.fails_deliver

        # Load lineage, RegSHO and global FTD coverage once.
        df_symbol_tenure, df_ticker_ciks = institutional_inputs.load_symbol_lineage(self._store, frames.universe)
        df_short = capital_inputs.load_source(self._store, Tables.short_interest, frames.universe)
        df_fails, df_coverage = institutional_inputs.load_fails(self._store, frames.universe)

        # Prepare class-aware tape denominators and the observed ZIP inventory.
        df_class_volume = secondary_class_volume(lines, bars, frames.trading_index)[0] if lines is not None else None
        stored_periods = sorted(df_coverage["period"].unique()) if df_coverage is not None else None

        # Compute both flow families with their existing publication clocks.
        return build_short_flow_feature_panel(
            frames,
            df_short,
            fails_history=df_fails,
            ftd_coverage=df_coverage,
            ftd_cache_dir=cache_dir,
            ftd_stored_periods=stored_periods,
            shares_out_history=shares,
            splits=splits,
            symbol_tenure=df_symbol_tenure,
            ticker_ciks=df_ticker_ciks,
            class_volume=df_class_volume,
            share_basis=share_basis,
        )

    def _ownership_panel(self, frames: PriceFrames) -> pd.DataFrame | None:
        last_session = pd.DatetimeIndex(frames.trading_index).max() if len(frames.trading_index) else None
        activist_halflife = self._decay_halflife("act")
        ownership_halflife = self._decay_halflife("bo")

        # Load the two schedules and independent global completeness frontiers.
        df_13d = capital_inputs.load_source(self._store, Tables.sec_13d, frames.universe)
        df_13g = capital_inputs.load_source(self._store, Tables.sec_13g, frames.universe)
        complete_through_13d = institutional_frontiers.schedule_complete_through(self._store, Tables.sec_13d, last_session)
        complete_through_13g = institutional_frontiers.schedule_complete_through(self._store, Tables.sec_13g, last_session)

        # Canonicalize events and apply schedule/identity masks in the builder.
        return build_ownership_feature_panel(
            frames,
            df_13d,
            df_13g,
            decay_halflife_act=activist_halflife,
            decay_halflife_bo=ownership_halflife,
            availability=self._availability,
            complete_through_13d=complete_through_13d,
            complete_through_13g=complete_through_13g,
        )

    def _decay_halflife(self, family: str) -> float:
        cfg = self._cfg.get("institutionals", {}).get("decay_halflife", {})
        return float(cfg.get(family, 63))
