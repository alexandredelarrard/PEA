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
from src.data_aggregate.utils.institutionals import short_flow_features as short_flow
from src.data_aggregate.utils.institutionals.insider_features import (
    EMISSION as INSIDER_EMISSION,
)
from src.data_aggregate.utils.institutionals.insider_features import (
    InsiderCoverage,
    build_insider_feature_panel,
    insider_market_cap,
    report_oversized_positions,
)
from src.data_aggregate.utils.institutionals.insider_positions import position_units
from src.data_aggregate.utils.institutionals.insider_quality import clean_transactions
from src.data_aggregate.utils.institutionals.ownership_features import (
    EMISSION as OWNERSHIP_EMISSION,
)
from src.data_aggregate.utils.institutionals.ownership_features import (
    build_ownership_feature_panel,
)
from src.data_aggregate.utils.institutionals.short_flow_features import (
    FtdPublication,
    ShortFlowCapital,
    build_short_flow_feature_panel,
    prepare_fails_quantities,
    secondary_class_volume,
    with_class_volume,
)
from src.data_store.schema import Tables
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
        df_panel = self.build_panel(full=full)

        # Replace an obsolete schema with the same completed panel.
        result = write_part(self._store, Tables.cube_part_institutionals, df_panel, window, drop_empty=True)
        if result == COLUMNS_CHANGED:
            write_part(self._store, Tables.cube_part_institutionals, df_panel, PartWindow(None, None), drop_empty=True)

    def _warmup(self) -> int:
        override = self._cfg.get("incremental", {}).get("warmup_trading_days")
        return int(override) if override is not None else self._part.warmup_trading_days

    def build_panel(self, full: bool = False) -> pd.DataFrame:
        """Compute with full event history; run controls the independently rewritten tail."""
        # Load shared price, class and capital inputs once.
        frames = capital_inputs.load_full_price_frames(self._store, self._context, self._FIELDS)
        df_lines = self._store.load(
            Tables.security_master,
            columns=institutional_inputs.CLASS_LINE_COLUMNS,
            where={"lineage_role": SECONDARY_CLASS, "canonical_company": list(frames.universe)},
            optional=True,
        )
        df_shares, _, basis, df_valuation_close, df_bars = capital_inputs.load_share_inputs(self._context, frames, self._bugfix, df_lines)
        quantity_basis = institutional_inputs.QuantityBasis.from_context(basis, pd.DatetimeIndex(frames.trading_index))

        # Build each family sequentially so raw sources do not overlap in memory.
        merger = PanelMerger(anchor=frames.skeleton())
        merger.add(self._insider_panel(frames, df_shares, quantity_basis, df_valuation_close), "insider-trading")
        merger.add(self._short_flow_panel(frames, df_shares, df_lines, df_bars, quantity_basis), "short-flow")
        merger.add(self._ownership_panel(frames), "beneficial-ownership")

        # Keep declared inputs even when a source has no qualifying observations.
        fields = [*INSIDER_EMISSION, *short_flow.EMISSION, *OWNERSHIP_EMISSION]
        return merger.to_long().reindex(columns=["date", "ticker", *(f"f_{name}" for name in fields)])

    def _insider_panel(
        self,
        frames: PriceFrames,
        shares: pd.DataFrame | None,
        quantity_basis: institutional_inputs.QuantityBasis,
        valuation_close: pd.DataFrame | None,
    ) -> pd.DataFrame | None:
        # Load transactions and their completeness/title evidence before cleaning.
        source = institutional_inputs.load_insider_transactions(self._store, frames.universe)
        if source is None or source.transactions.empty:
            return None
        df_insider = source.transactions
        df_roles = institutional_inputs.load_insider_role_evidence(self._store, df_insider)

        # Preserve execution USD while cleaning source-unit prices and public versions.
        df_trades, diagnostics = clean_transactions(
            df_insider, as_traded_quotes=valuation_close, role_evidence=df_roles, share_basis=quantity_basis.events
        )
        if "source_first" not in diagnostics:
            return None
        diagnostics["source_first"] = source.first_filed
        df_unpriced = diagnostics.get("unpriced_events", pd.DataFrame())
        df_positions = position_units(pd.concat([df_trades, df_unpriced], ignore_index=True), quantity_basis)
        df_shares_out = institutional_inputs.daily_share_quantities(shares, pd.DatetimeIndex(frames.trading_index), quantity_basis)
        report_oversized_positions(df_positions.iloc[: len(df_trades)], df_shares_out)
        df_mcap = insider_market_cap(shares, frames.close_split, frames.level_factor)
        coverage = InsiderCoverage.from_availability(self._availability, source.complete_through)
        return build_insider_feature_panel(frames, df_trades, df_positions, diagnostics, df_mcap, coverage)

    def _short_flow_panel(
        self,
        frames: PriceFrames,
        shares: pd.DataFrame | None,
        lines: pd.DataFrame | None,
        bars: pd.DataFrame,
        quantity_basis: institutional_inputs.QuantityBasis,
    ) -> pd.DataFrame | None:
        cache_dir = self._context.paths["DATA_STORE"] / self._context.config.local.paths.fails_deliver

        # Load lineage, RegSHO and global FTD coverage once.
        df_symbol_tenure, df_ticker_ciks = institutional_inputs.load_symbol_lineage(self._store, frames.universe)
        df_short = capital_inputs.load_source(self._store, Tables.short_interest, frames.universe)
        df_fails, df_coverage = institutional_inputs.load_fails(self._store, frames.universe)

        # Raw FINRA/FTD observations become ordinary shares on one explicit terminal basis.
        idx = pd.DatetimeIndex(frames.trading_index).normalize().unique().sort_values()
        columns = pd.Index(sorted(map(str, frames.universe)), name="ticker")
        df_tenure = short_flow._proven_tenure_mask(idx, columns, df_symbol_tenure, df_ticker_ciks)
        df_short = institutional_inputs.short_quantity_history(df_short, quantity_basis)
        df_shares_out = institutional_inputs.daily_share_quantities(shares, idx, quantity_basis)

        # Yahoo/class tape is already vendor-restated; apply its terminal class ratio once.
        df_tape = institutional_inputs.tape_share_quantities(frames.volume, quantity_basis)
        df_supported = None
        if lines is not None:
            df_class_volume, df_missing = secondary_class_volume(lines, bars, idx)
            df_tape = with_class_volume(df_tape, df_class_volume)
            df_supported = ~df_missing
        capital = ShortFlowCapital(df_shares_out, df_tape, df_supported)
        settlement = publication = None
        if df_coverage is not None or df_fails is not None:
            df_source = df_fails if df_fails is not None else pd.DataFrame(columns=["date", "ticker", "fails_quantity", "period"])
            df_coverage = df_coverage if df_coverage is not None else df_source[["date", "period"]]
            settlement = prepare_fails_quantities(df_source, df_coverage, idx, columns, df_tenure, quantity_basis)
            available = short_flow._ftd_available_dates(df_coverage, cache_dir, stored_periods=sorted(df_coverage["period"].unique()))
            publication = FtdPublication(df_coverage, available)
        return build_short_flow_feature_panel(frames, df_short, settlement, capital, publication, df_tenure)

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
