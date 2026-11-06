"""Insider, beneficial ownership and short-flow features on the full price calendar."""

from __future__ import annotations

from typing import cast

import pandas as pd
from omegaconf import DictConfig

from src.constants.constants import SECONDARY_CLASS
from src.context import Context
from src.data_aggregate.utils.common import inputs as capital_inputs
from src.data_aggregate.utils.common.availability import InstitutionalAvailability
from src.data_aggregate.utils.common.class_lines import load_class_lines
from src.data_aggregate.utils.common.incremental import COLUMNS_CHANGED, PART_REFRESH_TRADING_DAYS, PartWindow, plan_window, write_part
from src.data_aggregate.utils.common.panel_merge import PanelMerger
from src.data_aggregate.utils.common.parts import part_for
from src.data_aggregate.utils.common.peers_io import load_peers_or_raise
from src.data_aggregate.utils.common.quotes import lookup_asof, terminal_basis_factor
from src.data_aggregate.utils.institutionals import frontiers as institutional_frontiers
from src.data_aggregate.utils.institutionals import inputs as institutional_inputs
from src.data_aggregate.utils.institutionals import short_flow_features as short_flow
from src.data_aggregate.utils.institutionals.insider_features import (
    EMISSION as INSIDER_EMISSION,
)
from src.data_aggregate.utils.institutionals.insider_features import (
    InsiderCoverage,
    build_insider_feature_panel,
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
from src.data_aggregate.utils.price.clean import apply_volume_scale
from src.data_aggregate.utils.price.frames import PriceFrames, load_price_frames, load_trading_calendar
from src.data_store.schema import Tables
from src.utils.price.adjustments import load_bugfix
from src.utils.step import Step


class StepCubeInstitutionals(Step):
    _FIELDS = ("close_split", "close_total", "volume", "close_pit", "shares_outstanding_pit", "mcap_pit", "security_basis")

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

        self._write_panel(df_panel, window)

    @staticmethod
    def _output_keys(rows: pd.DataFrame) -> pd.MultiIndex:
        keys = list(Tables.cube_part_institutionals.pk)
        if rows.columns.duplicated().any() or not set(keys).issubset(rows):
            raise ValueError("Institutional output has missing keys or repeated columns")
        dates = pd.to_datetime(rows["date"], errors="raise")
        if (
            rows[keys].isna().any().any()
            or not dates.eq(dates.dt.normalize()).all()
            or not rows["ticker"].map(lambda value: isinstance(value, str) and bool(value)).all()
        ):
            raise ValueError("Institutional output has invalid date/ticker keys")
        index = pd.MultiIndex.from_arrays([dates, rows["ticker"]], names=keys)
        if index.has_duplicates:
            raise ValueError("Institutional output has repeated date/ticker keys")
        return index

    def _write_panel(self, rows: pd.DataFrame, window: PartWindow) -> None:
        """Write the ordinary refresh tail in one transaction; a schema change rebuilds the part."""
        self._output_keys(rows)
        part = Tables.cube_part_institutionals
        rows = rows.astype(dict.fromkeys([column for column in rows if column not in part.pk], "float64"))
        with self._store.transaction() as store:
            count = write_part(store, part, rows, window, drop_empty=True)
            if count == COLUMNS_CHANGED:
                count = write_part(store, part, rows, PartWindow(None, None), drop_empty=True)
        self._log.info("Institutional output: refresh from %s; wrote %s rows", window.refresh_from, count)

    def _warmup(self) -> int:
        override = self._cfg.get("incremental", {}).get("warmup_trading_days")
        return int(override) if override is not None else self._part.warmup_trading_days

    def build_panel(self) -> pd.DataFrame:
        """Compute with full event history; run controls the independently rewritten tail."""

        # Load shared price and class inputs once.
        frames = load_price_frames(self._store, peers=load_peers_or_raise(self._context), fields=self._FIELDS, since=None)
        df_lines = load_class_lines(self._store)

        # Build each family sequentially so raw sources do not overlap in memory.
        merger = PanelMerger(anchor=frames.skeleton())
        merger.add(self._insider_panel(frames, df_lines), "insider-trading")
        merger.add(self._short_flow_panel(frames, df_lines), "short-flow")
        merger.add(self._ownership_panel(frames), "beneficial-ownership")

        # Keep declared inputs even when a source has no qualifying observations.
        fields = [*INSIDER_EMISSION, *short_flow.EMISSION, *OWNERSHIP_EMISSION]
        return merger.to_long().reindex(columns=["date", "ticker", *(f"f_{name}" for name in fields)])

    def _insider_panel(self, frames: PriceFrames, identity_intervals: pd.DataFrame | None) -> pd.DataFrame | None:

        # Load transactions and their completeness/title evidence before cleaning.
        source = institutional_inputs.load_insider_transactions(self._store, frames.universe)
        if source is None or source.transactions.empty:
            return None

        # load and clean roles per person
        df_roles = institutional_inputs.load_insider_role_evidence(self._store, source.transactions, frames.universe)

        # Preserve execution USD while cleaning source-unit prices and public versions.
        df_trades, diagnostics = clean_transactions(df_insider=source.transactions, df_valuation_close=frames.close_pit, df_roles=df_roles)
        if "source_first" not in diagnostics:
            return None

        # Put unclassed positions in terminal units through the saved basis; report source-day trades against the saved count.
        diagnostics["source_first"] = source.first_filed
        df_unpriced = diagnostics.get("unpriced_events", pd.DataFrame())
        df_positions = position_units(pd.concat([df_trades, df_unpriced], ignore_index=True), frames.security_basis)
        report_oversized_positions(df_trades, frames.shares_outstanding_pit)

        coverage = InsiderCoverage.from_availability(self._availability, source.complete_through, identity_intervals)
        return build_insider_feature_panel(frames, df_trades, df_positions, diagnostics, frames.mcap_pit, coverage)

    def _short_flow_panel(self, frames: PriceFrames, lines: pd.DataFrame | None) -> pd.DataFrame | None:
        # Load observed fallback only for companies without any adjudicated master evidence; clean the raw class tape once.
        known = set(lines["canonical_company"].dropna()) if lines is not None else set()
        df_classes = (
            lines.loc[lines["lineage_role"].eq(SECONDARY_CLASS) & lines["canonical_company"].isin(frames.universe)] if lines is not None else None
        )
        df_finra = institutional_inputs.load_finra_tenure(self._store, sorted(set(frames.universe) - known))
        df_short = capital_inputs.load_source(self._store, Tables.short_interest, frames.universe)
        df_fails, df_coverage = institutional_inputs.load_fails(self._store, frames.universe)
        df_bars = apply_volume_scale(institutional_inputs.load_class_bars(self._store, df_classes), self._bugfix)

        # Put FINRA, FTD and outstanding quantities in each ticker's last saved security units (`q * F_T / F_d`).
        idx = pd.DatetimeIndex(frames.trading_index).normalize().unique().sort_values()
        columns = pd.Index(frames.universe, name="ticker")
        df_tenure = short_flow._proven_tenure_mask(idx, columns, lines, df_finra)
        df_factor = terminal_basis_factor(cast(pd.DataFrame, frames.security_basis))
        if df_short is not None:
            factor = lookup_asof(df_factor, df_short["ticker"], df_short["date"])
            df_short = df_short.assign(short_volume=df_short["short_volume"] * factor, total_volume=df_short["total_volume"] * factor)
        df_shares_out = cast(pd.DataFrame, frames.shares_outstanding_pit) * df_factor

        # Yahoo/class tape is already vendor-restated to the last bar; apply its terminal class ratio once.
        df_tape = frames.volume
        df_supported = ~df_tape.lt(0) if df_tape is not None else None
        if df_classes is not None:
            df_class_volume, df_missing = secondary_class_volume(df_classes, df_bars, idx)
            df_tape = with_class_volume(df_tape, df_class_volume)
            df_class_supported = (~df_missing).reindex(index=idx, columns=columns, fill_value=True)
            df_supported = df_class_supported if df_supported is None else df_supported & df_class_supported
        capital = ShortFlowCapital(df_shares_out, df_tape, df_supported)
        settlement = publication = None
        if df_coverage is not None or df_fails is not None:
            df_source = df_fails if df_fails is not None else pd.DataFrame(columns=["date", "ticker", "fails_quantity", "period"])
            df_coverage = df_coverage if df_coverage is not None else df_source[["date", "period"]]
            df_source = df_source.assign(fails_quantity=df_source["fails_quantity"] * lookup_asof(df_factor, df_source["ticker"], df_source["date"]))
            settlement = prepare_fails_quantities(df_source, df_coverage, idx, columns, df_tenure)
            available = short_flow._ftd_available_dates(df_coverage, self._cfg.institutionals.ftd_availability_calendar_days)
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
