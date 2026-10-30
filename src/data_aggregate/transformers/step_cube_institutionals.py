"""Insider, beneficial ownership and short-flow features on the full price calendar."""

from __future__ import annotations

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from src.constants.constants import SECONDARY_CLASS
from src.context import Context
from src.data_aggregate.utils.common import inputs as capital_inputs
from src.data_aggregate.utils.common.availability import InstitutionalAvailability
from src.data_aggregate.utils.common.incremental import PART_REFRESH_TRADING_DAYS, PartWindow, drop_empty_feature_rows, plan_window
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
        part = Tables.cube_part_institutionals
        keys = list(part.pk)
        existing = self._store.columns(part)
        if existing and not set(keys).issubset(existing):
            raise ValueError("Stored institutional output is missing date/ticker keys")
        stream = self._store.iter_load(part, columns=existing or list(rows.columns), order_by=keys)
        try:
            self._output_keys(rows)
            if rows.empty:
                raise ValueError("An empty institutional calculation cannot authorize historical replacement")
            fields = [column for column in rows if column not in keys]
            rows = rows.astype(dict.fromkeys(fields, "float64"))
            canonical = drop_empty_feature_rows(rows, keys, part)
            index = self._output_keys(canonical)
            values = canonical[fields].to_numpy(dtype="float64")
            seen = np.zeros(len(canonical), dtype=bool)
            same_schema = set(existing) == set(rows.columns)
            earliest: pd.Timestamp | None = None
            last_date: pd.Timestamp | None = None
            last_tickers: set[str] = set()
            compared = 0
            for chunk in stream:
                if set(chunk.columns) != set(existing):
                    raise ValueError("Stored institutional output changed schema inside its stream")
                stored_index = self._output_keys(chunk)
                if chunk.empty:
                    continue
                dates = pd.DatetimeIndex(stored_index.get_level_values("date"))
                if not dates.is_monotonic_increasing or (last_date is not None and dates[0] < last_date):
                    raise ValueError("Stored institutional output is not ordered by date")
                first_tickers = set(chunk.loc[dates == last_date, "ticker"]) if last_date is not None else set()
                if first_tickers & last_tickers:
                    raise ValueError("Stored institutional output repeats keys across chunks")
                last_tickers = set(chunk.loc[dates == dates[-1], "ticker"]) | (last_tickers if dates[-1] == last_date else set())
                last_date = dates[-1]
                positions = index.get_indexer(stored_index)
                matched = positions >= 0
                if seen[positions[matched]].any():
                    raise ValueError("Stored institutional output repeats candidate keys")
                seen[positions[matched]] = True
                changed = ~matched
                if same_schema:
                    old = chunk.loc[matched, fields].to_numpy(dtype="float64")
                    new = values[positions[matched]]
                    changed[matched] = ~((old == new) | (np.isnan(old) & np.isnan(new))).all(axis=1)
                    compared += len(old)
                if changed.any():
                    first = dates[changed].min()
                    earliest = first if earliest is None else min(earliest, first)
            if (~seen).any():
                first = pd.Timestamp(canonical.loc[~seen, "date"].min())
                earliest = first if earliest is None else min(earliest, first)
        finally:
            stream.close()
        cutoff = None if window.is_full or not same_schema else window.refresh_from
        if cutoff is not None and earliest is not None:
            cutoff = min(cutoff, earliest)
        tail = canonical if cutoff is None else canonical.loc[canonical["date"] >= cutoff]
        if cutoff is not None and (pd.to_datetime(tail["date"]) < cutoff).any():
            raise ValueError("Institutional replacement rows precede their corrected range")
        with self._store.transaction() as store:
            if cutoff is None:
                count = store.replace(part, tail)
            else:
                if set(store.columns(part)) != set(tail.columns):
                    raise ValueError("Institutional corrected range must retain the stored schema")
                count = store.append_tail(part, tail, cutoff, inclusive=True)
        self._log.info(
            "Institutional output compared %s rows; earliest change %s; refresh %s -> %s; wrote %s rows",
            compared,
            earliest,
            window.refresh_from,
            cutoff,
            count,
        )

    def _warmup(self) -> int:
        override = self._cfg.get("incremental", {}).get("warmup_trading_days")
        return int(override) if override is not None else self._part.warmup_trading_days

    def build_panel(self) -> pd.DataFrame:
        """Compute with full event history; run controls the independently rewritten tail."""

        # Load shared price, class and capital inputs once.
        frames = capital_inputs.load_full_price_frames(self._store, self._context, self._FIELDS)
        df_lines = self._store.load(
            Tables.security_master,
            columns=(*institutional_inputs.CLASS_LINE_COLUMNS, "security_id", "source", "source_symbol", "security_class", "lineage_role"),
            where={"source": "ftd"},
            optional=True,
        )
        df_classes = (
            df_lines.loc[df_lines["lineage_role"].eq(SECONDARY_CLASS) & df_lines["canonical_company"].isin(frames.universe)]
            if df_lines is not None
            else None
        )
        df_shares, _, basis, df_valuation_close, df_bars = capital_inputs.load_share_inputs(self._context, frames, self._bugfix, df_classes)
        quantity_basis = institutional_inputs.QuantityBasis.from_context(basis, pd.DatetimeIndex(frames.trading_index))

        # Build each family sequentially so raw sources do not overlap in memory.
        merger = PanelMerger(anchor=frames.skeleton())
        merger.add(self._insider_panel(frames, df_shares, quantity_basis, df_valuation_close, df_lines), "insider-trading")
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
        df_valuation_close: pd.DataFrame | None,
        identity_intervals: pd.DataFrame | None,
    ) -> pd.DataFrame | None:

        # Load transactions and their completeness/title evidence before cleaning.
        source = institutional_inputs.load_insider_transactions(self._store, frames.universe)
        if source is None or source.transactions.empty:
            return None

        # load and clean roles per person
        df_roles = institutional_inputs.load_insider_role_evidence(self._store, source.transactions, frames.universe)

        # Preserve execution USD while cleaning source-unit prices and public versions.
        df_trades, diagnostics = clean_transactions(
            df_insider=source.transactions, df_valuation_close=df_valuation_close, df_roles=df_roles, share_basis=quantity_basis.events
        )
        if "source_first" not in diagnostics:
            return None

        # get shares
        df_shares_out = institutional_inputs.daily_share_quantities(shares, pd.DatetimeIndex(frames.trading_index), quantity_basis)
        df_mcap = insider_market_cap(shares, frames.close_split, frames.level_factor)

        diagnostics["source_first"] = source.first_filed
        df_unpriced = diagnostics.get("unpriced_events", pd.DataFrame())
        df_positions = position_units(pd.concat([df_trades, df_unpriced], ignore_index=True), quantity_basis)
        report_oversized_positions(df_positions.iloc[: len(df_trades)], df_shares_out)

        coverage = InsiderCoverage.from_availability(self._availability, source.complete_through, identity_intervals)
        return build_insider_feature_panel(frames, df_trades, df_positions, diagnostics, df_mcap, coverage)

    def _short_flow_panel(
        self,
        frames: PriceFrames,
        shares: pd.DataFrame | None,
        lines: pd.DataFrame | None,
        bars: pd.DataFrame,
        quantity_basis: institutional_inputs.QuantityBasis,
    ) -> pd.DataFrame | None:
        # Load observed fallback only for companies without any adjudicated master evidence.
        known = set(lines["canonical_company"].dropna()) if lines is not None else set()
        df_finra = institutional_inputs.load_finra_tenure(self._store, sorted(set(frames.universe) - known))
        df_short = capital_inputs.load_source(self._store, Tables.short_interest, frames.universe)
        df_fails, df_coverage = institutional_inputs.load_fails(self._store, frames.universe)

        # Raw FINRA/FTD observations become ordinary shares on one explicit terminal basis.
        idx = pd.DatetimeIndex(frames.trading_index).normalize().unique().sort_values()
        columns = pd.Index(sorted(map(str, frames.universe)), name="ticker")
        df_tenure = short_flow._proven_tenure_mask(idx, columns, lines, df_finra)
        df_short = institutional_inputs.short_quantity_history(df_short, quantity_basis)
        df_shares_out = institutional_inputs.daily_share_quantities(shares, idx, quantity_basis)

        # Yahoo/class tape is already vendor-restated; apply its terminal class ratio once.
        df_tape = institutional_inputs.tape_share_quantities(frames.volume, quantity_basis)
        df_supported = ~df_tape.lt(0) if df_tape is not None else None
        if lines is not None:
            df_classes = lines.loc[lines["lineage_role"].eq(SECONDARY_CLASS) & lines["canonical_company"].isin(frames.universe)]
            df_class_volume, df_missing = secondary_class_volume(df_classes, bars, idx)
            df_tape = with_class_volume(df_tape, df_class_volume)
            df_class_supported = (~df_missing).reindex(index=idx, columns=columns, fill_value=True)
            df_supported = df_class_supported if df_supported is None else df_supported & df_class_supported
        capital = ShortFlowCapital(df_shares_out, df_tape, df_supported)
        settlement = publication = None
        if df_coverage is not None or df_fails is not None:
            df_source = df_fails if df_fails is not None else pd.DataFrame(columns=["date", "ticker", "fails_quantity", "period"])
            df_coverage = df_coverage if df_coverage is not None else df_source[["date", "period"]]
            settlement = prepare_fails_quantities(df_source, df_coverage, idx, columns, df_tenure, quantity_basis)
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
