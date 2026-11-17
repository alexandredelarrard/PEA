"""Insider, beneficial ownership and short-flow features on the full price calendar."""

from __future__ import annotations

from typing import cast

import pandas as pd
from omegaconf import DictConfig

from src.constants.constants import SECONDARY_CLASS
from src.context import Context
from src.data_aggregate.utils.common import inputs as capital_inputs
from src.data_aggregate.utils.common.availability import InstitutionalAvailability
from src.data_aggregate.utils.common.incremental import plan_window, write_part
from src.data_aggregate.utils.common.panel_merge import PanelMerger
from src.data_aggregate.utils.common.peers_io import load_peers_or_raise
from src.data_aggregate.utils.common.quotes import terminal_basis_factor, terminal_rows
from src.data_aggregate.utils.institutionals import frontiers as institutional_frontiers
from src.data_aggregate.utils.institutionals import inputs as institutional_inputs
from src.data_aggregate.utils.institutionals import short_flow_features as short_flow
from src.data_aggregate.utils.institutionals.insider_features import (
    EMISSION as INSIDER_EMISSION,
)
from src.data_aggregate.utils.institutionals.insider_features import (
    InsiderCoverage,
    build_insider_feature_panel,
)
from src.data_aggregate.utils.institutionals.insider_positions import position_units
from src.data_aggregate.utils.institutionals.insider_quality import clean_transactions
from src.data_aggregate.utils.institutionals.ownership_features import (
    EMISSION as OWNERSHIP_EMISSION,
)
from src.data_aggregate.utils.institutionals.ownership_features import (
    SUBJECT_COLUMNS,
    ScheduleSource,
    SubjectIdentity,
    build_ownership_feature_panel,
)
from src.data_aggregate.utils.institutionals.short_flow_features import (
    FtdPublication,
    FtdSource,
    ShortFlowCapital,
    build_short_flow_feature_panel,
    prepare_fails_quantities,
    secondary_class_volume,
    with_class_volume,
)
from src.data_aggregate.utils.price.clean import apply_volume_scale
from src.data_aggregate.utils.price.frames import PriceFrames, load_price_frames, load_trading_calendar
from src.data_store.schema import Table, Tables
from src.utils.data_policy import publication_policy
from src.utils.identity_tables import read_identity
from src.utils.price.adjustments import load_bugfix
from src.utils.step import Step
from src.utils.window import Window, WindowPlan


class StepCubeInstitutionals(Step):
    _FIELDS = ("close_split", "close_total", "volume", "close_pit", "shares_outstanding_pit", "mcap_pit", "security_basis")

    def __init__(self, context: Context, config: DictConfig) -> None:
        super().__init__(context=context, config=config)
        self._store = context.store
        self._cfg = config.data_aggregate
        self._bugfix = load_bugfix(context.config_dir)
        self._insider_lag = int(publication_policy(config, Tables.insider_transactions).lag.value)
        self._availability = InstitutionalAvailability.from_config(config)

    def run(self) -> dict[Table, WindowPlan]:
        # Plan the saved tail before loading the full feature dependencies.
        trading_index = load_trading_calendar(self._context)
        window = plan_window(self._context, Tables.cube_part_institutionals, trading_index)
        if window is None:
            return {}

        # Compute once; history-dependent features need the complete calendar.
        df_panel = self.build_panel(window.read, trading_index)

        self._write_panel(df_panel, window)
        return {Tables.cube_part_institutionals: window}

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

    def _write_panel(self, rows: pd.DataFrame, window: WindowPlan) -> None:
        """Publish the resolved dates/keys; a schema change requires an explicit rebuild."""
        self._output_keys(rows)
        part = Tables.cube_part_institutionals
        rows = rows.astype(dict.fromkeys([column for column in rows if column not in part.pk], "float64"))
        count = write_part(self._store, part, rows, window, drop_empty=True)
        self._log.info("Institutional output: refresh from %s; wrote %s rows", window.write.start, count)

    def _fill_unknown_units(self, df_factor: pd.DataFrame) -> pd.DataFrame:
        """A ticker with no basis on any date (units unverified) converts at `unknown_unit_factor`, with a warning.

        The factor feeds every path (positions, short volume, FTD, shares): a level ratio against the unconverted tape (turnover,
        `ic_ftd_to_adv20`) is off by the quote-unit ratio for an ADR."""
        unknown = df_factor.columns[df_factor.isna().all()]
        if len(unknown):
            factor = float(self._cfg.sec_thirteen.unknown_unit_factor)
            self._log.warning(
                "units unknown for %d ticker(s) %s: terminal-unit factor %s used, values may be off by the quote-unit ratio",
                len(unknown),
                list(unknown),
                factor,
            )
            df_factor = df_factor.copy()
            df_factor[unknown] = factor
        return df_factor

    def build_panel(self, window: Window, calendar: pd.DatetimeIndex) -> pd.DataFrame:
        """Compute with full event history; run controls the independently rewritten tail."""

        # Load shared price and class inputs once.
        frames = load_price_frames(self._context, peers=load_peers_or_raise(self._context), calendar=calendar, fields=self._FIELDS, window=window)
        df_lines = institutional_inputs.load_ftd_lines(self._store)
        df_lineage = read_identity(self._store, Tables.universe_cik_symbol, columns=list(institutional_inputs.LINEAGE_COLUMNS), optional=True)
        df_factor = terminal_basis_factor(cast(pd.DataFrame, frames.security_basis))
        df_factor = self._fill_unknown_units(df_factor)

        # Build each family sequentially so raw sources do not overlap in memory.
        merger = PanelMerger(anchor=frames.skeleton())
        merger.add(self._insider_panel(frames, df_lines, df_factor), "insider-trading")
        merger.add(self._short_flow_panel(frames, df_lines, df_lineage, df_factor), "short-flow")
        merger.add(self._ownership_panel(frames, df_lines, df_lineage), "beneficial-ownership")

        # Keep declared inputs even when a source has no qualifying observations.
        emission = {**INSIDER_EMISSION, **short_flow.EMISSION, **OWNERSHIP_EMISSION}
        columns = [column for name, mode in emission.items() for column in (f"f_{name}", *([f"f_{name}_vs_peers"] if mode == "raw+peers" else []))]
        return merger.to_long().reindex(columns=["date", "ticker", *columns])

    def _insider_panel(self, frames: PriceFrames, identity_intervals: pd.DataFrame | None, df_factor: pd.DataFrame) -> pd.DataFrame | None:

        # Load transactions and their completeness/title evidence before cleaning.
        source = institutional_inputs.load_insider_transactions(self._context, frames.universe, identity_intervals)
        if source is None or source.transactions.empty:
            return None
        source = institutional_inputs.lag_insider_source(
            source, self._insider_lag, pd.DatetimeIndex(frames.trading_index).normalize().unique().sort_values()
        )

        # load and clean roles per person
        df_roles = institutional_inputs.load_insider_role_evidence(self._context, source.transactions, frames.universe, frames.trading_index)

        # Preserve execution USD while cleaning source-unit prices and public versions.
        df_trades, diagnostics = clean_transactions(
            df_insider=source.transactions,
            df_valuation_close=frames.close_pit,
            df_roles=df_roles,
            df_shares_outstanding=frames.shares_outstanding_pit,
        )
        if "source_first" not in diagnostics:
            return None

        # Put unclassed positions in terminal units through the saved basis.
        diagnostics["source_first"] = source.first_filed
        df_unpriced = diagnostics.get("unpriced_events", pd.DataFrame())
        df_positions = position_units(pd.concat([df_trades, df_unpriced], ignore_index=True), df_factor)
        df_flows = diagnostics["other_flows"]
        diagnostics["other_flows"] = position_units(df_flows.assign(transaction_day=df_flows["transaction_day"].fillna(df_flows["day"])), df_factor)

        coverage = InsiderCoverage.from_availability(self._availability, source.complete_through, identity_intervals)
        return build_insider_feature_panel(frames, df_trades, df_positions, diagnostics, frames.mcap_pit, coverage, self._universe.sector_map)

    def _short_flow_panel(
        self, frames: PriceFrames, lines: pd.DataFrame | None, lineage: pd.DataFrame | None, df_factor: pd.DataFrame
    ) -> pd.DataFrame | None:
        # Build the short-volume series and its lineage-fallback evidence and the FTD series from the as-filed lines; clean the raw class tape once.
        df_classes = (
            lines.loc[lines["lineage_role"].eq(SECONDARY_CLASS) & lines["canonical_company"].isin(frames.universe)] if lines is not None else None
        )
        df_short, df_finra = institutional_inputs.load_short_volume(self._context, frames.universe, lines, lineage)
        df_fails, df_coverage = institutional_inputs.load_fails(self._context, frames.universe, lines, self._bugfix)
        df_bars = apply_volume_scale(institutional_inputs.load_class_bars(self._context, df_classes), self._bugfix)

        # Put FINRA, FTD and outstanding quantities in each ticker's last saved security units (`q * F_T / F_d`).
        idx = pd.DatetimeIndex(frames.trading_index).normalize().unique().sort_values()
        columns = pd.Index(frames.universe, name="ticker")
        df_tenure = short_flow._proven_tenure_mask(idx, columns, lines, df_finra)
        if df_short is not None:
            df_short = terminal_rows(df_short, ("short_volume", "total_volume"), df_factor)
        df_shares_out = cast(pd.DataFrame, frames.shares_outstanding_pit) * df_factor

        # Yahoo/class tape is already vendor-restated to the last bar; apply its terminal class ratio once.
        df_tape = frames.volume
        df_supported = ~df_tape.lt(0) if df_tape is not None else None
        if df_classes is not None:
            df_class_volume, df_missing = secondary_class_volume(df_classes, df_bars, idx, df_tape)
            df_tape = with_class_volume(df_tape, df_class_volume)
            df_class_supported = (~df_missing).reindex(index=idx, columns=columns, fill_value=True)
            df_supported = df_class_supported if df_supported is None else df_supported & df_class_supported
        capital = ShortFlowCapital(df_shares_out, df_tape, df_supported)
        settlement = publication = None
        if df_coverage is not None or df_fails is not None:
            df_source = df_fails if df_fails is not None else pd.DataFrame(columns=["date", "ticker", "fails_quantity", "period"])
            df_coverage = df_coverage if df_coverage is not None else df_source[["date", "period"]]
            df_source = terminal_rows(df_source, ("fails_quantity",), df_factor)
            settlement = prepare_fails_quantities(df_source, df_coverage, idx, columns, df_tenure)
            available = short_flow._ftd_available_dates(
                df_coverage, int(publication_policy(self._config, Tables.sec_fails_to_deliver_security).lag.value)
            )
            publication = FtdPublication(df_coverage, available)
        fails = FtdSource(settlement, publication) if settlement is not None and publication is not None else None
        return build_short_flow_feature_panel(frames, df_short, fails, capital, self._config, df_tenure)

    def _ownership_panel(self, frames: PriceFrames, lines: pd.DataFrame | None, lineage: pd.DataFrame | None) -> pd.DataFrame | None:
        last_session = pd.DatetimeIndex(frames.trading_index).max() if len(frames.trading_index) else None

        # Load the two schedules and independent global completeness frontiers.
        df_13d = capital_inputs.load_source(self._context, Tables.sec_13d, frames.universe, markers=True, columns=SUBJECT_COLUMNS)
        df_13g = capital_inputs.load_source(self._context, Tables.sec_13g, frames.universe, markers=True, columns=SUBJECT_COLUMNS)
        identity = SubjectIdentity.from_tables(lineage, lines) if lineage is not None and lines is not None else None
        if identity is None:
            self._log.warning("lineage or CUSIP lines absent -> 13D/13G filings are counted without the subject-side identity filter")
        complete_through_13d = institutional_frontiers.schedule_complete_through(self._context, Tables.sec_13d, last_session)
        complete_through_13g = institutional_frontiers.schedule_complete_through(self._context, Tables.sec_13g, last_session)

        # Count filings per schedule and apply each schedule's coverage mask in the builder.
        return build_ownership_feature_panel(
            frames,
            (ScheduleSource(Tables.sec_13d, df_13d, complete_through_13d), ScheduleSource(Tables.sec_13g, df_13g, complete_through_13g)),
            self._config,
            availability=self._availability,
            identity=identity,
        )
