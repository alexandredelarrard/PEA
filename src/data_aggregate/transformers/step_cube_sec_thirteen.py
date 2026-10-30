"""Load once, reconcile public13F books, calculate features and save one effective window."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import closing
from dataclasses import dataclass, replace

import pandas as pd
from omegaconf import DictConfig

from src.constants.constants import F13_SETTLE_TRADING_DAYS
from src.context import Context
from src.data_aggregate.utils.common import inputs as capital_inputs
from src.data_aggregate.utils.common.availability import InstitutionalAvailability
from src.data_aggregate.utils.common.incremental import COLUMNS_CHANGED, PART_REFRESH_TRADING_DAYS, PartWindow, plan_window, write_part
from src.data_aggregate.utils.common.level_basis import load_bugfix
from src.data_aggregate.utils.common.panel import build_peer_relative_panel
from src.data_aggregate.utils.common.panel_merge import PanelMerger
from src.data_aggregate.utils.common.parts import part_for
from src.data_aggregate.utils.common.price_frames import PriceFrames, load_trading_calendar
from src.data_aggregate.utils.sec_thirteen import inputs as thirteen_inputs
from src.data_aggregate.utils.sec_thirteen.holdings_clean import clean_public_batches
from src.data_aggregate.utils.sec_thirteen.institutional_books import iter_institutional_books, prepare_institutional_book_history
from src.data_aggregate.utils.sec_thirteen.institutional_features import EMISSION as INSTITUTIONAL_EMISSION
from src.data_aggregate.utils.sec_thirteen.institutional_features import compute_institutional_feature_history, project_institutional_features
from src.data_aggregate.utils.sec_thirteen.superinvestor_books import prepare_superinvestor_batches, prepare_superinvestor_book_history
from src.data_aggregate.utils.sec_thirteen.superinvestor_features import EMISSION as SUPERINVESTOR_EMISSION
from src.data_aggregate.utils.sec_thirteen.superinvestor_features import (
    SuperinvestorFeatureHistory,
    compute_superinvestor_feature_history,
    project_superinvestor_features,
)
from src.data_aggregate.utils.sec_thirteen.superinvestor_metrics import BehaviorMarket
from src.data_aggregate.utils.sec_thirteen.superinvestor_roster import load_manager_cohorts
from src.data_store.schema import Tables
from src.utils.price_adjustments import ShareBasis
from src.utils.step import Step
from src.utils.superinvestor_roster import filer_ciks


@dataclass(frozen=True)
class SecThirteenResources:
    """One loaded capital/security snapshot shared by both13F families."""

    frames: PriceFrames
    shares: pd.DataFrame | None
    basis: ShareBasis
    valuation_close: pd.DataFrame | None
    cusip_map: pd.DataFrame | None
    cohorts: Mapping[str, frozenset[str]] | None


@dataclass(frozen=True)
class PreparedSecThirteen:
    institutional: pd.DataFrame | None
    superinvestor: SuperinvestorFeatureHistory | None


class StepCubeSecThirteen(Step):
    _FIELDS = ("close_split", "level_factor")

    def __init__(self, context: Context, config: DictConfig) -> None:
        super().__init__(context=context, config=config)
        self._store = context.store
        self._cfg = config.build_cube
        self._part = part_for(Tables.cube_part_sec_thirteen)
        self._bugfix = load_bugfix(context.config_dir)
        cfg = config.get("data_availability")
        self._availability = InstitutionalAvailability.from_config(cfg) if cfg is not None else None

    def run(self, full: bool = False) -> None:
        
        # Plan the actual rewrite footprint before loading dependency payloads.
        calendar = load_trading_calendar(self._store)
        existing_columns = self._store.columns(Tables.cube_part_sec_thirteen)
        saved_dates = pd.DatetimeIndex(self._store.distinct(Tables.cube_part_sec_thirteen, "date"))
        window = plan_window(
            self._store,
            Tables.cube_part_sec_thirteen,
            full=False,
            warmup=self._warmup(),
            trading_index=calendar,
            refresh=PART_REFRESH_TRADING_DAYS,
        )
        if full:
            window = PartWindow(None, None)
        self._require_saved_calendar(saved_dates, calendar, window.refresh_from)

        # Schema fallback reuses the prepared full-history panel and performs one save.
        panel = self.build_panel(window, existing_columns=existing_columns)
        result = write_part(self._store, Tables.cube_part_sec_thirteen, panel, window, drop_empty=True, existing_columns=existing_columns)
        if result == COLUMNS_CHANGED:
            self._require_saved_calendar(saved_dates, calendar, None)
            write_part(self._store, Tables.cube_part_sec_thirteen, panel, PartWindow(None, None), drop_empty=True)

    @staticmethod
    def _require_saved_calendar(saved_dates: pd.DatetimeIndex, calendar: pd.DatetimeIndex, refresh_from: pd.Timestamp | None) -> None:
        rewrite_dates = saved_dates if refresh_from is None else saved_dates[saved_dates >= refresh_from]
        if not rewrite_dates.difference(calendar).empty:
            raise RuntimeError("SEC13F price calendar ends before the saved output frontier or omits a saved rewrite date; no rows may be replaced.")

    def _warmup(self) -> int:
        override = self._cfg.get("incremental", {}).get("warmup_trading_days")
        return int(override) if override is not None else self._part.warmup_trading_days

    def _load_resources(self) -> SecThirteenResources:
        # Read the shared dependencies once before any cleaning or feature work.
        frames = capital_inputs.load_full_price_frames(self._store, self._context, self._FIELDS)
        history = capital_inputs.load_source(self._store, Tables.superinvestor_roster)
        cusip_map = self._store.load(Tables.cusip_ticker_map, columns=["cusip", "ticker"], optional=True)
        shares, _, basis, valuation_close, _ = capital_inputs.load_share_inputs(self._context, frames, self._bugfix)

        # The ordered list is user-owned; membership never depends on portfolio prices.
        cohorts = None
        if history is not None:
            roster = filer_ciks(history.cik.dropna(), self._context.config_dir)
            cfg = self._cfg.get("institutionals", {}).get("superinvestor", {})
            cohorts = load_manager_cohorts(self._context.config_dir, roster, cfg.get("selection", {}).get("k", 20))
        return SecThirteenResources(frames, shares, basis, valuation_close, cusip_map, cohorts)

    def build_panel(self, window: PartWindow | None = None, *, existing_columns: Sequence[str] | None = None) -> pd.DataFrame:
        
        resources = self._load_resources()
        saved_columns = list(existing_columns)
        window = window or PartWindow(None, None)

        # Keep sparse full history for genuine lag comparisons and bounded daily projection.
        df_institutionals = self._institutional_features(resources)
        super_investors = self._superinvestor_history(resources)
        prepared = PreparedSecThirteen(df_institutionals, super_investors)
        self._require_saved_families(prepared, saved_columns)

        dates = resources.frames.trading_index
        output_dates = dates if window.since is None else dates[dates >= window.since]
        panels = self._project_panels(resources, prepared, output_dates)
        if window.is_full:
            return self._merge_panels(resources.frames, panels)

        # A schema change reprojects the same facts, never reloads or rebuilds books.
        supported = self._supported_columns(resources, prepared, output_dates, panels)
        if supported != set(saved_columns) - {"date", "ticker"}:
            panels = self._project_panels(resources, prepared, dates)
            return self._merge_panels(resources.frames, panels)
        output_frames = self._output_frames(resources.frames, output_dates)
        df_panel = self._merge_panels(output_frames, panels)
        if df_panel.empty:
            df_panel = output_frames.skeleton()
        for column in supported - set(df_panel.columns):
            df_panel[column] = pd.Series(index=df_panel.index, dtype="float32")
        return df_panel.reindex(columns=saved_columns)

    def _institutional_features(self, resources: SecThirteenResources) -> pd.DataFrame | None:
        packets = thirteen_inputs.iter_filing_vintages(self._store, until=resources.frames.trading_index.max())
        cfg = self._cfg.get("institutionals", {})
        if packets is None:
            return None

        # Preserve the existing broad cleaning, reconciliation and numerical equations.
        history = prepare_institutional_book_history(
            packets,
            resources.frames.trading_index,
            resources.cusip_map,
            resources.frames.universe,
            resources.valuation_close,
            cfg.get("value_basis"),
            settle_days=int(cfg.get("f13_settle_trading_days", F13_SETTLE_TRADING_DAYS)),
        )
        if history is None:
            return None
        compact, states = history
        splits = resources.basis.actual.rename(columns={"value": "ratio"})
        return compute_institutional_feature_history(iter_institutional_books(compact, states), splits=splits)

    def _superinvestor_history(self, resources: SecThirteenResources) -> SuperinvestorFeatureHistory | None:
        
        if resources.cohorts is None:
            return None
        
        packets = thirteen_inputs.iter_filing_vintages(
            self._store, resources.frames.trading_index.max(), sorted(resources.cohorts[""]), table=Tables.sec13f_manager_holdings
        )
        if packets is None:
            return None

        # Clean whole books once, retaining mapped off-universe positions and provenance.
        cfg = self._cfg.get("institutionals", {})
        age_days = int(cfg.get("superinvestor", {}).get("max_age_calendar_days", 90))
        with closing(clean_public_batches(packets, resources.cusip_map, None, resources.valuation_close, cfg.get("value_basis"))) as batches:
            prepared = prepare_superinvestor_batches(batches, resources.frames.trading_index)
            books = prepare_superinvestor_book_history(prepared, resources.frames.trading_index, age_days)
        if books is None:
            return None
        market = BehaviorMarket(resources.frames.trading_index, resources.valuation_close, resources.basis)
        return compute_superinvestor_feature_history(books, resources.cohorts, resources.frames.universe, market)

    @staticmethod
    def _output_frames(frames: PriceFrames, dates: pd.DatetimeIndex) -> PriceFrames:
        return replace(
            frames,
            trading_index=dates,
            close_split=frames.close_split.reindex(dates) if frames.close_split is not None else None,
            level_factor=frames.level_factor.reindex(dates) if frames.level_factor is not None else None,
        )

    @staticmethod
    def _require_saved_families(prepared: PreparedSecThirteen, saved_columns: Sequence[str]) -> None:
        failed = (("f_ic_inst_", prepared.institutional is None), ("f_ic_super_", prepared.superinvestor is None))
        for prefix, missing in failed:
            if missing and any(column.startswith(prefix) for column in saved_columns):
                raise RuntimeError(f"SEC13F source preparation failed for {prefix}; saved output is preserved.")

    def _project_panels(
        self, resources: SecThirteenResources, prepared: PreparedSecThirteen, dates: pd.DatetimeIndex
    ) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
        frames = self._output_frames(resources.frames, dates)
        broad = self._project_institutional_panel(prepared.institutional, frames, resources) if prepared.institutional is not None else None
        elite = None
        if prepared.superinvestor is not None:
            start = (
                self._availability.source_date(Tables.sec13f_manager_holdings)
                if self._availability is not None
                else resources.frames.trading_index.min()
            )
            elite = project_superinvestor_features(prepared.superinvestor, frames, start)
        return broad, elite

    def _project_institutional_panel(self, features: pd.DataFrame, frames: PriceFrames, resources: SecThirteenResources) -> pd.DataFrame:
        if features.empty:
            return pd.DataFrame(columns=["date", "ticker"])
        settle_days = int(self._cfg.get("institutionals", {}).get("f13_settle_trading_days", F13_SETTLE_TRADING_DAYS))
        fields = project_institutional_features(
            features,
            frames,
            resources.shares,
            resources.basis,
            settle_days,
            versioned=True,
            trading_calendar=resources.frames.trading_index,
        )
        emission = {name: INSTITUTIONAL_EMISSION[name] for name in fields}
        return build_peer_relative_panel(fields, frames.peers, emission=emission, availability=frames.availability)

    @staticmethod
    def _merge_panels(frames: PriceFrames, panels: tuple[pd.DataFrame | None, pd.DataFrame | None]) -> pd.DataFrame:
        merger = PanelMerger(anchor=frames.skeleton())
        merger.add(panels[0], "institutional (13F)")
        merger.add(panels[1], "superinvestor (13F)")
        return merger.to_long()

    @staticmethod
    def _feature_columns(panels: tuple[pd.DataFrame | None, pd.DataFrame | None]) -> set[str]:
        return set().union(*(set(panel.columns) - {"date", "ticker"} for panel in panels if panel is not None))

    @staticmethod
    def _possible_columns(resources: SecThirteenResources, prepared: PreparedSecThirteen) -> set[str]:
        fields: set[str] = set()
        broad = prepared.institutional
        if broad is not None and not broad.empty:
            fields.update(name for name in INSTITUTIONAL_EMISSION if name in broad and broad[name].notna().any())
            if resources.shares is not None and not resources.shares.empty:
                derived = {"inst_shares": "ic_inst_ownership_pct", "inst_value": "ic_inst_value_to_mcap"}
                fields.update(name for source, name in derived.items() if source in broad and broad[source].notna().any())
        if prepared.superinvestor is not None:
            fields.update(SUPERINVESTOR_EMISSION)
        return {"f_" + name for name in fields}

    def _supported_columns(
        self,
        resources: SecThirteenResources,
        prepared: PreparedSecThirteen,
        output_dates: pd.DatetimeIndex,
        panels: tuple[pd.DataFrame | None, pd.DataFrame | None],
    ) -> set[str]:
        supported = self._feature_columns(panels)
        unresolved = self._possible_columns(resources, prepared) - supported
        end = int(resources.frames.trading_index.searchsorted(output_dates[0])) if len(output_dates) else len(resources.frames.trading_index)
        span = max(len(output_dates), self._warmup() + PART_REFRESH_TRADING_DAYS + 1)

        # Probe historical support using the same prepared facts and family projectors.
        while unresolved and end:
            start = max(0, end - span)
            broad = prepared.institutional if any(column.startswith("f_ic_inst_") for column in unresolved) else None
            elite = prepared.superinvestor if any(column.startswith("f_ic_super_") for column in unresolved) else None
            probe = PreparedSecThirteen(broad, elite)
            probe_panels = self._project_panels(resources, probe, resources.frames.trading_index[start:end])
            supported.update(self._feature_columns(probe_panels))
            unresolved.difference_update(supported)
            end = start
        return supported
