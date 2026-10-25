"""Broad institutional and elite-manager 13F features from public accession books."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace
from functools import partial

import numpy as np
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
from src.data_aggregate.utils.sec_thirteen.holdings_clean import attach_tickers, clean_public_holdings
from src.data_aggregate.utils.sec_thirteen.institutional_books import iter_institutional_books, prepare_institutional_book_history
from src.data_aggregate.utils.sec_thirteen.institutional_features import (
    EMISSION as INSTITUTIONAL_EMISSION,
)
from src.data_aggregate.utils.sec_thirteen.institutional_features import (
    compute_institutional_feature_history,
    project_institutional_features,
)
from src.data_aggregate.utils.sec_thirteen.manager_selection import roster_members_on, select_managers
from src.data_aggregate.utils.sec_thirteen.superinvestor_features import (
    EMISSION as SUPERINVESTOR_EMISSION,
)
from src.data_aggregate.utils.sec_thirteen.superinvestor_features import (
    append_unknown_superinvestor_snapshot,
    compute_superinvestor_levels,
    iter_manager_feature_inputs,
    manager_selection_weights,
    project_superinvestor_features,
    public_state,
    superinvestor_contributions,
)
from src.data_aggregate.utils.sec_thirteen.superinvestor_roster import first_snapshot_date, roster_as_of
from src.data_store.schema import Tables
from src.utils.institutional_vintages import prepare_book_vintages
from src.utils.price_adjustments import ShareBasis
from src.utils.step import Step
from src.utils.superinvestor_roster import filer_ciks


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

        panel = self.build_panel(window, existing_columns=existing_columns)
        result = write_part(self._store, Tables.cube_part_sec_thirteen, panel, window, drop_empty=True, existing_columns=existing_columns)
        if result == COLUMNS_CHANGED:
            self._require_saved_calendar(saved_dates, calendar, None)
            write_part(self._store, Tables.cube_part_sec_thirteen, panel, PartWindow(None, None), drop_empty=True)

    @staticmethod
    def _require_saved_calendar(saved_dates: pd.DatetimeIndex, calendar: pd.DatetimeIndex, refresh_from: pd.Timestamp | None) -> None:
        """Preserve saved sessions absent from the calendar inside the actual deletion footprint."""
        rewrite_dates = saved_dates
        if refresh_from is not None:
            rewrite_dates = saved_dates[saved_dates >= refresh_from]
        if not rewrite_dates.difference(calendar).empty:
            raise RuntimeError("SEC13F price calendar ends before the saved output frontier or omits a saved rewrite date; no rows may be replaced.")

    def _warmup(self) -> int:
        override = self._cfg.get("incremental", {}).get("warmup_trading_days")
        return int(override) if override is not None else self._part.warmup_trading_days

    def build_panel(self, window: PartWindow | None = None, *, existing_columns: Sequence[str] | None = None) -> pd.DataFrame:
        """Prepare full dependency history once, then project only the effective output dates."""
        frames = capital_inputs.load_full_price_frames(self._store, self._context, self._FIELDS)
        history = capital_inputs.load_source(self._store, Tables.superinvestor_roster)
        cusip_map = self._store.load(Tables.cusip_ticker_map, columns=["cusip", "ticker"], optional=True)
        shares, _, basis, valuation_close, _ = capital_inputs.load_share_inputs(self._context, frames, self._bugfix)
        saved_columns = self._store.columns(Tables.cube_part_sec_thirteen) if existing_columns is None else list(existing_columns)
        window = window or PartWindow(None, None)
        session_positions = pd.Series(np.arange(len(frames.trading_index)), index=frames.trading_index)

        if history is not None:
            history["snapshot_date"] = pd.to_datetime(history["snapshot_date"])
        roster = filer_ciks(history["cik"].dropna(), self._context.config_dir) if history is not None else set()
        splits = basis.actual.rename(columns={"value": "ratio"})
        institutional_features = self._institutional_features(frames, splits, cusip_map, valuation_close)
        superinvestor_levels = self._superinvestor_levels(
            frames, splits, cusip_map, roster, history, valuation_close, session_positions=session_positions
        )
        self._require_saved_families(institutional_features, superinvestor_levels, saved_columns)

        output_dates = frames.trading_index if window.since is None else frames.trading_index[frames.trading_index >= window.since]
        output_frames = self._output_frames(frames, output_dates)
        institutional_panel, superinvestor_panel = self._project_panels(
            output_frames, institutional_features, superinvestor_levels, shares, basis, session_positions
        )
        if window.is_full:
            return self._merge_panels(output_frames, institutional_panel, superinvestor_panel)

        supported = self._supported_columns(
            frames,
            output_dates,
            institutional_features,
            superinvestor_levels,
            shares,
            basis,
            session_positions,
            institutional_panel,
            superinvestor_panel,
        )
        if supported != set(saved_columns) - {"date", "ticker"}:
            del institutional_panel, superinvestor_panel
            institutional_panel, superinvestor_panel = self._project_panels(
                frames, institutional_features, superinvestor_levels, shares, basis, session_positions
            )
            return self._merge_panels(frames, institutional_panel, superinvestor_panel)

        panel = self._merge_panels(output_frames, institutional_panel, superinvestor_panel)
        if panel.empty:
            panel = output_frames.skeleton()
        for column in supported - set(panel.columns):
            panel[column] = pd.Series(np.nan, index=panel.index, dtype="float32")
        return panel.reindex(columns=saved_columns)

    def _institutional_features(
        self,
        price_frames: PriceFrames,
        splits: pd.DataFrame | None,
        cusip_map: pd.DataFrame | None,
        valuation_close: pd.DataFrame | None = None,
    ) -> pd.DataFrame | None:

        packets = thirteen_inputs.iter_filing_vintages(self._store, price_frames.trading_index.max())
        cfg = self._cfg.get("institutionals", {})
        settle_days = int(cfg.get("f13_settle_trading_days", F13_SETTLE_TRADING_DAYS))

        if packets is None:
            return None
        history = prepare_institutional_book_history(
            packets,
            price_frames.trading_index,
            cusip_map,
            price_frames.universe,
            valuation_close,
            cfg.get("value_basis"),
            settle_days=settle_days,
        )
        if history is None:
            return None
        compact, states = history
        snapshots = iter_institutional_books(compact, states)
        return compute_institutional_feature_history(snapshots, splits=splits)

    def _superinvestor_levels(
        self,
        frames: PriceFrames,
        splits: pd.DataFrame | None,
        cusip_map: pd.DataFrame | None,
        roster: set[str],
        history: pd.DataFrame | None,
        valuation_close: pd.DataFrame | None = None,
        *,
        session_positions: pd.Series,
    ) -> dict[str, pd.DataFrame] | None:

        filings, holdings = (
            thirteen_inputs.load_manager_filing_vintages(self._store, frames.trading_index.max(), sorted(roster)) if roster else None
        ) or (pd.DataFrame(), pd.DataFrame())
        cfg = self._cfg.get("institutionals", {})
        stale_quarters = int(cfg.get("superinvestor", {}).get("stale_quarters", 4))
        decay_halflife = self._decay_halflife("super")
        snapshots: dict[str, list[pd.DataFrame]] = {}
        state = pd.DataFrame()

        if filings.empty:
            return None
        self._log.info(
            "Elite 13F books: %s rows across %s ever-listed managers (%s on today's roster)",
            len(holdings),
            len(roster),
            len(roster_as_of(history, config_dir=self._context.config_dir)),
        )

        holdings = clean_public_holdings(
            attach_tickers(holdings, cusip_map, frames.universe),
            valuation_close,
            cfg.get("value_basis"),
        )
        prepared = prepare_book_vintages(filings, holdings, frames.trading_index, settle_days=0)
        selector = self._superinvestor_selector(history)

        for day, state, conviction, states, observed_pairs in iter_manager_feature_inputs(prepared, holdings, frames.trading_index, stale_quarters):
            weights = manager_selection_weights(state, selector)
            public = public_state(state, weights)
            contributions, public, historical = superinvestor_contributions(
                conviction, state, states, weights, public, day, splits, stale_quarters, observed_pairs
            )
            if contributions.empty or public.empty:
                append_unknown_superinvestor_snapshot(snapshots, day)
                continue
            levels = compute_superinvestor_levels(
                contributions, public, historical, day, session_positions, observed_pairs.ticker.unique(), stale_quarters, decay_halflife
            )
            for name, frame in levels.items():
                snapshots.setdefault(name, []).append(frame)
        if not snapshots:
            return {}

        levels = {name: pd.concat(parts).sort_index() for name, parts in snapshots.items()}
        self._log.info("elite 13F levels: %s features over %s managers / %s quarters", len(levels), state.cik.nunique(), state.period.nunique())
        return levels

    @staticmethod
    def _output_frames(frames: PriceFrames, dates: pd.DatetimeIndex) -> PriceFrames:
        """Keep the logical universe while bounding the two dense price dependencies."""
        return replace(
            frames,
            trading_index=dates,
            close_split=frames.close_split.reindex(dates) if frames.close_split is not None else None,
            level_factor=frames.level_factor.reindex(dates) if frames.level_factor is not None else None,
        )

    @staticmethod
    def _require_saved_families(
        institutional_features: pd.DataFrame | None, superinvestor_levels: dict[str, pd.DataFrame] | None, saved_columns: Sequence[str]
    ) -> None:
        """Refuse replacement of a previously supported family after source preparation fails."""
        failed = (("f_ic_inst_", institutional_features is None), ("f_ic_super_", superinvestor_levels is None))
        for prefix, missing in failed:
            if missing and any(column.startswith(prefix) for column in saved_columns):
                raise RuntimeError(f"SEC13F source preparation failed for {prefix}; saved output is preserved.")

    def _project_institutional_panel(
        self,
        features: pd.DataFrame,
        frames: PriceFrames,
        shares: pd.DataFrame | None,
        share_basis: ShareBasis | None,
        trading_calendar: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """Project prepared quarter facts on the requested dates with existing denominator policies."""
        settle_days = int(self._cfg.get("institutionals", {}).get("f13_settle_trading_days", F13_SETTLE_TRADING_DAYS))
        if features.empty:
            return pd.DataFrame(columns=["date", "ticker"])
        fields = project_institutional_features(features, frames, shares, share_basis, settle_days, versioned=True, trading_calendar=trading_calendar)
        emission = {name: INSTITUTIONAL_EMISSION[name] for name in fields}
        return build_peer_relative_panel(fields, frames.peers, emission=emission, availability=frames.availability)

    def _project_superinvestor_panel(self, levels: dict[str, pd.DataFrame], frames: PriceFrames, session_positions: pd.Series) -> pd.DataFrame:
        """Project sparse selected-manager levels using their original full-calendar ordinals."""
        decay_halflife = self._decay_halflife("super")
        if not levels:
            return pd.DataFrame(columns=["date", "ticker"])
        family_start = min(frame.index.min() for frame in levels.values())
        if self._availability is not None:
            family_start = max(family_start, self._availability.source_date(Tables.sec13f_manager_holdings))
        fields = project_superinvestor_features(
            levels, pd.DataFrame(), frames, family_start, self._availability, decay_halflife, True, session_positions=session_positions
        )
        emission = {name: SUPERINVESTOR_EMISSION[name] for name in fields}
        return build_peer_relative_panel(fields, frames.peers, emission=emission, availability=frames.availability)

    def _project_panels(
        self,
        frames: PriceFrames,
        institutional_features: pd.DataFrame | None,
        superinvestor_levels: dict[str, pd.DataFrame] | None,
        shares: pd.DataFrame | None,
        share_basis: ShareBasis | None,
        session_positions: pd.Series,
    ) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
        institutional_panel = None
        superinvestor_panel = None
        if institutional_features is not None:
            institutional_panel = self._project_institutional_panel(
                institutional_features, frames, shares, share_basis, pd.DatetimeIndex(session_positions.index)
            )
        if superinvestor_levels is not None:
            superinvestor_panel = self._project_superinvestor_panel(superinvestor_levels, frames, session_positions)
        return institutional_panel, superinvestor_panel

    @staticmethod
    def _merge_panels(frames: PriceFrames, institutional_panel: pd.DataFrame | None, superinvestor_panel: pd.DataFrame | None) -> pd.DataFrame:
        merger = PanelMerger(anchor=frames.skeleton())
        merger.add(institutional_panel, "institutional (13F)")
        merger.add(superinvestor_panel, "superinvestor (elite 13F)")
        return merger.to_long()

    @staticmethod
    def _feature_columns(*panels: pd.DataFrame | None) -> set[str]:
        columns: set[str] = set()
        for panel in panels:
            if panel is not None:
                columns.update(set(panel.columns) - {"date", "ticker"})
        return columns

    @staticmethod
    def _possible_columns(
        institutional_features: pd.DataFrame | None, superinvestor_levels: dict[str, pd.DataFrame] | None, shares: pd.DataFrame | None
    ) -> set[str]:
        """Exclude absent producers; existing projectors decide actual historical support."""
        fields: set[str] = set()
        derived = {"inst_shares": "ic_inst_ownership_pct", "inst_value": "ic_inst_value_to_mcap"}
        if institutional_features is not None and not institutional_features.empty:
            fields.update(name for name in INSTITUTIONAL_EMISSION if name in institutional_features and institutional_features[name].notna().any())
            if shares is not None and not shares.empty:
                fields.update(
                    name for source, name in derived.items() if source in institutional_features and institutional_features[source].notna().any()
                )
        if superinvestor_levels:
            fields.update(name for name, frame in superinvestor_levels.items() if frame.notna().any().any())
        return {"f_" + name for name in fields}

    def _supported_columns(
        self,
        frames: PriceFrames,
        output_dates: pd.DatetimeIndex,
        institutional_features: pd.DataFrame | None,
        superinvestor_levels: dict[str, pd.DataFrame] | None,
        shares: pd.DataFrame | None,
        share_basis: ShareBasis | None,
        session_positions: pd.Series,
        institutional_panel: pd.DataFrame | None,
        superinvestor_panel: pd.DataFrame | None,
    ) -> set[str]:
        """Find historical-only support with bounded calls to the same family projections."""
        supported = self._feature_columns(institutional_panel, superinvestor_panel)
        unresolved = self._possible_columns(institutional_features, superinvestor_levels, shares) - supported
        end = int(frames.trading_index.searchsorted(output_dates[0]))
        span = max(len(output_dates), self._warmup() + PART_REFRESH_TRADING_DAYS + 1)

        while unresolved and end:
            start = max(0, end - span)
            probe_frames = self._output_frames(frames, frames.trading_index[start:end])
            broad = institutional_features if any(column.startswith("f_ic_inst_") for column in unresolved) else None
            elite = superinvestor_levels if any(column.startswith("f_ic_super_") for column in unresolved) else None
            probe_panels = self._project_panels(probe_frames, broad, elite, shares, share_basis, session_positions)
            supported.update(self._feature_columns(*probe_panels))
            unresolved.difference_update(supported)
            del probe_panels
            end = start
        return supported

    def _superinvestor_selector(self, history: pd.DataFrame | None) -> Callable[[pd.DataFrame], pd.Series] | None:
        cfg = self._cfg.get("institutionals", {}).get("superinvestor", {}).get("selection", {})
        raw = cfg.get("mode", "off")
        mode = "off" if raw in (False, None, "") else str(raw).lower()
        if mode in ("off", "none", "flat", "false"):
            return None
        first = first_snapshot_date(history)
        if first is None:
            self._log.warning("No roster snapshot -> selection falls back to flat 1.0.")
            return None
        roster_at = partial(roster_members_on, history=history, first=first, config_dir=self._context.config_dir, cache={})
        return partial(
            select_managers,
            roster_at=roster_at,
            mode=mode,
            k=int(cfg.get("k", 15)),
            min_quarters=int(cfg.get("min_quarters", 4)),
            min_positions=int(cfg.get("min_positions", 0)),
        )

    def _decay_halflife(self, family: str) -> float:
        cfg = self._cfg.get("institutionals", {}).get("decay_halflife", {})
        return float(cfg.get(family, 63))
