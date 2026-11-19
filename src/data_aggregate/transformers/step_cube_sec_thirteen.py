"""Read scoped public13F books, preserve pooled repair and save one bounded output window."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import closing
from dataclasses import dataclass
from typing import cast

import pandas as pd
from omegaconf import DictConfig

from src.context import Context
from src.data_aggregate.utils.common import inputs as capital_inputs
from src.data_aggregate.utils.common.availability import InstitutionalAvailability
from src.data_aggregate.utils.common.incremental import plan_window, write_part
from src.data_aggregate.utils.common.panel import build_peer_relative_panel
from src.data_aggregate.utils.common.panel_merge import PanelMerger
from src.data_aggregate.utils.common.peers_io import load_peers_or_raise
from src.data_aggregate.utils.common.quotes import lookup_asof
from src.data_aggregate.utils.price.frames import PriceFrames, load_price_frames, load_trading_calendar
from src.data_aggregate.utils.sec_thirteen import inputs as thirteen_inputs
from src.data_aggregate.utils.sec_thirteen.holdings_clean import build_consensus_quotes, clean_public_batches
from src.data_aggregate.utils.sec_thirteen.institutional_books import prepare_institutional_book_history
from src.data_aggregate.utils.sec_thirteen.institutional_features import EMISSION as INSTITUTIONAL_EMISSION
from src.data_aggregate.utils.sec_thirteen.institutional_features import compute_institutional_feature_history, project_institutional_features
from src.data_aggregate.utils.sec_thirteen.superinvestor_books import prepare_superinvestor_book_history
from src.data_aggregate.utils.sec_thirteen.superinvestor_features import EMISSION as SUPERINVESTOR_EMISSION
from src.data_aggregate.utils.sec_thirteen.superinvestor_features import compute_superinvestor_feature_history, project_superinvestor_features
from src.data_aggregate.utils.sec_thirteen.superinvestor_metrics import BehaviorMarket
from src.data_aggregate.utils.sec_thirteen.superinvestor_roster import load_manager_cohorts
from src.data_aggregate.utils.sec_thirteen.value_basis import MIN_ROW_COVERAGE, prepare_public_value_quotes
from src.data_store.schema import Table, Tables
from src.utils.identity_tables import CusipTags, universe_cusips
from src.utils.step import Step
from src.utils.superinvestor_roster import filer_ciks
from src.utils.window import Window, WindowPlan


@dataclass(frozen=True)
class SecThirteenResources:
    """One loaded capital/security snapshot shared by both13F families, with the CUSIP tags both read once."""

    frames: PriceFrames
    cohorts: Mapping[str, frozenset[str]] | None
    tags: CusipTags
    limits: Mapping[str, pd.Series]
    windows: Mapping[Table, Window | None]


@dataclass(frozen=True)
class PreparedSecThirteen:
    institutional: pd.DataFrame | None
    superinvestor: dict[str, pd.DataFrame] | None


class StepCubeSecThirteen(Step):
    _FIELDS = ("close_split", "close_pit", "shares_outstanding_pit", "mcap_pit", "security_basis")
    # Output-date frames; the saved basis F stays whole for quarter-end lookups before the output window.
    _OUTPUT_FIELDS = ("close_split", "close_pit", "shares_outstanding_pit", "mcap_pit")

    def __init__(self, context: Context, config: DictConfig) -> None:
        super().__init__(context=context, config=config)
        self._store = context.store
        self._cfg = config.data_aggregate
        self._availability = InstitutionalAvailability.from_config(config)

    def run(self) -> dict[Table, WindowPlan]:
        """Plan the effective window, prepare both families once, project them on the window and save it."""
        # Plan the actual rewrite footprint before loading dependency payloads.
        calendar = load_trading_calendar(self._context)
        existing_columns = self._store.columns(Tables.cube_part_sec_thirteen)
        window = plan_window(self._context, Tables.cube_part_sec_thirteen, calendar)
        if window is None:
            return {}
        saved_dates = pd.DatetimeIndex(
            self._store.distinct(Tables.cube_part_sec_thirteen, "date", where={**dict(window.scope), "date": window.write})
        )
        self._require_saved_calendar(saved_dates, calendar, window.write.start)

        # Load the shared dependencies once; keep both families' sparse full histories for genuine lag comparisons.
        resources = self._load_resources(window, calendar)
        prepared = PreparedSecThirteen(self._institutional_features(resources), self._superinvestor_history(resources))
        self._require_saved_families(prepared, existing_columns)

        # Project the output window; a full window is the whole history.
        dates = resources.frames.trading_index
        output_dates = dates[(dates >= window.write.start) & (dates <= window.write.end)]
        panels = self._project_panels(resources, prepared, output_dates)
        supported = self._feature_columns(panels)

        # Probe older windows, newest first, for columns the output window cannot show.
        unresolved = set() if window.is_full else self._possible_columns(resources, prepared) - supported
        end = int(dates.searchsorted(output_dates[0])) if len(output_dates) else len(dates)
        span = max(1, len(output_dates))
        while unresolved and end:
            start = max(0, end - span)
            broad = prepared.institutional if any(column.startswith("f_ic_inst_") for column in unresolved) else None
            elite = prepared.superinvestor if any(column.startswith("f_ic_super_") for column in unresolved) else None
            supported.update(self._feature_columns(self._project_panels(resources, PreparedSecThirteen(broad, elite), dates[start:end])))
            unresolved.difference_update(supported)
            end = start

        # Full rebuilds adopt the projected schema; incremental writes require the retained columns.
        if window.is_full:
            df_panel = self._merge_panels(resources.frames, panels)
        elif supported != set(existing_columns) - {"date", "ticker"}:
            raise ValueError("SEC13F feature columns changed; request an explicit full rebuild")
        else:
            output_frames = resources.frames.on(output_dates, *self._OUTPUT_FIELDS)
            df_panel = self._merge_panels(output_frames, panels)
            df_panel = output_frames.skeleton() if df_panel.empty else df_panel
            df_panel = df_panel.assign(**{column: pd.Series(index=df_panel.index, dtype="float32") for column in supported - set(df_panel.columns)})
            df_panel = df_panel.reindex(columns=existing_columns)

        write_part(self._store, Tables.cube_part_sec_thirteen, df_panel, window, drop_empty=True, existing_columns=existing_columns)
        return {Tables.cube_part_sec_thirteen: window}

    @staticmethod
    def _require_saved_calendar(saved_dates: pd.DatetimeIndex, calendar: pd.DatetimeIndex, refresh_from: pd.Timestamp | None) -> None:
        rewrite_dates = saved_dates if refresh_from is None else saved_dates[saved_dates >= refresh_from]
        if not rewrite_dates.difference(calendar).empty:
            raise RuntimeError("SEC13F price calendar ends before the saved output frontier or omits a saved rewrite date; no rows may be replaced.")

    def _load_resources(self, window: WindowPlan, calendar: pd.DatetimeIndex) -> SecThirteenResources:
        # Read the shared dependencies once before any cleaning or feature work.
        windows = capital_inputs.input_windows(self._context, Tables.cube_part_sec_thirteen, window.write)
        frames = load_price_frames(
            self._context, peers=load_peers_or_raise(self._context), calendar=calendar, fields=self._FIELDS, window=windows[Tables.cube_part_prices]
        )
        frames.require(*self._FIELDS)
        history = capital_inputs.load_source(self._context, Tables.superinvestor_roster, window=windows[Tables.superinvestor_roster])
        if windows[Tables.universe_cusip] is not None or windows[Tables.cusip_ticker_map] is not None:
            raise ValueError("SEC13F attribution requires full universe_cusip and cusip_ticker_map validity evidence")
        tags = CusipTags.load(self._store)

        # The ordered list is user-owned; membership never depends on portfolio prices.
        cohorts = None
        if history is not None:
            roster = filer_ciks(history.cik.dropna(), self._context.config_dir)
            cfg = self._cfg["sec_thirteen"]["superinvestor"]
            cohorts = load_manager_cohorts(self._context.config_dir, roster, cfg["selection"]["k"])

        # Issuer caps per quarter end bound every cleaned line: no position may exceed its issuer.
        limits = {
            "mcap": prepare_public_value_quotes(frames.mcap_pit),
            "shares_out": prepare_public_value_quotes(frames.shares_outstanding_pit),
        }
        return SecThirteenResources(frames, cohorts, tags, {name: series for name, series in limits.items() if series is not None}, windows)

    def _institutional_features(self, resources: SecThirteenResources) -> pd.DataFrame | None:
        # Universe positions: under a universe ticker or on a universe company's CUSIP (a class filed under its own ticker).
        universe = resources.frames.universe
        packets = thirteen_inputs.iter_filing_vintages(
            self._context,
            resources.windows[Tables.sec13f_hr],
            bond_lines=self._cfg["bond_lines"],
            universe=universe,
            cusips=universe_cusips(resources.tags.lines, universe),
        )
        if packets is None:
            return None

        # Each filed row tagged and in canonical-class shares before the value repair; amended books arrive whole and the
        # compact rows sum each company per manager-quarter after the resolver.
        with closing(clean_public_batches(packets, resources.tags, universe, resources.frames.close_pit, limits=resources.limits)) as batches:
            history = prepare_institutional_book_history(batches, resources.frames.trading_index, config=self._config)
        if history is None:
            return None
        compact, states = history

        # Quantities count only where the saved basis F supports the ticker on the report date.
        keys = compact.index.to_frame(index=False)
        df_basis = cast(pd.DataFrame, resources.frames.security_basis)
        if not compact.empty:
            compact = compact.assign(shares=compact["shares"].where(lookup_asof(df_basis, keys["ticker"], keys["period"]).notna().to_numpy()))
        return compute_institutional_feature_history(compact, states, df_basis=df_basis)

    def _superinvestor_history(self, resources: SecThirteenResources) -> dict[str, pd.DataFrame] | None:
        if resources.cohorts is None:
            return None

        # Whole roster books by the literal CIK list (an index scan; a subquery form scans every row).
        ciks = sorted(resources.cohorts[""])
        packets = thirteen_inputs.iter_filing_vintages(self._context, resources.windows[Tables.sec13f_hr], ciks, bond_lines=self._cfg["bond_lines"])
        if packets is None:
            return None

        # Clean whole books once, retaining off-universe and unmapped positions and provenance. Securities the price cube does not cover
        # are priced by one consensus per (quarter, CUSIP), built in a first pass over the same books.
        cfg = self._cfg["sec_thirteen"]
        age_days = int(cfg["superinvestor"]["max_age_calendar_days"])
        consensus = build_consensus_quotes(
            lambda: thirteen_inputs.iter_filing_vintages(
                self._context, resources.windows[Tables.sec13f_hr], ciks, bond_lines=self._cfg["bond_lines"]
            ),
            resources.tags,
            resources.frames.close_pit,
            resources.limits,
        )
        with closing(
            clean_public_batches(packets, resources.tags, None, resources.frames.close_pit, consensus=consensus, limits=resources.limits)
        ) as batches:
            books = prepare_superinvestor_book_history(batches, resources.frames.trading_index, age_days, self._config)
        if books is None:
            return None
        min_coverage = MIN_ROW_COVERAGE
        if resources.frames.close_pit is None or resources.frames.security_basis is None:
            raise ValueError("SEC13F superinvestor behavior requires close_pit and security_basis price fields")
        market = BehaviorMarket(resources.frames.trading_index, resources.frames.close_pit, resources.frames.security_basis, min_coverage)
        return compute_superinvestor_feature_history(books, resources.cohorts, resources.frames.universe, market)

    @staticmethod
    def _require_saved_families(prepared: PreparedSecThirteen, saved_columns: Sequence[str]) -> None:
        failed = (("f_ic_inst_", prepared.institutional is None), ("f_ic_super_", prepared.superinvestor is None))
        for prefix, missing in failed:
            if missing and any(column.startswith(prefix) for column in saved_columns):
                raise RuntimeError(f"SEC13F source preparation failed for {prefix}; saved output is preserved.")

    def _project_panels(
        self, resources: SecThirteenResources, prepared: PreparedSecThirteen, dates: pd.DatetimeIndex
    ) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
        """Each prepared family's daily panel on `dates`."""
        # Broad levels and changes, emitted raw through the peer-relative panel.
        frames = resources.frames.on(dates, *self._OUTPUT_FIELDS)
        broad = None if prepared.institutional is None else pd.DataFrame(columns=["date", "ticker"])
        if prepared.institutional is not None and not prepared.institutional.empty:
            fields = project_institutional_features(prepared.institutional, frames, self._config)
            emission = {name: INSTITUTIONAL_EMISSION[name] for name in fields}
            broad = build_peer_relative_panel(fields, frames.peers, emission=emission, availability=frames.availability)

        # Superinvestor levels from the source's availability start.
        elite = None
        if prepared.superinvestor is not None:
            start = self._availability.source_date(Tables.sec13f_hr) if self._availability is not None else resources.frames.trading_index.min()
            elite = project_superinvestor_features(prepared.superinvestor, frames, start)
        return broad, elite

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
        """Feature columns the prepared facts could support on some date."""
        fields: set[str] = set(SUPERINVESTOR_EMISSION) if prepared.superinvestor is not None else set()
        broad = prepared.institutional
        if broad is None or broad.empty:
            return {"f_" + name for name in fields}

        # Broad features with a value, and the ratios whose saved denominator and source amount both exist.
        fields.update(name for name in INSTITUTIONAL_EMISSION if name in broad and broad[name].notna().any())
        denominators = (
            (resources.frames.shares_outstanding_pit, "inst_shares", "ic_inst_ownership_pct"),
            (resources.frames.mcap_pit, "inst_value", "ic_inst_value_to_mcap"),
        )
        fields.update(
            name
            for df_denominator, source, name in denominators
            if cast(pd.DataFrame, df_denominator).gt(0).any().any() and source in broad and broad[source].notna().any()
        )
        return {"f_" + name for name in fields}
