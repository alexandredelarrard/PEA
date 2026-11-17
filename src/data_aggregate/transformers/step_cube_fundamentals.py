"""Build the fundamental cube from public histories and saved price capital.

Quantity evidence is loaded once; each family retains its publication, age and per-share rules."""

from __future__ import annotations

import pandas as pd
from omegaconf import DictConfig

from src.context import Context
from src.data_aggregate.utils.common.incremental import (
    plan_window,
    write_part,
)
from src.data_aggregate.utils.common.inputs import load_source
from src.data_aggregate.utils.common.panel_merge import PanelMerger
from src.data_aggregate.utils.common.peers_io import load_peers_or_raise
from src.data_aggregate.utils.common.pit import add_cube_time_growth
from src.data_aggregate.utils.fundamentals.disclosure_features import disclosure_return_fields
from src.data_aggregate.utils.fundamentals.dividend_features import build_dividend_feature_panel
from src.data_aggregate.utils.fundamentals.earnings_features import build_earnings_feature_panel
from src.data_aggregate.utils.fundamentals.employee_features import (
    EMPLOYEE_COLUMNS,
    build_employee_feature_panel,
    part_time_weight,
)
from src.data_aggregate.utils.fundamentals.feature_views import FEATURE_COLUMNS, HISTORY_FEATURES, build_fundamental_views
from src.data_aggregate.utils.fundamentals.fundamental_features import (
    build_fundamental_feature_panel,
    load_notes_num_scoped,
)
from src.data_aggregate.utils.fundamentals.inputs import FundamentalFeatureSettings, FundamentalHistory, FundamentalSupplement
from src.data_aggregate.utils.fundamentals.sector_features import build_sector_feature_panel
from src.data_aggregate.utils.fundamentals.source_join import SEC_CARRIED_COLUMNS, join_sec_columns
from src.data_aggregate.utils.price.frames import (
    PriceFrames,
    load_price_frames,
    load_trading_calendar,
)
from src.data_aggregate.utils.price.inputs import load_fundamental_share_observations
from src.data_store.schema import Table as SchemaTable
from src.data_store.schema import Tables
from src.utils.price.adjustments import load_bugfix
from src.utils.price.basis import load_price_identity, load_quantity_basis
from src.utils.publication import publication_dates, publication_rows
from src.utils.step import Step
from src.utils.window import Offset, Window, WindowPlan, shift_date


class StepCubeFundamentals(Step):
    # The price fields this step projects, declared like every sibling sub-step so the
    # Levels value the business; persisted total returns measure disclosure responses.
    _FIELDS = ("close_split", "level_factor", "ret", "sector_ret", "close_pit", "shares_outstanding_pit", "mcap_pit")

    def __init__(self, context: Context, config: DictConfig) -> None:
        super().__init__(context=context, config=config)
        self._cfg = config.data_aggregate
        self._store = context.store
        self._bugfix = load_bugfix(context.config_dir)

    def run(self) -> dict[SchemaTable, WindowPlan]:
        # Plan the public calculation and refresh clocks.
        trading_calendar = load_trading_calendar(self._context)
        window = plan_window(self._context, Tables.cube_part_fundamentals, trading_calendar)
        if window is None:
            return {}
        dividend_frames = load_price_frames(
            self._context, peers=load_peers_or_raise(self._context, self._config), calendar=trading_calendar, fields=self._FIELDS, window=window.read
        )
        frames = dividend_frames
        source_starts = {
            name: None if window.is_full else shift_date(window.write.start, Offset(item.value, item.unit), trading_calendar)
            for name, item in self._config.data.tables.cube_part_fundamentals.warmup.sources.items()
            if name != "fundamentals_employees"
        }
        financial_start = source_starts["financial_history"]
        employee_context = self._config.data.tables.cube_part_fundamentals.warmup.sources.fundamentals_employees
        employee_start = (
            None if financial_start is None else shift_date(financial_start, Offset(employee_context.value, employee_context.unit), trading_calendar)
        )
        history_fields: dict[str, pd.DataFrame] | None = None
        output_since: pd.Timestamp | None = None
        if not window.is_full:
            if window.write.start is None:
                raise RuntimeError("incremental fundamentals window requires an inclusive refresh boundary")
            output_since = window.write.start
            history_fields = self._load_history_fields(dividend_frames.universe, since=window.read.start, until=output_since)
            frames = dividend_frames.on(dividend_frames.trading_index[dividend_frames.trading_index >= output_since])

        # Load each source once and preserve raw financial cash units.
        df_fundamentals = self._load_fundamentals(frames.universe, trading_calendar, since=financial_start)
        earnings_start = source_starts["earnings_surprises"]
        earnings_window = None if earnings_start is None else Window(earnings_start, self._context.run.as_of)
        df_earnings = load_source(self._context, Tables.earnings_surprises, frames.universe, window=earnings_window)
        df_employees = self._load_employees(frames.universe, since=employee_start)
        identity = load_price_identity(self._context, list(frames.universe))
        basis = load_quantity_basis(self._context, frames.universe, self._bugfix, identity)
        df_share_history = load_fundamental_share_observations(self._context, df_fundamentals, basis, identity[0])
        histories = FundamentalHistory(df_fundamentals, df_share_history, basis)
        df_notes = load_notes_num_scoped(self._context, trading_calendar, tickers=frames.universe, since=source_starts["notes_num"])
        df_dividends = load_source(self._context, Tables.dividends, frames.universe)
        if df_earnings is not None:
            df_earnings = publication_rows(self._config, Tables.earnings_surprises, df_earnings, trading_calendar, "as_of")
        if df_employees is not None:
            df_employees = publication_rows(self._config, Tables.fundamentals_employees, df_employees, trading_calendar, "as_of")
        if df_dividends is not None:
            df_dividends = publication_rows(self._config, Tables.dividends, df_dividends, trading_calendar, "available_on")

        # Compute families on the same saved price and source-unit basis.
        merger = PanelMerger()
        merger.add(frames.skeleton().assign(_grid=1.0), "universe-grid")
        merger.add(
            build_fundamental_feature_panel(
                histories, frames, self._feature_settings(), FundamentalSupplement(df_earnings, df_notes), history_fields, output_since
            ),
            "peer-relative fundamental",
            "No fundamental features built (missing fundamentals).",
        )
        del df_notes
        merger.add(
            self._sector_kpi_panel(frames, df_fundamentals, history_fields, output_since),
            "sector-KPI",
            "No sector KPI features built (missing fundamentals).",
        )
        merger.add(
            build_earnings_feature_panel(df_earnings, frames, history_fields, output_since, basis),
            "earnings-expectation",
            "No earnings-expectation features built.",
        )
        merger.add(self._disclosure_panel(dividend_frames, df_earnings, output_since), "disclosure response")
        merger.add(
            self._employee_panel(frames, df_fundamentals, df_employees, history_fields, output_since),
            "workforce",
            "No workforce features built.",
        )
        merger.add(
            build_dividend_feature_panel(df_dividends, dividend_frames, histories, history_fields, output_since),
            "dividend",
            "No dividend features built (missing dividend history).",
        )

        # Restrict the completed panel before publishing the inclusive refresh.
        panel = merger.to_long().drop(columns=["_grid"], errors="ignore")
        panel = self._restrict_to_skeleton(panel, frames.skeleton()).reindex(columns=["date", "ticker", *FEATURE_COLUMNS])
        del frames, dividend_frames, histories, df_fundamentals, df_share_history, df_earnings, df_employees, df_dividends, basis
        write_part(self._store, Tables.cube_part_fundamentals, panel, window, drop_empty=True)
        return {Tables.cube_part_fundamentals: window}

    def _feature_settings(self) -> FundamentalFeatureSettings:
        """Read the existing financial history and intrinsic-model controls."""
        hist = self._cfg.hist
        features = self._cfg.fundamentals_features
        return FundamentalFeatureSettings(
            history_window=int(hist.window),
            history_min_periods=int(hist.min_periods),
            earnings_scale_quarters=int(features.earnings_scale_quarters),
            intrinsic_cfg=self._cfg.intrinsic,
        )

    # ---- inputs ---- #
    def _load_history_fields(
        self,
        universe: tuple[str, ...],
        *,
        since: pd.Timestamp | None,
        until: pd.Timestamp,
    ) -> dict[str, pd.DataFrame]:
        """Load persisted raw legs before the repair boundary as self-history context."""
        available = set(self._store.columns(Tables.cube_part_fundamentals))
        selected = [name for name in sorted(HISTORY_FEATURES) if f"f_{name}" in available]
        if not selected:
            return {}
        rows = self._store.load(
            Tables.cube_part_fundamentals,
            columns=["date", "ticker", *[f"f_{name}" for name in selected]],
            where={"ticker": list(universe)},
            since=since,
            until=pd.Timestamp(until) - pd.Timedelta(days=1),
        )
        if rows is None or rows.empty:
            return {}
        rows["date"] = pd.to_datetime(rows["date"], errors="coerce")
        rows["ticker"] = rows["ticker"].astype(str)
        fields = {name: rows.pivot(index="date", columns="ticker", values=f"f_{name}").sort_index() for name in selected}
        self._log.info(
            "Loaded persisted fundamentals history seed: %s rows, %s raw fields, %s..%s",
            len(rows),
            len(fields),
            rows["date"].min().date(),
            rows["date"].max().date(),
        )
        return fields

    def _load_fundamentals(self, universe: tuple[str, ...], calendar: pd.DatetimeIndex, since: pd.Timestamp | None = None) -> pd.DataFrame:
        """`sharadar_fundamentals_history` + three `sec_` columns + GICS, loaded ONCE for five builders.

        The Sharadar history is the spine; `sec_fundamentals_history` joins on the exact
        `(ticker, as_of)` key and is optional. The GICS join happens HERE, after the load:
        `universe_tickers` is the authority for sector membership, and without it every sector
        KPI is gated off (`sector_gates.row_gate` fails closed on the absent column)."""
        requested = None if since is None else Window(since, self._context.run.as_of)
        df = load_source(
            self._context,
            Tables.sharadar_fundamentals_history,
            universe,
            window=requested,
        )
        sec = load_source(
            self._context,
            Tables.sec_fundamentals_history,
            universe,
            columns=["ticker", "as_of", *SEC_CARRIED_COLUMNS],
            window=requested,
        )
        if df is None:
            raise Exception("No fundamentals history -> the fundamental, sector, workforce and dividend-payout features will be skipped.")
        self._log.info(
            "Loaded %s: %s rows, %s tickers%s (ONCE for five builders)",
            Tables.sharadar_fundamentals_history,
            len(df),
            df["ticker"].nunique(),
            f" since {pd.Timestamp(since).date()}" if since is not None else "",
        )
        if sec is None:
            self._log.warning("No %s -> the %s columns stay empty.", Tables.sec_fundamentals_history, ", ".join(SEC_CARRIED_COLUMNS))
        joined = join_sec_columns(df, sec)
        self._log.info(
            "Joined %s on exact (ticker, as_of): %s matched, %s Sharadar rows without SEC, %s SEC rows without Sharadar (not added)",
            Tables.sec_fundamentals_history,
            joined.matched,
            joined.sharadar_unmatched,
            joined.sec_unmatched,
        )
        # One eligibility clock covers every source field carried on the joined row.
        frame = publication_rows(self._config, Tables.sharadar_fundamentals_history, joined.frame, calendar, "as_of")
        carried = joined.frame[[f"sec_{field}" for field in SEC_CARRIED_COLUMNS]].notna().any(axis=1)
        sec_dates = publication_dates(self._config, Tables.sec_fundamentals_history, joined.frame, calendar)
        ready = pd.concat([frame["as_of"], sec_dates.reindex(frame.index).where(carried.reindex(frame.index))], axis=1).max(axis=1)
        ready = ready.where(~carried.reindex(frame.index) | sec_dates.reindex(frame.index).notna())
        frame["as_of"] = ready
        frame = frame.loc[ready.notna()]
        return self._universe.attach_gics(add_cube_time_growth(frame))

    def _load_employees(self, universe: tuple[str, ...], since: pd.Timestamp | None) -> pd.DataFrame | None:
        """Load projected employee history with its prior-year lookback."""
        requested = None if since is None else Window(since, self._context.run.as_of)
        df_employees = load_source(
            self._context,
            Tables.fundamentals_employees,
            universe,
            columns=list(EMPLOYEE_COLUMNS),
            window=requested,
        )
        if df_employees is None:
            self._log.warning("No %s -> related features skipped (run %s).", "employee headcount", "sec-employees-llm")
        return df_employees

    @staticmethod
    def _restrict_to_skeleton(
        panel: pd.DataFrame,
        skeleton: pd.DataFrame,
    ) -> pd.DataFrame:
        """Keep exactly the price-supported keys after all outer-aligned builders."""
        keys = skeleton[["date", "ticker"]].drop_duplicates()
        return keys.merge(
            panel,
            on=["date", "ticker"],
            how="left",
            validate="one_to_one",
        )

    # ---- panels ---- #

    def _sector_kpi_panel(
        self,
        frames: PriceFrames,
        fundamentals: pd.DataFrame | None,
        history_fields: dict[str, pd.DataFrame] | None,
        output_since: pd.Timestamp | None,
    ) -> pd.DataFrame | None:
        """Sector-specific KPIs (combined/loss ratio, NIM, efficiency ratio, FFO, inventory
        days, shareholder payout, net-debt/EBITDA, accruals), availability-gated per row so a
        KPI is null unless its sector reported the inputs."""
        if fundamentals is None:
            return None
        return build_sector_feature_panel(
            fundamentals,
            frames.peers,
            frames.trading_index,
            history_fields=history_fields,
            output_since=output_since,
        )

    def _disclosure_panel(
        self,
        frames: PriceFrames,
        earnings: pd.DataFrame | None,
        output_since: pd.Timestamp | None,
    ) -> pd.DataFrame | None:
        """Use the warmed return grid so incremental event windows retain their anchor."""
        if earnings is None:
            return None
        frames.require("ret", "sector_ret")
        assert frames.ret is not None and frames.sector_ret is not None
        fields = disclosure_return_fields(
            earnings,
            frames.ret,
            frames.sector_ret,
        )
        return build_fundamental_views(fields, frames.peers, output_since=output_since)

    def _employee_panel(
        self,
        frames: PriceFrames,
        fundamentals: pd.DataFrame | None,
        employees: pd.DataFrame | None,
        history_fields: dict[str, pd.DataFrame] | None,
        output_since: pd.Timestamp | None,
    ) -> pd.DataFrame | None:
        """Revenue per employee, headcount growth and elasticity: `fundamentals_employees` joined
        point-in-time onto `sharadar_fundamentals_history`, on the `FT + α·PT` headcount proxy."""
        if fundamentals is None or employees is None:
            return None
        return build_employee_feature_panel(
            fundamentals,
            employees,
            frames.peers,
            frames.trading_index,
            part_time_weight=part_time_weight(self._cfg),
            history_fields=history_fields,
            output_since=output_since,
        )
