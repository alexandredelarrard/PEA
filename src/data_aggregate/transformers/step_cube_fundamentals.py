"""
step_cube_fundamentals.py  (src/data_aggregate/transformers/step_cube_fundamentals.py)
----------------------------------------------------------------------------------
Everything keyed on SEC filings -> `cube_part_fundamentals`: the peer-relative fundamental
panel, the sector-scoped KPIs, earnings expectations, workforce and dividends.

WHY THESE FIVE TOGETHER. They all read `fundamentals_history`, and in the exploded DAG each
was its own task, so that table was loaded five separate times and every shared field
(`sharesOutstanding`, `totalRevenue`, `netIncome`, `freeCashflow`) was pivoted and
forward-filled once per task. Here it is loaded ONCE and the point-in-time frames are shared
through a single `PitFrames`, which is a pure memoization -- proved bit-identical by
`tests/data_aggregate/test_pit_cache.py`.

Quarter-basis and leak-free by construction: the SEC history is quarterly (TTM levels) and
every value is keyed on its FILING date (`as_of`); the point-in-time layer forward-fills each
value only from that date, so a feature on day d reflects the most recent quarter whose
10-Q/10-K was already public on d -- never a not-yet-filed one.

Warm-up 1560: the binding look-backs are `_self_history_z`'s rolling(1260) and the 5-year
dividend payout CAGR's 1260-day shift of a 252-day TTM series (1511 days end to end).
`sector` / `earnings` need ~none (they look back in filing space over the full source table),
so merging them into this part costs them a longer daily grid but no correctness.
"""

from __future__ import annotations

import pandas as pd
from omegaconf import DictConfig

from src.constants.constants import FUNDAMENTALS_REFRESH_TRADING_DAYS
from src.context import Context
from src.data_aggregate.utils.common.gics import attach_gics_columns
from src.data_aggregate.utils.common.incremental import (
    COLUMNS_CHANGED,
    plan_window,
    write_part,
)
from src.data_aggregate.utils.common.panel_merge import PanelMerger
from src.data_aggregate.utils.common.parts import part_for
from src.data_aggregate.utils.common.peers_io import load_peers_or_raise
from src.data_aggregate.utils.common.pit import PitFrames, add_cube_time_growth
from src.data_aggregate.utils.common.price_frames import (
    PriceFrames,
    load_price_frames,
    load_trading_calendar,
)
from src.data_aggregate.utils.fundamentals.dividend_features import build_dividend_feature_panel
from src.data_aggregate.utils.fundamentals.earnings_features import build_earnings_feature_panel
from src.data_aggregate.utils.fundamentals.employee_features import build_employee_feature_panel
from src.data_aggregate.utils.fundamentals.feature_views import FEATURE_COLUMNS, HISTORY_FEATURES
from src.data_aggregate.utils.fundamentals.fundamental_features import (
    build_fundamental_feature_panel,
    load_notes_num_scoped,
    load_pension_facts_scoped,
)
from src.data_aggregate.utils.fundamentals.sector_features import build_sector_feature_panel
from src.data_store.schema import Tables
from src.utils.step import Step

_INCREMENTAL_SOURCE_YEARS = 6


class StepCubeFundamentals(Step):
    # The price fields this step projects, declared like every sibling sub-step so the
    # projection stays visible and testable (valuation ratios need the close only).
    _FIELDS = ("close_split", "level_factor")

    def __init__(self, context: Context, config: DictConfig):
        super().__init__(context=context, config=config)
        self._cfg = config.build_cube
        self._part = part_for(Tables.cube_part_fundamentals)
        self._store = context.store

    def run(self, full: bool = False) -> None:
        # load inputs
        trading_calendar = load_trading_calendar(self._store)
        window = plan_window(
            self._store,
            Tables.cube_part_fundamentals,
            full=full,
            warmup=self._warmup(),
            trading_index=trading_calendar,
            refresh=FUNDAMENTALS_REFRESH_TRADING_DAYS,
        )
        dividend_frames = self._load_frames(window.since)
        frames = dividend_frames
        source_since: pd.Timestamp | None = None
        history_fields: dict[str, pd.DataFrame] | None = None
        output_since: pd.Timestamp | None = None
        if not window.is_full:
            if window.refresh_from is None:
                raise RuntimeError("incremental fundamentals window requires an inclusive refresh boundary")
            output_since = window.refresh_from
            source_since = pd.Timestamp(dividend_frames.trading_index.max()).normalize() - pd.DateOffset(years=_INCREMENTAL_SOURCE_YEARS)
            history_fields = self._load_history_fields(
                dividend_frames.universe,
                since=window.since,
                until=output_since,
            )
            frames = self._slice_frames(dividend_frames, output_since)

        fundamentals = self._load_fundamentals(frames.universe, since=source_since)
        earnings = self._load_optional(
            Tables.earnings_surprises,
            "earnings-surprise history",
            "fetch_earnings_surprises",
            frames.universe,
            since=source_since,
        )

        # ONE point-in-time cache for all five builders (see the module docstring)
        pit = PitFrames(fundamentals, frames.trading_index, frames.close_split, frames.level_factor)

        merger = PanelMerger(self._log)
        merger.add(frames.skeleton().assign(_grid=1.0), "universe-grid")
        merger.add(
            self._fundamental_panel(
                frames,
                fundamentals,
                earnings,
                pit,
                history_fields,
                output_since,
                source_since,
            ),
            "peer-relative fundamental",
            "No fundamental features built (missing fundamentals).",
        )
        merger.add(
            self._sector_kpi_panel(frames, fundamentals, pit, history_fields, output_since),
            "sector-KPI",
            "No sector KPI features built (missing fundamentals).",
        )

        # earnings
        merger.add(
            self._earnings_panel(frames, earnings, history_fields, output_since),
            "earnings-expectation",
            "No earnings-expectation features built.",
        )

        # employees
        merger.add(
            self._employee_panel(frames, fundamentals, pit, history_fields, output_since),
            "workforce",
            "No workforce features built.",
        )

        # dividends
        merger.add(
            self._dividend_panel(dividend_frames, fundamentals, pit, history_fields, output_since),
            "dividend",
            "No dividend features built (missing dividend history).",
        )
        self._log.info("PitFrames shared across the fundamentals builders: %s", pit.stats())

        panel = merger.to_long().drop(columns=["_grid"], errors="ignore")
        panel = self._restrict_to_skeleton(panel, frames.skeleton())
        if not window.is_full:
            panel = panel.reindex(columns=["date", "ticker", *FEATURE_COLUMNS])
        del frames, fundamentals, earnings, pit

        n = write_part(self._store, Tables.cube_part_fundamentals, panel, window, drop_empty=True)
        if n == COLUMNS_CHANGED:
            return self.run(full=True)

    def _warmup(self) -> int:
        override = self._cfg.get("incremental", {}).get("warmup_trading_days")
        return int(override) if override is not None else self._part.warmup_trading_days

    # ---- inputs ---- #
    def _load_frames(self, since: pd.Timestamp | None) -> PriceFrames:
        return load_price_frames(self._store, peers=load_peers_or_raise(self._context, self._config), fields=self._FIELDS, since=since)

    @staticmethod
    def _slice_frames(frames: PriceFrames, since: pd.Timestamp) -> PriceFrames:
        idx = frames.trading_index[frames.trading_index >= pd.Timestamp(since)]

        def _slice(frame: pd.DataFrame | None) -> pd.DataFrame | None:
            return None if frame is None else frame.reindex(idx)

        return PriceFrames(
            trading_index=idx,
            universe=frames.universe,
            peers=frames.peers,
            close_split=_slice(frames.close_split),
            close_total=_slice(frames.close_total),
            open=_slice(frames.open),
            high=_slice(frames.high),
            low=_slice(frames.low),
            volume=_slice(frames.volume),
            ret=_slice(frames.ret),
            sector_ret=_slice(frames.sector_ret),
            level_factor=_slice(frames.level_factor),
        )

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

    def _load_fundamentals(self, universe: tuple[str, ...], since: pd.Timestamp | None = None) -> pd.DataFrame | None:
        """`fundamentals_history` with GICS attached, loaded ONCE for five builders.

        The GICS join happens HERE, after the load, and returns a new frame rather than
        projecting the two columns out of the store: `sp500_tickers` is the authority for
        sector membership and `fundamentals_history` has never carried it, so this is a
        lookup, not a column the read could have asked for. Without it every sector KPI is
        gated off (`sector_gates.row_gate` fails closed on the absent column)."""
        df = self._context.store.load(
            Tables.fundamentals_history,
            optional=True,
            project=True,
            where={"ticker": list(universe)},
            since=since,
        )
        if df is None:
            raise Exception("No fundamentals history -> the fundamental, sector, workforce and dividend-payout features will be skipped.")
        self._log.info(
            "Loaded %s: %s rows, %s tickers%s (ONCE for five builders)",
            Tables.fundamentals_history,
            len(df),
            df["ticker"].nunique(),
            f" since {pd.Timestamp(since).date()}" if since is not None else "",
        )
        df = add_cube_time_growth(df)
        return attach_gics_columns(df, self._context, self._log)

    def _load_optional(
        self,
        table: str,
        what: str,
        fetcher: str,
        universe: tuple[str, ...],
        since: pd.Timestamp | None = None,
    ) -> pd.DataFrame | None:
        df = self._context.store.load(
            table,
            optional=True,
            where={"ticker": list(universe)},
            since=since,
        )
        if df is None:
            self._log.warning("No %s -> related features skipped (run %s).", what, fetcher)
            return None
        return df

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
    def _fundamental_panel(
        self,
        frames: PriceFrames,
        fundamentals: pd.DataFrame | None,
        earnings: pd.DataFrame | None,
        pit: PitFrames,
        history_fields: dict[str, pd.DataFrame] | None,
        output_since: pd.Timestamp | None,
        source_since: pd.Timestamp | None,
    ) -> pd.DataFrame | None:
        if fundamentals is None:
            return None
        hist = self._cfg.get("hist", {})
        # tag-scoped reads: only the two pension tags of each table, never the whole
        # multi-million-row facts tables
        return build_fundamental_feature_panel(
            fundamentals_history=fundamentals,
            peer_dict=frames.peers,
            trading_index=frames.trading_index,
            stock_close=frames.close_split,
            level_factor=frames.level_factor,
            intrinsic_cfg=self._cfg.get("intrinsic", {}),
            hist_window=int(hist.get("window", 1260)),
            hist_min_periods=int(hist.get("min_periods", 252)),
            earnings_history=earnings,  # PEGY projected-growth term
            pension_facts=load_pension_facts_scoped(
                self._context,
                tickers=frames.universe,
                since=source_since,
            ),
            notes_num=load_notes_num_scoped(
                self._context,
                tickers=frames.universe,
                since=source_since,
            ),
            history_fields=history_fields,
            output_since=output_since,
        )

    def _sector_kpi_panel(
        self,
        frames: PriceFrames,
        fundamentals: pd.DataFrame | None,
        pit: PitFrames,
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

    def _earnings_panel(
        self,
        frames: PriceFrames,
        earnings: pd.DataFrame | None,
        history_fields: dict[str, pd.DataFrame] | None,
        output_since: pd.Timestamp | None,
    ) -> pd.DataFrame | None:
        """Forward EPS yield, expected EPS growth and realized surprise. Genuinely historical
        and point-in-time: the forward estimate applies only within its own quarter, the
        actual only after the report."""
        if earnings is None:
            return None
        return build_earnings_feature_panel(
            earnings,
            frames.peers,
            frames.trading_index,
            stock_close=frames.close_split,
            level_factor=frames.level_factor,
            history_fields=history_fields,
            output_since=output_since,
        )

    def _employee_panel(
        self,
        frames: PriceFrames,
        fundamentals: pd.DataFrame | None,
        pit: PitFrames,
        history_fields: dict[str, pd.DataFrame] | None,
        output_since: pd.Timestamp | None,
    ) -> pd.DataFrame | None:
        """Revenue per employee and YoY headcount growth, from the `employees` column of
        `fundamentals_history` (10-K body-text headcount)."""
        if fundamentals is None:
            return None
        return build_employee_feature_panel(
            fundamentals,
            frames.peers,
            frames.trading_index,
            history_fields=history_fields,
            output_since=output_since,
        )

    def _dividend_panel(
        self,
        frames: PriceFrames,
        fundamentals: pd.DataFrame | None,
        pit: PitFrames,
        history_fields: dict[str, pd.DataFrame] | None,
        output_since: pd.Timestamp | None,
    ) -> pd.DataFrame | None:
        """TTM yield, 1y + 5y payout growth, payer flag, payout ratio, FCF coverage, dividend
        + buyback yield. RECONCILES the per-share ex-date history (`dividends`, primary) with
        the SEC cash-flow `dividendsPaid` total (gap-fill + payout/coverage). Non-payers get a
        real 0 yield so they rank correctly."""
        dividends = self._load_optional(Tables.dividends, "dividend history", "fetch_price_history -> StepExtractPrices", frames.universe)
        if dividends is None:
            return None
        return build_dividend_feature_panel(
            dividends,
            frames.peers,
            frames.trading_index,
            stock_close=frames.close_split,
            level_factor=frames.level_factor,
            fundamentals_history=fundamentals,
            history_fields=history_fields,
            output_since=output_since,
        )
