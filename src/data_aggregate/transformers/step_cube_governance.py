"""Build the point-in-time governance cube part."""

from __future__ import annotations

import pandas as pd
from omegaconf import DictConfig

from src.context import Context
from src.data_aggregate.utils.common.incremental import (
    COLUMNS_CHANGED,
    PART_REFRESH_TRADING_DAYS,
    plan_window,
    write_part,
)
from src.data_aggregate.utils.common.panel_merge import PanelMerger
from src.data_aggregate.utils.common.parts import part_for
from src.data_aggregate.utils.common.peers_io import load_peers_or_raise
from src.data_aggregate.utils.common.price_frames import (
    PriceFrames,
    load_price_frames,
    load_trading_calendar,
)
from src.data_aggregate.utils.governance.def14a_impute import (
    impute_def14a,
    impute_exec_comp,
)
from src.data_aggregate.utils.governance.director_comp import impute_director_comp
from src.data_aggregate.utils.governance.directors import (
    board_aggregates,
    fill_director_attributes,
    finalize_board_source,
    merge_board_aggregates,
)
from src.data_aggregate.utils.governance.panel import build_governance_feature_panel
from src.data_store.schema import Tables
from src.utils.step import Step

_PRICE_FIELDS = ("close_split", "close_total")


class StepCubeGovernance(Step):
    def __init__(self, context: Context, config: DictConfig):
        super().__init__(context=context, config=config)

        self._cfg = config.build_cube
        self._part = part_for(Tables.cube_part_governance)
        self._store = context.store

    def run(self, full: bool = False) -> None:
        calendar = load_trading_calendar(self._store)
        window = plan_window(
            self._store, Tables.cube_part_governance, full=full, warmup=self._warmup(), trading_index=calendar, refresh=PART_REFRESH_TRADING_DAYS
        )
        frames = self._load_frames(window.since)

        merger = PanelMerger(self._log, anchor=frames.skeleton())
        merger.add(
            self._governance_panel(frames), "governance", "No governance features built (def14a_llm empty — accrues as fetch_def14a_llm runs)."
        )

        panel = merger.to_long()
        del frames
        n = write_part(self._store, Tables.cube_part_governance, panel, window, drop_empty=True)
        if n == COLUMNS_CHANGED:
            return self.run(full=True)
        self._assert_current(calendar, window.last)

    def _assert_current(
        self,
        calendar: pd.DatetimeIndex,
        previous_max: pd.Timestamp | None,
    ) -> None:
        actual = pd.DatetimeIndex(pd.to_datetime(self._store.distinct(Tables.cube_part_governance, "date"))).normalize()
        expected_end = calendar.max().normalize()
        actual_end = actual.max().normalize() if len(actual) else None
        new_dates = calendar[:0] if previous_max is None else calendar[calendar > previous_max]
        missing = new_dates.difference(actual)
        if actual_end != expected_end or len(missing):
            raise RuntimeError(
                f"{Tables.cube_part_governance} incomplete: max={actual_end}, expected={expected_end}, missing_new_dates={len(missing)}"
            )
        self._log.info("Governance frontier current: %s (missing new dates=0)", expected_end.date())

    def _warmup(self) -> int:
        override = self._cfg.get("incremental", {}).get("warmup_trading_days")
        return int(override) if override is not None else self._part.warmup_trading_days

    # ---- inputs ---- #
    def _load_frames(self, since: pd.Timestamp | None) -> PriceFrames:
        return load_price_frames(self._store, peers=load_peers_or_raise(self._context, self._config), fields=_PRICE_FIELDS, since=since)

    def _load_def14a(self) -> pd.DataFrame | None:
        """Load the full proxy archive."""
        df = self._store.load(Tables.def14a_llm, optional=True)
        if df is None or df.empty:
            return None
        self._log.info("Loaded %s: %s rows x %s cols", Tables.def14a_llm.name, len(df), len(df.columns))
        return df

    def _load_votes(self) -> pd.DataFrame | None:
        """Load optional Item 5.07 vote records."""
        df = self._store.load(Tables.sec_8k_votes, optional=True)
        if df is None or df.empty:
            self._log.warning("No %s -> the four shareholder-dissent families are skipped.", Tables.sec_8k_votes.name)
            return None
        self._log.info("Loaded %s: %s rows x %s cols", Tables.sec_8k_votes.name, len(df), len(df.columns))
        return df

    def _load_exec_comp(self) -> pd.DataFrame | None:
        """Load and clean optional executive-compensation rows."""
        df = self._store.load(Tables.def14a_executive_comp, optional=True)
        if df is None or df.empty:
            self._log.warning("No %s -> the exact CEO Pay Slice family is skipped.", Tables.def14a_executive_comp.name)
            return None
        df, stats = impute_exec_comp(df)
        if stats:
            self._log.info("NEO comp clean-on-read: %s", ", ".join(f"{k}={v:,}" for k, v in stats.items()))
        return df

    def _load_directors(self) -> pd.DataFrame | None:
        """Load projected director rows and preserve source-aware fills."""
        table = Tables.def14a_directors
        if not self._store.exists(table):
            self._log.warning("No %s -> the board averages stay on the parent scalar and the board-quality family is skipped.", table.name)
            return None
        df = self._store.load(table, project=True, optional=True)
        if df is None or df.empty:
            return None
        df, stats = fill_director_attributes(df)
        if stats:
            self._log.info("Director clean-on-read: %s", ", ".join(f"{k}={v:,}" for k, v in stats.items()))
        return df

    def _load_director_comp(self) -> pd.DataFrame | None:
        """Load and clean optional Item 402(k) director compensation."""
        table = Tables.def14a_director_comp
        if not self._store.exists(table):
            self._log.warning("No %s -> the four director-pay features are skipped.", table.name)
            return None
        df = self._store.load(table, project=True, optional=True)
        if df is None or df.empty:
            return None
        df, stats = impute_director_comp(df)
        if stats:
            self._log.info("Director-comp clean-on-read: %s", ", ".join(f"{k}={v:,}" for k, v in stats.items()))
        return df

    def _load_fundamentals(self) -> pd.DataFrame | None:
        """Load revenue and point-in-time shares used by governance features."""
        df = self._store.load(Tables.fundamentals_history, optional=True)
        if df is None or df.empty:
            self._log.warning("No fundamentals history -> the pay-vs-revenue-growth misalignment feature is skipped.")
            return None
        return df

    # ---- panels ---- #
    def _governance_panel(self, frames: PriceFrames) -> pd.DataFrame | None:
        """Build governance features without mutating raw extraction tables."""
        df = self._load_def14a()
        if df is None:
            return None

        # ⚠ THE ORDER IS THE DESIGN (D35, §3.2). The directors table is filled per person, the
        # board averages are DERIVED from it, the derivation is merged into the parent rows under
        # D39's precedence -- and only THEN does `impute_def14a` run, so its forward carry is the
        # LAST resort rather than the first move. `CARRY_LEVELS` itself is unchanged: this is a
        # change of order, not of the fill rule.
        directors = self._load_directors()
        if directors is not None:
            df, mstats = merge_board_aggregates(df, board_aggregates(directors))
            for k, v in mstats.items():
                self._log.info("  board aggregate | %s: %s", k, f"{v:,}")

        df, stats = impute_def14a(df)
        if stats:
            self._log.info("DEF 14A clean-on-read: deduced %d missing cells across %d rules (raw table untouched).", sum(stats.values()), len(stats))
        # Step 4 of the precedence chain has now run, so the three-valued provenance can be
        # completed and said out loud: how much of each board average is EVIDENCE.
        df, sstats = finalize_board_source(df)
        for k, v in sstats.items():
            self._log.info("  board aggregate provenance | %s: %s", k, f"{v:,}")

        panel, tally = build_governance_feature_panel(
            df,
            frames.peers,
            frames.trading_index,
            fundamentals_history=self._load_fundamentals(),
            votes=self._load_votes(),
            exec_comp=self._load_exec_comp(),
            directors=directors,
            director_comp=self._load_director_comp(),
            close_total=frames.close_total,
            availability=frames.availability,
        )
        # Per-family row counts, the [0,1] rejects and the CEO-ballot split, one line each.
        # These are the numbers that make a thin family readable as a KNOWN property of the
        # ballot rather than a silent build failure -- `management_dissent_spread` in
        # particular is thin by construction (an exec-officer nominee stands in ~18% of
        # elections), and without the tally that looks identical to a broken join.
        for key in sorted(tally):
            self._log.info("  governance tally | %s: %s", key, f"{tally[key]:,}")
        return panel
