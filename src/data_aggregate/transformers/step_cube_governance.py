"""Build the point-in-time governance cube part."""

from __future__ import annotations

import pandas as pd
from omegaconf import DictConfig

from src.context import Context
from src.data_aggregate.utils.common.incremental import (
    COLUMNS_CHANGED,
    PartWindow,
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
from src.data_aggregate.utils.governance.clean import PAY_BACKFILL_STAGE, PAY_BACKFILL_WARNING, GovernanceInputs, clean_governance_inputs
from src.data_aggregate.utils.governance.panel import build_governance_feature_panel
from src.data_store.schema import Tables
from src.utils.filer_tables import def14a_evidence_mask
from src.utils.step import Step

_PRICE_FIELDS = ("close_split", "close_total")
_OWNERSHIP_COLUMNS = ("ticker", "accession_number", "holder_type", "holder_name", "shares")
#: the cleaning stage reads later proxies (stint constants, the historical pay backfill), so an
#: incremental run cannot restate the rows it does not rewrite
# A new proxy restates up to three fiscal years of CEO pay on older rows (the later-SCT
# backfill and the <= 2-proxy interpolation), so every run rewrites three years, not a week.
REFRESH_TRADING_DAYS = 756
_INCREMENTAL_WARNING = (
    "Incremental governance build rewrites from %s only: a new proxy can still restate older rows through "
    "the CEO stint constants and the ever-dual / ceiling refusals -> run build-governance -F once a year after the proxy season."
)


class StepCubeGovernance(Step):
    _FIELDS = _PRICE_FIELDS

    def __init__(self, context: Context, config: DictConfig):
        super().__init__(context=context, config=config)

        self._cfg = config.build_cube
        self._part = part_for(Tables.cube_part_governance)
        self._store = context.store

    def run(self, full: bool = False) -> None:
        calendar = load_trading_calendar(self._store)
        window = plan_window(
            self._store, Tables.cube_part_governance, full=full, warmup=self._warmup(), trading_index=calendar, refresh=REFRESH_TRADING_DAYS
        )
        frames = self._load_frames(window.since)

        merger = PanelMerger(anchor=frames.skeleton())
        merger.add(
            self._governance_panel(frames, window),
            "governance",
            "No governance features built (def14a_llm empty — accrues as fetch_def14a_llm runs).",
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
        """Load usable scalar observations, excluding archives and retry-only parents."""
        df = self._store.load(Tables.def14a_llm, project=True, optional=True)
        if df is not None:
            df = df.loc[def14a_evidence_mask(df)]
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
        """Load optional executive-compensation rows."""
        df = self._store.load(Tables.def14a_executive_comp, optional=True)
        if df is None or df.empty:
            self._log.warning("No %s -> the exact CEO Pay Slice family is skipped.", Tables.def14a_executive_comp.name)
            return None
        return df

    def _load_directors(self) -> pd.DataFrame | None:
        """Load projected director rows."""
        table = Tables.def14a_directors
        if not self._store.exists(table):
            self._log.warning("No %s -> the board averages stay on the parent scalar and the board-quality family is skipped.", table.name)
            return None
        df = self._store.load(table, project=True, optional=True)
        if df is None or df.empty:
            return None
        return df

    def _load_director_comp(self) -> pd.DataFrame | None:
        """Load optional Item 402(k) director compensation."""
        table = Tables.def14a_director_comp
        if not self._store.exists(table):
            self._log.warning("No %s -> the four director-pay features are skipped.", table.name)
            return None
        df = self._store.load(table, project=True, optional=True)
        if df is None or df.empty:
            return None
        return df

    def _load_ownership(self) -> pd.DataFrame | None:
        """Load the per-holder beneficial-ownership rows the insider-ownership estimate sums."""
        df = self._store.load(Tables.def14a_ownership, columns=_OWNERSHIP_COLUMNS, optional=True)
        if df is None or df.empty:
            self._log.warning("No %s -> insider ownership is not estimated from the holder rows.", Tables.def14a_ownership.name)
            return None
        return df

    def _load_fundamentals(self) -> pd.DataFrame | None:
        """Load revenue and point-in-time shares used by governance features."""
        df = self._store.load(Tables.fundamentals_history, optional=True)
        if df is None or df.empty:
            self._log.warning("No fundamentals history -> the pay-vs-revenue-growth misalignment feature is skipped.")
            return None
        return df

    # ---- panels ---- #
    def _governance_panel(self, frames: PriceFrames, window: PartWindow) -> pd.DataFrame | None:
        """Load the raw governance frames, clean them once, then build the features."""
        df = self._load_def14a()
        if df is None:
            return None
        if not window.is_full:
            self._log.warning(_INCREMENTAL_WARNING, pd.Timestamp(window.refresh_from or window.last).date())

        raw = GovernanceInputs(
            def14a=df,
            exec_comp=self._load_exec_comp(),
            directors=self._load_directors(),
            director_comp=self._load_director_comp(),
            votes=self._load_votes(),
            ownership=self._load_ownership(),
        )
        fundamentals = self._load_fundamentals()
        inputs, stats = clean_governance_inputs(raw, fundamentals)
        lookahead = {k.split("|", 1)[1]: v for k, v in stats.items() if k.startswith(f"{PAY_BACKFILL_STAGE}|")}
        for key, value in stats.items():
            if not key.startswith(f"{PAY_BACKFILL_STAGE}|"):
                self._log.info("  governance clean | %s: %s", key, f"{value:,}")
        if lookahead:
            self._log.warning(PAY_BACKFILL_WARNING, "; ".join(f"{k}: {v:,}" for k, v in lookahead.items()))

        panel, tally = build_governance_feature_panel(
            inputs.def14a,
            frames.peers,
            frames.trading_index,
            fundamentals_history=fundamentals,
            votes=inputs.votes,
            exec_comp=inputs.exec_comp,
            directors=inputs.directors,
            director_comp=inputs.director_comp,
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
