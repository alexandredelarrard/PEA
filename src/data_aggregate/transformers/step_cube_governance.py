"""Build the point-in-time governance cube part."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

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
from src.data_aggregate.utils.governance.clean import PAY_BACKFILL_STAGE, PAY_BACKFILL_WARNING, GovernanceInputs, clean_governance_inputs
from src.data_aggregate.utils.governance.def14a_impute import EXEC_COMPONENTS
from src.data_aggregate.utils.governance.panel import build_governance_feature_panel
from src.data_aggregate.utils.price.frames import (
    PriceFrames,
    load_price_frames,
    load_trading_calendar,
)
from src.data_store.schema import Table, Tables
from src.utils.data_policy import publication_policy, source_read_window
from src.utils.filer_tables import def14a_evidence_mask
from src.utils.publication import market_days, publication_rows
from src.utils.step import Step
from src.utils.window import Window, WindowPlan

_PRICE_FIELDS = ("close_split", "close_total", "shares_outstanding_pit")
_OWNERSHIP_COLUMNS = ("ticker", "accession_number", "holder_type", "holder_name", "shares")
_INCREMENTAL_WARNING = (
    "Incremental governance build rewrites from %s only: a new proxy can still restate older rows through "
    "the CEO stint constants and the ever-dual / ceiling refusals -> run build-governance -F once a year after the proxy season."
)


class StepCubeGovernance(Step):
    _FIELDS = _PRICE_FIELDS

    def __init__(self, context: Context, config: DictConfig):
        super().__init__(context=context, config=config)

        self._cfg = config.data_aggregate
        self._store = context.store

    def run(self) -> dict[Table, WindowPlan]:
        calendar = load_trading_calendar(self._context)
        window = plan_window(self._context, Tables.cube_part_governance, calendar)
        if window is None:
            return {}
        frames = self._load_frames(window.read, calendar)

        merger = PanelMerger(anchor=frames.skeleton())
        merger.add(
            self._governance_panel(frames, window),
            "governance",
            "No governance features built (def14a_llm empty — accrues as fetch_def14a_llm runs).",
        )

        panel = merger.to_long()
        del frames
        write_part(self._store, Tables.cube_part_governance, panel, window, drop_empty=True)
        self._assert_current(calendar, window)
        return {Tables.cube_part_governance: window}

    def _assert_current(
        self,
        calendar: pd.DatetimeIndex,
        window: WindowPlan,
    ) -> None:
        actual = pd.DatetimeIndex(
            pd.to_datetime(self._store.distinct(Tables.cube_part_governance, "date", where={**dict(window.scope), "date": window.write}))
        ).normalize()
        expected = calendar[(calendar >= window.write.start) & (calendar <= window.write.end)]
        expected_end = expected.max().normalize()
        actual_end = actual.max().normalize() if len(actual) else None
        new_dates = expected[:0] if window.frontier is None else expected[expected > window.frontier]
        missing = new_dates.difference(actual)
        if actual_end != expected_end or len(missing):
            raise RuntimeError(
                f"{Tables.cube_part_governance} incomplete: max={actual_end}, expected={expected_end}, missing_new_dates={len(missing)}"
            )
        self._log.info("Governance frontier current: %s (missing new dates=0)", expected_end.date())

    # ---- inputs ---- #
    def _load_frames(self, window: Window, calendar: pd.DatetimeIndex) -> PriceFrames:
        """Retain full count lineage while bounding the daily feature/return grid."""
        retained = source_read_window(self._config, Tables.cube_part_prices, window.end)
        if retained is None:
            raise ValueError("No configured price lineage is available")
        frames = load_price_frames(
            self._context, peers=load_peers_or_raise(self._context, self._config), calendar=calendar, fields=_PRICE_FIELDS, window=retained
        )
        return frames.on(frames.trading_index[frames.trading_index >= window.start], "close_split", "close_total")

    def _load_def14a(self, window: Window | None) -> pd.DataFrame | None:
        """Load usable scalar observations, excluding archives and retry-only parents."""
        if window is None:
            return None
        df = load_source(self._context, Tables.def14a_llm, window=window)
        if df is not None:
            df = df.loc[def14a_evidence_mask(df)]
        if df is None or df.empty:
            return None
        self._log.info("Loaded %s: %s rows x %s cols", Tables.def14a_llm.name, len(df), len(df.columns))
        return df

    def _load_votes(self, window: Window | None) -> pd.DataFrame | None:
        """Load optional Item 5.07 vote records."""
        if window is None:
            return None
        df = load_source(self._context, Tables.sec_8k_votes, window=window)
        if df is None or df.empty:
            self._log.warning("No %s -> the four shareholder-dissent families are skipped.", Tables.sec_8k_votes.name)
            return None
        self._log.info("Loaded %s: %s rows x %s cols", Tables.sec_8k_votes.name, len(df), len(df.columns))
        return df

    def _load_proxy_child(self, table: Table, parents: pd.DataFrame, columns: Sequence[str] | None = None) -> pd.DataFrame | None:
        """Read a proxy child once through the actual authorized parent filing keys."""
        retained = source_read_window(self._config, table, self._context.run.as_of)
        if retained is None:
            return None
        clock = str(publication_policy(self._config, table).clock)
        shared = [key for key in Tables.def14a_llm.pk if key in table.pk]
        keys = parents.loc[retained.mask(parents[clock]), [*shared, clock]].drop_duplicates()
        keys[clock] = market_days(keys[clock])
        if keys.empty:
            return None
        if table == Tables.def14a_executive_comp and columns is None:
            columns = [*table.pk, clock, "total", "reconciles", *EXEC_COMPONENTS]
        if columns is not None:
            columns = list(dict.fromkeys([*columns, *shared, clock]))
        filters = {key: keys[key].unique().tolist() for key in shared}
        df_rows = self._store.load(table, columns=columns, project=columns is None, where={**filters, clock: retained}, optional=True)
        if df_rows is None:
            return None
        df_rows[clock] = market_days(df_rows[clock])
        return df_rows.merge(keys, on=[*shared, clock], how="inner", validate="many_to_one")

    def _load_fundamentals(self, window: Window | None) -> pd.DataFrame | None:
        """Load revenue and point-in-time shares used by governance features."""
        if window is None:
            return None
        df = load_source(
            self._context, Tables.sharadar_fundamentals_history, columns=["ticker", "as_of", "fiscal_end", "totalRevenue"], window=window
        )
        if df is None or df.empty:
            self._log.warning("No fundamentals history -> the pay-vs-revenue-growth misalignment feature is skipped.")
            return None
        return df

    # ---- panels ---- #
    def _governance_panel(self, frames: PriceFrames, window: WindowPlan) -> pd.DataFrame | None:
        """Load the raw governance frames, clean them once, then build the features."""
        windows = {
            table: source_read_window(self._config, table, window.read.end)
            for table in (Tables.def14a_llm, Tables.sec_8k_votes, Tables.sharadar_fundamentals_history)
        }
        df = self._load_def14a(windows[Tables.def14a_llm])
        if df is None:
            return None
        if not window.is_full:
            self._log.warning(_INCREMENTAL_WARNING, pd.Timestamp(window.write.start or window.frontier).date())

        raw = GovernanceInputs(
            def14a=df,
            exec_comp=self._load_proxy_child(Tables.def14a_executive_comp, df),
            directors=self._load_proxy_child(Tables.def14a_directors, df),
            director_comp=self._load_proxy_child(Tables.def14a_director_comp, df),
            votes=self._load_votes(windows[Tables.sec_8k_votes]),
            ownership=self._load_proxy_child(Tables.def14a_ownership, df, _OWNERSHIP_COLUMNS),
        )
        fundamentals = self._load_fundamentals(windows[Tables.sharadar_fundamentals_history])
        inputs, stats = clean_governance_inputs(raw, frames)
        lookahead = {k.split("|", 1)[1]: v for k, v in stats.items() if k.startswith(f"{PAY_BACKFILL_STAGE}|")}
        for key, value in stats.items():
            if not key.startswith(f"{PAY_BACKFILL_STAGE}|"):
                self._log.info("  governance clean | %s: %s", key, f"{value:,}")
        if lookahead:
            self._log.warning(PAY_BACKFILL_WARNING, "; ".join(f"{k}: {v:,}" for k, v in lookahead.items()))

        # Apply each source's configured eligibility once, using the retained lineage calendar.
        calendar = pd.DatetimeIndex(frames.shares_outstanding_pit.index) if frames.shares_outstanding_pit is not None else frames.trading_index
        timed = {}
        for field, table in (
            ("def14a", Tables.def14a_llm),
            ("exec_comp", Tables.def14a_executive_comp),
            ("directors", Tables.def14a_directors),
            ("director_comp", Tables.def14a_director_comp),
            ("votes", Tables.sec_8k_votes),
        ):
            source = getattr(inputs, field)
            if source is not None:
                clock = str(publication_policy(self._config, table).clock)
                timed[field] = publication_rows(self._config, table, source, calendar, clock)
        inputs = replace(inputs, **timed)
        if fundamentals is not None:
            fundamentals = publication_rows(self._config, Tables.sharadar_fundamentals_history, fundamentals, calendar, "as_of")

        panel, tally = build_governance_feature_panel(
            inputs.def14a,
            frames.peers,
            frames.trading_index,
            fundamentals=fundamentals,
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
