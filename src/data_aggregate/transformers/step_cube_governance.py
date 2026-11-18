"""Build the point-in-time governance cube part."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from time import perf_counter

import pandas as pd
from omegaconf import DictConfig

from src.context import Context
from src.data_aggregate.utils.common.incremental import (
    plan_window,
    write_part,
)
from src.data_aggregate.utils.common.inputs import input_windows, load_source
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
from src.utils.data_policy import publication_policy
from src.utils.filer_tables import def14a_evidence_mask
from src.utils.publication import point_in_time
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

    def __init__(self, context: Context, config: DictConfig) -> None:
        super().__init__(context=context, config=config)

        self._cfg = config.data_aggregate
        self._store = context.store

    def run(self) -> dict[Table, WindowPlan]:
        calendar = load_trading_calendar(self._context)
        window = plan_window(self._context, Tables.cube_part_governance, calendar)
        if window is None:
            return {}
        windows = input_windows(self._context, Tables.cube_part_governance, window.write)
        frames = self._load_frames(windows[Tables.cube_part_prices], calendar)

        output_frames = frames.on(frames.trading_index[frames.trading_index >= window.read.start], "close_split", "close_total")
        merger = PanelMerger(anchor=output_frames.skeleton())
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
    def _load_frames(self, window: Window | None, calendar: pd.DatetimeIndex) -> PriceFrames:
        """Load retained prices once for count lineage and conditional schema-support probes."""
        return load_price_frames(
            self._context, peers=load_peers_or_raise(self._context, self._config), calendar=calendar, fields=_PRICE_FIELDS, window=window
        )

    def _load_def14a(self, window: Window | None) -> pd.DataFrame | None:
        """Load usable scalar observations, excluding archives and retry-only parents."""
        df = load_source(self._context, Tables.def14a_llm, window=window)
        if df is not None:
            df = df.loc[def14a_evidence_mask(df)]
        if df is None or df.empty:
            return None
        self._log.info("Loaded %s: %s rows x %s cols", Tables.def14a_llm.name, len(df), len(df.columns))
        return df

    def _load_votes(self, window: Window | None) -> pd.DataFrame | None:
        """Load optional Item 5.07 vote records."""
        df = load_source(self._context, Tables.sec_8k_votes, window=window)
        if df is None or df.empty:
            self._log.warning("No %s -> the four shareholder-dissent families are skipped.", Tables.sec_8k_votes.name)
            return None
        self._log.info("Loaded %s: %s rows x %s cols", Tables.sec_8k_votes.name, len(df), len(df.columns))
        return df

    def _load_proxy_child(self, table: Table, window: Window | None, columns: Sequence[str] | None = None) -> pd.DataFrame | None:
        """Read configured child history, including donors without a usable scalar parent."""
        clock = str(publication_policy(self._config, table).column)
        if table == Tables.def14a_executive_comp and columns is None:
            columns = [*table.pk, clock, "total", "reconciles", *EXEC_COMPONENTS]
        if columns is not None:
            columns = list(dict.fromkeys([*columns, *Tables.def14a_llm.pk, clock]))
        return load_source(self._context, table, columns=columns, window=window)

    def _load_fundamentals(self, window: Window | None) -> pd.DataFrame | None:
        """Load revenue and point-in-time shares used by governance features."""
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
        existing_columns = self._store.columns(Tables.cube_part_governance)
        windows = input_windows(self._context, Tables.cube_part_governance, window.write)
        df = self._load_def14a(windows[Tables.def14a_llm])
        if df is None:
            return None
        if not window.is_full:
            self._log.warning(_INCREMENTAL_WARNING, pd.Timestamp(window.write.start or window.frontier).date())

        raw = GovernanceInputs(
            def14a=df,
            exec_comp=self._load_proxy_child(Tables.def14a_executive_comp, windows[Tables.def14a_executive_comp]),
            directors=self._load_proxy_child(Tables.def14a_directors, windows[Tables.def14a_directors]),
            director_comp=self._load_proxy_child(Tables.def14a_director_comp, windows[Tables.def14a_director_comp]),
            votes=self._load_votes(windows[Tables.sec_8k_votes]),
            ownership=self._load_proxy_child(Tables.def14a_ownership, windows[Tables.def14a_ownership], _OWNERSHIP_COLUMNS),
        )
        fundamentals = self._load_fundamentals(windows[Tables.sharadar_fundamentals_history])
        retained = frames
        # Prepare independent dated roots before any cross-source cleaning or fill.
        calendar = retained.trading_index
        timed = {}
        for field, table in (
            ("def14a", Tables.def14a_llm),
            ("exec_comp", Tables.def14a_executive_comp),
            ("directors", Tables.def14a_directors),
            ("director_comp", Tables.def14a_director_comp),
            ("votes", Tables.sec_8k_votes),
            ("ownership", Tables.def14a_ownership),
        ):
            source = getattr(raw, field)
            if source is not None:
                source = point_in_time(source, self._config, table, calendar)
                source[str(publication_policy(self._config, table).column)] = source["as_of"]
                timed[field] = source
        raw = replace(raw, **timed)
        if fundamentals is not None:
            fundamentals = point_in_time(fundamentals, self._config, Tables.sharadar_fundamentals_history, calendar)
        inputs, stats = clean_governance_inputs(raw, retained)
        frames = retained.on(retained.trading_index[retained.trading_index >= window.read.start], "close_split", "close_total")
        lookahead = {k.split("|", 1)[1]: v for k, v in stats.items() if k.startswith(f"{PAY_BACKFILL_STAGE}|")}
        for key, value in stats.items():
            if not key.startswith(f"{PAY_BACKFILL_STAGE}|"):
                self._log.info("  governance clean | %s: %s", key, f"{value:,}")
        if lookahead:
            self._log.warning(PAY_BACKFILL_WARNING, "; ".join(f"{k}: {v:,}" for k, v in lookahead.items()))

        panel = self._project_panel(inputs, fundamentals, frames)

        # Only native historical emissions justify retaining an all-unknown bounded column.
        missing = set() if window.is_full else set(existing_columns) - set(panel.columns)
        end = int(retained.trading_index.searchsorted(frames.trading_index[0]))
        span = len(frames.trading_index)
        warmup = int(frames.trading_index.searchsorted(window.write.start))
        advance = max(1, span - warmup)
        if missing:
            self._log.info("Governance bounded schema lacks retained columns: %s", sorted(missing))
        while missing and end:
            start = max(0, end - span)
            historical = self._project_panel(inputs, fundamentals, retained.on(retained.trading_index[start:end], "close_split", "close_total"))
            if set(historical.columns) - set(existing_columns):
                raise ValueError("Governance feature columns changed; an explicit full rebuild is required")
            proven = missing.intersection(historical.columns)
            for column in sorted(proven):
                support = historical.loc[historical[column].notna() & ~historical[column].isin([float("inf"), float("-inf")]), "date"]
                self._log.info("Governance historical schema support %s: %s finite cells, %s..%s", column, len(support), support.min(), support.max())
            panel = panel.assign(**{column: pd.Series(index=panel.index, dtype="float32") for column in proven})
            missing.difference_update(proven)
            end = max(0, end - advance)
        return panel

    def _project_panel(self, inputs: GovernanceInputs, fundamentals: pd.DataFrame | None, frames: PriceFrames) -> pd.DataFrame:
        """Project the unchanged governance calculations and report their support and cost."""
        started = perf_counter()
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
        self._log.info(
            "Governance projection %s..%s: %s sessions, %s feature columns, %.3fs",
            frames.trading_index.min(),
            frames.trading_index.max(),
            len(frames.trading_index),
            len(panel.columns) - 2,
            perf_counter() - started,
        )

        # Per-family row counts, the [0,1] rejects and the CEO-ballot split, one line each.
        # These are the numbers that make a thin family readable as a KNOWN property of the
        # ballot rather than a silent build failure -- `management_dissent_spread` in
        # particular is thin by construction (an exec-officer nominee stands in ~18% of
        # elections), and without the tally that looks identical to a broken join.
        for key in sorted(tally):
            self._log.info("  governance tally | %s: %s", key, f"{tally[key]:,}")
        return panel
