"""
step_cube_governance.py  (src/data_aggregate/transformers/step_cube_governance.py)
---------------------------------------------------------------------------------
DEF 14A + Item 5.07 GOVERNANCE alpha -> `cube_part_governance`: shareholder dissent,
executive pay and its alignment, board structure, and the governance provisions that
entrench a board.

WHY ITS OWN PART, rather than a seventh panel in `extras`. Every other extras panel reads
ONE source table and emits one family. Governance reads five -- the proxy archive
(`def14a_llm`), its per-NEO and per-director children, the Item 5.07 vote record
(`sec_8k_votes`, the only SHAREHOLDER-CERTIFIED numbers in EDGAR), and `fundamentals_history`
for the revenue leg -- and emits nine feature families over them. It outgrew being one of six
panels in a step whose whole point is memory-bounded co-tenancy of unrelated sources.

TIME BASIS. Every source is FILING-space: an annual proxy or an 8-K stamped at its own
`as_of`, forward-filled onto the daily grid. So the features need no daily look-back at all,
and the part's warm-up exists for exactly one leg -- the 252-day trailing shareholder return
the pay-vs-performance family differences against pay growth. That is also the only reason
this step takes `close_total` (see `_FIELDS`).
"""
from __future__ import annotations

import pandas as pd
from omegaconf import DictConfig

from src.context import Context
from src.data_aggregate.utils.common.incremental import COLUMNS_CHANGED, plan_window, write_part
from src.data_aggregate.utils.common.panel_merge import PanelMerger
from src.data_aggregate.utils.common.parts import part_for
from src.data_aggregate.utils.common.peers_io import load_peers_or_raise
from src.data_aggregate.utils.common.price_frames import (
    PriceFrames, load_price_frames, load_trading_calendar,
)
from src.data_aggregate.utils.common.sources import project_existing
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


class StepCubeGovernance(Step):

    # ⚠ the ONLY feature step besides momentum/target allowed the total-return series:
    # the pay-vs-performance family needs a RETURN, not a level. What makes that safe here,
    # where it is forbidden in fundamentals/extras, is that this step builds no market cap
    # and no EV -- so `level_factor`, the other half of a correct level, is not read at all.
    # `close_split` is NOT a level input either: it is what `PriceFrames.skeleton()` keys the
    # universe grid on (the two bases share a non-null pattern, but close_total is derived
    # from close_split, so close_split is the one never null when the other is).
    _FIELDS = ("close_split", "close_total")

    def __init__(self, context: Context, config: DictConfig):
        super().__init__(context=context, config=config)

        self._cfg = config.build_cube
        self._part = part_for(Tables.cube_part_governance)
        self._store = context.store

    def run(self, full: bool = False) -> None:
        window = plan_window(self._store, Tables.cube_part_governance, full=full,
                             warmup=self._warmup(),
                             trading_index=load_trading_calendar(self._store))
        frames = self._load_frames(window.since)

        merger = PanelMerger(self._log)
        merger.add(frames.skeleton().assign(_grid=1.0), "universe-grid")
        merger.add(self._governance_panel(frames), "governance",
                   "No governance features built (def14a_llm empty — accrues as "
                   "fetch_def14a_llm runs).")

        panel = merger.to_long().drop(columns=["_grid"], errors="ignore")
        del frames
        n = write_part(self._store, Tables.cube_part_governance, panel, window, drop_empty=True)
        if n == COLUMNS_CHANGED:
            return self.run(full=True)

    def _warmup(self) -> int:
        override = self._cfg.get("incremental", {}).get("warmup_trading_days")
        return int(override) if override is not None else self._part.warmup_trading_days

    # ---- inputs ---- #
    def _load_frames(self, since: pd.Timestamp | None) -> PriceFrames:
        return load_price_frames(
            self._store, peers=load_peers_or_raise(self._context, self._config),
            fields=self._FIELDS, since=since)

    def _load_def14a(self) -> pd.DataFrame | None:
        """The proxy archive, read in FULL: one row per annual proxy per ticker, so the whole
        table is small enough that a projection would cost more in maintenance than it saves
        in memory (`utils/common/sources.py` deliberately lists no projection for it)."""
        df = self._store.load(Tables.def14a_llm, optional=True)
        if df is None or df.empty:
            return None
        self._log.info("Loaded %s: %s rows x %s cols",
                       Tables.def14a_llm.name, len(df), len(df.columns))
        return df

    def _load_votes(self) -> pd.DataFrame | None:
        """The Item 5.07 vote record, read in FULL and unprojected.

        `sec_8k_votes` is deliberately absent from `utils/common/sources.py::SOURCE_COLUMNS`:
        at ~35k rows over the whole 2010-2026 history it is two orders of magnitude smaller
        than anything that needs a projection, and it is WIDE (47 columns, 21 of them
        per-role vote sums) in a way that would make a column list a maintenance liability
        for no memory saved.

        `optional=True` because the vote archive accrues as the 8-K fetcher runs: a database
        part-way through its first fetch must degrade to "no dissent features", not raise.
        """
        df = self._store.load(Tables.sec_8k_votes, optional=True)
        if df is None or df.empty:
            self._log.warning("No %s -> the four shareholder-dissent families are skipped.",
                              Tables.sec_8k_votes.name)
            return None
        self._log.info("Loaded %s: %s rows x %s cols",
                       Tables.sec_8k_votes.name, len(df), len(df.columns))
        return df

    def _load_exec_comp(self) -> pd.DataFrame | None:
        """The per-NEO Summary Compensation Table rows, read in FULL and CLEANED ON READ.

        ~157k rows over 488 tickers -- small, and every column is needed: the exact CEO Pay
        Slice divides by the top five NEO totals, and `impute_exec_comp` reconstructs a NULL
        `total` from its seven components. That repair is not cosmetic: filings clearing the
        five-NEO bar go **8,026 -> 10,695 (+33%)**, which is the difference between a
        two-thirds-coverage feature and a near-complete one.

        `optional=True` -- absent, the pay slice is skipped and the other two pay families
        still build.
        """
        df = self._store.load(Tables.def14a_executive_comp, optional=True)
        if df is None or df.empty:
            self._log.warning("No %s -> the exact CEO Pay Slice family is skipped.",
                              Tables.def14a_executive_comp.name)
            return None
        df, stats = impute_exec_comp(df)
        if stats:
            self._log.info("NEO comp clean-on-read: %s",
                           ", ".join(f"{k}={v:,}" for k, v in stats.items()))
        return df

    def _load_directors(self) -> pd.DataFrame | None:
        """The per-DIRECTOR roster, PROJECTED and cleaned on read.

        134,490 rows over 12,239 filings, and the read is projected (`utils/common/sources.py`)
        because this table is wide with columns no governance builder touches -- `cik`,
        `gender`, `gender_basis` -- and it is read in the same build as the 21.7M-row 13F table.

        The clean-on-read is the per-PERSON fill (D37 agreement gate, D38 accrual). It runs
        BEFORE the board averages are derived from it, which is the whole mechanism: filling the
        child table is what makes the re-aggregation more than a no-op (§1.2).

        `optional=True` -- absent, the board averages fall back to the parent scalar plus
        interpolation exactly as phase 5 had them, and the board-quality family is skipped.
        """
        table = Tables.def14a_directors
        if not self._store.exists(table):
            self._log.warning("No %s -> the board averages stay on the parent scalar and the "
                              "board-quality family is skipped.", table.name)
            return None
        df = self._store.load(table, optional=True,
                              columns=project_existing(self._store.columns(table), table.name))
        if df is None or df.empty:
            return None
        df, stats = fill_director_attributes(df)
        if stats:
            self._log.info("Director clean-on-read: %s",
                           ", ".join(f"{k}={v:,}" for k, v in stats.items()))
        return df

    def _load_director_comp(self) -> pd.DataFrame | None:
        """The Item 402(k) Director Compensation Table, PROJECTED and cleaned on read.

        ⚠ Its coverage is a REGIME STAIRCASE (5.7% in the 1990s, 91.3% since 2020) because
        402(k) is a 2006 disclosure -- kind A under D25, reported in prose and never
        date-filtered in code. A build log showing ~64% here is showing the pre-2006 average of
        a post-2006 table, not a defect.

        ⚠ Six tickers have ZERO rows -- `APP`, `CRH`, `IBM`, `PANW`, `VTRS`, `WDAY` -- and that
        IS a defect: their Item 402(k) tables are not being parsed while everything else in the
        same document is. The count is logged rather than absorbed.
        """
        table = Tables.def14a_director_comp
        if not self._store.exists(table):
            self._log.warning("No %s -> the four director-pay features are skipped.", table.name)
            return None
        df = self._store.load(table, optional=True,
                              columns=project_existing(self._store.columns(table), table.name))
        if df is None or df.empty:
            return None
        df, stats = impute_director_comp(df)
        if stats:
            self._log.info("Director-comp clean-on-read: %s",
                           ", ".join(f"{k}={v:,}" for k, v in stats.items()))
        return df

    def _load_fundamentals(self) -> pd.DataFrame | None:
        """TWO columns are read off this frame, and the second is easy to miss.

        `totalRevenue` is the denominator leg of `ceo_pay_vs_revenue_growth`; absent, that one
        feature is skipped and the rest of the panel still builds.

        ⚠ `sharesOutstandingPit` IS THE DENOMINATOR OF `f_insider_ownership_pct` ITSELF, via
        `panel.economic_ownership`. A dual-class proxy prints per-class percentages and total
        voting power and NO combined economic column, so insider economic ownership is not a
        disclosed fact for those 104 tickers -- it is computed as the group's filed share count
        over this. Loaded in FULL on purpose: `fundamentals_history` is deliberately absent
        from `sources.SOURCE_COLUMNS`, and **if a projection is ever added for it, this column
        has to be in the list** or the ownership feature silently reverts to reporting whatever
        the extraction happened to read off a per-class column.
        """
        df = self._store.load(Tables.fundamentals_history, optional=True)
        if df is None or df.empty:
            self._log.warning("No fundamentals history -> the pay-vs-revenue-growth "
                              "misalignment feature is skipped.")
            return None
        return df

    # ---- panels ---- #
    def _governance_panel(self, frames: PriceFrames) -> pd.DataFrame | None:
        """CEO pay growth, pay-vs-revenue-growth misalignment, pay ratio, board
        independence / diversity / tenure, say-on-pay -- point-in-time from each proxy's
        `as_of` -- plus the four Item 5.07 shareholder-dissent families, point-in-time from
        each 8-K's `filing_date`.

        The raw extraction table is never mutated: the LLM-extracted proxy rows are cleaned
        ON READ, deducing cells the extraction left NaN from the identities that hold within
        and across a ticker's filings."""
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
            self._log.info("DEF 14A clean-on-read: deduced %d missing cells across %d rules "
                           "(raw table untouched).", sum(stats.values()), len(stats))
        # Step 4 of the precedence chain has now run, so the three-valued provenance can be
        # completed and said out loud: how much of each board average is EVIDENCE.
        df, sstats = finalize_board_source(df)
        for k, v in sstats.items():
            self._log.info("  board aggregate provenance | %s: %s", k, f"{v:,}")

        panel, tally = build_governance_feature_panel(
            df, frames.peers, frames.trading_index,
            fundamentals_history=self._load_fundamentals(),
            votes=self._load_votes(),
            exec_comp=self._load_exec_comp(),
            directors=directors,
            director_comp=self._load_director_comp(),
            close_total=frames.close_total)
        # Per-family row counts, the [0,1] rejects and the CEO-ballot split, one line each.
        # These are the numbers that make a thin family readable as a KNOWN property of the
        # ballot rather than a silent build failure -- `management_dissent_spread` in
        # particular is thin by construction (an exec-officer nominee stands in ~18% of
        # elections), and without the tally that looks identical to a broken join.
        for key in sorted(tally):
            self._log.info("  governance tally | %s: %s", key, f"{tally[key]:,}")
        return panel
