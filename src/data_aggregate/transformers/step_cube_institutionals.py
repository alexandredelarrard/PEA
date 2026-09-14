"""
step_cube_institutionals.py  (src/data_aggregate/transformers/step_cube_institutionals.py)
-----------------------------------------------------------------------------------------
The INFORMED-CAPITAL panels -> `cube_part_institutionals`. Four read a source, two are derived
from the first four:

    institutional    all-filer 13F ownership                   `ic_inst_*`   (sec13f_hr)
    superinvestor    elite-manager 13F conviction              `ic_super_*`  (sec13f_manager_holdings)
    insider          Forms 3/4/5 open-market trades            `ic_insider_*` (insider_transactions)
    beneficial-own.  Schedule 13D / 13G                        `ic_act_*` / `ic_bo_*`
    short-flow       RegSHO short VOLUME + fails-to-deliver    `ic_shortvol_*` / `ic_ftd_*`
    conditioning     the price path since each family's last   `ic_sig_*`     (derived)
                     disclosure -- the layer that makes the
                     panel move on a day with no filing
    cross-source     how many independent families / ACTORS    `ic_xs_*`      (derived)
                     agree on the name

THE TWO DERIVED PANELS READ A SINK, NOT THE SOURCES AGAIN (`utils/institutionals/sink.py`).
The insider event dates are the output of a 2M-row scope-and-repair pass and the elite ones of
the per-manager availability join, so each source panel drops its event dates, its bullish
actors and its handful of declared signal frames into a `ConditioningSink` on the way past.
Re-deriving either would double the most expensive read in the step.

WHY IT IS NO LONGER `extras`. "Extras" named no shared property, so it accreted whatever had
nowhere else to go -- a blended retail-attention panel sat here for months, defined but never
called from `run()`. Every panel that remains is a MOVE BY A DISCLOSING CAPITAL ALLOCATOR,
which is both the name and the membership rule: a panel belongs here iff its source is a
filing somebody was legally required to make. The attention panel is deleted, not moved.

THIS IS THE BIGGEST WIN OF THE COARSENING. These were six separate DAG tasks, each
re-running the whole price prologue, and three of them had to be SERIALIZED behind one
another (institutional -> superinvestor -> fundamental) purely to keep them off each
other's memory. Now they are one step reading a 160-day window, and the serialization
constraint disappears because the sub-steps run sequentially by construction.

MEMORY DISCIPLINE. Each `_*_panel` loads its own source into a LOCAL and returns a panel,
so the projected `sec13f_hr` read (the ~21.7M-row table, cut to 8 columns) and
`insider_transactions` are never resident at the same time. Peak is the largest single
source plus the accumulating panel.
"""
from __future__ import annotations

from typing import Sequence

import pandas as pd
from omegaconf import DictConfig

from src.data_store.schema import Table, Tables
from src.context import Context
from src.data_aggregate.utils.common.incremental import (
    COLUMNS_CHANGED, PartWindow, plan_window, write_part)
from src.data_aggregate.utils.common.panel_merge import PanelMerger
from src.data_aggregate.utils.common.parts import part_for
from src.data_aggregate.utils.common.peers_io import load_peers_or_raise
from src.data_aggregate.utils.common.price_frames import (
    PriceFrames, load_price_frames, load_trading_calendar,
)
from src.data_aggregate.utils.common.sources import project_existing
from src.data_aggregate.utils.institutionals.cross_source_features import (
    build_cross_source_panel,
)
from src.data_aggregate.utils.institutionals.insider_features import build_insider_feature_panel
from src.data_aggregate.utils.institutionals.institutional_features import (
    COVERAGE_BREAK_DEFAULT, build_institutional_feature_panel,
)
from src.data_aggregate.utils.institutionals.short_flow_features import (
    build_short_flow_feature_panel,
)
from src.data_aggregate.utils.institutionals.manager_selection import (
    eligibility, elite_weight, manager_concentration_score, selection_diagnostics,
)
from src.data_aggregate.utils.institutionals.ownership_features import (
    build_ownership_feature_panel,
)
from src.data_aggregate.utils.institutionals.signal_conditioning import (
    EXCURSION_LOOKBACK, build_signal_conditioning_panel,
)
from src.data_aggregate.utils.institutionals.sink import ConditioningSink
from src.data_aggregate.utils.institutionals.superinvestor_features import (
    build_superinvestor_feature_panel, load_superinvestor_holdings,
)
from src.utils.step import Step
from src.utils.superinvestor_roster import (
    first_snapshot_date, roster_as_of, roster_cik_union, roster_map_as_of,
)

# NOTE there is deliberately no SOURCE_COLUMNS map here. This module used to carry its own
# copy of one, which `_load_source` never read -- the projection comes from
# `utils/common/sources.py` via `project_existing`. The copy had already drifted (it omitted
# short_interest's `short_interest` / `avg_daily_volume` legs), which is the exact failure the
# parts registry was built to end: two declarations of one fact, only one of them live.


class StepCubeInstitutionals(Step):

    #: ⚠ SIX price fields, and each one is load-bearing. `close_split` is the LEVEL basis (market
    #: cap, and the insider cost anchor's own basis); `close_total` the RETURN basis every
    #: `ic_sig_*` return and excursion is taken on; `volume` backs ADV20 and the RegSHO coverage
    #: measurement; `level_factor` is the spinoff `S(d)` a market cap must carry; `sector_ret`
    #: makes the conditioning returns sector-RESIDUAL; `ret` is the persisted daily return the
    #: 20-day realized vol comes from (persisted rather than recomputed, so a trimmed
    #: incremental window reproduces the full build's first row).
    _FIELDS = ("close_split", "close_total", "volume", "level_factor", "sector_ret", "ret")

    def __init__(self, context: Context, config: DictConfig):
        super().__init__(context=context, config=config)
        self._cfg = config.build_cube
        self._part = part_for(Tables.cube_part_institutionals)
        self._store = context.store

    def run(self, full: bool = False) -> None:
        panel, window = self.build_panel(full=full)
        n = write_part(self._store, Tables.cube_part_institutionals, panel, window, drop_empty=True)
        if n == COLUMNS_CHANGED:
            return self.run(full=True)

    def build_panel(self, full: bool = False) -> tuple[pd.DataFrame, PartWindow]:
        """The merge chain, WITHOUT the write. Extracted so `validate institutionals` scores
        the same panel this step persists instead of carrying a second copy of the chain --
        the "two declarations of one fact" failure the `SOURCE_COLUMNS` note above records.

        `frames` and `shares` are locals and die with the frame on return, which is what the
        `del` before `write_part` used to buy."""
        window = plan_window(self._store, Tables.cube_part_institutionals, full=full,
                             warmup=self._warmup(),
                             trading_index=load_trading_calendar(self._store))

        # ⚠ `None`, NOT `window.since`, AND THAT IS THE WHOLE INCREMENTAL CONTRACT OF THIS PART.
        # The window still decides what gets WRITTEN -- `write_part` slices `rows` to the tail
        # itself -- but the panel is COMPUTED over the full trading calendar every time.
        #
        # A warm-up is the right instrument for a BOUNDED look-back: hand a 252-day rolling
        # statistic 390 days of context and its first rewritten row matches a rebuild exactly.
        # Six families here have no finite look-back at all -- age-since-last-event
        # (`ic_sig_*_age_days`, `ic_act_campaign_age_days`), is-this-holder-new
        # (`ic_bo_new_holder`), distinct-actors-to-date (`ic_xs_*`), the expanding owner
        # surprise, and every price path measured from an anchor that may be fifteen years
        # back. No value of `warmup_trading_days` reaches them.
        #
        # Worse than incomplete: WRONG. `decay.snap_to_grid` moves each event onto the first
        # trading day >= its date, so on a trimmed grid every event that predates the window
        # snaps ONTO the window's first day rather than being seen as historic. A 2010 13D
        # became a 2025 13D. Measured over a 567-day window on 2026-09-12, before this change:
        # 76 of 127 columns drifted from a rebuild and the drift reached the newest row --
        # the only row an append actually writes. `f_ic_bo_new_holder` was wrong on 490 of 491
        # tickers; `f_ic_sig_insider_age_days` by a median 674 days and up to 4,412.
        #
        # THE COST IS SMALL BECAUSE THE READS WERE NEVER TRIMMED. `_load_source` reads every
        # event table in full on both paths (22.5M `sec13f_hr` rows, 2.0M insider rows) -- that
        # is the dominant I/O and it is unchanged. What incremental still buys is the WRITE:
        # an append of ~5 dates against a 2.6M-row replace.
        price_frames = self._load_frames()

        # shares outstanding for the market-cap scaling shared by 13F / insider panels
        shares = self._load_shares_out()

        # `prices_splits` is read ONCE here rather than per panel: three families need the
        # same restatement (13F share counts, elite prior-quarter counts, insider filed
        # prices) and the table is small enough that the cost is the read, not the memory.
        splits = self._load_source(Tables.prices_splits)

        # What the two DERIVED panels consume. Filled by the four source panels as they run,
        # so nothing here re-reads a source -- see `sink.py`.
        sink = ConditioningSink()

        merger = PanelMerger(self._log)
        merger.add(price_frames.skeleton().assign(_grid=1.0), "universe-grid")
        merger.add(self._institutional_panel(price_frames, shares, splits), "institutional (13F)",
                   "No institutional (13F) features built.")
        merger.add(self._superinvestor_panel(price_frames, shares, splits, sink),
                   "superinvestor (elite 13F)",
                   "No superinvestor (elite 13F) features built.")
        merger.add(self._insider_panel(price_frames, shares, sink), "insider-trading",
                   "No insider-trading features built.")
        merger.add(self._short_flow_panel(price_frames, shares, splits, sink), "short-flow",
                   "No short-flow features built.")
        merger.add(self._ownership_panel(price_frames, sink), "beneficial-ownership",
                   "No beneficial-ownership features built.")
        merger.add(self._conditioning_panel(price_frames, splits, sink), "price-conditioning",
                   "No price-conditioning features built.")
        merger.add(self._cross_source_panel(price_frames, sink), "cross-source",
                   "No cross-source features built.")

        return self._restrict_to_grid(merger.to_long()), window

    def _restrict_to_grid(self, long: pd.DataFrame) -> pd.DataFrame:
        """Keep only rows the universe grid put there, then drop the marker.

        ⚠ THE `_grid` COLUMN WAS BEING DROPPED UNUSED, and that was a real hole.
        `PanelMerger.to_long` is an OUTER-aligned concat, so every (date, ticker) a source
        table carries but `cube_part_prices` does not was appended to the panel. Measured on
        the 2026-09-11 full build: **12 tickers / 14,982 rows** with no price row anywhere --
        `EA`, `EQR` and `AVB`, which are absent from `prices` entirely, plus nine recent
        spin-offs (`GEV`, `KVUE`, `GEHC`, `VLTO`, `SOLV`, `SNDK`, `Q`, `HONA`, `FDXF`).

        `assemble-cube` joins the parts onto a base built from `cube_part_prices`, so none of
        those rows ever reached the model. They still mattered: this part is what Phase 2.8's
        gates measure, every other part carries 491 tickers, and a coverage sheet computed
        over 503 is a sheet about a universe that does not exist.

        ⚠ THE RESTRICTION IS ON THE PAIR, NOT THE TICKER, and that is the bigger half of it.
        Measured on the 2026-09-12 build it removes **565,895 rows across 202 tickers** --
        mostly `AMZN`-class names on dates before their first price bar, where a 13F holding
        or an insider filing exists and the price grid does not. Almost all of those were
        already being discarded downstream as all-NaN rows (the write-side drop falls 31.6% ->
        19.0%), so the net row change is small; what changes is that the panel now states the
        same universe as every other part instead of arriving at it by accident.
        """
        if "_grid" not in long.columns:
            return long
        on_grid = long["_grid"].notna()
        if not on_grid.all():
            off = long.loc[~on_grid, ["date", "ticker"]].copy()
            off["ticker"] = off["ticker"].astype(str)
            on_tickers = set(long.loc[on_grid, "ticker"].astype(str))
            # ⚠ SAY WHICH AXIS. Most of these are IN-universe tickers on a date the price grid
            # does not cover -- before the name's first price bar, or after its last -- and
            # only a few are tickers the grid has never heard of. A message that called AMZN
            # "not in cube_part_prices" would send the next reader after a phantom.
            never = sorted(set(off["ticker"]) - on_tickers)
            self._log.warning(
                "cube_part_institutionals: dropped %s row(s) off the price grid, across %s "
                "ticker(s). %s of them appear nowhere in cube_part_prices (%s); the rest are "
                "in-universe names on dates the grid does not cover.",
                f"{int((~on_grid).sum()):,}", off['ticker'].nunique(), len(never),
                ", ".join(never[:15]) or "none")
        return long.loc[on_grid].drop(columns=["_grid"])

    def _warmup(self) -> int:
        override = self._cfg.get("incremental", {}).get("warmup_trading_days")
        return int(override) if override is not None else self._part.warmup_trading_days

    # ---- inputs ---- #
    def _load_frames(self) -> PriceFrames:
        """⚠ NO `since` PARAMETER, unlike every sibling step. This part's grid is the FULL
        trading calendar on both paths -- see the block in `build_panel`. Taking the argument
        and always passing `None` would leave the next reader thinking it was a choice."""
        return load_price_frames(
            self._store, peers=load_peers_or_raise(self._context, self._config),
            fields=self._FIELDS, since=None)

    def _load_source(self, table: Table,
                     universe: Sequence[str] | None = None) -> pd.DataFrame | None:
        """Load one source, PROJECTED to the columns its builder reads (see
        `utils/common/sources.py`); a table absent from that map loads in full.

        Pass `universe` for any table whose tickers become a feature CROSS-SECTION -- see
        `_scope_to_universe`. Lookup tables (`cusip_ticker_map`) and the split calendar are
        deliberately left whole: they are joined against, not ranked over.

        The projection is narrowed to the columns that actually exist: `read_table` resolves
        each via `tbl.c[name]` and raises `KeyError` otherwise, so demanding a column the
        builder treats as optional (short interest's `ic_shortvol_days_to_cover` inputs) killed
        the read instead of degrading.

        ⚠ TAKES A `Table`, NEVER A NAME STRING, and `project_existing` is keyed on
        `table.name`. Five registry entries have an attribute name that differs from their
        physical table (`Tables.short_interest` -> `sec_short_interest`), so a string literal
        here is a silent `exists() is False` and a whole feature family that never builds --
        which is exactly what `"short_interest"` did to the three `ic_shortvol_*` features
        while 956,640 rows sat in the table unread."""
        if not self._context.store.exists(table):
            self._log.warning("%s is absent -> its features are skipped.", table.name)
            return None
        columns = project_existing(self._store.columns(table), table.name)
        df = self._context.store.load(table, columns=columns, optional=True)
        if df is None:
            return None
        self._log.info("Loaded %s: %s rows x %s cols", table.name, len(df), len(df.columns))
        return self._scope_to_universe(df, table.name, universe)

    def _scope_to_universe(self, df: pd.DataFrame, label: str,
                           universe: Sequence[str] | None) -> pd.DataFrame:
        """Cut a source table to the cube's own cross-section.

        ⚠ THIS IS ABOUT `_xs`, NOT ABOUT ROW COUNTS. D26 defines `_xs` as a same-day
        percentile "ranking a ticker against every other ticker on that date", and in the cube
        that set is the universe. The source tables are wider than it -- `sec13f_hr` carries
        501 tickers and `sec_fails_to_deliver` 501 against the universe's 491 -- so every
        `_xs` leg in this part was ranking over a denominator no other part uses, and
        `peer_relative` was resolving baskets partly out of names the model never sees.
        `superinvestor_features` already took a `universe=` for exactly this reason; this
        applies the same rule to the other four families, at the read, where it also saves
        the work rather than doing it and discarding it.

        Off-universe rows are reported once per table rather than dropped in silence: the set
        is the S&P 500 membership boundary and it moves, so a jump in this number is a
        universe problem to look at, not noise.
        """
        if universe is None or "ticker" not in df.columns:
            return df
        keep = df["ticker"].astype(str).isin(set(map(str, universe)))
        if keep.all():
            return df
        dropped = df.loc[~keep, "ticker"].astype(str)
        self._log.info("%s: %s of %s rows are outside the %s-name universe (%s ticker(s): %s)"
                       " -- cut so the `_xs` cross-section is the cube's",
                       label, f"{int((~keep).sum()):,}", f"{len(df):,}", len(universe),
                       dropped.nunique(), ", ".join(sorted(dropped.unique())[:15]))
        return df.loc[keep]

    def _load_shares_out(self) -> pd.DataFrame | None:
        df = self._context.store.load(Tables.fundamentals_history, 
                                      columns=['ticker', 'as_of', 'sharesOutstandingPit'],
                                        optional=True)
        if df is None:
            self._log.warning("No fundamentals history -> the market-cap-scaled ownership "
                              "features are skipped.")
            return None
        return df

    # ---- panels ---- #
    def _institutional_panel(self, frames: PriceFrames, shares: pd.DataFrame | None,
                             splits: pd.DataFrame | None) -> pd.DataFrame | None:
        """The registry's eleven `ic_inst_*`: breadth SHARE (D28), split-restated share
        accumulation, new-buyer / exit ratios, cluster buying, Herfindahl concentration, net
        put/call sentiment, ownership %, value/market-cap weight and net $ flow -- stamped
        point-in-time with the 45-day filing lag.

        No sink: an all-filer aggregate has no single event date to condition on (registry
        section 7), and none of the cross-source inputs is an `ic_inst_*` feature."""
        holdings = self._load_source(Tables.sec13f_hr, frames.universe)
        if holdings is None:
            return None
        
        return build_institutional_feature_panel(
            holdings, frames.peers, frames.trading_index,
            shares_out_history=shares, stock_close=frames.close_split,
            level_factor=frames.level_factor, splits=splits,
            break_pct=float(self._institutionals_cfg().get("coverage_break_pct",
                                                           COVERAGE_BREAK_DEFAULT)))

    def _superinvestor_panel(self, frames: PriceFrames, shares: pd.DataFrame | None,
                             splits: pd.DataFrame | None,
                             sink: ConditioningSink) -> pd.DataFrame | None:
        """Elite-manager 13F conviction (Dataroma superinvestors), layered ON TOP of the
        all-filer features.

        Reads `sec13f_manager_holdings` -- the roster managers' WHOLE books -- not the
        S&P500-filtered `sec13f_hr`, because a portfolio weight is only comparable across
        managers when its denominator is the whole book (README Finding 2: the index sleeve
        is a median 47% of positions and swings 8%-100% by manager). `cusip_ticker_map`
        resolves the S&P500 leg for the numerator without touching that denominator, and
        `prices_splits` restates a prior quarter's share count so a 20-for-1 split is not
        read as a 1,900% purchase.

        ⚠ THE READ SCOPE IS THE EVER-LISTED UNION, NOT TODAY'S ROSTER (D19/D20). Loading
        today's 83 managers drops the 19 culled managers that have a book -- Arlington
        Value, Wintergreen, RBS Partners, Tilson among them -- and Dataroma's cull is
        performance-driven, so it is survivorship correlated with the selection criterion.
        Narrowing to who was listed at `q` is the selector's job and it can only narrow
        what was read."""
        union = roster_cik_union(self._context)
        if not union:
            self._log.warning("`superinvestor_roster` has no snapshot -> elite 13F features "
                              "skipped (run `data_extract superinvestors --seed`).")
            return None
        holdings = load_superinvestor_holdings(self._context, union)
        if holdings is None or holdings.empty:
            self._log.warning("No elite-manager 13F holdings -> superinvestor features skipped.")
            return None
        self._log.info("Elite 13F books: %s rows across %s ever-listed managers (%s on "
                       "today's roster)", len(holdings), len(union),
                       len(roster_map_as_of(self._context)))
        return build_superinvestor_feature_panel(
            holdings, union, frames.peers, frames.trading_index,
            shares_out_history=shares, stock_close=frames.close_split,
            level_factor=frames.level_factor,
            cusip_map=self._load_source(Tables.cusip_ticker_map),
            universe=frames.universe,
            splits=splits,
            selection=self._superinvestor_selector(),
            decay_halflife=float(self._decay_halflife("super")),
            stale_quarters=int(self._superinvestor_cfg().get("stale_quarters", 4)),
            sink=sink)

    def _institutionals_cfg(self) -> dict:
        """`build_cube.institutionals`, or an empty mapping."""
        return self._cfg.get("institutionals", {})

    def _superinvestor_cfg(self) -> dict:
        """`build_cube.institutionals.superinvestor`, or an empty mapping."""
        return self._institutionals_cfg().get("superinvestor", {})

    def _superinvestor_selector(self):
        """`sel(m, q)` as a callable over the manager state, or None for the flat 1.0.

        A CALLABLE rather than a Series because the selection needs the state the panel
        builder has just computed AND a roster accessor only this step can close over.

        `build_cube.institutionals.superinvestor.selection` drives it:
        `{mode: top_k|continuous|off, k, min_quarters, min_positions}`. Defaults are the
        measured ones -- see `manager_selection`, where the whole-book basis and the absent
        index-position floor are each worth ~1pp/yr on the corrected top-15 basket.

        ⚠ `roster_as_of` RETURNS THE EMPTY SET BEFORE THE FIRST SNAPSHOT (2013-01-01), which
        would zero every manager in 2011-2012 and delete two years of panel. Flooring the
        lookup at the first snapshot extrapolates the oldest roster backwards, which is a
        compromise, but the alternative -- today's roster -- is the exact bias D20 exists to
        remove.
        """
        cfg = self._superinvestor_cfg().get("selection", {})
        # ⚠ UNQUOTED `off` IN YAML IS THE BOOLEAN False, not the string -- PyYAML is 1.1 and
        # `off`/`no`/`on` are all booleans there. Accept both spellings rather than let a
        # `mode: off` that parsed as False fall through to `str(False) == "False"` and raise
        # deep inside `elite_weight`.
        raw = cfg.get("mode", "off")
        mode = "off" if raw in (False, None, "") else str(raw).lower()
        if mode in ("off", "none", "flat", "false"):
            return None
        first = first_snapshot_date(self._context)
        if first is None:
            self._log.warning("No roster snapshot -> selection falls back to flat 1.0.")
            return None
        k = int(cfg.get("k", 15))
        min_quarters = int(cfg.get("min_quarters", 4))
        min_positions = int(cfg.get("min_positions", 0))

        # ⚠ MEMOISED. `roster_as_of` reloads the WHOLE roster table on every call, and
        # `eligibility` asks it once per period -- 60 full-table reads per build without
        # this. The cache is per-selector-call, so a rebuild still re-reads the table once.
        cache: dict[pd.Timestamp, set[str]] = {}

        def roster_at(q) -> set[str]:
            key = max(pd.Timestamp(q), first)
            if key not in cache:
                cache[key] = roster_as_of(self._context, key)
            return cache[key]

        def selector(state: pd.DataFrame) -> pd.Series:
            ok = eligibility(state, roster_at, min_quarters=min_quarters,
                             min_positions=min_positions)
            scored = manager_concentration_score(state, eligible=ok)
            sel = elite_weight(scored, mode=mode, k=k)
            # `avail` is attached by the panel builder before it calls this, so the
            # diagnostic reads the same availability grid the selection was ranked on.
            diag = selection_diagnostics(sel, state)
            if not diag.empty:
                settled = diag[diag["n_public"] >= k]
                self._log.info(
                    "elite selection (%s, k=%s): %s of %s manager-quarters eligible, "
                    "live set %s-%s managers, median churn %s per filing date",
                    mode, k, int(ok.sum()), len(ok),
                    int(settled["n_selected"].min()) if len(settled) else 0,
                    int(settled["n_selected"].max()) if len(settled) else 0,
                    int(settled["churn"].median()) if len(settled) else 0)
            return sel

        return selector

    def _decay_halflife(self, family: str) -> float:
        """Trading-day half-life for `family`'s event decay, from
        `build_cube.institutionals.decay_halflife`. A tunable number, so it lives in the
        config and not in `constants/`."""
        cfg = self._cfg.get("institutionals", {}).get("decay_halflife", {})
        return float(cfg.get(family, 63))

    def _insider_panel(self, frames: PriceFrames, shares: pd.DataFrame | None,
                       sink: ConditioningSink) -> pd.DataFrame | None:
        """Fourteen Form 3/4/5 features: size-scaled open-market buying, cluster breadth,
        CEO/CFO/director legs, the buyer's own-history surprise, and the 10b5-1 split of
        selling. Point-in-time on the filing date (a Form 4 is due within ~2 business days).

        ⚠ THE DOLLAR FIGURES ARE SCOPED AND REPAIRED BEFORE THEY ARE SUMMED. Read raw,
        `value_usd` averages **$447,771,735,138** per Form 4 line -- 49.4% of which is ONE
        convertible-note row -- and the sells total $182,982,720tn against a real ~$1.45tn.
        After the scope cut and the price repair the mean is **$2,339,662** and the median
        $114,116. See `insider_quality` for which population each figure belongs to."""
        insider = self._load_source(Tables.insider_transactions, frames.universe)
        if insider is None:
            return None
        return build_insider_feature_panel(
            insider, frames.peers, frames.trading_index,
            shares_out_history=shares, stock_close=frames.close_split,
            level_factor=frames.level_factor,
            decay_halflife=float(self._decay_halflife("insider")),
            sink=sink)

    def _short_flow_panel(self, frames: PriceFrames, shares: pd.DataFrame | None,
                          splits: pd.DataFrame | None, sink: ConditioningSink) -> pd.DataFrame | None:
        """Volume-weighted RegSHO short-VOLUME ratios (5/20/60d), their self-history z, the
        two price-conditional interactions and short turnover, plus SEC fails-to-deliver
        (settlement stress) as a share of shares outstanding and of ADV20. RegSHO is lagged
        one trading day; FTD by ~2 months (its publication delay)."""
        short = self._load_source(Tables.short_interest, frames.universe)
        fails = self._load_source(Tables.sec_fails_to_deliver, frames.universe)
        if short is None and fails is None:
            return None
        return build_short_flow_feature_panel(
            short, frames.peers, frames.trading_index,
            fails_history=fails, volume=frames.volume, shares_out_history=shares,
            close_total=frames.close_total, splits=splits, sink=sink)

    def _ownership_panel(self, frames: PriceFrames,
                         sink: ConditioningSink) -> pd.DataFrame | None:
        """Schedule 13D activist (`ic_act_*`) and 13G passive-ownership (`ic_bo_*`) events.
        `sec_13d_transactions` is deliberately not read -- see `ownership_features`'s module
        docstring."""
        d13 = self._load_source(Tables.sec_13d, frames.universe)
        d13g = self._load_source(Tables.sec_13g, frames.universe)
        if d13 is None and d13g is None:
            return None
        return build_ownership_feature_panel(
            d13, d13g, frames.peers, frames.trading_index,
            decay_halflife_act=float(self._decay_halflife("act")),
            decay_halflife_bo=float(self._decay_halflife("bo")),
            sink=sink)

    def _conditioning_panel(self, frames: PriceFrames, splits: pd.DataFrame | None,
                            sink: ConditioningSink) -> pd.DataFrame | None:
        """The `ic_sig_*` layer: days since each family's last disclosure and the price path
        since, sector-residualized and vol-scaled. THE PANEL'S ONLY DAILY-MOVING FAMILY, and
        the direct evidence for the two-layer architecture (report acceptance test #13)."""
        if not sink.events:
            return None
        return build_signal_conditioning_panel(
            sink.events, frames.peers, frames.trading_index,
            close_total=frames.close_total, close_split=frames.close_split,
            sector_ret=frames.sector_ret, ret=frames.ret, splits=splits,
            excursion_lookback=int(self._institutionals_cfg().get("excursion_lookback",
                                                                  EXCURSION_LOOKBACK)))

    def _cross_source_panel(self, frames: PriceFrames,
                            sink: ConditioningSink) -> pd.DataFrame | None:
        """The `ic_xs_*` layer: how many independent families -- and how many distinct
        ACTORS -- are flagging this name at once, plus the both-sides conflict flag."""
        if not sink.signals and not sink.actors:
            return None
        return build_cross_source_panel(
            sink, frames.peers, frames.trading_index,
            universe=pd.Index(frames.universe))
