"""
step_cube_institutionals.py  (src/data_aggregate/transformers/step_cube_institutionals.py)
-----------------------------------------------------------------------------------------
The four INFORMED-CAPITAL panels -> `cube_part_institutionals`: all-filer 13F institutional
ownership, elite-manager (superinvestor) 13F, insider trading (Forms 3/4/5), and short
interest + fails-to-deliver.

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

import pandas as pd
from omegaconf import DictConfig

from src.data_store.schema import Table, Tables
from src.context import Context
from src.data_aggregate.utils.common.incremental import COLUMNS_CHANGED, plan_window, write_part
from src.data_aggregate.utils.common.panel_merge import PanelMerger
from src.data_aggregate.utils.common.parts import part_for
from src.data_aggregate.utils.common.peers_io import load_peers_or_raise
from src.data_aggregate.utils.common.price_frames import (
    PriceFrames, load_price_frames, load_trading_calendar,
)
from src.data_aggregate.utils.common.sources import project_existing
from src.data_aggregate.utils.institutionals.insider_features import build_insider_feature_panel
from src.data_aggregate.utils.institutionals.institutional_features import (
    build_institutional_feature_panel,
)
from src.data_aggregate.utils.institutionals.short_interest_features import (
    build_short_interest_feature_panel,
)
from src.data_aggregate.utils.institutionals.manager_selection import (
    eligibility, elite_weight, manager_concentration_score, selection_diagnostics,
)
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

    _FIELDS = ("close_split", "volume", "level_factor")

    def __init__(self, context: Context, config: DictConfig):
        super().__init__(context=context, config=config)
        self._cfg = config.build_cube
        self._part = part_for(Tables.cube_part_institutionals)
        self._store = context.store

    def run(self, full: bool = False) -> None:
        window = plan_window(self._store, Tables.cube_part_institutionals, full=full,
                             warmup=self._warmup(),
                             trading_index=load_trading_calendar(self._store))
        
        frames = self._load_frames(window.since)
        
        # shares outstanding for the market-cap scaling shared by 13F / insider panels
        shares = self._load_shares_out()

        merger = PanelMerger(self._log)
        merger.add(frames.skeleton().assign(_grid=1.0), "universe-grid")
        merger.add(self._institutional_panel(frames, shares), "institutional (13F)",
                   "No institutional (13F) features built.")
        merger.add(self._superinvestor_panel(frames, shares), "superinvestor (elite 13F)",
                   "No superinvestor (elite 13F) features built.")
        merger.add(self._insider_panel(frames, shares), "insider-trading",
                   "No insider-trading features built.")
        merger.add(self._short_interest_panel(frames), "short-interest",
                   "No short-interest features built.")
        
        panel = merger.to_long().drop(columns=["_grid"], errors="ignore")
        del frames, shares
        n = write_part(self._store, Tables.cube_part_institutionals, panel, window, drop_empty=True)
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

    def _load_source(self, table: Table) -> pd.DataFrame | None:
        """Load one source, PROJECTED to the columns its builder reads (see
        `utils/common/sources.py`); a table absent from that map loads in full.

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
        return df

    def _load_shares_out(self) -> pd.DataFrame | None:
        df = self._context.store.load(Tables.fundamentals_history, optional=True)
        if df is None:
            self._log.warning("No fundamentals history -> the market-cap-scaled ownership "
                              "features are skipped.")
            return None
        return df

    # ---- panels ---- #
    def _institutional_panel(self, frames: PriceFrames,
                             shares: pd.DataFrame | None) -> pd.DataFrame | None:
        """13F breadth, share/value accumulation, new-buyer / exiter counts, cluster buying,
        Herfindahl concentration, net put/call sentiment, ownership %, value/market-cap
        weight and net $ flow -- stamped point-in-time with the 45-day filing lag."""
        holdings = self._load_source(Tables.sec13f_hr)
        if holdings is None:
            return None
        return build_institutional_feature_panel(
            holdings, frames.peers, frames.trading_index,
            shares_out_history=shares, stock_close=frames.close_split,
            level_factor=frames.level_factor)

    def _superinvestor_panel(self, frames: PriceFrames,
                             shares: pd.DataFrame | None) -> pd.DataFrame | None:
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
            splits=self._load_source(Tables.prices_splits),
            selection=self._superinvestor_selector(),
            decay_halflife=float(self._decay_halflife("super")),
            stale_quarters=int(self._superinvestor_cfg().get("stale_quarters", 4)))

    def _superinvestor_cfg(self) -> dict:
        """`build_cube.institutionals.superinvestor`, or an empty mapping."""
        return self._cfg.get("institutionals", {}).get("superinvestor", {})

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

    def _insider_panel(self, frames: PriceFrames,
                       shares: pd.DataFrame | None) -> pd.DataFrame | None:
        """Fourteen Form 3/4/5 features: size-scaled open-market buying, cluster breadth,
        CEO/CFO/director legs, the buyer's own-history surprise, and the 10b5-1 split of
        selling. Point-in-time on the filing date (a Form 4 is due within ~2 business days).

        ⚠ THE DOLLAR FIGURES ARE SCOPED AND REPAIRED BEFORE THEY ARE SUMMED. Read raw,
        `value_usd` averages **$447,771,735,138** per Form 4 line -- 49.4% of which is ONE
        convertible-note row -- and the sells total $182,982,720tn against a real ~$1.45tn.
        After the scope cut and the price repair the mean is **$2,339,662** and the median
        $114,116. See `insider_quality` for which population each figure belongs to."""
        insider = self._load_source(Tables.insider_transactions)
        if insider is None:
            return None
        return build_insider_feature_panel(
            insider, frames.peers, frames.trading_index,
            shares_out_history=shares, stock_close=frames.close_split,
            level_factor=frames.level_factor,
            decay_halflife=float(self._decay_halflife("insider")))

    def _short_interest_panel(self, frames: PriceFrames) -> pd.DataFrame | None:
        """RegSHO short-volume ratio + its change, plus SEC fails-to-deliver (settlement
        stress). RegSHO is lagged one trading day; FTD by ~2 months (its publication delay)."""
        short = self._load_source(Tables.short_interest)
        fails = self._load_source(Tables.sec_fails_to_deliver)
        if short is None and fails is None:
            return None
        return build_short_interest_feature_panel(
            short, frames.peers, frames.trading_index,
            fails_history=fails, volume=frames.volume)
