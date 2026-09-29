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

from collections.abc import Sequence

import pandas as pd
from omegaconf import DictConfig

from src.constants.constants import F13_REVISION_MIN_MOVE, F13_SETTLE_TRADING_DAYS
from src.context import Context
from src.data_aggregate.utils.common.incremental import COLUMNS_CHANGED, PartWindow, plan_window, write_part
from src.data_aggregate.utils.common.panel_merge import PanelMerger
from src.data_aggregate.utils.common.parts import part_for
from src.data_aggregate.utils.common.price_frames import (
    PriceFrames,
    load_trading_calendar,
)
from src.data_aggregate.utils.institutionals import frontiers as institutional_frontiers
from src.data_aggregate.utils.institutionals import inputs as institutional_inputs
from src.data_aggregate.utils.institutionals.availability import InstitutionalAvailability
from src.data_aggregate.utils.institutionals.cross_source_features import (
    build_cross_source_panel,
)
from src.data_aggregate.utils.institutionals.insider_features import build_insider_feature_panel
from src.data_aggregate.utils.institutionals.insider_sources import (
    as_quarter,
    bulk_complete_through,
    overlay_insider_sources,
)
from src.data_aggregate.utils.institutionals.institutional_features import (
    COVERAGE_BREAK_DEFAULT,
    build_institutional_feature_panel,
)
from src.data_aggregate.utils.institutionals.manager_selection import (
    eligibility,
    elite_weight,
    manager_concentration_score,
    selection_diagnostics,
)
from src.data_aggregate.utils.institutionals.ownership_features import (
    build_ownership_feature_panel,
)
from src.data_aggregate.utils.institutionals.short_flow_features import (
    build_short_flow_feature_panel,
)
from src.data_aggregate.utils.institutionals.signal_conditioning import (
    EXCURSION_LOOKBACK,
    build_signal_conditioning_panel,
)
from src.data_aggregate.utils.institutionals.sink import ConditioningSink
from src.data_aggregate.utils.institutionals.superinvestor_features import (
    build_superinvestor_feature_panel,
    load_superinvestor_holdings,
)
from src.data_store.schema import Table, Tables
from src.utils.step import Step
from src.utils.superinvestor_roster import (
    first_snapshot_date,
    roster_as_of,
    roster_cik_union,
    roster_map_as_of,
)


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
        availability_config = config.get("data_availability")
        self._availability = InstitutionalAvailability.from_config(availability_config) if availability_config is not None else None
        freshness_config = config.get("source_freshness") or {}
        self._insider_bulk_cutover = freshness_config.get("insider_bulk_authoritative_through")

    def run(self, full: bool = False) -> None:
        panel, window = self.build_panel(full=full)
        n = write_part(self._store, Tables.cube_part_institutionals, panel, window, drop_empty=True)
        if n == COLUMNS_CHANGED:
            return self.run(full=True)

    def build_panel(self, full: bool = False) -> tuple[pd.DataFrame, PartWindow]:
        """The merge chain, WITHOUT the write. Extracted so `validate institutionals` scores
        the same panel this step persists instead of carrying a second copy of the chain --
        the "two declarations of one fact" failure the registry's `read_columns` exists to end.

        `frames` and `shares` are locals and die with the frame on return, which is what the
        `del` before `write_part` used to buy."""
        window = plan_window(
            self._store, Tables.cube_part_institutionals, full=full, warmup=self._warmup(), trading_index=load_trading_calendar(self._store)
        )

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

        merger = PanelMerger(self._log, anchor=price_frames.skeleton())
        merger.add(self._institutional_panel(price_frames, shares, splits), "institutional (13F)", "No institutional (13F) features built.")
        merger.add(
            self._superinvestor_panel(price_frames, shares, splits, sink), "superinvestor (elite 13F)", "No superinvestor (elite 13F) features built."
        )
        merger.add(self._insider_panel(price_frames, shares, sink), "insider-trading", "No insider-trading features built.")
        merger.add(self._short_flow_panel(price_frames, shares, splits, sink), "short-flow", "No short-flow features built.")
        merger.add(self._ownership_panel(price_frames, sink), "beneficial-ownership", "No beneficial-ownership features built.")
        merger.add(self._conditioning_panel(price_frames, splits, sink), "price-conditioning", "No price-conditioning features built.")
        merger.add(self._cross_source_panel(price_frames, sink), "cross-source", "No cross-source features built.")

        return merger.to_long(), window

    def _warmup(self) -> int:
        override = self._cfg.get("incremental", {}).get("warmup_trading_days")
        return int(override) if override is not None else self._part.warmup_trading_days

    # ---- inputs ---- #
    def _load_frames(self) -> PriceFrames:
        """⚠ NO `since` PARAMETER, unlike every sibling step. This part's grid is the FULL
        trading calendar on both paths -- see the block in `build_panel`. Taking the argument
        and always passing `None` would leave the next reader thinking it was a choice."""
        return institutional_inputs.load_full_price_frames(
            self._store,
            self._context,
            self._FIELDS,
        )

    def _load_source(self, table: Table, universe: Sequence[str] | None = None) -> pd.DataFrame | None:
        return institutional_inputs.load_source(self._store, self._log, table, universe)

    def _load_shares_out(self) -> pd.DataFrame | None:
        return institutional_inputs.load_shares_out(self._store, self._log)

    # ---- panels ---- #
    def _institutional_panel(self, price_frames: PriceFrames, shares: pd.DataFrame | None, splits: pd.DataFrame | None) -> pd.DataFrame | None:
        """The registry's eleven `ic_inst_*`: breadth SHARE (D28), split-restated share
        accumulation, new-buyer / exit ratios, cluster buying, Herfindahl concentration, net
        put/call sentiment, ownership %, value/market-cap weight and net $ flow -- stamped
        point-in-time on each period's AVAILABILITY DATE (the 45-day deadline snapped onto the
        trading calendar plus a settle buffer) over only the filings public by then, and
        re-emitted on each later date a material filing for that period arrives."""

        holdings = self._load_source(Tables.sec13f_hr, price_frames.universe)
        if holdings is None:
            return None

        cfg = self._institutionals_cfg()
        return build_institutional_feature_panel(
            price_frames,
            holdings,
            shares_out_history=shares,
            splits=splits,
            break_pct=float(cfg.get("coverage_break_pct", COVERAGE_BREAK_DEFAULT)),
            settle_trading_days=int(cfg.get("f13_settle_trading_days", F13_SETTLE_TRADING_DAYS)),
            revision_min_move=float(cfg.get("f13_revision_min_move", F13_REVISION_MIN_MOVE)),
        )

    def _superinvestor_panel(
        self, frames: PriceFrames, shares: pd.DataFrame | None, splits: pd.DataFrame | None, sink: ConditioningSink
    ) -> pd.DataFrame | None:
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

        roster = roster_cik_union(self._context)  # 106 managers as of 2026-09-08
        if not roster:
            self._log.warning("`superinvestor_roster` has no snapshot -> elite 13F features skipped (run `data_extract superinvestors --seed`).")
            return None

        holdings = load_superinvestor_holdings(self._context, roster)
        if holdings is None or holdings.empty:
            self._log.warning("No elite-manager 13F holdings -> superinvestor features skipped.")
            return None
        self._log.info(
            "Elite 13F books: %s rows across %s ever-listed managers (%s on today's roster)",
            len(holdings),
            len(roster),
            len(roster_map_as_of(self._context)),
        )

        return build_superinvestor_feature_panel(
            frames,
            holdings,
            roster,
            shares_out_history=shares,
            cusip_map=self._load_source(Tables.cusip_ticker_map),
            splits=splits,
            selection=self._superinvestor_selector(),
            decay_halflife=float(self._decay_halflife("super")),
            stale_quarters=int(self._superinvestor_cfg().get("stale_quarters", 4)),
            availability=self._availability,
            sink=sink,
        )

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
            ok = eligibility(state, roster_at, min_quarters=min_quarters, min_positions=min_positions)
            scored = manager_concentration_score(state, eligible=ok)
            sel = elite_weight(scored, mode=mode, k=k)
            # `avail` is attached by the panel builder before it calls this, so the
            # diagnostic reads the same availability grid the selection was ranked on.
            diag = selection_diagnostics(sel, state)
            if not diag.empty:
                settled = diag[diag["n_public"] >= k]
                self._log.info(
                    "elite selection (%s, k=%s): %s of %s manager-quarters eligible, live set %s-%s managers, median churn %s per filing date",
                    mode,
                    k,
                    int(ok.sum()),
                    len(ok),
                    int(settled["n_selected"].min()) if len(settled) else 0,
                    int(settled["n_selected"].max()) if len(settled) else 0,
                    int(settled["churn"].median()) if len(settled) else 0,
                )
            return sel

        return selector

    def _decay_halflife(self, family: str) -> float:
        """Trading-day half-life for `family`'s event decay, from
        `build_cube.institutionals.decay_halflife`. A tunable number, so it lives in the
        config and not in `constants/`."""
        cfg = self._cfg.get("institutionals", {}).get("decay_halflife", {})
        return float(cfg.get(family, 63))

    def _insider_panel(self, frames: PriceFrames, shares: pd.DataFrame | None, sink: ConditioningSink) -> pd.DataFrame | None:
        """Fourteen Form 3/4/5 features: size-scaled open-market buying, cluster breadth,
        CEO/CFO/director legs, the buyer's own-history surprise, and the 10b5-1 split of
        selling. Point-in-time on the filing date (a Form 4 is due within ~2 business days).

        ⚠ THE DOLLAR FIGURES ARE SCOPED AND REPAIRED BEFORE THEY ARE SUMMED. Read raw,
        `value_usd` averages **$447,771,735,138** per Form 4 line -- 49.4% of which is ONE
        convertible-note row -- and the sells total $182,982,720tn against a real ~$1.45tn.
        After the scope cut and the price repair the mean is **$2,339,662** and the median
        $114,116. See `insider_quality` for which population each figure belongs to."""
        bulk = self._load_source(Tables.insider_transactions, frames.universe)
        live = self._load_source(Tables.insider_transactions_live, frames.universe)
        insider = self._overlay_insider_sources(
            bulk,
            live,
            bulk_authoritative_through=self._insider_bulk_cutover,
        )
        if insider is None or insider.empty:
            return None
        _, latest_quarter = self._store.bounds(Tables.insider_transactions, "quarter")
        bulk_complete_through = self._insider_bulk_complete_through(
            latest_quarter,
            bulk,
            live,
            bulk_authoritative_through=self._insider_bulk_cutover,
        )
        live_complete_through = institutional_frontiers.insider_live_complete_through(
            self._store,
            self._log,
            frames.universe,
        )
        complete_through = self._insider_complete_through(
            bulk_complete_through,
            insider,
            live_complete_through,
        )
        return build_insider_feature_panel(
            frames,
            insider,
            shares_out_history=shares,
            decay_halflife=float(self._decay_halflife("insider")),
            availability=self._availability,
            complete_through=complete_through,
            sink=sink,
        )

    @staticmethod
    def _overlay_insider_sources(
        bulk: pd.DataFrame | None,
        live: pd.DataFrame | None,
        *,
        bulk_authoritative_through: object = None,
    ) -> pd.DataFrame | None:
        return overlay_insider_sources(
            bulk,
            live,
            bulk_authoritative_through=bulk_authoritative_through,
        )

    @staticmethod
    def _as_quarter(value: object) -> pd.Period | None:
        return as_quarter(value)

    @staticmethod
    def _insider_bulk_complete_through(
        latest_quarter: object,
        bulk: pd.DataFrame | None,
        live: pd.DataFrame | None,
        *,
        bulk_authoritative_through: object = None,
    ) -> pd.Timestamp | None:
        return bulk_complete_through(
            latest_quarter,
            bulk,
            live,
            bulk_authoritative_through=bulk_authoritative_through,
        )

    @staticmethod
    def _insider_complete_through(
        bulk_complete_through: object,
        insider: pd.DataFrame,
        live_complete_through: pd.Timestamp | None = None,
    ) -> pd.Timestamp | None:
        return institutional_frontiers.insider_complete_through(
            bulk_complete_through,
            insider,
            live_complete_through,
        )

    def _short_flow_panel(
        self, frames: PriceFrames, shares: pd.DataFrame | None, splits: pd.DataFrame | None, sink: ConditioningSink
    ) -> pd.DataFrame | None:
        """Volume-weighted RegSHO short-VOLUME ratios (5/20/60d), their self-history z, the
        two price-conditional interactions and short turnover, plus SEC fails-to-deliver
        (settlement stress) as a share of shares outstanding and of ADV20. RegSHO is lagged
        one trading day; FTD by ~2 months (its publication delay)."""
        short = self._load_source(Tables.short_interest, frames.universe)
        fails = self._load_source(Tables.sec_fails_to_deliver, frames.universe)
        universe = sorted(set(map(str, frames.universe)))
        symbol_tenure = self._store.load(
            Tables.symbol_tenure,
            columns=("symbol", "issuer_cik", "valid_from", "valid_to"),
            where={"symbol": universe},
            optional=True,
        )
        ticker_ciks = self._store.load(
            Tables.sp500_tickers,
            columns=("ticker", "cik"),
            where={"ticker": universe},
            optional=True,
        )
        return build_short_flow_feature_panel(
            frames,
            short,
            fails_history=fails,
            shares_out_history=shares,
            splits=splits,
            symbol_tenure=symbol_tenure,
            ticker_ciks=ticker_ciks,
            availability=self._availability,
            sink=sink,
        )

    def _ownership_panel(self, frames: PriceFrames, sink: ConditioningSink) -> pd.DataFrame | None:
        """Schedule 13D activist (`ic_act_*`) and 13G passive-ownership (`ic_bo_*`) events.
        `sec_13d_transactions` is deliberately not read -- see `ownership_features`'s module
        docstring."""
        sec_13d = self._load_source(Tables.sec_13d, frames.universe)
        sec_13g = self._load_source(Tables.sec_13g, frames.universe)
        expected_tickers = sorted(set(map(str, frames.universe)))
        complete_13d = self._schedule_complete_through(
            Tables.sec_13d,
            expected_tickers=expected_tickers,
        )
        complete_13g = self._schedule_complete_through(
            Tables.sec_13g,
            expected_tickers=expected_tickers,
        )
        return build_ownership_feature_panel(
            frames,
            sec_13d,
            sec_13g,
            decay_halflife_act=float(self._decay_halflife("act")),
            decay_halflife_bo=float(self._decay_halflife("bo")),
            availability=self._availability,
            complete_through_13d=complete_13d,
            complete_through_13g=complete_13g,
            sink=sink,
        )

    def _schedule_complete_through(
        self,
        table: Table,
        *,
        expected_tickers: Sequence[str],
    ) -> pd.Timestamp | None:
        return institutional_frontiers.schedule_complete_through(
            self._context,
            self._log,
            table,
            expected_tickers=expected_tickers,
        )

    def _conditioning_panel(self, frames: PriceFrames, splits: pd.DataFrame | None, sink: ConditioningSink) -> pd.DataFrame | None:
        """The `ic_sig_*` layer: days since each family's last disclosure and the price path
        since, sector-residualized and vol-scaled. THE PANEL'S ONLY DAILY-MOVING FAMILY, and
        the direct evidence for the two-layer architecture (report acceptance test #13)."""
        return build_signal_conditioning_panel(
            frames,
            sink.events,
            splits=splits,
            excursion_lookback=int(self._institutionals_cfg().get("excursion_lookback", EXCURSION_LOOKBACK)),
            frontiers=sink.frontiers,
        )

    def _cross_source_panel(self, frames: PriceFrames, sink: ConditioningSink) -> pd.DataFrame | None:
        """The `ic_xs_*` layer: how many independent families -- and how many distinct
        ACTORS -- are flagging this name at once, plus the both-sides conflict flag."""
        return build_cross_source_panel(frames, sink)
