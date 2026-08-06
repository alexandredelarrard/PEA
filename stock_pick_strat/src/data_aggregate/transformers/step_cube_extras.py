"""
step_cube_extras.py  (src/data_aggregate/transformers/step_cube_extras.py)
----------------------------------------------------------------------
The six OWNERSHIP / ATTENTION / GOVERNANCE panels -> `cube_part_extras`: DEF 14A
governance and executive pay, all-filer 13F institutional ownership, elite-manager
(superinvestor) 13F, insider trading, short interest + fails-to-deliver, and blended
retail attention.

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

import json

import pandas as pd
from omegaconf import DictConfig

from src.constants.constants import CUBE_PART_EXTRAS, SUPERINVESTORS_JSON
from src.context import Context
from src.data_aggregate.utils.common.incremental import COLUMNS_CHANGED, plan_window, write_part
from src.data_aggregate.utils.common.panel_merge import PanelMerger
from src.data_aggregate.utils.common.part_io import PartStore
from src.data_aggregate.utils.common.parts import PART_BY_NAME
from src.data_aggregate.utils.common.peers_io import load_peers_or_raise
from src.data_aggregate.utils.common.price_frames import (
    PriceFrames, load_price_frames, load_trading_calendar,
)
from src.data_aggregate.utils.common.sources import SOURCE_COLUMNS
from src.data_aggregate.utils.extras.attention_features import build_combined_attention_panel
from src.data_aggregate.utils.extras.def14a_impute import drop_implausible_def14a, impute_def14a
from src.data_aggregate.utils.extras.governance_features import build_governance_feature_panel
from src.data_aggregate.utils.extras.insider_features import build_insider_feature_panel
from src.data_aggregate.utils.extras.institutional_features import (
    build_institutional_feature_panel,
)
from src.data_aggregate.utils.extras.short_interest_features import (
    build_short_interest_feature_panel,
)
from src.data_aggregate.utils.extras.superinvestor_features import (
    build_superinvestor_feature_panel, load_superinvestor_holdings,
)
from src.utils.step import Step

_FUNDAMENTALS = "fundamentals_history"


class StepCubeExtras(Step):

    _FIELDS = ("close", "volume")

    def __init__(self, context: Context, config: DictConfig):
        super().__init__(context=context, config=config)
        self._cfg = config.build_cube
        self._part = PART_BY_NAME[CUBE_PART_EXTRAS]
        self._market_ticker = str(self._cfg.market_ticker)
        self._parts = PartStore(context.store, self._log)

    def run(self, full: bool = False) -> None:
        window = plan_window(self._parts, CUBE_PART_EXTRAS, full=full,
                             warmup=self._warmup(),
                             trading_index=load_trading_calendar(self._parts))
        frames = self._load_frames(window.since)
        # shares outstanding for the market-cap scaling shared by 13F / insider panels
        shares = self._load_shares_out()

        merger = PanelMerger(self._log)
        merger.add(frames.skeleton().assign(_grid=1.0), "universe-grid")
        merger.add(self._governance_panel(frames, shares), "governance/executive-pay",
                   "No governance/executive-pay features built (def14a_llm empty — "
                   "accrues as fetch_def14a_llm runs).")
        merger.add(self._institutional_panel(frames, shares), "institutional (13F)",
                   "No institutional (13F) features built.")
        merger.add(self._superinvestor_panel(frames, shares), "superinvestor (elite 13F)",
                   "No superinvestor (elite 13F) features built.")
        merger.add(self._insider_panel(frames, shares), "insider-trading",
                   "No insider-trading features built.")
        merger.add(self._short_interest_panel(frames), "short-interest",
                   "No short-interest features built.")
        merger.add(self._attention_panel(frames), "combined-attention",
                   "No attention data -> Wikipedia/Google-Trends features skipped.")

        panel = merger.to_long().drop(columns=["_grid"], errors="ignore")
        del frames, shares
        n = write_part(self._parts, CUBE_PART_EXTRAS, panel, window, self._log, drop_empty=True)
        if n == COLUMNS_CHANGED:
            return self.run(full=True)

    def _warmup(self) -> int:
        override = self._cfg.get("incremental", {}).get("warmup_trading_days")
        return int(override) if override is not None else self._part.warmup_trading_days

    # ---- inputs ---- #
    def _load_frames(self, since: pd.Timestamp | None) -> PriceFrames:
        return load_price_frames(
            self._parts, peers=load_peers_or_raise(self._context, self._config),
            market_ticker=self._market_ticker, fields=self._FIELDS, since=since)

    def _load_source(self, table: str) -> pd.DataFrame | None:
        """Load one source, PROJECTED to the columns its builder reads (see
        `utils/common/sources.py`); a table absent from that map loads in full."""
        df = self._context.store.load(table, columns=SOURCE_COLUMNS.get(table))
        if df is None or df.empty:
            return None
        self._log.info("Loaded %s: %s rows x %s cols", table, len(df), len(df.columns))
        return df

    def _load_shares_out(self) -> pd.DataFrame | None:
        df = self._context.store.load(_FUNDAMENTALS)
        if df.empty:
            self._log.warning("No fundamentals history -> the market-cap-scaled ownership "
                              "features are skipped.")
            return None
        return df

    # ---- panels ---- #
    def _governance_panel(self, frames: PriceFrames,
                          shares: pd.DataFrame | None) -> pd.DataFrame | None:
        """CEO pay growth, pay-vs-revenue-growth misalignment, pay ratio, board
        independence / diversity / tenure, say-on-pay. Point-in-time from each proxy's
        `as_of`.

        The raw extraction table is never mutated: the LLM-extracted proxy rows are
        cleaned ON READ (implausible values dropped, deducible cells imputed)."""
        df = self._load_source("def14a_llm")
        if df is None:
            return None
        df, drop_stats = drop_implausible_def14a(df)
        df, imp_stats = impute_def14a(df)
        stats = {**drop_stats, **imp_stats}
        if stats:
            self._log.info("DEF 14A clean-on-read: deduced %d missing cells across %d rules "
                           "(raw table untouched).", sum(stats.values()), len(stats))
        return build_governance_feature_panel(df, frames.peers, frames.trading_index,
                                              fundamentals_history=shares)

    def _institutional_panel(self, frames: PriceFrames,
                             shares: pd.DataFrame | None) -> pd.DataFrame | None:
        """13F breadth, share/value accumulation, new-buyer / exiter counts, cluster buying,
        Herfindahl concentration, net put/call sentiment, ownership %, value/market-cap
        weight and net $ flow -- stamped point-in-time with the 45-day filing lag."""
        holdings = self._load_source("sec13f_hr")
        if holdings is None:
            return None
        return build_institutional_feature_panel(
            holdings, frames.peers, frames.trading_index,
            shares_out_history=shares, stock_close=frames.close)

    def _superinvestor_panel(self, frames: PriceFrames,
                             shares: pd.DataFrame | None) -> pd.DataFrame | None:
        """Elite-manager 13F buy/sell evolution (Dataroma superinvestors), weighted by roster
        rank, layered ON TOP of the all-filer features. MEMORY-SAFE: reads ONLY the roster
        managers' rows (a handful of CIKs), never the whole ~21.7M-row table."""
        roster = self._load_superinvestor_roster()
        if not roster:
            self._log.warning("No superinvestors roster JSON -> elite 13F features skipped "
                              "(run fetch_superinvestors.build_superinvestors_json).")
            return None
        holdings = load_superinvestor_holdings(self._context, roster)
        if holdings is None or holdings.empty:
            self._log.warning("No elite-manager 13F holdings -> superinvestor features skipped.")
            return None
        n_mgrs = len(roster.get("cik_to_name") or roster.get("managers", []))
        self._log.info("Elite 13F subset: %s rows across %s managers", len(holdings), n_mgrs)
        return build_superinvestor_feature_panel(
            holdings, roster, frames.peers, frames.trading_index,
            shares_out_history=shares, stock_close=frames.close)

    def _load_superinvestor_roster(self) -> dict | None:
        path = self._context.paths["DATA_STORE"] / SUPERINVESTORS_JSON
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            self._log.warning("Superinvestors roster at %s is unreadable", path)
            return None

    def _insider_panel(self, frames: PriceFrames,
                       shares: pd.DataFrame | None) -> pd.DataFrame | None:
        """Trailing-window net open-market buying, buy/sell breadth, cluster-buy count and
        size-scaled net conviction from Forms 3/4/5. Point-in-time on the filing date (a
        Form 4 is due within ~2 business days)."""
        insider = self._load_source("insider_transactions")
        if insider is None:
            return None
        return build_insider_feature_panel(
            insider, frames.peers, frames.trading_index,
            shares_out_history=shares, stock_close=frames.close)

    def _short_interest_panel(self, frames: PriceFrames) -> pd.DataFrame | None:
        """RegSHO short-volume ratio + its change, plus SEC fails-to-deliver (settlement
        stress). RegSHO is lagged one trading day; FTD by ~2 months (its publication delay)."""
        short = self._load_source("short_interest")
        fails = self._load_source("fails_to_deliver")
        if short is None and fails is None:
            return None
        return build_short_interest_feature_panel(
            short, frames.peers, frames.trading_index,
            fails_history=fails, volume=frames.volume)

    def _attention_panel(self, frames: PriceFrames) -> pd.DataFrame | None:
        """Wikipedia pageviews and Google Trends are two noisy proxies of the same latent
        (public attention), so they are rank-BLENDED into one robust indicator rather than
        shipped as two correlated features. Empty only if BOTH sources are absent."""
        wiki = self._load_source("wiki_pageviews")
        trends = self._load_source("google_trends")
        if wiki is None and trends is None:
            return None
        return build_combined_attention_panel(wiki, trends, frames.peers,
                                              frames.trading_index)
