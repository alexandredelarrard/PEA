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

Warm-up 1320: the binding look-backs are `_self_history_z`'s rolling(1260) and the 5-year
dividend payout growth. `sector` / `earnings` need ~none (they look back in filing space over
the full source table), so merging them into this part costs them a longer daily grid but no
correctness.
"""
from __future__ import annotations

import pandas as pd
from omegaconf import DictConfig
from omegaconf import OmegaConf

from src.constants.constants import CUBE_PART_FUNDAMENTALS
from src.context import Context
from src.data_aggregate.utils.common.incremental import COLUMNS_CHANGED, plan_window, write_part
from src.data_aggregate.utils.common.panel_merge import PanelMerger
from src.data_aggregate.utils.common.part_io import PartStore
from src.data_aggregate.utils.common.parts import PART_BY_NAME
from src.data_aggregate.utils.common.peers_io import load_peers_or_raise
from src.data_aggregate.utils.common.pit import PitFrames
from src.data_aggregate.utils.common.price_frames import (
    PriceFrames, load_price_frames, load_trading_calendar,
)
from src.data_aggregate.utils.fundamentals.dividend_features import build_dividend_feature_panel
from src.data_aggregate.utils.fundamentals.earnings_features import build_earnings_feature_panel
from src.data_aggregate.utils.fundamentals.employee_features import build_employee_feature_panel
from src.data_aggregate.utils.fundamentals.fundamental_features import (
    build_fundamental_feature_panel, load_notes_num_scoped, load_pension_facts_scoped,
)
from src.data_aggregate.utils.fundamentals.sector_features import build_sector_feature_panel
from src.utils.step import Step

_FUNDAMENTALS = "fundamentals_history"
_EARNINGS = "earnings_surprises"
_DIVIDENDS = "dividends"


class StepCubeFundamentals(Step):

    _FIELDS = ("close",)

    def __init__(self, context: Context, config: DictConfig):
        super().__init__(context=context, config=config)
        self._cfg = config.build_cube
        self._part = PART_BY_NAME[CUBE_PART_FUNDAMENTALS]
        self._market_ticker = str(self._cfg.market_ticker)
        self._parts = PartStore(context.store, self._log)

    def run(self, full: bool = False) -> None:
        window = plan_window(self._parts, CUBE_PART_FUNDAMENTALS, full=full,
                             warmup=self._warmup(),
                             trading_index=load_trading_calendar(self._parts))
        frames = self._load_frames(window.since)
        fundamentals = self._load_fundamentals()
        earnings = self._load_optional(_EARNINGS, "earnings-surprise history",
                                       "fetch_earnings_surprises")
        
        # ONE point-in-time cache for all five builders (see the module docstring)
        pit = PitFrames(fundamentals, frames.trading_index, frames.close)

        merger = PanelMerger(self._log)
        merger.add(frames.skeleton().assign(_grid=1.0), "universe-grid")
        merger.add(self._fundamental_panel(frames, fundamentals, earnings, pit),
                   "peer-relative fundamental",
                   "No fundamental features built (missing fundamentals).")
        merger.add(self._sector_kpi_panel(frames, fundamentals, pit), "sector-KPI",
                   "No sector KPI features built (missing fundamentals).")
        merger.add(self._earnings_panel(frames, earnings), "earnings-expectation",
                   "No earnings-expectation features built.")
        merger.add(self._employee_panel(frames, fundamentals, pit), "workforce",
                   "No workforce features built.")
        merger.add(self._dividend_panel(frames, fundamentals, pit), "dividend",
                   "No dividend features built (missing dividend history).")
        self._log.info("PitFrames shared across the fundamentals builders: %s", pit.stats())

        panel = merger.to_long().drop(columns=["_grid"], errors="ignore")
        del frames, fundamentals, earnings, pit
        n = write_part(self._parts, CUBE_PART_FUNDAMENTALS, panel, window, self._log, drop_empty=True)
        if n == COLUMNS_CHANGED:
            return self.run(full=True)

    def _warmup(self) -> int:
        override = self._cfg.get("incremental", {}).get("warmup_trading_days")
        return int(override) if override is not None else self._part.warmup_trading_days

    def _intrinsic_cfg(self) -> dict:
        cfg = self._cfg.get("intrinsic", {})
        return OmegaConf.to_container(cfg, resolve=True) if cfg else {}

    # ---- inputs ---- #
    def _load_frames(self, since: pd.Timestamp | None) -> PriceFrames:
        return load_price_frames(
            self._parts, peers=load_peers_or_raise(self._context, self._config),
            market_ticker=self._market_ticker, fields=self._FIELDS, since=since)

    def _load_fundamentals(self) -> pd.DataFrame | None:
        df = self._context.store.load(_FUNDAMENTALS)
        if df.empty:
            self._log.warning("No fundamentals history -> the fundamental, sector, workforce "
                              "and dividend-payout features will be skipped.")
            return None
        self._log.info("Loaded %s: %s rows, %s tickers (ONCE for five builders)",
                       _FUNDAMENTALS, len(df), df["ticker"].nunique())
        return df

    def _load_optional(self, table: str, what: str, fetcher: str) -> pd.DataFrame | None:
        df = self._context.store.load(table)
        if df.empty:
            self._log.warning("No %s -> related features skipped (run %s).", what, fetcher)
            return None
        return df

    # ---- panels ---- #
    def _fundamental_panel(self, frames: PriceFrames, fundamentals: pd.DataFrame | None,
                           earnings: pd.DataFrame | None, pit: PitFrames) -> pd.DataFrame | None:
        if fundamentals is None:
            return None
        hist = self._cfg.get("hist", {})
        # tag-scoped reads: only the two pension tags of each table, never the whole
        # multi-million-row facts tables
        return build_fundamental_feature_panel(
            fundamentals_history=fundamentals,
            peer_dict=frames.peers,
            trading_index=frames.trading_index,
            stock_close=frames.close,
            intrinsic_cfg=self._intrinsic_cfg(),
            hist_window=int(hist.get("window", 1260)),
            hist_min_periods=int(hist.get("min_periods", 252)),
            earnings_history=earnings,                       # PEGY projected-growth term
            pension_facts=load_pension_facts_scoped(self._context),
            notes_num=load_notes_num_scoped(self._context),
        )

    def _sector_kpi_panel(self, frames: PriceFrames, fundamentals: pd.DataFrame | None,
                          pit: PitFrames) -> pd.DataFrame | None:
        """Sector-specific KPIs (combined/loss ratio, NIM, efficiency ratio, FFO, inventory
        days, shareholder payout, net-debt/EBITDA, accruals), availability-gated per row so a
        KPI is null unless its sector reported the inputs."""
        if fundamentals is None:
            return None
        return build_sector_feature_panel(fundamentals, frames.peers, frames.trading_index)

    def _earnings_panel(self, frames: PriceFrames,
                        earnings: pd.DataFrame | None) -> pd.DataFrame | None:
        """Forward EPS yield, expected EPS growth and realized surprise. Genuinely historical
        and point-in-time: the forward estimate applies only within its own quarter, the
        actual only after the report."""
        if earnings is None:
            return None
        return build_earnings_feature_panel(earnings, frames.peers, frames.trading_index,
                                            stock_close=frames.close)

    def _employee_panel(self, frames: PriceFrames, fundamentals: pd.DataFrame | None,
                        pit: PitFrames) -> pd.DataFrame | None:
        """Revenue per employee and YoY headcount growth, from the `employees` column of
        `fundamentals_history` (10-K body-text headcount). Headcount and the revenue it is
        divided by come from the SAME frame and the same `as_of`, which is why one source is
        passed twice."""
        if fundamentals is None:
            return None
        return build_employee_feature_panel(fundamentals, frames.peers, frames.trading_index,
                                            fundamentals_history=fundamentals)

    def _dividend_panel(self, frames: PriceFrames, fundamentals: pd.DataFrame | None,
                        pit: PitFrames) -> pd.DataFrame | None:
        """TTM yield, 1y + 5y payout growth, payer flag, payout ratio, FCF coverage, dividend
        + buyback yield. RECONCILES the per-share ex-date history (`dividends`, primary) with
        the SEC cash-flow `dividendsPaid` total (gap-fill + payout/coverage). Non-payers get a
        real 0 yield so they rank correctly."""
        dividends = self._load_optional(_DIVIDENDS, "dividend history",
                                        "fetch_price_history -> StepExtractPrices")
        if dividends is None:
            return None
        return build_dividend_feature_panel(dividends, frames.peers, frames.trading_index,
                                            stock_close=frames.close,
                                            fundamentals_history=fundamentals)
