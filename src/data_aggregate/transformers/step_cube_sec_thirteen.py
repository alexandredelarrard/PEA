"""Broad institutional and elite-manager 13F features from public accession books."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd
from omegaconf import DictConfig

from src.constants.constants import F13_SETTLE_TRADING_DAYS
from src.context import Context
from src.data_aggregate.sec_thirteen.utils import inputs as thirteen_inputs
from src.data_aggregate.sec_thirteen.utils.institutional_features import (
    COVERAGE_BREAK_DEFAULT,
    F13_REVISION_MIN_MOVE,
    build_institutional_feature_panel,
)
from src.data_aggregate.sec_thirteen.utils.manager_selection import eligibility, elite_weight, manager_concentration_score, selection_diagnostics
from src.data_aggregate.sec_thirteen.utils.superinvestor_features import attach_tickers, build_superinvestor_feature_panel
from src.data_aggregate.sec_thirteen.utils.superinvestor_roster import first_snapshot_date, roster_as_of
from src.data_aggregate.utils.common import inputs as capital_inputs
from src.data_aggregate.utils.common.availability import InstitutionalAvailability
from src.data_aggregate.utils.common.incremental import COLUMNS_CHANGED, PART_REFRESH_TRADING_DAYS, plan_window, write_part
from src.data_aggregate.utils.common.level_basis import load_bugfix
from src.data_aggregate.utils.common.panel_merge import PanelMerger
from src.data_aggregate.utils.common.parts import part_for
from src.data_aggregate.utils.common.price_frames import PriceFrames, load_trading_calendar
from src.data_store.schema import Tables
from src.utils.price_adjustments import ShareBasis
from src.utils.step import Step
from src.utils.superinvestor_roster import filer_ciks


class StepCubeSecThirteen(Step):
    _FIELDS = ("close_split", "level_factor")

    def __init__(self, context: Context, config: DictConfig) -> None:
        super().__init__(context=context, config=config)
        self._store = context.store
        self._cfg = config.build_cube
        self._part = part_for(Tables.cube_part_sec_thirteen)
        self._bugfix = load_bugfix(context.config_dir)
        cfg = config.get("data_availability")
        self._availability = InstitutionalAvailability.from_config(cfg) if cfg is not None else None

    def run(self, full: bool = False) -> None:
        window = plan_window(
            self._store,
            Tables.cube_part_sec_thirteen,
            full=full,
            warmup=self._warmup(),
            trading_index=load_trading_calendar(self._store),
            refresh=PART_REFRESH_TRADING_DAYS,
        )
        panel = self.build_panel(full=full)
        if write_part(self._store, Tables.cube_part_sec_thirteen, panel, window, drop_empty=True) == COLUMNS_CHANGED:
            self.run(full=True)

    def _warmup(self) -> int:
        override = self._cfg.get("incremental", {}).get("warmup_trading_days")
        return int(override) if override is not None else self._part.warmup_trading_days

    def build_panel(self, full: bool = False) -> pd.DataFrame:
        """Load shared inputs once and release each whole-book source after its panel."""
        frames = capital_inputs.load_full_price_frames(self._store, self._context, self._FIELDS)
        history = capital_inputs.load_source(self._store, Tables.superinvestor_roster)
        cusip_map = self._store.load(Tables.cusip_ticker_map, columns=["cusip", "ticker"], optional=True)
        shares, _, basis, valuation_close, _ = capital_inputs.load_share_inputs(self._context, frames, self._bugfix)
        if history is not None:
            history["snapshot_date"] = pd.to_datetime(history["snapshot_date"])
        roster = filer_ciks(history["cik"].dropna(), self._context.config_dir) if history is not None else set()
        splits = basis.actual.rename(columns={"value": "ratio"})
        merger = PanelMerger(anchor=frames.skeleton())
        merger.add(self._institutional_panel(frames, shares, splits, cusip_map, valuation_close, share_basis=basis), "institutional (13F)")
        merger.add(self._superinvestor_panel(frames, splits, cusip_map, roster, history, valuation_close), "superinvestor (elite 13F)")
        return merger.to_long()

    def _institutional_panel(
        self,
        price_frames: PriceFrames,
        shares: pd.DataFrame | None,
        splits: pd.DataFrame | None,
        cusip_map: pd.DataFrame | None,
        valuation_close: pd.DataFrame | None = None,
        *,
        share_basis: ShareBasis | None = None,
    ) -> pd.DataFrame | None:

        packet = thirteen_inputs.load_filing_vintages(self._store, price_frames.trading_index.max())
        if packet is None:
            return None
        filings, holdings = packet
        holdings = attach_tickers(holdings, cusip_map, price_frames.universe)

        cfg = self._cfg.get("institutionals", {})
        return build_institutional_feature_panel(
            price_frames,
            holdings,
            shares_out_history=shares,
            splits=splits,
            share_basis=share_basis,
            valuation_close=valuation_close,
            value_basis=cfg.get("value_basis"),
            filings=filings,
            break_pct=float(cfg.get("coverage_break_pct", COVERAGE_BREAK_DEFAULT)),
            settle_trading_days=int(cfg.get("f13_settle_trading_days", F13_SETTLE_TRADING_DAYS)),
            revision_min_move=float(cfg.get("f13_revision_min_move", F13_REVISION_MIN_MOVE)),
        )

    def _superinvestor_panel(
        self,
        frames: PriceFrames,
        splits: pd.DataFrame | None,
        cusip_map: pd.DataFrame | None,
        roster: set[str],
        history: pd.DataFrame | None,
        valuation_close: pd.DataFrame | None = None,
    ) -> pd.DataFrame | None:

        if not roster:
            self._log.warning("`superinvestor_roster` has no snapshot -> elite 13F features skipped (run `data_extract superinvestors -F`).")
            return None

        packet = thirteen_inputs.load_filing_vintages(self._store, frames.trading_index.max(), sorted(roster), table=Tables.sec13f_manager_holdings)
        if packet is None:
            self._log.warning("No elite-manager 13F holdings -> superinvestor features skipped.")
            return None
        filings, holdings = packet
        self._log.info(
            "Elite 13F books: %s rows across %s ever-listed managers (%s on today's roster)",
            len(holdings),
            len(roster),
            len(roster_as_of(history, config_dir=self._context.config_dir)),
        )

        return build_superinvestor_feature_panel(
            frames,
            holdings,
            roster,
            cusip_map=cusip_map,
            splits=splits,
            valuation_close=valuation_close,
            value_basis=self._cfg.get("institutionals", {}).get("value_basis"),
            filings=filings,
            selection=self._superinvestor_selector(history),
            decay_halflife=float(self._decay_halflife("super")),
            stale_quarters=int(self._cfg.get("institutionals", {}).get("superinvestor", {}).get("stale_quarters", 4)),
            availability=self._availability,
        )

    def _superinvestor_selector(self, history: pd.DataFrame | None) -> Callable[[pd.DataFrame], pd.Series] | None:
        cfg = self._cfg.get("institutionals", {}).get("superinvestor", {}).get("selection", {})
        raw = cfg.get("mode", "off")
        mode = "off" if raw in (False, None, "") else str(raw).lower()
        if mode in ("off", "none", "flat", "false"):
            return None
        first = first_snapshot_date(history)
        if first is None:
            self._log.warning("No roster snapshot -> selection falls back to flat 1.0.")
            return None
        k = int(cfg.get("k", 15))
        min_quarters = int(cfg.get("min_quarters", 4))
        min_positions = int(cfg.get("min_positions", 0))

        cache: dict[pd.Timestamp, set[str]] = {}

        def roster_at(q: object) -> set[str]:
            key = max(pd.Timestamp(q), first)
            if key not in cache:
                cache[key] = roster_as_of(history, key, config_dir=self._context.config_dir)
            return cache[key]

        def selector(state: pd.DataFrame) -> pd.Series:
            ok = eligibility(state, roster_at, min_quarters=min_quarters, min_positions=min_positions)
            scored = manager_concentration_score(state, eligible=ok)
            sel = elite_weight(scored, mode=mode, k=k)
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
        cfg = self._cfg.get("institutionals", {}).get("decay_halflife", {})
        return float(cfg.get(family, 63))
