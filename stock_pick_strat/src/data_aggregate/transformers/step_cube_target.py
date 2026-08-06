"""
step_cube_target.py  (src/data_aggregate/transformers/step_cube_target.py)
----------------------------------------------------------------------
Factor panel -> rolling multi-factor betas -> multi-horizon factor-neutral labels, persisted
as the two long parts the assemble step joins (`cube_part_targets`, `cube_part_betas`).

INCREMENTAL, with a twist the feature parts do not have: betas are backward-looking so they
just append dates after the stored max, but TARGETS are FORWARD-looking -- a label at date d
needs prices through d+horizon, so labels that were NaN on the last run MATURE into values
between runs. The trailing `max_horizon` window is therefore recomputed and OVERWRITTEN, not
merely extended.

Memory: every frame is local to `run()`; nothing is stashed on `self`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from src.constants.constants import CUBE_PART_BETAS, CUBE_PART_TARGETS
from src.context import Context
from src.data_aggregate.utils.assemble.cube import _betas_to_long, _labels_to_long
from src.data_aggregate.utils.common.gics import load_gics_maps
from src.data_aggregate.utils.common.incremental import (
    COLUMNS_CHANGED, plan_window, window_start, write_part,
)
from src.data_aggregate.utils.common.part_io import PartStore
from src.data_aggregate.utils.common.parts import PART_BY_NAME
from src.data_aggregate.utils.common.peers_io import load_peers_or_raise
from src.data_aggregate.utils.common.price_frames import (
    PriceFrames, load_price_frames, load_trading_calendar,
)
from src.data_aggregate.utils.common.prices import price_column_returns
from src.data_aggregate.utils.target.betas import estimate_all_betas
from src.data_aggregate.utils.target.factors import (
    assemble_factor_panel, build_style_factor_returns, macro_change_factors,
)
from src.data_aggregate.utils.target.targets import build_targets_multi
from src.utils.step import Step

_FUNDAMENTALS = "fundamentals_history"
_MACRO = "macro"
_COMMODITY_TICKERS = {"oil": "CL=F", "gold": "GC=F"}
_CURRENCY_TICKERS = {"USD/EUR": "USDEUR=X"}


class StepCubeTarget(Step):

    # the style factors need close + returns; the beta regression needs the peer-basket return
    _FIELDS = ("close", "ret", "sector_ret")

    def __init__(self, context: Context, config: DictConfig):
        super().__init__(context=context, config=config)
        self._cfg = config.build_cube
        self._part = PART_BY_NAME[CUBE_PART_TARGETS]
        self._market_ticker = str(self._cfg.market_ticker)
        self._other_tickers = tuple(config.data_extract.get("other_tickers", ()) or ())
        self._parts = PartStore(context.store, self._log)

    def run(self, full: bool = False) -> None:

        horizons = self._cfg.targets.horizons
        max_h = max(horizons)
        calendar = load_trading_calendar(self._parts)
        window = plan_window(self._parts, CUBE_PART_TARGETS, full=full,
                             warmup= self._part.warmup_trading_days, 
                             trading_index=calendar,
                             extra_back=max_h)

        frames = self._load_frames(window.since)
        fundamentals = self._load_fundamentals()
        panel, macro_cols = self._factor_panel(frames, fundamentals)
        betas = self._estimate_betas(frames, panel)
        labels = self._build_targets(frames, betas, panel, macro_cols, horizons)
        n = self._persist(labels, betas, window, calendar, max_h)
        
        if n == COLUMNS_CHANGED:
            return self.run(full=True)

    # ---- inputs ---- #
    def _load_frames(self, since: pd.Timestamp | None) -> PriceFrames:
        return load_price_frames(
            self._parts, peers=load_peers_or_raise(self._context, self._config),
            market_ticker=self._market_ticker, fields=self._FIELDS,
            with_market=True, other_tickers=self._other_tickers, since=since)

    def _load_fundamentals(self) -> pd.DataFrame | None:
        """Optional: without it the value / quality style factors are simply skipped."""
        df = self._context.store.load(_FUNDAMENTALS)
        if df.empty:
            self._log.warning("No fundamentals history -> value/quality style factors skipped.")
            return None
        return df

    # ---- factor panel ---- #
    def _style_factors(self, frames: PriceFrames,
                       fundamentals: pd.DataFrame | None) -> pd.DataFrame:
        frames.require("close", "ret")
        return build_style_factor_returns(frames.close, frames.ret, fundamentals,
                                          resvol_window=63)

    def _macro_changes(self, frames: PriceFrames) -> pd.DataFrame:
        macro = self._context.store.load(_MACRO)
        if macro.empty:
            self._log.warning("No macro data -> macro betas will be skipped.")
            return pd.DataFrame(index=frames.trading_index)
        return macro_change_factors(macro, frames.trading_index)

    def _asset_factors(self, frames: PriceFrames) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Commodity + FX factor returns, from the market part's own close frame. The factor
        function indexes by the frame it is handed, so this is numerically identical to the
        old call against the full untrimmed price panel."""
        if frames.other_close is None or frames.other_close.empty:
            empty = pd.DataFrame(index=frames.trading_index)
            return empty, empty
        return (price_column_returns(frames.other_close, _COMMODITY_TICKERS),
                price_column_returns(frames.other_close, _CURRENCY_TICKERS))

    def _factor_panel(self, frames: PriceFrames,
                      fundamentals: pd.DataFrame | None) -> tuple[pd.DataFrame, list[str]]:
        style = self._style_factors(frames, fundamentals)
        macro_chg = self._macro_changes(frames)
        commodity, currency = self._asset_factors(frames)
        if frames.mkt_ret is None:
            raise RuntimeError("market returns missing from cube_part_market -> re-run "
                               "`build-prices`")
        panel, macro_cols = assemble_factor_panel(
            frames.mkt_ret, style, commodity, currency, macro_chg)
        self._log.info("Factor panel: %s factors (%s style/market, %s macro)",
                       panel.shape[1], panel.shape[1] - len(macro_cols), len(macro_cols))
        return panel, macro_cols

    # ---- betas ---- #
    def _estimate_betas(self, frames: PriceFrames, panel: pd.DataFrame) -> dict:
        cfg = self._cfg.betas
        frames.require("ret", "sector_ret")
        betas = estimate_all_betas(
            frames.ret, panel, frames.sector_ret,
            window=cfg.window, min_obs=cfg.min_obs,
            ridge=cfg.get("ridge", 5.0), step=cfg.get("step", 5))

        # a ticker without the univariate market beta is unusable downstream (the L/S
        # optimizer neutralizes on it), so drop it rather than ship a hole
        to_kick = [t for t, b in betas.items() if "beta_market_simple" not in b]
        for t in to_kick:
            self._log.warning("beta_market_simple missing for %s -> dropped", t)
        betas = {t: v for t, v in betas.items() if t not in to_kick}
        if not betas:
            raise RuntimeError("no ticker produced betas -> factor panel or returns are empty")
        bm = np.nanmean([betas[t]["beta_market_simple"].mean() for t in betas])
        self._log.info("Estimated multi-factor betas for %s tickers "
                       "(mean beta_market_simple=%.2f)", len(betas), bm)
        return betas

    # ---- targets ---- #
    def _gics_groups(self) -> dict[str, dict[str, str]] | None:
        """Neutralize to the ACTUAL GICS sector + industry (per-day within-group demeaning)
        INSTEAD of the return-correlation peer basket, so sector / industry membership
        cannot predict the target -- if it could, it would dominate the model."""
        if not self._cfg.targets.get("neutralize_sectors", True):
            return None
        return load_gics_maps(self._context)

    def _build_targets(self, frames: PriceFrames, betas: dict, panel: pd.DataFrame,
                       macro_cols: list[str], horizons: list[int]) -> dict:
        cfg = self._cfg.targets
        # store EVERY configured target version (e.g. rank AND zscore) so the modelling step
        # can pick one via model.target_type without a cube rebuild
        label_types = list(cfg.get("labels", [cfg.get("label", "rank")]))
        sector_groups = self._gics_groups()
        labels = build_targets_multi(
            close=frames.close, stock_returns=frames.ret, peer_dict=frames.peers,
            betas=betas, factor_panel=panel, macro_cols=macro_cols,
            horizons=tuple(horizons), labels=tuple(label_types),
            min_names=cfg.min_names,
            neutralize_momentum=cfg.get("neutralize_momentum", True),
            sector_groups=sector_groups)
        non_null = sum(int(df.notna().sum().sum())
                       for per in labels.values() for df in per.values())
        self._log.info("Built factor-neutral targets %s for horizons %s "
                       "(GICS sector+industry-neutral=%s, non-null=%s)",
                       label_types, horizons, sector_groups is not None, non_null)
        return labels

    # ---- persist ---- #
    def _persist(self, labels: dict, betas: dict, window,
                 calendar: pd.DatetimeIndex, max_h: int) -> int:
        targets_long, betas_long = _labels_to_long(labels), _betas_to_long(betas)

        # targets: overwrite the trailing max_horizon window so MATURED labels refresh
        refresh_from = (None if window.is_full
                        else window_start(calendar, window.last, max_h))
        n = write_part(self._parts, CUBE_PART_TARGETS, targets_long, window, self._log,
                       refresh_from=refresh_from)
        if n == COLUMNS_CHANGED:
            return n
        # betas: backward-looking -> plain append after their OWN stored max
        beta_window = plan_window(self._parts, CUBE_PART_BETAS, full=window.is_full,
                                  warmup=0, trading_index=calendar)
        write_part(self._parts, CUBE_PART_BETAS, betas_long, beta_window, self._log)
        return n
