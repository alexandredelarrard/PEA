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

from src.data_store.schema import Tables
from src.data_aggregate.utils.assemble.cube import _betas_to_long, _labels_to_long
from src.data_aggregate.utils.common.gics import load_gics_maps
from src.data_aggregate.utils.common.pit import daily_market_cap
from src.data_aggregate.utils.common.incremental import (
    COLUMNS_CHANGED, plan_window, window_start, write_part,
)
from src.data_aggregate.utils.common.parts import part_for
from src.data_aggregate.utils.common.peers_io import load_peers_or_raise
from src.data_aggregate.utils.common.price_frames import (
    PriceFrames, load_price_frames, load_trading_calendar,
)
from src.data_aggregate.utils.target.betas import estimate_all_betas
from src.data_aggregate.utils.target.factors import (
    assemble_factor_panel, build_characteristics,
    characteristic_to_factor_return,
    gics_sector_excess_returns,
    macro_change_factors,
)
from src.data_aggregate.utils.target.targets import build_targets_multi, fitted_beta_columns

from src.constants.constants_price import (DAILY_MACRO_LEVELS, MACRO_CUBE_FACTORS,
                                     MACRO_MARKET_SERIES)
from src.context import Context
from src.utils.macro import load_macro_wide
from src.utils.step import Step


class StepCubeTarget(Step):
    """THE one place the cube touches macro data: a single `prices_macro` read, pivoted wide
    once, feeding every macro-derived column of the factor panel -- the market return, the
    commodity/FX factor returns, and the FRED change factors. Those used to be three reads
    across two tables (`cube_part_market` twice, `macro` once)."""

    # The price fields this step reads back from `cube_part_prices`: `ret` as well as
    # `close`, because the factor panel is built from returns. Declared rather than inlined
    # so the projection stays introspectable (see test_part_registry.py).
    # `close_split` for market cap and the size characteristic (the basis on which the
    # split factor cancels against `sharesbas`); `close_total` for momentum; `ret` for every
    # LABEL, which is now a forward COMPOUNDED total return rather than a price ratio.
    _FIELDS = ("close_split", "close_total", "ret")

    def __init__(self, context: Context, config: DictConfig):

        super().__init__(context=context, config=config)
        self._cfg = config.build_cube
        self._part = part_for(Tables.cube_part_targets)
        self._store = context.store

    def run(self, full: bool = False) -> None:

        # global variable definition 
        horizons = self._cfg.targets.horizons
        max_h = max(horizons)
        calendar = load_trading_calendar(self._store)
        window = plan_window(self._store, Tables.cube_part_targets, full=full,
                             warmup= self._part.warmup_trading_days, 
                             trading_index=calendar,
                             extra_back=max_h)

        # load inputs 
        price_frames = self._load_frames(window.since)
        fundamentals = self._load_fundamentals()

        # aggregate and compute needed netral variables
        panel, macro_cols = self._factor_panel(price_frames, fundamentals)
        sector_groups = load_gics_maps(self._context)
        # avg sector without market -- fed the panel's OWN market column, not a second read
        sector_excess = self._sector_factor(price_frames, sector_groups, panel["market"])

        # fit betas and build target neutrals to betas
        betas = self._estimate_betas(price_frames, panel, sector_excess)
        targets = self._build_targets(price_frames, betas, panel, macro_cols, horizons,
                                      sector_groups, sector_excess, fundamentals)
        n = self._persist(targets, betas, window, calendar, max_h)
        
        if n == COLUMNS_CHANGED:
            return self.run(full=True)

    # ---- inputs ---- #
    def _load_frames(self, since: pd.Timestamp | None) -> PriceFrames:
        return load_price_frames(
            store=self._store,
            peers=load_peers_or_raise(self._context, self._config),
            fields=self._FIELDS,
            since=since)

    def _load_fundamentals(self) -> pd.DataFrame:
        columns = ("ticker", "as_of", "totalRevenue", "sharesOutstanding", "netIncome", "freeCashflow", "stockholdersEquity")
        return self._context.store.load(Tables.fundamentals_history, columns=columns)

    def _load_macro(self) -> pd.DataFrame:
        """`prices_macro`, pivoted wide -- date-indexed, one column per series.

        Deliberately UNWINDOWED: the change factors ffill and diff off each series' own prior
        observation, so a `since=` read would need a warmup and would silently null the first
        row of every `d_*` column. ~15 narrow series over 31y, so the full read is cheap."""
        wide = load_macro_wide(self._store)
        if wide is None:
            raise RuntimeError(f"'{Tables.prices_macro}' is missing or empty -> no market or "
                               "macro factors. Run `data_extract macro` first.")
        return wide.set_index("date").sort_index()

    # ---- factor panel ---- #
    @staticmethod
    def _macro_changes(macro: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
        return macro_change_factors(macro, calendar, level_to_change=DAILY_MACRO_LEVELS)

    @staticmethod
    def _market_return(macro: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.Series:
        """The market factor: daily return of the market series, on the equity calendar.

        ffill BEFORE pct_change and then reindex, so a macro-calendar hole becomes a zero
        return rather than shifting the next day's return onto a two-day move."""
        if MACRO_MARKET_SERIES not in macro.columns:
            raise RuntimeError(f"'{MACRO_MARKET_SERIES}' missing from {Tables.prices_macro} -> "
                               "no market factor, so betas and epsilon are undefined.")
        s = macro[MACRO_MARKET_SERIES].astype(float)
        s = s.reindex(s.index.union(calendar)).ffill()
        return s.pct_change(fill_method=None).reindex(calendar)

    def _asset_factors(self, macro: pd.DataFrame,
                       calendar: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Commodity + FX factor returns from `prices_macro`. The panel column names are the
        `MACRO_CUBE_FACTORS` keys, unchanged from when these came out of `prices` via
        `cube_part_market`, so no beta or feature name downstream moves."""
        cols: dict[str, pd.Series] = {}
        for panel_col, series in MACRO_CUBE_FACTORS.items():
            if series not in macro.columns:
                self._log.warning("factor '%s' skipped: series '%s' absent from %s",
                                  panel_col, series, Tables.prices_macro)
                continue
            s = macro[series].astype(float)
            s = s.reindex(s.index.union(calendar)).ffill()
            cols[panel_col] = s.pct_change(fill_method=None).reindex(calendar)

        frame = pd.DataFrame(cols, index=calendar)
        # split back into the two families assemble_factor_panel expects; both are RETURNS,
        # so the split is presentational (it keeps the log line and the signature readable).
        fx = [c for c in frame.columns if MACRO_CUBE_FACTORS[c].startswith("fx_")]
        return frame.drop(columns=fx), frame[fx]

    def _factor_panel(self, frames: PriceFrames,
                      fundamentals: pd.DataFrame | None) -> tuple[pd.DataFrame, list[str]]:
        """Flat by design: style, macro, commodity/currency and market are each ONE call
        away, so every factor family that goes into the panel is visible here instead of
        behind another wrapper."""

        frames.require("close_split", "close_total", "ret")
        chars = build_characteristics(stock_close_total=frames.close_total,
                                      stock_ret=frames.ret,
                                      fundamentals_history=fundamentals,
                                      resvol_window=63,
                                      stock_close_split=frames.close_split)
        macro = self._load_macro()                     # ONE read, ONE pivot, all macro below
        macro_chg = self._macro_changes(macro, frames.trading_index)
        commodity, currency = self._asset_factors(macro, frames.trading_index)
        mkt_ret = self._market_return(macro, frames.trading_index)

        style_cols = {}
        for name, char in chars.items():
            char.name = name
            style_cols[name] = characteristic_to_factor_return(char, frames.ret)
        style = pd.DataFrame(style_cols)

        panel, macro_cols = assemble_factor_panel(
            mkt_ret, style, commodity, currency, macro_chg)

        self._log.info("Factor panel: %s factors (%s style/market, %s macro)",
                       panel.shape[1], panel.shape[1] - len(macro_cols), len(macro_cols))

        return panel, macro_cols

    # ---- GICS sector factor ---- #
    def _sector_factor(self, frames: PriceFrames, sector_groups: dict[str, dict[str, str]],
                       market_ret: pd.Series) -> pd.DataFrame | None:

        cfg = self._cfg.betas
        sector = sector_groups['sector']
        frames.require("ret")

        # `market_ret` is the panel's own `market` column, passed in rather than re-derived:
        # this de-markets the sector REGRESSOR before any beta is fit, so it must be the exact
        # same series the betas are later fit against.
        df_sector_neutral = gics_sector_excess_returns(
            stock_ret=frames.ret,
            sector_map=sector,
            market_ret=market_ret,
            window=cfg.window,
            min_obs=cfg.min_obs)
        
        self._log.info("GICS sector factor: %s sectors over %s tickers",
                       len(set(sector.values())), df_sector_neutral.notna().any().sum())
        
        return df_sector_neutral

    # ---- betas ---- #
    def _estimate_betas(self, frames: PriceFrames, panel: pd.DataFrame,
                        sector_excess: pd.DataFrame | None) -> dict:
        cfg = self._cfg.betas
        frames.require("ret")
        betas = estimate_all_betas(
            stock_returns=frames.ret,
            global_factors=panel,
            stock_sector_factor=sector_excess,
            window=cfg.window,
            min_obs=cfg.min_obs,
            ridge_alpha=cfg.get("ridge_alpha", 1),
            ridge_alpha_market=cfg.get("ridge_alpha_market", 0.5),
            step=cfg.get("step", 1),
            market_prior=cfg.get("market_prior", 1.0),
            ffill_limit=cfg.get("ffill_limit", 21))

        # `_assemble_output` already OMITS a ticker with no estimable window, and
        # `compute_epsilon` skips a ticker absent from this dict -- so an empty dict is the
        # only unrecoverable case and there is nothing to filter per-ticker.
        if not betas:
            raise RuntimeError("no ticker produced betas -> factor panel or returns are empty")

        bm = np.nanmean([b["beta_market"].mean() for b in betas.values()])
        self._log.info("Estimated multi-factor betas for %s tickers (mean beta_market=%.2f)",
                       len(betas), bm)
        return betas

    # ---- targets ---- #
    def _build_targets(self, frames: PriceFrames, betas: dict, panel: pd.DataFrame,
                       macro_cols: list[str], horizons: list[int],
                       sector_groups: dict[str, dict[str, str]],
                       sector_excess: pd.DataFrame | None,
                       fundamentals: pd.DataFrame) -> dict:

        cfg = self._cfg.targets
        frames.require("ret")

        # store EVERY configured target version (e.g. rank AND zscore) so the modelling step
        # can pick one via model.target_type without a cube rebuild
        label_types = list(cfg.get("labels", ["zscore", "rank", "epsilon"]))
        # recomputed here rather than reused from `PitFrames.market_cap`: that cache belongs to
        # the fundamentals sub-step, which runs later and over a different warm-up window.
        market_cap = (daily_market_cap(fundamentals, frames.close_split)
                      if cfg.get("neutralize_log_mcap", False) else None)
        labels = build_targets_multi(
            close_total=frames.close_total,
            betas=betas,
            factor_panel=panel,
            macro_cols=macro_cols,
            horizons=tuple(horizons),
            labels=tuple(label_types),
            min_names=cfg.min_names,
            neutralize_momentum=cfg.get("neutralize_momentum", True),
            sector_groups=sector_groups,
            sector_excess=sector_excess,
            stock_ret=frames.ret,
            vol_standardize=cfg.get("vol_standardize", False),
            market_cap=market_cap)

        non_null = sum(int(df.notna().sum().sum())
                       for per in labels.values() for df in per.values())
        # the log_mcap name count is the ONLY observable for the one silent failure mode: an
        # EMPTY market-cap frame becomes an all-zero design column `lstsq` absorbs exactly, so
        # the flag would no-op with the size tilt intact and no test or gate would fire.
        self._log.info("Built factor-neutral targets %s for horizons %s (projected orthogonal "
                       "to %s loadings + momentum + GICS industry + log_mcap on %s names, "
                       "non-null=%s)", label_types, horizons, len(fitted_beta_columns(betas)),
                       0 if market_cap is None else int(market_cap.notna().any().sum()),
                       non_null)
        return labels

    # ---- persist ---- #
    def _persist(self, labels: dict, betas: dict, window,
                 calendar: pd.DatetimeIndex, max_h: int) -> int:
        targets_long, betas_long = _labels_to_long(labels), _betas_to_long(betas)

        # targets: overwrite the trailing max_horizon window so MATURED labels refresh
        refresh_from = (None if window.is_full
                        else window_start(calendar, window.last, max_h))
        n = write_part(self._store, Tables.cube_part_targets, targets_long, window,
                       refresh_from=refresh_from)
        if n == COLUMNS_CHANGED:
            return n
        
        # betas: backward-looking -> plain append after their OWN stored max
        beta_window = plan_window(self._store, Tables.cube_part_betas, full=window.is_full,
                                  warmup=0, trading_index=calendar)
        # `write_part` returns COLUMNS_CHANGED *instead of writing*, so discarding this return
        # would silently leave the betas part missing both the new column and the run's rows
        # (adding a macro factor adds a beta column, which is exactly when this fires).
        if write_part(self._store, Tables.cube_part_betas, betas_long,
                      beta_window) == COLUMNS_CHANGED:
            return COLUMNS_CHANGED
        return n
