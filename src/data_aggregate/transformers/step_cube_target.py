"""
step_cube_target.py  (src/data_aggregate/transformers/step_cube_target.py)
----------------------------------------------------------------------
Factor panel -> rolling multi-factor betas -> multi-horizon factor-neutral labels, persisted
as the two parts the assemble step joins: `cube_part_targets` (WIDE -- one row per
(date, ticker), one `target_<label>_h<horizon>` column per pair) and `cube_part_betas`.

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

from src.constants.constants_price import DAILY_MACRO_LEVELS, MACRO_CUBE_FACTORS, MACRO_MARKET_SERIES
from src.context import Context
from src.data_aggregate.utils.assemble.cube import labels_to_wide
from src.data_aggregate.utils.common.incremental import (
    plan_window,
    write_part,
)
from src.data_aggregate.utils.common.inputs import input_windows, load_source
from src.data_aggregate.utils.common.peers_io import load_peers_or_raise
from src.data_aggregate.utils.price.checks import measure_seams
from src.data_aggregate.utils.price.frames import (
    PriceFrames,
    load_price_frames,
    load_trading_calendar,
)
from src.data_aggregate.utils.target.betas import _betas_to_long, estimate_all_betas
from src.data_aggregate.utils.target.factors import (
    assemble_factor_panel,
    build_characteristics,
    characteristic_to_factor_return,
    gics_sector_excess_returns,
    macro_change_factors,
)
from src.data_aggregate.utils.target.targets import build_targets_multi, fitted_beta_columns
from src.data_store.schema import Table, Tables
from src.utils.price.adjustments import load_bugfix
from src.utils.publication import market_days, point_in_time
from src.utils.step import Step
from src.utils.window import Window, WindowPlan


class StepCubeTarget(Step):
    """THE one place the cube touches macro data: a single `prices_macro` read, pivoted wide
    once, feeding every macro-derived column of the factor panel -- the market return, the
    commodity/FX factor returns, and the FRED change factors. Those used to be three reads
    across two tables (`cube_part_market` twice, `macro` once)."""

    # The price fields this step reads back from `cube_part_prices`: `ret` as well as
    # `close`, because the factor panel is built from returns. Declared rather than inlined
    # so the projection stays introspectable (see test_part_registry.py).
    # close_split anchors observed keys; valuation reads saved issuer cap only.
    _FIELDS = ("close_split", "close_total", "ret", "mcap_pit")

    def __init__(self, context: Context, config: DictConfig):
        super().__init__(context=context, config=config)
        self._cfg = config.data_aggregate
        self._store = context.store
        # ⚠ the `null_ret` register reaches this step too, and it must. This step RECOMPUTES
        # `momentum_characteristic` on its own `close_total` -- it does not read
        # `cube_part_momentum.mom_12_1` -- so masking the momentum feature leaves the LABEL
        # neutralized against a fabricated exposure unless the same register lands here.
        self._bugfix = load_bugfix(context.config_dir)

    def run(self) -> dict[Table, WindowPlan]:
        # global variable definition
        horizons = self._cfg.targets.horizons
        calendar = load_trading_calendar(self._context)
        window = plan_window(self._context, Tables.cube_part_targets, calendar)
        beta_window = plan_window(self._context, Tables.cube_part_betas, calendar)
        windows = {table: plan for table, plan in ((Tables.cube_part_targets, window), (Tables.cube_part_betas, beta_window)) if plan is not None}
        if not windows:
            return {}

        # Both outputs receive sufficient context even when only one is initially absent.
        declared = [input_windows(self._context, table, plan.write) for table, plan in windows.items()]
        reads = {
            table: None
            if any(mapping[table] is None for mapping in declared)
            else Window(min(mapping[table].start for mapping in declared), self._context.run.as_of)
            for table in declared[0]
        }
        price_frames = load_price_frames(
            self._context,
            peers=load_peers_or_raise(self._context, self._config),
            calendar=calendar,
            fields=self._FIELDS,
            window=reads[Tables.cube_part_prices],
        )
        fundamentals = self._load_fundamentals(calendar, reads[Tables.sharadar_fundamentals_history])
        macro = self._load_macro(calendar, reads[Tables.prices_macro])

        # ONE measurement, shared by the momentum STYLE FACTOR and the label's own momentum
        # exposure, so the two cannot end up masking different windows.
        seams = self._measure_seams(price_frames)

        # aggregate and compute needed netral variables
        panel, macro_cols = self._factor_panel(price_frames, fundamentals, macro, seams)
        sector_groups = {"sector": self._universe.sector_map, "industry_group": self._universe.industry_group_map}
        # avg sector without market -- fed the panel's OWN market column, not a second read
        sector_excess = self._sector_factor(price_frames, sector_groups, panel["market"])

        # fit betas and build target neutrals to betas
        betas = self._estimate_betas(price_frames, panel, sector_excess)
        targets = self._build_targets(price_frames, (betas, panel, macro_cols), horizons, (sector_groups, sector_excess), seams)
        self._persist(targets, betas, windows, price_frames.skeleton())
        return windows

    # ---- inputs ---- #
    def _measure_seams(self, frames: PriceFrames) -> dict[str, list[pd.Timestamp]]:
        """The registered `null_ret` basis changes, re-measured on THIS step's `close_total`."""
        frames.require("close_total")
        assert frames.close_total is not None
        seams = measure_seams(frames.close_total, self._bugfix)
        listed = sum(len(v) for v in (self._bugfix.get("null_ret") or {}).values())
        self._log.info(
            "Seam masks: %s of %s registered null_ret entries measured (%s ticker(s)) "
            "-> masked in the momentum style factor and the label's momentum exposure",
            sum(len(v) for v in seams.values()),
            listed,
            len(seams),
        )
        return seams

    def _load_fundamentals(self, calendar: pd.DatetimeIndex, window: Window | None) -> pd.DataFrame | None:
        columns = ("ticker", "as_of", "netIncome", "freeCashflow", "stockholdersEquity")
        df_rows = load_source(self._context, Tables.sharadar_fundamentals_history, columns=columns, window=window)
        return None if df_rows is None else point_in_time(df_rows, self._config, Tables.sharadar_fundamentals_history, calendar)

    def _load_macro(self, calendar: pd.DatetimeIndex, window: Window | None) -> pd.DataFrame:
        """Delay shared beta/target inputs by trading sessions; SPY/gold/oil stay current."""
        series = sorted({MACRO_MARKET_SERIES, *MACRO_CUBE_FACTORS.values(), *DAILY_MACRO_LEVELS})
        df_rows = load_source(self._context, Tables.prices_macro, series, columns=["date", "ticker", "close"], window=window)
        if df_rows is None:
            raise RuntimeError(f"{Tables.prices_macro.name}: configured macro history is unavailable")
        no_lag = self._config.copy()
        no_lag.data.tables[Tables.prices_macro.name].pit.lag_days = 0
        df_rows = point_in_time(df_rows, no_lag, Tables.prices_macro, calendar)
        immediate = df_rows["ticker"].isin((MACRO_MARKET_SERIES, "gold", "oil"))
        lag = int(self._config.data.tables[Tables.prices_macro.name].pit.lag_days)
        positions = calendar.get_indexer(df_rows["as_of"]) + np.where(immediate, 0, lag)
        eligible = positions < len(calendar)
        df_rows = df_rows.loc[eligible].copy()
        df_rows["as_of"] = calendar.take(positions[eligible])
        df_rows = df_rows.sort_values("date", key=market_days, kind="stable")
        df_rows["as_of"] = df_rows["as_of"].dt.as_unit(calendar.unit)
        return df_rows.pivot_table(index="as_of", columns="ticker", values="close", aggfunc="last").sort_index().rename_axis("date")

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
            raise RuntimeError(f"'{MACRO_MARKET_SERIES}' missing from {Tables.prices_macro} -> no market factor, so betas and epsilon are undefined.")
        s = macro[MACRO_MARKET_SERIES].astype(float)
        s = s.reindex(s.index.union(calendar)).ffill()
        return s.pct_change(fill_method=None).reindex(calendar)

    def _asset_factors(self, macro: pd.DataFrame, calendar: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Commodity + FX factor returns from `prices_macro`. The panel column names are the
        `MACRO_CUBE_FACTORS` keys, unchanged from when these came out of `prices` via
        `cube_part_market`, so no beta or feature name downstream moves."""
        cols: dict[str, pd.Series] = {}
        for panel_col, series in MACRO_CUBE_FACTORS.items():
            if series not in macro.columns:
                self._log.warning("factor '%s' skipped: series '%s' absent from %s", panel_col, series, Tables.prices_macro)
                continue
            s = macro[series].astype(float)
            s = s.reindex(s.index.union(calendar)).ffill()
            cols[panel_col] = s.pct_change(fill_method=None).reindex(calendar)

        frame = pd.DataFrame(cols, index=calendar)
        # split back into the two families assemble_factor_panel expects; both are RETURNS,
        # so the split is presentational (it keeps the log line and the signature readable).
        fx = [c for c in frame.columns if MACRO_CUBE_FACTORS[c].startswith("fx_")]
        return frame.drop(columns=fx), frame[fx]

    def _factor_panel(
        self, frames: PriceFrames, fundamentals: pd.DataFrame | None, macro: pd.DataFrame, seams: dict[str, list[pd.Timestamp]] | None = None
    ) -> tuple[pd.DataFrame, list[str]]:
        """Flat by design: style, macro, commodity/currency and market are each ONE call
        away, so every factor family that goes into the panel is visible here instead of
        behind another wrapper."""

        frames.require("close_total", "ret")
        assert frames.close_total is not None
        assert frames.ret is not None
        chars = build_characteristics(
            stock_close_total=frames.close_total,
            stock_ret=frames.ret,
            fundamentals=fundamentals,
            resvol_window=63,
            mcap_pit=frames.mcap_pit,
            seams=seams,
        )
        macro_chg = self._macro_changes(macro, frames.trading_index)
        commodity, currency = self._asset_factors(macro, frames.trading_index)
        mkt_ret = self._market_return(macro, frames.trading_index)

        style_cols = {}
        for name, char in chars.items():
            char.name = name
            style_cols[name] = characteristic_to_factor_return(char, frames.ret)
        style = pd.DataFrame(style_cols)

        panel, macro_cols = assemble_factor_panel(mkt_ret, style, commodity, currency, macro_chg)

        self._log.info("Factor panel: %s factors (%s style/market, %s macro)", panel.shape[1], panel.shape[1] - len(macro_cols), len(macro_cols))

        return panel, macro_cols

    # ---- GICS sector factor ---- #
    def _sector_factor(self, frames: PriceFrames, sector_groups: dict[str, dict[str, str]], market_ret: pd.Series) -> pd.DataFrame | None:
        cfg = self._cfg.betas
        sector = sector_groups["sector"]
        frames.require("ret")
        assert frames.ret is not None

        # `market_ret` is the panel's own `market` column, passed in rather than re-derived:
        # this de-markets the sector REGRESSOR before any beta is fit, so it must be the exact
        # same series the betas are later fit against.
        df_sector_neutral = gics_sector_excess_returns(
            stock_ret=frames.ret, sector_map=sector, market_ret=market_ret, window=cfg.window, min_obs=cfg.min_obs
        )

        self._log.info("GICS sector factor: %s sectors over %s tickers", len(set(sector.values())), df_sector_neutral.notna().any().sum())

        return df_sector_neutral

    # ---- betas ---- #
    def _estimate_betas(self, frames: PriceFrames, panel: pd.DataFrame, sector_excess: pd.DataFrame | None) -> dict:
        cfg = self._cfg.betas
        frames.require("ret")
        assert frames.ret is not None
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
            ffill_limit=cfg.get("ffill_limit", 21),
        )

        # `_assemble_output` already OMITS a ticker with no estimable window, and
        # `compute_epsilon` skips a ticker absent from this dict -- so an empty dict is the
        # only unrecoverable case and there is nothing to filter per-ticker.
        if not betas:
            raise RuntimeError("no ticker produced betas -> factor panel or returns are empty")

        bm = np.nanmean([b["beta_market"].mean() for b in betas.values()])
        self._log.info("Estimated multi-factor betas for %s tickers (mean beta_market=%.2f)", len(betas), bm)
        return betas

    # ---- targets ---- #
    def _build_targets(
        self,
        frames: PriceFrames,
        factors: tuple[dict, pd.DataFrame, list[str]],
        horizons: list[int],
        sectors: tuple[dict[str, dict[str, str]], pd.DataFrame | None],
        seams: dict[str, list[pd.Timestamp]] | None = None,
    ) -> dict:
        """Build labels from the fitted factor bundle and matched sector exposures."""
        betas, panel, macro_cols = factors
        sector_groups, sector_excess = sectors
        cfg = self._cfg.targets
        frames.require("close_total", "ret")
        assert frames.close_total is not None

        # store EVERY configured target version (e.g. rank AND zscore) so the modelling step
        # can pick one via model.target_type without a cube rebuild
        label_types = list(cfg.get("labels", ["zscore", "rank", "epsilon"]))
        market_cap = frames.mcap_pit if cfg.get("neutralize_log_mcap", False) else None
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
            market_cap=market_cap,
            seams=seams,
        )

        non_null = sum(int(df.notna().sum().sum()) for per in labels.values() for df in per.values())
        # the log_mcap name count is the ONLY observable for the one silent failure mode: an
        # EMPTY market-cap frame becomes an all-zero design column `lstsq` absorbs exactly, so
        # the flag would no-op with the size tilt intact and no test or gate would fire.
        self._log.info(
            "Built factor-neutral targets %s for horizons %s (projected orthogonal "
            "to %s loadings + momentum + GICS industry + log_mcap on %s names, "
            "non-null=%s)",
            label_types,
            horizons,
            len(fitted_beta_columns(betas)),
            0 if market_cap is None else int(market_cap.notna().any().sum()),
            non_null,
        )
        return labels

    # ---- persist ---- #
    def _persist(self, labels: dict, betas: dict, windows: dict[Table, WindowPlan], valid_keys: pd.DataFrame) -> None:
        targets_wide = labels_to_wide(labels, valid_keys=valid_keys)
        betas_long = _betas_to_long(betas).merge(valid_keys[["date", "ticker"]], on=["date", "ticker"], how="inner", validate="one_to_one")
        outputs = {Tables.cube_part_targets: targets_wide, Tables.cube_part_betas: betas_long}
        with self._store.transaction() as store:
            for table, window in windows.items():
                write_part(store, table, outputs[table], window)
