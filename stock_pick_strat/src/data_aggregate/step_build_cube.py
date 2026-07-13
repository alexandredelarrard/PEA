import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from src.utils.step import Step
from src.context import Context
from src.data_aggregate.utils import data_utils as du
from src.data_aggregate.utils.betas import estimate_all_betas
from src.data_aggregate.utils.targets import build_targets
from src.data_aggregate.utils.features import build_feature_panel
from src.data_aggregate.utils.fundamental_features import build_fundamental_feature_panel
from src.data_aggregate.utils.analyst_features import build_analyst_feature_panel
from src.data_aggregate.utils.earnings_features import build_earnings_feature_panel
from src.data_aggregate.utils.management_features import build_management_feature_panel
from src.data_aggregate.utils.employee_features import build_employee_feature_panel
from src.data_aggregate.utils.factors import (
    build_style_factor_returns,
    macro_change_factors,
    assemble_factor_panel,
    commodity_factor_returns,
    currency_factor_returns
)
from src.data_aggregate.utils.cube import build_cube_dataframe
from src.data_peers.step_deduce_peers import StepDeducePeers
from src.data_peers.utils.sector_peers import compute_sector_returns


class StepBuildCube(Step):

    def __init__(self, context: Context, config: DictConfig):
        super().__init__(context=context, config=config)
        self._cfg = config.build_cube

    def run(self):
        self.load_prices()
        self.normalize_prices()
        self.load_peers()
        self.load_fundamentals_and_macro()
        self.build_factor_panel()
        self.estimate_betas()
        self.build_targets()
        self.build_features()
        self.build_fundamental_features()
        self.build_earnings_features()
        self.build_analyst_features()
        self.build_management_features()
        self.build_employee_features()
        self.aggregate_cube()
        self.save_cube()

    # ------------------------------------------------------------------ #
    def load_prices(self):
        path = self._context.paths["PRICES_PATH"]
        self._log.info("Loading prices from %s", path)
        self.prices_long = pd.read_parquet(path)

      
    def normalize_prices(self):
        cfg = self._cfg
        raw = du.prices_long_to_multiindex(self.prices_long)

        self.close = du.extract_field(raw, "Close")
        self.open_ = du.extract_field(raw, "Open")
        self.high = du.extract_field(raw, "High") if "High" in raw.columns.get_level_values(0) else None
        self.low = du.extract_field(raw, "Low") if "Low" in raw.columns.get_level_values(0) else None

        trading_days = self.close[cfg.market_ticker].notna()
        self.close = self.close.loc[trading_days]
        self.open_ = self.open_.loc[trading_days]
        if self.high is not None:
            self.high = self.high.reindex(self.close.index)
            self.low = self.low.reindex(self.close.index)

        self.returns = du.daily_returns(self.close)
        self.mkt_ret = self.returns[cfg.market_ticker]
        self.market_close = self.close[cfg.market_ticker]

        drop_cols = self._config.data_extract.other_tickers
        self.stock_ret = self.returns.drop(columns=drop_cols)
        self.stock_close = self.close.drop(columns=drop_cols)
        self.stock_open = self.open_.drop(columns=drop_cols)
        self.stock_high = (self.high.drop(columns=drop_cols, errors="ignore")
                           if self.high is not None else None)
        self.stock_low = (self.low.drop(columns=drop_cols, errors="ignore")
                          if self.low is not None else None)

        self._log.info("Normalized prices: %s dates, %s stocks",
                       self.close.shape[0], self.stock_ret.shape[1])

    def load_peers(self):
        self.peers = StepDeducePeers(context=self._context, config=self._config).run()
        self.sector_ret = compute_sector_returns(self.stock_ret, self.peers)
        n = sum(1 for p in self.peers.values() if p)
        self._log.info("Sector returns ready for %s / %s tickers", n, len(self.peers))

    def load_fundamentals_and_macro(self):
        """Load fundamentals history and macro; both optional but recommended."""
        fpath = self._context.paths["FUNDAMENTALS_HISTORY_PATH"]
        self.fundamentals = pd.read_parquet(fpath) if fpath.exists() else None
        if self.fundamentals is None:
            self._log.warning("No fundamentals history -> value/quality factors "
                              "and peer-relative fundamentals will be skipped.")

        mpath = self._context.paths["MACRO_PATH"]
        self.macro = pd.read_parquet(mpath) if mpath.exists() else None
        if self.macro is None:
            self._log.warning("No macro file -> macro betas will be skipped.")

        apath = self._context.paths["ANALYST_ESTIMATES_HISTORY_PATH"]
        self.analyst = pd.read_parquet(apath) if apath.exists() else None
        if self.analyst is None:
            self._log.warning("No analyst-estimate history -> analyst features skipped.")

        epath = self._context.paths["EARNINGS_SURPRISES_PATH"]
        self.earnings = pd.read_parquet(epath) if epath.exists() else None
        if self.earnings is None:
            self._log.warning("No earnings-surprise history -> earnings expectation "
                              "features skipped (run fetch_earnings_surprises).")

        gpath = self._context.paths["MANAGEMENT_HISTORY_PATH"]
        self.management = pd.read_parquet(gpath) if gpath.exists() else None
        if self.management is None:
            self._log.warning("No management/ownership history -> governance features "
                              "skipped (run fetch_management).")

        wpath = self._context.paths["EMPLOYEES_HISTORY_PATH"]
        self.employees = pd.read_parquet(wpath) if wpath.exists() else None
        if self.employees is None:
            self._log.warning("No employee-count history -> workforce features skipped "
                              "(run fetch_employees; needs FMP_API_KEY).")

    def _intrinsic_cfg(self) -> dict:
        cfg = self._cfg.get("intrinsic", {})
        return OmegaConf.to_container(cfg, resolve=True) if cfg else {}

    def build_factor_panel(self):
        """Style (price + fundamentals) + macro (changes) -> shared factor panel."""
        cfg = self._cfg.get("factors", {})
        resvol_window = cfg.get("resvol_window", 63)

        style = build_style_factor_returns(
            self.stock_close, self.stock_ret, self.fundamentals, resvol_window
        )

        if self.macro is not None:
            macro_chg = macro_change_factors(self.macro, self.stock_close.index)
        else:
            macro_chg = pd.DataFrame(index=self.stock_close.index)
        self.macro_cols = list(macro_chg.columns)

        #retreive commo info

        commodity_returns = commodity_factor_returns(self.close, tickers={"oil": "CL=F", "gold": "GC=F"})
        currency_returns = currency_factor_returns(self.close, tickers={"USD/EUR": "USDEUR=X"})

        self.factor_panel, self.macro_cols = assemble_factor_panel(self.mkt_ret, style, commodity_returns, currency_returns, macro_chg)
        self._log.info(
            "Factor panel: %s factors (%s style/market, %s macro)",
            self.factor_panel.shape[1],
            self.factor_panel.shape[1] - len(self.macro_cols),
            len(self.macro_cols),
        )

    def estimate_betas(self):
        cfg = self._cfg.betas
        self.betas = estimate_all_betas(
            self.stock_ret,
            self.factor_panel,                 # market is inside the panel now
            self.sector_ret,
            window=cfg.window,
            min_obs=cfg.min_obs,
            ridge=cfg.get("ridge", 5.0),
            step=cfg.get("step", 5),
        )

        # check if beta_market_simple is in the betas
        to_kick = []
        for t in self.betas:
            if "beta_market_simple" not in self.betas[t]:
                to_kick.append(t)
                self._log.warning(f"beta_market_simple not in the {t}")
                continue
        
        self.betas = {t: v for t, v in self.betas.items() if t not in to_kick}

        bm = np.nanmean([self.betas[t]["beta_market_simple"].mean() for t in self.betas])
        self._log.info("Estimated multi-factor betas for %s tickers "
                        "(mean beta_market_simple=%.2f)", len(self.betas), bm)

    def build_targets(self):
        cfg = self._cfg.targets
        self.labels = build_targets(
            close=self.stock_close,
            stock_returns=self.stock_ret,
            peer_dict=self.peers,
            betas=self.betas,
            factor_panel=self.factor_panel,
            macro_cols=self.macro_cols,
            horizons=tuple(cfg.horizons),
            label=cfg.label,
            min_names=cfg.min_names,
            neutralize_momentum=cfg.get("neutralize_momentum", True),
        )
        non_null = sum(int(df.notna().sum().sum()) for df in self.labels.values())
        self._log.info("Built factor-neutral targets for horizons %s (non-null=%s)",
                       list(cfg.horizons), non_null)

    def build_features(self):
        cfg = self._cfg.features
        self.feature_panel = build_feature_panel(
            self.stock_close, self.stock_open, self.sector_ret,
            method=cfg.standardize_method,
            high=self.stock_high, low=self.stock_low,
        )
        self._log.info("Price feature panel: %s rows, %s features",
                       len(self.feature_panel), len(self.feature_panel.columns) - 2)

    def build_fundamental_features(self):
        """Peer-relative fundamentals (firm vs direct competitors) -> merged in.

        Quarter-basis and leak-free by construction: the SEC history is now
        quarterly (TTM levels) and every value is keyed on its FILING date
        (`as_of`); `fundamentals_to_daily` forward-fills each value only from
        that date, so a feature on day d reflects the most recent quarter whose
        10-Q/10-K was already public on d -- never a not-yet-filed quarter.
        """
        fund_panel = build_fundamental_feature_panel(
            self.fundamentals, self.peers, self.stock_close.index,
            stock_close=self.stock_close, intrinsic_cfg=self._intrinsic_cfg(),
        )
        if fund_panel.empty:
            self._log.warning("No fundamental features built (missing fundamentals).")
            return
        before = len(self.feature_panel.columns) - 2
        self.feature_panel = self.feature_panel.merge(
            fund_panel, on=["date", "ticker"], how="left"
        )
        added = len(self.feature_panel.columns) - 2 - before
        self._log.info("Merged %s peer-relative fundamental features", added)

    def build_earnings_features(self):
        """Historical earnings-expectation features: forward EPS yield, expected
        EPS growth, and realized surprise (beat/miss). Built from the earnings
        archive (fetch_earnings_surprises); genuinely historical and point-in-
        time (forward estimate only within its own quarter, actual only after
        the report)."""
        panel = build_earnings_feature_panel(
            self.earnings, self.peers, self.stock_close.index,
            stock_close=self.stock_close,
        )
        if panel.empty:
            self._log.warning("No earnings-expectation features built.")
            return
        before = len(self.feature_panel.columns) - 2
        self.feature_panel = self.feature_panel.merge(
            panel, on=["date", "ticker"], how="left"
        )
        added = len(self.feature_panel.columns) - 2 - before
        cov = panel.drop(columns=["date", "ticker"]).notna().any(axis=1).mean()
        self._log.info("Merged %s earnings-expectation features (row coverage %.1f%%)",
                       added, 100 * cov)

    def build_analyst_features(self):
        """Sell-side analyst-estimate features (level, revisions, and estimates
        vs our intrinsic value). Point-in-time from each `as_of`, so leak-free;
        coverage only accrues as the estimate history is collected over time."""
        panel = build_analyst_feature_panel(
            self.analyst, self.peers, self.stock_close.index,
            stock_close=self.stock_close, fundamentals_history=self.fundamentals,
            intrinsic_cfg=self._intrinsic_cfg(),
        )
        if panel.empty:
            self._log.warning("No analyst features built (missing/empty estimate history).")
            return
        before = len(self.feature_panel.columns) - 2
        self.feature_panel = self.feature_panel.merge(
            panel, on=["date", "ticker"], how="left"
        )
        added = len(self.feature_panel.columns) - 2 - before
        cov = panel.drop(columns=["date", "ticker"]).notna().any(axis=1).mean()
        self._log.info("Merged %s analyst-estimate features (row coverage %.1f%%)",
                       added, 100 * cov)

    def build_management_features(self):
        """Governance / ownership / workforce features (founder-led, family-owned,
        insider & institutional ownership, net insider buying, CEO age, revenue per
        employee). Point-in-time from each `as_of`, so leak-free; coverage only
        accrues as the management snapshot is collected over time."""
        panel = build_management_feature_panel(
            self.management, self.peers, self.stock_close.index,
        )
        if panel.empty:
            self._log.warning("No management/ownership features built.")
            return
        before = len(self.feature_panel.columns) - 2
        self.feature_panel = self.feature_panel.merge(
            panel, on=["date", "ticker"], how="left"
        )
        added = len(self.feature_panel.columns) - 2 - before
        cov = panel.drop(columns=["date", "ticker"]).notna().any(axis=1).mean()
        self._log.info("Merged %s management/ownership features (row coverage %.1f%%)",
                       added, 100 * cov)

    def build_employee_features(self):
        """Workforce features (revenue per employee, YoY headcount growth) from the
        FMP historical employee-count archive. Genuinely historical and point-in-
        time (stepwise from each filing's `as_of`), so backtestable."""
        panel = build_employee_feature_panel(
            self.employees, self.peers, self.stock_close.index,
            fundamentals_history=self.fundamentals,
        )
        if panel.empty:
            self._log.warning("No workforce features built.")
            return
        before = len(self.feature_panel.columns) - 2
        self.feature_panel = self.feature_panel.merge(
            panel, on=["date", "ticker"], how="left"
        )
        added = len(self.feature_panel.columns) - 2 - before
        cov = panel.drop(columns=["date", "ticker"]).notna().any(axis=1).mean()
        self._log.info("Merged %s workforce features (row coverage %.1f%%)",
                       added, 100 * cov)

    def aggregate_cube(self):
        self.cube = build_cube_dataframe(
            self.feature_panel, self.labels, self.betas, self.peers,
        )
        self._log.info("Aggregated cube: %s rows, %s horizons, %s tickers",
                       len(self.cube), self.cube["target_horizon"].nunique(),
                       self.cube["ticker"].nunique())

    def save_cube(self):
        cube_path = self._context.paths["CUBE_PATH"]
        cube_path.parent.mkdir(parents=True, exist_ok=True)
        self.cube.to_parquet(cube_path, index=False)
        self._log.info("Saved cube to %s", cube_path)
