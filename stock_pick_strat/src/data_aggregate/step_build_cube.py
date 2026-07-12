import numpy as np
import pandas as pd
from omegaconf import DictConfig

from src.utils.step import Step
from src.context import Context
from src.data_aggregate.utils import data_utils as du
from src.data_aggregate.utils.betas import estimate_all_betas
from src.data_aggregate.utils.targets import build_targets
from src.data_aggregate.utils.features import build_feature_panel
from src.data_aggregate.utils.fundamental_features import build_fundamental_feature_panel
from src.data_aggregate.utils.factors import (
    build_style_factor_returns,
    build_macro_factor_changes,
    assemble_factor_panel,
)
from src.data_aggregate.utils.cube import build_cube_dataframe
from src.data_aggregate.step_deduce_peers import StepDeducePeers
from src.modelling.utils_model.sector_peers import compute_sector_returns


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

        trading_days = self.close[cfg.market_ticker].notna()
        self.close = self.close.loc[trading_days]
        self.open_ = self.open_.loc[trading_days]

        self.returns = du.daily_returns(self.close)
        self.mkt_ret = self.returns[cfg.market_ticker]
        self.market_close = self.close[cfg.market_ticker]

        drop_cols = self._config.data_extract.other_tickers
        self.stock_ret = self.returns.drop(columns=drop_cols)
        self.stock_close = self.close.drop(columns=drop_cols)
        self.stock_open = self.open_.drop(columns=drop_cols)

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

    def build_factor_panel(self):
        """Style (price + fundamentals) + macro (changes) -> shared factor panel."""
        cfg = self._cfg.get("factors", {})
        resvol_window = cfg.get("resvol_window", 63)

        style = build_style_factor_returns(
            self.stock_close, self.stock_ret, self.fundamentals, resvol_window
        )

        if self.macro is not None:
            macro_chg = build_macro_factor_changes(self.macro, self.stock_close.index)
        else:
            macro_chg = pd.DataFrame(index=self.stock_close.index)
        self.macro_cols = list(macro_chg.columns)

        self.factor_panel = assemble_factor_panel(self.mkt_ret, style, macro_chg)
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
        )
        non_null = sum(int(df.notna().sum().sum()) for df in self.labels.values())
        self._log.info("Built factor-neutral targets for horizons %s (non-null=%s)",
                       list(cfg.horizons), non_null)

    def build_features(self):
        cfg = self._cfg.features
        self.feature_panel = build_feature_panel(
            self.stock_close, self.stock_open, self.sector_ret,
            method=cfg.standardize_method,
        )
        self._log.info("Price feature panel: %s rows, %s features",
                       len(self.feature_panel), len(self.feature_panel.columns) - 2)

    def build_fundamental_features(self):
        """Peer-relative fundamentals (firm vs direct competitors) -> merged in."""
        fund_panel = build_fundamental_feature_panel(
            self.fundamentals, self.peers, self.stock_close.index
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
