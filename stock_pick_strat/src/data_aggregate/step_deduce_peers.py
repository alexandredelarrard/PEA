import pandas as pd
from omegaconf import DictConfig

from src.utils.step import Step
from src.context import Context
from src.data_aggregate.utils import data_utils as du
from src.modelling.utils_model.sector_peers import (
    build_peer_dict,
    load_peer_dict,
    save_peer_dict,
)

class StepDeducePeers(Step):

    def __init__(self, context: Context, config: DictConfig):
        super().__init__(context=context, config=config)
        self._cfg = config.build_cube

    def run(self):
        peers_path = self._context.paths["SECTOR_PEERS_PATH"]
        if peers_path.exists():
            self.peers = load_peer_dict(peers_path)
            n_with_peers = sum(1 for p in self.peers.values() if p)
            self._log.info(
                "Loaded existing peer dict from %s (%s tickers, %s with peers)",
                peers_path,
                len(self.peers),
                n_with_peers,
            )
            return self.peers

        self.load_prices()
        self.normalize_prices()
        self.build_peers()
        self.save_peers()
        
        return self.peers

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
        self.stock_ret = self.returns.drop(columns=[cfg.market_ticker])

        self._log.info(
            "Normalized prices: %s dates, %s stocks",
            self.close.shape[0],
            self.stock_ret.shape[1],
        )

    def build_peers(self):
        cfg = self._cfg.peers
        self.peers = build_peer_dict(
            self.stock_ret,
            top_k=cfg.top_k,
            weighting=cfg.weighting,
            min_obs=cfg.min_obs,
        )
        n_with_peers = sum(1 for p in self.peers.values() if p)
        self._log.info("Built peer baskets for %s / %s tickers", n_with_peers, len(self.peers))

    def save_peers(self):
        peers_path = self._context.paths["SECTOR_PEERS_PATH"]
        save_peer_dict(self.peers, peers_path)
        self._log.info("Saved peer dict to %s", peers_path)
