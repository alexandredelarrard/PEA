import pandas as pd
from omegaconf import DictConfig

from src.utils.step import Step
from src.context import Context
from src.data_aggregate.utils import data_utils as du
from src.data_peers.utils.embeddings import (
    fetch_business_descriptions,
    get_openai_embeddings,
)
from src.data_peers.utils.sector_peers import (
    build_peer_dict,
    build_peer_dict_hybrid,
    cosine_similarity_matrix,
    load_peer_dict,
    save_peer_dict,
)


class StepDeducePeers(Step):
    """
    Peer baskets from BOTH return correlation and business-description embedding
    similarity (OpenAI). Correlation captures statistical co-movement; embeddings
    capture actual business similarity, so a stock spuriously correlated with an
    unrelated cohort is pulled back to its real peers. Set peers.use_embeddings:
    false to fall back to correlation-only.
    """

    def __init__(self, context: Context, config: DictConfig):
        super().__init__(context=context, config=config)
        self._cfg = config.peers

    def run(self):
        peers_path = self._context.paths["SECTOR_PEERS_PATH"]
        if peers_path.exists():
            self.peers = load_peer_dict(peers_path)
            n = sum(1 for p in self.peers.values() if p)
            self._log.info("Loaded peer dict from %s (%s tickers, %s with peers)",
                           peers_path, len(self.peers), n)
            return self.peers

        self.load_prices()
        self.normalize_prices()
        self.build_peers()
        self.save_peers()
        return self.peers

    def load_prices(self):
        self._log.info("Loading prices from DB table 'prices'")
        self.prices_long = self._context.store.load("prices")

    def normalize_prices(self):
        mkt = self._config.build_cube.market_ticker

        raw = du.prices_long_to_multiindex(self.prices_long)
        self.close = du.extract_field(raw, "Close")
        trading_days = self.close[mkt].notna()
        self.close = self.close.loc[trading_days]
        self.returns = du.daily_returns(self.close)
        self.stock_ret = self.returns.drop(columns=[mkt])
        self._log.info("Normalized prices: %s dates, %s stocks",
                       self.close.shape[0], self.stock_ret.shape[1])

    def _embedding_similarity(self):
        """Fetch descriptions -> OpenAI embeddings (cached) -> cosine similarity.
        Returns None on any failure so the caller falls back to correlation-only."""
        tickers = list(self.stock_ret.columns)
        try:
            descriptions = fetch_business_descriptions(tickers, store=self._context.store)
            if not descriptions:
                self._log.warning("No business descriptions fetched -> corr-only peers")
                return None
            emb = get_openai_embeddings(
                descriptions,
                model=self._cfg.get("embedding_model", "text-embedding-3-small"),
                store=self._context.store,
            )
            if emb.empty:
                return None
            self._log.info("Embedded %s / %s tickers", len(emb), len(tickers))
            return cosine_similarity_matrix(emb)
        except Exception as e:
            self._log.warning("Embedding step failed (%s) -> corr-only peers", e)
            return None

    def build_peers(self):
        
        use_emb = self._cfg.get("use_embeddings", False)
        embed_sim = self._embedding_similarity() if use_emb else None

        if embed_sim is not None:
            self.peers = build_peer_dict_hybrid(
                self.stock_ret, embed_sim,
                top_k=self._cfg.top_k, weighting=self._cfg.weighting, min_obs=self._cfg.min_obs,
                w_corr=self._cfg.get("w_corr", 0.5), w_embed=self._cfg.get("w_embed", 0.5),
            )
            self._log.info("Built HYBRID (corr + embedding) peer baskets "
                           "(w_corr=%.2f w_embed=%.2f)",
                           self._cfg.get("w_corr", 0.5), self._cfg.get("w_embed", 0.5))
        else:
            self.peers = build_peer_dict(
                self.stock_ret, top_k=self._cfg.top_k,
                weighting=self._cfg.weighting, min_obs=self._cfg.min_obs)
            self._log.info("Built CORRELATION-ONLY peer baskets")

        n = sum(1 for p in self.peers.values() if p)
        self._log.info("Peer baskets for %s / %s tickers", n, len(self.peers))

    def save_peers(self):
        peers_path = self._context.paths["SECTOR_PEERS_PATH"]
        save_peer_dict(self.peers, peers_path)
        self._log.info("Saved peer dict to %s", peers_path)
