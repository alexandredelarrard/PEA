import pandas as pd
from omegaconf import DictConfig

from src.utils.step import Step
from src.constants.constants_price import MACRO_MARKET_SERIES
from src.context import Context
from src.data_store.schema import Tables
from src.utils.macro import load_macro_series
from src.utils.universe import load_universe_tickers
from src.data_aggregate.utils.common import data_utils as du
from src.data_peers.utils.embeddings import (
    fetch_business_descriptions,
    get_openai_embeddings,
    load_embedded_tickers,
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
            return self.load_pre_computed_peers(peers_path)
        
        self.load_prices()
        self.normalize_prices()
        self.build_peers()
        self.save_peers()
        return self.peers

    def load_prices(self):
        self._log.info("Loading prices from DB table 'prices'")
        self.prices_long = self._context.store.load("prices")

    def load_pre_computed_peers(self, peers_path):
        self.peers = load_peer_dict(peers_path)
        n = sum(1 for p in self.peers.values() if p)
        self._log.info("Loaded peer dict from %s (%s tickers, %s with peers)",
                        peers_path, len(self.peers), n)
        return self.peers

    def normalize_prices(self):

        raw = du.prices_long_to_multiindex(self.prices_long)
        # `CloseTotal`: the peer graph is built from the CORRELATION of daily returns, so it
        # needs the buy-and-hold path. On the price-only series two names with different
        # dividend policies look less correlated than they are.
        self.close_total = du.extract_field(raw, "CloseTotal")

        # The trading calendar is the days the MARKET traded, which now lives in
        # `prices_macro` rather than as a column inside this equity frame. Same definition as
        # the cube's (du.get_trading_days), just sourced from the table that owns it.
        market = load_macro_series(self._context.store, MACRO_MARKET_SERIES)
        if market is None:
            raise RuntimeError(f"'{Tables.prices_macro}' has no '{MACRO_MARKET_SERIES}' rows -> "
                               "no trading calendar for the peer graph. Run `data_extract macro`.")
        
        self.close_total = self.close_total.loc[market.reindex(self.close_total.index).notna()]
        self.returns = du.daily_returns(self.close_total)
        # no market column to drop: `prices` is the equity universe and nothing else
        self.stock_ret = self.returns
        # restrict to the authoritative universe (sp500_tickers) so peers are built
        # ONLY among analysed names — swap that table and the peer graph reroutes.
        universe = [t for t in load_universe_tickers(self._context)
                    if t in self.stock_ret.columns]
        if universe:
            self.stock_ret = self.stock_ret[universe]
        else:
            self._log.warning("sp500_tickers empty -> peers over ALL priced names; "
                              "seed the universe table to scope peers")
        self._log.info("Normalized prices: %s dates, %s stocks",
                       self.close_total.shape[0], self.stock_ret.shape[1])

    def _embedding_similarity(self):
        """Fetch descriptions -> OpenAI embeddings (cached) -> cosine similarity.
        Returns None on any failure so the caller falls back to correlation-only.

        INCREMENTAL: `ticker_embeddings` is the single 'done' gate — tickers already
        in that table skip BOTH the Yahoo description fetch and the OpenAI embedding;
        only tickers missing from it are (re)processed. `universe` still returns the
        full cached-plus-new matrix for the similarity computation."""
        store = self._context.store
        tickers = list(self.stock_ret.columns)
        try:
            done = load_embedded_tickers(store)
            todo = [t for t in tickers if t not in done]
            self._log.info("Embeddings: %s/%s tickers already in ticker_embeddings, "
                           "%s to (re)process", len(tickers) - len(todo), len(tickers), len(todo))
            # descriptions (Yahoo) only for the not-done tickers; already-embedded
            # tickers never touch Yahoo or OpenAI again.
            descriptions = fetch_business_descriptions(todo, store=store) if todo else {}
            emb = get_openai_embeddings(
                descriptions,
                model=self._cfg.get("embedding_model", "text-embedding-3-small"),
                store=store,
                universe=tickers,          # return every ticker's vector (cached + new)
            )
            if emb.empty:
                self._log.warning("No embeddings available -> corr-only peers")
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
