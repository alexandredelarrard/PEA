import json

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from src.utils.step import Step
from src.context import Context
from src.utils.universe import load_universe_tickers
from src.constants.constants import SUPERINVESTORS_JSON
from src.data_aggregate.utils import data_utils as du
from src.data_aggregate.utils.betas import estimate_all_betas
from src.data_aggregate.utils.targets import build_targets_multi
from src.data_aggregate.utils.features import build_feature_panel
from src.data_aggregate.utils.fundamental_features import build_fundamental_feature_panel
from src.data_aggregate.utils.earnings_features import build_earnings_feature_panel
from src.data_aggregate.utils.governance_features import build_governance_feature_panel
from src.data_aggregate.utils.def14a_impute import impute_def14a
from src.data_aggregate.utils.sector_features import build_sector_feature_panel
from src.data_aggregate.utils.employee_features import build_employee_feature_panel
from src.data_aggregate.utils.dividend_features import build_dividend_feature_panel
from src.data_aggregate.utils.attention_features import build_combined_attention_panel
from src.data_aggregate.utils.earnings_call_features import (
    build_earnings_call_feature_panel,
    score_earnings_calls,
)
from src.data_aggregate.utils.earnings_call_embeddings import embed_earnings_calls
from src.data_aggregate.utils.institutional_features import build_institutional_feature_panel
from src.data_aggregate.utils.superinvestor_features import build_superinvestor_feature_panel
from src.data_aggregate.utils.insider_features import build_insider_feature_panel
from src.data_aggregate.utils.short_interest_features import build_short_interest_feature_panel
from src.data_aggregate.utils.composites import build_composites as build_composite_signals
from src.data_aggregate.utils.factors import (
    build_style_factor_returns,
    macro_change_factors,
    assemble_factor_panel,
    commodity_factor_returns,
    currency_factor_returns
)
from src.data_aggregate.utils.cube import build_cube_dataframe, _betas_to_long, _labels_to_long
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
        self.build_sector_features()
        self.build_earnings_features()
        self.build_governance_features()
        self.build_employee_features()
        self.build_dividend_features()
        self.build_attention_features()
        self.build_institutional_features()
        self.build_superinvestor_features()
        self.build_insider_features()
        self.build_short_interest_features()
        self.build_earnings_call_features()
        self.build_composite_signals()
        self.aggregate_cube()
        self.save_cube()

    # ------------------------------------------------------------------ #
    def load_prices(self):
        self._log.info("Loading prices from DB table 'prices'")
        self.prices_long = self._context.store.load("prices")

      
    def normalize_prices(self):
        cfg = self._cfg
        raw = du.prices_long_to_multiindex(self.prices_long)

        self.close = du.extract_field(raw, "Close")
        self.open_ = du.extract_field(raw, "Open")
        self.high = du.extract_field(raw, "High") if "High" in raw.columns.get_level_values(0) else None
        self.low = du.extract_field(raw, "Low") if "Low" in raw.columns.get_level_values(0) else None
        self.volume = du.extract_field(raw, "Volume") if "Volume" in raw.columns.get_level_values(0) else None

        trading_days = self.close[cfg.market_ticker].notna()
        # Surface interior calendar holes BEFORE dropping: dates where a quorum of
        # stocks trade but the market_ticker (which defines the calendar) is
        # missing get dropped for the WHOLE universe -> the classic "no stock for
        # month X" cube symptom. Warn so it is not silent; heal via price extraction.
        stock_cov = (self.close.drop(columns=[cfg.market_ticker], errors="ignore")
                     .notna().sum(axis=1))
        quorum = 0.5 * max(1, self.close.shape[1] - 1)
        holes = self.close.index[(~trading_days) & (stock_cov >= quorum)]
        if len(holes):
            self._log.warning(
                "%s (market_ticker) missing on %d date(s) where >=50%% of stocks "
                "trade (%s .. %s) -> these dates are dropped for the ENTIRE universe. "
                "Re-run price extraction to backfill %s (interior-gap heal in "
                "fetch_prices).", cfg.market_ticker, len(holes),
                holes.min().date(), holes.max().date(), cfg.market_ticker)
        self.close = self.close.loc[trading_days]
        self.open_ = self.open_.loc[trading_days]
        if self.high is not None:
            self.high = self.high.reindex(self.close.index)
            self.low = self.low.reindex(self.close.index)

        self.returns = du.daily_returns(self.close)
        self.mkt_ret = self.returns[cfg.market_ticker]
        self.market_close = self.close[cfg.market_ticker]

        # Analysis universe = the `sp500_tickers` table (single entry point). Restrict
        # every stock_* frame to it so the cube is built ONLY for the analysed names —
        # swap that table (e.g. Russell 1000) and the whole cube reroutes. Fall back to
        # "all priced names minus benchmark/macro" if the table is not yet seeded.
        avail = set(self.close.columns)
        universe = [t for t in load_universe_tickers(self._context) if t in avail]
        if not universe:
            drop_cols = set(self._config.data_extract.other_tickers)
            universe = [c for c in self.close.columns if c not in drop_cols]
            self._log.warning("sp500_tickers empty/unseeded -> cube universe = all priced "
                              "names minus benchmark/macro; seed it to scope the cube")

        def _sub(f):
            return f[[c for c in universe if c in f.columns]] if f is not None else None

        self.stock_ret = _sub(self.returns)
        self.stock_close = _sub(self.close)
        self.stock_open = _sub(self.open_)
        self.stock_high = _sub(self.high)
        self.stock_low = _sub(self.low)
        self.stock_volume = _sub(self.volume.reindex(self.close.index)
                                 if self.volume is not None else None)

        self._log.info("Normalized prices: %s dates, %s stocks",
                       self.close.shape[0], self.stock_ret.shape[1])

    def load_peers(self):
        self.peers = StepDeducePeers(context=self._context, config=self._config).run()
        self.sector_ret = compute_sector_returns(self.stock_ret, self.peers)
        n = sum(1 for p in self.peers.values() if p)
        self._log.info("Sector returns ready for %s / %s tickers", n, len(self.peers))

    def _load_or_none(self, table: str) -> pd.DataFrame | None:
        """Load a DB table, returning None when it is absent/empty (so every
        downstream feature builder keeps its 'optional source' semantics)."""
        df = self._context.store.load(table)
        return None if df.empty else df

    def load_fundamentals_and_macro(self):
        """Load fundamentals history and macro; both optional but recommended."""
        self.fundamentals = self._load_or_none("fundamentals_history")
        if self.fundamentals is None:
            self._log.warning("No fundamentals history -> value/quality factors "
                              "and peer-relative fundamentals will be skipped.")

        self.macro = self._load_or_none("macro")
        if self.macro is None:
            self._log.warning("No macro data -> macro betas will be skipped.")

        self.earnings = self._load_or_none("earnings_surprises")
        if self.earnings is None:
            self._log.warning("No earnings-surprise history -> earnings expectation "
                              "features skipped (run fetch_earnings_surprises).")

        self.def14a = self._load_or_none("def14a_llm")
        if self.def14a is None:
            self._log.warning("No DEF 14A LLM proxy history -> executive-pay/board "
                              "features skipped (run fetch_def14a_llm).")
        else:
            self.def14a, imp_stats = impute_def14a(self.def14a)
            if imp_stats:
                self._log.info("DEF 14A clean-on-read: deduced %d missing cells across "
                               "%d rules (raw table untouched).", sum(imp_stats.values()),
                               len(imp_stats))

        self.employees = self._load_or_none("employees_history")
        if self.employees is None:
            self._log.warning("No employee-count history -> workforce features skipped "
                              "(run fetch_employees; needs FMP_API_KEY).")

        self.pension_facts = self._load_or_none("pension_facts")
        if self.pension_facts is None:
            self._log.warning("No pension_facts (Financial Statement Data Sets) -> the "
                              "companyfacts pensionDeficit is used for off-BS leverage.")

        self.notes_num = self._load_or_none("notes_num")
        if self.notes_num is None:
            self._log.warning("No notes_num (Financial Statement & Notes sets) -> footnote "
                              "pension detail (PBO/plan assets/funded ratio) skipped "
                              "(run fetch_financial_notes).")

        self.insider = self._load_or_none("insider_transactions")
        if self.insider is None:
            self._log.warning("No insider_transactions -> insider-trading features skipped "
                              "(run fetch_insider_transactions).")

        self.dividends = self._load_or_none("dividends")
        if self.dividends is None:
            self._log.warning("No dividend history -> dividend/shareholder-yield "
                              "features skipped (dividends come from the price download, "
                              "fetch_price_history -> StepExtractPrices).")

        self.wiki_pageviews = self._load_or_none("wiki_pageviews")
        self.google_trends = self._load_or_none("google_trends")
        if self.wiki_pageviews is None and self.google_trends is None:
            self._log.warning("No attention data -> Wikipedia/Google-Trends features "
                              "skipped (run fetch_wiki_pageviews / fetch_google_trends).")

        self.institutional = self._load_or_none("institutional_holdings")
        if self.institutional is None:
            self._log.warning("No 13F holdings -> institutional-ownership features "
                              "skipped (run fetch_13f).")

        self.short_interest = self._load_or_none("short_interest")
        if self.short_interest is None:
            self._log.warning("No short-volume data -> short-interest features "
                              "skipped (run fetch_short_interest).")

        self.fails_to_deliver = self._load_or_none("fails_to_deliver")
        if self.fails_to_deliver is None:
            self._log.warning("No fails-to-deliver data -> FTD features skipped "
                              "(run fetch_fails_to_deliver).")

        self.earnings_call_sections = self._load_or_none("earnings_call_sections")
        if self.earnings_call_sections is None:
            self._log.warning("No earnings_call_sections -> earnings-call sentiment/text "
                              "features skipped (run fetch_earnings_calls).")

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

    def _gics_group_maps(self) -> dict[str, dict[str, str]]:
        """Ticker -> GICS sector / industry_group maps (from sp500_tickers) used to
        neutralize the target to the ACTUAL sector + industry (not the peer basket)."""
        ref = self._context.store.load("sp500_tickers")
        maps: dict[str, dict[str, str]] = {}
        for col in ("sector", "industry_group"):
            if not ref.empty and col in ref.columns:
                maps[col] = {str(t): str(g) for t, g in zip(ref["ticker"], ref[col])
                             if pd.notna(g) and str(g).strip()}
        return maps

    def build_targets(self):
        cfg = self._cfg.targets
        # store EVERY configured target version (e.g. rank AND zscore) in the cube
        # so the modelling step can pick one via model.target_type without a rebuild
        label_types = list(cfg.get("labels", [cfg.get("label", "rank")]))
        # neutralize the target to the ACTUAL GICS sector + industry (per-day within-
        # group demeaning) INSTEAD of the return-correlation peer basket, so sector /
        # industry membership can't predict the target (else they dominate the model).
        sector_groups = self._gics_group_maps() if cfg.get("neutralize_sectors", True) else None
        self.labels = build_targets_multi(
            close=self.stock_close,
            stock_returns=self.stock_ret,
            peer_dict=self.peers,
            betas=self.betas,
            factor_panel=self.factor_panel,
            macro_cols=self.macro_cols,
            horizons=tuple(cfg.horizons),
            labels=tuple(label_types),
            min_names=cfg.min_names,
            neutralize_momentum=cfg.get("neutralize_momentum", True),
            sector_groups=sector_groups,
        )
        non_null = sum(int(df.notna().sum().sum())
                       for per in self.labels.values() for df in per.values())
        self._log.info("Built factor-neutral targets %s for horizons %s "
                       "(GICS sector+industry-neutral=%s, non-null=%s)",
                       label_types, list(cfg.horizons), sector_groups is not None, non_null)

    def build_features(self):
        cfg = self._cfg.features
        self.feature_panel = build_feature_panel(
            self.stock_close, self.stock_open, self.sector_ret,
            method=cfg.standardize_method,
            high=self.stock_high, low=self.stock_low,
            volume=getattr(self, "stock_volume", None),
            seasonal_horizons=list(self._cfg.targets.horizons),
        )
        self._log.info("Price feature panel: %s rows, %s features (volume liquidity: %s)",
                       len(self.feature_panel), len(self.feature_panel.columns) - 2,
                       "yes" if getattr(self, "stock_volume", None) is not None else "no")

    def build_fundamental_features(self):
        """Peer-relative fundamentals (firm vs direct competitors) -> merged in.

        Quarter-basis and leak-free by construction: the SEC history is now
        quarterly (TTM levels) and every value is keyed on its FILING date
        (`as_of`); `fundamentals_to_daily` forward-fills each value only from
        that date, so a feature on day d reflects the most recent quarter whose
        10-Q/10-K was already public on d -- never a not-yet-filed quarter.
        """
        hist = self._cfg.get("hist", {})
        fund_panel = build_fundamental_feature_panel(
            self.fundamentals, self.peers, self.stock_close.index,
            stock_close=self.stock_close, intrinsic_cfg=self._intrinsic_cfg(),
            hist_window=int(hist.get("window", 1260)),
            hist_min_periods=int(hist.get("min_periods", 252)),
            earnings_history=getattr(self, "earnings", None),   # PEGY projected-growth term
            pension_facts=getattr(self, "pension_facts", None), # bulk off-BS pension deficit
            notes_num=getattr(self, "notes_num", None),         # footnote PBO / plan assets
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

    def build_sector_features(self):
        """Sector-specific fundamental KPIs (combined/loss ratio, NIM, efficiency
        ratio, FFO, inventory days, shareholder payout, net-debt/EBITDA, accruals).
        Availability-gated per row (a KPI is null unless its sector reported the
        inputs), then peer-relative + neutralized like the other panels."""
        panel = build_sector_feature_panel(
            self.fundamentals, self.peers, self.stock_close.index,
        )
        if panel.empty:
            self._log.warning("No sector KPI features built (missing fundamentals).")
            return
        before = len(self.feature_panel.columns) - 2
        self.feature_panel = self.feature_panel.merge(
            panel, on=["date", "ticker"], how="left"
        )
        added = len(self.feature_panel.columns) - 2 - before
        cov = panel.drop(columns=["date", "ticker"]).notna().any(axis=1).mean()
        self._log.info("Merged %s sector-KPI features (row coverage %.1f%%)",
                       added, 100 * cov)

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

    def build_governance_features(self):
        """Executive-pay & board features from the LLM-extracted DEF 14A archive
        (CEO pay growth, pay-vs-revenue-growth misalignment, pay ratio, board
        independence/diversity/tenure, say-on-pay). Peer-relative and point-in-time
        from each proxy's `as_of`; coverage accrues as fetch_def14a_llm runs."""
        panel = build_governance_feature_panel(
            self.def14a, self.peers, self.stock_close.index,
            fundamentals_history=self.fundamentals,
        )
        if panel.empty:
            self._log.warning("No governance/executive-pay features built "
                              "(def14a_llm empty — accrues as fetch_def14a_llm runs).")
            return
        before = len(self.feature_panel.columns) - 2
        self.feature_panel = self.feature_panel.merge(
            panel, on=["date", "ticker"], how="left"
        )
        added = len(self.feature_panel.columns) - 2 - before
        cov = panel.drop(columns=["date", "ticker"]).notna().any(axis=1).mean()
        self._log.info("Merged %s governance/executive-pay features (row coverage %.1f%%)",
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

    def build_dividend_features(self):
        """Dividend / shareholder-yield features (TTM yield, 1y + 5y payout growth,
        payer flag, payout ratio, FCF coverage, dividend + buyback yield). RECONCILES
        the two dividend sources: the per-share ex-date history (`dividends`, primary)
        and the SEC cash-flow `dividendsPaid` total (from `fundamentals`, gap-fill +
        payout/coverage). Point-in-time; non-payers get a real 0 yield so they rank
        correctly."""
        panel = build_dividend_feature_panel(
            getattr(self, "dividends", None), self.peers, self.stock_close.index,
            stock_close=self.stock_close, fundamentals_history=self.fundamentals,
        )
        if panel.empty:
            self._log.warning("No dividend features built (missing dividend history).")
            return
        before = len(self.feature_panel.columns) - 2
        self.feature_panel = self.feature_panel.merge(
            panel, on=["date", "ticker"], how="left"
        )
        added = len(self.feature_panel.columns) - 2 - before
        cov = panel.drop(columns=["date", "ticker"]).notna().any(axis=1).mean()
        self._log.info("Merged %s dividend features (row coverage %.1f%%)",
                       added, 100 * cov)

    def build_attention_features(self):
        """Retail-attention features: Wikipedia pageviews and Google Trends search
        interest are two noisy proxies of the same latent (public attention), so
        they are rank-BLENDED into one robust indicator (f_attn_spike/level) rather
        than shipped as two correlated features. Point-in-time (trailing windows).
        Empty only if BOTH sources are absent (single source -> that source alone)."""
        idx = self.stock_close.index
        panel = build_combined_attention_panel(
            getattr(self, "wiki_pageviews", None),
            getattr(self, "google_trends", None),
            self.peers, idx,
        )
        if panel.empty:
            return
        before = len(self.feature_panel.columns) - 2
        self.feature_panel = self.feature_panel.merge(
            panel, on=["date", "ticker"], how="left"
        )
        added = len(self.feature_panel.columns) - 2 - before
        cov = panel.drop(columns=["date", "ticker"]).notna().any(axis=1).mean()
        self._log.info("Merged %s combined-attention features (wiki+Google Trends "
                       "rank-blend; row coverage %.1f%%)", added, 100 * cov)

    def build_institutional_features(self):
        """13F institutional-ownership features (breadth, share/value accumulation,
        new-buyer / exiter counts, cluster buying, Herfindahl concentration, net
        put/call option sentiment, ownership %, value/market-cap weight, net $ flow).
        Aggregated across all 13F managers per quarter and stamped point-in-time with
        the 45-day filing lag (a quarter's positions only become features ~45d later)."""
        panel = build_institutional_feature_panel(
            getattr(self, "institutional", None), self.peers, self.stock_close.index,
            shares_out_history=self.fundamentals, stock_close=self.stock_close,
        )
        if panel.empty:
            self._log.warning("No institutional (13F) features built.")
            return
        before = len(self.feature_panel.columns) - 2
        self.feature_panel = self.feature_panel.merge(
            panel, on=["date", "ticker"], how="left"
        )
        added = len(self.feature_panel.columns) - 2 - before
        cov = panel.drop(columns=["date", "ticker"]).notna().any(axis=1).mean()
        self._log.info("Merged %s institutional (13F) features (row coverage %.1f%%)",
                       added, 100 * cov)

    def _load_superinvestors(self) -> dict | None:
        """Read the persisted superinvestors roster JSON (built by
        fetch_superinvestors.build_superinvestors_json). None if it is absent."""
        path = self._context.paths["DATA_STORE"] / SUPERINVESTORS_JSON
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            self._log.warning("Superinvestors roster at %s is unreadable", path)
            return None

    def build_superinvestor_features(self):
        """Elite-manager 13F buy/sell-evolution features (Dataroma superinvestors),
        each weighted by its roster rank, layered ON TOP of the all-filer institutional
        features. Reads the roster JSON; skipped if it has not been built."""
        roster = self._load_superinvestors()
        if not roster:
            self._log.warning("No superinvestors roster JSON -> elite 13F features "
                              "skipped (run fetch_superinvestors.build_superinvestors_json).")
            return
        panel = build_superinvestor_feature_panel(
            getattr(self, "institutional", None), roster, self.peers,
            self.stock_close.index, shares_out_history=self.fundamentals,
            stock_close=self.stock_close,
        )
        if panel.empty:
            self._log.warning("No superinvestor (elite 13F) features built.")
            return
        before = len(self.feature_panel.columns) - 2
        self.feature_panel = self.feature_panel.merge(
            panel, on=["date", "ticker"], how="left"
        )
        added = len(self.feature_panel.columns) - 2 - before
        cov = panel.drop(columns=["date", "ticker"]).notna().any(axis=1).mean()
        n_mgrs = len(roster.get("cik_to_name") or roster.get("managers", []))
        self._log.info("Merged %s superinvestor (elite 13F) features "
                       "(%d managers; row coverage %.1f%%)", added, n_mgrs, 100 * cov)

    def build_insider_features(self):
        """Insider-trading features (trailing-window net open-market buying, buy/sell
        breadth, cluster-buy count, size-scaled net conviction) from Forms 3/4/5.
        Point-in-time on the filing date (Form 4 filed within ~2 business days)."""
        panel = build_insider_feature_panel(
            getattr(self, "insider", None), self.peers, self.stock_close.index,
            shares_out_history=self.fundamentals, stock_close=self.stock_close,
        )
        if panel.empty:
            self._log.warning("No insider-trading features built.")
            return
        before = len(self.feature_panel.columns) - 2
        self.feature_panel = self.feature_panel.merge(
            panel, on=["date", "ticker"], how="left"
        )
        added = len(self.feature_panel.columns) - 2 - before
        cov = panel.drop(columns=["date", "ticker"]).notna().any(axis=1).mean()
        self._log.info("Merged %s insider-trading features (row coverage %.1f%%)",
                       added, 100 * cov)

    def build_short_interest_features(self):
        """Short-selling-pressure features (RegSHO short-volume ratio + its change) plus
        SEC fails-to-deliver (settlement stress). Point-in-time: RegSHO is lagged one
        trading day; FTD is lagged ~2 months (its publication delay) inside the builder."""
        panel = build_short_interest_feature_panel(
            getattr(self, "short_interest", None), self.peers, self.stock_close.index,
            fails_history=getattr(self, "fails_to_deliver", None),
            volume=getattr(self, "stock_volume", None),
        )
        if panel.empty:
            self._log.warning("No short-interest features built.")
            return
        before = len(self.feature_panel.columns) - 2
        self.feature_panel = self.feature_panel.merge(
            panel, on=["date", "ticker"], how="left"
        )
        added = len(self.feature_panel.columns) - 2 - before
        cov = panel.drop(columns=["date", "ticker"]).notna().any(axis=1).mean()
        self._log.info("Merged %s short-interest features (row coverage %.1f%%)",
                       added, 100 * cov)

    def build_earnings_call_features(self):
        """Earnings-call TEXT features. Scores each parsed transcript section with the
        local FinBERT-tone model (cached + incremental in `earnings_call_sentiment`, so
        the GPU pass runs once), then derives the smart per-call KPIs — tone level &
        momentum, the Q&A-vs-scripted candor gap, the hedging (uncertainty) ratio, the
        disclosure-length change, and vocabulary novelty — as peer-relative, point-in-
        time features (a call at date d only affects features on d+1 onward). Skipped
        cleanly when there are no transcripts or the ML stack/model is unavailable."""
        sections = getattr(self, "earnings_call_sections", None)
        if sections is None:
            return
        sentiment = score_earnings_calls(self._context, sections=sections)
        # OpenAI-embedding pass (cached/incremental; no-op without an API key): Q&A coherence
        # cos(question, answer) + quarter-to-quarter embedding drift -> extra f_ec_* features.
        embeddings = embed_earnings_calls(self._context, sections=sections)
        panel = build_earnings_call_feature_panel(
            sentiment, self.peers, self.stock_close.index, sections=sections,
            embeddings=embeddings,
        )
        if panel.empty:
            self._log.warning("No earnings-call features built (unscored transcripts "
                              "or sentiment model unavailable).")
            return
        before = len(self.feature_panel.columns) - 2
        self.feature_panel = self.feature_panel.merge(
            panel, on=["date", "ticker"], how="left"
        )
        added = len(self.feature_panel.columns) - 2 - before
        cov = panel.drop(columns=["date", "ticker"]).notna().any(axis=1).mean()
        self._log.info("Merged %s earnings-call sentiment/text features (row coverage %.1f%%)",
                       added, 100 * cov)

    def build_composite_signals(self):
        """Append thematic COMPOSITE columns (comp_<theme>) to the feature panel:
        each theme averages its (sign-oriented, re-standardized) member features.
        ADDITIVE -- raw features are kept, so no information is lost. Configured
        under build_cube.composites in build_cube.yml."""
        cfg = self._cfg.get("composites", {}) or {}
        if not cfg.get("enabled", False):
            return
        groups = OmegaConf.to_container(cfg.get("groups", {}), resolve=True) or {}
        if not groups:
            self._log.warning("composites.enabled but no groups configured.")
            return
        before = len(self.feature_panel.columns) - 2
        self.feature_panel = build_composite_signals(
            self.feature_panel, groups, method=cfg.get("method", "zscore"),
        )
        added = len(self.feature_panel.columns) - 2 - before
        self._log.info("Built %s composite signals: %s", added,
                       [f"comp_{t}" for t in groups])

    def aggregate_cube(self):
        self.cube = build_cube_dataframe(
            self.feature_panel, self.labels, self.betas, self.peers,
        )
        self._add_categorical_codes()
        self._log.info("Aggregated cube: %s rows, %s horizons, %s tickers",
                       len(self.cube), self.cube["target_horizon"].nunique(),
                       self.cube["ticker"].nunique())

    def _add_categorical_codes(self):
        """Attach GICS sector / industry_group as INTEGER category codes
        (deterministic sorted mapping; unknown -> -1) so LightGBM can make native
        non-linear categorical splits on them. Stored as ints so they flow through
        the numeric panel path unchanged; the linear ensemble member ignores them
        (they are listed under inputs.categoricals, not inputs.columns)."""
        ref = self._context.store.load("sp500_tickers")
        for col in ("sector", "industry_group"):
            if ref.empty or col not in ref.columns:
                self._log.warning("sp500_tickers has no '%s' -> categorical skipped", col)
                continue
            m = dict(zip(ref["ticker"].astype(str), ref[col].astype("string")))
            cats = self.cube["ticker"].astype(str).map(m).astype("category")
            self.cube[col] = cats.cat.codes.astype("int16")     # unknown / NaN -> -1
            self._log.info("Added categorical '%s' (%d categories) to cube",
                           col, cats.cat.categories.size)

    def save_cube(self):
        # full rebuild each run -> replace the table (truncate + fast COPY)
        n = self._context.store.replace("cube", self.cube)
        self._log.info("Saved cube to DB table 'cube' (%s rows)", n)

    # ================================================================== #
    # EXPLODED, memory-light execution for the Airflow `data_aggregation` DAG.
    # Each feature group (and the target) runs STANDALONE — loading only prices +
    # peers + its OWN source table(s) — and persists a compact part to the DB. A
    # final `assemble` step reads the parts and builds the cube. So no single run
    # ever holds all source tables at once, and the parts compute in PARALLEL.
    # The monolithic run() above is unchanged (still used by main.py / tests).
    # ================================================================== #

    # feature group -> (source tables to load onto self, builder method)
    _GROUP_SOURCES: dict[str, tuple[tuple[str, ...], str]] = {
        "price":          ((), "build_features"),
        "fundamental":    (("fundamentals_history", "pension_facts", "notes_num",
                            "earnings_surprises"), "build_fundamental_features"),
        "sector":         (("fundamentals_history",), "build_sector_features"),
        "earnings":       (("earnings_surprises",), "build_earnings_features"),
        "governance":     (("def14a_llm", "fundamentals_history"), "build_governance_features"),
        "employee":       (("employees_history", "fundamentals_history"), "build_employee_features"),
        "dividend":       (("dividends", "fundamentals_history"), "build_dividend_features"),
        "attention":      (("wiki_pageviews", "google_trends"), "build_attention_features"),
        "institutional":  (("institutional_holdings", "fundamentals_history"),
                           "build_institutional_features"),
        "superinvestor":  (("institutional_holdings", "fundamentals_history"),
                           "build_superinvestor_features"),
        "insider":        (("insider_transactions", "fundamentals_history"), "build_insider_features"),
        "short_interest": (("short_interest", "fails_to_deliver"), "build_short_interest_features"),
        "earnings_call":  (("earnings_call_sections",), "build_earnings_call_features"),
    }
    _TABLE_TO_ATTR: dict[str, str] = {
        "fundamentals_history": "fundamentals", "earnings_surprises": "earnings",
        "def14a_llm": "def14a", "employees_history": "employees", "dividends": "dividends",
        "wiki_pageviews": "wiki_pageviews", "google_trends": "google_trends",
        "institutional_holdings": "institutional", "insider_transactions": "insider",
        "short_interest": "short_interest", "fails_to_deliver": "fails_to_deliver",
        "pension_facts": "pension_facts", "notes_num": "notes_num",
        "earnings_call_sections": "earnings_call_sections",
    }

    def _prereqs(self) -> None:
        """Shared minimum for any standalone step: prices (trading calendar + returns) + peers
        (peer baskets + sector returns; read from the SECTOR_PEERS_PATH cache the deduce-peers step
        wrote, so this is cheap and does NOT recompute)."""
        self.load_prices()
        self.normalize_prices()
        self.load_peers()

    def _load_source(self, table: str) -> None:
        setattr(self, self._TABLE_TO_ATTR[table], self._load_or_none(table))
        if table == "def14a_llm" and getattr(self, "def14a", None) is not None:
            self.def14a, _ = impute_def14a(self.def14a)

    def _skeleton(self) -> pd.DataFrame:
        """The (date, ticker) universe grid (cells that have a price) the merge-based feature
        builders left-join onto."""
        s = self.stock_close.reset_index()
        idx = s.columns[0]
        m = (s.melt(id_vars=idx, var_name="ticker", value_name="_v")
             .dropna(subset=["_v"]).rename(columns={idx: "date"}))
        return m[["date", "ticker"]].reset_index(drop=True)

    @staticmethod
    def _norm_date(df: pd.DataFrame) -> pd.DataFrame:
        if df is not None and not df.empty and "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        return df

    def _persist_part(self, name: str, df: pd.DataFrame) -> int:
        """Full-rebuild an AD-HOC intermediate part table (not in the schema registry). Pre-create
        it from the frame's own dtypes via to_sql(head(0)) so store.replace's registry-PK lookup is
        skipped, then let store.replace do the fast COPY load into it."""
        if df is None or df.empty:
            return 0
        df.head(0).to_sql(name, self._context.store.engine, if_exists="replace", index=False)
        return self._context.store.replace(name, df)

    def run_feature_group(self, group: str) -> None:
        """Standalone: build ONE feature group's panel and persist it as `cube_part_<group>`
        (compact: only rows carrying >=1 of that group's features)."""
        if group not in self._GROUP_SOURCES:
            raise ValueError(f"unknown feature group '{group}' "
                             f"(known: {sorted(self._GROUP_SOURCES)})")
        sources, method = self._GROUP_SOURCES[group]
        self._prereqs()
        for t in sources:
            self._load_source(t)
        if method == "build_features":
            self.build_features()                        # SETS self.feature_panel (price panel)
        else:
            self.feature_panel = self._skeleton()        # merge-based builders left-join onto this
            getattr(self, method)()
        fcols = [c for c in self.feature_panel.columns if c not in ("date", "ticker")]
        if not fcols:
            self._log.warning("Feature group '%s' produced no features -> nothing persisted.", group)
            return
        part = self.feature_panel[self.feature_panel[fcols].notna().any(axis=1)]
        n = self._persist_part(f"cube_part_{group}", part)
        self._log.info("Persisted cube_part_%s: %s rows x %s feature cols.", group, n, len(fcols))

    def run_target(self) -> None:
        """Standalone: factor panel + betas + multi-horizon targets -> persist the LONG target &
        beta tables (`cube_part_targets` / `cube_part_betas`) the assemble step joins. Needs no
        feature sources, so it runs beside the feature groups."""
        self._prereqs()
        self._load_source("fundamentals_history")        # style factors + shares-out
        self.macro = self._load_or_none("macro")
        self.build_factor_panel()
        self.estimate_betas()
        self.build_targets()
        nt = self._persist_part("cube_part_targets", _labels_to_long(self.labels))
        nb = self._persist_part("cube_part_betas", _betas_to_long(self.betas))
        self._log.info("Persisted cube_part_targets (%s rows) + cube_part_betas (%s rows).", nt, nb)

    def assemble_cube_from_parts(self) -> None:
        """Final step: read every persisted part, merge features + composites + betas + peers +
        targets into the cube, and save it. Loads NO raw source tables and recomputes NO features
        (mirrors build_cube_dataframe on the persisted long forms)."""
        self._prereqs()                                  # peers dict (for the `peers` column)
        panel = None
        for group in self._GROUP_SOURCES:
            t = f"cube_part_{group}"
            if not self._context.store.exists(t):
                self._log.warning("Feature part '%s' missing -> skipped.", t)
                continue
            p = self._norm_date(self._context.store.load(t))
            if p.empty:
                continue
            panel = p if panel is None else panel.merge(p, on=["date", "ticker"], how="outer")
        if panel is None or panel.empty:
            raise RuntimeError("No feature parts found -> run the features-* steps first.")
        self.feature_panel = panel
        self.build_composite_signals()                   # composites over the merged feature panel

        betas_long = self._norm_date(self._context.store.load("cube_part_betas"))
        targets_long = self._norm_date(self._context.store.load("cube_part_targets"))
        base = self.feature_panel.merge(betas_long, on=["date", "ticker"], how="left")
        base["peers"] = base["ticker"].map(
            lambda t: json.dumps(self.peers.get(t, {}), ensure_ascii=False))
        self.cube = targets_long.merge(base, on=["date", "ticker"], how="inner")
        self.cube = (self.cube.sort_values(["date", "ticker", "target_horizon"])
                     .reset_index(drop=True))
        self._add_categorical_codes()
        self.save_cube()
