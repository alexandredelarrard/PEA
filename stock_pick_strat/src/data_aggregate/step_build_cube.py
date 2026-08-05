import gc
import json

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from sqlalchemy import text

from src.utils.step import Step
from src.context import Context
from src.utils.universe import load_universe_tickers
from src.constants.constants import SUPERINVESTORS_JSON, CUBE_INCREMENTAL_WARMUP_TRADING_DAYS
from src.data_aggregate.utils import data_utils as du
from src.data_aggregate.utils.betas import estimate_all_betas
from src.data_aggregate.utils.targets import build_targets_multi
from src.data_aggregate.utils.features import build_feature_panel
from src.data_aggregate.utils.fundamental_features import (
    build_fundamental_feature_panel, load_notes_num_scoped, load_pension_facts_scoped,
)
from src.data_aggregate.utils.earnings_features import build_earnings_feature_panel
from src.data_aggregate.utils.governance_features import build_governance_feature_panel
from src.data_aggregate.utils.def14a_impute import drop_implausible_def14a, impute_def14a
from src.data_aggregate.utils.sector_features import build_sector_feature_panel
from src.data_aggregate.utils.employee_features import build_employee_feature_panel
from src.data_aggregate.utils.dividend_features import build_dividend_feature_panel
from src.data_aggregate.utils.attention_features import build_combined_attention_panel
from src.data_aggregate.utils.earnings_call_features import (
    build_earnings_call_feature_panel,
    build_earnings_call_embedding_panel,
    score_earnings_calls,
    sentiment_kpis_streamed,
)
from src.data_aggregate.utils.earnings_call_embeddings import (
    embed_earnings_calls, embedding_kpis_streamed,
)
from src.data_aggregate.utils.institutional_features import build_institutional_feature_panel
from src.data_aggregate.utils.superinvestor_features import (
    build_superinvestor_feature_panel, load_superinvestor_holdings,
)
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

# join keys every feature panel carries; everything else in a panel is a feature column
PANEL_KEYS = ["date", "ticker"]


class FeatureCollisionError(ValueError):
    """Two feature panels emit the same feature name -- merging would split it into
    pandas `_x` / `_y` columns instead of failing loudly."""


class StepBuildCube(Step):

    # STATIC wiring tables, kept at CLASS scope so they can be inspected without
    # constructing a Step (which needs a Context + a live DB). They were instance
    # attributes built inside __init__, which rebuilt the same literals per instance and
    # made `StepBuildCube._GROUP_WARMUP_TRADING_DAYS` unreachable from tests. Read-only:
    # nothing mutates either dict, so `self._X` lookups resolve here unchanged.
    # ---- incremental part maintenance (read latest date -> recompute tail -> append) ---- #
    # PER-PART warm-up (trading days): the LONGEST look-back each part computes ON THE DAILY PRICE
    # GRID, + a safety buffer. `_trim_window` only trims the price-derived frames; SOURCE tables
    # (fundamentals / 13F / insider / def14a / ...) are loaded FULL, so a builder that looks back in
    # FILING/QUARTER space reads all its history regardless of the trim and needs ~no grid warm-up
    # (floored at ~6 months for safety). This is the per-part sanity check — each part reads only as
    # far back as its features actually need, so the light parts stay light.
    # feature group -> (source tables to load onto self, builder method)
    _GROUP_SOURCES: dict[str, tuple[tuple[str, ...], str]] = {
        "price":          ((), "build_features"),
        # pension_facts/notes_num are NOT preloaded: the builder reads only the 2 pension tags of
        # each (load_pension_facts_scoped / load_notes_num_scoped), never the whole facts tables.
        "fundamental":    (("fundamentals_history", "earnings_surprises"),
                            "build_fundamental_features"),
        "sector":         (("fundamentals_history",), "build_sector_features"),
        "earnings":       (("earnings_surprises",), "build_earnings_features"),
        "governance":     (("def14a_llm", "fundamentals_history"), "build_governance_features"),
        # headcount now lives in fundamentals_history's `employees` column (the
        # separate employees_history table is retired) -> one source, not two
        "employee":       (("fundamentals_history",), "build_employee_features"),
        "dividend":       (("dividends", "fundamentals_history"), "build_dividend_features"),
        "attention":      (("wiki_pageviews", "google_trends"), "build_attention_features"),
        "institutional":  (("sec13f_hr", "fundamentals_history"),
                            "build_institutional_features"),
        # only fundamentals_history preloaded: the elite 13F rows are read directly (roster CIKs
        # only) by load_superinvestor_holdings — never the whole ~20M-row sec13f_hr table.
        "superinvestor":  (("fundamentals_history",), "build_superinvestor_features"),
        "insider":        (("insider_transactions", "fundamentals_history"), "build_insider_features"),
        "short_interest": (("short_interest", "fails_to_deliver"), "build_short_interest_features"),
        # earnings calls split into two independent parts (own DAG tasks): the FinBERT/LM sentiment
        # KPIs and the OpenAI-embedding Q&A-coherence/drift KPIs.
        # no preloaded source: scoring streams the sections per ticker + the KPIs stream the cache
        # per ticker itself (loading the full sections table here is what OOM-crashed the task).
        "earnings_call_sentiment": ((), "build_earnings_call_sentiment_features"),
        # no preloaded source: embedding streams the sections per call + the KPI cache per ticker
        # itself (loading the full sections/embeddings tables here is what OOM-crashed the task).
        "earnings_call_embedding": ((), "build_earnings_call_embedding_features"),
    }
    _TABLE_TO_ATTR: dict[str, str] = {
        "fundamentals_history": "fundamentals", "earnings_surprises": "earnings",
        "def14a_llm": "def14a", "dividends": "dividends",
        "wiki_pageviews": "wiki_pageviews", "google_trends": "google_trends",
        "sec13f_hr": "institutional", "insider_transactions": "insider",
        "short_interest": "short_interest", "fails_to_deliver": "fails_to_deliver",
        "pension_facts": "pension_facts", "notes_num": "notes_num",
        "earnings_call_sections": "earnings_call_sections",
    }
    _GROUP_WARMUP_TRADING_DAYS: dict[str, int] = {
        # --- DAILY-grid look-backs (real warm-up required) ---
        "price":          1320,   # seasonal_h*: close.shift(252 * seasonal_years=5) = 1260
        "fundamental":    1320,   # _self_history_z rolling(1260) + Beneish shift(252)
        "dividend":       1320,   # 5y payout growth: shift(5 * 252) = 1260
        "employee":        320,   # YoY headcount / rev-per-employee: shift(252)
        "short_interest":  160,   # short-vol rolling(63) + FTD shift(40) = 103
        "attention":       130,   # spike rolling(63) / level rolling(21)
        # --- FILING/QUARTER-space look-backs over the FULL source (grid warm-up ~0; 6mo floor) ---
        "sector":          130,   # _yearly_lag over fundamentals (as_of order)
        "earnings":        130,   # trailing-4Q rolling over REPORTED quarters
        "governance":      130,   # YoY fiscal change over annual proxies
        "institutional":   130,   # QoQ vs the prior 13F period
        "superinvestor":   130,   # QoQ vs the prior 13F period
        "insider":         130,   # rolling('180D') over the FULL transaction calendar
        "earnings_call_sentiment": 130,   # QoQ over reported quarters (+1d transcript lag)
        "earnings_call_embedding": 130,   # QoQ embedding drift over reported quarters
    }
    # Per-source COLUMN PROJECTION for the exploded feature builds: load ONLY the columns each
    # builder reads (union across the groups that consume the table). Cuts memory on the TALL tables
    # so parallel builds don't OOM the DB/VM (sec13f_hr ~20M rows was the crasher). A
    # table absent here loads in FULL — used for the SMALL tables (fundamentals_history 22k rows,
    # pension/notes/def14a/employees/earnings, all tiny) where projection saves ~nothing, and for
    # earnings_call_sections whose `text` column IS needed by the incremental scoring/embedding pass.
    _SOURCE_COLUMNS: dict[str, list[str]] = {
        # institutional_features + superinvestor_features
        "sec13f_hr": ["cik", "period", "ticker", "shares", "value_usd",
                      "call_value", "put_value", "filing_date"],
        # insider_features (its own required set: ticker/filing_date/transaction_code/value_usd)
        "insider_transactions":   ["ticker", "filing_date", "transaction_code", "value_usd"],
        # short_interest_features (RegSHO short/total volume + reported short interest / ADV)
        "short_interest":         ["date", "ticker", "short_volume", "total_volume",
                                    "short_interest", "avg_daily_volume"],
        "fails_to_deliver":       ["date", "ticker", "fails_quantity"],
        # attention_features
        "wiki_pageviews":         ["date", "ticker", "pageviews"],
        "google_trends":          ["date", "ticker", "search_interest"],
    }

    def __init__(self, context: Context, config: DictConfig):
        super().__init__(context=context, config=config)
        self._cfg = config.build_cube

        # targets/betas: style momentum shift(252) + beta window(63); the forward horizon is added at
        # the call site (targets look FORWARD to mature labels).
        self._TARGET_WARMUP_TRADING_DAYS = 320

        # ticker 
        self.market_ticker = config.build_cube.market_ticker
        self.tickers = load_universe_tickers(self._context)
        self._log.info(f"Ticker universe built for {len(self.tickers)} tickers")

    
    def run(self):

        # load prices 
        self.load_prices()
        self.normalize_prices()

        # peers 
        self.load_peers()

        # load it all
        self.load_fundamentals_and_macro()

        # 
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

    def _merge_panel(self, panel: pd.DataFrame) -> int:
        """Left-merge one feature panel onto the working panel; return how many columns
        it added.

        Raises `FeatureCollisionError` when the panel re-uses a feature name already in
        the working panel. A plain `merge` on the keys silently renames BOTH sides to
        `<name>_x` / `<name>_y`, which is how 20 such columns reached the live cube: the
        fundamental and sector panels each emitted `interest_coverage`,
        `net_debt_to_ebitda`, `gross_profitability`, `cash_conversion_cycle` and
        `sbc_intensity` under DIFFERENT formulas, so which one a model saw depended on
        merge order. Exactly one panel must own each feature name.
        """
        clash = sorted((set(panel.columns) & set(self.feature_panel.columns))
                        - set(PANEL_KEYS))
        if clash:
            raise FeatureCollisionError(
                f"feature name(s) already in the cube panel: {clash}. "
                "Give the feature a single owning panel (or rename it) -- merging would "
                "silently split it into _x / _y columns."
            )
        before = len(self.feature_panel.columns) - len(PANEL_KEYS)
        self.feature_panel = self.feature_panel.merge(panel, on=PANEL_KEYS, how="left")
        return len(self.feature_panel.columns) - len(PANEL_KEYS) - before


    def _attach_panel(self, panel: pd.DataFrame, label: str,
                        empty_msg: str | None = None) -> None:
        """Merge one feature panel and log what it contributed.

        Every `build_*_features` method ended in the same four steps -- bail out when the
        panel is empty, merge through the collision guard, compute row coverage, log it.
        Ten copies of that tail meant ten places to keep in step, and the only thing that
        ever differed was the wording."""
        if panel is None or panel.empty:
            self._log.warning(empty_msg or f"No {label} features built.")
            return
        added = self._merge_panel(panel)
        cov = panel.drop(columns=PANEL_KEYS).notna().any(axis=1).mean()
        self._log.info("Merged %s %s features (row coverage %.1f%%)", added, label, 100 * cov)

    # 1. ------------------------------------------------------------------ #
    def load_prices(self):
        self._log.info("Loading prices from DB table 'prices'")
        self.prices_long = self._context.store.load("prices")

    def normalize_prices(self):

        # pivot table 
        pivot_price = du.prices_long_to_multiindex(self.prices_long)

        # get sub tables from price
        self.close = du.extract_field(pivot_price, "Close")
        self.open = du.extract_field(pivot_price, "Open")
        self.high = du.extract_field(pivot_price, "High") 
        self.low = du.extract_field(pivot_price, "Low") 
        self.volume = du.extract_field(pivot_price, "Volume") 

        # Surface interior calendar holes BEFORE dropping: dates where a quorum of
        # stocks trade but the market_ticker (which defines the calendar) is
        trading_days = du.get_trading_days(self.close, self.market_ticker)

        self.close = self.close.loc[trading_days]
        self.open = self.open.loc[trading_days]
        self.high = self.high.loc[trading_days]
        self.low = self.low.loc[trading_days]

        self.returns = du.daily_returns(self.close)
        self.mkt_ret = self.returns[self.market_ticker]
        self.market_close = self.close[self.market_ticker]

        # Analysis universe = the `sp500_tickers` table (single entry point). Restrict
        # every stock_* frame to it so the cube is built ONLY for the analysed names
        universe = set(self.tickers).intersection(set(self.close.columns))
        if not universe:
            raise Exception("sp500_tickers empty/unseeded -> cube universe is empty")

        self.stock_ret = du._sub(self.returns, universe)
        self.stock_close = du._sub(self.close, universe)
        self.stock_open = du._sub(self.open, universe)
        self.stock_high = du._sub(self.high, universe)
        self.stock_low = du._sub(self.low, universe)
        self.stock_volume = du._sub(self.volume.reindex(self.close.index), universe)
        self._log.info(f"Normalized prices: {self.close.shape[0]} dates, {len(universe)} stocks")

    def load_peers(self):
        self.peers = StepDeducePeers(context=self._context, config=self._config).run()
        self.sector_ret = compute_sector_returns(self.stock_ret, self.peers)
        n = sum(1 for p in self.peers.values() if p)
        self._log.info("Sector returns ready for %s / %s tickers", n, len(self.peers))

    def build_factor_panel(self):
        """Style (price + fundamentals) + macro (changes) -> shared factor panel."""

        # first load fundamental history
        self.fundamentals = self._load_or_none("fundamentals_history")
        if self.fundamentals is None:
            self._log.warning("No fundamentals history -> value/quality factors "
                                "and peer-relative fundamentals will be skipped.")

        style = build_style_factor_returns(
            self.stock_close, self.stock_ret, self.fundamentals, resvol_window=63
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

    def _load_or_none(self, table: str, columns: list[str] | None = None) -> pd.DataFrame | None:
        df = self._context.store.load(table, columns=columns)
        return None if df.empty else df

    def load_fundamentals_and_macro(self):
        """Load fundamentals history and macro; both optional but recommended."""

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
            self.def14a, drop_stats = drop_implausible_def14a(self.def14a)
            self.def14a, imp_stats = impute_def14a(self.def14a)
            imp_stats = {**drop_stats, **imp_stats}
            if imp_stats:
                self._log.info("DEF 14A clean-on-read: deduced %d missing cells across "
                               "%d rules (raw table untouched).", sum(imp_stats.values()),
                               len(imp_stats))

        # scoped reads: the panel uses only 2 tags of each -> never load the whole facts table
        self.pension_facts = load_pension_facts_scoped(self._context)
        if self.pension_facts is None:
            self._log.warning("No pension_facts (Financial Statement Data Sets) -> the "
                              "companyfacts pensionDeficit is used for off-BS leverage.")

        self.notes_num = load_notes_num_scoped(self._context)
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

        self.institutional = self._load_or_none("sec13f_hr")
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

        # scoped tag-filtered reads (DAG path leaves these unset -> read only the pension tags,
        # never the whole notes_num/pension_facts tables); the monolithic path pre-scopes them.
        pension_facts = getattr(self, "pension_facts", None)
        if pension_facts is None:
            pension_facts = load_pension_facts_scoped(self._context)

        # NOTES NUMS
        notes_num = getattr(self, "notes_num", None)
        if notes_num is None:
            notes_num = load_notes_num_scoped(self._context)

        fund_panel = build_fundamental_feature_panel(
            fundamentals_history=self.fundamentals, 
            peer_dict=self.peers, 
            trading_index=self.stock_close.index,
            stock_close=self.stock_close, 
            intrinsic_cfg=self._intrinsic_cfg(),
            hist_window=int(hist.get("window", 1260)),
            hist_min_periods=int(hist.get("min_periods", 252)),
            earnings_history=getattr(self, "earnings", None),   # PEGY projected-growth term
            pension_facts=pension_facts,                        # bulk off-BS pension deficit (2 tags)
            notes_num=notes_num,                                # footnote PBO / plan assets (2 tags)
        )
        if fund_panel.empty:
            self._log.warning("No fundamental features built (missing fundamentals).")
            return
        added = self._merge_panel(fund_panel)
        self._log.info("Merged %s peer-relative fundamental features", added)

    def build_sector_features(self):
        """Sector-specific fundamental KPIs (combined/loss ratio, NIM, efficiency
        ratio, FFO, inventory days, shareholder payout, net-debt/EBITDA, accruals).
        Availability-gated per row (a KPI is null unless its sector reported the
        inputs), then peer-relative + neutralized like the other panels."""
        panel = build_sector_feature_panel(
            self.fundamentals, self.peers, self.stock_close.index,
        )
        self._attach_panel(panel, "sector-KPI", empty_msg="No sector KPI features built (missing fundamentals).")

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
        self._attach_panel(panel, "earnings-expectation", empty_msg="No earnings-expectation features built.")

    def build_governance_features(self):
        """Executive-pay & board features from the LLM-extracted DEF 14A archive
        (CEO pay growth, pay-vs-revenue-growth misalignment, pay ratio, board
        independence/diversity/tenure, say-on-pay). Peer-relative and point-in-time
        from each proxy's `as_of`; coverage accrues as fetch_def14a_llm runs."""
        panel = build_governance_feature_panel(
            self.def14a, self.peers, self.stock_close.index,
            fundamentals_history=self.fundamentals,
        )
        self._attach_panel(panel, "governance/executive-pay", empty_msg="No governance/executive-pay features built " "(def14a_llm empty — accrues as fetch_def14a_llm runs).")

    def build_employee_features(self):
        """Workforce features (revenue per employee, YoY headcount growth) from the
        `employees` column of `fundamentals_history` (10-K body-text headcount, see
        `fundamentals_employees.py`). Genuinely historical and point-in-time
        (stepwise from each filing's `as_of`), so backtestable. Headcount and the
        revenue it is divided by now come from the SAME frame and the same `as_of`,
        which is also why one source is passed twice here."""
        panel = build_employee_feature_panel(
            self.fundamentals, self.peers, self.stock_close.index,
            fundamentals_history=self.fundamentals,
        )
        self._attach_panel(panel, "workforce", empty_msg="No workforce features built.")

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
        self._attach_panel(panel, "dividend", empty_msg="No dividend features built (missing dividend history).")

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
        added = self._merge_panel(panel)
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
        self._attach_panel(panel, "institutional (13F)", empty_msg="No institutional (13F) features built.")

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
        features. Reads the roster JSON; skipped if it has not been built. MEMORY-SAFE:
        reads ONLY the roster managers' 13F rows (a handful of CIKs) via
        `load_superinvestor_holdings`, never the whole ~20M-row sec13f_hr table."""
        roster = self._load_superinvestors()
        if not roster:
            self._log.warning("No superinvestors roster JSON -> elite 13F features "
                              "skipped (run fetch_superinvestors.build_superinvestors_json).")
            return
        holdings = load_superinvestor_holdings(self._context, roster)   # elite subset only
        if holdings is None or holdings.empty:
            self._log.warning("No elite-manager 13F holdings -> superinvestor features skipped.")
            return
        panel = build_superinvestor_feature_panel(
            holdings, roster, self.peers,
            self.stock_close.index, shares_out_history=self.fundamentals,
            stock_close=self.stock_close,
        )
        if panel.empty:
            self._log.warning("No superinvestor (elite 13F) features built.")
            return
        added = self._merge_panel(panel)
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
        self._attach_panel(panel, "insider-trading", empty_msg="No insider-trading features built.")

    def build_short_interest_features(self):
        """Short-selling-pressure features (RegSHO short-volume ratio + its change) plus
        SEC fails-to-deliver (settlement stress). Point-in-time: RegSHO is lagged one
        trading day; FTD is lagged ~2 months (its publication delay) inside the builder."""
        panel = build_short_interest_feature_panel(
            getattr(self, "short_interest", None), self.peers, self.stock_close.index,
            fails_history=getattr(self, "fails_to_deliver", None),
            volume=getattr(self, "stock_volume", None),
        )
        self._attach_panel(panel, "short-interest", empty_msg="No short-interest features built.")

    def _merge_ec_panel(self, panel: pd.DataFrame, label: str) -> None:
        """Left-merge an earnings-call feature panel into the working feature panel + log coverage."""
        if panel is None or panel.empty:
            self._log.warning("No %s features built (no transcripts / model or API key absent).", label)
            return
        added = self._merge_panel(panel)
        cov = panel.drop(columns=["date", "ticker"]).notna().any(axis=1).mean()
        self._log.info("Merged %s %s features (row coverage %.1f%%)", added, label, 100 * cov)

    def build_earnings_call_sentiment_features(self):
        """Earnings-call SENTIMENT features. Scores each transcript section with the local
        FinBERT-tone + Loughran-McDonald pass (cached/incremental in `earnings_call_sentiment`, so
        the GPU pass runs once), then derives the per-call KPIs — tone level & momentum, the
        Q&A-vs-scripted candor gap, the hedging (uncertainty) ratio, disclosure-length change and
        vocabulary novelty — as peer-relative, point-in-time `f_ec_*` features. MEMORY-SAFE: never
        preloads the full sections table — scoring streams the text per ticker, and the KPIs are
        streamed back per ticker (`sentiment_kpis_streamed`)."""
        score_earnings_calls(self._context)                     # lazy, iterative, cache-incremental
        per_call = sentiment_kpis_streamed(self._context)       # per-ticker KPI stream (bounded memory)
        if per_call is None or per_call.empty:
            self._log.warning("No earnings-call sentiment cache -> sentiment features skipped.")
            return
        panel = build_earnings_call_feature_panel(
            None, self.peers, self.stock_close.index, embeddings=None, per_call=per_call)
        self._merge_ec_panel(panel, "earnings-call sentiment")

    def build_earnings_call_embedding_features(self):
        """Earnings-call EMBEDDING features. Runs the OpenAI-embedding pass (cached/incremental;
        no-op without an API key) and derives the Q&A-coherence (cosine of a question vs its answer)
        + quarter-to-quarter narrative-drift `f_ec_*` KPIs. Independent of the sentiment pass (call
        dates come from the embedding rows' own `as_of`), so it runs as its own step without the GPU
        tone model. MEMORY-SAFE: never preloads the full sections/embeddings tables — embedding is
        streamed per call, and the KPIs are streamed back per ticker (`embedding_kpis_streamed`)."""
        embed_earnings_calls(self._context)                  # lazy, iterative, cache-incremental
        ekpi, asof = embedding_kpis_streamed(self._context)  # per-ticker KPI stream (bounded memory)
        if ekpi is None or ekpi.empty:
            self._log.warning("No earnings-call embeddings -> embedding features skipped.")
            return
        panel = build_earnings_call_embedding_panel(
            None, self.peers, self.stock_close.index, sections=asof, ekpi=ekpi)
        self._merge_ec_panel(panel, "earnings-call embedding")

    def build_earnings_call_features(self):
        """Monolithic earnings-call features (main.py / tests): sentiment + embedding, merged into
        the feature panel. The Airflow DAG runs the two as SEPARATE tasks
        (earnings_call_sentiment ∥ earnings_call_embedding)."""
        self.build_earnings_call_sentiment_features()
        self.build_earnings_call_embedding_features()

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
            log=self._log,          # warns about configured members absent from the panel
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

    def _apply_categorical_codes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Attach GICS sector / industry_group as INTEGER category codes to `df`
        (deterministic sorted mapping; unknown -> -1) so LightGBM can make native
        non-linear categorical splits on them. Stored as ints so they flow through
        the numeric panel path unchanged; the linear ensemble member ignores them
        (they are listed under inputs.categoricals, not inputs.columns). By ticker, so
        it is horizon-independent and can be applied ONCE to the pre-horizon-merge base."""
        ref = self._context.store.load("sp500_tickers")
        for col in ("sector", "industry_group"):
            if ref is None or ref.empty or col not in ref.columns:
                self._log.warning("sp500_tickers has no '%s' -> categorical skipped", col)
                continue
            m = dict(zip(ref["ticker"].astype(str), ref[col].astype("string")))
            cats = df["ticker"].astype(str).map(m).astype("category")
            df[col] = cats.cat.codes.astype("int16")            # unknown / NaN -> -1
            self._log.info("Added categorical '%s' (%d categories)", col, cats.cat.categories.size)
        return df

    def _add_categorical_codes(self):
        self.cube = self._apply_categorical_codes(self.cube)

    @staticmethod
    def _downcast_float32(df: pd.DataFrame | None) -> pd.DataFrame | None:
        """Downcast float64 columns to float32 (feature z-scores / ranks / returns need no float64
        precision) -> halves the wide panel + every horizon slice below. Keys / ints / object cols
        untouched."""
        if df is None or df.empty:
            return df
        f64 = df.select_dtypes(include=["float64"]).columns
        if len(f64):
            df[f64] = df[f64].astype("float32")
        return df

    def save_cube(self):
        # full rebuild each run -> replace the table (truncate + fast COPY)
        n = self._context.store.replace("cube", self.cube)
        self._log.info("Saved cube to DB table 'cube' (%s rows)", n)

    def _prereqs(self) -> None:
        """Shared minimum for any standalone step: prices (trading calendar + returns) + peers
        (peer baskets + sector returns; read from the SECTOR_PEERS_PATH cache the deduce-peers step
        wrote, so this is cheap and does NOT recompute)."""
        self.load_prices()
        self.normalize_prices()
        self.load_peers()

    def _load_source(self, table: str) -> None:
        # project to only the columns the consuming builder(s) read (see _SOURCE_COLUMNS); tables
        # absent from the map load in full.
        setattr(self, self._TABLE_TO_ATTR[table],
                self._load_or_none(table, columns=self._SOURCE_COLUMNS.get(table)))
        if table == "def14a_llm" and getattr(self, "def14a", None) is not None:
            self.def14a, _ = drop_implausible_def14a(self.def14a)
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

    def _warmup(self, group: str | None = None) -> int:
        """Warm-up (trading days) to pad before the first new date, sized PER PART to the longest
        daily look-back that part needs (`_GROUP_WARMUP_TRADING_DAYS`). A config
        `build_cube.incremental.warmup_trading_days` overrides every part (escape hatch); an unknown
        group falls back to the global constant."""
        override = self._cfg.get("incremental", {}).get("warmup_trading_days")
        if override is not None:
            return int(override)
        if group is not None and group in self._GROUP_WARMUP_TRADING_DAYS:
            return int(self._GROUP_WARMUP_TRADING_DAYS[group])
        return CUBE_INCREMENTAL_WARMUP_TRADING_DAYS

    def _part_max_date(self, name: str) -> pd.Timestamp | None:
        """Latest `date` already stored in an ad-hoc part table (None if it is absent/empty)."""
        if not self._context.store.exists(name):
            return None
        try:
            with self._context.store.engine.connect() as c:
                v = c.execute(text(f'SELECT MAX(date) FROM "{name}"')).scalar()
        except Exception:                                # noqa: BLE001 (table shape unexpected)
            return None
        return None if v is None else pd.Timestamp(v).normalize()

    def _part_columns(self, name: str) -> list[str] | None:
        """Column names of an existing part table (None if absent) — to detect a feature-set change
        that would break an append (-> caller falls back to a full rebuild)."""
        if not self._context.store.exists(name):
            return None
        try:
            with self._context.store.engine.connect() as c:
                return list(c.execute(text(f'SELECT * FROM "{name}" LIMIT 0')).keys())
        except Exception:                                # noqa: BLE001
            return None

    def _window_start(self, last: pd.Timestamp, n_back: int) -> pd.Timestamp:
        """The date `n_back` trading days BEFORE `last` on the (untrimmed) price calendar."""
        idx = self.stock_close.index
        pos = int(idx.searchsorted(pd.Timestamp(last).normalize()))
        return idx[max(0, pos - n_back)]

    def _trim_window(self, since: pd.Timestamp) -> None:
        """Slice every price-derived frame to `date >= since`, so the builders compute ONLY the
        trailing window (the tail we keep has full warm-up; earlier window rows are discarded)."""
        def cut(x):
            return None if x is None else x.loc[x.index >= since]
        for a in ("close", "open_", "high", "low", "volume", "returns", "mkt_ret", "market_close",
                  "stock_ret", "stock_close", "stock_open", "stock_high", "stock_low",
                  "stock_volume", "sector_ret"):
            if hasattr(self, a):
                setattr(self, a, cut(getattr(self, a)))

    def _append_rows(self, name: str, df: pd.DataFrame, cutoff: pd.Timestamp,
                     inclusive: bool = False) -> int:
        """Idempotently replace the tail of a part: DELETE rows with date > cutoff (>= if
        `inclusive`) then append `df`. Idempotent so a re-run of the same day never duplicates."""
        op = ">=" if inclusive else ">"
        cut = pd.Timestamp(cutoff).strftime("%Y-%m-%d")
        with self._context.store.engine.begin() as c:
            c.execute(text(f'DELETE FROM "{name}" WHERE date {op} :d'), {"d": cut})
        if df is None or df.empty:
            return 0
        df.to_sql(name, self._context.store.engine, if_exists="append", index=False)
        return len(df)

    def cube_parts_status(self) -> dict:
        """Report the latest date + row count of EVERY cube part (+ the cube / predictions), so the
        DAG can push it to XCom and flag drift. `lag_vs_cube_days` = how far a part trails the cube's
        max date; `behind` lists parts more than one build behind (a gap worth attention)."""
        names = ([f"cube_part_{g}" for g in self._GROUP_SOURCES]
                 + ["cube_part_targets", "cube_part_betas",
                    "cube", "predictions", "cube_signal", "predictions_latest"])
        parts: dict[str, dict] = {}
        for name in names:
            info = {"exists": False, "max_date": None, "rows": None}
            if self._context.store.exists(name):
                info["exists"] = True
                mx = self._part_max_date(name)
                info["max_date"] = mx.strftime("%Y-%m-%d") if mx is not None else None
                with self._context.store.engine.connect() as c:
                    info["rows"] = int(c.execute(text(f'SELECT COUNT(*) FROM "{name}"')).scalar())
            parts[name] = info

        cube_max = parts["cube"]["max_date"]
        behind = []
        if cube_max is not None:
            cmax = pd.Timestamp(cube_max)
            for name, info in parts.items():
                if name in ("cube", "predictions", "cube_signal", "predictions_latest"):
                    continue
                if info["exists"] and info["max_date"] is not None:
                    lag = int((cmax - pd.Timestamp(info["max_date"])).days)
                    info["lag_vs_cube_days"] = lag
                    if lag > 4:                          # more than ~one build behind the cube
                        behind.append(name)
                elif not info["exists"]:
                    behind.append(name)
        report = {"as_of": pd.Timestamp.today().normalize().strftime("%Y-%m-%d"),
                  "cube_max_date": cube_max, "ok": not behind, "behind": behind, "parts": parts}
        self._log.info("=== Cube parts status @ %s (cube max=%s, ok=%s) ===",
                       report["as_of"], cube_max, report["ok"])
        for name, info in parts.items():
            self._log.info("  %-26s exists=%-5s max=%-11s rows=%-9s lag_vs_cube=%s",
                           name, info["exists"], info["max_date"] or "-",
                           info["rows"] if info["rows"] is not None else "-",
                           info.get("lag_vs_cube_days", "-"))
        if behind:
            self._log.warning("Cube parts BEHIND / missing (%d): %s", len(behind), ", ".join(behind))
        return report

    def run_feature_group(self, group: str, full: bool = False) -> None:
        """Standalone: build ONE feature group's panel and persist `cube_part_<group>`.

        INCREMENTAL by default: read the part's latest date, recompute only a warm-up-padded
        trailing window (the backward-looking features reproduce the tail exactly once the warm-up
        covers their longest look-back), and APPEND the rows after that date — instead of
        truncating + reloading the whole 15y part. `full=True` (or a missing / column-changed part)
        forces a full rebuild + replace."""
        if group not in self._GROUP_SOURCES:
            raise ValueError(f"unknown feature group '{group}' "
                             f"(known: {sorted(self._GROUP_SOURCES)})")
        sources, method = self._GROUP_SOURCES[group]
        part = f"cube_part_{group}"
        self._prereqs()
        last = None if full else self._part_max_date(part)
        if last is not None:
            self._trim_window(self._window_start(last, self._warmup(group)))
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
        rows = self.feature_panel[self.feature_panel[fcols].notna().any(axis=1)]

        if last is None:                                 # first build (or forced) -> full replace
            n = self._persist_part(part, rows)
            self._log.info("Persisted cube_part_%s (FULL): %s rows x %s feature cols.",
                           group, n, len(fcols))
            return
        
        # incremental append: only rows strictly after the stored max date
        tail = rows[rows["date"] > last]
        existing = self._part_columns(part)
        if existing is not None and set(existing) != set(tail.columns):
            self._log.warning("cube_part_%s feature set changed (%s vs %s) -> full rebuild.",
                              group, len(existing), len(tail.columns))
            return self.run_feature_group(group, full=True)
        n = self._append_rows(part, tail, cutoff=last)
        self._log.info("Appended cube_part_%s (INCREMENTAL): +%s rows after %s (x %s feature cols).",
                       group, n, last.date(), len(fcols))

    def run_target(self, full: bool = False) -> None:
        """Standalone: factor panel + betas + multi-horizon targets -> persist the LONG target &
        beta parts (`cube_part_targets` / `cube_part_betas`) the assemble step joins.

        INCREMENTAL by default. Betas are backward-looking -> append dates after the stored max.
        Targets are FORWARD-looking (a target at date d needs prices to d+horizon), so recent dates
        that were NaN MATURE into values between runs: recompute + overwrite the trailing
        `max_horizon` window (not just new dates) so those matured labels are refreshed."""
        self._prereqs()
        horizons = list(self._cfg.targets.horizons)
        max_h = int(max(horizons)) if horizons else 0
        last_t = None if full else self._part_max_date("cube_part_targets")
        last_b = None if full else self._part_max_date("cube_part_betas")
        if last_t is not None:
            # warm-up for the style-momentum(252)/beta(63) stats + the forward horizon (maturing labels)
            override = self._cfg.get("incremental", {}).get("warmup_trading_days")
            base = int(override) if override is not None else self._TARGET_WARMUP_TRADING_DAYS
            self._trim_window(self._window_start(last_t, base + max_h))
        self._load_source("fundamentals_history")        # style factors + shares-out
        self.macro = self._load_or_none("macro")
        self.build_factor_panel()
        self.estimate_betas()
        self.build_targets()
        targets_long, betas_long = _labels_to_long(self.labels), _betas_to_long(self.betas)

        if last_t is None:                               # first build (or forced) -> full replace
            nt = self._persist_part("cube_part_targets", targets_long)
            nb = self._persist_part("cube_part_betas", betas_long)
            self._log.info("Persisted cube_part_targets (%s) + cube_part_betas (%s) (FULL).", nt, nb)
            return
        # targets: overwrite the trailing max_horizon window so matured labels refresh
        refresh_from = self._window_start(last_t, max_h)
        nt = self._append_rows("cube_part_targets",
                               targets_long[targets_long["date"] >= refresh_from],
                               cutoff=refresh_from, inclusive=True)
        # betas: backward-looking -> just append dates after the stored max
        cutoff_b = last_b if last_b is not None else last_t
        nb = self._append_rows("cube_part_betas", betas_long[betas_long["date"] > cutoff_b],
                               cutoff=cutoff_b)
        self._log.info("Refreshed cube_part_targets (+%s rows from %s) + appended cube_part_betas "
                       "(+%s rows after %s) (INCREMENTAL).", nt, refresh_from.date(), nb,
                       pd.Timestamp(cutoff_b).date())

    def assemble_cube_from_parts(self) -> None:
        """Final step: read every persisted part, merge features + composites + betas + peers +
        targets into the cube, and save it. Loads NO raw source tables and recomputes NO features
        (mirrors build_cube_dataframe on the persisted long forms).

        MEMORY-LIGHT (this step was OOM-killing the DAG): the cube is LONG by target_horizon, so a
        single `targets.merge(base)` broadcasts every feature column across all horizons at once =
        dates x tickers x horizons x ~200 cols held in RAM, then replaced in one shot. Instead we
        (1) float32 every feature part, (2) build the wide `base` (features + composites + betas +
        peers + categoricals) ONCE, and (3) STREAM the write one target_horizon at a time -> peak
        memory drops by the horizon factor (only one horizon slice resident) and there is no giant
        final serialization spike."""
        self._prereqs()                                  # peers dict (for the `peers` column)
        panel = None
        for group in self._GROUP_SOURCES:
            t = f"cube_part_{group}"
            if not self._context.store.exists(t):
                self._log.warning("Feature part '%s' missing -> skipped.", t)
                continue
            p = self._downcast_float32(self._norm_date(self._context.store.load(t)))
            if p is None or p.empty:
                continue
            panel = p if panel is None else panel.merge(p, on=["date", "ticker"], how="outer")
        if panel is None or panel.empty:
            raise RuntimeError("No feature parts found -> run the features-* steps first.")
        self.feature_panel = panel
        self.build_composite_signals()                   # composites over the merged feature panel

        # build the wide, horizon-INDEPENDENT base once (features + betas + peers + categoricals)
        betas_long = self._downcast_float32(self._norm_date(self._context.store.load("cube_part_betas")))
        base = self.feature_panel.merge(betas_long, on=["date", "ticker"], how="left")
        self.feature_panel = panel = betas_long = None   # free the wide intermediates
        # peers JSON PRECOMPUTED per ticker (a few hundred unique strings SHARED across every row)
        # -> the old per-row json.dumps built ~millions of DISTINCT strings, a large object-column
        # memory hog broadcast into each horizon.
        peer_json = {t: json.dumps(self.peers.get(t, {}), ensure_ascii=False)
                     for t in base["ticker"].unique()}
        base["peers"] = base["ticker"].map(peer_json)
        base = self._apply_categorical_codes(base)       # by ticker -> broadcasts to every horizon
        base = base.set_index(["date", "ticker"]).sort_index()   # index once -> fast per-slice join

        targets_long = self._downcast_float32(self._norm_date(self._context.store.load("cube_part_targets")))
        if targets_long is None or targets_long.empty:
            raise RuntimeError("cube_part_targets missing/empty -> run the target step first.")
        horizons = sorted(pd.to_numeric(targets_long["target_horizon"], errors="coerce")
                          .dropna().unique().tolist())
        # STREAM the write in bounded ROW-CHUNKS: only ~chunk rows x cols (+ its COPY buffer) are
        # ever materialised on top of `base`, and each horizon is a PK-disjoint plain COPY-append.
        # The first chunk `replace`s (clears the table + creates the schema); the rest `bulk_seed`
        # (chunked COPY-append, NO slow unchunked upsert — that was the horizon-2 OOM). gc.collect()
        # after every chunk hands the freed arrays + CSV buffer back before the next allocation.
        chunk_rows = 200_000
        total, first = 0, True
        for h in horizons:
            tg = targets_long[targets_long["target_horizon"] == h].set_index(["date", "ticker"])
            for j in range(0, len(tg), chunk_rows):
                chunk = tg.iloc[j:j + chunk_rows].join(base, how="inner").reset_index()
                if chunk.empty:
                    continue
                if first:
                    self._context.store.replace("cube", chunk); first = False
                else:
                    self._context.store.bulk_seed("cube", chunk)
                total += len(chunk)
                chunk = None
                gc.collect()
            self._log.info("Cube horizon %s streamed (running total %s rows).", h, total)
            tg = None
            gc.collect()
        self._log.info("Saved cube to DB table 'cube' (%s rows across %d horizons)",
                       total, len(horizons))
