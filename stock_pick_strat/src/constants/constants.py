"""
constants.py  (src/constants/constants.py)
-------------------------------------------
Project-wide constants. Import these instead of hardcoding the same date
formats or SEC endpoints across modules, so a change happens in one place.
"""
from __future__ import annotations

# --------------------------------------------------------------------------- #
# Date formats                                                                 #
# --------------------------------------------------------------------------- #
DATE_FORMAT = "%Y-%m-%d"          # ISO day — as_of / filing / query dates
DATE_FORMAT_COMPACT = "%Y%m%d"    # SEC / FINRA daily-file name stamps

# --------------------------------------------------------------------------- #
# Dual-class share redundancy                                                  #
# --------------------------------------------------------------------------- #
# Some companies trade under TWO tickers for the SAME business (e.g. Alphabet
# GOOGL/GOOG, Fox FOXA/FOX, News Corp NWSA/NWS). Their returns correlate ~1.0, so
# in the peer calc the twin would be a stock's own #1 "peer" and would double-count
# that company in everyone else's basket -> flawed peers. We keep the PRIMARY
# (class A / more liquid) and map each redundant SECONDARY (class B/C) to it: the
# secondary is dropped as a peer CANDIDATE and instead inherits its primary's
# basket. Extend as the universe adds dual-class names (e.g. "UA": "UAA").
DUAL_CLASS_SECONDARY_TO_PRIMARY: dict[str, str] = {
    "GOOG": "GOOGL",   # Alphabet   class C -> class A
    "FOX": "FOXA",     # Fox        class B -> class A
    "NWS": "NWSA",     # News Corp  class B -> class A
}

# --------------------------------------------------------------------------- #
# SEC EDGAR endpoints (free, no key; require a descriptive User-Agent)         #
# --------------------------------------------------------------------------- #
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
SEC_SUBMISSIONS_PAGE_URL = "https://data.sec.gov/submissions/{name}"
SEC_COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
SEC_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_ARCHIVES_BASE_URL = "https://www.sec.gov/Archives/edgar/data"
# EDGAR company-name search (atom): the authoritative NAME -> CIK lookup. Filtered to
# 13F-HR filers so a fund name resolves to its institutional-manager CIK. {company}
# must be URL-quoted. Response: one <company-info> block per match with <cik> +
# <conformed-name> (tags are lower-case).
SEC_EDGAR_COMPANY_SEARCH_URL = (
    "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&company={company}"
    "&type=13F-HR&dateb=&owner=include&count=10&output=atom")

# SEC bulk quarterly structured data sets (free TSV zips; {quarter} = e.g. "2024q1").
# insider = Forms 3/4/5 officer/director transactions; finstmt = primary-statement
# XBRL facts (num/sub) incl. the balance-sheet net pension liability.
SEC_INSIDER_URL_TEMPLATE = (
    "https://www.sec.gov/files/structureddata/data/insider-transactions-data-sets/"
    "{quarter}_form345.zip")
SEC_FINSTMT_URL_TEMPLATE = (
    "https://www.sec.gov/files/dera/data/financial-statement-data-sets/{quarter}.zip")
SEC_INSIDER_FIRST_YEAR = 2011      # earliest insider-transactions data set
SEC_FINSTMT_FIRST_YEAR = 2009      # earliest financial-statement data set (2009q2)

# SEC "Financial Statement AND Notes" data sets: like finstmt but ALSO carry the
# NOTES (footnote) facts — numeric (num.tsv, incl. footnote PBO / plan-asset /
# funded-status detail) AND the narrative TEXT blocks (txt.tsv, for embedding /
# sentiment). Files are .tsv (not .txt). {period} is either quarterly "YYYYqQ" OR
# monthly "YYYY_MM": the SEC now consolidates months into a quarter after ~1 year,
# so at any time only the last ~12 months exist as monthly and older data as
# quarterly (the fetcher probes both and skips 404s). ~300-450MB per file.
SEC_FINNOTES_URL_TEMPLATE = (
    "https://www.sec.gov/files/dera/data/financial-statement-notes-data-sets/"
    "{period}_notes.zip")
SEC_FINNOTES_FIRST_YEAR = 2009     # earliest notes data set (2009q1)

# SEC Fails-to-Deliver: semi-monthly settlement-fails files ({period} = "YYYYMMa" for
# settlement dates 1-15, "YYYYMMb" for 16-end). Daily grain (ticker x settlement date).
# The SAME cnsfails{period}.zip files (identical pipe format) live under TWO paths:
#   * current path       -> 2017-06b onward
#   * FOIA "legacy" path  -> 2009-07a .. 2017-06a  (pre-2017-06 history)
# The fetcher picks the path by period (with a cross-fallback for the boundary file).
SEC_FTD_URL_TEMPLATE = "https://www.sec.gov/files/data/fails-deliver-data/cnsfails{period}.zip"
SEC_FTD_LEGACY_URL_TEMPLATE = ("https://www.sec.gov/files/data/"
                               "frequently-requested-foia-document-fails-deliver-data/cnsfails{period}.zip")
SEC_FTD_LEGACY_LAST_PERIOD = "201706a"   # last period on the legacy path (>= 201706b uses the current path)
SEC_FTD_FIRST_YEAR = 2009          # earliest FTD file overall (2009-07, legacy path) -> full 15y coverage

# moves from LEGACY URLto NEW on second half of june 2017
# NEW  href="https://www.sec.gov/files/data/fails-deliver-data/cnsfails202007b.zip"
# LEGACY href= "https://www.sec.gov/files/data/frequently-requested-foia-document-fails-deliver-data/cnsfails201301a.zip"

# --------------------------------------------------------------------------- #
# Google Trends (unofficial API — retail-attention proxy). The explore call    #
# returns widget tokens; the multiline call returns the interest-over-time     #
# series for a token. Priming the home URL first sets the required NID cookie.  #
# --------------------------------------------------------------------------- #
GOOGLE_TRENDS_HOME_URL = "https://trends.google.com/?geo=US"
GOOGLE_TRENDS_EXPLORE_URL = "https://trends.google.com/trends/api/explore"
GOOGLE_TRENDS_MULTILINE_URL = "https://trends.google.com/trends/api/widgetdata/multiline"

# --------------------------------------------------------------------------- #
# Earnings-call transcripts (The Motley Fool — free, full text, no API key)    #
# and local FinBERT-tone sentiment scoring of the parsed sections.             #
# --------------------------------------------------------------------------- #
MOTLEY_FOOL_BASE_URL = "https://www.fool.com"
MOTLEY_FOOL_TRANSCRIPT_INDEX_URL = "https://www.fool.com/earnings-call-transcripts/"
MOTLEY_FOOL_HEADERS = {"User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                                      "Chrome/124.0 Safari/537.36")}
# raw transcript HTML + link-index cache, relative to DATA_STORE (non-tabular artifact)
EARNINGS_CALL_CACHE_DIR = "call_transcripts"
EARNINGS_CALL_SECTIONS_TABLE = "earnings_call_sections"
# Motley Fool politeness: base inter-request pause (seconds) for the quote-page discovery
# AND the transcript HTML download. Deliberately slow — fool.com sits behind Cloudflare and
# throttles (429) after a short burst; the per-host slowdown in polite_http then ratchets
# this up further. Reporting-lag grace (days): the just-ended calendar quarter is not required
# until this many days after quarter-end, so a not-yet-reported quarter never forces a request.
EARNINGS_CALL_REQUEST_PAUSE = 2.5
EARNINGS_CALL_REPORT_GRACE_DAYS = 50

# HuggingFace backbone: clean S&P 500 earnings-call transcripts 2005-2025 (MIT license,
# 33k+ transcripts / 685 companies, full verbatim `content` + speaker-segmented
# `structured_content`). Downloaded ONCE as a single ~1.8 GB parquet, cached under the
# call_transcripts dir; the Motley Fool crawl then only fills the recent gap past its cut.
HF_TRANSCRIPTS_DATASET = "kurry/sp500_earnings_transcripts"
HF_TRANSCRIPTS_PARQUET_URL = (
    "https://huggingface.co/datasets/kurry/sp500_earnings_transcripts/"
    "resolve/main/parquet_files/part-0.parquet")
HF_TRANSCRIPTS_CACHE = "hf_sp500_transcripts.parquet"

# Per-call sentiment / text-metrics cache (one row per ticker / quarter / tag). The
# EXPENSIVE, call-intrinsic scores (FinBERT tone probs + word count + lexicon ratios)
# live here so the GPU pass runs once; the cross-call KPIs (tone delta, Q&A gap,
# length delta, vocabulary novelty) are cheap and derived at cube-build time.
EARNINGS_CALL_SENTIMENT_TABLE = "earnings_call_sentiment"
# OpenAI-embedding cache for earnings calls: one row PER SPEAKER TURN (question / answer /
# prepared), each with its own embedding + raw text + person + tag + exchange_idx (links a
# question to its answer turns) + model/run stamp + as_of (call date). The Q&A-coherence
# (cosine of a question vs its answer turns) + quarter-to-quarter drift cube features are
# DERIVED from these turns at build time. See earnings_call_embeddings.py.
EARNINGS_CALL_EMBEDDING_TABLE = "earning_calls_embedding"
EARNINGS_CALL_EMBED_MODEL = "text-embedding-3-small"     # cheap, 1536-dim; cost-efficient default
# per-turn `tag` values in EARNINGS_CALL_EMBEDDING_TABLE
EARNINGS_CALL_TAG_QUESTION = "question"      # a sell-side analyst turn (asks)
EARNINGS_CALL_TAG_ANSWER = "answer"          # a management turn answering the current question
EARNINGS_CALL_TAG_PREPARED = "prepared"      # a prepared-remarks (scripted) management turn
# Sections we score for tone (the high-signal prose); 'participants'/'full' are skipped
# for KPIs ('full' stays in the sections table as a format-proof fallback).
EARNINGS_CALL_SCORED_TAGS = ("prepared_remarks", "qa")

# FinBERT-tone: finance-domain tone classifier (positive / neutral / negative),
# trained on analyst reports & earnings text. ~440MB, runs locally on GPU (fits 6GB)
# or CPU; free (HuggingFace). Sections longer than the 512-token BERT window are
# chunked and length-weighted (see src/utils/nlp_sentiment.py).
FINBERT_TONE_MODEL = "yiyanghkust/finbert-tone"
FINBERT_MAX_TOKENS = 512

# --------------------------------------------------------------------------- #
# SEC footnote NARRATIVE (`notes_text`) -> risk/compliance features (data_aggregate).
# The raw high-signal TextBlocks are embedded (OpenAI) + NLP-scored into per-filing,
# per-theme features (narrative drift, risk-anchor similarity, tone/litigious density,
# disclosure-length dynamics), then made peer-relative for the cube. See
# src/data_aggregate/utils/notes_features.py.
# --------------------------------------------------------------------------- #
NOTES_TEXT_TABLE = "notes_text"
NOTES_EMBEDDING_TABLE = "notes_embedding"          # cache: 1 row per (ticker, adsh, tag), pooled vector
NOTES_EMBED_MODEL = "text-embedding-3-small"       # cheap, 1536-dim (shared with the earnings-call layer)
# risk/compliance THEME <- the footnote TextBlock tags that carry it (see fetch_financial_notes
# `_NOTES_TEXT_TAGS`). Drift/tone/length are tracked per tag and aggregated to the theme.
NOTES_THEME_TAGS: dict[str, tuple[str, ...]] = {
    "litigation": ("CommitmentsAndContingenciesDisclosureTextBlock",
                   "LegalMattersAndContingenciesTextBlock"),
    "going_concern": ("SubstantialDoubtAboutGoingConcernTextBlock",),
    "revenue_rec": ("RevenueFromContractWithCustomerTextBlock",
                    "RevenueRecognitionPolicyTextBlock", "RevenueRecognitionTextBlock"),
    # `UseOfEstimates` intentionally EXCLUDED: it is mostly a canned boilerplate paragraph and its
    # apparent drift is dominated by filers re-tagging content (e.g. ASC 606) -> not a risk signal.
    "critical_estimates": ("SignificantAccountingPoliciesTextBlock",
                           "OrganizationConsolidationAndPresentationOfFinancialStatements"
                           "DisclosureAndSignificantAccountingPoliciesTextBlock"),
    "concentration": ("ConcentrationRiskDisclosureTextBlock",),
}

# Named RISK / COMPLIANCE archetypes: each note embedding is scored by cosine to these anchor
# phrases (feature B) -> "how close is this disclosure to a known risk pattern", trackable over time.
NOTES_RISK_ANCHORS: dict[str, str] = {
    "litigation_loss": ("It is probable that the company will incur a material adverse loss from "
                        "pending litigation, and it recorded a charge or accrual for legal "
                        "settlements, damages, fines or penalties."),
    "regulatory_action": ("The company is subject to a government or regulatory investigation, "
                          "subpoena, consent decree, or enforcement action alleging violations."),
    "going_concern": ("There is substantial doubt about the company's ability to continue as a "
                      "going concern due to recurring losses and liquidity problems."),
    "covenant_breach": ("The company was not in compliance with its debt covenants and obtained a "
                        "waiver or amendment from its lenders to avoid default."),
    "impairment": ("The company recognized a goodwill or long-lived asset impairment charge because "
                   "expected future cash flows and fair value declined."),
    "control_weakness": ("A material weakness was identified in the company's internal control over "
                         "financial reporting."),
    "restatement": ("The company restated previously issued financial statements to correct a "
                    "material misstatement or accounting error."),
    "customer_concentration": ("A substantial portion of the company's revenue or credit exposure is "
                               "concentrated in a single large customer, counterparty, supplier or region."),
}

# --------------------------------------------------------------------------- #
# Dataroma "superinvestors" — a curated roster of proven long-term investors.  #
# We scrape the roster, resolve each manager to its SEC 13F CIK, rank the top  #
# N by 13F long-equity AUM, and persist a weighted subset JSON so the elite    #
# "smart-money" 13F features can be recomputed reproducibly (fetch_superinvestors #
# -> superinvestor_features). Dataroma exposes NO returns and NO CIK, so the    #
# roster is the curation and CIKs are resolved from cached 13F filer names.     #
# --------------------------------------------------------------------------- #
DATAROMA_HOME_URL = "https://www.dataroma.com/m/home.php"
DATAROMA_HEADERS = {"User-Agent": "Mozilla/5.0 (research; valar_analytics@gmail.com)"}
# roster JSON, relative to DATA_STORE (non-tabular artifact, like sec_bulk_cache)
SUPERINVESTORS_JSON = "superinvestors/superinvestors.json"
SUPERINVESTORS_DEFAULT_TOP_N = 30
# manager weighting scheme within the selected top-N: "rank" (linear decay, top
# gets the most), "aum" (proportional to 13F AUM), or "equal"
SUPERINVESTORS_WEIGHTING = "rank"
# Dataroma manager code -> SEC 13F CIK (zero-padded 10-digit, as in sp500_tickers),
# for the few names the fuzzy filer-name match misses.
SUPERINVESTOR_CIK_OVERRIDES: dict[str, str] = {
    "BRK": "0001067983",   # Berkshire Hathaway  (Warren Buffett)
    "HA" : "0000827280",
    "VAN" : "0000858172",
    "RC" : "0001570775", 
    "DAC": "0000200217",
    "PI": "0001549574",
    "MPF": "0000932223",
    "DAV": "0000200305",
    "T" : "0001002778",
    "oa" : "0000885665"
}

SEC_FORM13F_URL_DICT = {
    "2013q2": "https://www.sec.gov/files/structureddata/data/form-13f-data-sets/2013q2_form13f.zip",
    "2013q3": "https://www.sec.gov/files/structureddata/data/form-13f-data-sets/2013q3_form13f.zip",
    "2013q4": "https://www.sec.gov/files/structureddata/data/form-13f-data-sets/2013q4_form13f.zip",
    "2014q1": "https://www.sec.gov/files/structureddata/data/form-13f-data-sets/2014q1_form13f.zip",
    "2014q2": "https://www.sec.gov/files/structureddata/data/form-13f-data-sets/2014q2_form13f.zip",
    "2014q3": "https://www.sec.gov/files/structureddata/data/form-13f-data-sets/2014q3_form13f.zip",
    "2014q4": "https://www.sec.gov/files/structureddata/data/form-13f-data-sets/2014q4_form13f.zip",
    "2015q1": "https://www.sec.gov/files/structureddata/data/form-13f-data-sets/2015q1_form13f.zip",
    "2015q2": "https://www.sec.gov/files/structureddata/data/form-13f-data-sets/2015q2_form13f.zip",
    "2015q3": "https://www.sec.gov/files/structureddata/data/form-13f-data-sets/2015q3_form13f.zip",
    "2015q4": "https://www.sec.gov/files/structureddata/data/form-13f-data-sets/2015q4_form13f.zip",
    "2016q1": "https://www.sec.gov/files/structureddata/data/form-13f-data-sets/2016q1_form13f.zip",
    "2016q2": "https://www.sec.gov/files/structureddata/data/form-13f-data-sets/2016q2_form13f.zip",
    "2016q3": "https://www.sec.gov/files/structureddata/data/form-13f-data-sets/2016q3_form13f.zip",
    "2016q4": "https://www.sec.gov/files/structureddata/data/form-13f-data-sets/2016q4_form13f.zip",
    "2017q1": "https://www.sec.gov/files/structureddata/data/form-13f-data-sets/2017q1_form13f.zip",
    "2017q2": "https://www.sec.gov/files/structureddata/data/form-13f-data-sets/2017q2_form13f.zip",
    "2017q3": "https://www.sec.gov/files/structureddata/data/form-13f-data-sets/2017q3_form13f.zip",
    "2017q4": "https://www.sec.gov/files/structureddata/data/form-13f-data-sets/2017q4_form13f.zip",
    "2018q1": "https://www.sec.gov/files/structureddata/data/form-13f-data-sets/2018q1_form13f.zip",
    "2018q2": "https://www.sec.gov/files/structureddata/data/form-13f-data-sets/2018q2_form13f.zip",
    "2018q3": "https://www.sec.gov/files/structureddata/data/form-13f-data-sets/2018q3_form13f.zip",
    "2018q4": "https://www.sec.gov/files/structureddata/data/form-13f-data-sets/2018q4_form13f.zip",
    "2019q1": "https://www.sec.gov/files/structureddata/data/form-13f-data-sets/2019q1_form13f.zip",
    "2019q2": "https://www.sec.gov/files/structureddata/data/form-13f-data-sets/2019q2_form13f.zip",
    "2019q3": "https://www.sec.gov/files/structureddata/data/form-13f-data-sets/2019q3_form13f.zip",
    "2019q4": "https://www.sec.gov/files/structureddata/data/form-13f-data-sets/2019q4_form13f.zip",
    "2020q1": "https://www.sec.gov/files/structureddata/data/form-13f-data-sets/2020q1_form13f.zip",
    "2020q2": "https://www.sec.gov/files/structureddata/data/form-13f-data-sets/2020q2_form13f.zip",
    "2020q3": "https://www.sec.gov/files/structureddata/data/form-13f-data-sets/2020q3_form13f.zip",
    "2020q4": "https://www.sec.gov/files/structureddata/data/form-13f-data-sets/2020q4_form13f.zip",
    "2021q1": "https://www.sec.gov/files/structureddata/data/form-13f-data-sets/2021q1_form13f.zip",
    "2021q2": "https://www.sec.gov/files/structureddata/data/form-13f-data-sets/2021q2_form13f.zip",
    "2021q3": "https://www.sec.gov/files/structureddata/data/form-13f-data-sets/2021q3_form13f.zip",
    "2021q4": "https://www.sec.gov/files/structureddata/data/form-13f-data-sets/2021q4_form13f.zip",
    "2022q1": "https://www.sec.gov/files/structureddata/data/form-13f-data-sets/2022q1_form13f.zip",
    "2022q2": "https://www.sec.gov/files/structureddata/data/form-13f-data-sets/2022q2_form13f.zip",
    "2022q3": "https://www.sec.gov/files/structureddata/data/form-13f-data-sets/2022q3_form13f.zip",
    "2022q4": "https://www.sec.gov/files/structureddata/data/form-13f-data-sets/2022q4_form13f.zip",
    "2023q1": "https://www.sec.gov/files/structureddata/data/form-13f-data-sets/2023q1_form13f.zip",
    "2023q2": "https://www.sec.gov/files/structureddata/data/form-13f-data-sets/2023q2_form13f.zip",
    "2023q3": "https://www.sec.gov/files/structureddata/data/form-13f-data-sets/2023q3_form13f.zip",
    "2023q4": "https://www.sec.gov/files/structureddata/data/form-13f-data-sets/2023q4_form13f.zip",
    "2024q1": "https://www.sec.gov/files/structureddata/data/form-13f-data-sets/01mar2024-31may2024_form13f.zip",
    "2024q2": "https://www.sec.gov/files/structureddata/data/form-13f-data-sets/01jun2024-31aug2024_form13f.zip",
    "2024q3": "https://www.sec.gov/files/structureddata/data/form-13f-data-sets/01sep2024-30nov2024_form13f.zip",
    "2024q4": "https://www.sec.gov/files/structureddata/data/form-13f-data-sets/01dec2024-28feb2025_form13f.zip",
    "2025q1": "https://www.sec.gov/files/structureddata/data/form-13f-data-sets/01mar2025-31may2025_form13f.zip",
    "2025q2": "https://www.sec.gov/files/structureddata/data/form-13f-data-sets/01jun2025-31aug2025_form13f.zip",
    "2025q3": "https://www.sec.gov/files/structureddata/data/form-13f-data-sets/01sep2025-30nov2025_form13f.zip",
    "2025q4": "https://www.sec.gov/files/structureddata/data/form-13f-data-sets/01dec2025-28feb2026_form13f.zip",
    "2026q1": "https://www.sec.gov/files/structureddata/data/form-13f-data-sets/01mar2026-31may2026_form13f.zip",
    "2026q2": "https://www.sec.gov/files/structureddata/data/form-13f-data-sets/01jun2026-31aug2026_form13f.zip",
    "2026q3": "https://www.sec.gov/files/structureddata/data/form-13f-data-sets/01sep2026-30nov2026_form13f.zip",
    "2026q4": "https://www.sec.gov/files/structureddata/data/form-13f-data-sets/01dec2026-28feb2027_form13f.zip",
}

# --------------------------------------------------------------------------- #
# Long-history multi-asset ALLOCATION series (FRED, free, deep history)        #
# --------------------------------------------------------------------------- #
# A separate, deeper-history pull than the `macro` feature table: total-return /
# level series for the risk-parity + trend asset-allocation sleeve, back to ~1995
# (scoped by data_extract.macro_asset_years_history). Written to MACRO_ASSET_PRICES_TABLE.
#
# Equity: FRED's S&P 500 (`SP500`) is license-truncated to ~10y daily, so the
# long equity leg uses the Wilshire 5000 Total Market Full Cap index (near-identical
# US equity beta, daily since ~1971). 10Y bond is stored as a TOTAL-RETURN index
# reconstructed from the constant-maturity yield (carry + duration*Δyield) — the raw
# yield is kept alongside for transparency. Cash = 3M T-bill secondary-market rate.
MACRO_ASSET_PRICES_TABLE = "macro_asset_prices"
# HYBRID source (verified July 2026): FRED's API no longer serves a broad daily S&P
# (SP500 is license-truncated to ~10y) or ANY gold series (the London fixes were
# removed ~2025), so the RATES/CASH/FX legs come from FRED (its strong long history)
# and the EQUITY + GOLD legs from yfinance (the pipeline's existing market source).
# FRED series id -> column name in MACRO_ASSET_PRICES_TABLE:
MACRO_ASSET_FRED_SERIES = {
    "DGS10": "yield_10y",           # 10Y constant-maturity Treasury yield -> bond_10y_tr
    "DTB3": "cash_rate",            # 3-month T-bill secondary market rate (cash leg)
    "DEXUSEU": "fx_usdeur",         # USD per EUR (FX leg; NaN before the euro, 1999)
    "VIXCLS": "vix",                # CBOE VIX (since 1990) — REGIME SIGNAL, not an asset
}
# columns stored for their SIGNAL value only (regime detection), never allocated to as an
# asset — asset_returns_from_macro whitelists the tradable legs, so these are excluded there.
MACRO_ASSET_SIGNAL_COLUMNS = ("vix",)
# yfinance symbol -> column name (daily, auto-adjusted so each is a total-return proxy):
# SPY = S&P 500 total-return (since 1993); GC=F = COMEX gold front future (since 2000);
# XLE = Energy Select Sector SPDR (since 1998) — the "commodity via ENERGY EQUITIES" sleeve
# (no futures): the rate/inflation-shock diversifier that was +~60% in the 2022 selloff.
MACRO_ASSET_YF_SERIES = {
    "SPY": "equity_tr",
    "GC=F": "gold",
    "XLE": "energy",
}
MACRO_ASSET_GOLD_COLUMN = "gold"
# reconstructed 10Y total-return index column + its maturity assumption
MACRO_ASSET_BOND_TR_COLUMN = "bond_10y_tr"
MACRO_ASSET_BOND_MATURITY_YEARS = 10
# CORE daily level series used to judge table freshness (lag ~1 business day)
MACRO_ASSET_CORE_LEVEL_COLUMNS = ("equity_tr", "yield_10y", "cash_rate")

# --------------------------------------------------------------------------- #
# Multi-asset trend (time-series-momentum) sleeve — StepTrendAssetClass output #
# --------------------------------------------------------------------------- #
# Daily NET return of the directional cross-asset trend book (one row per date, no ticker),
# consumed by StepBacktest as a diversifying sleeve to blend with the equity L/S alpha + SPY.
TREND_ASSET_RETURNS_TABLE = "trend_asset_returns"
# model artifact (params + vol-target calibration) under paths["MODELS_DIR"]
TREND_ASSET_MODEL_FILE = "trend_asset_model.json"

# --------------------------------------------------------------------------- #
# Data-freshness / gap check (StepCheckFreshness) — runs at the tail of the    #
# nightly extraction DAG, before triggering aggregation, so prediction never   #
# runs on stale inputs. Each source maps to (table, observation-date column,   #
# cadence); the cadence sets how old the latest observed date may be before it #
# is flagged (RED). Thresholds are generous by design — they fold in weekends/ #
# holidays (daily) and the normal reporting/filing lag (quarterly/yearly), so  #
# only a genuine GAP beyond one extra cycle trips the warning. Where a filing  #
# date exists (SEC notes/pension/insider) it is used instead of the period-end #
# so freshness reflects WHEN data was published, not the period it covers.     #
# --------------------------------------------------------------------------- #
DATA_FRESHNESS_SOURCES: dict[str, tuple[str, str, str]] = {
    # label:                 (table,                  date_col,           cadence)
    "prices":                ("prices",               "date",             "daily"),
    "macro":                 ("macro",                "date",             "daily"),
    "macro_asset_prices":    ("macro_asset_prices",   "date",             "daily"),
    "short_interest":        ("short_interest",       "date",             "daily"),
    "wiki_pageviews":        ("wiki_pageviews",       "date",             "daily"),
    "google_trends":         ("google_trends",        "date",             "weekly"),
    "fails_to_deliver":      ("fails_to_deliver",     "date",             "biweekly"),
    "notes_num":             ("notes_num",            "filed",            "biweekly"),
    "notes_text":            ("notes_text",           "filed",            "biweekly"),
    "insider_transactions":  ("insider_transactions", "filing_date",      "quarterly"),
    "fundamentals_history":  ("fundamentals_history", "as_of",            "quarterly"),
    "earnings_surprises":    ("earnings_surprises",   "earnings_date",    "quarterly"),
    "institutional_holdings":("institutional_holdings","period",          "quarterly"),
    "pension_facts":         ("pension_facts",        "filed",            "quarterly"),
    "earnings_call_sections":("earnings_call_sections","as_of",           "quarterly"),
    "employees_history":     ("employees_history",    "as_of",            "yearly"),
    "def14a_llm":            ("def14a_llm",           "as_of",            "yearly"),
}
# how many days old the latest observed date may be, per cadence, before RED
DATA_FRESHNESS_MAX_AGE_DAYS: dict[str, int] = {
    "daily": 4, "weekly": 10, "biweekly": 20, "monthly": 45,
    "quarterly": 140, "yearly": 460,
}
# cadence tiers in the order the freshness report walks them (daily -> yearly)
DATA_FRESHNESS_CADENCE_ORDER: tuple[str, ...] = (
    "daily", "weekly", "biweekly", "monthly", "quarterly", "yearly")
# To report WHICH tickers got a new fundamentals filing (new earnings) since the last run, the
# freshness gate snapshots the per-ticker latest fundamentals date to this JSON (under DATA_STORE,
# so it persists on the host ./data mount) and diffs it next run. Keyed off the "fundamentals_history"
# source in DATA_FRESHNESS_SOURCES above.
FRESHNESS_SNAPSHOT_DIR = "freshness"
FUNDAMENTALS_SNAPSHOT_FILE = "fundamentals_latest_by_ticker.json"
FRESHNESS_FUNDAMENTALS_SOURCE = "fundamentals_history"
