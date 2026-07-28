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

# Tickers EXCLUDED from the modelling universe for INSUFFICIENT HISTORY (< 4 years of price
# data): recent IPOs / spin-offs that can't support the cube's multi-year look-backs (seasonal,
# self-history z-scores, QoQ drift) or a walk-forward backtest. Filtered out in
# `load_universe_tickers`. Point-in-time snapshot (as of 2026-07, latest price 2026-07-20) --
# REFRESH periodically: a name crosses the 4y threshold and should be removed (GEHC ~= 3.6y is
# closest), and new spin-offs appear. All are genuine separate entities, not dual-class twins.
INSUFFICIENT_HISTORY_TICKERS: frozenset[str] = frozenset({
    "HONA",   # Honeywell Aerospace spin-off (2026)
    "FDXF",   # FedEx Freight spin-off (2026)
    "Q",      # recent listing (2025)
    "SNDK",   # Sandisk / Western Digital spin-off (2025)
    "GEV",    # GE Vernova spin-off (2024)
    "SOLV",   # Solventum / 3M spin-off (2024)
    "VLTO",   # Veralto / Danaher spin-off (2023)
    "KVUE",   # Kenvue / J&J spin-off (2023)
    "GEHC",   # GE HealthCare spin-off (2022, ~3.6y -- closest to the 4y cutoff)
})

# Companies that LEFT the S&P 500 in the last ~15 years (removed 2011+ and not since re-added),
# from the Wikipedia "Selected changes to the S&P 500 components" table. Held here so the universe
# can be made SURVIVORSHIP-BIAS-FREE: a backtest that trains only on today's members overstates
# returns because the losers that were delisted / acquired / demoted (BBBY, SIVB, FRC, SHLD, MON,
# YHOO, TWTR, ...) are silently excluded. Union these into `sp500_tickers` to include them.
# CAVEATS: (1) many are delisted/acquired -> yfinance has PRICES up to delisting but EDGAR
# fundamentals need a CURRENT ticker->CIK match, so those resolve thinly; (2) some symbols were
# later REUSED by a different company (ticker recycling); (3) truly bias-free use wants POINT-IN-TIME
# membership (a name in the universe only WHILE it was in the index) -- a static union approximates.
FORMER_SP500_TICKERS: frozenset[str] = frozenset({
    "AA", "AAL", "AAP", "ABMD", "ACE", "ADS", "ADT", "AET", "AGN", "AIV",
    "AKS", "ALK", "ALTR", "ALXN", "AMG", "AMTM", "AN", "ANDV", "ANF", "ANR",
    "ANSS", "APC", "APOL", "ARG", "ARNC", "ATI", "ATVI", "AVP", "AYE", "AYI",
    "BBBY", "BBWI", "BCR", "BEAM", "BHF", "BHI", "BIG", "BIO", "BMC", "BMS",
    "BRCM", "BTU", "BWA", "BXLT", "CA", "CAG", "CAM", "CBE", "CCE", "CDAY",
    "CE", "CELG", "CEPH", "CERN", "CFN", "CHK", "CLF", "CMA", "CMCSK", "CNX",
    "COL", "COTY", "COV", "CPB", "CPGX", "CPRI", "CPWR", "CSC", "CSRA", "CTLT",
    "CTRA", "CTXS", "CVC", "CVH", "CXO", "CZR", "DAY", "DF", "DFS", "DISCA",
    "DISCK", "DISH", "DNB", "DNR", "DO", "DPS", "DRE", "DTV", "DV", "DWDP",
    "DXC", "EMC", "EMN", "ENDP", "ENPH", "EP", "EPAM", "ESRX", "ESV", "ETFC",
    "ETSY", "EVHC", "FBHS", "FDO", "FHN", "FII", "FL", "FLIR", "FLR", "FLS",
    "FLT", "FMC", "FOSL", "FOX", "FRC", "FRX", "FTI", "FTR", "GAS", "GENZ",
    "GGP", "GHC", "GMCR", "GME", "GNW", "GPS", "GR", "GT", "HAR", "HBI",
    "HCBK", "HES", "HFC", "HNZ", "HOG", "HOLX", "HOT", "HP", "HRB", "HSP",
    "IGT", "ILMN", "INFO", "IPG", "IPGP", "ITT", "JCP", "JDSU", "JEF", "JNPR",
    "JNS", "JOY", "JWN", "K", "KFT", "KMX", "KRFT", "KSS", "KSU", "LEG",
    "LIFE", "LKQ", "LLL", "LLTC", "LM", "LNC", "LO", "LSI", "LUMN", "LVLT",
    "LW", "LXK", "M", "MAC", "MAT", "MBC", "MDP", "MEE", "MFE", "MHK",
    "MHS", "MI", "MJN", "MKTX", "MMI", "MNK", "MOH", "MOLX", "MON", "MRO",
    "MTCH", "MUR", "MWW", "MXIM", "NAVI", "NBL", "NBR", "NE", "NFX", "NKTR",
    "NLSN", "NOV", "NOVL", "NSM", "NVLS", "NWL", "NYX", "OGN", "OI", "PAYC",
    "PBCT", "PBI", "PCL", "PCP", "PCS", "PDCO", "PENN", "PETM", "PGN", "PLL",
    "POM", "POOL", "PRGO", "PVH", "PXD", "QEP", "QRVO", "R", "RAI", "RDC",
    "RE", "RHI", "RHT", "RIG", "RRC", "RRD", "RSH", "RTN", "S", "SAI",
    "SBNY", "SCG", "SE", "SEDG", "SEE", "SHLD", "SIAL", "SIG", "SIVB", "SLE",
    "SLG", "SLM", "SNI", "SOLS", "SPLS", "SRCL", "STI", "STJ", "SUN", "SVU",
    "SWN", "SWY", "TDC", "TE", "TEG", "TFX", "TGNA", "THC", "TIE", "TIF",
    "TLAB", "TRIP", "TSS", "TWC", "TWTR", "TWX", "TYC", "UA", "UAA", "UNM",
    "URBN", "VAR", "VFC", "VIAB", "VNO", "VNT", "WBA", "WCG", "WFM", "WFR",
    "WHR", "WIN", "WLTW", "WPX", "WU", "WYN", "X", "XEC", "XL", "XLNX",
    "XRAY", "XRX", "YHOO", "ZION",
})

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

# 8-K material-event item codes. Pulled STRUCTURED from the EDGAR submissions JSON (`filings.*.items`)
# — no document parsing, ~100% fill for post-2004 8-Ks. Table keyed per filing; raw comma-separated
# `items` stored + a count. The curated high-signal codes (leading distress/governance events) are
# mapped for the downstream feature layer; item structure exists from 2004-08 onward.
SEC_8K_ITEMS_TABLE = "sec_8k_items"
SEC_8K_CACHE_DIR = "sec_8k_cache"        # raw submissions JSON, relative to DATA_STORE
SEC_8K_FORMS = ["8-K", "8-K/A"]
SEC_8K_HIGH_SIGNAL_ITEMS = {
    "1.01": "material_agreement_entered", "1.02": "material_agreement_terminated",
    "1.03": "bankruptcy", "2.05": "restructuring_costs", "2.06": "impairment",
    "3.01": "delisting_or_covenant", "4.01": "auditor_change", "4.02": "non_reliance_restatement",
    "5.02": "exec_or_director_change", "5.03": "bylaw_change", "8.01": "other_events",
}

# SC 13D activist filings (>5% stake WITH intent to influence) + amendments — the event-driven
# catalyst signal. Pulled from the SUBJECT company's EDGAR submissions (SEC indexes 13D/13G under
# the subject CIK, so the shared list_filings works), one row per filing at ~100% event fill.
SEC_13D_TABLE = "sec_13d"
SEC_13D_CACHE_DIR = "sec_13d_cache"      # raw submissions JSON, relative to DATA_STORE
SEC_13D_FORMS = ["SC 13D", "SC 13D/A"]   # activist (13G = passive is deliberately excluded)

# 10-K narrative sections: Item 1A (Risk Factors) + Item 7 (MD&A). Downloaded from the annual 10-K,
# section-carved to raw text, stored for later embedding/drift features (YoY risk-factor additions,
# MD&A tone drift — reusing the notes-embedding machinery). One row per (ticker, accession, section).
FILING_TEXT_TABLE = "filing_risk_text"
FILING_TEXT_CACHE_DIR = "sec_filings_text"   # raw 10-K/10-Q HTML, relative to DATA_STORE
# MD&A lives in DIFFERENT items per form: 10-K Item 7 (annual) and 10-Q Item 2 (quarterly). Both are
# extracted so the MD&A tone/drift signal is QUARTERLY. Risk Factors are taken from the 10-K (Item 1A,
# the substantive annual set); 10-Q Part II Item 1A is usually "no material change" so it is skipped.
FILING_TEXT_FORMS = ["10-K", "10-Q"]
FILING_SECTION_RISK = "risk_factors"         # 10-K Item 1A
FILING_SECTION_MDA = "mda"                   # 10-K Item 7 / 10-Q Item 2
FILING_TEXT_MIN_CHARS = 1500                 # below this a "section" is a TOC/cross-ref stub, not the body

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
EARNINGS_CALL_REQUEST_PAUSE = 2
EARNINGS_CALL_REPORT_GRACE_DAYS = 50
# Map an earnings REPORT date (from earnings_surprises) back into the fiscal quarter it reports:
# a report lands ~4-8 weeks after quarter-end, so shifting the report date back this many days puts
# it inside the reported quarter (Feb report -> prior Q4, late-Apr report -> Q1). Used to demand
# only quarters a ticker has ACTUALLY released, instead of a blanket calendar guess.
EARNINGS_REPORT_TO_QUARTER_LAG_DAYS = 45
# Tickers that hold NO earnings call, so a transcript can never be downloaded -> skipped entirely by
# the earnings-call gap logic (no wasted request, not flagged as "missing"). Berkshire Hathaway is
# the classic case (Buffett publishes a letter + holds the annual meeting, but no quarterly call).
# Extend as other no-call names surface.
NO_EARNINGS_CALL_TICKERS: frozenset[str] = frozenset({"BRK-B", "BRK-A"})

# HuggingFace backbone: clean S&P 500 earnings-call transcripts 2005-2025 (MIT license,
# 33k+ transcripts / 685 companies, full verbatim `content` + speaker-segmented
# `structured_content`). Downloaded ONCE as a single ~1.8 GB parquet, cached under the
# call_transcripts dir; the Motley Fool crawl then only fills the recent gap past its cut.
HF_TRANSCRIPTS_DATASET = "kurry/sp500_earnings_transcripts"
HF_TRANSCRIPTS_PARQUET_URL = (
    "https://huggingface.co/datasets/kurry/sp500_earnings_transcripts/"
    "resolve/main/parquet_files/part-0.parquet")
HF_TRANSCRIPTS_CACHE = "hf_sp500_transcripts.parquet"
# The HF backbone is a ONE-TIME historical load (2005 .. ~2025Q1). Once earnings_call_sections
# already spans that range, re-scanning the 1.8 GB parquet only to find every (ticker, quarter)
# already ingested is pure waste (minutes of "nothing happens"). So ingest_hf_transcripts skips the
# scan when the table's quarter coverage reaches back to EARLY and forward to LATE. Quarters are
# fixed-width "YYYYQN", so a plain string MIN/MAX compares chronologically.
HF_BACKBONE_EARLY_QUARTER = "2005Q4"   # table min quarter must be <= this (deep history is present)
HF_BACKBONE_LATE_QUARTER = "2025Q1"    # table max quarter must be >= this (HF's ~2025 cut is reached)

# Roic AI earnings-call transcripts API — the PRIMARY recent-gap source (after the HF backbone,
# before Motley Fool): a clean JSON API covering ~2y of history on the FREE tier (5 req/min). Auth
# is the `apikey` QUERY param (not a header). `list` returns the available (year, quarter, date) per
# ticker; `transcript` returns {symbol, year, quarter, date, content} for one fiscal quarter.
ROIC_EARNINGS_LIST_URL = "https://api.roic.ai/v2/company/earnings-calls/list/{ticker}"
ROIC_EARNINGS_TRANSCRIPT_URL = "https://api.roic.ai/v2/company/earnings-calls/transcript/{ticker}"
ROIC_REQUEST_PAUSE = 12.5              # free tier = 5 req/min -> >= 12s between calls

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


# --------------------------------------------------------------------------- #
# Incremental cube-part builds (Airflow data_aggregation DAG)                  #
# --------------------------------------------------------------------------- #
# Each cube_part_<group> is rebuilt INCREMENTALLY: read its latest date, recompute only the
# trailing tail and append the new rows (instead of truncating + reloading 15y every run). The
# rolling/peer-relative feature builders need history BEFORE the first new date, so the trailing
# window is padded by this many trading days of warm-up. It MUST exceed the longest look-back any
# feature uses (the fundamental peer-relative history window ~1260 trading days is the binding one),
# else tail values would be computed on a truncated history. Bump it if a longer-window feature is added.
CUBE_INCREMENTAL_WARMUP_TRADING_DAYS = 1400


# --------------------------------------------------------------------------- #
# GICS sectors / industry groups (values as stored in `sp500_tickers`, carried  #
# onto every `fundamentals_history` row by the extractor)                      #
# --------------------------------------------------------------------------- #
GICS_SECTOR_ENERGY = "Energy"
GICS_SECTOR_FINANCIALS = "Financials"
GICS_SECTOR_REAL_ESTATE = "Real Estate"
GICS_SECTOR_UTILITIES = "Utilities"

GICS_GROUP_BANKS = "Banks"
GICS_GROUP_FINANCIAL_SERVICES = "Financial Services"
GICS_GROUP_INSURANCE = "Insurance"
GICS_GROUP_EQUITY_REITS = "Equity Real Estate Investment Trusts (REITs)"
GICS_GROUP_REAL_ESTATE_MGMT = "Real Estate Management & Development"
GICS_GROUP_PHARMA_BIOTECH = "Pharmaceuticals, Biotechnology & Life Sciences"

# The GICS scope each sector-KPI FAMILY is DEFINED for, as (level, accepted values).
# `sector_gates.py` masks every sector KPI with this instead of asking "did the filer
# report tag X", which mis-fired in BOTH directions:
#   * a tag that is not sector-exclusive leaked the KPI into the wrong sector --
#     `InterestIncomeExpenseNet` is used by 59 non-Financials, so bank NIM / ROA /
#     operating margin were computed for industrials & health care; `OperatingLease-
#     LeaseIncome` did the same for FFO on utilities and IT names;
#   * a tag that IS sector-exclusive but rarely tagged starved it -- only 3 of 21
#     Energy names tag `OilAndGasProperty*`, so EBITDAX / DD&A intensity were empty
#     for 86% of the sector.
# `fundamentals_history` carries sector + industry_group only (no sub-industry), so
# `energy` is scoped at sector level: services / refiners simply report no exploration
# expense or oil&gas property, leaving those KPIs NaN as before.
SECTOR_KPI_SCOPE: dict[str, tuple[str, tuple[str, ...]]] = {
    "bank":       ("industry_group", (GICS_GROUP_BANKS,)),
    "insurance":  ("industry_group", (GICS_GROUP_INSURANCE,)),
    "financials": ("sector",         (GICS_SECTOR_FINANCIALS,)),
    "reit":       ("industry_group", (GICS_GROUP_EQUITY_REITS,)),
    "energy":     ("sector",         (GICS_SECTOR_ENERGY,)),
    "utilities":  ("sector",         (GICS_SECTOR_UTILITIES,)),
    "pharma":     ("industry_group", (GICS_GROUP_PHARMA_BIOTECH,)),
}
