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