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
# SEC EDGAR endpoints (free, no key; require a descriptive User-Agent)         #
# --------------------------------------------------------------------------- #
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
SEC_SUBMISSIONS_PAGE_URL = "https://data.sec.gov/submissions/{name}"
SEC_COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
SEC_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_ARCHIVES_BASE_URL = "https://www.sec.gov/Archives/edgar/data"

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
}