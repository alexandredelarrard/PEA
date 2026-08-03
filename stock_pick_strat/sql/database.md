# DATA_DICTIONARY.md

This document provides a comprehensive overview of the extracted data tables in the codebase, designed for consumption by both human developers and LLM/coding agents.

---

## 1. High-Level Summary Table

| Table Name | Source | Primary Key | Refresh Frequency | Short Description |
| :--- | :--- | :--- | :--- | :--- |
| **`prices`** | Yahoo Finance (`yfinance`) | `(ticker, date)` | Daily | Daily OHLCV price and trading volume data per equity ticker. |
| **`dividends`** | Yahoo Finance (`yfinance`) | `(ticker, date)` | Daily / Event-driven | Historical cash dividend payout history and ex-dividend dates per ticker. |
| **`earnings_surprises`** | Yahoo Finance (`yfinance`) | `(ticker, earnings_date)` | Daily / Post-release | Historical quarterly EPS consensus estimate vs. actual release and surprise percentage. |
| **`fails_to_deliver`** | SEC Market FOIA Data | `(ticker, date)` | Semi-Monthly (Lagged 1–2 months) | Settlement failure counts and dollar values (CNS fails) per ticker date. |
| **`macro`** | FRED (Federal Reserve) | `(date)` | Daily | US treasury yield curve benchmarks, VIX, 10Y breakeven inflation, and BAA credit spreads. |
| **`macro_asset_prices`** | FRED / Yahoo Finance | `(date)` | Daily | Daily cross-asset total return prices and rates (Gold, Energy, Bonds, FX, Cash). |
| **`fundamentals_facts`** | SEC Company Facts (XBRL) | `(ticker, accession_number, field, fiscal_year, fiscal_period, duration_type)` | Daily / Post-filing | Unadjusted atomic XBRL financial facts extracted directly from 10-K/10-Q filings. |
| **`fundamentals_history`** | SEC 10-K/10-Q Parsed | `(ticker, as_of)` | Daily / Post-filing | As-reported point-in-time financial statement metrics, ratios, and parsed headcount. |
| **`pension_facts`** | SEC Financial Statement Sets | `(cik, tag, ddate, qtrs)` | Quarterly / Post-filing | Corporate benefit pension plan details, liabilities, funded status, and service costs per CIK. |
| **`notes_num`** | SEC Financial Statement Footnotes | `(adsh, tag, ddate, qtrs)` | Quarterly / Post-filing | Standardized numeric values extracted from SEC financial statement footnotes. |
| **`notes_text`** | SEC Financial Statement Footnotes | `(adsh, tag, ddate, qtrs)` | Quarterly / Post-filing | Narrative text and escaped HTML disclosures extracted from SEC footnote tables. |
| **`sec13f_hr`** | SEC Form 13F-HR / SEC Bulk | `(cik, period, ticker, cusip)` | Quarterly (45-day lag) | Institutional manager ($100M+ AUM) long equity, option (Call/Put), and debt holdings by CUSIP. |
| **`insider_transactions`** | SEC Insider Data Sets (Forms 3, 4, 5) | `(accession_number, security_type, transaction_sk)` | Quarterly bulk zips / Daily | Open-market and option transactions by C-suite officers, board directors, and >10% owners. |
| **`sec_13d`** | SEC Schedules 13D / 13D/A | `(ticker, accession_number, rp_seq)` | Daily / Filing-based | Activist investor (>5% stake) filing metadata, voting power, group details, and Items 3–6 text. |
| **`sec_13d_transactions`** | SEC Schedules 13D / 13D/A | `(ticker, accession_number, trade_seq)` | Daily / Filing-based | Granular acquisition and disposition trade executions disclosed inside activist 13D disclosures. |
| **`sec_8k`** | SEC Form 8-K filings | `(ticker, accession_number)` | Daily / Filing-based | Material corporate event disclosures (earnings releases, executive changes, M&A, press releases). |
| **`filing_risk_text`** | SEC Forms 10-K / 10-Q | `(ticker, accession_number, section)` | Quarterly / Annually | Parsed narrative body sections (Risk Factors, MD&A, Legal Proceedings) from annual/quarterly filings. |
| **`earnings_call_sections`** | Web Crawl (ROIC / Motley Fool) | `(ticker, quarter, tag)` | Quarterly post-release | Extracted text sections (Prepared Remarks, Q&A) from quarterly earnings conference call transcripts. |
| **`earnings_call_sentiment`** | Internal NLP / LLM Pipeline | `(ticker, quarter, tag)` | Quarterly post-release | Positive, negative, neutral sentiment scores, uncertainty ratios, and word counts per transcript section. |
| **`earning_calls_embedding`** | Internal Vector Embedding Model | `(ticker, quarter, seq)` | Quarterly post-release | Dense vector embeddings for transcript chunks for semantic search and thematic similarity analysis. |

---

## 2. Table Schemas & Functional Details

### Market & Price Data

#### `prices`
- **Primary Key:** `("ticker", "date")`[cite: 1]
- **Source:** Yahoo Finance API (`yfinance`)
- **Refresh Frequency:** Daily
- **Description:** Stores daily equity market bars containing `open`, `high`, `low`, `close`, and `volume`[cite: 1]. Primary market reference table for pricing and volatility signals[cite: 1].

#### `dividends`
- **Primary Key:** `("ticker", "date")`[cite: 1]
- **Source:** Yahoo Finance API (`yfinance`)
- **Refresh Frequency:** Daily / Event-driven
- **Description:** Per-ticker cash dividend amounts indexed by ex-dividend date (`date`)[cite: 1].

#### `fails_to_deliver`
- **Primary Key:** `("ticker", "date")`[cite: 1]
- **Source:** SEC Market Data semi-monthly FOIA releases (`YYYYMMa`/`YYYYMMb`)
- **Refresh Frequency:** Semi-monthly (published with a 1–2 month lag)
- **Description:** Daily counts (`fails_quantity`) and dollar values (`fails_value`) of Continuous Net Settlement (CNS) failed trades[cite: 1]. Key indicator for settlement stress and short-squeeze potential.

#### `earnings_surprises`
- **Primary Key:** `("ticker", "earnings_date")`[cite: 1]
- **Source:** Yahoo Finance (`yfinance`)
- **Refresh Frequency:** Daily / Post-earnings release
- **Description:** Captures historical quarterly earnings announcements, comparing consensus EPS estimates (`eps_estimate`) against realized results (`eps_actual`) and surprise percentages (`surprise_pct`)[cite: 1].

---

### Macroeconomics

#### `macro` & `macro_asset_prices`
- **Primary Key:** `macro` → `("date")`[cite: 1] \| `macro_asset_prices` → `("date")`[cite: 1]
- **Source:** Federal Reserve Economic Data (FRED) & Yahoo Finance
- **Refresh Frequency:** Daily
- **Description:** 
  - `macro`: Contains benchmark Treasury yield curves (3M, 2Y, 10Y, 30Y), yield curve spreads (10Y-2Y, 10Y-3M), VIX index, BAA credit spreads, and 10-year breakeven inflation rates[cite: 1].
  - `macro_asset_prices`: Historical daily prices for multi-asset benchmarks including Gold, Energy, Total Return Bond/Equity indices, and FX rates (USD/EUR)[cite: 1].

---

### Corporate Fundamentals & SEC Disclosures

#### `fundamentals_facts`
- **Primary Key:** `("ticker", "accession_number", "field", "fiscal_year", "fiscal_period", "duration_type")`[cite: 1]
- **Source:** SEC EDGAR Company Facts API (XBRL)
- **Refresh Frequency:** Daily / Incremental as 10-K/10-Q filings drop
- **Description:** Raw, unadjusted XBRL financial tags and values reported in SEC filings[cite: 1]. Tracks original vs. amended accessions and duration windows (`duration_type` e.g., 3M vs 12M)[cite: 1].

#### `fundamentals_history`
- **Primary Key:** `("ticker", "as_of")`[cite: 1]
- **Source:** SEC EDGAR filings (Processed & Standardized)
- **Refresh Frequency:** Daily / Post-filing parse
- **Description:** Point-in-time standardized financial statement metrics (Income Statement, Balance Sheet, Cash Flow), valuation metrics, credit ratios, and employee headcount extracted from 10-K filings[cite: 1].

#### `pension_facts`
- **Primary Key:** `("cik", "tag", "ddate", "qtrs")`[cite: 1]
- **Source:** SEC Financial Statement Data Sets / XBRL filings
- **Refresh Frequency:** Quarterly / Filing-based
- **Description:** Detailed corporate pension and post-retirement benefit plan facts[cite: 1]. Tracks funded statuses, defined benefit obligations, service costs, and discount assumptions[cite: 1].

#### `notes_num` & `notes_text`
- **Primary Key:** Both use `("adsh", "tag", "ddate", "qtrs")`[cite: 1]
- **Source:** SEC Financial Statement Footnotes Data Sets
- **Refresh Frequency:** Quarterly / Filing-based
- **Description:**
  - `notes_num`: Granular numeric data embedded within SEC filing footnote disclosures[cite: 1].
  - `notes_text`: Text strings and escaped HTML tables from financial statement footnotes[cite: 1].

---

### SEC Ownership & Institutional Transactions

#### `sec13f_hr`
- **Primary Key:** `("cik", "period", "ticker", "cusip")`[cite: 1]
- **Source:** SEC Form 13F-HR quarterly filings
- **Refresh Frequency:** Quarterly (45 days after quarter-end)
- **Description:** Complete long equity portfolio disclosures for institutional investment managers managing $\ge \$100\text{M}$ AUM[cite: 1]. Decomposes holdings into long equity (`value_usd`, `shares`), call options (`call_value`), put options (`put_value`), and debt instruments (`debt_value`)[cite: 1].

#### `insider_transactions`
- **Primary Key:** `("accession_number", "security_type", "transaction_sk")`[cite: 1]
- **Source:** SEC Insider Transactions Bulk Datasets (Forms 3, 4, and 5)
- **Refresh Frequency:** Quarterly bulk TSV zips / Daily updates
- **Description:** Executive and 10% shareholder trade logs[cite: 1]. Identifies individual insiders (`owner_cik`, `owner_name`), insider roles (Officer, Director, 10% Owner), transaction codes (e.g., `P` for open market purchase, `S` for sale), shares traded, and share prices[cite: 1].

#### `sec_13d` & `sec_13d_transactions`
- **Primary Key:** 
  - `sec_13d` → `("ticker", "accession_number", "rp_seq")`[cite: 1]
  - `sec_13d_transactions` → `("ticker", "accession_number", "trade_seq")`[cite: 1]
- **Source:** SEC Schedule 13D and 13D/A filings
- **Refresh Frequency:** Daily / Event-driven (within 5 business days of crossing 5%)
- **Description:**
  - `sec_13d`: Captures activist investor stakes (>5% ownership with intent to influence control)[cite: 1]. Contains voting/dispositive power breakdown, group membership, and text from Item 3 (Source of Funds), Item 4 (Purpose of Transaction/Strategy), Item 5 (Interest in Securities), and Item 6 (Contracts/Understandings)[cite: 1].
  - `sec_13d_transactions`: Item 5 trade execution table detailing individual buy/sell trades made during the activist accumulation window[cite: 1].

---

### SEC Filings Text & Unstructured NLP

#### `sec_8k`
- **Primary Key:** `("ticker", "accession_number")`[cite: 1]
- **Source:** SEC Form 8-K filings via EDGAR
- **Refresh Frequency:** Daily / Event-driven
- **Description:** Current report disclosures for major corporate events[cite: 1]. Flags presence of earnings releases (`has_earnings`) or press releases (`has_press_release`) and parses specific item text (`item_text`)[cite: 1].

#### `filing_risk_text`
- **Primary Key:** `("ticker", "accession_number", "section")`[cite: 1]
- **Source:** SEC Form 10-K and 10-Q filing body text
- **Refresh Frequency:** Quarterly / Annually post-filing
- **Description:** Structured section text extracted from annual and quarterly reports[cite: 1]. Standardized sections include Item 1A (Risk Factors), Item 7 (MD&A), and Legal Proceedings[cite: 1].

#### `earnings_call_sections`, `earnings_call_sentiment`, & `earning_calls_embedding`
- **Primary Keys:**
  - `earnings_call_sections`: `("ticker", "quarter", "tag")`[cite: 1]
  - `earnings_call_sentiment`: `("ticker", "quarter", "tag")`[cite: 1]
  - `earning_calls_embedding`: `("ticker", "quarter", "seq")`[cite: 1]
- **Source:** Online web crawlers (ROIC / Motley Fool transcripts) + Internal NLP Processing
- **Refresh Frequency:** Quarterly post-earnings release
- **Description:**
  - `earnings_call_sections`: Unstructured text extracted per transcript section (`tag` e.g., `prepared_remarks`, `q_and_a`)[cite: 1].
  - `earnings_call_sentiment`: Pre-calculated Loughran-McDonald sentiment distribution scores (`sent_pos`, `sent_neg`, `sent_neu`) and uncertainty ratios[cite: 1].
  - `earning_calls_embedding`: Dense vector representations of transcript sentences/chunks for semantic search and AI embedding models[cite: 1].