# Data sources

Scope: the external sources, their keys, their real limits, and the quirks that have already cost
debugging time. All free or freemium **except Sharadar**, which is a paid subscription. For the
tables they land in, see [data_schema.md](data_schema.md); for current coverage,
[database.md](database.md).

## The sources

| Domain | Source | Key? | Lands in | Fetcher |
|---|---|---|---|---|
| Prices (OHLCV) | yfinance | no | `prices` | `prices/fetch_prices.py` |
| Dividends (ex-dates) | yfinance | no | `dividends` | `prices/fetch_dividends.py` |
| S&P 500 constituents | Wikipedia | no | `sp500_tickers` | `prices/fetch_tickers.py` |
| Benchmark / commodity / energy | yfinance (`SPY`, `^VIX`, `CL=F`, `GC=F`, `XLE`) | no | `prices_macro` | `prices/fetch_macro.py` (close only, via `download_ohlcv`) |
| Rates / credit / breakeven / FX + derived legs (~1995→) | FRED (incl. `DEXUSEU` for FX) | `FRED_API_KEY` | `prices_macro` | `prices/fetch_macro.py` (same fetcher — one table, one source per series) |
| Fundamentals history | SEC EDGAR per-filing XBRL (edgartools) | `SEC_USER_AGENT` | `fundamentals_facts` → `fundamentals_history_sec` | `fundamentals/fetch_fundamentals_edgar.py` |
| Fundamentals (vendor, **paid**) | **Sharadar SF1** via the DIRECT API `api.sharadar.com` | `SHARADAR_API_KEY` | `fundamentals_sharadar` → `fundamentals_history` | `fundamentals_sharadar/fetch_sharadar.py` |
| Entity dimension / corporate actions / index membership | Sharadar `tickers`, `actions`, `sp500` | `SHARADAR_API_KEY` | `sharadar_tickers`, `sharadar_actions`, `sharadar_sp500` | `fundamentals_sharadar/fetch_sharadar.py` |
| Employee headcount | SEC 10-K **body text** | `SEC_USER_AGENT` | `fundamentals_history_sec.employees` | `fundamentals/fundamentals_employees.py` |
| Earnings surprises / forward P/E | yfinance | no | `earnings_surprises` | `fundamentals/fetch_earnings_surprises.py` |
| Pension facts | SEC Financial Statement Data Sets (zip) | `SEC_USER_AGENT` | `pension_facts` | `fundamentals/fetch_financial_statements.py` |
| Footnote numbers + narrative | SEC Financial Statement **and Notes** sets (zip, `.tsv`) | `SEC_USER_AGENT` | `notes_num`, `notes_text` | `fundamentals/fetch_financial_notes.py` |
| Institutional holdings | SEC Form 13F bulk sets | `SEC_USER_AGENT`, `OPENFIGI_API_KEY` (optional) | `sec13f_hr`, `cusip_ticker_map` | `prices/fetch_13f.py`, `fetch_cusip_map.py` |
| Elite-manager subset | Dataroma roster → CIK filter over 13F | no | `data/superinvestors/superinvestors.json` | `prices/fetch_superinvestors.py` |
| Insider trades | SEC Insider Data Sets (Forms 3/4/5, quarterly zips) | `SEC_USER_AGENT` | `insider_transactions` | `prices/fetch_insider_transactions.py` |
| Governance / comp / ownership | SEC **DEF 14A** via OpenAI structured output | `OPENAI_API_KEY`, `SEC_USER_AGENT` | `def14a_llm` | `structure/fetch_def14a_llm.py` |
| Governance (deterministic) | SEC DEF 14A via edgartools `ProxyStatement` | `SEC_USER_AGENT` | `sec_def14a` + 4 children | `structure/fetch_def14a_edgar.py` |
| Corporate events | SEC Form 8-K | `SEC_USER_AGENT` | `sec_8k` | `structure/fetch_8k_edgar.py` |
| Activist stakes | SEC Schedule 13D / 13D-A | `SEC_USER_AGENT` | `sec_13d`, `sec_13d_transactions` | `structure/fetch_13d_edgar.py` |
| Filing narrative | SEC 10-K Item 1A / Item 7, 10-Q Item 2 | `SEC_USER_AGENT` | `sec_filing_text` | `structure/fetch_filing_text.py` |
| Short volume | FINRA RegSHO daily files | no | `short_interest` | `prices/fetch_short_interest.py` |
| Settlement fails | SEC Market FOIA (semi-monthly zips) | no | `sec_fails_to_deliver` | `prices/fetch_fails_to_deliver.py` |
| Retail attention | Wikipedia pageviews API | no | `wiki_pageviews` | `behavioral/fetch_wiki_pageviews.py` |
| Retail attention | Google Trends | no | `google_trends` | `behavioral/fetch_google_trends.py` |
| Earnings-call transcripts (deep history) | HuggingFace `kurry/sp500_earnings_transcripts` | no | `earnings_call_sections` | `behavioral/fetch_hf_transcripts.py` |
| Earnings-call transcripts (recent gaps) | Roic AI → Motley Fool quote pages | no | `earnings_call_sections` | `behavioral/fetch_roic_transcripts.py`, `utils_missing_quarters.py` |
| Call tone | local **FinBERT-tone** (torch, GPU) + LM uncertainty lexicon | no | `earnings_call_sentiment` | `utils/nlp_sentiment.py` |
| Call / notes / business embeddings | OpenAI `text-embedding-3-small` | `OPENAI_API_KEY` | `earning_calls_embedding`, `notes_embedding`, `ticker_embeddings` | `utils/openai_embeddings.py` |

Environment variables live in a git-ignored `.env` at the repo root (see `.env.example`), loaded by
`Context._load_env` via `find_dotenv(usecwd=True)`. `SEC_USER_AGENT` must be a real
`"Name email@domain"` — SEC EDGAR rejects requests without it.

## Shared plumbing

- [src/utils/polite_http.py](../src/utils/polite_http.py) — `curl_cffi` **TLS impersonation** with
  rotation + rate limiting. This is what gets past the anti-bot 429s that plain `requests` collects
  (notably Google Trends and Motley Fool).
- [src/utils/ssl_setup.py](../src/utils/ssl_setup.py)`::configure_corporate_ca()` — builds the
  combined corporate CA bundle. **Required behind the corporate TLS proxy**, and it must run before
  any module imports `curl_cffi` (which freezes its bundle at import). `tests/conftest.py` calls it
  at import time for exactly this reason; without it, live tests fail as
  `CERTIFICATE_VERIFY_FAILED`, which looks like a source-coverage failure.
- [data_extract/utils/common/](../src/data_extract/utils/common/) — `bulk_cache.py` (zip caching &
  self-healing), `sec_utils.py` (rate limiting ~10 req/s, state), `form_registry.py`
  (`FORM_REGISTRY`), `rate_limit.py`, `parallel_fetch.py`, `run_manifest.py`, `llm_extractor.py`.
- Airflow pools cap the load: `sec_bulk` 2, `sec_api` 2, `scrape` 2, `aggregate` 3.

## Free-source realities you must design around

**SEC XBRL is the fundamentals backbone** — genuine point-in-time history keyed on filing date,
~10-15 years deep. Everything else about fundamentals is downstream of it.

**Forward P/E and 13F accrue point-in-time going forward only.** yfinance and 13F have no clean
back-history for those, so the features build up over successive runs. Do not expect 15 years.

**13F is a long-only quarterly snapshot with a 45-day filing lag**, split into stock / call / put /
debt. Institutional "moves" come from **quarter-over-quarter share deltas, not value deltas** —
value moves with price and would encode the return you are trying to predict.

**Sector-specific line items** (bank NII, insurance premiums/claims, REIT rental income, energy
DD&A) are extracted and turned into sector KPIs (NIM, combined ratio, FFO, …), **gated by
availability** and normalized at the GICS industry-group level. See
`utils/common/sector_gates.py` and `constants.SECTOR_KPI_SCOPE`.

## Known traps, by source

### Sharadar SF1 (the paid vendor layer)

**The channel.** Direct API only — `https://api.sharadar.com/v1.0`. **Never** `data.nasdaq.com`,
and never the `nasdaqdatalink` / `quandl` libraries with this key: those speak a different channel
that names the filing-date column **`datekey`** and ships no `fiscalperiod`. Our filing-date column
is **`date`**, and it sits inside the primary key, so the two channels are not interchangeable.

**Request shape — three defaults that silently truncate:**
- **`from` defaults to "1 year ago"**, `limit` to 10000, `sort` to `date.desc`. Always pass
  `date.gte` and `sort` explicitly; an omitted bound quietly returns one year of history.
- **`fields=` drops an unavailable field with no warning** — a typo yields a missing column, not an
  error. `client._validate_header` asserts the response header against the stored contract both
  ways for exactly this reason.
- `limit` **above** 10000 *is* honoured on `/data/fundamentals` (50000 returned all 22,530 rows of
  2024 ARQ), and `offset` paging works with no duplicate keys. Paging is belt-and-braces.
- The `tickers` endpoint has a **filter** called `table`, which collides with the wrapper's own
  first argument — hence `sharadar_get`'s positional-only `/`.

**Entitlement — measured 2026-08-26, after the upgrade to the paid tier:**
- **The whole SF1 universe is entitled.** No ticker returns 403; a ticker-less query spans 5,780
  distinct tickers in 2024 alone, and arbitrary micro-caps return rows.
- **History reaches filing date 1993-12-22** (earliest `calendardate` 1993-03-31). Megacaps start
  1994: AAPL 1994-01-26, MSFT 1994-02-14, GE 1994-03-11, JPM 1994-03-25.
- `configs.yml`'s `sharadar_years_history: 31` therefore sets a cold-start floor of ~1995-08 and
  leaves ~2,700 ARQ rows across 539 tickers unfetched **by choice**. Raise the knob to ~34 to take
  the full depth; it costs no extra requests, only larger responses.
- **`bulk/fundamentals` still returns 404** — the bulk download is not part of this subscription.
- **403 means NOT ENTITLED, not throttled.** It should no longer occur, but the classification path
  is kept: `polite_http.http_get` retries 403 four times with exponential backoff, so a roster loop
  would burn minutes per denied ticker. Classify off a single `get_once`, never the retrying path.

**Units and conventions — the ones that produce plausible-looking wrong numbers:**
- **Only 8 columns are USD-converted.** Everything else is the filer's reporting currency while
  `marketcap`/`price` are always USD, so a non-USD row mixes units *within itself*. We assert USD
  off `sharadar_tickers.currency` and REFUSE to write a non-USD filer (D20).
- Money columns are **actual units** in SF1 but **USD millions** in the `daily` table — a 10⁶ factor
  between two tables of the same subscription.
- Ratio columns are **decimal fractions**, not percentages, despite the 2019 dictionary typing them
  `%`. `evebit` is `bigint` and comes back integer-truncated.
- **`de` is liabilities/equity**, not debt/equity, despite the name.
- `capex` and the `ncf*` legs are stored **negative**; the repo's `capex` is `non_negative`, so the
  map flips the sign — and NULLs the ~1% of rows (mostly GS) that are positive rather than writing a
  negative into a column that cannot hold one.
- **The whole share-count and per-share block is retroactively SPLIT-ADJUSTED**, and `sharefactor`
  is 1.0 on those rows so it does not flag it. `build_ttm.deadjust_splits` corrects it *after* the
  four-quarter aggregation — de-adjusting the quarters first mixes two bases inside one window.
- **`lastupdated` is a per-TICKER reprocessing stamp, not a per-row change stamp**, so it is useless
  as an incremental watermark. A Sharadar restatement is picked up by `-F/--full`, not by a resume.

**Grain:**
- **Only the `AR*` dimensions are point-in-time.** `MR*` rows mutate in place and are not stored.
- **Q4 is CONSTRUCTED** as `ARY - Σ(Q1..Q3)`, so `ΣARQ == ARY` is a tautology (measured `+0.000%`)
  and can never be a quality check. It can still produce absurd LEVELS — that is what
  `gate_implausible_quarters` measures instead.
- **Quarterly dimensions are US-domestic-only.** ADR (form 20) and Canadian (form 40) filers have no
  ARQ/MRQ at all — relevant the moment the universe widens past the S&P 500.
- SF1 covers the **primary share class only**.
- **41 fields are zero-filled**, and a `0` may mean "not applicable" (a bank has no inventory) or
  "absent, and we wrote a zero" (`intexp = 0` for JPM is provably false). The verdict is per-field
  and human-approved in `configs/sharadar/sharadar_zero_rules.json`.
- ⚠ `contraticker` is the literal string **`"N/A"`**, not NULL, and it is a PK member of
  `sharadar_actions` — so the side tables must be read with `keep_default_na=False` or a PK value
  becomes NULL.

### Fundamentals / XBRL

- **`companyfacts` drops dimensioned facts.** The per-filing walk exists because the aggregate
  companyfacts endpoint silently omits them.
- **Never trust a tag name.** Measure coverage on real cached filings before believing a tag is
  populated.
- **Never compare two independently forward-filled columns.** They ffill from different filing dates,
  so a derived ratio mixes two as-of dates.
- **Multi-class share counts.** A filer with more than one class of common stock tags **no
  undimensioned share count anywhere** — every fact sits on `StatementClassOfStockAxis`, the classes
  disagree, and the dimension rules refuse them all, so `shares_outstanding` (hence market cap) came
  out NULL for the whole multi-class cohort. `build_tag_frames` therefore never admits a
  class-dimensioned fact as the company total, and rebuilds the total by summing the **cover-page**
  classes only (`dei:EntityCommonStockSharesOutstanding`), which the SEC cover page requires to be an
  exhaustive per-class enumeration. **The balance-sheet parenthetical is never summed** — measured
  incomplete or overlapping on 6 of 36 filers. Where classes do not convert 1:1, the sum is put into
  the traded class's units using factors the filers tag themselves (`CommonStockConversionRatio`,
  `EconomicEquivalentPercentage`, `SharesOutstandingAsConvertedBasis`). All **fill-only** — absent
  the hook, the plain sum stands.
- **Consolidated basis.** Market cap, `netIncome` (`ProfitLoss` first) and `stockholdersEquity`
  (incl-NCI first) are all on the **whole consolidated group**, matching `totalRevenue`/`totalAssets`
  (which have no parent-only US-GAAP concept) and matching what every vendor publishes. Previously
  income was the parent's slice while revenue was the group's, so a high-NCI filer's ratios were
  built from two different companies (IBKR's parent takes 22.6% of income; `sales_yield` was ~3.8×
  too high). For an **Up-C**, the share count is grossed up by the tagged parent-ownership %, but
  only when the class sum demonstrably does not already cover the non-controlling holders — the
  filing decides. *Known artifact*: an Up-C's NCI income escapes the parent's corporate-tax layer, so
  a consolidated P/E reads cheaper than a buyer of the traded class gets (IBKR 33.6× vs 39.4×). This
  is shared with every vendor; the alternative needs a parent-level revenue that does not exist.
- **Tag ledger.** [src/utils/fundamentals_tag_ledger.py](../src/utils/fundamentals_tag_ledger.py)
  collapses `fundamentals_facts` into `source_tag` eras and flags boundaries where the *level* jumps
  across a concept switch — i.e. two measures spliced into one column. Flag-only; writes
  `data/gaps/fundamentals_tag_{ledger,breaks}.csv`. `n_boundaries` separates a one-time cutover from
  a systematic per-filing swap; `n_tickers_same_switch` separates a taxonomy migration (fix the
  candidate list) from one filer's mis-tagging (deny-list entry). It complements
  `analyze_history.py::detect_source_tag_misalignment`, which compares period-end vs interim tags
  *within* a fiscal year and deliberately ignores cross-year cutovers.
- **Over-strict guards cost real data.** A previous Q4 guard nulled 745 *correct* rows. Size any
  such fix by replaying it over the existing table before shipping it.

### DEF 14A

**edgartools' proxy HTML parser is silently wrong, not absent.** Every row goes through
[def14a_validate.py](../src/data_extract/utils/structure/def14a_validate.py). Only the XBRL-backed
block of `sec_def14a` is trusted unconditionally; the HTML-parsed child tables are best-effort and
are complemented by the LLM path. **Rule: never fabricate** — write a value only when
deterministically recoverable, else NaN.

### Earnings calls

- **Free via Motley Fool / HuggingFace, not FMP.** URL-slug carries the metadata; `bs4` is needed for
  the nested body; the Q&A split hinges on the operator hand-off.
- **Three header formats** exist for speaker turns; the per-turn split handles all three plus
  cleaning, and emits `answer_idx` so a question links to its answers.
- `constants.NO_EARNINGS_CALL_TICKERS = {BRK-B, BRK-A}` — Berkshire holds no call.
- `EARNINGS_CALL_REPORT_GRACE_DAYS = 50`, `EARNINGS_REPORT_TO_QUARTER_LAG_DAYS = 45` —
  the transcript is not available on the earnings date.
- The HF backbone is checked against `HF_BACKBONE_EARLY_QUARTER = "2005Q4"` /
  `HF_BACKBONE_LATE_QUARTER = "2025Q1"` so a truncated download is caught.
- Roic AI free tier is **5 req/min** → `ROIC_REQUEST_PAUSE = 12.5s`. Budget accordingly.

### Google Trends

`curl_cffi` TLS impersonation beats the 429. Weekly 15-year history requires **chunking into ≤4-year
windows and stitching** — a single long request is silently rescaled to monthly.

### Superinvestors (Dataroma)

No returns and no CIKs on the site; a broken SSL chain; manager names carry an `"Updated"` suffix.
The roster is resolved to CIKs via `constants.SUPERINVESTOR_CIK_OVERRIDES` and then used as a
manager-CIK filter over `sec13f_hr` to produce the `f_super_*` features. Best-effort: a Dataroma
failure must never break price extraction.

### SEC bulk sets — three different products, do not confuse them

| Product | Grain | Notes |
|---|---|---|
| Insider Data Sets | Forms 3/4/5 | quarterly zips from 2011 (`SEC_INSIDER_FIRST_YEAR`) |
| Financial Statement Data Sets | `num`/`sub` | from 2009; the pension source |
| Financial Statement **and Notes** Data Sets | `.tsv`, rolling monthly | from 2009; footnote numbers + text. Filter `dimn == 0` for the consolidated/undimensioned facts |

Fails-to-deliver has **two** URL templates: the legacy path up to `SEC_FTD_LEGACY_LAST_PERIOD =
"201706a"`, the current path from `201706b`.

### LLM cost discipline

Validate a slice with **no-LLM diagnostics first**, before spending calls. Narrow the text you send:
it is both cheaper and more accurate than a whole filing.
