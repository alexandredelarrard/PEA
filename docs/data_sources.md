# Data sources

Scope: the external sources, their keys, their real limits, and the quirks that have already cost
debugging time. All free or freemium **except Sharadar**, which is a paid subscription. For the
tables they land in, see [data_schema.md](data_schema.md); for current coverage,
[database.md](database.md).

## The sources

| Domain | Source | Key? | Lands in | Fetcher |
|---|---|---|---|---|
| Prices (OHLCV) | yfinance | no | `prices` | `prices/fetch_prices.py` (`auto_adjust=False` → BOTH `close_split` and `close_total`; `-F/--full` re-pulls history, and a ticker that split after its last stored bar is re-pulled automatically — split adjustment is RETROACTIVE) |
| Dividends (ex-dates) | yfinance | no | `dividends` | `prices/fetch_dividends.py` |
| Share splits (ex-dates) | yfinance | no | `prices_splits` | `prices/fetch_splits.py` (fills nine holes in `sharadar_actions`; ⚠ its `Stock Splits` column also carries SPINOFF factors, so `field_map.split_events` shape-tests both vendors) |
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
| Pay-versus-Performance (deterministic) | SEC DEF 14A **inline XBRL** (ECD taxonomy), read direct from `filing.xbrl()` | `SEC_USER_AGENT` | `sec_def14a` (2023+ by regulation) | `structure/def14a_ecd.py`, `structure/fetch_def14a_edgar.py` |
| Corporate events | SEC Form 8-K | `SEC_USER_AGENT` | `sec_8k` | `structure/fetch_8k_edgar.py` |
| Shareholder votes | SEC Form 8-K **Item 5.07**, re-read from `sec_8k.item_text` (no new download) | `OPENAI_API_KEY` | `sec_8k_votes` (2010-03+ by regulation) | `structure/fetch_8k_votes_llm.py` |
| Governance child tables | flattened out of the SAME paid `def14a_llm.def14a_json` blob | — | `def14a_directors`, `def14a_executive_comp`, `def14a_director_comp`, `def14a_ownership` | `structure/fetch_def14a_llm.py` |
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
- **ARQ's grain is one row per FILING, not per quarter** — amendments included. A quarter that was
  amended arrives twice, on one `reportperiod` under two `date`s: IBM's 2004-09-30 as the 10-Q of
  2004-10-28 and the 10-Q/A of 2004-11-01, and likewise KO (2002-03-11/13) and GOOGL
  (2007-05-09/10), all EDGAR-confirmed. Measured table-wide: **543 duplicate
  `(ticker, calendardate)` groups over 316 tickers** — 439 pure re-publications, 97 genuine
  restatements, 7 class-A collisions (below). `build_ttm._one_row_per_quarter` keeps the EARLIEST
  filing, because AR\* is as-reported and taking the amendment would file a restatement under an
  earlier publication date. A repeated quarter breaks a contiguity check exactly as a missing one
  does, so de-duplicating is a correctness requirement, not tidiness.
- **`calendardate` is a per-ticker FISCAL OFFSET, not a bounded normalisation.** It drifts as far as
  the filer's calendar demands and in EITHER direction: AVGO's 2024-02-04 maps FORWARD to 2024-03-31
  (+56d), WMT's 1995-07-31 BACKWARD to 1995-06-30 (−31d). So neither an absolute-drift cap nor a
  "the quarter containing `reportperiod`" containment test is valid — a 45-day cap deleted 239
  correct rows over 4 tickers, and containment measures 6,083 false rejects. **Sequence on the
  quarter ordinals, validate the window on `reportperiod` span** (`build_ttm.TTM_SPAN_DAYS`).
- **Class A**: 7 groups over BBY, GPN, KR and OKE are TWO REAL quarters whose fiscal ends normalise
  onto ONE `calendardate`. Any de-duplication must therefore key on `reportperiod`; keying on
  `calendardate` deletes a real quarter.
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

**edgartools' proxy HTML parser is silently wrong, not absent** — which is why the whole
HTML-parsed block and its four child tables were DELETED rather than repaired. A parser that
returns a fabricated `0.5` for a "*" percent, misses a "(in thousands)" fee header (KO: the same
fee read as 32,104 one year and 32,104,000 the next), and invents pay-ratio legs cannot be
repaired into a source. `sec_def14a` is now exactly the filer-tagged ECD block; everything a proxy
says in PROSE belongs to `def14a_llm` and its four child tables.

**The ECD reader also cannot use `ProxyStatement`'s accessors**: they filter on `concept ==` only
and take `.iloc[0]`, so on a co-PEO year document order decides which executive survives — BA's
2025 proxy silently drops one of Ortberg / Calhoun (and with it a CAP of −23,875,735).
[def14a_ecd.py](../src/data_extract/utils/structure/def14a_ecd.py) reads the facts frame directly.
Filers discriminate PEO facts two incompatible ways, so **the axis filter is conditional**: a
fixed `dim_ecd_ExecutiveCategoryAxis == 'ecd:PeoMember'` returns ZERO rows on every filing
measured (BA/NKE/SBUX tag `IndividualAxis` only; AAPL's amounts are undimensioned while its
26 `ecd:PeoName` facts are dimensioned, 21 of them as NEOs).

**`peo_actually_paid_comp` is negative on real filings** (NKE 2025: −10,924,243). Compensation
Actually Paid subtracts prior-year unvested fair value. There is no `abs()` on this path.

**Rule: never fabricate** — write a value only when deterministically recoverable, else NaN.

**The table-anchored carve is what makes the LLM path work.** `prepare_def14a_sections` routes
each section through a table classifier first (header signatures → TSV) and falls back to an
anchor carve, with per-section budgets. Measured mean payload dropped 51,198 → 42,225 chars
(−18%) while the tables the model needs got *more* of the budget, not less. Two positional facts
are load-bearing: `_TOC_SKIP_FRAC` is a front-matter FLOOR, and `_DIRECTOR_MAX_FRAC = 0.40` is a
CEILING on where the director-bios window may start — without it T's window landed at 53.6% of
the document (pension-assumptions prose, 0 of 11 directors) and EOG's at 70.0%. The ceiling moved
exactly those two filings and took in-block directors from 191 to 203 of 243.

### Shareholder votes — Form 8-K Item 5.07

**Item 5.07 is the only source of certified vote counts, and it starts 2010-03.** Rel. 33-9089
moved the disclosure out of 10-Q Part II Item 4. **No vote number is ever tagged in XBRL** —
Apple's 2025 annual-meeting 8-K carries 21 facts, 100% of them `dei:` cover-page tags — the SEC
publishes no data set, and no vendor publishes a free parse. `sec_8k_votes` reads the narratives
`fetch_8k_edgar` **already stores** in `sec_8k.item_text`, so it costs no new download.

**N-PX is not a substitute.** ~51% of N-PX filings contain zero vote records, 13F managers report
say-on-pay only (Rule 14Ad-1), funds hold ~33% of US equity, and 2,684 N-PX filings in 2025
mention Apple's CUSIP alone — that is the parsing bill to *partially* reconstruct one meeting
whose own 8-K gives the complete tally in one document.

**An LLM, not a parser, and it was measured.** Head-to-head on 60 filings / 690 hand-read rows:
rows fully correct 82.8% (parser) vs 82.5% (LLM), but **missed rows 39 vs 1** and **filings fully
clean 40% vs 65%**. The parser's failures are layout-driven and unbounded — 12 header
vocabularies across 34 filings, 41% combined-table layouts, 20.0% writing "Withheld" for
"Against", a vertical `For: 1,234` layout it misses 100% of the time, and a dropped-header case
that yields a silent COLUMN PERMUTATION.

**There is no accuracy gate, and that is a finding rather than an omission.** A vote table prints
no independent total and the dominant error is a column permutation, which is invariant under
sums: every computable check combined gives recall 0.56 / precision 0.56, with 7 of 16 known-bad
filings passing clean. `nominee_sum_matches` is stored as a monitor (computable on 96% of
filings, true on 91%) and never used as a filter. What does the work is a **fabrication guard** —
the model invented a whole table ("John Doe" / "Jane Smith", 250,000,000 votes) for a truncated
filing — so a comma-grouped number must exist in the source, and a nominee's name *and* at least
one of its counts must be printed there.

**Amendments are unioned, never deduped.** Of 190 multi-filing meetings, "latest wins" is correct
on 33 (17%) and unioning the group on 173 (91%), because **71% of amendments carry no vote
numbers at all**. Read on `(ticker, period_of_report)`. In a contested election the *first* 8-K is
the preliminary one — DIS `0000950157-24-000595` says its results exclude "shares voted on the
blue proxy card distributed by Trian" and the 8-K/A switches from Against to Withhold — but only
14 filings corpus-wide say "preliminary", so `mentions_preliminary` is a weak flag that resolves
8 of the 17 genuine restatements and no more.

**Zero rows is a normal outcome.** 8.8% of Item 5.07 filings are 5.07(d) board responses with no
tally at all; a separate 1.0% store a truncated `item_text` because edgartools' item splitter cut
at a filer's own "Item 1." numbering. The two are logged under distinct reasons. Note the length
floor cannot be the truncation defence: the shortest GENUINE tally measured is **554 chars**
(TDG's 2014 and 2019 special meetings) against HWM 2019's 508-char stub, so a floor high enough
to reject the stub destroys two correct filings.

### Schedule 13D

- **EDGAR renamed the form at the 2024-12-17 structured-XML mandate.** Filings through
  2024-12-16 are `SC 13D` / `SC 13D/A`; from 2024-12-17 they are `SCHEDULE 13D` /
  `SCHEDULE 13D/A`. `get_filings(form=...)` matches **exactly**, so
  `constants.SEC_13D_FORMS` lists **both pairs** — listing one silently truncates the table
  at the changeover with no error (measured: 461 filings across 91 tickers went missing,
  and `sec_13d` stopped dead on 2024-12-16).
- **`has_structured_data` means "has XML", not "is modern".** False for essentially every
  pre-mandate filing, True for essentially every one since. Pre-mandate it nulled
  edgartools' 0 defaults by accident; post-mandate filers defer the cover-page numbers to
  the narrative (`commentContent`: *"Rows 7, 8, 9, 10, 11, and 13: See Item 5"*) while
  tagging all six numerics 0. `_is_placeholder_numerics` nulls those and keeps the comment
  in `reporting_person_comment`; a genuine full disposal reports zeros with **no** comment
  and keeps them.
- **Item 3/4/5/6 prose is regex-carved** from `filing.text()` when there is no XML, using
  the **union** of a line-anchored and an anywhere-matching anchor set. Each reads filings
  the other cannot: line-anchoring rejects mid-prose cross-references (*"as described in
  Item 4"*) but cannot read a filing rendered as a single line; the anywhere-matcher can,
  but loses Item 3's end boundary when Item 4's caption varies, swallowing Item 4 whole.
- Amendment item coverage well below 100% is **Rule 13d-2(a) working as intended** — an
  amendment restates only materially changed items. Not a carve deficiency to chase.
- 13G (passive stakes) is **deliberately excluded**; this table is activist-only.

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
