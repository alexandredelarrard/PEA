---
title: Data sources
description: External providers, credentials, coverage boundaries, and source-specific failure modes.
type: reference
tags:
  - wiki
  - reference
  - data-sources
  - extraction
---
# Data sources

## Purpose

This page is the source-facing operating contract: where data comes from, which credentials it needs, which table receives it, how far history really reaches, and which source-specific behaviors must not be normalized away. The extraction package is described in [data extraction](../modules/data-extract.md), and table grains are in the [table catalog](./table-catalog.md).

## Source map

| Domain | Provider | Credentials | Destination | Primary implementation |
| --- | --- | --- | --- | --- |
| Equity OHLCV, dividends, splits | yfinance | none | `prices`, `dividends`, `prices_splits` | [price fetcher](../../src/data_extract/utils/prices/fetch_prices.py) |
| Benchmark, volatility, commodity, energy | yfinance | none | `prices_macro` | [fetch_macro.py](../../src/data_extract/utils/prices/fetch_macro.py) |
| Rates, credit, breakeven, FX | FRED | `FRED_API_KEY` | `prices_macro` | [fetch_macro.py](../../src/data_extract/utils/prices/fetch_macro.py) |
| Current S&P 500 roster | Wikipedia | none | `sp500_tickers` | [universe fetcher](../../src/data_extract/utils/prices/fetch_tickers.py) |
| Per-filing fundamentals | SEC EDGAR XBRL | `SEC_USER_AGENT` | `fundamentals_facts` → `fundamentals_history_sec` | [SEC fundamentals fetcher](../../src/data_extract/utils/fundamentals/fetch_fundamentals_sec.py) |
| Vendor fundamentals | Sharadar Direct API | `SHARADAR_API_KEY` | `fundamentals_sharadar` → `fundamentals_history` | [Sharadar fetcher](../../src/data_extract/utils/fundamentals_sharadar/fetch_sharadar.py) |
| Vendor entity/actions/index history | Sharadar | `SHARADAR_API_KEY` | `sharadar_tickers`, `sharadar_actions`, `sharadar_sp500` | same producer |
| Employee headcount | SEC annual filing text plus OpenAI structured output | `SEC_USER_AGENT`, `OPENAI_API_KEY` | `fundamentals_employees` | [headcount fetcher](../../src/data_extract/utils/fundamentals/fundamentals_employees.py) |
| Earnings surprises | yfinance | none | `earnings_surprises` | fundamentals fetchers |
| Pension and note datasets | SEC bulk ZIPs | `SEC_USER_AGENT` | `pension_facts`, `notes_num`, `notes_text` | fundamentals fetchers |
| Institutional holdings | SEC EDGAR 13F-HR listing by filing date (edgartools), CUSIP map from OpenFIGI | `SEC_USER_AGENT`; optional OpenFIGI key | `sec13f_hr`, `sec13f_manager_holdings` (roster managers), `cusip_ticker_map` | [13F fetcher](../../src/data_extract/utils/institutionals/fetch_13f.py) |
| Elite-manager roster/books | Dataroma, Wayback, SEC 13F | `SEC_USER_AGENT` | `superinvestor_roster`, `sec13f_manager_holdings` | [superinvestor fetcher](../../src/data_extract/utils/institutionals/fetch_superinvestors.py), [manager catch-up](../../src/data_extract/utils/institutionals/fetch_13f_managers.py) |
| Insider transactions | SEC quarterly datasets plus daily ownership XML | `SEC_USER_AGENT` | `insider_transactions` (one table, both sources), `insider_footnotes` | [EDGAR fetcher](../../src/data_extract/utils/institutionals/fetch_insider_edgar.py), [zip fetcher](../../src/data_extract/utils/institutionals/fetch_insider_transactions.py) |
| Governance and compensation | SEC DEF 14A plus OpenAI structured output | `SEC_USER_AGENT`, `OPENAI_API_KEY` | `def14a_llm` and four child tables | [DEF 14A fetcher](../../src/data_extract/utils/structure/def14a/fetch.py) and [GPT extraction](../modules/gpt-extract.md) |
| Pay-versus-performance | DEF 14A inline XBRL | `SEC_USER_AGENT` | `sec_def14a` | DEF 14A ECD reader |
| Corporate events and shareholder votes | SEC 8-K plus OpenAI for Item 5.07 | SEC and OpenAI keys | `sec_8k`, `sec_8k_votes` | institutional then structure utilities |
| Activist/passive stakes | SEC Schedule 13D/13G | `SEC_USER_AGENT` | `sec_13d`, `sec_13d_transactions`, `sec_13g` | institutional utilities |
| Filing narrative | SEC 10-K/10-Q | `SEC_USER_AGENT` | `sec_filing_text` | structure utilities |
| Short volume and settlement fails | FINRA RegSHO, SEC | none | `short_interest`, `sec_fails_to_deliver` | [FTD fetcher](../../src/data_extract/utils/institutionals/fetch_fails_to_deliver.py) and institutional utilities |
| Earnings calls | HuggingFace dataset `defeatbeta/yahoo-finance-data` | none | `earnings_call_sections` | `src/data_extract/utils/behavioral/fetch_earnings_call_transcripts.py` |
| Tone and embeddings | local FinBERT/lexicon and OpenAI | OpenAI only for embeddings | `earnings_call_sentiment`, `earning_calls_embedding` | text aggregation (`src/data_aggregate/utils/text/`) and [gpt_extract](../../src/gpt_extract/) |

The [notes fetcher](../../src/data_extract/utils/fundamentals/fetch_financial_notes.py) applies an explicit archive-date policy. For periods through August 2026, `available_at` is **estimated** as the 12th of the following month, moved to Monday when the 12th falls on a weekend (`2021_08` → 2021-09-13). For September 2026 archives onward, a new download uses its successful completion date in New York; an already-cached ZIP without a recorded download date falls back to its file modification date. HTTP `Last-Modified` is not used. `financial-notes --repair-availability` corrects existing clocks with primary-key-only updates, without rereading ZIP payloads. The [notes feature builder](../../src/data_aggregate/utils/fundamentals/fundamental_features.py) starts numeric facts at `max(filed, available_at)` and forward-fills the latest eligible annual value for at most 460 calendar days unless replaced sooner. The historical 12th is a modelling assumption, **not a verified SEC publication date**; late releases and corrected archives can still create look-ahead bias.

The [financial-statements fetcher](../../src/data_extract/utils/fundamentals/fetch_financial_statements.py) uses the analogous quarterly rule for `pension_facts`: historical ZIPs through 2026q2 are estimated at quarter end plus 12 calendar days, advanced to Monday after a weekend (2025q2 → 2025-07-14). From 2026q3, a first successful download is dated on its New York completion day; a cached ZIP with no stored clock falls back to its modification date, and reruns retain the stored quarter clock. The table retains each ZIP quarter as a separate vintage, so a later corrected value cannot overwrite what an earlier archive contained. [The pension feature reader](../../src/data_aggregate/utils/fundamentals/fundamental_features.py) starts each value at `max(filed, available_at)`, rejects missing clocks, and expires it after 460 calendar days unless a newer value arrives. The historical +12 date is an assumption, not proof of SEC publication; a ZIP actually posted later would still imply look-ahead risk.

Secrets live only in the ignored root `.env`, loaded by [Context](../../src/context.py). Never copy credentials into config, logs, reports, or wiki content. `SEC_USER_AGENT` must identify a real contact.

## Shared transport and extraction plumbing

- [polite_http.py](../../src/utils/polite_http.py) supplies rate limiting and TLS impersonation for sources that reject ordinary clients.
- [ssl_setup.py](../../src/utils/ssl_setup.py) exports the Windows trust store to the ignored `.cache/corporate_ca_bundle.pem`; Linux Airflow processes reuse that file through the repository bind mount before `curl_cffi` imports freeze certificate state. TLS verification remains enabled.
- `src/data_extract/utils/common/` owns bulk caches, SEC request state, rate limiting, parallel entity walks, run manifests, and incremental resume helpers. Each mechanism exists once:
  - [edgar_driver.py](../../src/data_extract/utils/common/edgar_driver.py): every per-ticker EDGAR fetcher (8-K, 13D, 13G, DEF 14A ECD, filing text, SEC fundamentals, EDGAR insider) declares one frozen `EdgarFetch` (tables, build function, `identity_aware`, optional `minimum_since` floor, optional `listing_since` start that overrides the manifest window, optional `done_where` filter on the stored-accession dedup set) and `run_edgar_fetch` walks it: listing window, accession dedup on `tables[0]`, guarded thread pool, per-ticker upsert, run-manifest record. There is no partial-success mode: a failed ticker raises `IncompleteEdgarRunError` before any manifest entry advances, and a recorded run is always `coverage_complete`. `FilingStamp` reads the filing-level values every fetcher stamps (filing CIK, filing date, guarded `period_of_report`, document URL) once per filing; `build_filing_rows` is the generic single-table builder.
  - [parallel_fetch.py](../../src/data_extract/utils/common/parallel_fetch.py): `run_per_ticker` returns results in scope-row order; a programming error aborts the pool, any other exception logs and yields `None`.
  - [sec_atom.py](../../src/data_extract/utils/common/sec_atom.py): the owner-inclusive company-browse Atom feed (page URL, fetch, entry parse, lazy paging in `iter_atom_pages`, the `keep_atom_entry` form/date/accession filter) shared by the Schedule 13D/13G subject search in `registrant.py` and the EDGAR insider listing. A failed page raises `AtomPageError` with its offset; each caller owns its failure policy.
  - [item_carve.py](../../src/data_extract/utils/common/item_carve.py): item-heading regexes and span scan shared by the 10-K/10-Q section carver and the 13D item carver. [frame_sanitize.py](../../src/data_extract/utils/common/frame_sanitize.py) `finalise_frame` de-duplicates rows on the table primary key before any upsert.
  - [bulk_cache.py](../../src/data_extract/utils/common/bulk_cache.py): `read_zip_tables` (with one `ZipRead` spec per member) reads every SEC tab-separated bulk zip (statements, notes, insider, symbol tenure, entity lineage); each caller picks `on_corrupt="delete"` (re-download) or `"skip"`. `pending_periods` / `mark_processed` decide which cached periods a bulk fetcher re-parses: all of them on reparse or when the scope gained members since the last sidecar, otherwise only periods with no stored row.
  - [run_manifest.py](../../src/data_extract/utils/common/run_manifest.py) `manifest_window` relists the full window when no run is recorded, the ticker set differs from the recorded one (a same-size swap included), or the last full rescan is too old; manifest writes are atomic (temporary file plus replace).
- [registrant.py](../../src/data_extract/utils/common/registrant.py) is the single authority on which legal CIKs may supply a ticker's filings.
- `DataStore.ensure_table` creates a cold table under a process-wide lock with a re-check, so threaded first writers cannot race the `CREATE` ([store.py](../../src/data_store/store.py)).
- [gpt_extract](../../src/gpt_extract/) is the shared LLM service; source packages provide tasks and schemas, not duplicate OpenAI clients.

## Price-basis contract

`prices` contains equities only. Market and macro series are named rows in `prices_macro`, preventing benchmark symbols from contaminating wide cross-sectional ranks.

Use `close_split` for levels such as market capitalization, execution price, enterprise value, yields, and ATR inputs. Use `close_total` for returns, momentum, volatility, betas, and labels. Split/spinoff corrections that affect cube levels are applied when building `cube_part_prices`; the raw vendor table remains auditable. See [cube build](../flows/cube-build.md).

## Sharadar SF1

The supported channel is the Sharadar Direct API. Its field names and entitlements differ from Nasdaq Data Link clients, so the channels are not interchangeable.

Operational rules:

- Always pass an explicit start bound, sort, and paging policy; provider defaults can silently return only recent history.
- Validate response headers because an unknown field may be omitted instead of rejected.
- Only AR dimensions are point-in-time. MR dimensions mutate and are not stored.
- ARQ rows are filing-grain, so amendments can repeat a report period. Quarter resolution keys on the actual report period, not normalized calendar date.
- Q4 is constructed from the annual total and first three quarters; the equality is not an independent quality check.
- Currency, unit, sign, and split-adjustment conventions are field-specific. The producer asserts USD eligibility and applies the approved zero/sign rules before merging.
- `lastupdated` is a per-ticker processing stamp, not a row watermark. Full refresh is the restatement path.
- Read `fundamentals_sharadar` projected: it is one of the widest extract tables.

## SEC fundamentals and identity

Per-filing XBRL is used because aggregate company-facts feeds can omit dimensioned facts. Candidate concepts are priority-coalesced per period, then narrow non-negative and per-issuer deny rules can reject a bad candidate without pinning an alternative.

Never join SEC data by free-text entity name. Identifier fields such as CIK, CUSIP, and accession number remain text to preserve leading zeros.

A ticker's price history follows the economic entity, while filings follow legal registrants. Identity-consuming EDGAR fetchers load `symbol_tenure` and `entity_lineage` once, expand same-CIK historical aliases, and persist a per-ticker identity-scope fingerprint in the run manifest. `build_identity` precomputes each entity's (symbol, CIK) pairs, so `Identity.filing_scope(ticker)` in [identity.py](../../src/data_extract/utils/common/identity.py) returns a `FilingScope` (entity, roster CIK, every entity CIK, symbols, same-CIK aliases) without rescanning the tables; `load_edgar_scope` in the driver fingerprints all tickers from it once per run. A fingerprint change relists the full configured window for only that ticker; unchanged tickers keep their incremental frontier. Multi-CIK consolidating history still requires a complete curated dated chain: without one, SPLIT forms list only the roster CIK (or the curated chain) and log a warning naming the uncurated CIKs, so predecessor history stays missing until it is curated in `configs/sec/registrant_cutover.json`. Any listed filing whose CIK is outside the issuer lineage is dropped, because `Company(alias)` resolves a reused symbol to its current holder. [registrant.py](../../src/data_extract/utils/common/registrant.py) owns the scope rules, while [edgar_driver.py](../../src/data_extract/utils/common/edgar_driver.py) owns fingerprint invalidation and manifest advancement.

Both identity tables come from one pass over the cached Form 3/4/5 quarter zips: `scan_form345_cache` in [symbol_tenure.py](../../src/data_extract/utils/common/symbol_tenure.py) returns a `Form345Scan` holding each quarter's (symbol, issuer CIK) tenure aggregate and the distinct (issuer, reporting-owner) CIK pairs; `build_symbol_tenure` consumes the first and [entity_lineage.py](../../src/data_extract/utils/common/entity_lineage.py) the second, with no database reload in between. A corrupt zip is skipped and kept. Each table's `replace` is skipped when `incremental.matches_stored` finds the rebuilt frame identical to the stored one (a dtype mismatch only causes a rewrite).

Employee headcount is an independent annual-filing extraction. [fundamentals_employees.py](../../src/data_extract/utils/fundamentals/fundamentals_employees.py) walks 10-K, 10-K/A, and 10-K405 through the dated registrant resolver, checks the actual filing CIK against entity lineage, and sends filing text to the configured cheap OpenAI model. Only an issuer-wide count supported by source text is saved. The quote is located on letters and digits only (immune to line-break hyphens, split words and stray punctuation), and ` ... `-joined fragments must sit within 2,000 characters with no date or excerpt gap between them. The model's count must be a stated number or a sum of stated components. Part-time components are dropped (full-time only), `61 thousand` reads as 61,000, and an open bound moves half a unit of its precision (`over 13,000` -> 13,500, `nearly 2.2 million` -> 2,150,000). Ranges, image-only and unsupported claims stay null. `as_of` is the SEC filing date. The table alone records what is decided: a filing with no supported count (image-only, not disclosed, ambiguous) is stored as a NULL row, so it is never sent to the LLM again and does not fail the run; only a ticker that could not be read keeps the run incomplete. Each run lists the whole `years_history` window and skips filing dates that already have a row, count or NULL, before reading filing text. A filing listed in [configs/sec/employees_manual_roster.json](../../configs/sec/employees_manual_roster.json) takes its count (or null) from there instead of the LLM; the pipeline only reads that file. To have a filing re-decided, delete its table row (and its roster entry, if any). `--full` re-decides every filing in the window, the roster still winning for its accessions. The Sharadar merge ignores NULL rows, so a prior year's count keeps carrying. The XBRL facts rebuild does not write or delete employee history.

The form-specific combination policy is:

| Form family | Policy | Reason |
| --- | --- | --- |
| Event forms: 8-K, 13D, 13G, Forms 3/4/5 | Union all declared registrants | An event remains relevant even if indexed under a predecessor after a corporate boundary. |
| Consolidating forms: 10-K, 10-Q, DEF 14A and their carved text | Dated split | Unioning predecessor and successor financial statements can mix a subsidiary into the parent. |

A form without an explicit policy raises. A name change with the same CIK is not a registrant cutover.

## Institutional coverage

Source history and usable feature history differ:

- 13F rows may exist decades earlier, but broad manager coverage becomes usable only at the measured regime boundary. Coverage is measured on managers, not ticker count.
- 13F holdings are an S&P 500 slice; complete manager books provide the denominator for manager portfolio weights.
- `thirteen-f` is one walk over every 13F-HR filed from the stored `sec13f_hr` `filing_date` watermark minus a 7-day lookback (the configured history on an empty table). The listing is sorted oldest-first by (filing date, amendment flag, accession), so an amendment always overwrites its original, even one filed the same day, and a crash never leaves the watermark past an unsaved filing. Each filing is parsed once and feeds both tables: the S&P 500 ticker slice to `sec13f_hr` and, for CIKs on `superinvestor_roster`, the complete CUSIP book to `sec13f_manager_holdings`; within a save batch the last filed row wins on each primary key. A `filing_window` backfill neither reads nor advances the watermark. See [fetch_13f.py](../../src/data_extract/utils/institutionals/fetch_13f.py).
- `thirteen-f-managers` is a per-CIK catch-up for every CIK ever on the roster: each CIK lists its own 13F-HR filings and reads only those its stored book does not show yet. A filing is done when `sec13f_manager_holdings` holds rows with its own (period, filing date) pair, or a later filing of the same period is stored (a restatement replaces its original's rows); a CIK with no stored book reads its whole history. The stored rows are the only retry state. A transient read failure (throttle, 5xx, timeout, unreachable SEC) holds that whole quarter back with a WARNING, so the next run retries it; any other read failure, such as an unparseable information table, skips only that filing with an ERROR naming the CIK, accession and period, and the quarter's readable filings are saved. Same parser and save as the main walk; the last filed row wins per (cik, period, cusip). An empty roster raises `SuperinvestorRosterEmptyError`. See [fetch_13f_managers.py](../../src/data_extract/utils/institutionals/fetch_13f_managers.py).
- Superinvestor hand resolutions (Dataroma code to CIK overrides and recorded-unresolvable codes with reasons) live in [superinvestor_overrides.json](../../configs/sec/superinvestor_overrides.json), read by `load_superinvestor_overrides(config_dir)`; an override wins over any stored or EDGAR resolution.
- Schedule 13D and 13G share one row builder and one issuer-guarded filing walk in [schedule_rows.py](../../src/data_extract/utils/institutionals/schedule_rows.py); each form module declares a `ScheduleSpec` (columns, form-specific fields, numeric trust, reporting-person CIK, event-date parse, blank normalisation).
- Insider bulk history begins at the dataset's filing-date floor. Older transaction dates do not prove older publication coverage.
- RegSHO history is a moving provider window; the oldest stored anomaly is not a guaranteed recoverable boundary. Its CIK-less symbols, and those in FTD, resolve through [identity.py](../../src/data_extract/utils/common/identity.py): dot and slash share-class spellings use the roster's hyphen spelling. Berkshire Class A is excluded from the Class B series as a separate traded security. The [manual tenure file](../../configs/sec/symbol_tenure_manual.json) records the evidenced `BK` to `BNY` transition at the 2026-05-21 market open; earlier `BNY` tenure belongs to an unrelated fund. When the latest observed interval of an exact current roster symbol ends solely because insider filings stop, the resolver may continue the current entity as `roster_tenure_proxy` if no later entity held the symbol and no manual interval explicitly ended it. The stored `valid_to` remains the last filing observation.
- SEC fails-to-deliver history has a fixed publication start. Each row persists its source ZIP `period`; the short-flow builder calculates availability from that tag without a metadata table. A real-data audit found day-15 rows in 141 `b` ZIPs, so settlement day cannot reliably identify the ZIP half. Historical periods use period end plus 15 calendar days, advanced past weekends; the builder snaps to the next trading session, including holidays. For only the highest stored period whose end is within 60 days of New York today, a cached ZIP's New York modification date overrides that estimate when it is today or yesterday. Missing, future, or older timestamps use the estimate. File copying can change the timestamp, and the override expires after day 1; it is not immutable first-publication evidence. See the [FTD fetcher](../../src/data_extract/utils/institutionals/fetch_fails_to_deliver.py), [bulk cache](../../src/data_extract/utils/common/bulk_cache.py), and [short-flow builder](../../src/data_aggregate/utils/institutionals/short_flow_features.py).
- Schedule 13D/13G filer identity and event dates exist historically, while reliable ownership numerics begin with structured-data mandates.
- Missing data after a global source start can still be unavailable for a security, denominator, filing, or completeness frontier.

Insider transactions live in one table, `insider_transactions`, keyed (`accession_number`, `security_type`, `row_sequence`), with a `source` column (`edgar` or `zip`). One `insider-transactions` run ingests the pending zip quarters first, then EDGAR:

- **EDGAR daily is authoritative.** [fetch_insider_edgar.py](../../src/data_extract/utils/institutionals/fetch_insider_edgar.py) lists each ticker's Forms 3/4/5 from that ticker's own latest stored `filing_date` minus 7 days (the configured history when no row is stored under the ticker; rows carry the CIK-resolved universe ticker, so only a renamed roster ticker whose rows still carry the old label pays that full listing) and skips accessions already stored with `source='edgar'`. A zip-sourced accession in that window is re-read and replaced whole: its EDGAR rows keep the zip `quarter`, then its zip rows are deleted, so no accession holds rows from both sources.
- **A zip quarter fills what EDGAR missed.** [fetch_insider_transactions.py](../../src/data_extract/utils/institutionals/fetch_insider_transactions.py) inserts only the filings EDGAR has not stored (`source='zip'`); on a filing EDGAR already holds it only stamps `quarter` on the stored keys. Per quarter it logs a WARNING `insider <q>: X / N filings missing from EDGAR (x%), added from zip; top: ...`, where N counts the zip filings filed on or after the quarter's earliest EDGAR filing date, plus an INFO line with the EDGAR-only filing count and the mismatch rate on shared rows (code, transaction date, shares, price, holding after, primary owner; numbers within 0.005). A quarter with no EDGAR rows logs `loaded from zip (no EDGAR coverage)` instead.
- **Identity rejects are not stored.** The CIK-first screen drops rows whose issuer does not resolve to a universe entity and logs one WARNING per run (filings, rows, P/S rows, reason split, top claimed tickers), or one INFO line when nothing was excluded. The stored-row sweep runs after a zip run only on a full-universe run (no `-t`): it deletes stored filings that now fail the screen and logs the same summary. A `-t` subset run skips it with an INFO line, so it never deletes another company's rows. There is no quarantine table and no per-ticker coverage table.
- **The completeness frontier is the run manifest.** Only the EDGAR run writes the `insider_transactions` manifest entry (`coverage_complete`, tickers, `last_run_date`); the zip ingest never records a run, so it cannot erase or advance that proof. See [source availability](../concepts/source-availability.md).

The zip data sets round every number to 2 decimals and lack `footnote_ids` and `acceptance_datetime`; EDGAR's values win on overlap. The zip download must stay even when EDGAR covers a quarter: `symbol_tenure` and `entity_lineage` are built from the cached zips. EDGAR can still miss filings silently on an Atom 503 (see the [TODO](../TODO.md)); the per-quarter missing-from-EDGAR warning is where that loss shows.

Both insider paths share one contract in [insider_common.py](../../src/data_extract/utils/institutionals/insider_common.py): the bulk TSV reader ([fetch_insider_transactions.py](../../src/data_extract/utils/institutionals/fetch_insider_transactions.py)) and the EDGAR Form 3/4/5 XML parser ([insider_edgar_parser.py](../../src/data_extract/utils/institutionals/insider_edgar_parser.py)) extract the same declarative `INSIDER_FIELDS` string frame plus one owner row per reporting owner, type it once with `build_insider_frame`, and screen it with `screen_insider_rows`. The encoding is shared: one numeric parser (`parse_number`), `value_usd` = shares × price filled by the stated total, absent role relationships as 0, a blank ticker as NULL. Only the date formats differ (`%d-%b-%Y`/ISO for the zip, mixed parsing for the XML). `owner_summary` applies the joint-filing rule: the primary owner is the best role rank (Officer, Director, 10% owner, Other), then the lowest numeric CIK; `owner_ciks` and `n_reporting_owners` keep every owner and the role flags are OR'ed across them, so a joint filing stays one row per trade. The zip reader prefilters accessions on the SUBMISSION verdict (`screened_accessions`) before building transaction rows and ranks the surrogate key per (accession, table) into `row_sequence`. The stored-row sweep reads distinct issuer CIKs first and loads full rows only for rejected CIKs. The EDGAR listing pages the owner-inclusive Atom feed per form family through `sec_atom.py`.

## Governance and shareholder votes

DEF 14A prose uses table-aware section carving before LLM extraction. Structured Pay-versus-Performance facts are read directly from filing XBRL, including legitimate negative compensation-actually-paid values. A value is emitted only when deterministically recoverable; parsers never fabricate a replacement. The ECD reader's listing is floored at 2022-12-16, the Item 402(v) effective date (`_PVP_EFFECTIVE`, passed as `EdgarFetch.minimum_since` in [fetch_def14a_edgar.py](../../src/data_extract/utils/structure/fetch_def14a_edgar.py)); no earlier proxy carries ECD facts.

The DEF 14A LLM extractor ([def14a/fetch.py](../../src/data_extract/utils/structure/def14a/fetch.py)) keeps its own cross-registrant lister rather than `resolve_registrant_filings`; moving it needs a listing-equivalence check first (see [TODO](../TODO.md)). Its listing window comes from `manifest_window` keyed on the requested ticker set, and its resume check reads `def14a_llm` tickers with a scoped `distinct`. DEF 14A and Item 5.07 vote extraction both size their LLM worker pool from `gpt.threads` in [gpt.yml](../../configs/gpt.yml) unless a caller pins one explicitly; employee extraction pins one thread per ticker.

Item 5.07 is the certified shareholder-vote source and begins with its regulatory regime. Important semantics:

- one election row can aggregate many nominees while retaining nominee-level JSON;
- “against” and “withheld” are legally distinct standards;
- amendments are unioned unless evidence identifies a genuine restatement;
- validation monitors cannot reliably detect every column permutation, so source-text fabrication guards are mandatory;
- zero proposal rows can be a legitimate board-response filing rather than an extraction failure.

Both DEF 14A and vote extraction use the [SEC and LLM flow](../flows/sec-llm-extraction.md).

## Earnings-call transcripts and attention data

Earnings-call transcripts have one source: the free HuggingFace dataset `defeatbeta/yahoo-finance-data`, file `data/US/stock_earning_call_transcripts.parquet` (about 2.27 GB, one parquet sorted by symbol in 1,195 row groups). Its terms allow research and non-commercial use. The file is rebuilt about daily near 05:00 UTC; calls held on 2026-09-30 were already in the 2026-10-01 build, so publication lag is at most one build. History starts 2005-10-11. The full load of 2026-10-02 stored 33,591 calls for 487 roster tickers; BRK-B, ED, EXPD and NVR are absent from the dataset. Evidence: `reports/validate/2026-10-01-earnings-call-extraction-gaps/` (`01-research.md`, `03-implementation.md`).

Source facts the extractor relies on:

- `report_date` is the call date (it matched an independent call date on 1,477 of 1,487 shared calls). It is date-only, so a pre-market call cannot be told from an after-close call; the cube therefore shows a call from the next session.
- `(symbol, fiscal_year, fiscal_quarter)` is unique. `transcripts_id` is NULL on 2,914 of 33,676 roster calls, so a re-issued call is detected by a changed id **or** a changed call date.
- Provider fiscal relabels put 23 roster call dates of 10 tickers (DG, DLTR, HD, LOW, LULU, TGT, ULTA, WSM, PCAR, RJF) under two fiscal labels. One call is kept per (ticker, call date): the label one ordinal below the next later call's label, otherwise the highest.
- Each call is a list of `paragraph_number`, `speaker`, `content`; `paragraph_number` starts at 1 and is contiguous. Symbols use the roster hyphen spelling (`BF-B`); a `.` spelling is still mapped to `-`.
- Speaker labels are unreliable for a few issuers (TJX), and some transcripts are defective (truncated Q&A, answers merged into analyst paragraphs). Such calls fail the quality gate and produce null features.
- A source-schema change raises before any read; the extractor never guesses a new layout.

The table stores raw paragraphs. The speaker-turn split, text cleaning, FinBERT sentiment and embeddings run at aggregate time (see [cube aggregation](../modules/data-aggregate.md)), so a split fix needs no refetch.

A quarter counts as covered only when the speaker-turn split returns `ok` and the cleaned prepared remarks and management answers are both present with at least 100 combined cleaned words. Stored paragraphs alone do not establish coverage. A re-issued call (same ticker and quarter, new `transcripts_id` or call date) has its old paragraphs deleted, its sentiment and embedding rows invalidated, and a pending refresh marker written at the earlier of the two call dates; the text aggregate uses that date to repair persisted signals and acknowledges the marker only after the write succeeds. Sentiment resume keys carry the cleaned-text cache version (`speaker-clean-v2`), and embedding resume keys carry the model tag `text-embedding-3-small:speaker-clean-v2`, so a cleaning change re-scores and re-embeds through the normal path.

There is no Wikipedia pageview or Google Trends fetcher.

## Incremental and cost discipline

- Resume from database frontiers through `max_date`, `max_date_by`, and the shared resume helper. Never load a table merely to find where to continue.
- Match frontier grain to provider grain: entity-specific for per-ticker APIs, global for market-wide day files.
- Persist per entity or durable chunk so an interrupted expensive walk keeps completed work.
- Re-list SEC histories periodically to self-heal missed or out-of-order filings.
- Cache bulk archives and immutable source files under `data/`.
- Validate representative slices without LLM calls before paying for a broad run.
- A source boundary marks possible availability, not automatic zeros or complete ticker coverage.

## Related

- [Table catalog](./table-catalog.md)
- [Live database](./live-database.md)
- [Data access and storage](../guides/data-access.md)
- [Add a data source](../guides/add-a-data-source.md)
- [Source availability](../concepts/source-availability.md)
