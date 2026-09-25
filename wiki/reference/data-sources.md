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
| Equity OHLCV, dividends, splits | yfinance | none | `prices`, `dividends`, `prices_splits` | [prices utilities](../../src/data_extract/utils/prices/) |
| Benchmark, volatility, commodity, energy | yfinance | none | `prices_macro` | [fetch_macro.py](../../src/data_extract/utils/prices/fetch_macro.py) |
| Rates, credit, breakeven, FX | FRED | `FRED_API_KEY` | `prices_macro` | [fetch_macro.py](../../src/data_extract/utils/prices/fetch_macro.py) |
| Current S&P 500 roster | Wikipedia | none | `sp500_tickers` | [prices utilities](../../src/data_extract/utils/prices/) |
| Per-filing fundamentals | SEC EDGAR XBRL | `SEC_USER_AGENT` | `fundamentals_facts` → `fundamentals_history_sec` | [fundamentals utilities](../../src/data_extract/utils/fundamentals/) |
| Vendor fundamentals | Sharadar Direct API | `SHARADAR_API_KEY` | `fundamentals_sharadar` → `fundamentals_history` | [fundamentals_sharadar](../../src/data_extract/utils/fundamentals_sharadar/) |
| Vendor entity/actions/index history | Sharadar | `SHARADAR_API_KEY` | `sharadar_tickers`, `sharadar_actions`, `sharadar_sp500` | same producer |
| Employee headcount | SEC 10-K body text | `SEC_USER_AGENT` | `fundamentals_employees` | [fundamentals utilities](../../src/data_extract/utils/fundamentals/) |
| Earnings surprises | yfinance | none | `earnings_surprises` | fundamentals fetchers |
| Pension and note datasets | SEC bulk ZIPs | `SEC_USER_AGENT` | `pension_facts`, `notes_num`, `notes_text` | fundamentals fetchers |
| Institutional holdings | SEC 13F bulk | `SEC_USER_AGENT`; optional OpenFIGI key | `sec13f_hr`, `cusip_ticker_map` | [institutional utilities](../../src/data_extract/utils/institutionals/) |
| Elite-manager roster/books | Dataroma, Wayback, SEC 13F | `SEC_USER_AGENT` | `superinvestor_roster`, `sec13f_manager_holdings` | institutional utilities |
| Insider transactions | SEC quarterly datasets plus daily ownership XML | `SEC_USER_AGENT` | canonical, live, coverage, footnote, and quarantine tables | institutional utilities |
| Governance and compensation | SEC DEF 14A plus OpenAI structured output | `SEC_USER_AGENT`, `OPENAI_API_KEY` | `def14a_llm` and four child tables | [structure utilities](../../src/data_extract/utils/structure/) and [GPT extraction](../modules/gpt-extract.md) |
| Pay-versus-performance | DEF 14A inline XBRL | `SEC_USER_AGENT` | `sec_def14a` | DEF 14A ECD reader |
| Corporate events and shareholder votes | SEC 8-K plus OpenAI for Item 5.07 | SEC and OpenAI keys | `sec_8k`, `sec_8k_votes` | institutional then structure utilities |
| Activist/passive stakes | SEC Schedule 13D/13G | `SEC_USER_AGENT` | `sec_13d`, `sec_13d_transactions`, `sec_13g` | institutional utilities |
| Filing narrative | SEC 10-K/10-Q | `SEC_USER_AGENT` | `sec_filing_text` | structure utilities |
| Short volume and settlement fails | FINRA RegSHO, SEC | none | `short_interest`, `sec_fails_to_deliver` | institutional utilities |
| Retail attention | Wikipedia and Google Trends | none | `wiki_pageviews`, `google_trends` | [behavioral utilities](../../src/data_extract/utils/behavioral/) |
| Earnings calls | HuggingFace, Roic AI, Motley Fool | none | `earnings_call_sections` | behavioral utilities |
| Tone and embeddings | local FinBERT/lexicon and OpenAI | OpenAI only for embeddings | sentiment and embedding tables | behavioral and [gpt_extract](../../src/gpt_extract/) |

Secrets live only in the ignored root `.env`, loaded by [Context](../../src/context.py). Never copy credentials into config, logs, reports, or wiki content. `SEC_USER_AGENT` must identify a real contact.

## Shared transport and extraction plumbing

- [polite_http.py](../../src/utils/polite_http.py) supplies rate limiting and TLS impersonation for sources that reject ordinary clients.
- [ssl_setup.py](../../src/utils/ssl_setup.py) configures the corporate CA bundle before `curl_cffi` imports freeze certificate state.
- [data_extract/utils/common](../../src/data_extract/utils/common/) owns bulk caches, SEC request state, rate limiting, parallel entity walks, the form registry, run manifests, and incremental resume helpers.
- [registrant.py](../../src/data_extract/utils/common/registrant.py) is the single authority on which legal CIKs may supply a ticker's filings.
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

A ticker's price history follows the economic entity, while filings follow legal registrants. [registrant.py](../../src/data_extract/utils/common/registrant.py) applies form-specific combination policy:

| Form family | Policy | Reason |
| --- | --- | --- |
| Event forms: 8-K, 13D, 13G, Forms 3/4/5 | Union all declared registrants | An event remains relevant even if indexed under a predecessor after a corporate boundary. |
| Consolidating forms: 10-K, 10-Q, DEF 14A and their carved text | Dated split | Unioning predecessor and successor financial statements can mix a subsidiary into the parent. |

A form without an explicit policy raises. A name change with the same CIK is not a registrant cutover.

## Institutional coverage

Source history and usable feature history differ:

- 13F rows may exist decades earlier, but broad manager coverage becomes usable only at the measured regime boundary. Coverage is measured on managers, not ticker count.
- 13F holdings are an S&P 500 slice; complete manager books provide the denominator for manager portfolio weights.
- Insider bulk history begins at the dataset's filing-date floor. Older transaction dates do not prove older publication coverage.
- RegSHO history is a moving provider window; the oldest stored anomaly is not a guaranteed recoverable boundary.
- SEC fails-to-deliver history has a fixed publication start.
- Schedule 13D/13G filer identity and event dates exist historically, while reliable ownership numerics begin with structured-data mandates.
- Missing data after a global source start can still be unavailable for a security, denominator, filing, or completeness frontier.

The canonical insider reader reconciles quarterly bulk files with the daily EDGAR tail at accession grain. Before a parity-approved quarterly promotion, the live copy wins overlap; afterward the bulk copy wins. Rows from the two representations are never mixed inside a filing. See the [TODO](../TODO.md) for unresolved historical coverage work.

## Governance and shareholder votes

DEF 14A prose uses table-aware section carving before LLM extraction. Structured Pay-versus-Performance facts are read directly from filing XBRL, including legitimate negative compensation-actually-paid values. A value is emitted only when deterministically recoverable; parsers never fabricate a replacement.

Item 5.07 is the certified shareholder-vote source and begins with its regulatory regime. Important semantics:

- one election row can aggregate many nominees while retaining nominee-level JSON;
- “against” and “withheld” are legally distinct standards;
- amendments are unioned unless evidence identifies a genuine restatement;
- validation monitors cannot reliably detect every column permutation, so source-text fabrication guards are mandatory;
- zero proposal rows can be a legitimate board-response filing rather than an extraction failure.

Both DEF 14A and vote extraction use the [SEC and LLM flow](../flows/sec-llm-extraction.md).

## Transcripts and attention data

Earnings-call history combines a deep HuggingFace backbone with recent-gap sources. Transcript availability is later than the earnings event, and call sections retain speaker/Q&A structure. Expensive per-call sentiment is cached separately from cross-call cube features.

Google Trends must be fetched in bounded windows before stitching because a long request can change its sampling frequency. Wikipedia and Google series have shorter usable histories than prices; long rolling windows therefore cover fewer observations.

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
