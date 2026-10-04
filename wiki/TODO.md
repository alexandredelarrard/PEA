---
title: TODO
description: Accepted backlog for source backfills and deferred data-contract work.
type: backlog
tags:
  - wiki
  - todo
  - data-quality
---
# TODO

Deferred work that requires new source history, authoritative eligibility evidence, or a broader data-contract change belongs here—not in production features with insufficient support.

## L/S point-in-time gates

### P0 — Purge the train-to-holdout label boundary

The [trainer](../src/modelling/steps/step_long_short.py) filters labelled rows by feature date through `train.end_date`, while the [L/S scorer](../src/strategies/utils/ls_model.py) starts at that same metadata date. The configured boundary is 2022-01-01 in [training](../configs/modellling.yml) and [portfolio](../configs/portfolio.yml). Forward 30/60/90-day labels near the training end can therefore contain returns from the nominal holdout; the per-fold CV embargo does not purge this separate boundary.

Before interpreting any backtest, make the first holdout decision strictly later than the latest return used by every training label (or remove the overlapping training rows). Test the boundary for every horizon and persist the effective label-end and first tradable holdout dates in run diagnostics.

### P0 — Rebuild historical universe and sector membership

The [price-part builder](../src/data_aggregate/transformers/step_cube_prices.py) uses the current S&P 500 roster for its historical grid, and [L/S sector neutralization](../src/strategies/step_ls.py) uses the current ticker-to-sector map. This can exclude former members and apply future classifications to past trades. Use a dated, point-in-time membership and sector history, reconciling the existing `sharadar_sp500` source before relying on it. Validate additions, removals, delistings, symbol changes, and the earliest eligible date per security; compare strategy results with the present survivor-only universe. Do not expand to Russell 1000 until this boundary is sound.

## Institutionals follow-up validation

- **P1 — Archive revisions and FTD timing.** Historical [FTD ZIP availability](./concepts/source-availability.md) dates are conservative estimates, not verified first-publication timestamps; a cached latest ZIP supplies only a short-lived local modification date. Test extra lags and an FTD-family ablation against the L/S holdout. Check whether revised source content can rewrite a pre-revision feature value; freeze prior published content if required by that evidence. Reassess the negative FTD monotonic constraint in [LightGBM configuration](../configs/models/lgbm_modelling.yml) against fold stability and economic meaning.
- **P1 — Full prefix replay and model re-evaluation.** Replay each institutional source family through the same historical prefix cutoff, then incremental-update it and prove pre-cutoff rows do not change except for an explicitly dated correction. Run the [institutionals quality driver](../scripts/institutionals_feature_quality.py) and a new model fit after the feature taxonomy changes. Keep correlated candidates for the model comparison, as agreed; remove one only on measured out-of-sample evidence.
- **P2 — Explain residual missingness.** Establish exact filing-discovery and absence frontiers for 13D/13G events; audit the remaining long interior hole runs one ticker and period at a time, including the five broad-13F coverage gaps. Separate source absence, ticker/CIK lineage, denominator ineligibility, and legitimate zero. Correct confirmed extraction or lineage defects and re-run coverage; never convert unknowns to zero solely to close a hole.



## Schedule 13D/13G ownership numerics

Reliable structured ownership values begin with the SEC mandate on 2024-12-17. The level and per-filer delta versions of `percent_of_class`, their cross-sectional transforms, and related power/aggregate fields remain excluded from `cube_part_institutionals`.

Restore them only after:

- parsing pre-mandate HTML/SGML cover pages point-in-time on `filing_date`;
- producing one canonical value per filing without summing repeated joint-filer values;
- reconciling pre-mandate output against overlapping XML and reporting disagreement rates;
- proving stable multi-year coverage across tickers and form variants;
- retaining enough history for representative training, validation, and test folds; and
- completing a new data-validation report.

Mean imputation over unavailable decades is not acceptable. Raw fields remain in the [source tables](./reference/table-catalog.md) for audit and future backfill.

## Pre-mandate beneficial-ownership event coverage

Investigate the roughly sixfold rise in Schedule 13D event rows around the structured-data mandate. Event-only features remain because filer identity and filing date exist before the mandate, but confirmed historical filing-discovery gaps must be backfilled before treating event intensity as time-comparable.

Acceptance evidence must separate a real regulatory/reporting shift from fetcher form-name or discovery coverage.

## Security-specific fails-to-deliver eligibility

The current FTD availability boundary is table-wide. Research and implement security-specific eligibility before interpreting pre-listing, post-delisting, suspended, or symbol-transition intervals as observed zero.

Required work:

- obtain authoritative primary-listing, delisting, suspension, and symbol-transition dates;
- reconcile those intervals with SEC FTD observations;
- keep vendor-backfilled price rows outside verified trading eligibility unavailable;
- build a 15-ticker edge-case table, including explicit treatment of the SMCI source-suspension gap; and
- prove in validation that no FTD feature appears before verified security eligibility.

Do not put these source-specific intervals in `symbol_tenure_manual.json`; that register has a different identity purpose.

## Point-in-time shares-outstanding split basis

Repair `fundamentals_history.sharesOutstandingPit` around stock splits before restoring guarded `ic_inst_ownership_pct` cells. The institutional rebuild on 2026-09-24 nulled 160 otherwise non-null cells above two times shares outstanding; bounded holes remained visible for AMCR, DD, DUK, and NVDA.

The numerator is present, but the point-in-time denominator is on an incompatible split basis. Clipping or forward-filling would fabricate ownership.

Required work:

- reconcile `sharesOutstandingPit` with split events and vendor-basis `sharesOutstanding`;
- backfill a corrected point-in-time denominator without applying future split knowledge;
- compare raw 13F shares against both denominator bases around every affected split;
- rebuild and validate `ic_inst_ownership_pct`; and
- remove the guard only if the corrected series is economically bounded and has no unexplained interior holes.

## EDGAR extraction follow-ups

Deferred from the shared-driver refactor of `src/data_extract`. Stored-value changes need explicit approval plus a DB cleanup; each item names the condition that should reopen it.

### Stored-value divergences (approval required)

- **P1 — Recycled-symbol aliases left foreign filings in stored tables.** `Company(alias)` resolves through EDGAR's *current* ticker map, so an identity alias of an unregistered ticker used to pull another company's filings. `resolve_registrant_filings` in [registrant.py](../src/data_extract/utils/common/registrant.py) now drops any listed filing whose CIK is outside the issuer lineage, so new runs are clean, but rows stored before that remain: measured in `sec_8k`, ALB holds 582 accessions from AllianceBernstein (CIK 825313), ALGN 72 from Allegro MicroSystems (866291), DASH 129 from Fabrinet (1408710), TFC 490 from Beacon Financial (1108134); `sec_def14a` ITW/BAC are clean. The DEF 14A LLM lister in [def14a/fetch.py](../src/data_extract/utils/structure/def14a/fetch.py) still has its own lister (one listing check found +31 Northern Trust proxies on ITW and +32 fund proxies on BAC under the old resolver). Trigger: before 8-K event features are trusted for these tickers; fix = delete the foreign accessions, then re-run the listing-equivalence check before moving the DEF 14A lister.
- **P2 — Schedule 13D blanks and reporting-person CIK.** `sec_13d.cusip` stores `''` on 3,685 of 4,791 rows where `sec_13g` stores NULL for the same default; 572 of 1,106 structured 13D rows lack `reporting_person_cik` because 13D has no header-CIK backfill (13G has one). Both live in the per-form `ScheduleSpec` in [schedule_rows.py](../src/data_extract/utils/institutionals/schedule_rows.py). Trigger: when a 13D feature reads `cusip` or filer CIK, or at the next 13D rebuild.
- **P2 — Insider live versus bulk encodings.** In [insider_common.py](../src/data_extract/utils/institutionals/insider_common.py): `value_usd` priority is reversed (bulk shares × price first, live stated total first; 6 disagreeing rows in 2026q1); role flags are 0/1 in bulk but NULL when the live checkbox is absent (about 1,700 live rows have NULL `is_director` where bulk has 0); a live row with no ticker stores `''`; and the numeric parsers differ (bulk pandas `to_numeric`, live Python `float` after stripping `,`/`$`, which disagree in the last digit on 5,678 of 40,000 random long decimals). Trigger: before the next parity-approved bulk promotion, or when an insider feature reads `value_usd` or role flags across the bulk/live boundary.
- **P3 — Text cleaning for 8-K and filing text.** `sec_8k` item text and `sec_filing_text` get no cp1252 normalisation (13D does) and skip the NUL strip. Trigger: when a text consumer meets mojibake or NUL bytes, or at the next full 8-K/filing-text rebuild.

### Fetch correctness and robustness

- **P1 — Live insider listing truncates silently on an Atom 503.** `ownership_filings` in [fetch_insider_edgar.py](../src/data_extract/utils/institutionals/fetch_insider_edgar.py) swallows an SEC 503 mid-pagination and returns a truncated list (observed: JPM Form 4 at offset 5100), so the run reports success with missing filings. Trigger: before the next insider live backfill; the page failure should fail the ticker.
- **P2 — `sec13f_hr` batch duplicate-key risk.** A save batch in [fetch_13f.py](../src/data_extract/utils/institutionals/fetch_13f.py) is de-duplicated on the book key `(cik, period, cusip)`; the `sec13f_hr` slice is cut after the CUSIP-to-ticker merge and is not re-checked on its own key `(cik, period, ticker, cusip)`, and Postgres rejects an upsert that touches one key twice. Trigger: any change to the CUSIP map shape or batch de-duplication, or a 13F batch upsert failure.
- **P2 — Insider stored-row sweep and mixed-CIK accessions.** The sweep in [fetch_insider_transactions.py](../src/data_extract/utils/institutionals/fetch_insider_transactions.py) deletes a whole accession that contains a rejected issuer CIK but quarantines only the rejected rows, so a kept row in a mixed-CIK accession is lost (0 such accessions live). Trigger: when the sweep first reports a mixed-CIK accession.
- **P2 — Two-table EDGAR fetchers never retry a failed secondary save.** `run_edgar_fetch` in [edgar_driver.py](../src/data_extract/utils/common/edgar_driver.py) builds its done set from `tables[0]` accessions, and `tables[0]` is saved first. If `sec_13d` saves and `sec_13d_transactions` fails, or `insider_transactions_live` saves and `insider_footnotes` / quarantine fails, the ticker fails and the manifest holds, but the accession is skipped from then on. Pre-existing. Fix = save `tables[0]` last, or intersect the stored accessions of every non-completion table (an accession may legitimately have no secondary rows, so this needs a design). Trigger: a secondary-table save failure in a log, or the next driver change.
- **P3 — 13F manager catch-up residuals.** In [fetch_13f_managers.py](../src/data_extract/utils/institutionals/fetch_13f_managers.py): a deterministic-broken filing that is the newest of its period is re-read and logs the same ERROR on every run (no side state records it); and two filings of one period filed the same day share one (period, filing date) pair, so if the main walk saves one and the other fails transiently, both count as done. Trigger: the same ERROR repeating across runs, or a same-day same-period manager filing pair.
- **P3 — Live-test tripwire scope.** `tests/live_guard.py` does not see asyncio Proactor connects (`ConnectEx`), raw DB-API cursors from `engine.raw_connection()`, or `socket.getaddrinfo` DNS lookups. Nothing in `src`, `tests` or `scripts` uses these paths today. Trigger: the first async HTTP client, raw cursor or new DNS-dependent test; fix = patch them or document the limits in the guard docstring.
- **P2 — edgartools Windows cache rename race.** Threads building `Company` for the same CIK can hit a `PermissionError` on the edgartools cache rename. Trigger: when a threaded EDGAR walk logs that error, or before raising worker counts.
- **P3 — `ensure_zip` limiter and close.** `ensure_zip` in [bulk_cache.py](../src/data_extract/utils/common/bulk_cache.py) and the notes scrape bypass the SEC rate limiter, do not close a streamed response on a non-200, and a mid-download exception aborts the loop. Trigger: if a bulk download is throttled or leaks connections.

### Performance and code health

- **P3 — Memoised per-run EDGAR listing.** Each fetcher calls `Company(ticker).get_filings` per ticker (five or more listings per ticker per run). Memoise per run once a memory budget is set; resident memory is the risk. Trigger: when listing time dominates a nightly run.
- **P3 — Parallel RegSHO and earnings-surprise downloads.** RegSHO day files (about 3,900 on a cold start) and per-ticker earnings surprises download sequentially. Trigger: a cold start or full refresh of either source.
- **P3 — `xbrl_linkbase._resolve_once` is 220 lines.** [xbrl_linkbase.py](../src/data_extract/utils/fundamentals/xbrl_linkbase.py) sits outside the EDGAR walk. Trigger: the next functional change to linkbase resolution.
- **P3 — `isin` on Arrow string columns.** pandas `isin` on Arrow-backed string columns is about 15× slower than on object columns (for example `filter_footnotes`). Trigger: when a bulk insider quarter's filter step regresses or the frame dtype backend changes.
- **P3 — Unused `pytrends` dependency.** Nothing imports it; remove it from `pyproject.toml` (line 29) and `airflow/requirements-airflow.txt` (line 16) with a `poetry lock`. Trigger: next dependency update.
- **P2 — KPI catalogue leaf/evidence check covers only the last ticker.** In `_catalogue_at` in [kpi_catalogue.py](../src/data_extract/utils/fundamentals/kpi_catalogue.py), the leaf/not-leaf and `evidence` checks sit after the per-ticker loop, so each register checks only its last ticker block (2 of 19 blocks checked; 0 violations today, so the gap is latent). Trigger: the next edit to the KPI register or a new ticker block; fix = move the check inside the loop.
- **P3 — Sharadar gap check builds TTM without Yahoo splits.** `gap_check` in [gap_check.py](../src/data_extract/utils/fundamentals_sharadar/gap_check.py) calls `build_ttm` without `yf_splits`, while the merge in [merge_history.py](../src/data_extract/utils/fundamentals_sharadar/merge_history.py) passes both vendors' splits, so the gap report can disagree with the merged history on split-affected tickers. Trigger: when a gap report flags a split-affected ticker the merge handles correctly.
- **P3 — Stale comments in risk-zone files.** [schema.py](../src/data_store/schema.py) (about line 888) names the deleted `_filter_universe`, and [context.py](../src/context.py) (about line 125) names `kpi_catalogue.resolve_config_dir`, now in `config_paths`; [store.py](../src/data_store/store.py) (about line 384) cites the retired `fetch_hf_transcripts` as an example. Trigger: the next approved edit to any of these files.
- **P3 — `src/utils/crawler.py` has no source consumer.** Its last callers (Google Trends, wiki pageviews, the retired earnings-call crawlers) are gone; only `tests/utils/test_crawler.py` imports it, and the [polite_http.py](../src/utils/polite_http.py) module docstring still says Google Trends shares `resolve_proxy` through it. Trigger: the next scraper that needs IP rotation reuses it, or delete it with its test and fix that docstring.
- **P3 — Minor simplifications from the extraction refactor review.** Behaviour-neutral unless noted; take each with the next edit to its file.
  - [edgar_driver.py](../src/data_extract/utils/common/edgar_driver.py): inline the one-use `_build_ticker` hop into `_walk_ticker`; move `num_or_null` (13D/13G only) into [schedule_rows.py](../src/data_extract/utils/institutionals/schedule_rows.py); let `manifest_window` take the already-loaded manifest entry instead of reading the JSON twice (also in the DEF 14A lister).
  - [fetch_13f.py](../src/data_extract/utils/institutionals/fetch_13f.py): collapse `_record`'s if/else (only `backfill_window` differs); delete `position_type` (test-only caller); share one suspect-price warning with the manager catch-up and drop the leading underscore on the names `fetch_13f_managers` imports.
  - 13D/13G: derive `fetch_13g_edgar._COLS` from the 13D columns; run 13D frames through `finalise_frame` like 13G (low risk: PK dedup); build `_ITEM_HEADING_ANYWHERE` from `item_heading` (low risk); one cached `FilingStamp.text` for the best-effort text read in 13D and filing text.
  - Insider: one `source` profile (bulk/live) instead of the `value_rule` / `numeric_rule` / `date_formats` triple in [insider_common.py](../src/data_extract/utils/institutionals/insider_common.py); drop the two redundant empty-frame conditionals in `fetch_insider_edgar.py`.
  - Elsewhere: delete the unread `on_date` parameter of `Identity.owns`; vectorise `fundamentals_employees._done_dates`; use `Counter` for the vote tallies in `votes/fetch.py`; one `bulk_cache` helper for the period-archive fetch shared by financial statements and notes (keep each caller's clock precedence); `pad_cik_series` in `fetch_tickers.py` (low risk: odd inputs); read `manifest_full_rescan_days` directly in `def14a/fetch.py` instead of a re-declared default.
  - Constants (needs approval): move `SEC_13D_FORMS`, `SEC_13G_FORMS`, `SEC_INSIDER_FORMS` and `SEC_8K_FORMS`, each with one `src` consumer, next to their fetchers.

## Earnings-call tone drift over time

Transcript language drifts over the years, and nothing in the earnings-call features removes it. Measured on a stratified sample of 2,940 calls (140 per year, 2006–2026, 473 tickers), comparing 2006–10 with 2021–25:

- positive words rise in every role by 1.0 to 2.1 per 1,000 words per decade, and negative words fall by 0.5 to 0.9;
- courtesy sentences in analyst questions go from 8.6 to 13.9 per 1,000 words, as a step (2.6–3.5 in 2011–13, then 12–15 from 2015), which looks like a change in transcription style rather than in behaviour (unverified);
- courtesy in answers goes from 1.6 to 2.4 and in prepared remarks from 1.25 to 1.64 per 1,000 words.

The lexicon counts use a 40-word subset of each Loughran-McDonald list, not FinBERT; courtesy is now removed before scoring, but the positive/negative word drift is in the substantive text. The `f_ec_*` features have no cross-sectional normalization by design, so a common upward drift reaches the raw levels, and the prior-only issuer-history scores (`*_vs_hist`, 5-year window) carry it too: every issuer's current call reads above its own history. Evidence: `reports/validate/2026-10-01-earnings-call-extraction-gaps/01-research.md` (text-noise follow-up) and `_out/clean_politeness_by_year.png`.

Reconsider when any of these holds:

- full FinBERT scoring is done and the date-mean of `f_ec_tone` or `f_ec_tone_vs_hist` shows a trend over years;
- the earnings-call validator raises a PSI drift warning (`drift_psi_warn`) on a tone feature; or
- a model review finds the tone features' importance or sign unstable across CV folds by era.

Candidate fixes to compare then: a per-date cross-sectional demeaning of tone only, or a time-detrended issuer history. Keep the 12-column contract unless the measurement justifies a change.

## Documentation retirement

After this migration is reviewed:

- confirm every root instruction and package manifest resolves to [the wiki](./OVERVIEW.md);
- confirm the OpenKnowledge audit has no links into the retired documentation tree;
- delete only legacy docs that are fully represented here;
- keep this TODO as the accepted backlog; and
- append the retirement event to [wiki log](./log.md).

## Related

- [Data sources](./reference/data-sources.md)
- [Live database](./reference/live-database.md)
- [Source availability](./concepts/source-availability.md)
- [Cube aggregation](./modules/data-aggregate.md)


# extraction data
- short interest : extract also the Lit exchange NYSE /NASDAQ, from 2009 for all (now is 2018)
- insiders trading : sec form 3/4/5 -> sec since 2003, zip since Q1 2006 -> take sec. Done for the latest ones. So should be quick + add the Tickers fixed in gov
- earnings surprises starts 1999-08, but empty till ~2003
- financial notes (text & nums) 2009 from sec XBLR (zip), but possible directly from fillings (edgar)
- fix volume to be adjusted to spinoffs in price
- fix EC extraction history and gaps. 2/3 are available today vs 90% potential
- fix employee count, lots of MVs and gaps. Play with LLM extract.
- fine tune def 14 data extraction

# other data checks
- check data is consistent over time, even for latest 2026 month ?
- check dag data extract and data aggregate run smoothly end to end
- Review the price split / spinoff over time (price validation)
- refine modelling to be as stable as possible
- review periods when IC drops for few weeks / months
- add other strats decorrelated : - Super investors replica ?

# # Signals :
- include move from peers when new results are available -> move all peers info
- create variables based on clients stock move, geography graph, etc.

# universe expand:
- After the point-in-time S&P 500 gate above, evaluate Russell 1000 expansion with the same membership and delisting controls.

## Extraction resume reads only the database

**P1 — make it a separate plan and refactor.** Today each fetcher decides what to fetch in its own way:

- the run manifest (`extraction_manifest.json`) gives the EDGAR walks, the DEF 14A LLM fetch and the earnings-call fetch their listing window, ticker set and 30-day full rescan;
- the `{table}_universe.json` marker files do the same for pension, Notes, FTD and insider bulk;
- the other fetchers read a single last-date frontier.

The target: every fetcher reads only its own tables, the source's own listing and `entity_lineage`. A missing table means a full build. Holes, missing filings and new tickers are found on every run. A repair means deleting the wrong rows, and the next run fetches them again.

Defects to fix (read-only survey of 33 fetchers, 2026-10-03; the per-fetcher table is in `reports/validate/2026-10-02-entity-symbol-lineage/_out/resume_survey.md`):

- **Holes behind the frontier are never found:**
  - prices older than 7 sessions;
  - dividends, splits;
  - Sharadar fundamentals, actions, S&P 500;
  - short interest, 13F, 13F managers;
  - earnings surprises.
- **New tickers:**
  - 13F, short interest, splits (1 year only) and dividends never backfill a new ticker;
  - prices re-pulls the whole universe when one ticker is new.
- **Zero-row units are redone every run:** a 5.07 votes filing with no rows is sent to the LLM again; a ZIP period with no universe row is re-parsed.
- **"Done" read from the wrong place:**
  - 13D treats an accession as done once it is in `sec_13d`, even when its transactions failed to save;
  - Notes treats a period as done when either `notes_num` or `notes_text` has it.
- **`fetch_13f` writes a new manager's book inline,** which sets that manager's frontier before `fetch_13f_managers` has fetched its history.
- **A `-t` subset run overwrites the stored ticker set** in the manifest and the marker files, so the next full run relists or re-parses everything.
- **Full-table reads** in earnings surprises, the DEF 14A LLM fetch, the CUSIP map and the Sharadar roster.

Design proposed for the plan:

1. **Two store reads.** `frontier(table, by, date_col, where)` returns MIN, MAX and COUNT per key in one GROUP BY query. `keys(table, cols, where)` returns the stored keys for a scope.
2. **One planner, `resume.py`.** Each fetcher declares its table, key columns, date column and mode. The planner returns the work items: full history for a new key, the forward window, holes, and the overlap re-fetch.
3. **Three modes:**
   - **Listed by the source** (EDGAR filings, ZIP periods, 13F filings, earnings calls, votes): fetch what the source lists minus what is stored. `Company(cik).get_filings()` already downloads the full history on every call, so the EDGAR window and the 30-day rescan can go.
   - **Daily series** (prices, macro, short interest, Sharadar): the calendar gives the expected dates, so a row count below it reveals a hole.
   - **Sparse events** (dividends, splits): fetch forward from the last date checked.
4. **Recording "checked, nothing found".** Options: a NULL placeholder row in each data table; one `extract_coverage(table_name, unit, n_rows, checked_through, checked_at)` table that would absorb `insider_transactions_live_coverage` (the recommended option); or recording nothing.

Acceptance:

- A source test finds no resume decision that reads the manifest or a marker file.
- Each mode has fixtures for: a missing table, a new key, the forward window, an interior hole, an old gap in a source-listed unit, a zero-row unit, and a `-t` run followed by a full run.
- A read-only comparison on the live database shows each fetcher's new plan against its old rule.
- `EXPLAIN` on the largest tables shows each frontier query is indexed.
