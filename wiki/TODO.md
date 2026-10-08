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
- **P2 — Late insider filings count as fresh news.** Insider features are stamped on `filing_date`, so a trade reported months late (a Form 5 or a late Form 4, e.g. an August 2005 purchase filed in March 2006) enters every trailing window and decayed leg at full weight on its filing day, exactly like a trade done that week. This is point-in-time correct (nothing before the filing date sees it) but may dilute the signal: a 7-month-old purchase says less about the next quarter than last week's. Nothing in [insider_quality.py](../src/data_aggregate/utils/institutionals/insider_quality.py) or [insider_features.py](../src/data_aggregate/utils/institutionals/insider_features.py) filters on the reporting lag `filing_date - transaction_date`. To do: (1) measure, on the scoped P/S rows of the refilled `insider_transactions`, the share of rows and dollars by lag bucket (≤ 2 business days, ≤ 30, ≤ 60, > 60 days) per year and by form (4 vs 5); (2) build two candidates on the same publication clock — a filter that ignores records whose lag exceeds N days (N = 30 or 60), and a separate `ic_insider_late_filing_share` feature that leaves the sums unchanged; both read only `filing_date` and `transaction_date`, known on the filing day, so neither can leak; (3) keep a change only if it improves out-of-sample IC and fold stability against the current features on the L/S holdout. Trigger: after the insider refill, before the next insider feature review.
- **P2 — Explain residual missingness.** Establish exact filing-discovery and absence frontiers for 13D/13G events; audit the remaining long interior hole runs one ticker and period at a time, including the five broad-13F coverage gaps. Separate source absence, ticker/CIK lineage, denominator ineligibility, and legitimate zero. Correct confirmed extraction or lineage defects and re-run coverage; never convert unknowns to zero solely to close a hole.
- **Handled — PSKY short volume under PARA/PARAA.** About 400 PSKY sessions were missing from `sec_short_interest` because the old symbol resolver had no PARA/PARAA tenure for PSKY. The security master maps PARA (CUSIP 92556H206, class B, `secondary_class`) and PARAA (92556H107, `canonical_predecessor`) to PSKY's predecessor issuer 0000813828 and sums both; no manual entry is needed. Re-count PSKY's days after the cutover's `short-interest -F`.
- **Handled — false second symbol tenures.** Each is now dropped at build by a cited `rejected` entry in [symbol_tenure_manual.json](../configs/sec/symbol_tenure_manual.json) (the build logs `rejected F/0000712537 ...`, `rejected AXP/0001387156 ...`), and short volume resolves through the security master, so it no longer reads them. The cutover's `short-interest -F` supersedes `--repair-gaps`; re-count the gaps after it. The original finding: **False second symbol tenures drop short-volume days.** `symbol_tenure` holds tenures derived from a few mis-tagged Form 3/4/5 filings, which make the RegSHO resolver read a universe symbol as `ambiguous` and drop its rows: F -> CIK 0000712537 (First Commonwealth Financial), 2018-04-26..2019-04-25, 6 filings, which costs Ford 183 sessions (2018-08-01..2019-04-24); AXP -> AirXpanders (CIK 0001387156), 2017-07-07..2018-09-05, 20 filings; and 1-2 day tenures on NTRS (Illinois Tool Works), PM (FAC Propertys), AME (AMC Entertainment) and SPG (SpyGlass Pharma). After the one-time `short-interest --repair-gaps` of 2026-10-05, 1,120 ticker-days (18 tickers) are still missing: PSKY 400 (above), SMCI 349 (absent at FINRA), F 183, EXE 156, and 1-4 days on ticker-change dates.
- **P2 — One joint 13F filing sits under two managers.** In `sec13f_hr`, the 2015-12-31 books of CIK 0001622431 and 0001054880 are the same 35 rows (35 of 35 equal in shares and value), filed 2016-02-12; 0001622431's 2015-06-30 and 2015-09-30 books mix its own filings (filed 2015-08-10 and 2015-11-12) with rows of that 2016-02-12 filing. Its own 2015-12-31 filing held 3 rows. The 2026-10-05 CIK normalisation (stage B, keep the later-filed row per key) replaced 2 of those 3 rows (VZ, XOM) with the joint book's values. The rows as they were are in the run dir export `_out/phase11/state_before_B/sec13f_hr_affected_rows.csv`. Fix: attribute a joint 13F-HR to its filer only (or split it by the other-manager list), then delete the misattributed rows and refetch both managers' 2015 periods. Trigger: before a superinvestor or 13F feature reads these managers' 2015 books.



## Entity-symbol-lineage branch: re-runs after the resume merge

Folded into the [identity cutover](./guides/run-the-pipeline.md#identity-cutover-one-off-user-run): Stage C is its step 4 (LLM stop raised to $50), the false tenures and PSKY are handled (entries above), the per-ticker re-runs are its rebuilds and gate diff, and the second AC-003 check is an after-step. Only the earnings-call item stays open after it. The `2026-10-02-entity-symbol-lineage` branch takes `dev` after `feat/db-derived-resume` merges, fixes the identity gaps, re-runs the affected tables and diffs them. Context and evidence: the final report, in the local, gitignored run dir `reports/validate/2026-10-03-db-derived-extraction-resume/05-final-report.md`.


- **P2 — 12 tickers with no stored earnings call:** ED, EXPD, FDXF, GEHC, GEV, HONA, KVUE, NVR, Q, SNDK, SOLV, VLTO. Check ED, EXPD and NVR first; they are old companies.
- **Retired — second AC-003 check.** `_out/old_rule_plan.py` replays the pre-lineage `scope.registrants` listing that the identity layer removed, so it cannot run on the merged code, and any listing difference would be the intended identity change, not a resume regression. The post-cutover `resume-plan` is the substitute evidence.

## Insider transactions follow-ups

Deferred from the single-table merge of `insider_transactions` (findings F-007..F-014 and R-04 in the local, gitignored run dir `reports/validate/2026-10-03-insider-transactions-merge/defects.md`).

- **P2 — Amendment double counts (R-04).** Linked amendment rows never enter the copy collapse in [insider_quality.py](../src/data_aggregate/utils/institutionals/insider_quality.py). Measured on the refilled table (all history, scoped P/S): (a) joint reporters who each file a value-changing 4/A count twice from the 4/A day, 81 rows / $449m as filed; (b) a linked 4/A adding a cell a co-owner already reported counts twice, 194 rows / $763m over 5,737 new-cell rows (e.g. NTRS `0000073124-06-000091` vs `-000072`). Small next to the 6,813 copies the collapse removes, but material in dollars. Trigger: the next insider feature review, or before `buy_value` features are re-tuned.
- **P2 — An undownloadable zip quarter hides under a complete frontier (F-007).** `fetch_insider_transactions` in [fetch_insider_transactions.py](../src/data_extract/utils/institutionals/fetch_insider_transactions.py) skips a quarter that no URL serves with only per-URL WARNINGs; the EDGAR window then starts after the last stored quarter, so nothing but the stored quarters shows the hole. Fix: fail or WARN-summarise a skipped non-newest quarter, and/or hold the frontier at the last quarter end before it. Trigger: the next zip-path change at the SEC, or any refill.
- **P3 — Repeat collapse merges different people opening equal new positions (F-011).** `_collapse_copies` keys on (ticker, day, code, shares, price, holding); when the holding equals the shares bought it cannot tell two new positions apart. DDOG IPO day 2019-09-23: two directors' purchases collapse into two others', so `distinct_buyers_120d` reads 4 instead of 6; about 6 zero-start groups / $482m since 2020, some legitimate (TKO). Fix idea: a zero-start group collapses only when the accessions share an owner or filer agent. Trigger: the next insider feature review.
- **P3 — REQ-007 WARNING overstates EDGAR losses (F-012).** `report_zip_quarter` counts N from the quarter-wide earliest EDGAR filing date, while EDGAR listing is per ticker since the merge; after a rebuild, a `-t` run or a skipped quarter the WARNING inflates (2026q2 replay 71.0 %). Fix: N per ticker. Trigger: the next time the WARNING is used to judge EDGAR completeness.
- **P3 — Half-cent tolerance in the mismatch line (F-013).** `_same` compares `gap.le(0.005)` on floats, so the zip's half-up rounding of a half-cent EDGAR value (116.695 vs 116.70) counts as a mismatch: 65 of 66 in the 2026q2 replay. Fix: compare `round_half_up(edgar)` with the zip value, or allow a few ulps. Trigger: with F-012.
- **P3 — `validate pull` fails on `insider_transactions` (F-014).** `_coerce` in [io.py](../src/validate/io.py) fixes a float policy for `footnote_ids` from an all-null first chunk (zip rows come first), then fails on `'F5,F6'`. Fix: decide the policy on the first non-null value or from the DB column type. Trigger: the next validation-library run on an insider table.

## Superinvestor roster follow-ups

Follow-ups from the verified point-in-time roster run (`harness/sec13f-superinvestor-roster`; local ignored evidence `reports/validate/2026-10-04-sec13f-superinvestor-roster/`) and the 2026-10-07 legacy parser audit (`reports/validate/2026-10-07-superinvestor-legacy-p1/`). See [Superinvestor roster (Dataroma)](./reference/data-sources.md#superinvestor-roster-dataroma) and the [refresh procedure](./guides/run-the-pipeline.md#superinvestor-roster).

- **P2 — Recover missing manager quarters and resolve ghost periods (R-06).** The finite audit still has **228 never-stored books across 19 CIKs**: **15 in the feature window** and 213 earlier. Prioritize `0000098758` and `0000846222` (each quarter from 2011Q3 through 2013Q1), and `0001099281` (2012Q3). These 228 had no immutable local source cache in this audit; that is not evidence that SEC sources are unavailable. **Next session:** acquire and freeze the original/amendment filings for those 15 gaps, verify complete books with the merged parser, and separately check every non-quarter-end period against its SEC cover before correcting it. Keep Greenhaven/pre-XML recovery in this scope. Do not invent placeholder positions or normalize ghost dates without source evidence. These never-stored gaps are separate from the 90 stored unresolved books below. Owner: `harness/13f-managers-cusip`.
- **P2 — Rebuild and validate superinvestor features after the data cleanup.** No cube rebuild has been performed. The audit lacked `cube_part_prices`, so production feature-impact certification abstained; the raw-price counterfactual is not production certification. **Next session, after the P1 repair:** build the missing price part, fully rebuild `cube_part_institutionals` with `build-institutionals -F`, then run `assemble-cube` and the relevant read-only [validation](./guides/validate-a-change.md). Check full-book weight denominators, concentration, manager eligibility and filing/availability clocks before using the repaired features. Commands are owned by [aggregate CLI](../src/data_aggregate/cli.py).

- **P2 — `fetch_13f_managers` code health.** Its private cross-imports from `fetch_13f.py` and the dual-padding lookup in `_warn_empty_books`. Trigger: after `feat/db-derived-resume` merges (it rewrites `fetch_13f.py`).
- **P3 — Optional recovery of 90 unresolved stored books.** The 2026-10-07 audit counted 90 / 7,369 stored roster-manager books (**1.22%**; one CIK/report-period per book): 81 before 2011Q3 and 9 within the feature window. This includes already NULL/unavailable books, not 90 new corrupt books. **User decision: defer further source recovery while this share is 2% or less.** This does not defer the P1 cleanup of corrupt stored amounts. If reopened, recover immutable original/amendment sources and require complete, source-verified books; retain unavailable amounts when verification fails. Trigger: the unresolved share rises above 2% or the user explicitly requests recovery.
- **P3 — `step_super_investors` survivorship.** The replication sleeve builds its history from today's roster (expanded to chain members), not the roster at each date. Pre-existing.
- **P3 — `lmvtx` reads a diluted book after 2018Q3 (F-002 / F-010).** The `0000820330` chain in [overrides.json](../configs/superinvestors/overrides.json) is right on identity: ClearBridge Investments `0001348883` holds the Legg Mason Value Trust book from 2018-09-30 (the fund's N-CSR records the advisory agreement moving on 2018-07-01). But the fund (about $2.3bn) is about 2% of that filer's ~1,125-position, ~$116bn book, and no 13F filer isolates it. So over the 16 snapshots from 2018-12-31 to 2022-09-30, `lmvtx` adds a diversified holder on about 1,100 names to breadth, ownership-count and `continuous`-weight features. Options: an `inactive` range for `lmvtx` from 2018-09-30, or excluding the code. Today `inactive` only exempts a code from the activity gate and readers still use its CIK, so either option also needs the roster readers to drop the code in that range. Owner: user policy decision.
- **P3 — Successor CIKs log without a name.** `fetch_13f_managers` looks log names up by filer CIK, so a chain successor logs nameless. Cosmetic.
- **P3 — Stale `schema.py` comment.** The `superinvestor_roster` comment in [schema.py](../src/data_store/schema.py) (lines 104–121, "13 captures") predates the quarterly history. Edit when `schema.py` is next touched (guarded zone).
- **Trigger — Holdings-fingerprint resolver.** Match a Dataroma holdings page against 13F books to resolve a code mechanically. Build it if a refresh leaves more than two codes unresolved.

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

## Schedule 13G holder identity

- **P1 — Re-registered holders read as new holders (`ic_bo_new_holder`).** The feature keys a holder on the raw filer CIK, so a large holder that changes its SEC registration shows up as a brand-new holder on every stock it owns. BlackRock moved to CIK 0002012383 (88 false new-holder events in 2024, 154 in 2025); Vanguard split into Vanguard Capital Management (487 tickers) and Vanguard Portfolio Management (170 tickers) in 2026. The share of 13G filings counted as a new holder goes from about 7 % a year (2018–2022) to 21 % in 2025 and 40 % in 2026 (figures from the 2026-10-06 ticker-coverage session, not re-measured). This is holder-side identity; the issuer-side `entity_lineage` does not cover it.
  - Fix: a curated, dated holder map in `configs/` (holder CIK -> canonical holder id, `valid_from`, cited evidence), one-to-many for splits such as Vanguard; `ic_bo_new_holder` counts a holder as new only when the canonical id never held the stock before. Reuse the superinvestor roster's filer-chain key (manager ID = oldest CIK of the chain).
  - Detection: flag a holder CIK that appears on hundreds of tickers in the quarter an established holder disappears (a `missing_cutover`-style action item), so the next re-registration is caught.
  - Check: the new-holder share of 13G filings returns to the 2018–2022 level; needs an aggregate rebuild (user-run).
  - Trigger: before the next model retrain that uses `ic_bo_new_holder`.

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

- **Closed — recycled-symbol aliases (F-P8-1).** Listing is by CIK only and `identity-propagate` purges the stored foreign rows; the cleanup itself is the identity cutover under [Entity and symbol lineage follow-ups](#entity-and-symbol-lineage-follow-ups).
- **P2 — Schedule 13D blanks and reporting-person CIK.** `sec_13d.cusip` stores `''` on 3,685 of 4,791 rows where `sec_13g` stores NULL for the same default; 572 of 1,106 structured 13D rows lack `reporting_person_cik` because 13D has no header-CIK backfill (13G has one). Both live in the per-form `ScheduleSpec` in [schedule_rows.py](../src/data_extract/utils/institutionals/schedule_rows.py). Trigger: when a 13D feature reads `cusip` or filer CIK, or at the next 13D rebuild.
- **P3 — Text cleaning for 8-K and filing text.** `sec_8k` item text and `sec_filing_text` get no cp1252 normalisation (13D does) and skip the NUL strip. Trigger: when a text consumer meets mojibake or NUL bytes, or at the next full 8-K/filing-text rebuild.

### Fetch correctness and robustness

- **P2 — `sec13f_hr` batch duplicate-key risk.** A save batch in [fetch_13f.py](../src/data_extract/utils/institutionals/fetch_13f.py) is de-duplicated on the book key `(cik, period, cusip)`; the `sec13f_hr` slice is cut after the CUSIP-to-ticker merge and is not re-checked on its own key `(cik, period, ticker, cusip)`, and Postgres rejects an upsert that touches one key twice. Trigger: any change to the CUSIP map shape or batch de-duplication, or a 13F batch upsert failure.
- **P2 — Insider stored-row sweep and mixed-CIK accessions.** The sweep in [fetch_insider_transactions.py](../src/data_extract/utils/institutionals/fetch_insider_transactions.py) deletes a whole accession that contains a rejected issuer CIK and counts only the rejected rows in its warning, so a kept row in a mixed-CIK accession is lost silently (0 such accessions live). Trigger: when the sweep first reports a mixed-CIK accession.
- **P3 — 13F manager catch-up residuals.** In [fetch_13f_managers.py](../src/data_extract/utils/institutionals/fetch_13f_managers.py): a deterministic-broken filing that is the newest of its period is re-read and logs the same ERROR on every run (no side state records it); and two filings of one period filed the same day share one (period, filing date) pair, so if the main walk saves one and the other fails transiently, both count as done. Trigger: the same ERROR repeating across runs, or a same-day same-period manager filing pair.
- **P3 — Live-test tripwire scope.** `tests/live_guard.py` does not see asyncio Proactor connects (`ConnectEx`), raw DB-API cursors from `engine.raw_connection()`, or `socket.getaddrinfo` DNS lookups. Nothing in `src`, `tests` or `scripts` uses these paths today. Trigger: the first async HTTP client, raw cursor or new DNS-dependent test; fix = patch them or document the limits in the guard docstring.
- **P2 — edgartools Windows cache rename race.** Threads building `Company` for the same CIK can hit a `PermissionError` on the edgartools cache rename. Trigger: when a threaded EDGAR walk logs that error, or before raising worker counts.
- **P3 — `ensure_zip` limiter and close.** `ensure_zip` in [bulk_cache.py](../src/data_extract/utils/common/bulk_cache.py) and the notes scrape bypass the SEC rate limiter, do not close a streamed response on a non-200 (F-010: now reached for every insider quarter that falls back to its second SEC path), and a mid-download exception aborts the loop. Trigger: if a bulk download is throttled or leaks connections.

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
  - [edgar_driver.py](../src/data_extract/utils/common/edgar_driver.py): inline the one-use `_build_ticker` hop into `_walk_ticker`; move `num_or_null` (13D/13G only) into [schedule_rows.py](../src/data_extract/utils/institutionals/schedule_rows.py).
  - [fetch_13f.py](../src/data_extract/utils/institutionals/fetch_13f.py): collapse `_record`'s if/else (only `backfill_window` differs); delete `position_type` (test-only caller); share one suspect-price warning with the manager catch-up and drop the leading underscore on the names `fetch_13f_managers` imports.
  - 13D/13G: derive `fetch_13g_edgar._COLS` from the 13D columns; run 13D frames through `finalise_frame` like 13G (low risk: PK dedup); build `_ITEM_HEADING_ANYWHERE` from `item_heading` (low risk); one cached `FilingStamp.text` for the best-effort text read in 13D and filing text.
  - Insider: `date_formats` is the last per-source argument of `build_insider_frame` in [insider_common.py](../src/data_extract/utils/institutionals/insider_common.py); fold it into the source; drop the two redundant empty-frame conditionals in `fetch_insider_edgar.py`.
  - Elsewhere: delete the unread `on_date` parameter of `Identity.owns`; vectorise `fundamentals_employees._done_dates`; use `Counter` for the vote tallies in `votes/fetch.py`; one `bulk_cache` helper for the period-archive fetch shared by financial statements and notes (keep each caller's clock precedence); `pad_cik_series` in `fetch_tickers.py` (low risk: odd inputs).
  - Constants (needs approval): move `SEC_13D_FORMS`, `SEC_13G_FORMS`, `SEC_INSIDER_FORMS` and `SEC_8K_FORMS`, each with one `src` consumer, next to their fetchers.

## Entity and symbol lineage follow-ups

Deferred from the entity-safe EDGAR filing scope and dated lineage work (`reports/validate/2026-10-02-entity-symbol-lineage/`). The flag list is regenerated by `validate identity` (`identity_flags.csv`); act on it in `configs/sec/*.json` as described in [data sources](./reference/data-sources.md).

- **Done 2026-10-06/07 — identity cutover.** Run on the live database from the [cutover runbook](./guides/run-the-pipeline.md): new identity tables, security master (1,678 rows, 500 companies), per-security tapes and insider lineage columns; purged sec_8k 9,070, fundamentals_facts 31,280, sec_8k_votes 179, sec_13d 9, sec_13g 52, insider_transactions 1,338 rows. A live defect (`store.copy_load` read `''` as NULL in NOT NULL text keys) was fixed during the run.
- **P2 — Restatements never reach the prior-year side of YoY features.** An amendment filed within 365 days emits a row with `as_of` = its filing date and `fiscal_end` = the latest known period (the restated period goes to `amended_fiscal_end`); one filed later emits nothing (`MAX_AMENDMENT_LAG_DAYS`). `pit.fiscal_prior_positions` matches the prior-year row by `fiscal_end`, so it always reads the original, frozen prior-year TTM: when a restatement touches the prior-year window, YoY compares a restated current TTM with an unrestated prior one. Look-ahead-safe, but a basis mismatch. Option: let an amendment also publish a version of the restated period itself (`fiscal_end` = amended period, `as_of` = filing date) so later YoY comparisons pick it up; this changes the history grain and every YoY feature, so it needs a decision and an aggregate rebuild.
- **Fixed 2026-10-07 — filer-typed period of report.** The SEC header of 14 filings names a period earlier than the quarter or year they report (MRVL's Q1 FY2019 10-Q states 2018-02-03, VTRS's Q1 2012 10-Q states 2011-03-30, BKR's Q1 2018 10-Q states 2017-03-31), so their rows landed on the wrong `fiscal_end`. The history builder now takes the filing's own quarterly or annual duration end when it ended by the filing date and is more than 7 days after the header; forward-tagged contexts (FDS, GD) are ignored. The 14 tickers were rebuilt.
- **P1 — `fundamentals-history-sec` replay cost.** Measured during the cutover: about 4.5 to 6 min per ticker, and 2.5 to 3.5 h for multi-filer histories (APO, DUK, MRVL, XOM), so a universe `--rebuild-history` is a multi-day job and the nightly `fundamentals` task, which calls the same builder for every ticker, cannot finish in a night unless most tickers short-circuit. Measure the nightly path, then parallelise per ticker or cache the replay (see the replay-cost memory). The cutover rebuilt only the 40 tickers whose facts or windows changed; the other 451 keep their pre-cutover history (no identity input changed for them). Trigger: before the DAGs are re-enabled.
- **Done 2026-10-08 — `fundamentals_employees` identity and coverage (the P1 identity entry and its P2 coverage twin; merged into `dev` 2026-10-08 as d8d3dc02).** The table carries the filer `cik`, `accession_number`, form, components, basis, status and quotes, and `identity-propagate` purges it. On the 50 focus tickers (4 with no rows, 46 stale): missing owned original 10-K dates 562 → 0; 48 of 50 have a 2025+ count (IR and PEG do not, below); foreign rows 0; a 30-row quote audit passed 29/30, 30/30 with the table's "(in thousands)" scale. The rest of the universe is user-run ([runbook](./guides/large-backfills-and-recovery.md#employee-headcount-universe-run)); open items are under [Employee headcount follow-ups](#employee-headcount-follow-ups).
- **Handled — symbol conflicts no longer drop FTD history.** FTD and short volume resolve per security through `security_master` (by CUSIP), and `market_boundaries` in `security_master_manual.json` curate APTV/DLPH, NWSA, FOXA, EXE and WBD. Check those tickers' FTD row counts in the cutover's gate diff.
- **Handled — MRK predecessor window.** `registrant_cutover.json` gives old Merck (0000064978) MRK's consolidating history to 2009-11-03 (`basis` `reverse_merger_accounting_predecessor`); Schering-Plough's pre-merger 8-Ks are no longer MRK's (dated event windows).
- **P2 — Missing cutovers without a successor filing.** KMI and CDW each have a second CIK filing one after the roster CIK, but no 8-K12B/8-K12G3 exists to date a register entry (LBO and re-IPO). Until decided, consolidating forms list the roster CIK only (DOW and TPL are now in the register). Trigger: a decision per company or an SEC filing that dates the seam.
- **P2 — Merger reviews CB, DOC, DELL.** The roster CIK traded under another symbol until it took the ticker (ACE→CB, PEAK→DOC, DVMT→DELL). Confirm the accounting predecessor; the default roster-CIK window is most likely right. PLD goes with the traded-security realignment below.
- **Handled — tape mixes.** `security_master` now sums only a company's canonical and secondary-class lines (ratio-converted) and keeps preferreds (FITBP, IIVIP) and acquired companies' lines (SGP before 2009-11-03) apart; a predecessor's own line (ACE, MYL) is the one canonical line on its dates. The remaining `manual_review` tape-mix flags are review items for the master's roles.
- **P2 — Purge and validator judge junk CIK strings differently.** `identity-propagate` judges every non-blank filer CIK, so a CIK string with no digit is purged as foreign, while the validator skips it. Recommended: adopt the validator's rule in the purge (never judge a CIK with no digit). Behaviour change. Trigger: the next edit of [filer_tables.py](../src/utils/filer_tables.py) or a junk CIK in a purge WARNING.
- **Handled — the insider EDGAR walk runs in `sec_api`.** The DAG runs `insider-zip` in `sec_bulk`, then `insider-edgar` in `sec_api`.
- **P1 — Traded-security realignment for PLD, JCI, DD and DOW.** Before a merger a ticker's history should be the security whose prices we carry, after it the combined company. Yahoo's pre-seam prices are AMB's for PLD (2011-04-01: Yahoo 36.04 = AMB 36.04, old ProLogis 15.98), Tyco's for JCI (Yahoo's 2007, 2012 and 2016 split factors are Tyco's) and Dow Chemical's for DD (the 2000-06-19 3:1 split), while the register predecessor is the accounting predecessor (old ProLogis 0000899881, old Johnson Controls 0000053669, old DuPont 0000030554). DOW has no price before 2019-03-20, and its register predecessor TDCC (0000029915, to 2019-04-01) is the company DD's pre-2017 prices show. MRK, LIN, EVRG, BKR, STE and VTRS are aligned. Work: (1) register = traded-security predecessor: PLD loses its entry (AMB's CIK is PLD's own), JCI becomes Tyco 0000833444 throughout, DD becomes Dow Chemical before 2017-08-31, DOW starts at 2019-04-01 or needs dated CIK membership (one CIK in two entities at different dates, which the one-entity-per-CIK invariant forbids); (2) reverse what followed the accounting view: the PLD/DD vendor replacement and exchange ratios, the PLD insider rule (AMB becomes canonical, old ProLogis acquired), Tyco's 1,357 FTD lines (become JCI's; old JCI's become acquired) and AMB's 373, the gate hypotheses and vendor exceptions, then the `deferred_to_traded_security` entries; (3) re-run identity, `identity-propagate --every-ticker`, filings, merged history, insider re-stamp and tapes for the four; (4) a validator rule comparing Yahoo's pre-seam price with the register predecessor's FTD price at every register seam. Evidence: run dir `plans/02-revision-4-q2.md` §13.10. Trigger: before any feature that mixes prices and financials across these seams is trusted.
- **P2 — `identity-propagate` re-parses whole caches inside the 7-day window (F-105 residual).** `entity_lineage.scope_changed_at` is per scope, not per CIK, so "the scope gained a CIK" cannot be told from the stored rows. 14 Notes / 7 pension / 1 insider tickers hold rows but none from a holdable CIK (old DuPont, old Dow, ICE, JCI, LIN and STE predecessors with no curated pension tag; XOM's new holdco CIK 0002115436), so while one sits inside its window its family re-parses the whole cache each night: after the cutover's cold build, 7 nights of Notes and statements re-parses. Idempotent, so correctness is unaffected. Fix (user decision: DDL or table markers): a per-CIK change stamp on `entity_lineage` CIK rows, or a NULL-marker row per re-parsed (ticker, CIK) in the family tables. Trigger: before unpausing the DAGs, if the nightly memory budget is tight.
- **P2 — Filing-count drift warning.** Nothing warns when a ticker's new quarter holds far fewer filings of a form than its trailing average (e.g. the last 3 quarters). Add a per-ticker, per-form drift warning to `validate`. Trigger: the next validation-library change.
- **P2 — Roster CIK vs SEC ticker list.** `sp500_tickers.cik` comes from the Wikipedia scrape and nothing checks it (XOM carried a wrong CIK until 2026-09; a lagging page keeps an old CIK after a holdco change, so the new CIK's filings are never listed). Add an `identity_flags` action item `roster_cik_mismatch` when a roster ticker's CIK differs from the CIK `sec_company_tickers` gives that ticker today. It is a flag only: the snapshot never dates or resolves a CIK, and a confirmed change still needs a dated `registrant_cutover.json` entry. Trigger: with the drift warning, or the next roster CIK incident.
- **P2 — DOW 2018Q1–Q2 and VMC 2006Q1–Q3 missing in merged history.** Merged history is Sharadar-spined and no predecessor vendor series exists for TDCC or Legacy Vulcan, although the SEC filings exist (recorded in `vendor_coverage_exceptions.json`). Option: let the SEC block fill a vendor-missing quarter. Trigger: a fundamentals review covering DOW or VMC.
- **P3 — A scoped tape re-stamp counts a moved line twice (F-117).** `restamp_short_volume` and `restamp_fails` build the ticker grain from the fresh stamps of every loaded row, including rows held for out-of-scope companies. A `-t` run inside both companies' 7-day window therefore counts a line that moved between two universe companies under both (the mirror case under-counts the in-scope company) until the next unscoped run. Fix: compute the grain with held rows at their stored stamps, or exclude held rows. Verify: `_scripts/rv3_probe_scoped_move.py` prints "no double count". Interim: run the tapes unscoped.
- **P3 — A conflict CUSIP's manual override needs `ftd-download --full` (F-110).** A CUSIP flagged `security_issuer_conflict` has no master row, so its stored lines are purged and a manual issuer override has no observations to act on until `ftd-download --full` (then `identity-tables`, `fails-to-deliver -F`). Both fixes change nightly behaviour: rescanning manual CUSIP-6 prefixes with no stored line re-reads all 414 cached zips every night; keeping unattributed lines with a NULL role reuses the only pending marker, so they would be re-stamped nightly. Decide between a one-off rescan flag on `ftd-download` for named CUSIPs and a non-NULL `unattributed` role value.
- **P3 — Insider amendment dates are inherited only in the re-stamp (F-111).** The ingest paths stamp without pooled originals (`EdgarFetch.parse` has no store access), so an undated amendment whose original sits in another quarter keeps the `period_of_report`/`filing_date` fallback until a re-stamp pools stored originals (every unstamped row, every ticker inside its window). Mitigation: the second unscoped `identity-propagate` after `insider-transactions -F` in the cutover. Later: re-stamp, after each insider run, the tickers that received undated amendments.
- **P3 — EDGAR insider listing walks co-registrant CIKs.** `insider_filings` lists every event CIK, so co-registrant filings are read and then rejected at the screen: a few wasted requests. Drop co-registrant CIKs from the insider listing.
- **P3 — RE short volume before the EG rename stays unsummed.** Canonical lines run forward over FINRA days only for class lines, so RE's FINRA days between its last FTD fail and the 2023-07-10 rename to EG (same CUSIP) are kept raw but not summed. Fix: a forward run for canonical lines with the reuse guard of the backward run.
- **P3 — Evergy 2018-07-31 cover share count.** `fundamentals_facts` stores 217,687,849 for 10-Q `0001711269-18-000012`, where Sharadar has 271,687,849 and the share bridge gives 271.3 m: a 54,000,000 digit transposition, the filer's or the parser's. Only that row's `fundamentals_history_sec.sharesOutstanding` is affected.
- **P3 — BHGE 2017Q2 share count and a period stamp.** BHGE's own 2017Q2 10-Q cover count is the shell's 100 shares (`us-gaap:CommonStockSharesOutstanding`); its Class A/B counts are dimensioned and so not extracted. Separately, `fundamentals_facts` stamps BHGE 10-Q `0001701605-18-000052` (filed 2018-04-27) with `period_of_report` 2017-03-31, where the FS data set gives 2018-03-31.
- **Fixed 2026-10-07 — KO dividend-factor test on live prices.** Root cause: `price-history` re-pulled a ticker only after a split, so a new dividend left every earlier `close_total` on the old basis; it now also re-pulls after a dividend, from the earliest stored bar. The 194 affected tickers were re-pulled and the 7,301 bars stored before the 31-year floor (1995-09-01..1995-10-06) deleted; the live test passes.
- **P3 — PSKY `close_total` seam from Yahoo.** Yahoo serves only 4 PSKY sessions for 2026-07..09, so the bars stored earlier keep an older dividend basis and `D(d)` steps once. A source gap: re-pull PSKY when Yahoo serves the full window, or register the window in `yf_price_bugfix.json`.
- **P3 — 24 pre-existing Pyright errors** in `earnings_call_features.py`, `def14a/fetch.py` (`_completed_accessions`) and `validate/checks/earnings_calls.py`, on lines this work did not change. Trigger: the next edit to those files.
- **P3 — Stale identity prose on other pages.** [Nightly data refresh](./flows/nightly-data-refresh.md), [Data extraction](./modules/data-extract.md), [DAGs and infrastructure](./modules/dags-and-infrastructure.md) and [Large backfills and recovery](./guides/large-backfills-and-recovery.md) still describe identity-scope fingerprints, aliases and `identity-tables` after `insider-transactions`. The canonical text is in [data sources](./reference/data-sources.md) and the [run guide](./guides/run-the-pipeline.md). Trigger: the next edit of those pages.

- **Done 2026-10-07 — cutover verification.** `validate identity` after the cutover: PASS, 0 foreign rows (35,075 before), 13/13 continuity breaks explained, no vendor exception healed. Regression-gate diff against the 2026-10-06 snapshot: filings and insider 0 unexplained; tapes 1 (APA, below) after re-stamping six renamed companies; merged-history and price changes attributed (a merged table stale before the cutover, the 194-ticker price re-pull, APH's split). Live grain, KO dividend and identity tests pass.
- **P2 — The cutover re-stamp missed pre-cutover short-volume stamps.** `restamp_targets` selects companies whose master or lineage rows changed recently, so rows stamped by the old resolver for RTX (UTX), HWM (ARNC), IR (GDI), TT (IR), EXE (CHK) and BNY (BK) stayed unresolved from the last FTD sighting to the rename (204 rows) until a manual `restamp_short_volume` for those six re-stamped 6,111 rows. Fix: the cutover (or any identity rule change) re-stamps every company once. Trigger: the next identity rule change or a re-run of the cutover.
- **P3 — APA short volume on 2021-02-26.** The holding-company CUSIP change gives the old and new CUSIP a master interval on the same day, so the conflict rule leaves that day unresolved (1 row). Fix: the new CUSIP's interval starts the session after the old one ends.
- **P3 — Gate attribution gaps.** The regression gate labels DD (+32), MRVL (+45) and FERG (+8) predecessor periods under another reason than `register_window`, so three hypotheses read FAIL although the rows are present; merged-history changes are explained only through `fundamentals_history_sec`, so a merged table stale in the snapshot (employees, SEC block) reads UNEXPLAINED. Fix the gate's reason labels and compare the merged SEC block with the snapshot's `fundamentals_history_sec`. Trigger: the next gate run.

### Identity manual review of 2026-10-08 (open items)

Deferred from the identity manual-review run (local run dir `reports/validate/2026-10-08-identity-manual-review/`). Rule applied: a ticker's fundamentals follow its traded price series; a CIK whose price stops joins the survivor as an event-only acquired target.

- **DELL price history before 2018-12-28.** Yahoo's DELL series before 2018-12-28 is DVMT (the VMware tracking stock), unrelated to Dell Technologies' fundamentals. No mechanism starts a ticker's price history at a date; the price pipeline needs a dated start register, then DELL's usable history starts 2018-12-28.
- **VMRK comparatives.** AvalonBay (0000915912) is recorded as VMRK's acquired target (Yahoo's VMRK is EQR's series). If the first post-merger VMRK reports present AvalonBay as the accounting predecessor, their comparatives need the PLD-style handling.
- **Stale tape symbols.** DWDP→DD (2019-06-03, 8-K 0001193125-19-163322) is left to the traded-security realignment run, which owns DD; LTR→L has no SEC text found yet.
- **`dei` recapture.** The series-code slash rule and the `LegalEntityAxis` drop are merged; the stored `dei` rows change only at a `notes-download --full` recapture followed by `identity-tables`. Replayed offline on the 2019+ zips, conflicts fall 43 → 2; the two left (CMS-PB on CMS and Consumers Energy, SREA on Sempra and its utilities) are one security listed on two registrants' covers without a legal-entity dimension. Pre-2019 zips were not replayed.
- **Live apply.** The merged configs take effect at the next `identity_tables` → `identity_propagate` DAG run (`identity-propagate --every-ticker` re-stamps the FINRA seam rows of FOXA, DOC, TT, VTRS and COHR).
- **PEAK on 2024-03-01.** Healthpeak's last PEAK session (2.19M shares on FINRA) stays unattributed: the 42250P103 line from 2024-03-01 is keyed on DOC, and a second boundary on the CUSIP would overlap it. Needs a per-symbol boundary on one CUSIP.
- **Remaining tape-mix actions.** FITBP (an open `form345` tenure while the master's FITBP lines stop at 2026-06-09 and gap 2013–2019) and HUBA (the overlap runs to 2016-08-18, the master's HUBA line ends 2015-12-24) stay ACTION items for a tenure curation.
- **Tape-mix flag reads the previous master.** `build_entity_lineage` judges tape-mix overlaps against the stored `security_master`, rebuilt after the lineage; a new acquired line is information only from the following run.

## Employee headcount follow-ups

Deferred from the employee headcount coverage run (local, gitignored run dir `reports/validate/2026-10-07-employee-headcount-coverage/`, findings F1–F6 in `03-implementation.md`). The branch merged into `dev` without the independent validation stage, so F2, F3 and the scope-jump rule below are still open.

- **P1 — Universe run, then cube rebuild (user-run).** The live table was recreated and holds only the 50 focus tickers plus A; `fundamentals_history.employees_sec` is dropped. Run `fundamentals-employees` over the universe without `-F`, in one process, about $0.0014 per filing (~$17 for ~12,600 undecided dates), then `validate employees`, then rebuild `cube_part_fundamentals` with `-F`. Steps: [runbook](./guides/large-backfills-and-recovery.md#employee-headcount-universe-run).

- **P2 — The contractor guard rejects "complementary" outside a workforce (F4).** `_CONTRACTOR_LABEL_RE` matches "complementary" anywhere in the number's phrase, so IR's "over 40 complementary service and repair centers … and over 21,000 employees" is rejected in 6 filings, and IR has no 2025+ count. Fix: require a workforce noun after `complementary`, then delete IR's `unsupported` rows and rerun it. Trigger: the next fetcher change.
- **P3 — Image-only filings stored as `not_disclosed` (F6).** PEG FY2022–26 give the headcount in an image; the model returns `not_disclosed`, so the table does not tell image-only from verified non-disclosure. Trigger: with the OCR trigger below.
- **P2 — Derived `full_part` totals have no total quote (F2).** `validate employees` demands a quote per stored component, but a total stored as FT + PT, where no total is stated, has none by design (156 false provenance findings on the 50). Fix in the validator: accept a total equal to its quoted FT + PT. Trigger: before the universe run's validation is read.
- **P2 — `10-K405/A` outside the validator's form scope (F3).** The fetcher lists `10-K405/A`; the cached index and the validator's `ANNUAL_FORMS` do not, so 12 owned amendments read as stale rows and as accessions missing from the index (true stale rows: 0). Fix: add the form to `ANNUAL_FORMS` and the index filter. Trigger: with F2.
- **P3 — Validator scope-jump rule differs from the spec.** `validate employees` flags a same-basis YoY change above 50 %; the spec (REQ-011) asked for a value differing from both neighbours by a log ratio of 1.4 while revenue stays flat, which separates scope errors from real M&A. Report-only today. Trigger: with the `dei:EntityNumberOfEmployees` cross-check.
- **Trigger — OCR for image-only counts.** Build it when more than 1 % of owned 10-Ks end image-only.
- **Trigger — `dei:EntityNumberOfEmployees` cross-check.** Add it if scope jumps the prompt cannot fix persist in `employees_scope_jumps.csv`.
- **Trigger — scope-error auto-correction.** The validator only detects scope jumps; correct them automatically only when detection shows a recurring, mechanical pattern.

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
- earnings surprises starts 1999-08, but empty till ~2002
- financial notes (text & nums) 2009 from sec XBLR (zip), but possible directly from fillings (edgar)
- fix volume to be adjusted to spinoffs in price

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
