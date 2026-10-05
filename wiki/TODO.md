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
- **P2 — PSKY short volume stored under PARA/PARAA.** About 400 PSKY sessions are missing from `sec_short_interest`, although FINRA's RegSHO day files carry them under the old symbols PARA and PARAA. `Identity.candidate_symbols` in [identity.py](../src/data_extract/utils/common/identity.py) has no PARA/PARAA tenure for PSKY, so the fetch drops those rows. The user deferred the mapping on 2026-10-05. Fix: add the PARA/PARAA → PSKY symbol tenure to [symbol_tenure_manual.json](../configs/sec/symbol_tenure_manual.json), then re-read PSKY with `short-interest -t PSKY -F`, alone and outside 01:00 (a scoped `-F` upserts only its own tickers' rows). Trigger: before a short-flow feature review that covers PSKY, or the next symbol-tenure change.
- **P1 — False second symbol tenures drop short-volume days (for the entity-symbol-lineage branch).** `symbol_tenure` holds tenures derived from a few mis-tagged Form 3/4/5 filings, which make the RegSHO resolver read a universe symbol as `ambiguous` and drop its rows: F -> CIK 0000712537 (First Commonwealth Financial), 2018-04-26..2019-04-25, 6 filings, which costs Ford 183 sessions (2018-08-01..2019-04-24); AXP -> AirXpanders (CIK 0001387156), 2017-07-07..2018-09-05, 20 filings; and 1-2 day tenures on NTRS (Illinois Tool Works), PM (FAC Propertys), AME (AMC Entertainment) and SPG (SpyGlass Pharma). After the one-time `short-interest --repair-gaps` of 2026-10-05, 1,120 ticker-days (18 tickers) are still missing: PSKY 400 (above), SMCI 349 (absent at FINRA), F 183, EXE 156, and 1-4 days on ticker-change dates. The entity-symbol-lineage branch owns the fix (a minimum-evidence rule or manual overrides for a second tenure). Follow-up after it: run `short-interest -c ./configs --repair-gaps` once, alone and outside 01:00 (about 7 min, a safe upsert), and re-count the gaps. Trigger: when that branch merges.
- **P2 — One joint 13F filing sits under two managers.** In `sec13f_hr`, the 2015-12-31 books of CIK 0001622431 and 0001054880 are the same 35 rows (35 of 35 equal in shares and value), filed 2016-02-12; 0001622431's 2015-06-30 and 2015-09-30 books mix its own filings (filed 2015-08-10 and 2015-11-12) with rows of that 2016-02-12 filing. Its own 2015-12-31 filing held 3 rows. The 2026-10-05 CIK normalisation (stage B, keep the later-filed row per key) replaced 2 of those 3 rows (VZ, XOM) with the joint book's values. The rows as they were are in the run dir export `_out/phase11/state_before_B/sec13f_hr_affected_rows.csv`. Fix: attribute a joint 13F-HR to its filer only (or split it by the other-manager list), then delete the misattributed rows and refetch both managers' 2015 periods. Trigger: before a superinvestor or 13F feature reads these managers' 2015 books.



## Insider transactions follow-ups

Deferred from the single-table merge of `insider_transactions` (findings F-007..F-014 and R-04 in the local, gitignored run dir `reports/validate/2026-10-03-insider-transactions-merge/defects.md`).

- **P1 — Ingest insider 2026q1.** The 2026-10-04 refill skipped 2026q1: before the two-path zip fallback (F-005) the quarter was requested only from the new SEC path and got HTTP 404. Only its 77 EDGAR re-reads are stored, so every 2026 feature window reads its trades as absent. Once the branch is merged, run once from the main tree, with no `-F` and no `-t` (2026q1 is still pending, and the main tree caches `data/sec_insider_transactions/2026q1.zip`, so nothing is downloaded): `rtk "$PY" -m src data_extract insider-transactions`. Expect `insider 2026q1: 12525 / 12602 filings missing from EDGAR (99.4%)`: an artefact of the hole and of the quarter-wide N (F-012 below), not an EDGAR loss. Then rebuild the cube: `data_aggregate build-institutionals` (a full build, since the retired columns go away) and `assemble-cube`. Verify: 82 distinct `quarter` values (2006q1 to 2026q2) and about 12,602 accessions in 2026q1. Trigger: right after the merge, before any cube build.
- **P2 — Amendment double counts (R-04).** Linked amendment rows never enter the copy collapse in [insider_quality.py](../src/data_aggregate/utils/institutionals/insider_quality.py). Measured on the refilled table (all history, scoped P/S): (a) joint reporters who each file a value-changing 4/A count twice from the 4/A day, 81 rows / $449m as filed; (b) a linked 4/A adding a cell a co-owner already reported counts twice, 194 rows / $763m over 5,737 new-cell rows (e.g. NTRS `0000073124-06-000091` vs `-000072`). Small next to the 6,813 copies the collapse removes, but material in dollars. Trigger: the next insider feature review, or before `buy_value` features are re-tuned.
- **P2 — An undownloadable zip quarter hides under a complete frontier (F-007).** `fetch_insider_transactions` in [fetch_insider_transactions.py](../src/data_extract/utils/institutionals/fetch_insider_transactions.py) skips a quarter that no URL serves with only per-URL WARNINGs; the EDGAR window then starts after the last stored quarter, so nothing but the stored quarters shows the hole. Fix: fail or WARN-summarise a skipped non-newest quarter, and/or hold the frontier at the last quarter end before it. Trigger: the next zip-path change at the SEC, or any refill.
- **P3 — Repeat collapse merges different people opening equal new positions (F-011).** `_collapse_copies` keys on (ticker, day, code, shares, price, holding); when the holding equals the shares bought it cannot tell two new positions apart. DDOG IPO day 2019-09-23: two directors' purchases collapse into two others', so `distinct_buyers_120d` reads 4 instead of 6; about 6 zero-start groups / $482m since 2020, some legitimate (TKO). Fix idea: a zero-start group collapses only when the accessions share an owner or filer agent. Trigger: the next insider feature review.
- **P3 — REQ-007 WARNING overstates EDGAR losses (F-012).** `report_zip_quarter` counts N from the quarter-wide earliest EDGAR filing date, while EDGAR listing is per ticker since the merge; after a rebuild, a `-t` run or a skipped quarter the WARNING inflates (2026q2 replay 71.0 %). Fix: N per ticker. Trigger: the next time the WARNING is used to judge EDGAR completeness.
- **P3 — Half-cent tolerance in the mismatch line (F-013).** `_same` compares `gap.le(0.005)` on floats, so the zip's half-up rounding of a half-cent EDGAR value (116.695 vs 116.70) counts as a mismatch: 65 of 66 in the 2026q2 replay. Fix: compare `round_half_up(edgar)` with the zip value, or allow a few ulps. Trigger: with F-012.
- **P3 — `validate pull` fails on `insider_transactions` (F-014).** `_coerce` in [io.py](../src/validate/io.py) fixes a float policy for `footnote_ids` from an all-null first chunk (zip rows come first), then fails on `'F5,F6'`. Fix: decide the policy on the first non-null value or from the DB column type. Trigger: the next validation-library run on an insider table.

## Superinvestor roster follow-ups

Deferred from the verified point-in-time roster run (branch `harness/sec13f-superinvestor-roster`; local, gitignored run dir `reports/validate/2026-10-04-sec13f-superinvestor-roster/`). See [Superinvestor roster (Dataroma)](./reference/data-sources.md#superinvestor-roster-dataroma).

- **P1 — Post-merge rebuild (user).** After the branch merges into `dev`: rerun `superinvestors --seed` from the merged code, unpause the `data_extraction` DAG, then rebuild the cube and re-baseline the aggregate fingerprint. The elite-manager features shift by design: managers with a book per quarter rise from about 51 to 65 on average.
- **P1 — Garbage rows from legacy text 13F books (F-001).** The Phase 5 catch-up (`years_history` 31, periods back to 1995) imported 5,317 garbage `sec13f_manager_holdings` rows in 72 pre-XML text-era books; pre-existing books hold 1,290 more in 57. Garbage means a common row with shares 0 and `value_usd` > 0, one position above $1tn (max 3.27e17), or unsplit information-table text in `issuer_name`; e.g. Davis `0001036325` 2010-03-31, Wells Fargo with shares 0 and `value_usd` 327,403,410,520,673,000. Inside the feature window (period ≥ 2011-09-30), 133 rows in 5 books give wrong concentration and weight denominators: Third Avenue `0001099281` 2012-03-31 and 2012-06-30, Pabrai (Dalal Street) `0001549575` 2012-12-31 and 2013-03-31, ValueAct `0001418814` 2013-03-31. Cause: `_read_filing` in [fetch_13f.py](../src/data_extract/utils/institutionals/fetch_13f.py) misaligns legacy text information tables, and `_save_book` only counts implied-price outliers, never rejects a row. Not a value-unit issue: clean 1000× filings occur at the same rate in new and pre-existing books (4.5% vs 4.0%), and `repair_value_basis` already repairs them. **Parser ported (branch `fix/13f-legacy-parser`):** `_read_filing` now re-reads a pre-XML text book that EdgarTools failed on, that disagrees with the source (entry count, CUSIPs, Summary Page value total), or that holds a garbage row, with the source-checked [legacy_13f_fallback.py](../src/data_extract/utils/institutionals/legacy_13f_fallback.py); an unverifiable book is a deterministic read failure and is never stored. XML-era reads are unchanged. **Data repair done (2026-10-05, run folder `fixes/F-001-repair.md`):**
  - The 34 verifiable books were deleted and refetched: 4,234 rows went out and 4,233 verified rows came back, with 0 garbage and totals equal to the replay.
  - The 95 other books were classified by the gap between the stored total and the SEC cover total, read in $1000s, dollars or $ billions:
    - 65 books within 10% were kept as stored.
    - 30 books (5 in the feature window: Harris `0000813917` 2011-09-30 and 2011-12-31, Third Avenue `0001099281` 2012-06-30, Pabrai `0001549575` 2012-12-31 and 2013-03-31) kept their positions, but all nine amount columns were set to NULL (4,044 rows).
  - `load_superinvestor_holdings` drops any book whose `value_usd` is all NULL, so that manager-quarter reads as unfiled. The catch-up never re-reads these books, because their `(period, filing_date)` is still stored.
  - **Remaining:**
    - The 65 kept books still carry 4,492 garbage rows: 4,313 zero-share common rows (94 in-window), 17 negative and 194 issuer-text rows.
    - The XML book `0001061165` 2014-12-31 has 5 positions above $1tn. The whole book is 1000x; `repair_value_basis` measures a median implied-price/close ratio of 1000.0 on it and divides it.
    - The 30 NULL-amount books, plus 58 issuer-text names inside them, are waiting for a better parser (`harness/13f-managers-cusip`).
    - Every catch-up run logs 246 ERRORs on 228 never-stored (cik, period) filings across 19 CIKs. 15 of them are in-window, and they are the R-06 missing quarters.
    - Fixed (`ea37ef0`): `0000098758` 2011-06-30 was held back as "transient" on every run because `is_rate_limited` matched "429" inside the cover text "460,429"; an unverifiable legacy book is now always a deterministic failure.
    - The 228 never-stored (cik, period) are re-read on every catch-up (cheap: archives are cached locally) and log an ERROR each time. Options: record them as all-NULL placeholder books like the 30 above, or log them at INFO once known.
- **P2 — `sec13f_hr` unpadded-CIK duplicates (R-07).** The same manager is stored padded and unpadded. Owner: `feat/db-derived-resume` Phase 11.
- **P2 — Ghost non-quarter-end manager periods and pre-2013 text 13F books (R-06).** Legacy-era manager periods that are not quarter ends, and the pre-XML text 13F books (Greenhaven, the 15 missing 2011Q3 to 2013Q1 quarters at 0000098758, 0000846222 and 0001099281). Owner: `harness/13f-managers-cusip`.
- **P2 — `fetch_13f_managers` code health.** Its private cross-imports from `fetch_13f.py` and the dual-padding lookup in `_warn_empty_books`. Trigger: after `feat/db-derived-resume` merges (it rewrites `fetch_13f.py`).
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
