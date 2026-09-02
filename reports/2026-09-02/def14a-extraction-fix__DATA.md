---
type: DATA
session_id: cbba673e-f8e0-4596-b06d-a136e202dd08
generated_at: 2026-09-02T21:36:12+00:00
baseline: {head_sha: b5cb15603ee36f282ee2c50e58a257a441d1fbce}
generator: scripts/dod/data_profile.py@1
---

## 1. Scope

**SAMPLE SCOPE** — a metric without its scope is not a measurement:

- tables: def14a_llm, sec_8k, sec_def14a
- tickers: **all** (no ticker filter)
- since: **no lower bound**
- row limit per table: **none**
- full-scope tables (eligible to set the baseline): def14a_llm, sec_def14a, sec_8k

**What was asked:** implement the eight-phase plan in
`reports/planning/active-tasks/2026-09-01-def14a-extraction-fix/` — replace the silently-wrong
edgartools proxy HTML block with an LLM path plus four child tables, slim `sec_def14a` to the
filer-tagged ECD/PVP block, add `sec_8k_votes` from the already-stored Item 5.07 narratives, then
run the 23-ticker comparison, gate it, cut over and rebuild.

**Read §3 as the BEFORE state.** Phases 0-5 are code-complete and committed; **Phase 6's DEF 14A
re-extraction was explicitly retained by the user and NOT run here**, so the live tables in §3 are
still the ones this work exists to replace. `def14a_llm`'s 8,667 rows and 45 columns are the OLD
contract (three retired technology columns included), `sec_def14a`'s 46 columns are the OLD wide
shape, and the four new `def14a_*` child tables plus `sec_8k_votes` do not exist in Postgres yet —
they are created by that rebuild. The code is proven by the unit suites and the deterministic ECD
run in §4, not by these row counts.

## 2. Gates

| Gate | Check | Verdict | Detail |
|---|---|---|---|
| D1 | declared PK unique over the rows profiled | **PASS** | unique across 3 table(s): def14a_llm, sec_def14a, sec_8k |
| D2 | row count not decreased | **N/A** | no full-scope baseline to compare against — this run records one |
| D3 | no column lost | **N/A** | no baseline columns recorded yet |
| D4 | date range covers the expected window | **N/A** | no --expect-through given |
| D5 | per-field null rate not worse | **N/A** | no full-scope baseline null rates to compare against |

**All gates pass** (N/A gates are stated above, not skipped).

## 3. Metrics

_Observed values only — no verdicts. `rows`, `date_min` and `date_max` are **table-wide** (server-side); every other number is over the **sample** described in §1. Do not compare across the two._

**Tables**

| table | exists | rows | sampled | cols | pk | pk_absent_cols | pk_dupes | date_min | date_max | sample_date_min | sample_date_max |
|---|---|---|---|---|---|---|---|---|---|---|---|
| def14a_llm | yes | 8,667 | 8,667 | 45 | ticker,accession_number | — | 0 | 1995-09-13 00:00:00 | 2026-08-17 00:00:00 | 1995-09-13 | 2026-08-17 |
| sec_8k | yes | 249,839 | 249,839 | 14 | ticker,accession_number,item | — | 0 | 1995-09-01 | 2026-09-01 | 1995-09-01 | 2026-09-01 |
| sec_def14a | yes | 3,662 | 3,662 | 46 | ticker,accession_number | — | 0 | 1995-09-13 | 2026-07-24 | 1995-09-13 | 2026-07-24 |

**Fields** (worst null rate first, top 60)

| table | field | dtype | null_% | nunique | mean | std | min | p01 | p50 | p99 | max | mad_outliers |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sec_def14a | auditor_name | str | 97.79 | 10 | — | — | — | — | — | — | — | — |
| sec_def14a | award_dates_predetermined | float64 | 95.33 | 2 | 0.871345 | 0.335801 | 0 | 0 | 1 | 1 | 1 | 22 |
| sec_def14a | mnpi_disclosure_timed_for_comp_value | float64 | 95.19 | 2 | 0.0113636 | 0.106295 | 0 | 0 | 0 | 0.25 | 1 | 2 |
| sec_def14a | award_timing_mnpi_considered | float64 | 95 | 2 | 0.393443 | 0.489854 | 0 | 0 | 0 | 1 | 1 | 0 |
| sec_def14a | total_fees_prior | float64 | 94.27 | 202 | 7.04917e+10 | 1.02164e+12 | -6.841e+09 | 9,044.27 | 1.00715e+07 | 6.3826e+08 | 1.4805e+13 | 14 |
| sec_def14a | total_fees_current | float64 | 94.16 | 204 | 5.58677e+10 | 8.16883e+11 | 0 | 33,238.6 | 1.03376e+07 | 5.8168e+08 | 1.195e+13 | 14 |
| sec_def14a | other_fees_prior | float64 | 93.77 | 127 | 4.90249e+07 | 6.63015e+08 | -2e+06 | -8,885.56 | 39,000 | 2.8796e+08 | 1e+10 | 58 |
| sec_def14a | other_fees_current | float64 | 93.66 | 125 | 4.3778e+08 | 6.56511e+09 | -1.4e+07 | 0 | 29,500 | 3.0294e+08 | 1e+11 | 63 |
| sec_def14a | audit_fees_prior | float64 | 93.09 | 240 | 4.57972e+10 | 7.28027e+11 | 6,000 | 140,256 | 7.221e+06 | 1.01888e+09 | 1.158e+13 | 10 |
| sec_def14a | audit_fees_current | float64 | 92.98 | 242 | 3.81588e+10 | 6.11306e+11 | 75,000 | 98,440 | 8e+06 | 9.3256e+08 | 9.8e+12 | 11 |
| sec_def14a | audit_related_fees_prior | float64 | 92.55 | 219 | 5.889e+09 | 9.72599e+10 | 0 | 0 | 575,667 | 2.37413e+07 | 1.607e+12 | 45 |
| sec_def14a | insider_trading_policy_adopted | float64 | 92.55 | 2 | 0.996337 | 0.0605228 | 0 | 1 | 1 | 1 | 1 | 1 |
| sec_def14a | audit_related_fees_current | float64 | 92.38 | 234 | 5.91664e+09 | 9.87828e+10 | 0 | 0 | 678,600 | 2.37297e+07 | 1.65e+12 | 37 |
| sec_def14a | tax_fees_prior | float64 | 91.97 | 239 | 5.47157e+09 | 9.37803e+10 | -3.9e+07 | -351,647 | 743,046 | 3.03859e+07 | 1.608e+12 | 33 |
| sec_def14a | tax_fees_current | float64 | 91.92 | 247 | 1.35353e+09 | 2.32494e+10 | -1.5e+07 | -337,704 | 636,914 | 2.955e+07 | 4e+11 | 39 |
| sec_def14a | net_income | float64 | 86.26 | 501 | 1.44374e+10 | 1.14067e+11 | -2.28e+10 | -2.43956e+09 | 1.4801e+09 | 1.1171e+11 | 1.55245e+12 | 58 |
| sec_def14a | company_selected_measure_name | str | 85.83 | 159 | — | — | — | — | — | — | — | — |
| sec_def14a | peo_name | str | 85.53 | 213 | — | — | — | — | — | — | — | — |
| sec_def14a | peo_actually_paid_comp | float64 | 85.47 | 532 | 3.37353e+07 | 1.30297e+08 | -6.86374e+08 | -7.95448e+07 | 1.81818e+07 | 3.39553e+08 | 2.268e+09 | 55 |
| sec_def14a | peo_total_comp | float64 | 85.47 | 531 | 1.93979e+07 | 2.02398e+07 | 12,566 | 40,018.7 | 1.60306e+07 | 8.73053e+07 | 2.47579e+08 | 21 |
| sec_def14a | company_selected_measure_value | float64 | 85.06 | 524 | 1.42054e+10 | 1.21031e+11 | -1.431e+10 | -13.5672 | 23.27 | 2.58999e+11 | 1.69671e+12 | 241 |
| sec_def14a | neo_avg_actually_paid_comp | float64 | 84.79 | 557 | 1.0433e+07 | 2.17871e+07 | -3.27082e+07 | -9.97207e+06 | 5.54668e+06 | 9.87132e+07 | 3.32353e+08 | 66 |
| sec_def14a | neo_avg_total_comp | float64 | 84.79 | 557 | 7.04827e+06 | 5.81969e+06 | 4,518 | 1.04153e+06 | 5.14266e+06 | 2.71099e+07 | 6.14342e+07 | 52 |
| sec_def14a | peer_group_tsr | float64 | 84.79 | 419 | 158.65 | 57.1499 | 24.4 | 64.4936 | 146 | 318.128 | 547 | 17 |
| sec_def14a | total_shareholder_return | float64 | 84.79 | 497 | 176.624 | 125.794 | 5.15 | 38.4 | 144.513 | 710.584 | 1,182.35 | 34 |
| sec_def14a | company_name | str | 84.76 | 179 | — | — | — | — | — | — | — | — |
| sec_def14a | ceo_pay_ratio_median_employee_comp | float64 | 77.39 | 827 | 477,598 | 3.56081e+06 | 5.36536 | 175.27 | 71,445 | 1.53401e+07 | 8.33617e+07 | 38 |
| sec_def14a | ceo_pay_ratio_ceo_comp | float64 | 77.17 | 836 | 1.45473e+08 | 2.35358e+09 | 48 | 13,391.5 | 1.34761e+07 | 1.51373e+09 | 6.62725e+10 | 50 |
| sec_def14a | audit_fiscal_year_current | float64 | 76.87 | 25 | 2,014.96 | 6.57298 | 2,002 | 2,002 | 2,015 | 2,025 | 2,026 | 0 |
| sec_def14a | audit_fiscal_year_prior | float64 | 76.87 | 25 | 2,013.95 | 6.57443 | 2,001 | 2,001 | 2,014 | 2,024 | 2,025 | 0 |
| def14a_llm | ceo_ownership_pct | float64 | 76.07 | 910 | 0.0775922 | 0.156449 | 0 | 0.0001 | 0.018 | 0.82 | 1 | 396 |
| sec_def14a | ceo_pay_ratio | float64 | 73.05 | 482 | 255.667 | 335.759 | 0 | 1.92972 | 174 | 1,663.34 | 5,294 | 76 |
| def14a_llm | median_employee_pay | float64 | 55.43 | 3,801 | 89,658.4 | 54,716.3 | 5,091 | 10,180 | 76,519 | 259,798 | 598,414 | 93 |
| def14a_llm | ceo_pay_ratio | float64 | 55.08 | 1,026 | 400.404 | 5,079.53 | 0 | 3.676 | 179 | 1,935.32 | 246,804 | 323 |
| def14a_llm | say_on_pay_support_pct | float64 | 44.71 | 447 | 0.91046 | 0.101463 | 0.111 | 0.46 | 0.94 | 0.99 | 1 | 418 |
| def14a_llm | n_technology_directors | float64 | 43.52 | 15 | 3.74525 | 2.31735 | 0 | 1 | 3 | 10 | 14 | 1 |
| def14a_llm | pct_technology_directors | float64 | 43.52 | 94 | 0.360052 | 0.230253 | 0 | 0.067 | 0.308 | 1 | 1 | 0 |
| def14a_llm | insider_ownership_pct | float64 | 42.69 | 1,474 | 0.0853982 | 0.145784 | 0 | 0.00126222 | 0.0254 | 0.693736 | 1 | 1,032 |
| def14a_llm | lead_independent_director | float64 | 30.11 | 2 | 0.713555 | 0.452137 | 0 | 0 | 1 | 1 | 1 | 0 |
| def14a_llm | ceo_since_year | float64 | 22.85 | 53 | 2,010.07 | 9.08713 | 1,966 | 1,985 | 2,012 | 2,025 | 2,026 | 47 |
| def14a_llm | avg_other_public_boards | float64 | 22.48 | 223 | 1.25403 | 0.563952 | 0 | 0.25 | 1.182 | 3 | 6.5 | 104 |
| def14a_llm | ceo_option_awards | float64 | 22.04 | 3,497 | 2.7375e+06 | 3.18458e+07 | 0 | 0 | 624,996 | 1.93145e+07 | 2.28399e+09 | 1,052 |
| def14a_llm | ceo_age | float64 | 21.45 | 67 | 57.5194 | 6.86174 | 28 | 40.07 | 58 | 75 | 95 | 59 |
| def14a_llm | ceo_non_equity_incentive | float64 | 19.7 | 4,746 | 2.27143e+06 | 2.30886e+06 | 0 | 0 | 1.81214e+06 | 1.09215e+07 | 2.25337e+07 | 188 |
| def14a_llm | ceo_is_founder | float64 | 18 | 2 | 0.0794991 | 0.270535 | 0 | 0 | 0 | 1 | 1 | 565 |
| def14a_llm | ceo_stock_awards | float64 | 16.57 | 6,186 | 7.37916e+06 | 1.84515e+07 | -6.40016e+06 | 0 | 5.09968e+06 | 4.4531e+07 | 8.13179e+08 | 239 |
| def14a_llm | ceo_bonus | float64 | 16.25 | 1,013 | 498,669 | 2.06513e+06 | 0 | 0 | 0 | 8.25252e+06 | 9e+07 | 444 |
| def14a_llm | ceo_all_other_comp | float64 | 16.18 | 6,512 | 422,500 | 2.03562e+06 | -758,687 | 0 | 126,612 | 4.31419e+06 | 6.84843e+07 | 702 |
| def14a_llm | ceo_equity_pay_pct | float64 | 14.93 | 894 | 0.552206 | 0.267086 | -0.241 | 0 | 0.62 | 0.97528 | 3.082 | 6 |
| def14a_llm | total_neo_comp | float64 | 14.87 | 7,357 | 3.24662e+07 | 4.76011e+07 | 1 | 1.71913e+06 | 2.44834e+07 | 1.46452e+08 | 2.28404e+09 | 310 |
| def14a_llm | ceo_total_comp | float64 | 14.85 | 7,309 | 1.50099e+07 | 3.68753e+07 | 0 | 103,433 | 1.17222e+07 | 6.72824e+07 | 2.28404e+09 | 207 |
| def14a_llm | pct_independent_directors | float64 | 14.18 | 116 | 0.742929 | 0.2614 | 0 | 0 | 0.857 | 0.95441 | 2.75 | 1,090 |
| def14a_llm | avg_director_age | float64 | 14.1 | 1,164 | 61.9717 | 3.88782 | 40 | 50.805 | 62.2 | 70.9095 | 85.5 | 95 |
| def14a_llm | ceo_salary | float64 | 13.85 | 3,322 | 1.08092e+06 | 489,922 | 0 | 1 | 1.04998e+06 | 2.99997e+06 | 9.3636e+06 | 128 |
| def14a_llm | avg_board_tenure | float64 | 13.33 | 1,063 | 8.23064 | 3.62765 | 0 | 1.111 | 7.75 | 18.9909 | 43.75 | 96 |
| def14a_llm | auditor_fees | float64 | 13.12 | 6,546 | 1.24028e+07 | 1.58608e+07 | 57.6 | 597,537 | 7.1175e+06 | 8.88e+07 | 1.50807e+08 | 701 |
| def14a_llm | independent_chair | float64 | 11.75 | 2 | 0.308406 | 0.461866 | 0 | 0 | 0 | 1 | 1 | 0 |
| def14a_llm | n_five_percent_holders | float64 | 9.57 | 21 | 3.26167 | 1.55435 | 0 | 1 | 3 | 8 | 23 | 56 |
| def14a_llm | ceo_is_board_chair | float64 | 8.61 | 2 | 0.485292 | 0.499815 | 0 | 0 | 0 | 1 | 1 | 0 |
| def14a_llm | technology_committee | float64 | 8.02 | 2 | 0.105118 | 0.306725 | 0 | 0 | 0 | 1 | 1 | 838 |

## 4. Evidence

- baseline file: `reports/baselines/data_profile.json` (2 table(s) recorded)
- `def14a_llm`: 8,667 rows, 45 cols, 8,667 sampled
- `sec_8k`: 249,839 rows, 14 cols, 249,839 sampled
- `sec_def14a`: 3,662 rows, 46 cols, 3,662 sampled

## 5. Regressions, gaps and deliberate omissions

- **The DEF 14A re-extraction was not run, by the user's instruction.** Phase 6 Steps 1
  (proxy/vote LLM legs), 3 (gate), 4 (cutover) and 5 (post-cutover verification) are therefore
  outstanding. Ten of the fourteen G1-G14 gates read PENDING in `COMPARISON.md` — they have **no
  measurement, not a failing one**, and the distinction is why they are not reported as passes.
- **Wasted LLM spend, my error: ~$5.** `TaskStop` killed the shell wrapper but not the Python
  children, so two abandoned proxy runs kept extracting — one for ~34 minutes — after I believed
  they were stopped. Found by enumerating command lines, killed by PID (never by image name).
  Verifying the process is actually dead after a stop is the fix.
- **A pre-2001 defect I introduced in the Phase-6 runner and fixed.** 88 of 656 cached filings
  carry `primaryDocument == ""`, so `_doc_url` builds a bare directory URL and Phase 0's
  `cache_htm` holds a **10,217-char EDGAR directory listing** rather than the proxy; the real
  document is the `.txt` sibling (159,203 chars on A's 2000 filing). Reading `cache_htm` fed the
  model a folder index and those filings returned **0-2 non-null fields**, which reads as
  "pre-2001 does not extract" when the cause is opening the wrong file. Fixed and verified: A
  2000-01-13 went **2 non-null fields -> 27**. Production was never affected — it uses Phase 1's
  corrected `_doc_url`.
- **The smoke-set parquets in `new/` were DELETED rather than left in place.** They covered only
  each ticker's oldest filing and were built through the folder-index bug, so every fill rate
  computed off them was wrong. A stale artifact that reads like a validation run is worse than a
  missing one; `new/README.md` records what is and is not there.
- **The vote role map cannot be measured from the database yet.** `_role_source` joins to
  `def14a_executive_comp` / `def14a_director_comp`, which do not exist until the cutover creates
  them, so a DB-sourced map returned **96.8% `unmatched`** — a statement about the missing tables,
  not about the join. Step 1's `--votes` leg therefore sources its role frames from `new/`.
- **A real over-rejection in the Phase-5 fabrication guard, found by running it on live data.**
  PTC's 2010 filing wraps a narrow name column so the vote numbers land *between* the halves of
  the name (`Paul` / 100,753,338 / `A. Lacy`), and a contiguous substring check discarded three
  correctly-read nominees. Fixed with a token-level fallback; the measured fabrications ("John
  Doe" / "Jane Smith") still fail it. The one remaining weakness — a name assembled from words
  present elsewhere in the document — is documented in the docstring and covered by a test.
- **`sec_8k_votes` has no accuracy gate, deliberately.** A vote table prints no independent total
  and the dominant error is a column permutation, which is invariant under sums: all computable
  checks combined give recall 0.56 / precision 0.56, with 7 of 16 known-bad filings passing clean.
  `nominee_sum_matches` is stored as a monitor and never used as a filter.
- **XOM's transposed vote layout rests on the prompt and no test.** XOM is not in the 23-ticker
  Phase-0 baseline and there is no other cached source for that geometry. Every other geometry the
  plan named was re-sourced from the 333 real baseline narratives.
- **D2-D5 came back N/A**: no full-scope baseline existed for these tables before this run, so
  this run records one. They become real checks on the next profile.
- **The length floor cannot catch a truncated Item 5.07.** The shortest GENUINE tally measured is
  554 chars (TDG's 2014 and 2019 special meetings) against HWM 2019's 508-char stub, so a floor
  high enough to reject the stub destroys two correct filings. The comma-grouped-number rule does
  that work instead. Recorded so the floor is not "tightened" later.
- **`sec_def14a_*` row counts in `docs/database.md` describe retired data.** The four tables are
  no longer written by any code; they survive in Postgres only until the cutover drops them.

## 6. Next actions

- **Yours, and the only remaining destructive step.** Take the pre-cutover snapshot first —
  it is the sole rollback that exists, because truncate-and-rebuild was chosen over migration:
  `"$PY" scripts/def14a_baseline.py -c ./configs --tag pre-cutover`.
- Then, in order: `"$PY" scripts/def14a_cutover.py -c ./configs --confirm` (dry-run verified: 7
  statements — drop the 4 retired child tables, drop `sec_def14a` because 46->25 columns cannot be
  reshaped by `CREATE TABLE IF NOT EXISTS`, truncate `def14a_llm`, drop the 3 technology columns),
  then `def14a -t AAPL` **single-ticker first** to create the tables warm (`ensure_table` is a
  check-then-create with no lock, and threaded writers on a cold table can silently lose rows),
  then the universe runs: `def14a`, `def14a-edgar`, `sec-8k-votes`.
- **Budget it before starting: ~34 hours and ~$150.** Measured, not estimated — a modern proxy is
  a ~130k-char payload at **~94 s serially**, which is why the per-filing LLM calls in both
  fetchers now run 12-wide (**9.1-10.6 s/filing measured, no 429s**). That is ~23 h for 8,700
  proxies and ~11 h for 6,657 vote filings. `def14a_llm` upserts per ticker and dedups on
  accession, so the run is interruptible and resumable without losing paid tokens.
- Optionally run Step 1 first to gate before touching the DB — it writes **only parquet**:
  `--proxy --workers 12` (~656 calls, ~$11, ~100 min) then `--votes --workers 12` (~320 calls,
  ~$2), then `scripts/compare_def14a_baseline.py` to fill in the ten PENDING gates.
- After the rebuild: `"$PY" scripts/def14a_postcutover_verify.py -c ./configs` re-runs the defect
  assertions **against Postgres** rather than parquet, which is the only way to see the
  `DATE -> datetime.date` round-trip bug class.
- Update `docs/database.md`'s size table and the "five tables registered but not yet in Postgres"
  bullet with the real post-rebuild counts, and refresh the aggregate fingerprint baseline (it
  will move — features were dropped, which is expected, not a regression).
- Leave the plan directory in `active-tasks/` until the rebuild lands; Phases 0-5 are ✅ and
  Phase 6 is partially complete. 

```json dod-metrics
{
  "baseline_head_sha": "b5cb15603ee36f282ee2c50e58a257a441d1fbce",
  "content_hash": "sha256:4fc021eb3a5b77a1ace97ba756311bf4808a6ba5a1fcd7fa55a71d1a7df454fd",
  "gates": {
    "D1": "PASS",
    "D2": "N/A",
    "D3": "N/A",
    "D4": "N/A",
    "D5": "N/A"
  },
  "generator": "scripts/dod/data_profile.py@1",
  "metrics": {
    "parts_behind": null,
    "stale_sources": null,
    "tables": {
      "def14a_llm": {
        "columns": [
          "ticker",
          "as_of",
          "period",
          "accession_number",
          "company_name",
          "fiscal_year_extract",
          "n_directors",
          "board_size",
          "avg_director_age",
          "avg_board_tenure",
          "pct_independent_directors",
          "pct_female_directors",
          "avg_other_public_boards",
          "ceo_name_proxy",
          "ceo_age",
          "ceo_since_year",
          "ceo_is_founder",
          "ceo_is_board_chair",
          "ceo_salary",
          "ceo_bonus",
          "ceo_stock_awards",
          "ceo_option_awards",
          "ceo_non_equity_incentive",
          "ceo_all_other_comp",
          "ceo_total_comp",
          "ceo_equity_pay_pct",
          "n_neos",
          "total_neo_comp",
          "insider_ownership_pct",
          "ceo_ownership_pct",
          "n_five_percent_holders",
          "independent_chair",
          "lead_independent_director",
          "classified_board",
          "dual_class_shares",
          "poison_pill",
          "majority_voting",
          "say_on_pay_support_pct",
          "ceo_pay_ratio",
          "median_employee_pay",
          "auditor_fees",
          "def14a_json",
          "n_technology_directors",
          "pct_technology_directors",
          "technology_committee"
        ],
        "date_col": "as_of",
        "date_max": "2026-08-17 00:00:00",
        "date_min": "1995-09-13 00:00:00",
        "exists": true,
        "fields": {
          "accession_number": {
            "dtype": "str",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 8667
          },
          "as_of": {
            "dtype": "datetime64[us]",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 2822
          },
          "auditor_fees": {
            "dtype": "float64",
            "mad_center": 7117500.0,
            "mad_outliers": 701,
            "mad_scale": 4367502.0,
            "max": 150807000.0,
            "mean": 12402750.955192564,
            "min": 57.6,
            "null_rate": 0.13118726202838352,
            "nulls": 1137,
            "nunique": 6546,
            "p01": 597537.17,
            "p25": 3568000.0,
            "p50": 7117500.0,
            "p75": 14300000.0,
            "p99": 88800000.0,
            "std": 15860815.799657442
          },
          "avg_board_tenure": {
            "dtype": "float64",
            "mad_center": 7.75,
            "mad_outliers": 96,
            "mad_scale": 2.032,
            "max": 43.75,
            "mean": 8.230639376996805,
            "min": 0.0,
            "null_rate": 0.13326410522672205,
            "nulls": 1155,
            "nunique": 1063,
            "p01": 1.111,
            "p25": 5.9,
            "p50": 7.75,
            "p75": 10.0,
            "p99": 18.990870000000026,
            "std": 3.627645918077744
          },
          "avg_director_age": {
            "dtype": "float64",
            "mad_center": 62.2,
            "mad_outliers": 95,
            "mad_scale": 2.200000000000003,
            "max": 85.5,
            "mean": 61.97167884486232,
            "min": 40.0,
            "null_rate": 0.14099457713164879,
            "nulls": 1222,
            "nunique": 1164,
            "p01": 50.805,
            "p25": 59.889,
            "p50": 62.2,
            "p75": 64.333,
            "p99": 70.90952,
            "std": 3.8878235868187683
          },
          "avg_other_public_boards": {
            "dtype": "float64",
            "mad_center": 1.182,
            "mad_outliers": 104,
            "mad_scale": 0.31800000000000006,
            "max": 6.5,
            "mean": 1.2540299151659475,
            "min": 0.0,
            "null_rate": 0.22476058613130265,
            "nulls": 1948,
            "nunique": 223,
            "p01": 0.25,
            "p25": 0.889,
            "p50": 1.182,
            "p75": 1.545,
            "p99": 3.0,
            "std": 0.5639517506882608
          },
          "board_size": {
            "dtype": "float64",
            "mad_center": 11.0,
            "mad_outliers": 207,
            "mad_scale": 1.0,
            "max": 32.0,
            "mean": 10.818469015795868,
            "min": 1.0,
            "null_rate": 0.05042113764855198,
            "nulls": 437,
            "nunique": 31,
            "p01": 6.0,
            "p25": 9.0,
            "p50": 11.0,
            "p75": 12.0,
            "p99": 18.0,
            "std": 2.454861495910196
          },
          "ceo_age": {
            "dtype": "float64",
            "mad_center": 58.0,
            "mad_outliers": 59,
            "mad_scale": 4.0,
            "max": 95.0,
            "mean": 57.51938895417156,
            "min": 28.0,
            "null_rate": 0.21449175031729548,
            "nulls": 1859,
            "nunique": 67,
            "p01": 40.07000000000001,
            "p25": 53.0,
            "p50": 58.0,
            "p75": 62.0,
            "p99": 75.0,
            "std": 6.8617423625522385
          },
          "ceo_all_other_comp": {
            "dtype": "float64",
            "mad_center": 126612.0,
            "mad_outliers": 702,
            "mad_scale": 114284.0,
            "max": 68484271.0,
            "mean": 422500.4276324845,
            "min": -758687.0,
            "null_rate": 0.16176300911503405,
            "nulls": 1402,
            "nunique": 6512,
            "p01": 0.0,
            "p25": 28711.0,
            "p50": 126612.0,
            "p75": 350608.0,
            "p99": 4314186.239999999,
            "std": 2035616.316831769
          },
          "ceo_bonus": {
            "dtype": "float64",
            "mad_center": 0.0,
            "mad_outliers": 444,
            "mad_scale": 498669.3891720623,
            "max": 90000000.0,
            "mean": 498669.3891720623,
            "min": 0.0,
            "null_rate": 0.16245529018114688,
            "nulls": 1408,
            "nunique": 1013,
            "p01": 0.0,
            "p25": 0.0,
            "p50": 0.0,
            "p75": 0.0,
            "p99": 8252524.360000009,
            "std": 2065128.27284163
          },
          "ceo_equity_pay_pct": {
            "dtype": "float64",
            "mad_center": 0.62,
            "mad_outliers": 6,
            "mad_scale": 0.138,
            "max": 3.082,
            "mean": 0.5522062932320629,
            "min": -0.241,
            "null_rate": 0.1493019499250029,
            "nulls": 1294,
            "nunique": 894,
            "p01": 0.0,
            "p25": 0.432,
            "p50": 0.62,
            "p75": 0.732,
            "p99": 0.9752799999999997,
            "std": 0.2670862018774359
          },
          "ceo_is_board_chair": {
            "dtype": "float64",
            "mad_center": 0.0,
            "mad_outliers": 0,
            "mad_scale": 0.4852922610781467,
            "max": 1.0,
            "mean": 0.4852922610781467,
            "min": 0.0,
            "null_rate": 0.08607361255336334,
            "nulls": 746,
            "nunique": 2,
            "p01": 0.0,
            "p25": 0.0,
            "p50": 0.0,
            "p75": 1.0,
            "p99": 1.0,
            "std": 0.4998151866035545
          },
          "ceo_is_founder": {
            "dtype": "float64",
            "mad_center": 0.0,
            "mad_outliers": 565,
            "mad_scale": 0.07949908540875193,
            "max": 1.0,
            "mean": 0.07949908540875193,
            "min": 0.0,
            "null_rate": 0.17999307718933888,
            "nulls": 1560,
            "nunique": 2,
            "p01": 0.0,
            "p25": 0.0,
            "p50": 0.0,
            "p75": 0.0,
            "p99": 1.0,
            "std": 0.2705351715099743
          },
          "ceo_name_proxy": {
            "dtype": "str",
            "null_rate": 0.05157493942540672,
            "nulls": 447,
            "nunique": 1415
          },
          "ceo_non_equity_incentive": {
            "dtype": "float64",
            "mad_center": 1812135.0,
            "mad_outliers": 188,
            "mad_scale": 1207408.5,
            "max": 22533686.0,
            "mean": 2271434.9676724137,
            "min": 0.0,
            "null_rate": 0.1969539633091035,
            "nulls": 1707,
            "nunique": 4746,
            "p01": 0.0,
            "p25": 702862.5,
            "p50": 1812135.0,
            "p75": 3145093.5,
            "p99": 10921518.749999989,
            "std": 2308862.7983171833
          },
          "ceo_option_awards": {
            "dtype": "float64",
            "mad_center": 624996.0,
            "mad_outliers": 1052,
            "mad_scale": 624996.0,
            "max": 2283988504.0,
            "mean": 2737495.7726550247,
            "min": 0.0,
            "null_rate": 0.22037613937925465,
            "nulls": 1910,
            "nunique": 3497,
            "p01": 0.0,
            "p25": 0.0,
            "p50": 624996.0,
            "p75": 2699044.0,
            "p99": 19314527.399999727,
            "std": 31845773.434080057
          },
          "ceo_ownership_pct": {
            "dtype": "float64",
            "mad_center": 0.018,
            "mad_outliers": 396,
            "mad_scale": 0.0131,
            "max": 1.0,
            "mean": 0.07759218547960463,
            "min": 0.0,
            "null_rate": 0.7607015114803277,
            "nulls": 6593,
            "nunique": 910,
            "p01": 0.0001,
            "p25": 0.0106,
            "p50": 0.018,
            "p75": 0.051,
            "p99": 0.82,
            "std": 0.15644858259782182
          },
          "ceo_pay_ratio": {
            "dtype": "float64",
            "mad_center": 179.0,
            "mad_outliers": 323,
            "mad_scale": 86.0,
            "max": 246804.0,
            "mean": 400.40350620182835,
            "min": 0.0,
            "null_rate": 0.5508249682704511,
            "nulls": 4774,
            "nunique": 1026,
            "p01": 3.6760000000000006,
            "p25": 111.0,
            "p50": 179.0,
            "p75": 301.0,
            "p99": 1935.3199999999997,
            "std": 5079.527556573672
          },
          "ceo_salary": {
            "dtype": "float64",
            "mad_center": 1049976.0,
            "mad_outliers": 128,
            "mad_scale": 249976.0,
            "max": 9363600.0,
            "mean": 1080923.0047462168,
            "min": 0.0,
            "null_rate": 0.13845621322256838,
            "nulls": 1200,
            "nunique": 3322,
            "p01": 1.0,
            "p25": 847708.0,
            "p50": 1049976.0,
            "p75": 1319231.5,
            "p99": 2999971.62,
            "std": 489921.55423257843
          },
          "ceo_since_year": {
            "dtype": "float64",
            "mad_center": 2012.0,
            "mad_outliers": 47,
            "mad_scale": 6.0,
            "max": 2026.0,
            "mean": 2010.0671452071183,
            "min": 1966.0,
            "null_rate": 0.2284527518172378,
            "nulls": 1980,
            "nunique": 53,
            "p01": 1985.0,
            "p25": 2005.0,
            "p50": 2012.0,
            "p75": 2017.0,
            "p99": 2025.0,
            "std": 9.087131422708115
          },
          "ceo_stock_awards": {
            "dtype": "float64",
            "mad_center": 5099681.0,
            "mad_outliers": 239,
            "mad_scale": 3426477.0,
            "max": 813178818.0,
            "mean": 7379158.642823951,
            "min": -6400156.0,
            "null_rate": 0.16568593515634014,
            "nulls": 1436,
            "nunique": 6186,
            "p01": 0.0,
            "p25": 2009766.0,
            "p50": 5099681.0,
            "p75": 9143790.5,
            "p99": 44530968.09999993,
            "std": 18451488.09175131
          },
          "ceo_total_comp": {
            "dtype": "float64",
            "mad_center": 11722163.5,
            "mad_outliers": 207,
            "mad_scale": 4908784.0,
            "max": 2284044884.0,
            "mean": 15009867.002260162,
            "min": 0.0,
            "null_rate": 0.14849428868120457,
            "nulls": 1287,
            "nunique": 7309,
            "p01": 103433.34000000003,
            "p25": 7464110.5,
            "p50": 11722163.5,
            "p75": 17529080.0,
            "p99": 67282362.51,
            "std": 36875304.22899395
          },
          "classified_board": {
            "dtype": "float64",
            "mad_center": 0.0,
            "mad_outliers": 1263,
            "mad_scale": 0.14758121056321571,
            "max": 1.0,
            "mean": 0.14758121056321571,
            "min": 0.0,
            "null_rate": 0.012576439367716626,
            "nulls": 109,
            "nunique": 2,
            "p01": 0.0,
            "p25": 0.0,
            "p50": 0.0,
            "p75": 0.0,
            "p99": 1.0,
            "std": 0.35470508649081134
          },
          "company_name": {
            "dtype": "str",
            "null_rate": 0.055036344755970926,
            "nulls": 477,
            "nunique": 914
          },
          "def14a_json": {
            "dtype": "str",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 8271
          },
          "dual_class_shares": {
            "dtype": "float64",
            "mad_center": 0.0,
            "mad_outliers": 770,
            "mad_scale": 0.08997429305912596,
            "max": 1.0,
            "mean": 0.08997429305912596,
            "min": 0.0,
            "null_rate": 0.012576439367716626,
            "nulls": 109,
            "nunique": 2,
            "p01": 0.0,
            "p25": 0.0,
            "p50": 0.0,
            "p75": 0.0,
            "p99": 1.0,
            "std": 0.28616164714280967
          },
          "fiscal_year_extract": {
            "dtype": "float64",
            "mad_center": 2017.0,
            "mad_outliers": 0,
            "mad_scale": 5.0,
            "max": 2026.0,
            "mean": 2016.2320106783156,
            "min": 1999.0,
            "null_rate": 0.04915195569401177,
            "nulls": 426,
            "nunique": 28,
            "p01": 2000.0,
            "p25": 2012.0,
            "p50": 2017.0,
            "p75": 2021.0,
            "p99": 2025.0,
            "std": 6.324904500440444
          },
          "independent_chair": {
            "dtype": "float64",
            "mad_center": 0.0,
            "mad_outliers": 0,
            "mad_scale": 0.3084063276245261,
            "max": 1.0,
            "mean": 0.3084063276245261,
            "min": 0.0,
            "null_rate": 0.11745702088381216,
            "nulls": 1018,
            "nunique": 2,
            "p01": 0.0,
            "p25": 0.0,
            "p50": 0.0,
            "p75": 1.0,
            "p99": 1.0,
            "std": 0.4618655142856151
          },
          "insider_ownership_pct": {
            "dtype": "float64",
            "mad_center": 0.0254,
            "mad_outliers": 1032,
            "mad_scale": 0.016399999999999998,
            "max": 1.0,
            "mean": 0.08539824748004832,
            "min": 0.0,
            "null_rate": 0.42690665743625245,
            "nulls": 3700,
            "nunique": 1474,
            "p01": 0.0012622172280000001,
            "p25": 0.0126,
            "p50": 0.0254,
            "p75": 0.086,
            "p99": 0.693736,
            "std": 0.1457838211397394
          },
          "lead_independent_director": {
            "dtype": "float64",
            "mad_center": 1.0,
            "mad_outliers": 0,
            "mad_scale": 0.2864454350338451,
            "max": 1.0,
            "mean": 0.7135545649661549,
            "min": 0.0,
            "null_rate": 0.3011422637590862,
            "nulls": 2610,
            "nunique": 2,
            "p01": 0.0,
            "p25": 0.0,
            "p50": 1.0,
            "p75": 1.0,
            "p99": 1.0,
            "std": 0.4521373668660428
          },
          "majority_voting": {
            "dtype": "float64",
            "mad_center": 0.0,
            "mad_outliers": 0,
            "mad_scale": 0.4053295932678822,
            "max": 1.0,
            "mean": 0.4053295932678822,
            "min": 0.0,
            "null_rate": 0.012807199723087574,
            "nulls": 111,
            "nunique": 2,
            "p01": 0.0,
            "p25": 0.0,
            "p50": 0.0,
            "p75": 1.0,
            "p99": 1.0,
            "std": 0.490984408239871
          },
          "median_employee_pay": {
            "dtype": "float64",
            "mad_center": 76519.0,
            "mad_outliers": 93,
            "mad_scale": 28658.0,
            "max": 598414.0,
            "mean": 89658.43720683407,
            "min": 5091.0,
            "null_rate": 0.5542863736010153,
            "nulls": 4804,
            "nunique": 3801,
            "p01": 10179.98,
            "p25": 54074.0,
            "p50": 76519.0,
            "p75": 116145.5,
            "p99": 259797.82000000015,
            "std": 54716.334299530485
          },
          "n_directors": {
            "dtype": "float64",
            "mad_center": 11.0,
            "mad_outliers": 233,
            "mad_scale": 1.0,
            "max": 33.0,
            "mean": 10.706254563154053,
            "min": 1.0,
            "null_rate": 0.05180569978077766,
            "nulls": 449,
            "nunique": 30,
            "p01": 5.0,
            "p25": 9.0,
            "p50": 11.0,
            "p75": 12.0,
            "p99": 18.0,
            "std": 2.472162121750774
          },
          "n_five_percent_holders": {
            "dtype": "float64",
            "mad_center": 3.0,
            "mad_outliers": 56,
            "mad_scale": 1.0,
            "max": 23.0,
            "mean": 3.2616738964021432,
            "min": 0.0,
            "null_rate": 0.09565016730125764,
            "nulls": 829,
            "nunique": 21,
            "p01": 1.0,
            "p25": 2.0,
            "p50": 3.0,
            "p75": 4.0,
            "p99": 8.0,
            "std": 1.554353451880351
          },
          "n_neos": {
            "dtype": "float64",
            "mad_center": 5.0,
            "mad_outliers": 1,
            "mad_scale": 1.2582947289345185,
            "max": 12.0,
            "mean": 4.3496932515337425,
            "min": 1.0,
            "null_rate": 0.07845852082612208,
            "nulls": 680,
            "nunique": 12,
            "p01": 1.0,
            "p25": 4.0,
            "p50": 5.0,
            "p75": 5.0,
            "p99": 8.0,
            "std": 1.9384566459822874
          },
          "n_technology_directors": {
            "dtype": "float64",
            "mad_center": 3.0,
            "mad_outliers": 1,
            "mad_scale": 2.0,
            "max": 14.0,
            "mean": 3.745250255362615,
            "min": 0.0,
            "null_rate": 0.43521403022960653,
            "nulls": 3772,
            "nunique": 15,
            "p01": 1.0,
            "p25": 2.0,
            "p50": 3.0,
            "p75": 5.0,
            "p99": 10.0,
            "std": 2.3173501641956458
          },
          "pct_female_directors": {
            "dtype": "float64",
            "mad_center": 0.235,
            "mad_outliers": 8,
            "mad_scale": 0.092,
            "max": 1.0,
            "mean": 0.238534039148098,
            "min": 0.0,
            "null_rate": 0.06276681666089766,
            "nulls": 544,
            "nunique": 88,
            "p01": 0.0,
            "p25": 0.154,
            "p50": 0.235,
            "p75": 0.333,
            "p99": 0.545,
            "std": 0.1243381450282205
          },
          "pct_independent_directors": {
            "dtype": "float64",
            "mad_center": 0.857,
            "mad_outliers": 1090,
            "mad_scale": 0.06000000000000005,
            "max": 2.75,
            "mean": 0.7429294165098144,
            "min": 0.0,
            "null_rate": 0.1418022383754471,
            "nulls": 1229,
            "nunique": 116,
            "p01": 0.0,
            "p25": 0.714,
            "p50": 0.857,
            "p75": 0.9,
            "p99": 0.9544100000000008,
            "std": 0.2614003645574829
          },
          "pct_technology_directors": {
            "dtype": "float64",
            "mad_center": 0.308,
            "mad_outliers": 0,
            "mad_scale": 0.154,
            "max": 1.0,
            "mean": 0.36005188968335033,
            "min": 0.0,
            "null_rate": 0.43521403022960653,
            "nulls": 3772,
            "nunique": 94,
            "p01": 0.067,
            "p25": 0.182,
            "p50": 0.308,
            "p75": 0.5,
            "p99": 1.0,
            "std": 0.23025324751991133
          },
          "period": {
            "dtype": "datetime64[us]",
            "null_rate": 0.044536748586592824,
            "nulls": 386,
            "nunique": 2945
          },
          "poison_pill": {
            "dtype": "float64",
            "mad_center": 0.0,
            "mad_outliers": 33,
            "mad_scale": 0.0038560411311053984,
            "max": 1.0,
            "mean": 0.0038560411311053984,
            "min": 0.0,
            "null_rate": 0.012576439367716626,
            "nulls": 109,
            "nunique": 2,
            "p01": 0.0,
            "p25": 0.0,
            "p50": 0.0,
            "p75": 0.0,
            "p99": 0.0,
            "std": 0.06198081130703718
          },
          "say_on_pay_support_pct": {
            "dtype": "float64",
            "mad_center": 0.94,
            "mad_outliers": 418,
            "mad_scale": 0.027000000000000024,
            "max": 1.0,
            "mean": 0.9104599895659431,
            "min": 0.111,
            "null_rate": 0.44709818853121036,
            "nulls": 3875,
            "nunique": 447,
            "p01": 0.46,
            "p25": 0.91,
            "p50": 0.94,
            "p75": 0.96,
            "p99": 0.99,
            "std": 0.10146342545305989
          },
          "technology_committee": {
            "dtype": "float64",
            "mad_center": 0.0,
            "mad_outliers": 838,
            "mad_scale": 0.10511791269443051,
            "max": 1.0,
            "mean": 0.10511791269443051,
            "min": 0.0,
            "null_rate": 0.08018922349140417,
            "nulls": 695,
            "nunique": 2,
            "p01": 0.0,
            "p25": 0.0,
            "p50": 0.0,
            "p75": 0.0,
            "p99": 1.0,
            "std": 0.30672453182299964
          },
          "ticker": {
            "dtype": "str",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 497
          },
          "total_neo_comp": {
            "dtype": "float64",
            "mad_center": 24483437.0,
            "mad_outliers": 310,
            "mad_scale": 11555269.5,
            "max": 2284044884.0,
            "mean": 32466221.34738683,
            "min": 1.0,
            "null_rate": 0.14872504903657552,
            "nulls": 1289,
            "nunique": 7357,
            "p01": 1719134.1199999999,
            "p25": 14555007.0,
            "p50": 24483437.0,
            "p75": 38968712.0,
            "p99": 146451536.01999947,
            "std": 47601051.766766444
          }
        },
        "kind": "extract",
        "pk": [
          "ticker",
          "accession_number"
        ],
        "pk_checked_cols": [
          "ticker",
          "accession_number"
        ],
        "pk_checked_rows": 8667,
        "pk_complete": true,
        "pk_duplicate_rows": 0,
        "pk_missing_cols": [],
        "rows": 8667,
        "sample_date_max": "2026-08-17",
        "sample_date_min": "1995-09-13",
        "sampled_rows": 8667,
        "scope": {
          "limit": null,
          "since": null,
          "tickers": null
        },
        "table": "def14a_llm"
      },
      "sec_8k": {
        "columns": [
          "ticker",
          "cik",
          "accession_number",
          "form",
          "filing_date",
          "period_of_report",
          "n_items",
          "is_amendment",
          "has_earnings",
          "has_press_release",
          "primary_document",
          "item",
          "item_tag",
          "item_text"
        ],
        "date_col": "filing_date",
        "date_max": "2026-09-01",
        "date_min": "1995-09-01",
        "exists": true,
        "fields": {
          "accession_number": {
            "dtype": "str",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 125687
          },
          "cik": {
            "dtype": "str",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 406
          },
          "filing_date": {
            "dtype": "object",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 7651
          },
          "form": {
            "dtype": "str",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 2
          },
          "has_earnings": {
            "dtype": "float64",
            "mad_center": 0.0,
            "mad_outliers": 0,
            "mad_scale": 0.22695138251445318,
            "max": 1.0,
            "mean": 0.22695138251445318,
            "min": 0.0,
            "null_rate": 0.005807740184678933,
            "nulls": 1451,
            "nunique": 2,
            "p01": 0.0,
            "p25": 0.0,
            "p50": 0.0,
            "p75": 0.0,
            "p99": 1.0,
            "std": 0.4188617418961959
          },
          "has_press_release": {
            "dtype": "float64",
            "mad_center": 0.0,
            "mad_outliers": 0,
            "mad_scale": 0.2913345250173116,
            "max": 1.0,
            "mean": 0.2913345250173116,
            "min": 0.0,
            "null_rate": 0.005807740184678933,
            "nulls": 1451,
            "nunique": 2,
            "p01": 0.0,
            "p25": 0.0,
            "p50": 0.0,
            "p75": 1.0,
            "p99": 1.0,
            "std": 0.4543782023249083
          },
          "is_amendment": {
            "dtype": "float64",
            "mad_center": 0.0,
            "mad_outliers": 4386,
            "mad_scale": 0.01755530561681723,
            "max": 1.0,
            "mean": 0.01755530561681723,
            "min": 0.0,
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 2,
            "p01": 0.0,
            "p25": 0.0,
            "p50": 0.0,
            "p75": 0.0,
            "p99": 1.0,
            "std": 0.1313285418129598
          },
          "item": {
            "dtype": "str",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 41
          },
          "item_tag": {
            "dtype": "str",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 30
          },
          "item_text": {
            "dtype": "str",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 234406
          },
          "n_items": {
            "dtype": "int64",
            "mad_center": 2.0,
            "mad_outliers": 5719,
            "mad_scale": 0.5549854106044293,
            "max": 12.0,
            "mean": 2.304295966602492,
            "min": 1.0,
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 10,
            "p01": 1.0,
            "p25": 2.0,
            "p50": 2.0,
            "p75": 3.0,
            "p99": 5.0,
            "std": 0.9168971427571354
          },
          "period_of_report": {
            "dtype": "object",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 8989
          },
          "primary_document": {
            "dtype": "str",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 101934
          },
          "ticker": {
            "dtype": "str",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 406
          }
        },
        "kind": "extract",
        "pk": [
          "ticker",
          "accession_number",
          "item"
        ],
        "pk_checked_cols": [
          "ticker",
          "accession_number",
          "item"
        ],
        "pk_checked_rows": 249839,
        "pk_complete": true,
        "pk_duplicate_rows": 0,
        "pk_missing_cols": [],
        "rows": 249839,
        "sample_date_max": "2026-09-01",
        "sample_date_min": "1995-09-01",
        "sampled_rows": 249839,
        "scope": {
          "limit": null,
          "since": null,
          "tickers": null
        },
        "table": "sec_8k"
      },
      "sec_def14a": {
        "columns": [
          "ticker",
          "cik",
          "accession_number",
          "form",
          "filing_date",
          "period_of_report",
          "company_name",
          "has_xbrl",
          "has_individual_executive_data",
          "peo_name",
          "peo_total_comp",
          "peo_actually_paid_comp",
          "neo_avg_total_comp",
          "neo_avg_actually_paid_comp",
          "total_shareholder_return",
          "peer_group_tsr",
          "net_income",
          "company_selected_measure_name",
          "company_selected_measure_value",
          "insider_trading_policy_adopted",
          "award_timing_mnpi_considered",
          "award_dates_predetermined",
          "mnpi_disclosure_timed_for_comp_value",
          "ceo_pay_ratio_ceo_comp",
          "ceo_pay_ratio_median_employee_comp",
          "ceo_pay_ratio",
          "auditor_name",
          "audit_fiscal_year_current",
          "audit_fiscal_year_prior",
          "audit_fees_current",
          "audit_fees_prior",
          "audit_related_fees_current",
          "audit_related_fees_prior",
          "tax_fees_current",
          "tax_fees_prior",
          "other_fees_current",
          "other_fees_prior",
          "total_fees_current",
          "total_fees_prior",
          "n_voting_proposals",
          "n_say_on_pay_proposals",
          "n_director_election_proposals",
          "n_auditor_ratification_proposals",
          "n_equity_plan_proposals",
          "n_shareholder_proposals",
          "n_board_against_recommendations"
        ],
        "date_col": "filing_date",
        "date_max": "2026-07-24",
        "date_min": "1995-09-13",
        "exists": true,
        "fields": {
          "accession_number": {
            "dtype": "str",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 3662
          },
          "audit_fees_current": {
            "dtype": "float64",
            "mad_center": 8000000.0,
            "mad_outliers": 11,
            "mad_scale": 4484000.0,
            "max": 9800000000000.0,
            "mean": 38158826277.41634,
            "min": 75000.0,
            "null_rate": 0.929819770617149,
            "nulls": 3405,
            "nunique": 242,
            "p01": 98440.0,
            "p25": 4452000.0,
            "p50": 8000000.0,
            "p75": 14200000.0,
            "p99": 932559999.9999956,
            "std": 611305568735.4377
          },
          "audit_fees_prior": {
            "dtype": "float64",
            "mad_center": 7221000.0,
            "mad_outliers": 10,
            "mad_scale": 4779000.0,
            "max": 11580000000000.0,
            "mean": 45797199514.411064,
            "min": 6000.0,
            "null_rate": 0.9309120699071546,
            "nulls": 3409,
            "nunique": 240,
            "p01": 140256.0,
            "p25": 4000000.0,
            "p50": 7221000.0,
            "p75": 13537500.0,
            "p99": 1018879999.9999802,
            "std": 728026714995.0686
          },
          "audit_fiscal_year_current": {
            "dtype": "float64",
            "mad_center": 2015.0,
            "mad_outliers": 0,
            "mad_scale": 5.0,
            "max": 2026.0,
            "mean": 2014.9598583234947,
            "min": 2002.0,
            "null_rate": 0.7687056253413436,
            "nulls": 2815,
            "nunique": 25,
            "p01": 2002.0,
            "p25": 2010.0,
            "p50": 2015.0,
            "p75": 2021.0,
            "p99": 2025.0,
            "std": 6.572979575504445
          },
          "audit_fiscal_year_prior": {
            "dtype": "float64",
            "mad_center": 2014.0,
            "mad_outliers": 0,
            "mad_scale": 5.0,
            "max": 2025.0,
            "mean": 2013.9492325855963,
            "min": 2001.0,
            "null_rate": 0.7687056253413436,
            "nulls": 2815,
            "nunique": 25,
            "p01": 2001.0,
            "p25": 2009.0,
            "p50": 2014.0,
            "p75": 2020.0,
            "p99": 2024.0,
            "std": 6.574434420135162
          },
          "audit_related_fees_current": {
            "dtype": "float64",
            "mad_center": 678600.0,
            "mad_outliers": 37,
            "mad_scale": 634600.0,
            "max": 1650000000000.0,
            "mean": 5916640256.827957,
            "min": 0.0,
            "null_rate": 0.923812124522119,
            "nulls": 3383,
            "nunique": 234,
            "p01": 0.0,
            "p25": 110292.5,
            "p50": 678600.0,
            "p75": 1859740.5,
            "p99": 23729679.999999993,
            "std": 98782756344.53842
          },
          "audit_related_fees_prior": {
            "dtype": "float64",
            "mad_center": 575667.0,
            "mad_outliers": 45,
            "mad_scale": 546167.0,
            "max": 1607000000000.0,
            "mean": 5889004358.424909,
            "min": 0.0,
            "null_rate": 0.9254505734571272,
            "nulls": 3389,
            "nunique": 219,
            "p01": 0.0,
            "p25": 90000.0,
            "p50": 575667.0,
            "p75": 2000000.0,
            "p99": 23741319.999999996,
            "std": 97259909291.46788
          },
          "auditor_name": {
            "dtype": "str",
            "null_rate": 0.9778809393773894,
            "nulls": 3581,
            "nunique": 10
          },
          "award_dates_predetermined": {
            "dtype": "float64",
            "mad_center": 1.0,
            "mad_outliers": 22,
            "mad_scale": 0.1286549707602339,
            "max": 1.0,
            "mean": 0.8713450292397661,
            "min": 0.0,
            "null_rate": 0.9533042053522666,
            "nulls": 3491,
            "nunique": 2,
            "p01": 0.0,
            "p25": 1.0,
            "p50": 1.0,
            "p75": 1.0,
            "p99": 1.0,
            "std": 0.3358009796019315
          },
          "award_timing_mnpi_considered": {
            "dtype": "float64",
            "mad_center": 0.0,
            "mad_outliers": 0,
            "mad_scale": 0.39344262295081966,
            "max": 1.0,
            "mean": 0.39344262295081966,
            "min": 0.0,
            "null_rate": 0.9500273074822502,
            "nulls": 3479,
            "nunique": 2,
            "p01": 0.0,
            "p25": 0.0,
            "p50": 0.0,
            "p75": 1.0,
            "p99": 1.0,
            "std": 0.48985381956960233
          },
          "ceo_pay_ratio": {
            "dtype": "float64",
            "mad_center": 174.0,
            "mad_outliers": 76,
            "mad_scale": 85.0,
            "max": 5294.0,
            "mean": 255.6665587290832,
            "min": 0.0,
            "null_rate": 0.7304751501911524,
            "nulls": 2675,
            "nunique": 482,
            "p01": 1.9297216085578444,
            "p25": 103.0,
            "p50": 174.0,
            "p75": 277.0,
            "p99": 1663.3399999999997,
            "std": 335.75874770169963
          },
          "ceo_pay_ratio_ceo_comp": {
            "dtype": "float64",
            "mad_center": 13476051.0,
            "mad_outliers": 50,
            "mad_scale": 5438205.5,
            "max": 66272534010.0,
            "mean": 145473490.72727272,
            "min": 48.0,
            "null_rate": 0.7717094483888586,
            "nulls": 2826,
            "nunique": 836,
            "p01": 13391.55,
            "p25": 8614796.25,
            "p50": 13476051.0,
            "p75": 19631615.75,
            "p99": 1513725778.599998,
            "std": 2353577825.037878
          },
          "ceo_pay_ratio_median_employee_comp": {
            "dtype": "float64",
            "mad_center": 71445.00390625,
            "mad_outliers": 38,
            "mad_scale": 31711.541321215987,
            "max": 83361678.0,
            "mean": 477597.9795485013,
            "min": 5.365364308342133,
            "null_rate": 0.7738940469688694,
            "nulls": 2834,
            "nunique": 827,
            "p01": 175.27,
            "p25": 50607.67403846154,
            "p50": 71445.00390625,
            "p75": 117529.44132833857,
            "p99": 15340060.000000004,
            "std": 3560814.5787564605
          },
          "cik": {
            "dtype": "str",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 151
          },
          "company_name": {
            "dtype": "str",
            "null_rate": 0.8476242490442382,
            "nulls": 3104,
            "nunique": 179
          },
          "company_selected_measure_name": {
            "dtype": "str",
            "null_rate": 0.8582741671217914,
            "nulls": 3143,
            "nunique": 159
          },
          "company_selected_measure_value": {
            "dtype": "float64",
            "mad_center": 23.27,
            "mad_outliers": 241,
            "mad_scale": 23.192999999999998,
            "max": 1696714000000.0,
            "mean": 14205350580.320578,
            "min": -14310000000.0,
            "null_rate": 0.8506280720917532,
            "nulls": 3115,
            "nunique": 524,
            "p01": -13.567160000000001,
            "p25": 5.08,
            "p50": 23.27,
            "p75": 2335850000.0,
            "p99": 258998519999.99017,
            "std": 121030573194.90338
          },
          "filing_date": {
            "dtype": "object",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 1931
          },
          "form": {
            "dtype": "str",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 1
          },
          "has_individual_executive_data": {
            "dtype": "float64",
            "mad_center": 0.0,
            "mad_outliers": 242,
            "mad_scale": 0.06608410704533042,
            "max": 1.0,
            "mean": 0.06608410704533042,
            "min": 0.0,
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 2,
            "p01": 0.0,
            "p25": 0.0,
            "p50": 0.0,
            "p75": 0.0,
            "p99": 1.0,
            "std": 0.24846298678690043
          },
          "has_xbrl": {
            "dtype": "float64",
            "mad_center": 0.0,
            "mad_outliers": 558,
            "mad_scale": 0.1523757509557619,
            "max": 1.0,
            "mean": 0.1523757509557619,
            "min": 0.0,
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 2,
            "p01": 0.0,
            "p25": 0.0,
            "p50": 0.0,
            "p75": 0.0,
            "p99": 1.0,
            "std": 0.35943380578255213
          },
          "insider_trading_policy_adopted": {
            "dtype": "float64",
            "mad_center": 1.0,
            "mad_outliers": 1,
            "mad_scale": 0.003663003663003663,
            "max": 1.0,
            "mean": 0.9963369963369964,
            "min": 0.0,
            "null_rate": 0.9254505734571272,
            "nulls": 3389,
            "nunique": 2,
            "p01": 1.0,
            "p25": 1.0,
            "p50": 1.0,
            "p75": 1.0,
            "p99": 1.0,
            "std": 0.06052275326688024
          },
          "mnpi_disclosure_timed_for_comp_value": {
            "dtype": "float64",
            "mad_center": 0.0,
            "mad_outliers": 2,
            "mad_scale": 0.011363636363636364,
            "max": 1.0,
            "mean": 0.011363636363636364,
            "min": 0.0,
            "null_rate": 0.9519388312397596,
            "nulls": 3486,
            "nunique": 2,
            "p01": 0.0,
            "p25": 0.0,
            "p50": 0.0,
            "p75": 0.0,
            "p99": 0.25,
            "std": 0.10629534937475535
          },
          "n_auditor_ratification_proposals": {
            "dtype": "float64",
            "mad_center": 1.0,
            "mad_outliers": 0,
            "mad_scale": 0.4612233752048061,
            "max": 2.0,
            "mean": 0.5415073730202076,
            "min": 0.0,
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 3,
            "p01": 0.0,
            "p25": 0.0,
            "p50": 1.0,
            "p75": 1.0,
            "p99": 1.0,
            "std": 0.5010752926124811
          },
          "n_board_against_recommendations": {
            "dtype": "float64",
            "mad_center": 0.0,
            "mad_outliers": 325,
            "mad_scale": 0.3560895685417804,
            "max": 11.0,
            "mean": 0.3560895685417804,
            "min": 0.0,
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 11,
            "p01": 0.0,
            "p25": 0.0,
            "p50": 0.0,
            "p75": 0.0,
            "p99": 4.0,
            "std": 0.9447535086962124
          },
          "n_director_election_proposals": {
            "dtype": "float64",
            "mad_center": 1.0,
            "mad_outliers": 0,
            "mad_scale": 0.4489350081922447,
            "max": 2.0,
            "mean": 0.5636264336428182,
            "min": 0.0,
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 3,
            "p01": 0.0,
            "p25": 0.0,
            "p50": 1.0,
            "p75": 1.0,
            "p99": 1.0,
            "std": 0.5085112894356776
          },
          "n_equity_plan_proposals": {
            "dtype": "float64",
            "mad_center": 0.0,
            "mad_outliers": 235,
            "mad_scale": 0.06526488257782632,
            "max": 2.0,
            "mean": 0.06526488257782632,
            "min": 0.0,
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 3,
            "p01": 0.0,
            "p25": 0.0,
            "p50": 0.0,
            "p75": 0.0,
            "p99": 1.0,
            "std": 0.2514104941422405
          },
          "n_say_on_pay_proposals": {
            "dtype": "float64",
            "mad_center": 0.0,
            "mad_outliers": 75,
            "mad_scale": 0.3623702894593119,
            "max": 2.0,
            "mean": 0.3623702894593119,
            "min": 0.0,
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 3,
            "p01": 0.0,
            "p25": 0.0,
            "p50": 0.0,
            "p75": 1.0,
            "p99": 2.0,
            "std": 0.5216259081563668
          },
          "n_shareholder_proposals": {
            "dtype": "float64",
            "mad_center": 0.0,
            "mad_outliers": 243,
            "mad_scale": 0.3148552703440743,
            "max": 9.0,
            "mean": 0.3148552703440743,
            "min": 0.0,
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 10,
            "p01": 0.0,
            "p25": 0.0,
            "p50": 0.0,
            "p75": 0.0,
            "p99": 4.0,
            "std": 0.8280080574220109
          },
          "n_voting_proposals": {
            "dtype": "float64",
            "mad_center": 3.0,
            "mad_outliers": 0,
            "mad_scale": 2.0,
            "max": 13.0,
            "mean": 2.8612779901693064,
            "min": 0.0,
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 14,
            "p01": 0.0,
            "p25": 0.0,
            "p50": 3.0,
            "p75": 4.0,
            "p99": 9.0,
            "std": 2.4164559900323974
          },
          "neo_avg_actually_paid_comp": {
            "dtype": "float64",
            "mad_center": 5546679.0,
            "mad_outliers": 66,
            "mad_scale": 3237716.0,
            "max": 332352803.0,
            "mean": 10432962.145421904,
            "min": -32708200.0,
            "null_rate": 0.8478973238667394,
            "nulls": 3105,
            "nunique": 557,
            "p01": -9972071.6,
            "p25": 2964564.0,
            "p50": 5546679.0,
            "p75": 11072167.0,
            "p99": 98713221.71999949,
            "std": 21787126.70269895
          },
          "neo_avg_total_comp": {
            "dtype": "float64",
            "mad_center": 5142656.0,
            "mad_outliers": 52,
            "mad_scale": 1886725.0,
            "max": 61434191.0,
            "mean": 7048265.97486535,
            "min": 4518.0,
            "null_rate": 0.8478973238667394,
            "nulls": 3105,
            "nunique": 557,
            "p01": 1041525.04,
            "p25": 3651796.0,
            "p50": 5142656.0,
            "p75": 7990849.0,
            "p99": 27109910.71999999,
            "std": 5819685.362259284
          },
          "net_income": {
            "dtype": "float64",
            "mad_center": 1480100000.0,
            "mad_outliers": 58,
            "mad_scale": 1073978000.0,
            "max": 1552449000000.0,
            "mean": 14437388017.52485,
            "min": -22800000000.0,
            "null_rate": 0.8626433642818132,
            "nulls": 3159,
            "nunique": 501,
            "p01": -2439560000.0,
            "p25": 637747000.0,
            "p50": 1480100000.0,
            "p75": 3694500000.0,
            "p99": 111709700000.00027,
            "std": 114066708509.27585
          },
          "other_fees_current": {
            "dtype": "float64",
            "mad_center": 29500.0,
            "mad_outliers": 63,
            "mad_scale": 29500.0,
            "max": 100000000000.0,
            "mean": 437779929.80172414,
            "min": -14000000.0,
            "null_rate": 0.9366466411796832,
            "nulls": 3430,
            "nunique": 125,
            "p01": 0.0,
            "p25": 2950.0,
            "p50": 29500.0,
            "p75": 232250.0,
            "p99": 302939999.9999997,
            "std": 6565113327.167686
          },
          "other_fees_prior": {
            "dtype": "float64",
            "mad_center": 39000.0,
            "mad_outliers": 58,
            "mad_scale": 39000.0,
            "max": 10000000000.0,
            "mean": 49024866.0,
            "min": -2000000.0,
            "null_rate": 0.9377389404696886,
            "nulls": 3434,
            "nunique": 127,
            "p01": -8885.56,
            "p25": 2000.0,
            "p50": 39000.0,
            "p75": 241953.75,
            "p99": 287959999.99999744,
            "std": 663014981.4781516
          },
          "peer_group_tsr": {
            "dtype": "float64",
            "mad_center": 146.0,
            "mad_outliers": 17,
            "mad_scale": 27.5,
            "max": 547.0,
            "mean": 158.65048833034112,
            "min": 24.4,
            "null_rate": 0.8478973238667394,
            "nulls": 3105,
            "nunique": 419,
            "p01": 64.4936,
            "p25": 122.29,
            "p50": 146.0,
            "p75": 187.18,
            "p99": 318.12799999999936,
            "std": 57.149920775311784
          },
          "peo_actually_paid_comp": {
            "dtype": "float64",
            "mad_center": 18181842.5,
            "mad_outliers": 55,
            "mad_scale": 12113384.5,
            "max": 2268000000.0,
            "mean": 33735260.631578945,
            "min": -686374419.0,
            "null_rate": 0.8547241944292736,
            "nulls": 3130,
            "nunique": 532,
            "p01": -79544781.00999999,
            "p25": 7480242.5,
            "p50": 18181842.5,
            "p75": 33613470.5,
            "p99": 339552907.4599992,
            "std": 130296662.31387492
          },
          "peo_name": {
            "dtype": "str",
            "null_rate": 0.8552703440742764,
            "nulls": 3132,
            "nunique": 213
          },
          "peo_total_comp": {
            "dtype": "float64",
            "mad_center": 16030630.0,
            "mad_outliers": 21,
            "mad_scale": 6399702.5,
            "max": 247579143.0,
            "mean": 19397865.20488722,
            "min": 12566.0,
            "null_rate": 0.8547241944292736,
            "nulls": 3130,
            "nunique": 531,
            "p01": 40018.67,
            "p25": 10394358.75,
            "p50": 16030630.0,
            "p75": 22926442.75,
            "p99": 87305303.99999966,
            "std": 20239808.917293493
          },
          "period_of_report": {
            "dtype": "object",
            "null_rate": 0.030311305297651556,
            "nulls": 111,
            "nunique": 1967
          },
          "tax_fees_current": {
            "dtype": "float64",
            "mad_center": 636914.0,
            "mad_outliers": 39,
            "mad_scale": 630414.0,
            "max": 400000000000.0,
            "mean": 1353525453.0033784,
            "min": -15000000.0,
            "null_rate": 0.9191698525395958,
            "nulls": 3366,
            "nunique": 247,
            "p01": -337703.6999999999,
            "p25": 135248.75,
            "p50": 636914.0,
            "p75": 2000000.0,
            "p99": 29550000.000000127,
            "std": 23249402030.79897
          },
          "tax_fees_prior": {
            "dtype": "float64",
            "mad_center": 743045.5,
            "mad_outliers": 33,
            "mad_scale": 726045.5,
            "max": 1608000000000.0,
            "mean": 5471572104.207483,
            "min": -39000000.0,
            "null_rate": 0.9197160021845986,
            "nulls": 3368,
            "nunique": 239,
            "p01": -351646.7799999999,
            "p25": 146945.25,
            "p50": 743045.5,
            "p75": 2000000.0,
            "p99": 30385909.999999963,
            "std": 93780336891.9212
          },
          "ticker": {
            "dtype": "str",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 151
          },
          "total_fees_current": {
            "dtype": "float64",
            "mad_center": 10337650.0,
            "mad_outliers": 14,
            "mad_scale": 5691000.0,
            "max": 11950000000000.0,
            "mean": 55867723809.060745,
            "min": 0.0,
            "null_rate": 0.9415619879847078,
            "nulls": 3448,
            "nunique": 204,
            "p01": 33238.57,
            "p25": 5497750.0,
            "p50": 10337650.0,
            "p75": 20421000.0,
            "p99": 581680000.0000002,
            "std": 816883365214.3065
          },
          "total_fees_prior": {
            "dtype": "float64",
            "mad_center": 10071500.0,
            "mad_outliers": 14,
            "mad_scale": 5675938.5,
            "max": 14805000000000.0,
            "mean": 70491680582.91905,
            "min": -6841000000.0,
            "null_rate": 0.9426542872747132,
            "nulls": 3452,
            "nunique": 202,
            "p01": 9044.269999999997,
            "p25": 5142500.0,
            "p50": 10071500.0,
            "p75": 18137438.75,
            "p99": 638259999.999998,
            "std": 1021642754051.3145
          },
          "total_shareholder_return": {
            "dtype": "float64",
            "mad_center": 144.513,
            "mad_outliers": 34,
            "mad_scale": 43.977000000000004,
            "max": 1182.35,
            "mean": 176.62424236983844,
            "min": 5.15,
            "null_rate": 0.8478973238667394,
            "nulls": 3105,
            "nunique": 497,
            "p01": 38.4,
            "p25": 107.96,
            "p50": 144.513,
            "p75": 200.51,
            "p99": 710.5839999999989,
            "std": 125.79361806366516
          }
        },
        "kind": "extract",
        "pk": [
          "ticker",
          "accession_number"
        ],
        "pk_checked_cols": [
          "ticker",
          "accession_number"
        ],
        "pk_checked_rows": 3662,
        "pk_complete": true,
        "pk_duplicate_rows": 0,
        "pk_missing_cols": [],
        "rows": 3662,
        "sample_date_max": "2026-07-24",
        "sample_date_min": "1995-09-13",
        "sampled_rows": 3662,
        "scope": {
          "limit": null,
          "since": null,
          "tickers": null
        },
        "table": "sec_def14a"
      }
    }
  },
  "scope": {
    "limit": null,
    "since": null,
    "tables": [
      "def14a_llm",
      "sec_8k",
      "sec_def14a"
    ],
    "tickers": [],
    "unknown_tables": []
  },
  "session_id": "cbba673e-f8e0-4596-b06d-a136e202dd08",
  "type": "DATA"
}
```

