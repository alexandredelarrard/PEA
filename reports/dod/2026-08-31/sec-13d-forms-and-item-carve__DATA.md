---
type: DATA
session_id: 9d1a6164-51cc-4126-953b-a46b98db87d9
generated_at: 2026-09-01T01:37:57+00:00
baseline: {head_sha: f023398b62fa78cb80446c0d7dee9aadab1d07dc}
generator: scripts/dod/data_profile.py@1
---

## 1. Scope

**SAMPLE SCOPE** — a metric without its scope is not a measurement:

- tables: sec_13d, sec_13d_transactions
- tickers: **all** (no ticker filter)
- since: **no lower bound**
- row limit per table: **none**
- full-scope tables (eligible to set the baseline): sec_13d, sec_13d_transactions

**What was asked:** implement the five-phase plan in
`reports/planning/active-tasks/2026-08-31-sec-13d-forms-and-item-carve-plan.md` — fix the
`sec_13d` ingestion outage, the Item 3/4 carve corruption, and the post-mandate false-zero risk.
Concretely: list EDGAR's post-mandate form strings, thread `-F/--full`, replace the single-anchor
item carve with a guarded union of two anchor sets, normalize item-body encoding and whitespace,
add `_is_placeholder_numerics` plus a `reporting_person_comment` column, and document (**not
execute**) the table rebuild.

**Read this profile as the BEFORE state.** The rebuild is deliberately out of scope, so the live
table still holds only the 1,666 legacy rows this work exists to replace. Every number in §3 is
the defect being fixed, not the result of fixing it — the code changes are proven by the
measurements in §4, against live EDGAR, not by this table.

## 2. Gates

| Gate | Check | Verdict | Detail |
|---|---|---|---|
| D1 | declared PK unique over the rows profiled | **PASS** | unique across 2 table(s): sec_13d, sec_13d_transactions |
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
| sec_13d | yes | 1,666 | 1,666 | 29 | ticker,accession_number,rp_seq | — | 0 | 2011-08-22 | 2024-12-16 | 2011-08-22 | 2024-12-16 |
| sec_13d_transactions | yes | 1,725 | 1,725 | 10 | ticker,accession_number,trade_seq | — | 0 | 2012-01-19 | 2024-10-30 | 2012-01-19 | 2024-10-30 |

**Fields** (worst null rate first, top 39)

| table | field | dtype | null_% | nunique | mean | std | min | p01 | p50 | p99 | max | mad_outliers |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sec_13d | aggregate_amount | object | 100 | 0 | — | — | — | — | — | — | — | — |
| sec_13d | amendment_number | object | 100 | 0 | — | — | — | — | — | — | — | — |
| sec_13d | date_of_event | object | 100 | 0 | — | — | — | — | — | — | — | — |
| sec_13d | is_group_member | object | 100 | 0 | — | — | — | — | — | — | — | — |
| sec_13d | percent_of_class | object | 100 | 0 | — | — | — | — | — | — | — | — |
| sec_13d | reporting_person_citizenship | object | 100 | 0 | — | — | — | — | — | — | — | — |
| sec_13d | shared_dispositive_power | object | 100 | 0 | — | — | — | — | — | — | — | — |
| sec_13d | shared_voting_power | object | 100 | 0 | — | — | — | — | — | — | — | — |
| sec_13d | sole_dispositive_power | object | 100 | 0 | — | — | — | — | — | — | — | — |
| sec_13d | sole_voting_power | object | 100 | 0 | — | — | — | — | — | — | — | — |
| sec_13d | type_of_reporting_person | object | 100 | 0 | — | — | — | — | — | — | — | — |
| sec_13d | item3_source_of_funds | str | 56.66 | 604 | — | — | — | — | — | — | — | — |
| sec_13d | item6_contracts_understandings | str | 42.26 | 843 | — | — | — | — | — | — | — | — |
| sec_13d | item4_purpose_of_transaction | str | 38.12 | 956 | — | — | — | — | — | — | — | — |
| sec_13d | item5_interest_in_securities | str | 18.79 | 1,344 | — | — | — | — | — | — | — | — |
| sec_13d_transactions | price_per_share | float64 | 2.38 | 794 | 65.2331 | 98.0058 | 0 | 0.17 | 45.325 | 497.71 | 893.42 | 45 |
| sec_13d_transactions | trade_date | object | 1.91 | 462 | — | — | — | — | — | — | — | — |
| sec_13d | reporting_person_cik | str | 0.66 | 226 | — | — | — | — | — | — | — | — |
| sec_13d | reporting_person_name | str | 0.66 | 241 | — | — | — | — | — | — | — | — |
| sec_13d_transactions | quantity | float64 | 0.52 | 1,193 | 312,702 | 976,471 | 4 | 71.3 | 77,668.5 | 3.21752e+06 | 1.86485e+07 | 238 |
| sec_13d | accession_number | str | 0 | 1,666 | — | — | — | — | — | — | — | — |
| sec_13d | cik | str | 0 | 162 | — | — | — | — | — | — | — | — |
| sec_13d | cusip | str | 0 | 1 | — | — | — | — | — | — | — | — |
| sec_13d | doc_url | str | 0 | 1,666 | — | — | — | — | — | — | — | — |
| sec_13d | filing_date | object | 0 | 1,216 | — | — | — | — | — | — | — | — |
| sec_13d | form | str | 0 | 2 | — | — | — | — | — | — | — | — |
| sec_13d | has_structured_data | float64 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| sec_13d | is_amendment | float64 | 0 | 2 | 0.890756 | 0.312038 | 0 | 0 | 1 | 1 | 1 | 182 |
| sec_13d | issuer_name | str | 0 | 178 | — | — | — | — | — | — | — | — |
| sec_13d | primary_document | str | 0 | 1,597 | — | — | — | — | — | — | — | — |
| sec_13d | rp_seq | int64 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| sec_13d | ticker | str | 0 | 155 | — | — | — | — | — | — | — | — |
| sec_13d_transactions | accession_number | str | 0 | 52 | — | — | — | — | — | — | — | — |
| sec_13d_transactions | cik | str | 0 | 16 | — | — | — | — | — | — | — | — |
| sec_13d_transactions | filing_date | object | 0 | 49 | — | — | — | — | — | — | — | — |
| sec_13d_transactions | reporting_person_name | str | 0 | 37 | — | — | — | — | — | — | — | — |
| sec_13d_transactions | ticker | str | 0 | 16 | — | — | — | — | — | — | — | — |
| sec_13d_transactions | trade_seq | int64 | 0 | 387 | 83.6012 | 100.754 | 0 | 0 | 37 | 368.76 | 386 | 283 |
| sec_13d_transactions | transaction_type | str | 0 | 19 | — | — | — | — | — | — | — | — |

## 4. Evidence

- baseline file: `reports/baselines/data_profile.json` (2 table(s) recorded)
- `sec_13d`: 1,666 rows, 29 cols, 1,666 sampled
- `sec_13d_transactions`: 1,725 rows, 10 cols, 1,725 sampled


**Live-EDGAR measurements (the actual verification — the table above is the un-rebuilt BEFORE):**

- **Outage confirmed and fixed.** `sec_13d.filing_date` maxes at **2024-12-16**, exactly the
  mandate changeover. With both form eras listed, **322 post-mandate filings / 1,351 reporting-person
  rows** across the 155 tickers already in the table become reachable; all were invisible before.
- **Item carve**, over 182 originals (every `SC 13D` in the table) + 200 seeded-random amendments:
  item3 contamination **3.8% → 0%** (originals) and **2.5% → 0%** (amendments); item4 coverage
  **92.9% → 98.9%** and **61.5% → 70.0%**; item3/5/6 all up; **zero regressions** on either
  population. Worst case MNST `0001341004-15-000486`: item3 **17,776 → 825 chars**, item4
  recovered from a miss to 16,872 chars.
- **Normalization**: of 1,186 carved bodies, **0** retain cp1252 `\x80-\x9f` bytes, a 3+
  box-drawing run, a non-breaking space or ragged whitespace — against 42.4% / 84.0% of raw filings
  carrying the first two. Carve coverage was bit-identical before and after.
- **False-zero guard**, over the 322-filing post-mandate backlog (`has_structured_data` is 99.9%
  true there, so the old accidental guard is fully disarmed): **19** all-zero-with-comment rows
  correctly nulled, **230** all-zero-no-comment rows correctly preserved.
- Tests: `test_fetch_8k_13d_edgar.py` **25 → 38 passed**; `test_form_registry.py` 6 passed.

## 5. Regressions, gaps and deliberate omissions

- **The rebuild was NOT executed — this is the headline gap.** Scope was code + tests only. Until
  `docs/runbook.md`'s procedure is run, the live table keeps all 1,666 defective rows and none of
  the 461-filing backlog. Everything in §3 is still the broken shape.
- **D2/D3/D5 came back N/A: no baseline existed before this run.** This run records the first one.
  That means row-count, column-loss and null-rate regressions are unguarded *for this run
  specifically* — the next profile can compare, this one had nothing to compare to. D1 (PK
  uniqueness) did run and passed on all 3,391 rows across both tables.
- **New defect found while profiling, not in the plan: `cusip` is the empty string on all 1,666
  rows** (`nunique=1`, `null_rate=0`). Because it is `''` and not NULL it reads as *fully
  populated* to every null-rate check, including D5 — a null-rate gate can never catch this
  column. The pre-mandate SGML parse leaves `security_info.cusip` empty; the post-mandate XML path
  does populate it (verified: CVNA `0001104659-25-019162` → `77664L108`), so the rebuild fixes it
  as a side effect. Not fixed here, because doing so would mean touching the pre-mandate path this
  plan deliberately leaves alone.
- **11 rows (0.66%) have a NULL `reporting_person_name`** — the no-persons fallback row. Expected
  by design, unchanged by this work.
- **17 pre-existing test failures found and FIXED** (they were not caused by this task -- first
  reproduced at clean HEAD in a detached worktree). All three causes were fixtures left behind by
  commit `10f51d5`, which moved production forward without updating the stubs that stand in for
  it: (a) `run_manifest._manifest_path` began resolving the filename through
  `context.config.local.filename.extraction` while the fake contexts carried only `paths`;
  (b) `load_cik_mapping` projects six `sp500_tickers` columns server-side while the driver
  fixture wrote two, so the SELECT raised `KeyError: 'name'`; (c) `list_filings` and
  `_process_filing` each gained a leading `context` argument that the DEF 14A stubs never took,
  which bound `since` twice. `tests/data_extract/{structure,common,utils}` now runs
  **149 passed, 3 skipped, 0 failed** (was 132 passed / 17 failed).
- **The whole-directory `tests/data_extract` run was not carried to completion** (live-network
  fundamentals/sharadar tests, past an hour). The three subdirectories that can reach the four
  files this task touches were run in full.
- **One success criterion was refined, not met literally.** "No row has `percent_of_class == 0`
  while its comment is non-empty" would require nulling 3 real rows whose 0.0% is a genuine
  rounded disclosure sitting beside non-zero share counts (CALFINCO: 18,632,216 shares against
  54,730,851,778,811 outstanding). The criterion's stated intent — no row claims a 0% stake the
  filer did not disclose — is met. Covered by a test in each direction.
- **One deviation from the plan's spec**, driven by measurement: `_CHAR_NORMALIZATION` covers the
  whole cp1252 C1 block (derived), not the 8 codepoints the plan listed, because the residual was
  `\x80` = the euro sign in a real KDP consideration figure. Deleting it would have changed a
  disclosed currency.
- 13G (passive) filings, legal-boilerplate stripping, cover-page-row stripping and page-furniture
  removal remain deliberately out of scope, per the plan's measured 1.9–2.6% cost/benefit call.

## 6. Next actions

1. **Run the rebuild** (`docs/runbook.md` → "Rebuilding `sec_13d`"). `-F` is mandatory: the DELETE
   does not touch the run manifest, so an incremental run resumes from the last run date and
   fetches nothing. Expect ~1,700 re-fetched filings plus the 461 never-ingested ones.
2. **Re-profile with `--expect-through` and `--update-baseline` afterwards.** That turns D4 into a
   real check and lets D2/D3/D5 compare against the baseline this run just recorded — the rebuild
   is exactly the change those gates exist to police (row count should rise, `rp_seq` should stop
   being all-zero, `date_of_event` / `percent_of_class` / `cusip` should stop being empty).
3. **Confirm `cusip` populates on the rebuilt rows.** If pre-mandate rows still come back `''`,
   decide explicitly whether to source it from the filing header or write NULL — `''` is worse
   than NULL because it defeats every null-rate gate.
4. **Consider auditing the rest of the suite for the same class of staleness.** All three causes
   fixed here were stubs that drifted from a production signature and failed loudly only later;
   nothing guarantees the untested-in-this-pass directories (`fundamentals`, `sharadar`,
   `prices`, `behavioral`) are free of it.
5. Consider whether `item5` embeddings underperform enough to revisit cover-page-row stripping
   (measured to affect 29.6% of item5 bodies at 0.5% false positives).
```json dod-metrics
{
  "baseline_head_sha": "f023398b62fa78cb80446c0d7dee9aadab1d07dc",
  "content_hash": "sha256:86fd29e9a5cec2cf7a75894785ebc30928231dbba89a30a1bc13e03f1e1ac5ce",
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
      "sec_13d": {
        "columns": [
          "ticker",
          "cik",
          "accession_number",
          "form",
          "filing_date",
          "rp_seq",
          "is_amendment",
          "amendment_number",
          "cusip",
          "issuer_name",
          "date_of_event",
          "has_structured_data",
          "reporting_person_name",
          "reporting_person_cik",
          "reporting_person_citizenship",
          "type_of_reporting_person",
          "is_group_member",
          "sole_voting_power",
          "shared_voting_power",
          "sole_dispositive_power",
          "shared_dispositive_power",
          "aggregate_amount",
          "percent_of_class",
          "item3_source_of_funds",
          "item4_purpose_of_transaction",
          "item5_interest_in_securities",
          "item6_contracts_understandings",
          "primary_document",
          "doc_url"
        ],
        "date_col": "filing_date",
        "date_max": "2024-12-16",
        "date_min": "2011-08-22",
        "exists": true,
        "fields": {
          "accession_number": {
            "dtype": "str",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 1666
          },
          "aggregate_amount": {
            "dtype": "object",
            "null_rate": 1.0,
            "nulls": 1666,
            "nunique": 0
          },
          "amendment_number": {
            "dtype": "object",
            "null_rate": 1.0,
            "nulls": 1666,
            "nunique": 0
          },
          "cik": {
            "dtype": "str",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 162
          },
          "cusip": {
            "dtype": "str",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 1
          },
          "date_of_event": {
            "dtype": "object",
            "null_rate": 1.0,
            "nulls": 1666,
            "nunique": 0
          },
          "doc_url": {
            "dtype": "str",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 1666
          },
          "filing_date": {
            "dtype": "object",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 1216
          },
          "form": {
            "dtype": "str",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 2
          },
          "has_structured_data": {
            "dtype": "float64",
            "mad_center": 0.0,
            "mad_outliers": 0,
            "mad_scale": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "min": 0.0,
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 1,
            "p01": 0.0,
            "p25": 0.0,
            "p50": 0.0,
            "p75": 0.0,
            "p99": 0.0,
            "std": 0.0
          },
          "is_amendment": {
            "dtype": "float64",
            "mad_center": 1.0,
            "mad_outliers": 182,
            "mad_scale": 0.1092436974789916,
            "max": 1.0,
            "mean": 0.8907563025210085,
            "min": 0.0,
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 2,
            "p01": 0.0,
            "p25": 1.0,
            "p50": 1.0,
            "p75": 1.0,
            "p99": 1.0,
            "std": 0.31203838897078934
          },
          "is_group_member": {
            "dtype": "object",
            "null_rate": 1.0,
            "nulls": 1666,
            "nunique": 0
          },
          "issuer_name": {
            "dtype": "str",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 178
          },
          "item3_source_of_funds": {
            "dtype": "str",
            "null_rate": 0.5666266506602641,
            "nulls": 944,
            "nunique": 604
          },
          "item4_purpose_of_transaction": {
            "dtype": "str",
            "null_rate": 0.38115246098439376,
            "nulls": 635,
            "nunique": 956
          },
          "item5_interest_in_securities": {
            "dtype": "str",
            "null_rate": 0.187875150060024,
            "nulls": 313,
            "nunique": 1344
          },
          "item6_contracts_understandings": {
            "dtype": "str",
            "null_rate": 0.4225690276110444,
            "nulls": 704,
            "nunique": 843
          },
          "percent_of_class": {
            "dtype": "object",
            "null_rate": 1.0,
            "nulls": 1666,
            "nunique": 0
          },
          "primary_document": {
            "dtype": "str",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 1597
          },
          "reporting_person_cik": {
            "dtype": "str",
            "null_rate": 0.006602641056422569,
            "nulls": 11,
            "nunique": 226
          },
          "reporting_person_citizenship": {
            "dtype": "object",
            "null_rate": 1.0,
            "nulls": 1666,
            "nunique": 0
          },
          "reporting_person_name": {
            "dtype": "str",
            "null_rate": 0.006602641056422569,
            "nulls": 11,
            "nunique": 241
          },
          "rp_seq": {
            "dtype": "int64",
            "mad_center": 0.0,
            "mad_outliers": 0,
            "mad_scale": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "min": 0.0,
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 1,
            "p01": 0.0,
            "p25": 0.0,
            "p50": 0.0,
            "p75": 0.0,
            "p99": 0.0,
            "std": 0.0
          },
          "shared_dispositive_power": {
            "dtype": "object",
            "null_rate": 1.0,
            "nulls": 1666,
            "nunique": 0
          },
          "shared_voting_power": {
            "dtype": "object",
            "null_rate": 1.0,
            "nulls": 1666,
            "nunique": 0
          },
          "sole_dispositive_power": {
            "dtype": "object",
            "null_rate": 1.0,
            "nulls": 1666,
            "nunique": 0
          },
          "sole_voting_power": {
            "dtype": "object",
            "null_rate": 1.0,
            "nulls": 1666,
            "nunique": 0
          },
          "ticker": {
            "dtype": "str",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 155
          },
          "type_of_reporting_person": {
            "dtype": "object",
            "null_rate": 1.0,
            "nulls": 1666,
            "nunique": 0
          }
        },
        "kind": "extract",
        "pk": [
          "ticker",
          "accession_number",
          "rp_seq"
        ],
        "pk_checked_cols": [
          "ticker",
          "accession_number",
          "rp_seq"
        ],
        "pk_checked_rows": 1666,
        "pk_complete": true,
        "pk_duplicate_rows": 0,
        "pk_missing_cols": [],
        "rows": 1666,
        "sample_date_max": "2024-12-16",
        "sample_date_min": "2011-08-22",
        "sampled_rows": 1666,
        "scope": {
          "limit": null,
          "since": null,
          "tickers": null
        },
        "table": "sec_13d"
      },
      "sec_13d_transactions": {
        "columns": [
          "ticker",
          "cik",
          "accession_number",
          "filing_date",
          "trade_seq",
          "reporting_person_name",
          "trade_date",
          "transaction_type",
          "quantity",
          "price_per_share"
        ],
        "date_col": "filing_date",
        "date_max": "2024-10-30",
        "date_min": "2012-01-19",
        "exists": true,
        "fields": {
          "accession_number": {
            "dtype": "str",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 52
          },
          "cik": {
            "dtype": "str",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 16
          },
          "filing_date": {
            "dtype": "object",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 49
          },
          "price_per_share": {
            "dtype": "float64",
            "mad_center": 45.325,
            "mad_outliers": 45,
            "mad_scale": 33.735,
            "max": 893.42,
            "mean": 65.23308069774346,
            "min": 0.0,
            "null_rate": 0.023768115942028985,
            "nulls": 41,
            "nunique": 794,
            "p01": 0.17,
            "p25": 26.5975,
            "p50": 45.325,
            "p75": 100.14750000000001,
            "p99": 497.71,
            "std": 98.00581806641176
          },
          "quantity": {
            "dtype": "float64",
            "mad_center": 77668.5,
            "mad_outliers": 238,
            "mad_scale": 74844.5,
            "max": 18648500.0,
            "mean": 312702.072974359,
            "min": 4.0,
            "null_rate": 0.0052173913043478265,
            "nulls": 9,
            "nunique": 1193,
            "p01": 71.30000000000001,
            "p25": 13082.5,
            "p50": 77668.5,
            "p75": 235641.75,
            "p99": 3217525.0,
            "std": 976470.5973321679
          },
          "reporting_person_name": {
            "dtype": "str",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 37
          },
          "ticker": {
            "dtype": "str",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 16
          },
          "trade_date": {
            "dtype": "object",
            "null_rate": 0.019130434782608695,
            "nulls": 33,
            "nunique": 462
          },
          "trade_seq": {
            "dtype": "int64",
            "mad_center": 37.0,
            "mad_outliers": 283,
            "mad_scale": 32.0,
            "max": 386.0,
            "mean": 83.60115942028986,
            "min": 0.0,
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 387,
            "p01": 0.0,
            "p25": 11.0,
            "p50": 37.0,
            "p75": 129.0,
            "p99": 368.76,
            "std": 100.75444408707408
          },
          "transaction_type": {
            "dtype": "str",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 19
          }
        },
        "kind": "extract",
        "pk": [
          "ticker",
          "accession_number",
          "trade_seq"
        ],
        "pk_checked_cols": [
          "ticker",
          "accession_number",
          "trade_seq"
        ],
        "pk_checked_rows": 1725,
        "pk_complete": true,
        "pk_duplicate_rows": 0,
        "pk_missing_cols": [],
        "rows": 1725,
        "sample_date_max": "2024-10-30",
        "sample_date_min": "2012-01-19",
        "sampled_rows": 1725,
        "scope": {
          "limit": null,
          "since": null,
          "tickers": null
        },
        "table": "sec_13d_transactions"
      }
    }
  },
  "scope": {
    "limit": null,
    "since": null,
    "tables": [
      "sec_13d",
      "sec_13d_transactions"
    ],
    "tickers": [],
    "unknown_tables": []
  },
  "session_id": "9d1a6164-51cc-4126-953b-a46b98db87d9",
  "type": "DATA"
}
```

