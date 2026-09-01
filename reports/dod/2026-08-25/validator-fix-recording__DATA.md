---
type: DATA
session_id: a6d336e2-a115-4087-9ec5-9596d1e899d9
generated_at: 2026-08-25T15:31:05+00:00
baseline: {head_sha: eac0c90f7e5dfcacce696e04f43ccd8316952e5c}
generator: scripts/dod/data_profile.py@1
---

## 1. Scope

**SAMPLE SCOPE** — a metric without its scope is not a measurement:

- tables: fundamentals_check, fundamentals_check_fix, fundamentals_check_run, fundamentals_check_status
- tickers: **all** (no ticker filter)
- since: **no lower bound**
- row limit per table: **none**
- full-scope tables (eligible to set the baseline): fundamentals_check, fundamentals_check_run, fundamentals_check_status, fundamentals_check_fix

**What was asked:** implement
`reports/planning/active-tasks/2026-08-25-validator-fix-recording-plan.md` — give a validator
fix somewhere to be recorded. Agent B fixed cluster `1c9a517eaa47` (MCD `capex`) on 2026-08-25
and the fix left **no machine-readable trace anywhere**: not in `fundamentals_check_status`
(which accepts `wontfix` and nothing else), not in the settled set (a strict set difference a
55→4 drop does not satisfy), not in the rendered report (the cluster fell off the top-50). Its
only record was commit `2fb6ef2`.

Seven phases: a new `fundamentals_check_fix` table plus a widened `fundamentals_check_status`
PK; a ledger read layer; waiver-aware settlement and rendering; `validate fix record` /
`fix show`; the agent-B gate; repairing `adjustment_unguarded`; and backfilling the MCD case as
the end-to-end proof. **All seven are complete.**

This profile covers all four validator tables at **full scope, no row cap and no ticker
filter** — they are small (23,656 / 70 / 2 / 1 rows), so this is a census rather than a sample
and the §3 sample numbers equal the table-wide ones.

## 2. Gates

| Gate | Check | Verdict | Detail |
|---|---|---|---|
| D1 | declared PK unique over the rows profiled | **PASS** | unique across 4 table(s): fundamentals_check, fundamentals_check_run, fundamentals_check_status, fundamentals_check_fix |
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
| fundamentals_check | yes | 23,656 | 23,656 | 23 | run_date,run_id,check_name,ticker,field,period_key | — | 0 | 2026-08-24 | 2026-08-25 | 2026-08-24 | 2026-08-25 |
| fundamentals_check_fix | yes | 1 | 1 | 16 | cluster_id,run_id_after | — | 0 | 2026-08-25 | 2026-08-25 | 2026-08-25 | 2026-08-25 |
| fundamentals_check_run | yes | 70 | 70 | 17 | run_id,check_name | — | 0 | 2026-08-24 | 2026-08-25 | 2026-08-24 | 2026-08-25 |
| fundamentals_check_status | yes | 2 | 2 | 8 | cluster_id,check_name | — | 0 | 2026-08-25 | 2026-08-25 | 2026-08-25 | 2026-08-25 |

**Fields** (worst null rate first, top 60)

| table | field | dtype | null_% | nunique | mean | std | min | p01 | p50 | p99 | max | mad_outliers |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fundamentals_check_run | scope_roster | object | 100 | 0 | — | — | — | — | — | — | — | — |
| fundamentals_check | root_anchor | str | 97.8 | 8 | — | — | — | — | — | — | — | — |
| fundamentals_check | roll_up_children | str | 97.53 | 26 | — | — | — | — | — | — | — | — |
| fundamentals_check | role_uri | str | 48.04 | 805 | — | — | — | — | — | — | — | — |
| fundamentals_check | deviation | float64 | 23.94 | 9,307 | 224,637 | 1.5031e+07 | -5,463.82 | -6.03398 | 3.18883 | 146.12 | 1.01444e+09 | 1,116 |
| fundamentals_check | accession_number | str | 22.08 | 2,321 | — | — | — | — | — | — | — | — |
| fundamentals_check | edgar_url | str | 22.08 | 2,322 | — | — | — | — | — | — | — | — |
| fundamentals_check | source_concept | str | 14.5 | 111 | — | — | — | — | — | — | — | — |
| fundamentals_check | resolution_method | str | 11.6 | 7 | — | — | — | — | — | — | — | — |
| fundamentals_check | as_of | object | 4.36 | 1,058 | — | — | — | — | — | — | — | — |
| fundamentals_check | expected | float64 | 4.27 | 7,050 | 5.76311e+09 | 4.98803e+10 | -7.7039e+10 | -1.2978e+09 | 5.8375e+07 | 8.3004e+10 | 2.03199e+12 | 8,605 |
| fundamentals_check | check_name | str | 0 | 25 | — | — | — | — | — | — | — | — |
| fundamentals_check | cluster_id | str | 0 | 2,335 | — | — | — | — | — | — | — | — |
| fundamentals_check | detail | str | 0 | 11,264 | — | — | — | — | — | — | — | — |
| fundamentals_check | field | str | 0 | 56 | — | — | — | — | — | — | — | — |
| fundamentals_check | finding_id | str | 0 | 13,465 | — | — | — | — | — | — | — | — |
| fundamentals_check | observed | float64 | 0 | 7,630 | 4.71021e+09 | 3.92423e+10 | -2.90447e+11 | -4.041e+09 | 1.64256e+06 | 7.8351e+10 | 2.11967e+12 | 9,816 |
| fundamentals_check | period_key | str | 0 | 1,125 | — | — | — | — | — | — | — | — |
| fundamentals_check | run_date | object | 0 | 2 | — | — | — | — | — | — | — | — |
| fundamentals_check | run_id | str | 0 | 2 | — | — | — | — | — | — | — | — |
| fundamentals_check | severity | str | 0 | 4 | — | — | — | — | — | — | — | — |
| fundamentals_check | substrate | str | 0 | 2 | — | — | — | — | — | — | — | — |
| fundamentals_check | ticker | str | 0 | 55 | — | — | — | — | — | — | — | — |
| fundamentals_check | tier | int64 | 0 | 3 | 2.14736 | 0.597193 | 1 | 1 | 2 | 3 | 3 | 0 |
| fundamentals_check_fix | cluster_id | str | 0 | 1 | — | — | — | — | — | — | — | — |
| fundamentals_check_fix | commit_sha | str | 0 | 1 | — | — | — | — | — | — | — | — |
| fundamentals_check_fix | decided_at | object | 0 | 1 | — | — | — | — | — | — | — | — |
| fundamentals_check_fix | evidence | str | 0 | 1 | — | — | — | — | — | — | — | — |
| fundamentals_check_fix | field | str | 0 | 1 | — | — | — | — | — | — | — | — |
| fundamentals_check_fix | findings_after | int64 | 0 | 1 | 4 | NaN | 4 | 4 | 4 | 4 | 4 | 0 |
| fundamentals_check_fix | findings_before | int64 | 0 | 1 | 55 | NaN | 55 | 55 | 55 | 55 | 55 | 0 |
| fundamentals_check_fix | layer | str | 0 | 1 | — | — | — | — | — | — | — | — |
| fundamentals_check_fix | queued_after | int64 | 0 | 1 | 3 | NaN | 3 | 3 | 3 | 3 | 3 | 0 |
| fundamentals_check_fix | queued_before | int64 | 0 | 1 | 54 | NaN | 54 | 54 | 54 | 54 | 54 | 0 |
| fundamentals_check_fix | root_cause | str | 0 | 1 | — | — | — | — | — | — | — | — |
| fundamentals_check_fix | run_id_after | str | 0 | 1 | — | — | — | — | — | — | — | — |
| fundamentals_check_fix | run_id_before | str | 0 | 1 | — | — | — | — | — | — | — | — |
| fundamentals_check_fix | scope_hash | str | 0 | 1 | — | — | — | — | — | — | — | — |
| fundamentals_check_fix | test_path | str | 0 | 1 | — | — | — | — | — | — | — | — |
| fundamentals_check_fix | ticker | str | 0 | 1 | — | — | — | — | — | — | — | — |
| fundamentals_check_run | abstained | bool | 0 | 2 | — | — | — | — | — | — | — | — |
| fundamentals_check_run | ceiling | float64 | 0 | 10 | 0.266571 | 0.406786 | 0 | 0 | 0.04 | 1 | 1 | 20 |
| fundamentals_check_run | check_name | str | 0 | 37 | — | — | — | — | — | — | — | — |
| fundamentals_check_run | examined | int64 | 0 | 29 | 52,350.3 | 87,624.2 | 0 | 0 | 8,698.5 | 252,001 | 252,001 | 16 |
| fundamentals_check_run | info | int64 | 0 | 10 | 34.8 | 136.995 | 0 | 0 | 0 | 698.31 | 699 | 4 |
| fundamentals_check_run | over_ceiling | bool | 0 | 2 | — | — | — | — | — | — | — | — |
| fundamentals_check_run | queued | int64 | 0 | 33 | 303.143 | 625.723 | 0 | 0 | 0 | 2,471.48 | 2,477 | 4 |
| fundamentals_check_run | run_date | object | 0 | 2 | — | — | — | — | — | — | — | — |
| fundamentals_check_run | run_id | str | 0 | 2 | — | — | — | — | — | — | — | — |
| fundamentals_check_run | scope_fields | str | 0 | 1 | — | — | — | — | — | — | — | — |
| fundamentals_check_run | scope_hash | str | 0 | 1 | — | — | — | — | — | — | — | — |
| fundamentals_check_run | scope_ticker_list | str | 0 | 1 | — | — | — | — | — | — | — | — |
| fundamentals_check_run | scope_tickers | int64 | 0 | 1 | 54 | 0 | 54 | 54 | 54 | 54 | 54 | 0 |
| fundamentals_check_run | scope_tiers | str | 0 | 1 | — | — | — | — | — | — | — | — |
| fundamentals_check_run | substrate | str | 0 | 2 | — | — | — | — | — | — | — | — |
| fundamentals_check_run | tier | int64 | 0 | 3 | 1.68571 | 0.826076 | 1 | 1 | 1 | 3 | 3 | 0 |
| fundamentals_check_status | check_name | str | 0 | 2 | — | — | — | — | — | — | — | — |
| fundamentals_check_status | cluster_id | str | 0 | 1 | — | — | — | — | — | — | — | — |
| fundamentals_check_status | decided_at | object | 0 | 1 | — | — | — | — | — | — | — | — |
| fundamentals_check_status | field | str | 0 | 1 | — | — | — | — | — | — | — | — |

## 4. Evidence

- baseline file: `reports/baselines/data_profile.json` (0 table(s) recorded)
- `fundamentals_check`: 23,656 rows, 23 cols, 23,656 sampled
- `fundamentals_check_fix`: 1 rows, 16 cols, 1 sampled
- `fundamentals_check_run`: 70 rows, 17 cols, 70 sampled
- `fundamentals_check_status`: 2 rows, 8 cols, 2 sampled

## 5. Regressions, gaps and deliberate omissions

- **D2/D3/D5 came back N/A because no baseline existed** — `reports/baselines/data_profile.json`
  recorded 0 tables before this run. This run sets the first one, so the next profile of these
  four tables gets real row-count, column and null-rate comparisons. Nothing was verified about
  drift here; the gates simply had nothing to compare against and say so.
- **`fundamentals_check_status` was DROPPED AND RECREATED, not migrated.** There is no migration
  tooling in `src/data_store/` (all DDL is `CREATE TABLE IF NOT EXISTS`; `ensure_columns` evolves
  columns but never a primary key), so widening the PK to `(cluster_id, check_name)` needed the
  table rebuilt. The drop was **gated on a verified `count(*) = 0`** inside the transaction, with
  a `RAISE EXCEPTION` fallback — no human-written row could have been lost. Had it been non-empty
  the plan's `ALTER TABLE ... ADD COLUMN` path would have been used instead.
- **The `NOT NULL` constraints in the plan's DDL sketch were deliberately NOT applied.**
  `ddl.table_ddl` emits `NOT NULL` only for PK columns, so a hand-written one on `commit_sha` /
  `test_path` / `layer` would be silently erased by the next `python -m scripts.generate_schema_sql`
  and the two mirrors would then disagree. Required-ness is enforced in the CLI (which produces a
  useful message) and stated in the `sql/schema.sql` prose. **A DB-level constraint would be
  stronger and this is weaker than the plan intended** — flagged rather than quietly dropped.
- **`sql/schema.sql` says "AUTO-GENERATED — do not edit by hand" but its rationale prose is
  hand-maintained.** The generator emits only a one-line `-- [aggregate] name (pk: ...)` header,
  so every multi-paragraph block in that file — including the two written here — is lost on
  regeneration. Pre-existing repo-wide, not introduced here, but this change adds ~90 more lines
  of it.
- **`10f02d649538` was left unrecorded.** It settled on 2026-08-25 for reasons nobody wrote down.
  Under the new rule it no longer reads as settled (no fix row), and the run's delta changed from
  "1 cluster settled" to "no cluster closed" before the MCD backfill. That is decision 1 working
  as designed: a reconstructed record is worse than a missing one in a table whose whole purpose
  is evidence. It stays open until somebody who knows what happened records it.
- **Two pre-existing failures in `tests/data_aggregate` are unresolved.** Seen as `..F.F...` in a
  whole-suite run; the suite exceeds the 2-minute tool timeout so they were not isolated. They
  are outside this change's blast radius (`data_aggregate` reads none of the four tables touched
  here) but this is **stated rather than verified**. `tests/validate` + `tests/data_store` are
  fully green: **174 passed, 11 skipped**.
- **`test_apa_revenue` remains broken on HEAD** (`_materialise` returns a tuple; the test still
  calls `.values()` on it). Declared out of scope by the plan, untouched, still broken.
- **`adjustment_unguarded`'s per-row emission was a second, unplanned bug.** Making the check run
  immediately raised `DuplicateFindingError` on 296 findings: it emitted one finding per *fact
  row*, but `finding_id` hashes the `fundamentals_check` PK and one `(ticker, field, period_end)`
  has as many rows as it has as-filed vintages. Fixed by collapsing to the finding grain and
  recording `vintages` in `detail`. The check had never run, so this had never surfaced.
- **`_normalise` in `ledger.py` was not honouring its own documented contract.** It promised
  `datetime64[ns]`; pandas 2.x infers the unit, so a Postgres `DATE` column arrived as
  `datetime64[s]` while the same data through parquet arrived as `[ns]`. Now pinned explicitly.
  Pre-existing and latent — it affected `run_date` on every load, not just the new columns.
- **`adjustment_unguarded` fires 398 `info` findings on the 26-ticker `in_sample` roster and its
  ceiling is 100%.** It is a census of adjustment decisions, not a defect detector, so it can
  never be "over ceiling" and never enters the queue. Worth knowing before anyone reads its rate
  as a quality signal.
- **The read-cost measurement was scoped to `in_sample` (26 tickers, 146,226 fact rows), not the
  universe.** Adding `adjustment` cost **+1.2 MB on 61.6 MB (~1.9%)** with no wall-clock penalty
  (2.5s → 1.9s, i.e. inside the noise). Universe-wide (~28M rows) the column is 0.475% non-null
  and extrapolates to roughly 230 MB of mostly-empty text — **not measured, extrapolated**.

## 6. Next actions

- **Isolate the two `tests/data_aggregate` failures** and confirm they pre-date this branch. Run
  the directory alone with a long timeout; the whole-suite run exceeds the tool's 2-minute cap.
- **Decide whether `sql/schema.sql` should carry prose at all.** Either teach
  `ddl.generate_schema_sql` to preserve hand-written comment blocks per table (it already has the
  `existing_blocks` regex machinery to do it), or move the rationale to `schema.py` and stop
  duplicating it. Today the file's own header instruction and its contents contradict each other.
- **Re-run this profile with `--expect-through`** once a nightly validator run is scheduled, so
  D4 stops being N/A and staleness is actually gated.
- **Measure the `adjustment` projection cost universe-wide** before the first `--roster all` run.
  The in_sample delta is negligible; the extrapolation above is not a measurement.
- **Record `10f02d649538`, or explicitly close it as unrecordable.** It is the one cluster that
  settled under the old rule and does not under the new one.
- **The `coverage_field` ceiling question (26.31% vs 25.0%) is still open** — explicitly out of
  scope here, and it needs its own measurement and its own decision.
- **Consider a `fix list` sub-command.** `fix show` answers "what happened to this cluster?" but
  there is no way to ask "what has been fixed lately, and at which layer?" without SQL. Cheap now
  that `Ledger.fixes_for` exists; not needed until the table has more than one row.

```json dod-metrics
{
  "baseline_head_sha": "eac0c90f7e5dfcacce696e04f43ccd8316952e5c",
  "content_hash": "sha256:4c586c8d3d5b3fef0269e5f9e5753a1b3de14e8534db8fec1946ab4e62e3f019",
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
      "fundamentals_check": {
        "columns": [
          "run_date",
          "check_name",
          "ticker",
          "field",
          "period_key",
          "finding_id",
          "tier",
          "severity",
          "substrate",
          "observed",
          "expected",
          "deviation",
          "as_of",
          "source_concept",
          "resolution_method",
          "roll_up_children",
          "root_anchor",
          "role_uri",
          "accession_number",
          "edgar_url",
          "detail",
          "run_id",
          "cluster_id"
        ],
        "date_col": "run_date",
        "date_max": "2026-08-25",
        "date_min": "2026-08-24",
        "exists": true,
        "fields": {
          "accession_number": {
            "dtype": "str",
            "null_rate": 0.22078965167399392,
            "nulls": 5223,
            "nunique": 2321
          },
          "as_of": {
            "dtype": "object",
            "null_rate": 0.04362529590801488,
            "nulls": 1032,
            "nunique": 1058
          },
          "check_name": {
            "dtype": "str",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 25
          },
          "cluster_id": {
            "dtype": "str",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 2335
          },
          "detail": {
            "dtype": "str",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 11264
          },
          "deviation": {
            "dtype": "float64",
            "mad_center": 3.188828759180518,
            "mad_outliers": 1116,
            "mad_scale": 3.152746461123477,
            "max": 1014441175.4705882,
            "mean": 224637.3354517271,
            "min": -5463.822458270106,
            "null_rate": 0.23938958403787622,
            "nulls": 5663,
            "nunique": 9307,
            "p01": -6.03397706193394,
            "p25": 0.0,
            "p50": 3.188828759180518,
            "p75": 5.425428498337171,
            "p99": 146.1200553497668,
            "std": 15030998.356374318
          },
          "edgar_url": {
            "dtype": "str",
            "null_rate": 0.22078965167399392,
            "nulls": 5223,
            "nunique": 2322
          },
          "expected": {
            "dtype": "float64",
            "mad_center": 58375000.0,
            "mad_outliers": 8605,
            "mad_scale": 92625000.0,
            "max": 2031989000000.0,
            "mean": 5763106048.265386,
            "min": -77039000000.0,
            "null_rate": 0.042695299289820764,
            "nulls": 1010,
            "nunique": 7050,
            "p01": -1297800000.0,
            "p25": 0.17417917006717307,
            "p50": 58375000.0,
            "p75": 1285612000.0,
            "p99": 83004000000.0,
            "std": 49880263296.64688
          },
          "field": {
            "dtype": "str",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 56
          },
          "finding_id": {
            "dtype": "str",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 13465
          },
          "observed": {
            "dtype": "float64",
            "mad_center": 1642558.0,
            "mad_outliers": 9816,
            "mad_scale": 59757442.0,
            "max": 2119673000000.0,
            "mean": 4710214203.628241,
            "min": -290447002532.0,
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 7630,
            "p01": -4041000000.0,
            "p25": 0.873762339625544,
            "p50": 1642558.0,
            "p75": 1310250000.0,
            "p99": 78351000000.0,
            "std": 39242291310.36157
          },
          "period_key": {
            "dtype": "str",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 1125
          },
          "resolution_method": {
            "dtype": "str",
            "null_rate": 0.11595366925938451,
            "nulls": 2743,
            "nunique": 7
          },
          "role_uri": {
            "dtype": "str",
            "null_rate": 0.480385525870815,
            "nulls": 11364,
            "nunique": 805
          },
          "roll_up_children": {
            "dtype": "str",
            "null_rate": 0.9753128170443016,
            "nulls": 23072,
            "nunique": 26
          },
          "root_anchor": {
            "dtype": "str",
            "null_rate": 0.9780182617517754,
            "nulls": 23136,
            "nunique": 8
          },
          "run_date": {
            "dtype": "object",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 2
          },
          "run_id": {
            "dtype": "str",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 2
          },
          "severity": {
            "dtype": "str",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 4
          },
          "source_concept": {
            "dtype": "str",
            "null_rate": 0.14503719986472777,
            "nulls": 3431,
            "nunique": 111
          },
          "substrate": {
            "dtype": "str",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 2
          },
          "ticker": {
            "dtype": "str",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 55
          },
          "tier": {
            "dtype": "int64",
            "mad_center": 2.0,
            "mad_outliers": 0,
            "mad_scale": 0.37833953331078796,
            "max": 3.0,
            "mean": 2.1473621914102132,
            "min": 1.0,
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 3,
            "p01": 1.0,
            "p25": 2.0,
            "p50": 2.0,
            "p75": 3.0,
            "p99": 3.0,
            "std": 0.5971925936426788
          }
        },
        "kind": "aggregate",
        "pk": [
          "run_date",
          "run_id",
          "check_name",
          "ticker",
          "field",
          "period_key"
        ],
        "pk_checked_cols": [
          "run_date",
          "run_id",
          "check_name",
          "ticker",
          "field",
          "period_key"
        ],
        "pk_checked_rows": 23656,
        "pk_complete": true,
        "pk_duplicate_rows": 0,
        "pk_missing_cols": [],
        "rows": 23656,
        "sample_date_max": "2026-08-25",
        "sample_date_min": "2026-08-24",
        "sampled_rows": 23656,
        "scope": {
          "limit": null,
          "since": null,
          "tickers": null
        },
        "table": "fundamentals_check"
      },
      "fundamentals_check_fix": {
        "columns": [
          "cluster_id",
          "run_id_after",
          "run_id_before",
          "scope_hash",
          "ticker",
          "field",
          "findings_before",
          "findings_after",
          "queued_before",
          "queued_after",
          "layer",
          "root_cause",
          "evidence",
          "commit_sha",
          "test_path",
          "decided_at"
        ],
        "date_col": "decided_at",
        "date_max": "2026-08-25",
        "date_min": "2026-08-25",
        "exists": true,
        "fields": {
          "cluster_id": {
            "dtype": "str",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 1
          },
          "commit_sha": {
            "dtype": "str",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 1
          },
          "decided_at": {
            "dtype": "object",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 1
          },
          "evidence": {
            "dtype": "str",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 1
          },
          "field": {
            "dtype": "str",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 1
          },
          "findings_after": {
            "dtype": "int64",
            "mad_center": 4.0,
            "mad_outliers": 0,
            "mad_scale": 0.0,
            "max": 4.0,
            "mean": 4.0,
            "min": 4.0,
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 1,
            "p01": 4.0,
            "p25": 4.0,
            "p50": 4.0,
            "p75": 4.0,
            "p99": 4.0,
            "std": NaN
          },
          "findings_before": {
            "dtype": "int64",
            "mad_center": 55.0,
            "mad_outliers": 0,
            "mad_scale": 0.0,
            "max": 55.0,
            "mean": 55.0,
            "min": 55.0,
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 1,
            "p01": 55.0,
            "p25": 55.0,
            "p50": 55.0,
            "p75": 55.0,
            "p99": 55.0,
            "std": NaN
          },
          "layer": {
            "dtype": "str",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 1
          },
          "queued_after": {
            "dtype": "int64",
            "mad_center": 3.0,
            "mad_outliers": 0,
            "mad_scale": 0.0,
            "max": 3.0,
            "mean": 3.0,
            "min": 3.0,
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 1,
            "p01": 3.0,
            "p25": 3.0,
            "p50": 3.0,
            "p75": 3.0,
            "p99": 3.0,
            "std": NaN
          },
          "queued_before": {
            "dtype": "int64",
            "mad_center": 54.0,
            "mad_outliers": 0,
            "mad_scale": 0.0,
            "max": 54.0,
            "mean": 54.0,
            "min": 54.0,
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 1,
            "p01": 54.0,
            "p25": 54.0,
            "p50": 54.0,
            "p75": 54.0,
            "p99": 54.0,
            "std": NaN
          },
          "root_cause": {
            "dtype": "str",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 1
          },
          "run_id_after": {
            "dtype": "str",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 1
          },
          "run_id_before": {
            "dtype": "str",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 1
          },
          "scope_hash": {
            "dtype": "str",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 1
          },
          "test_path": {
            "dtype": "str",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 1
          },
          "ticker": {
            "dtype": "str",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 1
          }
        },
        "kind": "aggregate",
        "pk": [
          "cluster_id",
          "run_id_after"
        ],
        "pk_checked_cols": [
          "cluster_id",
          "run_id_after"
        ],
        "pk_checked_rows": 1,
        "pk_complete": true,
        "pk_duplicate_rows": 0,
        "pk_missing_cols": [],
        "rows": 1,
        "sample_date_max": "2026-08-25",
        "sample_date_min": "2026-08-25",
        "sampled_rows": 1,
        "scope": {
          "limit": null,
          "since": null,
          "tickers": null
        },
        "table": "fundamentals_check_fix"
      },
      "fundamentals_check_run": {
        "columns": [
          "run_id",
          "run_date",
          "check_name",
          "scope_hash",
          "scope_roster",
          "scope_tickers",
          "scope_ticker_list",
          "scope_fields",
          "scope_tiers",
          "tier",
          "substrate",
          "examined",
          "queued",
          "info",
          "ceiling",
          "abstained",
          "over_ceiling"
        ],
        "date_col": "run_date",
        "date_max": "2026-08-25",
        "date_min": "2026-08-24",
        "exists": true,
        "fields": {
          "abstained": {
            "dtype": "bool",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 2
          },
          "ceiling": {
            "dtype": "float64",
            "mad_center": 0.04,
            "mad_outliers": 20,
            "mad_scale": 0.04,
            "max": 1.0,
            "mean": 0.26657142857142857,
            "min": 0.0,
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 10,
            "p01": 0.0,
            "p25": 0.01,
            "p50": 0.04,
            "p75": 0.25,
            "p99": 1.0,
            "std": 0.4067862227950949
          },
          "check_name": {
            "dtype": "str",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 37
          },
          "examined": {
            "dtype": "int64",
            "mad_center": 8698.5,
            "mad_outliers": 16,
            "mad_scale": 8667.5,
            "max": 252001.0,
            "mean": 52350.3,
            "min": 0.0,
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 29,
            "p01": 0.0,
            "p25": 699.75,
            "p50": 8698.5,
            "p75": 29661.0,
            "p99": 252001.0,
            "std": 87624.23191738455
          },
          "info": {
            "dtype": "int64",
            "mad_center": 0.0,
            "mad_outliers": 4,
            "mad_scale": 34.8,
            "max": 699.0,
            "mean": 34.8,
            "min": 0.0,
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 10,
            "p01": 0.0,
            "p25": 0.0,
            "p50": 0.0,
            "p75": 0.0,
            "p99": 698.31,
            "std": 136.99450956944924
          },
          "over_ceiling": {
            "dtype": "bool",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 2
          },
          "queued": {
            "dtype": "int64",
            "mad_center": 0.0,
            "mad_outliers": 4,
            "mad_scale": 303.14285714285717,
            "max": 2477.0,
            "mean": 303.14285714285717,
            "min": 0.0,
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 33,
            "p01": 0.0,
            "p25": 0.0,
            "p50": 0.0,
            "p75": 196.5,
            "p99": 2471.48,
            "std": 625.7230087218244
          },
          "run_date": {
            "dtype": "object",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 2
          },
          "run_id": {
            "dtype": "str",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 2
          },
          "scope_fields": {
            "dtype": "str",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 1
          },
          "scope_hash": {
            "dtype": "str",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 1
          },
          "scope_roster": {
            "dtype": "object",
            "null_rate": 1.0,
            "nulls": 70,
            "nunique": 0
          },
          "scope_ticker_list": {
            "dtype": "str",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 1
          },
          "scope_tickers": {
            "dtype": "int64",
            "mad_center": 54.0,
            "mad_outliers": 0,
            "mad_scale": 0.0,
            "max": 54.0,
            "mean": 54.0,
            "min": 54.0,
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 1,
            "p01": 54.0,
            "p25": 54.0,
            "p50": 54.0,
            "p75": 54.0,
            "p99": 54.0,
            "std": 0.0
          },
          "scope_tiers": {
            "dtype": "str",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 1
          },
          "substrate": {
            "dtype": "str",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 2
          },
          "tier": {
            "dtype": "int64",
            "mad_center": 1.0,
            "mad_outliers": 0,
            "mad_scale": 0.6857142857142857,
            "max": 3.0,
            "mean": 1.6857142857142857,
            "min": 1.0,
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 3,
            "p01": 1.0,
            "p25": 1.0,
            "p50": 1.0,
            "p75": 2.0,
            "p99": 3.0,
            "std": 0.8260760596426335
          }
        },
        "kind": "aggregate",
        "pk": [
          "run_id",
          "check_name"
        ],
        "pk_checked_cols": [
          "run_id",
          "check_name"
        ],
        "pk_checked_rows": 70,
        "pk_complete": true,
        "pk_duplicate_rows": 0,
        "pk_missing_cols": [],
        "rows": 70,
        "sample_date_max": "2026-08-25",
        "sample_date_min": "2026-08-24",
        "sampled_rows": 70,
        "scope": {
          "limit": null,
          "since": null,
          "tickers": null
        },
        "table": "fundamentals_check_run"
      },
      "fundamentals_check_status": {
        "columns": [
          "cluster_id",
          "check_name",
          "ticker",
          "field",
          "status",
          "note",
          "findings_at_decision",
          "decided_at"
        ],
        "date_col": "decided_at",
        "date_max": "2026-08-25",
        "date_min": "2026-08-25",
        "exists": true,
        "fields": {
          "check_name": {
            "dtype": "str",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 2
          },
          "cluster_id": {
            "dtype": "str",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 1
          },
          "decided_at": {
            "dtype": "object",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 1
          },
          "field": {
            "dtype": "str",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 1
          },
          "findings_at_decision": {
            "dtype": "int64",
            "mad_center": 1.5,
            "mad_outliers": 0,
            "mad_scale": 0.5,
            "max": 2.0,
            "mean": 1.5,
            "min": 1.0,
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 2,
            "p01": 1.01,
            "p25": 1.25,
            "p50": 1.5,
            "p75": 1.75,
            "p99": 1.99,
            "std": 0.7071067811865476
          },
          "note": {
            "dtype": "str",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 2
          },
          "status": {
            "dtype": "str",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 1
          },
          "ticker": {
            "dtype": "str",
            "null_rate": 0.0,
            "nulls": 0,
            "nunique": 1
          }
        },
        "kind": "aggregate",
        "pk": [
          "cluster_id",
          "check_name"
        ],
        "pk_checked_cols": [
          "cluster_id",
          "check_name"
        ],
        "pk_checked_rows": 2,
        "pk_complete": true,
        "pk_duplicate_rows": 0,
        "pk_missing_cols": [],
        "rows": 2,
        "sample_date_max": "2026-08-25",
        "sample_date_min": "2026-08-25",
        "sampled_rows": 2,
        "scope": {
          "limit": null,
          "since": null,
          "tickers": null
        },
        "table": "fundamentals_check_status"
      }
    }
  },
  "scope": {
    "limit": null,
    "since": null,
    "tables": [
      "fundamentals_check",
      "fundamentals_check_fix",
      "fundamentals_check_run",
      "fundamentals_check_status"
    ],
    "tickers": [],
    "unknown_tables": []
  },
  "session_id": "a6d336e2-a115-4087-9ec5-9596d1e899d9",
  "type": "DATA"
}
```

