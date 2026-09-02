# Phase 6 — 23-ticker comparison, gate, cutover, report 🔄

**Goal**: prove the new implementation beats the baseline on the 23 tickers over 2000-2026, get your
sign-off, then **you** truncate and rerun. Finish with the DATA definition-of-done report.

**This is the only phase that touches live data.**

---

## Step 1 — Run the new pipeline into parquet (no DB writes)

- [ ] For each of the 23 tickers, over the full `years_history` window, run the new path off Phase 0's
      **on-disk filing cache**: `html_to_text` + `def14a_tables` → `prepare_def14a_sections` →
      `LLMExtractor` → `_flatten` + the four row builders → parquet under
      `.../2026-09-01-def14a-extraction-fix/new/`.
- [x] Same for the ECD path (`def14a_ecd`) → `sec_def14a` parquet. This one needs live
      `filing.xbrl()` calls; scope it to the 2023+ filings of the 23 tickers (~90 filings).
- [ ] Same for `sec_8k_votes`, reading `item_text` **from the DB** (read-only, projected, `where`
      on ticker + item) and writing parquet.
- [ ] **Estimated LLM spend**: 23 tickers × ~26 proxies ≈ **600 proxy calls**, plus ~23 × 14 ≈ **320
      vote calls**. Under $10 at `gpt-5-mini` payload sizes. Log the actual token counts — the full
      rerun is ~15× this and you will want the number before starting it.
- [ ] Record the per-filing payload chars so the payload metric is real, not asserted.

## Step 2 — The comparison

- [x] `"$PY" scripts/compare_def14a_baseline.py` → `COMPARISON.md` + stdout.
- [ ] Every check from Phase 0's table, reported PASS/FAIL, most important first. The gate:

| # | Gate | Baseline | Required |
|---|---|---|---|
| G1 | exec-comp rows with any component > $1e9 | > 0 | **0** |
| G2 | ownership rows where shares/percent diverge ~10× | > 0 | **0** |
| G3 | `peo_total_comp == 0.0` (SBUX FY23-25) | 3 | **0** |
| G4 | distinct PEOs recorded for BA/NKE co-PEO years | 1 | **2** (`n_peos`, `peo_names_all`) |
| G5 | accessions with `n_neos == 1`, 2012+ | ~21.7% | **< 8%** |
| G6 | `auditor_name` fill | 2.05% | **> 80%** |
| G7 | director-comp rows / post-2008 proxies | 0 | **> 90%** |
| G8 | say-on-pay values < 0.50 present | 0 (floored) | **≥ 3** (JPM/INTC/SPG) |
| G9 | pre-2001 rows fully NULL | 401/422 | **< 10%** |
| G10 | mean carve payload chars | 51,198 (measured) | **≤ 45,000** (re-set — see note) |
| G11 | `sec_8k_votes` rows / 5.07 filings with a comma-number | 0 | **> 90%** |
| G12 | `pct_female_directors` fill | (record it) | **must not fall** |
| G13 | `gender_basis` populated wherever `gender` is set | n/a (field is new) | **100%** |
| G14 | filings where `n_women_directors_vs_inferred == 0` | (record it) | **materially higher than baseline** |

- [ ] **G10 was re-set from ≤ 40,000 to ≤ 45,000** (approved 2026-09-02). The 40,000 derived from a
      36,544-char estimate that predates this plan's own +3,000 widenings of PAY RATIO / SAY ON PAY
      and its new AUDITOR NAME slice, so it was unreachable as specified; measured is 42,225 (−18%
      from 51,198). It is *not* 50,000, which would sit at the baseline and certify only "not worse
      than the code being replaced". Full reasoning and the regression it must catch are recorded in
      `PHASE-2-table-anchored-carve.md`. **Report the mean / median / max, not just the pass flag** —
      the binding term is the 20,000-char director-bios window the plan deliberately keeps.
- [ ] **G12-G14 are the gender-upgrade gates.** `gender` is kept because it carries alpha, so the
      bar is *accuracy up at no cost in coverage*, not merely "still present". Report alongside them:
      the `gender_basis` distribution (how much is `stated`/`honorific` versus `name`), and the count
      of people the consensus pass **overturned**. An overturn count of 0 on real data means the
      name key never matched anyone across filings — a silent failure of the mechanism, not a clean
      bill of health.
- [ ] **Hand-check 20 rows** across the 23 tickers against the actual filings — 5 exec-comp, 5
      director-comp, 5 ownership, 5 vote rows, chosen to include A / AMAT / PG / CAT / PFE. Fill-rate
      deltas prove recall; only opening the filing proves correctness, and every conclusion in the
      research came from doing exactly this.
- [ ] Write the outcome into `COMPARISON.md` with the filing URLs, so the evidence is re-checkable.

## Step 3 — Gate

- [ ] Present `COMPARISON.md`. **Do not proceed without your explicit go**, including the G12-G14 call on the gender upgrade.
- [ ] If a gate fails, fix in the owning phase and re-run Steps 1-2. Do not proceed on a partial pass.

## Step 4 — Cutover (yours to run)

Ordered, because `existing_filings` dedups on accession — an incremental run after a parser change
skips everything, which is why this is a truncate-and-rebuild rather than a migration.

- [ ] **Stop the running extraction first.** Both the DEF 14A and 8-K backfills are live. Find the
      PID and kill **by PID only** — never by image name; a blanket `python.exe` kill has already
      destroyed a multi-hour SEC download in this repo once.
- [ ] Re-run the Phase 0 snapshot to capture the final pre-cutover state (the backfill has advanced
      since Phase 0). Keep both snapshots — Phase 0's is the comparison baseline, this one is the
      rollback.
- [ ] Drop the four retired tables:

```sql
DROP TABLE IF EXISTS sec_def14a_executive_comp;
DROP TABLE IF EXISTS sec_def14a_director_comp;
DROP TABLE IF EXISTS sec_def14a_ownership;
DROP TABLE IF EXISTS sec_def14a_votes;
```

- [ ] Truncate the two rebuilt tables (`sec_def14a` changes column set; `def14a_llm` must be
      re-extracted because the accession dedup would otherwise skip every filing):

```sql
TRUNCATE TABLE sec_def14a;
TRUNCATE TABLE def14a_llm;
```

- [ ] **Drop the three retired `def14a_llm` columns.** Phase 3 removed them from `sql/schema.sql`,
      but `CREATE TABLE IF NOT EXISTS` cannot retire a column on a database that already has one,
      so the live table keeps them until this runs. A truncate does *not* remove them — it would
      leave three permanently-NULL columns that read downstream as "this company discloses no
      technology directors" rather than "we stopped extracting an opinion".

```sql
ALTER TABLE def14a_llm
    DROP COLUMN IF EXISTS n_technology_directors,
    DROP COLUMN IF EXISTS pct_technology_directors,
    DROP COLUMN IF EXISTS technology_committee;
```

- [ ] `sec_def14a`'s column set changed, so let `store.ensure_table` recreate it, or `DROP` it too.
      **Caveat**: `ensure_table` is a check-then-create with no lock, and threaded writers on a
      **cold** table can silently lose rows. Run the first fetch **single-ticker** (`-t AAPL`) to
      create the table warm, then run the universe.
- [ ] Clear the run manifest entries for `def14a_llm` / `def14a_edgar` so the next run does a full
      rescan rather than a manifest-narrowed window.
- [ ] Full rerun:

```bash
rtk "$PY" -m src data_extract def14a       -c ./configs      # LLM path + 4 child tables
rtk "$PY" -m src data_extract def14a-edgar -c ./configs      # ECD XBRL only
rtk "$PY" -m src data_extract sec-8k-votes -c ./configs      # new
```

- [ ] Expected spend: ~8,700 proxies + ~6,700 vote filings. Use the Step-1 token measurement to size
      it before starting.
- [ ] `def14a_llm` upserts **per ticker**, so an interrupted run loses nothing already extracted and
      resumes on the accession dedup.

## Step 5 — Post-cutover verification against Postgres

Parquet-cached harnesses hide the `DATE → datetime.date` round-trip bug class entirely, so the
defect assertions must be re-run **against the real DB**, not only against parquet.

- [ ] Re-run the G1-G14 checks with SQL over the whole rebuilt tables, not just the 23 tickers.
- [ ] Coverage: `def14a_llm` should reach ~500 tickers; `sec_def14a` should hold roughly
      **93% of 2023 and 99% of 2025** S&P 500 proxies, and the 2023 gaps should decompose as **27
      non-December fiscal year ends** plus a handful of filers who published a PVP table with no XBRL
      (APP, BSX, GDDY were the measured three). If the shortfall does not decompose that way, the
      dimension filter is dropping rows.
- [ ] Confirm the ECD columns are **0% filled for 1995-2022 and 81-100% for 2023-2026**. That is
      correct behaviour, not a gap — the ECD block **did not exist in any DEF 14A before 2023**
      (Rel. 34-95607, FY ending ≥ 2022-12-16). Do not "fix" it.
- [ ] Cube smoke test: `StepBuildCube`'s governance panel builds, and `governance_features` produces
      the same feature names as before minus the dropped ones. Aggregate fingerprint will move —
      expected, and the baseline needs updating with a note saying why.
- [ ] Spot-check three `def14a_json` blobs for NUL characters and non-ASCII mangling. The old edgar
      path had **7.06% of exec-comp rows carrying U+0097** (a cp1252 em-dash mis-decode); the LLM
      path should have none.

## Step 6 — Docs and report

- [ ] `docs/data_schema.md` — 4 tables removed, 5 added (`def14a_executive_comp`,
      `def14a_director_comp`, `def14a_ownership`, `def14a_directors`, `sec_8k_votes`), `sec_def14a` and `def14a_llm`
      column lists updated.
- [ ] `docs/database.md` — real measured coverage after the rerun. Replace the "23 of 500 tickers /
      ~95% NaN" warning with what is now true.
- [ ] `docs/data_sources.md` — DEF 14A section: the ECD 2023 threshold, why the HTML block was
      deleted, the table-anchored carve, and the Item 5.07 source with its two hard rules.
- [ ] `docs/config.md` — only if a knob was added (the plan adds none).
- [ ] `AGENTS.md` — no change expected (cap 70 lines).
- [ ] **DATA definition-of-done report** via the `dod-data-report` skill:
      `reports/<YYYY-MM-DD>/def14a-extraction-fix__DATA.md`. It must carry the measured before/after
      table, the hand-checked rows with filing URLs, the accepted losses, and the real LLM spend.
- [ ] Move this plan directory from `active-tasks/` to wherever completed plans live in this repo.

---

## RESULTS — tooling complete and gated; the DEF 14A re-extraction is YOURS

**You retained the re-extraction** ("do not rerun the whole def14, I will do it. Finish the plan
without doing def14 extraction"). So Steps 2, 6 and every piece of tooling are done and verified;
Steps 1 (LLM legs), 3, 4 and 5 are prepared, dry-run where possible, and handed over. **Nothing
in the database was changed by this session.**

### What ran

| Step | State | Evidence |
|---|---|---|
| 1 — ECD leg | ✅ **complete** | 85 rows, 25 columns, all 23 tickers, 2023+. Deterministic, no LLM. `new/sec_def14a.parquet` |
| 1 — proxy / vote legs | ⬜ **handed over** | runner written, smoke-tested, throughput measured; ~656 + ~320 calls outstanding |
| 2 — comparison | ✅ **runs** | `COMPARISON.md`: **2 PASS / 0 FAIL / 12 PENDING** |
| 3 — gate | ⬜ | 10 of 14 gates have no measurement yet |
| 4 — cutover | ⬜ **handed over** | `scripts/def14a_cutover.py` dry-run verified: 7 statements |
| 5 — post-cutover verify | ⬜ **handed over** | `scripts/def14a_postcutover_verify.py` written; runs read-only and correctly reports the pre-cutover state as failing |
| 6 — docs + DoD | ✅ **complete** | `docs/data_schema.md`, `docs/database.md`, `docs/data_sources.md`, `reports/2026-09-02/def14a-extraction-fix__DATA.md` |

### The gates that ARE measured

Both come off the deterministic ECD run, so they cost nothing and are reproducible:

- **G3 PASS** — 0 rows with `peo_total_comp == 0.0`, against a baseline of 3. SBUX FY23-25 now
  reads 95,801,676 and 30,992,773 where the retired path had a zero matrix.
- **G4 PASS** — BA 2025 `n_peos = 2` with `peo_names_all` = "David Calhoun,Gregory Smith,Robert
  K. Ortberg"; NKE 2025 `n_peos = 2` with "Elliott Hill,John Donahoe II". The old
  `.iloc[0]` accessor kept one PEO per year and document order decided which.
- Sign preservation: **3 negative `peo_actually_paid_comp`** rows survive, including NKE 2025's
  −10,924,243. There is no `abs()` on this path.
- Live-DB spot checks (read-only, whole table not just the 23 tickers): `peo_actually_paid_comp`
  fill **87.8% / 96.6% / 94.0% / 92.4%** for 2023/24/25/26, and `def14a_json` carries **0 NULs
  and 0 cp1252 mojibake** in 200 sampled blobs — the retired path had U+0097 on 7.06% of rows.

**The other ten gates report `-`/PENDING, not FAIL.** They have no measurement. That distinction
is the whole point of the row: a PENDING gate is an unanswered question, and reporting it as a
pass would be the one thing this phase exists to prevent.

### Step 4 was not executable as the plan wrote it — measured, then fixed

The plan says "run the CLI commands". Serially that is **not finishable**: one modern proxy is a
~130k-char payload on a reasoning model and takes **~94 seconds** (measured over A's filings), so
8,700 proxies is **~9.5 days**.

The per-filing LLM calls in `fetch_def14a_llm` and `fetch_8k_votes_llm` now run **12-wide**
(`_LLM_WORKERS`), measured at **9.1-10.6 s/filing with zero 429s** — a ~10x cut. Every write
stays exactly where it was: one `_save_ticker_rows` per ticker, on the main thread. That
deliberately preserves two properties the plan relies on — "an interrupted run loses no paid
tokens", and immunity from `store.ensure_table`'s check-then-create race, since no worker ever
touches the store.

**Budget the real thing at ~34 h and ~$150**: ~23 h for 8,700 proxies, ~11 h for 6,657 vote
filings. `LLMExtractor` now records token usage per call (`last_usage` / `totals`, lock-guarded)
because the plan forbids starting the backfill on an estimate and nothing in the repo had ever
captured the numbers.

### Defects found by running the tooling on live data

1. **The pre-2001 folder-index trap (my bug, in the Phase-6 runner).** 88 of 656 cached filings
   carry `primaryDocument == ""`, so `_doc_url` builds a bare *directory* URL and Phase 0's
   `cache_htm` holds a **10,217-char EDGAR directory listing** instead of the proxy. The real
   document is the `.txt` sibling — 159,203 chars on A's 2000 filing. Reading `cache_htm` fed the
   model a folder index, and those filings came back with **0-2 non-null fields**, which reads as
   "pre-2001 does not extract" (i.e. G9 fails) when the cause is opening the wrong file. Verified
   after the fix on A 2000-01-13: **2 non-null fields → 27**, `company_name` "Agilent
   Technologies", `board_size` 6, CEO "Edward W. Barnholt", auditor "PricewaterhouseCoopers LLP",
   5 NEOs, plus 5 exec-comp / 11 ownership / 6 director rows. **Production was never affected** —
   it uses Phase 1's corrected `_doc_url`.
2. **A real over-rejection in the Phase-5 fabrication guard.** PTC's 2010 filing wraps a narrow
   name column, so the vote numbers land BETWEEN the halves of the name:

   ```
   Paul                       100,753,338     1,735,851     7,486,441
   A. Lacy
   ```

   `"Paul A. Lacy"` is a substring at no whitespace normalisation, and three CORRECTLY read
   nominees were discarded. Fixed with a token fallback (contiguous first, else every substantial
   token must appear); the measured fabrications still fail it. Modern filings show **0
   rejections** — AAPL 2026 and JPM 2026 each returned 5 and 7 clean proposals.
3. **The vote role map is unmeasurable from the DB pre-cutover.** `_role_source` joins to
   `def14a_executive_comp` / `def14a_director_comp`, which the cutover CREATES, so a DB-sourced
   map returned **96.8% `unmatched`** — a fact about the missing tables, not about the join.
   Step 1's `--votes` leg reads its role frames from `new/` instead.

### Mistakes worth recording

- **`TaskStop` kills the shell wrapper, not the Python children.** Two abandoned proxy runs kept
  extracting — one for ~34 minutes — after I believed they were stopped, both against the buggy
  pre-2001 path. **~$5 of wasted LLM spend.** Found by enumerating command lines and killed by
  PID (never by image name — a blanket `python.exe` kill has already destroyed a multi-hour SEC
  download in this repo once). Always verify the process is gone after stopping a task.
- **The smoke-set parquets in `new/` were deleted, not left behind.** They covered only each
  ticker's oldest filing and were produced through the folder-index bug, so every fill rate off
  them was wrong. `COMPARISON.md` now honestly says PENDING. `new/README.md` records why.

### Fixture substitution (Step 2's hand-check)

The plan's 20-row hand-check across A / AMAT / PG / CAT / PFE needs the proxy leg's output, so it
is outstanding with it. What was hand-checked instead, on real filings: BA's and NKE's co-PEO
years and SBUX's non-zero PVP (above), AAPL 2025 / JPM 2025 / TDG 2020 / AEE 2026 / GE 2019+2024 /
JPM 2017 / TDG 2014 vote tables read line-by-line into `tests/.../fixtures/item507_texts.json`,
and Agilent's 2000 proxy end-to-end.

## Rollback

| If | Then |
|---|---|
| A gate fails before Step 4 | Nothing has changed in the DB. Fix in the owning phase. |
| The rerun produces worse data than the snapshot | The pre-cutover snapshot (Step 4) restores the 23 tickers for comparison; the full old tables are gone, which is the cost you accepted in choosing truncate-and-rebuild over migration. |
| G12-G14 fail (the gender upgrade made things worse) | The consensus pass is a separate, idempotent finalisation step — skip it and the extraction still stands, leaving `gender` at its current first-name-prior quality plus the new `gender_basis` provenance. Decide at Step 3, before truncation. |
| Code needs reverting | Phases 1-5 are separate commits; each reverts independently. |

## Notes

- Kill the running extraction **by PID**. Never by image name.
- Do not start the full rerun until Step 1's token measurement exists. ~15× the validation run is a
  real bill and a real wall-clock cost.
- The aggregate fingerprint baseline will move. That is an expected consequence of dropping features,
  not a regression — update it with the reason recorded.
