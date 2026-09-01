# Phase 6 — 23-ticker comparison, gate, cutover, report ⬜

**Goal**: prove the new implementation beats the baseline on the 23 tickers over 2000-2026, get your
sign-off, then **you** truncate and rerun. Finish with the DATA definition-of-done report.

**This is the only phase that touches live data.**

---

## Step 1 — Run the new pipeline into parquet (no DB writes)

- [ ] For each of the 23 tickers, over the full `years_history` window, run the new path off Phase 0's
      **on-disk filing cache**: `html_to_text` + `def14a_tables` → `prepare_def14a_sections` →
      `LLMExtractor` → `_flatten` + the four row builders → parquet under
      `.../2026-09-01-def14a-extraction-fix/new/`.
- [ ] Same for the ECD path (`def14a_ecd`) → `sec_def14a` parquet. This one needs live
      `filing.xbrl()` calls; scope it to the 2023+ filings of the 23 tickers (~90 filings).
- [ ] Same for `sec_8k_votes`, reading `item_text` **from the DB** (read-only, projected, `where`
      on ticker + item) and writing parquet.
- [ ] **Estimated LLM spend**: 23 tickers × ~26 proxies ≈ **600 proxy calls**, plus ~23 × 14 ≈ **320
      vote calls**. Under $10 at `gpt-5-mini` payload sizes. Log the actual token counts — the full
      rerun is ~15× this and you will want the number before starting it.
- [ ] Record the per-filing payload chars so the payload metric is real, not asserted.

## Step 2 — The comparison

- [ ] `"$PY" scripts/compare_def14a_baseline.py` → `COMPARISON.md` + stdout.
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
| G10 | mean carve payload chars | ~50,300 | **≤ 40,000** |
| G11 | `sec_8k_votes` rows / 5.07 filings with a comma-number | 0 | **> 90%** |
| G12 | `pct_female_directors` fill | (record it) | **must not fall** |
| G13 | `gender_basis` populated wherever `gender` is set | n/a (field is new) | **100%** |
| G14 | filings where `n_women_directors_vs_inferred == 0` | (record it) | **materially higher than baseline** |

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
