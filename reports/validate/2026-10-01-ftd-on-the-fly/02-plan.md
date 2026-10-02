# FTD on-the-fly publication — implementation plan

Status: **proposed for approval** on 2026-10-01. Governing inputs: [approved specification](./01-spec.md) and [research](./01-research.md). This plan authorizes no product edit until the user approves it.

## Objective and working state

Remove every runtime and declared `sec_ftd_vintages` dependency. Compute a FTD ZIP's feature publication date from its persisted `period` at build time, with the approved short-lived cached-ZIP timestamp override. Preserve extraction resume, cumulative FTD balance semantics, and all unrelated feature fields.

- Base: `dev` at `7ce4ad7a857f973c84201a3599ff5a317f5e90f3`.
- Isolated implementation: branch `harness/ftd-on-the-fly`, worktree `reports/worktrees/ftd-on-the-fly` under the repository root. All relative commands below run there unless stated otherwise.
- Evidence root: `reports/validate/2026-10-01-ftd-on-the-fly/` in that worktree. The worktree and report are ignored by the base repository. Preserve unrelated edits in the base worktree.
- Python: `C:/Users/de larrard alexandre/AppData/Local/pypoetry/Cache/virtualenvs/stock-pick-strat-lkf53h9P-py3.13/Scripts/python.exe`. Every shell command begins with `rtk`.
- Baseline: 31 focused extraction/short-flow tests passed in 10.92 s. The fixed short pytest targets `../../pt-ftd-red-20261001`, `../../pt-ftd-green-20261001`, and `../../pt-ftd-reg-20261001` resolve under the base repository's ignored `reports/` directory; all three were verified absent on 2026-10-01. Recheck the exact target before running because pytest may clear an existing `--basetemp` directory.

## Current behavior and desired compatibility

The extractor currently builds and backfills a second vintage table, the cube step loads it as a prerequisite, and the feature function otherwise guesses the ZIP half from settlement day. That guess is false for day-15 rows actually supplied by a `b` ZIP. The desired build reads the already stored `period`, performs a small per-period date calculation, and checks at most one cached ZIP's filesystem timestamp. It adds no large read, persistent table, network fetch, or new config. FTD feature publication dates may change; feature names, grain, payload values, non-FTD fields, extraction source identity, and incremental resume remain compatible. Existing saved cube rows are not rewritten by this run.

Migration: removing the registry entry and generated CREATE TABLE block changes future bootstrap schema only. A legacy physical `sec_ftd_vintages` table may remain in an existing PostgreSQL database but no product path may use it. A later physical cleanup requires separate explicit authorization and a DB migration plan; it is outside this run.

## Exact product write scope

| Area | Files | Intended change |
| --- | --- | --- |
| Source ingestion | `src/data_extract/utils/institutionals/fetch_fails_to_deliver.py` | Remove vintage backfill/write and HTTP-observation bookkeeping; keep ZIP parsing, source `period`, resume, and full replacement. |
| Feature build | `src/data_aggregate/utils/institutionals/short_flow_features.py`, `src/data_aggregate/transformers/step_cube_institutionals.py` | Require source `period`, derive ZIP dates once per build, read only the relevant cache file's mtime for the newest eligible stored period, and stop loading vintage metadata. |
| Shared fixed policy | `src/constants/constants.py` | Name the fixed 15-day lag, strict 2-day cache window, and existing 60-day recent-period bound; do not add a new YAML knob. |
| Registry/DDL | `src/data_store/schema.py`, generated `sql/schema.sql` | Delete only the `sec_ftd_vintages` declaration and its generated DDL block. The physical PostgreSQL table is not dropped. |
| Read-only quality script | `scripts/institutionals_feature_quality.py` | Remove `sec_ftd_vintages` from source requirements; retain source identity checks for the remaining tables. |
| Focused tests | `tests/data_extract/institutionals/test_fails_to_deliver.py`, `tests/data_aggregate/test_short_flow_features.py`, `tests/data_aggregate/test_institutional_step_contract.py`, `tests/data_aggregate/aggregate_fingerprint.py` | Replace table assumptions with known-truth clock/cache cases, absent-table integration, and unchanged extraction/feature contracts. Add the required persisted `period` to the synthetic aggregate fixture, without changing the protected fingerprint baseline. Tests print a sanity conclusion. |
| Targeted wiki sync | `wiki/reference/table-catalog.md`, `wiki/reference/data-sources.md`, `wiki/modules/data-extract.md`, `wiki/modules/data-aggregate.md`, `wiki/concepts/source-availability.md`, `wiki/log.md`, `wiki/TODO.md` | Correct only FTD availability/table descriptions and record source-grounded change history. Use OpenKnowledge MCP for every read/write. |

Audit artifacts are confined to the evidence root; read-only baseline and candidate scripts may live in its `_scripts/` directory. No edits to `context.py`, `utils/step.py`, `data/`, configs, the aggregate fingerprint baseline, or unrelated dirty files. No physical table drop, persisted cube build, backfill, push, PR, or deployment. Encountering a required protected file, new source contract, or need for DB mutation triggers a revised plan and approval before that action.

## Decision ledger

1. **Date names.** `period` is the persisted ZIP tag (`YYYYMMa/b`). `available_date` is the estimated calendar date or fresh cache mtime's New York date. `as_of` is the first trading session on or after `available_date` in the cube's session index. Do not derive the ZIP from settlement day.
2. **Historical clock.** `a` ends on day 15; `b` ends at calendar month end. Add 15 calendar days, then move a weekend to Monday. Existing session-index search handles market holidays. A ZIP's latest settlement must not exceed its publication date.
3. **Freshest ZIP.** Select the highest stored `period` whose period end is not in the future and is at most 60 calendar days old. Only its existing `cnsfails{period}.zip` cache file may override the estimate when `0 <= (New York today - New York cache-mtime date).days < 2`. Missing/future/older mtime falls back to the historical clock. Read mtime, never touch the cache. The date is intentionally time-dependent after day 1.
4. **Fail closed.** Malformed/missing `period` and one settlement day assigned to multiple ZIPs raise clear errors. A missing cache is a documented fallback, not a missing-data failure. One cumulative latest balance per ZIP is published, preserving unavailable `NaN` and observed zero.
5. **Schema boundary.** Remove the code declaration and generated CREATE TABLE SQL; keep the existing live physical table untouched because the approved spec excludes a destructive drop.
6. **Review discipline.** A narrow existing-code change is preferred to a new availability service/table. Any required file outside the table above or changed policy is a plan revision before edit.
7. **2026-10-01 test-fixture scope amendment.** The approved aggregate regression check could not run because its synthetic FTD frame omitted the now-required `period`; 30 other broad tests passed. `tests/data_aggregate/aggregate_fingerprint.py` is added only to tag its synthetic rows by the half-month already used by the old fallback. The fingerprint baseline file remains protected and unchanged. This is a test-input repair under the approved FTD regression objective, not a new product behavior.
8. **2026-10-01 correctness-review fix.** A partial-universe build can omit rows from the globally latest ZIP. Read global distinct FTD periods through `context.store` and pass that small list to the date derivation; only the globally latest eligible period may use cache mtime. Add a known-truth regression for an older ZIP with fresh mtime in a ticker-scoped build. The independent re-review found no remaining issue.
9. **2026-10-01 validation-scope amendment.** The full persisted-cube pull took 754 seconds; grain, coverage, profile, bounds, catalogue, and leakage were run. Coverage found stale rows, catalogue found legacy undocumented columns, and leakage found a critical unrelated beneficial-ownership issue. Exact redundancy scanned 600,000 of 3,265,378 rows over several minutes with no candidate FTD coverage, so it was stopped before a result; clip and timeseries were not started. These checks inspect unchanged persisted rows and cannot prove this candidate. The read-only FTD candidate check, focused tests, aggregate regression, and completed generic checks are the decision evidence. Record all missing/failed statuses, without treating the skipped checks as passes.

## Acceptance trace

| Criterion | RED or inspection evidence | GREEN / validation evidence |
| --- | --- | --- |
| AC-001 | Existing vintage-requiring tests and source reference audit | Extractor/step tests with no vintage table; registry/DDL and `rg` audit |
| AC-002 | New known-truth cases fail on the old API | `a`/`b`, weekend, holiday, and 0/1/2-day cases pass with sanity prints |
| AC-003 | Read-only projected AAPL 2024-04-15 `202404b` sample | In-memory feature run proves no state before `202404b` publication; domain validator evidence |
| AC-004 | Missing/invalid period and conflicting ZIP assignment cases fail on current code | Explicit error/fallback cases plus full/incremental extraction tests pass |
| AC-005 | 31-test baseline and captured candidate column set | Aggregate regression/institutional contract checks; explain only FTD clock changes, no baseline update |
| AC-006 | Current targeted wiki statements and report inventory | OpenKnowledge changes and self-contained HTML report with exact results/limits |

## Ordered phases

### Phase 1 — Baseline and RED contract tests · ✅

AC-001/002/004/005. This phase precedes source edits so old behavior is available for comparison. Bounded files: three scoped test files and `reports/validate/2026-10-01-ftd-on-the-fly/_scripts/capture_ftd_baseline.py`. First capture the old FTD publication dates and feature-column set on a projected AAPL 2024 window using `context.store` and the old in-memory feature path; write `baseline-ftd.json` under the run directory. Add the smallest known-truth tests for period-based publication, latest-cache 0/1/2-day threshold, missing/future/old cache, duplicate mapping, and missing/malformed periods. Add a step test that denies the vintage table and requires FTD output. Capture failing output in `03-implementation.md`, including which failure demonstrates each new rule. Test workdir is the isolated worktree. Before pytest, resolve and confirm `../../pt-ftd-red-20261001` is still absent.

Commands: `rtk "C:/Users/de larrard alexandre/AppData/Local/pypoetry/Cache/virtualenvs/stock-pick-strat-lkf53h9P-py3.13/Scripts/python.exe" reports/validate/2026-10-01-ftd-on-the-fly/_scripts/capture_ftd_baseline.py --out reports/validate/2026-10-01-ftd-on-the-fly/baseline-ftd.json` (timeout 120 s); `rtk "C:/Users/de larrard alexandre/AppData/Local/pypoetry/Cache/virtualenvs/stock-pick-strat-lkf53h9P-py3.13/Scripts/python.exe" -m pytest tests/data_extract/institutionals/test_fails_to_deliver.py tests/data_aggregate/test_short_flow_features.py tests/data_aggregate/test_institutional_step_contract.py -q -s --basetemp=../../pt-ftd-red-20261001` (timeout 120 s). Expected RED is only the newly specified behavior; unrelated failures require investigation. Done when the baseline artifact exists and each new contract has a meaningful failing assertion. Recovery: correct the test or input assumption before source edits; never weaken a valid assertion to make RED disappear.

### Phase 2 — Minimal implementation and GREEN · ✅

AC-001/002/004/005. This depends on Phase 1's RED evidence. Bounded files: the extractor, two aggregate files, constants, registry/generated DDL, quality script, and the same focused tests listed above. Remove extraction metadata work, implement the clock beside FTD feature publication, pass cache directory through the cube step, remove quality-script dependency and table registry, regenerate SQL from the registry. Preserve `period` projection and stable output columns. Run only needed targeted tests first, then the broader boundary/aggregate tests. Record file-by-file diff and decisions in `03-implementation.md`. Expected GREEN is all targeted tests passing and no product vintage reference. Done when the generated SQL diff contains only the table block removal and the broader checks pass or have a scoped, diagnosed failure.

Commands: `rtk "C:/Users/de larrard alexandre/AppData/Local/pypoetry/Cache/virtualenvs/stock-pick-strat-lkf53h9P-py3.13/Scripts/python.exe" -m pytest tests/data_extract/institutionals/test_fails_to_deliver.py tests/data_aggregate/test_short_flow_features.py tests/data_aggregate/test_institutional_step_contract.py -q -s --basetemp=../../pt-ftd-green-20261001` (timeout 120 s); `rtk "C:/Users/de larrard alexandre/AppData/Local/pypoetry/Cache/virtualenvs/stock-pick-strat-lkf53h9P-py3.13/Scripts/python.exe" -m scripts.generate_schema_sql` (timeout 120 s; generated file scope checked afterward); `rtk "C:/Users/de larrard alexandre/AppData/Local/pypoetry/Cache/virtualenvs/stock-pick-strat-lkf53h9P-py3.13/Scripts/python.exe" -m pytest tests/data_store/test_store_boundary.py tests/data_aggregate/test_cube_incremental.py tests/data_aggregate/test_aggregate_regression.py tests/data_aggregate/test_part_registry.py -q -s --basetemp=../../pt-ftd-reg-20261001` (timeout 180 s); `rtk rg -n "sec_ftd_vintages" src scripts sql tests` (expect no production reference; old test names may be renamed). Recovery: revert only candidate edits in the isolated branch if necessary, preserving audit files and unrelated work; fix the concrete failing contract before continuing.

### Phase 3 — Real-data and integration validation · ✅ (findings and incomplete broad scans reported)

AC-003/005. This depends on Phase 2 GREEN and compares against the Phase 1 artifact. Bounded files: audit `_scripts/check_ftd_candidate.py` and validation outputs only; no product writes. Use `context.store` for scoped, projected reads. Build the affected FTD series in memory from live rows and pass the base worktree's `data/sec_fails_to_deliver` cache as an explicit **read-only** argument because the isolated worktree has no ignored cache. Record the exact ZIP path and mtime. Specifically inspect AAPL 2024-04-15 in `202404b`, its period publication date, and sessions immediately before/after; measure row/feature counts and compare feature columns and publication values to `baseline-ftd.json`, explaining expected FTD-only clock differences. The candidate script is the read-only FTD domain validator for this change. Generic validators inspect the already persisted cube part and therefore cannot alone prove candidate output. Do not save candidate data to PostgreSQL. Put exact commands, output paths, row windows, and any non-zero validator status in `04-validation-01.md`. Done when the real-row assertion passes, differences are explained, and every validator result is recorded; a generic validator's exit 3 is ABSTAINED rather than pass.

Commands (timeout 180 s each; `profile` precedes `redundancy`; 0 = PASS, 1 = FINDING, 3 = ABSTAINED):

```powershell
rtk "C:/Users/de larrard alexandre/AppData/Local/pypoetry/Cache/virtualenvs/stock-pick-strat-lkf53h9P-py3.13/Scripts/python.exe" reports/validate/2026-10-01-ftd-on-the-fly/_scripts/check_ftd_candidate.py --cache-dir "C:/Users/de larrard alexandre/OneDrive - The Boston Consulting Group, Inc/Documents/repos_github/PEA/data/sec_fails_to_deliver" --baseline reports/validate/2026-10-01-ftd-on-the-fly/baseline-ftd.json --out reports/validate/2026-10-01-ftd-on-the-fly/candidate-ftd.json
rtk "C:/Users/de larrard alexandre/AppData/Local/pypoetry/Cache/virtualenvs/stock-pick-strat-lkf53h9P-py3.13/Scripts/python.exe" -m src validate pull -T cube_part_institutionals -o reports/validate/2026-10-01-ftd-on-the-fly/pull
rtk "C:/Users/de larrard alexandre/AppData/Local/pypoetry/Cache/virtualenvs/stock-pick-strat-lkf53h9P-py3.13/Scripts/python.exe" -m src validate grain -T cube_part_institutionals -o reports/validate/2026-10-01-ftd-on-the-fly/pull
rtk "C:/Users/de larrard alexandre/AppData/Local/pypoetry/Cache/virtualenvs/stock-pick-strat-lkf53h9P-py3.13/Scripts/python.exe" -m src validate coverage -T cube_part_institutionals -o reports/validate/2026-10-01-ftd-on-the-fly/pull
rtk "C:/Users/de larrard alexandre/AppData/Local/pypoetry/Cache/virtualenvs/stock-pick-strat-lkf53h9P-py3.13/Scripts/python.exe" -m src validate profile -T cube_part_institutionals -o reports/validate/2026-10-01-ftd-on-the-fly/pull
rtk "C:/Users/de larrard alexandre/AppData/Local/pypoetry/Cache/virtualenvs/stock-pick-strat-lkf53h9P-py3.13/Scripts/python.exe" -m src validate bounds -T cube_part_institutionals -o reports/validate/2026-10-01-ftd-on-the-fly/pull
rtk "C:/Users/de larrard alexandre/AppData/Local/pypoetry/Cache/virtualenvs/stock-pick-strat-lkf53h9P-py3.13/Scripts/python.exe" -m src validate redundancy -T cube_part_institutionals -o reports/validate/2026-10-01-ftd-on-the-fly/pull
rtk "C:/Users/de larrard alexandre/AppData/Local/pypoetry/Cache/virtualenvs/stock-pick-strat-lkf53h9P-py3.13/Scripts/python.exe" -m src validate clip -T cube_part_institutionals -o reports/validate/2026-10-01-ftd-on-the-fly/pull
rtk "C:/Users/de larrard alexandre/AppData/Local/pypoetry/Cache/virtualenvs/stock-pick-strat-lkf53h9P-py3.13/Scripts/python.exe" -m src validate timeseries -T cube_part_institutionals -o reports/validate/2026-10-01-ftd-on-the-fly/pull
rtk "C:/Users/de larrard alexandre/AppData/Local/pypoetry/Cache/virtualenvs/stock-pick-strat-lkf53h9P-py3.13/Scripts/python.exe" -m src validate leakage -T cube_part_institutionals -o reports/validate/2026-10-01-ftd-on-the-fly/pull
rtk "C:/Users/de larrard alexandre/AppData/Local/pypoetry/Cache/virtualenvs/stock-pick-strat-lkf53h9P-py3.13/Scripts/python.exe" -m src validate catalogue -T cube_part_institutionals -o reports/validate/2026-10-01-ftd-on-the-fly/pull --catalogue scripts/cube_institutionals_catalogue.py
```

The catalogue module exposes `CATALOGUE` (verified at line 143). Record each actual command and status. Recovery: fix source if candidate findings are real; classify pre-existing data/validator limitations without changing the DB.

### Phase 4 — Documentation, independent review, final report · ✅

AC-001/005/006. This depends on Phase 3's measured candidate. Bounded files: the listed wiki pages and audit report files; fixes to prior scoped product files only if a reviewer finds a real defect. Read and edit wiki Markdown only via OpenKnowledge MCP; cite changed source locations and append the wiki log. Run diff-focused correctness and Ponytail simplicity reviews independently as Harness Validate requires; remedy material findings on isolated fix branches and re-run affected checks. Confirm source reference audit, expected diff, branch cleanliness, and no protected files/DB mutations. Generate `05-correctness-review.md`, `05-ponytail-review.md`, `06-final.md`, and a self-contained `report.html` under the evidence root. Done when reviews, wiki sync, final report, and completion checklist below agree; unresolved findings must be called out rather than marking readiness. Recovery: revise the candidate and validation evidence for real findings.

Commands: `rtk rg -n "sec_ftd_vintages" src scripts sql tests` (expect no production reference); search the affected wiki pages for that name **through OpenKnowledge MCP**, reviewing any historical mention in context; `rtk cmd /c git diff --check`; `rtk cmd /c git status --short`; targeted pytest from Phase 2 after any fix. HTML must stand alone without external assets. No commit/push/PR is implied by this plan.

## Completion checklist

- [x] AC-001 through AC-006 each have linked evidence, with RED, GREEN, real-data, and reviewer outcomes recorded.
- [x] No runtime/script/registry/generated-DDL vintage dependency; legacy physical table explicitly left untouched.
- [x] Historical/0/1/2-day/cache-fallback cases, fail-closed period cases, and extraction resume/replacement all pass.
- [x] Scoped real-row candidate and before/after column/value comparison explain FTD clock changes; generic persisted-data checks are not misrepresented as candidate checks.
- [x] Only approved paths changed; wiki accessed only through OpenKnowledge; report is self-contained and all limits/findings disclosed.

## Approval requested

Approve this exact plan and start implementation, including edits to `src/constants/constants.py`, `src/data_store/schema.py`, generated `sql/schema.sql`, the listed code/tests/wiki files, and creation of local harness evidence. Approval does **not** include a physical PostgreSQL table drop, DB writes, `data/` edits, config changes, or aggregate fingerprint changes.
