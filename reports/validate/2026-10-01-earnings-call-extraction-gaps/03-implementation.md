# Implementation — defeatbeta EC extraction

**Spec**: 01-spec.md v3 · **Research**: 01-research.md · **Plan**: 02-plan.md v2 (approved 2026-10-02).
**Base**: `dev` @ `71a9ecd`. **Branch / worktree**: `harness/earnings-call-extraction-gaps` at `reports/worktrees/earnings-call-extraction-gaps`.
**Started**: 2026-10-02 (orchestrator: main session). **Precondition for P5**: the user disabled the Airflow DAG (2026-10-02).

## Phase table

| Phase | Status | ACs | Changed paths | Commit |
|---|---|---|---|---|
| P0 Preflight | ✅ | — | worktree only | — |
| P1 Split library | ✅ | AC-005..007 | `src/utils/earnings_call_split.py`, `tests/utils/test_speaker_turn_split.py`, 16 fixtures | `ede92af` |
| P2 Schema | ✅ | REQ-001 | `src/data_store/schema.py`, `sql/schema.sql` | `0b46784` |
| P3 Extractor | ✅ | AC-003, AC-004 | new extractor + test; `configs/data.yml`, `cli.py`, `step_extract_behavioral.py`, DAG + DAG test, `docker-compose.yml`, `utils_earnings_call_cache.py` | `ad4bab9`, `d887eec` |
| P4 Aggregation rewiring | ✅ | AC-007, AC-008 | `earnings_call_embeddings.py`, `earnings_call_features.py`, `checks/earnings_calls.py`, `configs/validate.yml`, `constants.py` (version), tests, `tests/fixtures/earnings_call_rows.py` | `3ed3585` |
| P5 Live cutover | ⬜ | AC-001..003, 006, 008 | | |
| P6 Wiki removal | ✅ | AC-009 | 3 deleted, 8 edited (docstrings, DAG test, incremental test) | `63edc8b` |
| P7 Legacy retirement | ✅ (code; DB and file deletions await the user) | AC-009 | 6 modules + 8 tests deleted; cache util, constants, `configs/paths.yml`, comments; 28 files, +48/−2,710 | `56dea8b` |
| P8 Docs | ✅ | AC-010 | 11 wiki pages in the main tree (OK MCP) + `wiki/TODO.md` tone-drift entry + `log.md`; memory notes rewritten | — (wiki lives in the main tree, uncommitted) |

## P0 — baseline

- **Worktree** created at `71a9ecd`. `src/data_aggregate/utils/target` is tracked at this commit, so nothing was copied. Copying `.env` was blocked by a permission rule; the suite ran without it.
- **Full pytest** (`_out/p0-pytest-full.txt`): `8 failed, 1961 passed, 29 skipped` in 33 min. Pre-existing failures:
  1. `tests/dags/test_dag_matches_part_registry.py::test_dag_chain_is_derived_from_the_registry`
  2. `tests/data_aggregate/test_betas.py::test_persisted_betas_require_a_current_own_price`
  3. `tests/data_aggregate/test_fundamental_features.py::test_real_panel_wellformed_and_bounded`
  4. `tests/data_aggregate/test_institutional_features.py::test_first_publication_shares_chg_error_does_not_regress`
  5. `tests/data_aggregate/test_panel_merge_duplicates.py::test_availability_is_applied_before_peer_statistics`
  6. `tests/data_extract/behavioral/test_wiki_incremental.py::test_wiki_incremental_reads_last_date_per_ticker` — removed by P6
  7. `tests/data_extract/common/test_identity.py::test_every_flagged_group_resolves_to_its_reviewed_verdict` — a `reports/planning` CSV absent from the worktree
  8. `tests/data_extract/prices/test_adjustment_basis.py::test_the_dividend_factor_is_monotone_and_terminates_at_one`
- **DB snapshot** (`_out/p0-db.json`):

| Table | Rows | Size | Distinct calls | Tickers |
|---|---|---|---|---|
| `earnings_call_sections` | 110,994 | 1,618 MB | 28,338 | 498 |
| `earnings_call_sentiment` | 55,863 | 11 MB | 27,933 | 497 |
| `earning_calls_embedding` | 1,379,014 | 8,658 MB | 27,898 | 497 |
| `cube_part_text` | 1,798,087 | 559 MB | — | 498 |

- `cube_part_text` has 30 `f_ec_*` columns (15 base/peer pairs), 2005-11-01 → 2026-09-04.
- Anomalies noted: SJM 2027Q1 has no `full` row; sentiment and embeddings each miss about 400 calls; the cube stops at 2026-09-04.

## P1 — speaker-turn split library

- **API**: `clean_paragraphs`, `qa_start`, `label_turns`, `split_call -> CallSplit(prepared_remarks, qa, turns, status)`. Pure; no `src` consumer yet.
- **RED**: the test file failed to collect (`ModuleNotFoundError`). **GREEN**: `pytest tests/utils/test_speaker_turn_split.py -q` → `69 passed`. The exact Q&A start and status hold on 16/16 hand-labelled calls; 120/120 analyst turns are tagged question; 0 management turns are tagged question.
- **Sample** (585 calls, `_out/p1-sample-check.txt`):

| Measure | Result | Target |
|---|---|---|
| `ok` | 99.49 % (582 ok, 3 no_qa) | ≥ 98.5 % |
| Prepared share q05/q50/q95 | 0.19 / 0.34 / 0.57 | q05 ≥ 0.15 |
| ok calls passing `assess_earnings_call_sections` | 100 % | — |
| Analyst turns labelled answer (follow-ups) | 8.7 % → 0.3 % | — |
| Question turns > 300 words | 214 → 19 | — |
| AXON 2025Q4 exchanges | 2 (up to 83 answers each) → 24 (≤ 5) | — |

- The 3 `no_qa` calls are two transcripts without a Q&A session and one fireside interview.
- **Logic additions over the prototype**, each measured on the sample: a short host turn that opens the queue starts the Q&A; a strong hand-off phrase counts without the word floor; more hand-off phrases; zero-width characters stripped from speaker names (IBM); a new speaker asking a short question after an answer is tagged question.
- **Open issues** (accepted, small): MCHP 2025Q4 starts the Q&A early on a third executive's short "?" turn; TJX source speaker labels are unreliable; IR-read questions (PLTR, COIN) are dropped as logistics.

## P2 — schema

- **Registry**: `Table("earnings_call_sections", ("ticker", "quarter", "paragraph"), date_col="as_of", date_type_cols=("as_of",), freshness="quarterly")`.
- **DDL**: one hand-spliced hunk in `sql/schema.sql` (`_out/p2-schema-sql.diff`): `tag`, `url`, `text` removed; `paragraph BIGINT NOT NULL`, `as_of DATE NOT NULL`, `transcript_id BIGINT`, `speaker TEXT`, `content TEXT`; the `as_of` index is kept. No live DB statement.
- **Verify**: `pytest tests/data_store -q` → `22 passed, 19 skipped`.
- **Expected breakage until P4** (`_out/p2-ec-impact.txt`): 4 tests whose fixtures write old-shape frames — `test_earnings_call_features.py::test_sentiment_kpis_streamed_equals_batch`, `::test_malformed_refresh_marker_survives_until_cube_write_ack`, `tests/validate/test_earnings_calls.py::test_earnings_call_validator_reports_coverage_schema_and_quality`, `::test_coverage_uses_point_in_time_lineage_and_separates_no_call_names`.

## P3 — fast extractor

- **Module**: `fetch_earnings_call_transcripts.extract_earnings_calls(context, cfg, full=, tickers=)`. Reads are pinned to the repo SHA. The no-op compares the file's LFS sha256, stored in the run manifest's existing `identity_scope_fingerprints` slot under the dataset path. `manifest_window(reconcile_days)` schedules the full reconcile.
- **Source measurements** (`_out/p3-duplicates.txt`):
  - 0 duplicate (symbol, fiscal_year, fiscal_quarter) keys;
  - `transcripts_id` is NULL on 2,914 of 33,676 roster calls, so a re-issue is detected when the id **or** the date changes;
  - 23 roster calls on 10 tickers appear on one date under two fiscal labels (a provider relabel). `one_call_per_date` keeps the label whose ordinal is one below the next call's label. Results: DG 2026-03-12 → 2026Q4, PCAR 2011-04-21 → 2011Q1, RJF 2015-07-23 → 2015Q3, LOW 2022-11-16 → 2023Q3; 0 duplicates remain (33,653 calls).
- **Wiring**: CLI `extract-earnings-calls [-F] [-t]`; the step calls the extractor; one DAG task in the default pool. The `scrape` pool was removed from the DAG and from airflow-init, since only the EC tasks used it.
- **Config**: a new `earnings_calls` block (`lookback_days: 45`, `read_workers: 4`, `reconcile_days: 7`). `local.paths.call_transcripts` is legacy-only and goes in P7.
- **Re-issue**: deletes the old paragraphs, invalidates derivatives and writes a pending refresh marker at the earlier call date, through a shared `pending_refresh_markers` helper.
- **Verify**:
  - targeted: 13 passed, plus 9 after the follow-up, including a live HF smoke test (AAPL 84 calls, MSFT 83 calls, 12 s);
  - `tests/data_extract tests/dags`: `3 failed, 984 passed, 22 skipped`, exactly the P0 baseline.
- **Timing** (`_out/p3-timing.txt`): revision 2.2 s; footer 2.3 s; 549/1,195 row groups selected for the full run; index read 70 s with 4 workers (49 s with 8); incremental selection 223 row groups.

## P4 — aggregation and validator rewiring

- **Reads**:
  - Call keys come from `store.distinct` (ticker, then quarter per ticker); no text is read for them.
  - Texts are read projected (`ticker, quarter, paragraph, as_of, speaker, content`), scoped by ticker (batches of 25 when streaming) and ordered by paragraph, then passed through `split_call`.
  - Non-ok calls and calls failing the gate are skipped. `as_of` comes from the paragraphs; `_daily_frame` is untouched.
- **Embeddings**:
  - Removed: the header regexes, `split_turns`, `split_qa_exchanges`, `split_qa_pairs`, the duplicated role and cleaning rules, `_section_calls` and `_calls_from_frame` (−355 lines).
  - `_yield_call_turns` yields `CallSplit.turns`. Keys, tag values and ordering are unchanged; `person` is always the source speaker.
- **Cache**: `EARNINGS_CALL_SENTIMENT_CACHE_VERSION = "speaker-v1"`. Embeddings have no version key; completeness is judged on (ticker, quarter, model). P5 must therefore start from an empty `earning_calls_embedding`, which the planned `*_legacy` rename provides.
- **Validator** (thresholds in `configs/validate.yml`):
  - kept: per-ticker coverage;
  - added: per-calendar-quarter coverage vs the roster; grain checks (duplicate PK, null paragraph or as_of, more than one as_of per call, missing paragraph 1, non-contiguous paragraphs); split status counts with ok rate ≥ 0.985 and prepared q05 ≥ 0.15; as_of vs the nearest `earnings_surprises` date.
- **Verify**:
  - `pytest tests/data_aggregate -k earnings_call -q` → 28 passed;
  - `tests/validate/test_earnings_calls.py` → 3 passed;
  - the header-regex grep returns nothing;
  - `tests/data_aggregate tests/validate` → `4 failed, 708 passed`, the 4 P0 baseline items (`_out/p4-pytest.txt`).
- **PIT test**: a Wednesday call is first visible on Thursday; a Friday call on Monday.
- **Accepted carry-overs**: calls that never pass the gate are re-read and re-split each run, as before.

## P4b — text cleaning (commit `166a9f2`)

- **Paths**: `src/utils/earnings_call_split.py`, `src/utils/text_metrics.py`, `src/constants/constants.py`, `earnings_call_embeddings.py`, the split, embeddings and limits tests, and the COIN fixture.
- **Splitter fix**: sentences are now split only at terminal punctuation followed by whitespace. Broken decimals in the FinBERT input go from 95,260 broken / 0 intact to 171 / 90,335; the 171 are already broken in the source.
- **Cleaner** (one shared function):
  - removes courtesy, backchannel, self-introduction, addressed-name and hand-off sentences anywhere;
  - removes boilerplate and the roster from prepared remarks;
  - replaces names removed mid-clause with "a colleague";
  - keeps hedges.
- **Precision** (true hits out of 20): courtesy, self-introduction and addressed name 20/20; boilerplate 19–20/20; roster 19/20; hand-off 19/20.
- **Residual noise** (% of words, before → after):

| Measure | Before | After |
|---|---|---|
| boilerplate, prepared | 5.0 % | 1.55 % (the rest carries figures) |
| roster, prepared | 0.30 % | 0.001 % |
| courtesy, Q&A | 1.73 % | 0.25 % |
| logistics, Q&A | 2.33 % | 0.07 % |
| analyst text, Q&A | 27.1 % | 0 |
| operator text, Q&A | 4.2 % | 0 |

- **Turns**: prepared 14,580 → 14,375; questions 66,125 → 60,345; answers 83,838 → 69,750 (10-word minimum). Exchange indices stay contiguous.
- **Sentiment Q&A** is now management answers only. 9 of 2,880 ok sample calls fail the gate because the transcript is defective (truncated Q&A, or answers merged into analyst paragraphs).
- **585-call ok rate**: 99.32 %. Prepared share q05/q50/q95: 0.15/0.34/0.56.
- **Cache versions**:
  - sentiment: `speaker-clean-v1`;
  - embeddings: model tag `text-embedding-3-small:speaker-clean-v1`, so the 72 probe rows are treated as stale and re-embedded by the normal path.
- **Tests**: RED first (`_out/p4b-pytest-red.txt`); GREEN: split + aggregate 103 passed, validate + utils 120 passed.
- **Cost**: splitting takes 82 ms per call (33 ms before).
- **Accepted**: calls with no prepared management text (e.g. COIN live-on-X) drop out of the sentiment features.

## P5 — live cutover (PAUSED at the cost gate, 2026-10-02)

- **Done**:
  - snapshot `_cache/p5_cube_part_text.dump` (1,798,087 rows);
  - 3 tables, PKs and indexes renamed to `*_legacy` (`_out/p5-renames.txt`, with the rollback steps).
- **Full extract**: 33,591 calls and 2,804,060 paragraphs in 1,119 s (18.6 min: misses the 15 min target, under the 30 min replan trigger; 1,091 s of it is DB writes).
- **No-op re-run**: 1.3–1.7 s in-process.
- **Parity** (`_out/p5-parity.txt`): exact — 33,591 = 33,591, 0 only-in-source, 0 only-in-DB, 0 `as_of` mismatches, 487 tickers.
- **Validator** (`_out/p5-validator.txt`):
  - split ok 98.71 %; prepared q05/q50/q95 0.165/0.363/0.581; grain clean;
  - `as_of` vs release: 23,312 same day, 6,157 at +1 day, 963 at 2–7 days, 756 beyond 7 days;
  - 1 score-9 finding, pre-existing: the live `cube_part_text` still carries an older 30-column `f_ec_*` set instead of the 12 approved columns. `build-text -F` resolves it.
- **Cost probe** (`_out/p5-cost-probe.log`, 72 calls written to the new sentiment and embedding tables):
  - OpenAI: about 10.5k tokens per call, ≈ $7 for all calls (passes);
  - FinBERT on CPU: 27–37 s per call, ≈ 280 h for all calls (**gate failed**).
- **User decision (2026-10-02)**: the user runs FinBERT; the agent does everything else. First pause the code work and measure the text noise in the new table (lengths per person, empty and one-word turns, names, pleasantries, politeness drift over time) before any embedding run.

### P5 continuation (2026-10-02 afternoon)

- **Embedding stage**: `build-text` always scores FinBERT before embedding, so a throwaway driver (`_scripts/p5_embed_only.py`) calls the production `embed_earnings_calls`.
- **Cost on cleaned text**: 9,150 tokens per call, projected at $6.15 for all calls.
- **Full embedding pass**: 6 shards (`_out/p5-embed-shard*.out`), CPU-bound on the `float8[]` inserts at about 100 calls/min. 5,317 calls done at 15:01; ETA about 19:45.
- **FinBERT**: run by the user. Command and checks in `_out/p5-finbert-command.txt`; driver `_scripts/p5_score_only.py`. About 33,155 valid calls, 66.3k sections, 575k windows of 510 tokens.
- **Incremental extract**: no new revision (sha `cd41c0ac6c9e`, fingerprint matches); no-op in 3.9 s in-process.
- **Validator re-run**: BLOCKED. It hit the 30 min background limit under CPU load. PIDs 9876 and 30904 are still running; the permission system refused to stop them, and the decision is the user's. Re-run after the embedding pass.
- **User decision (2026-10-02, later)**: no full embedding, FinBERT or cube runs by the agent; the user runs them. The agent proves the chain end to end on a 3-ticker sample, then moves to P7 and P8. The 6 embedding shards were stopped by PID at 7,451 calls; the cache resumes.
- **Next**: `_out/p5-next.txt` (`build-text -F` after FinBERT, then the coverage, PIT and fingerprint checks).

### P5 end-to-end sample (ABNB, PLTR, CEG)

- **Run**: driver `_scripts/p5_e2e_build_text.py` = the production `StepCubeText.run(full=True)` scoped to the 3 tickers, because `build-text` has no `-t`. It finished in 61 min with exit 0 and no errors; FinBERT ran on CPU.
- **Caches**: all 78 calls were scored (156 sections) and embedded under the `speaker-clean-v1` tags.
- **Cube**: `cube_part_text` was **replaced** (`store.replace`). It now holds only the 3 sample tickers (3,780 rows) with exactly the 12 approved `f_ec_*` columns. Rollback: `_cache/p5_cube_part_text.dump`. The user reruns the full cube.
- **Non-null share** after each ticker's first call:
  - ABNB: 0.99 for the level features, 0.82 for history, 0.95 for delta and distance;
  - PLTR: similar, with coherence NULL on 249 sessions whose calls have a single question (by design);
  - CEG: 0.82, because its 2008–11 predecessor-issuer calls are masked by price availability, and one source call is missing (2024Q4).
- **PIT**: on 4 calls, the value on the `as_of` session is still the previous call's (or NULL), and the new value appears on the next session. Example: ABNB call 2025-08-06, tone 0.4137 → 0.5857 on 2025-08-07.
- **Ranges**: tone [0.21, 0.90]; qa_gap [−1.18, 0.25]; distances [0.023, 0.336]; 0 inf or NaN.
- **Validator**: `-T earnings_call_sections` was the wrong target: the check validates the `cube_part_text` feature schema and reads the transcripts internally. It is being re-run with `-T cube_part_text` (`_out/p5-validator-3.txt`).
- **Validator `-T cube_part_text`** (`_out/p5-validator-3-cube.txt`, `_out/earnings_calls.json`, 17:35→19:07, 92 min under machine load):
  - extraction over all 33,591 calls: 33,091 valid, 500 malformed; split ok 98.63 % (floor 98.5 %); prepared share q05 0.20 (floor 0.15); median ticker coverage 0.90;
  - the score-9 schema finding is gone (12 columns, 3,780 rows);
  - 16 score-7 findings, all recent-vs-train drift checks: PSI 0.41–4.3 and `*_vs_hist` missingness −0.85. They are an artefact of a 3-ticker cube (a handful of values per distribution; history features absent in the early train years) and must be re-judged after the user's full cube rebuild.
  - A redundant second run was stopped by PID; the cube had not changed.
- **Note for P8**: the `tone_delta` and `length_delta` docstrings say "vs the prior call", but the code requires the prior **consecutive** quarter.

## P7 — ROIC, Fool and kurry retirement (commit `56dea8b`)

- **Deleted**: `fetch_earnings_calls.py`, `fetch_roic_transcripts.py`, `fetch_hf_transcripts.py`, `utils_missing_quarters.py`, `utils_split_qa.py`, `utils_behavior.py`, and their 8 tests (30 test functions).
- **Removed**:
  - constants: `FOOL_BASE`, `EARNINGS_CALL_REQUEST_PAUSE`, `EARNINGS_CALL_REPORT_GRACE_DAYS` and the three `ROIC_*` constants;
  - config: `local.paths.call_transcripts` (`configs/paths.yml`);
  - the legacy save function in `utils_earnings_call_cache.py`.
- **Kept**: `EARNINGS_REPORT_TO_QUARTER_LAG_DAYS` and `NO_EARNINGS_CALL_TICKERS`, both still used; the generic `Crawler` and `polite_http` helpers, which have their own tests and a Google Trends consumer.
- **Docstring**: `tone_delta` and `length_delta` now say "prior consecutive quarter", matching the code.
- **Grep residue** (`_out/p7-grep.txt`, 52 hits):
  - 46 are the ROIC ratio and 4 a Sharadar `missing_quarters` column — unrelated;
  - 1 is the English word "fooling";
  - 1 is a docstring in `store.py:380` naming the deleted `fetch_hf_transcripts`, left because `data_store` is ask-before-edit.
- **Full pytest** (3 shards after the 30 min cap, `_out/p7-pytest-full.txt`): `7 failed, 2,019 passed, 26 skipped`, against P0's 8 / 1,961 / 29. The 7 are the P0 failures minus the wiki test (removed in P6); **no new failures**.
- **User confirmed both deletions (2026-10-02).** The agent's combined `DROP TABLE` + `rm -rf data/call_transcripts` command was denied by the permission system, so the exact commands were handed to the user to run. Targets verified just before: exactly 3 `*_legacy` tables (8,658 / 1,618 / 11 MB); `data/call_transcripts` = 113 ticker folders + `hf_sp500_transcripts.parquet` (1.74 GB) + `transcript_index.json`.
- **Pending the user's go-ahead** (`_out/p7-pending-deletions.txt`):
  - tables: `earning_calls_embedding_legacy` (8,658 MB), `earnings_call_sections_legacy` (1,618 MB), `earnings_call_sentiment_legacy` (11 MB), about 10 GB in total;
  - files: `data/call_transcripts`, 2.07 GB (kurry parquet 1.82 GB, Fool HTML 247 MB, index JSON). It is the same host directory as the Airflow bind mount.

## P9 fixes (commit `8161e52`)

- **F-001**: on full and reconcile runs, `stale_calls` lists stored keys absent from the deduplicated source; `_remove_calls` invalidates their derivatives, writes a pending marker and deletes them. Incremental runs never delete. The new validator check `ticker_dates_with_multiple_calls` files a score-8 finding when it is above 0. Covered by 3 tmp-parquet tests and 1 validator test.
- **F-016**: a Q&A-only speaker counts as management when they speak in ≥ 3 exchanges, are never announced, never read the queue and are not labelled analyst. New fixtures: TXN 2025Q1, BMY 2026Q1, JPM 2024Q3.
  - Live scan: ok calls with a ≥ 1,000-word speaker tagged question, 61 → 3 of 5,241.
  - 585-call ok rate: 99.32 %.
  - Cache tags move to `speaker-clean-v2`.
  - Side effect: calls where an IR host reads the questions (NFLX, DIS) now have almost no Q&A exchanges.
- **F-002**: a pending marker survives until its call is actually scored.
- **Verify**:
  - `--collect-only`: 2,069 collected;
  - split, validate, extract and dags: 108 passed, 1 failed (the P0 baseline);
  - aggregate EC: 30 passed.
- **User re-baselines R-1 to R-3** are recorded in the plan.
- **Legacy deletions**: the three `*_legacy` tables were dropped by the user (verified: 0 tables). `data/call_transcripts` still holds 457 files; the user retries.

## Re-validation and merge (2026-10-03)

- **Re-validation**: CONDITIONAL. F-001, F-002 and F-016 were verified independently.
  - F-016: speakers of ≥ 1,000 words tagged question fall from 413 to 63 over all history, and from 61 to 3 since 2024.
  - Side effect: recent calls with 0 Q&A exchanges go from 8 to 16. These are NFLX, DIS and MCK, where an IR host reads the questions.
  - Thresholds: AC-001 R-1 passes (lowest quarter 97.54 %); AC-006 R-3 has 0 of 5,241 calls failing.
  - Still open: AC-008 waits for the user's rebuild; F-005 is open.
- **`dev` had moved**: it carried the user's own commits, including the P8 wiki pages (`091278b`), a duplicate of the pageviews removal (`191fc65`) and a `run_manifest` change (`48d80d9`). That change still keeps `identity_scope_fingerprints`, `get_entry`, `manifest_window` and `record_run`.
- **Merge**: `dev` was merged into the branch with no conflicts (`61f18c6`).
  - Collect-only: 2,070 tests.
  - EC, extract, common, DAG, utils and store: 408 passed, 1 failed (P0 baseline `test_identity`; the DAG-chain baseline is fixed on `dev`).
  - Aggregate EC: 30 passed (`_out/merge-pytest-a.txt`, `_out/merge-pytest-b.txt`).
  - Then merged into `dev` with `--no-ff` as `35be44e`. Not pushed.

## P6 — Wikipedia pageviews removal

- **Deleted**: `fetch_wiki_pageviews.py`, `test_wiki_incremental.py`, `test_wiki_resolution.py` (all pageviews-only).
- **Edited**: docstrings in both extract steps, `polite_http.py`, `crawler.py`, `docker-compose.yml`; the pageviews assertion in `tests/dags/test_extraction_dag.py`; the wiki day-window assertions in `test_rate_limit_and_incremental.py`.
- **Not needed**: no CLI, DAG, schema, constant or config reference existed; no feature reads pageviews.
- **Verify**: `pytest tests/data_extract tests/dags tests/utils/test_polite_http.py -q` → `3 failed, 981 passed, 22 skipped`. The 3 failures are P0 baseline items 1, 7, 8; the wiki failure is gone. Evidence: `_out/p6-pytest.txt`, `_out/p6-grep.txt`.
- **Grep residue**: one docstring mention in the protected `common/incremental.py:11`, left untouched.
- **Docs for P8**: `wiki/flows/nightly-data-refresh.md`, `wiki/guides/run-the-pipeline.md`, `wiki/reference/live-database.md`.

## Deviations

- P3: the revision is stored in `record_run(identity_scope_fingerprints=...)`. `record_run` has no free field, and this slot already carries input fingerprints forward across runs, so `data_store` and `run_manifest` stay unedited.
- P3: added a same-date dedup rule; the plan did not foresee the source relabel.
- P2: `paragraph` and `transcript_id` are `BIGINT` (the file's and `ddl.sql_type`'s integer convention) instead of `INT`.
- P6: the protected `common/incremental.py` keeps a docstring mention of wiki pageviews (protection outranks a comment cleanup).
