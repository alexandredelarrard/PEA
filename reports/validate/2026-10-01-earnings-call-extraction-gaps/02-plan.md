# Plan v2 — defeatbeta EC extraction, speaker-turn split, legacy and wiki retirement

**Status**: APPROVED v2 (user, 2026-10-02). Airflow DAG disabled by the user (P5 precondition met).
**Spec**: 01-spec.md v3 · **Research**: 01-research.md · **Run**: 00-run.md.
**Base**: `dev` @ `71a9ecd` → branch `harness/earnings-call-extraction-gaps`, worktree `reports/worktrees/earnings-call-extraction-gaps`. The unrelated dirty files stay in the main tree.

## 1. Objective

- **Extraction**: one fast, incremental extractor writes the raw defeatbeta paragraphs into a re-schemed `earnings_call_sections`.
- **Split**: one shared, pure speaker-turn split rebuilds what sentiment and embeddings consume today.
- **Features**: the 12 `f_ec_*` features and `cube_part_text` are rebuilt with an unchanged contract.
- **Retirement**: ROIC, Fool, kurry and the Wikipedia pageviews fetcher are removed.

New features are out of scope.

## 2. Execution protocol: one sub-agent per phase

1. **Dispatch.** The orchestrator, the main session, sends each phase to a **fresh** `general-purpose` sub-agent in the worktree. The prompt contains only:
   - the phase section of this plan, plus the spec and research paths;
   - the AGENTS.md hard rules;
   - the allowed write paths;
   - the base commit;
   - the exact verify commands;
   - the stop gates.
2. **Token discipline.**
   - Every shell command is prefixed with `rtk`.
   - Tests run as `rtk "$PY" -m pytest <targets> -q`; only failures and the summary line are reported.
   - No full logs. Large outputs go to `_out/` files, and only their paths come back.
   - Read files by line range, not whole.
3. **Hand-back contract.**
   - The sub-agent commits only its owned paths on the branch.
   - It writes `_out/pNN-*.txt` evidence.
   - It returns at most 300 words: commit SHA, changed paths, exact verify results, deviations and open issues.
   - It never edits 02-plan, 00-run or 03-implementation; those are the orchestrator's.
4. **Review and close.**
   - The orchestrator reviews `git diff --stat` and the targeted hunks plus the evidence.
   - It updates the phase marker, `03-implementation.md` and `00-run.md`.
   - The sub-agent is **stopped** (TaskStop) as soon as it hands back. A follow-up correction goes to a new sub-agent with a short finding contract.
5. **Stop gates.**
   - A sub-agent never performs a gated action: live DB rename or drop, cache deletion, cost overrun, fingerprint update.
   - It stops and returns the measured numbers; the orchestrator asks the user.
6. **Concurrency.** Only P1 and P6 run in parallel, since their write sets are disjoint. Everything else runs sequentially.

## 3. Current state → end state

**Current state:**

- **Extract**: HF kurry → gap logic (calendar vs fiscal bug) → ROIC → Fool (wrong `as_of`). A regex split anchored on `Operator`.
- **`earnings_call_sections`**: PK `(ticker, quarter, tag)`, 1.6 GB of duplicated texts.
- **Aggregation**:
  - `earnings_call_features.py` scores the section texts.
  - `earnings_call_embeddings.py` re-parses speaker headers (`split_turns`).

**End state:**

- **`earnings_call_sections`**: PK `(ticker, quarter, paragraph)` with `as_of`, `transcript_id`, `speaker`, `content`, filled by `extract_earnings_calls()`.
- **Split**: `src/utils/earnings_call_split.py` serves aggregation and validate.
- **Embeddings**: take turns from the split.
- **Sentiment**: takes the header-free section texts from the split.
- **Unchanged**:
  - KPI math;
  - the derived schemas;
  - `_daily_frame`.

## 4. Decision ledger

| ID | Decision | Status |
|---|---|---|
| U1–U10 | See spec v3 | DECIDED (user) |
| P-1 | Paragraph grain; columns `ticker, quarter, paragraph, as_of DATE, transcript_id INT, speaker TEXT, content TEXT` | DECIDED (U7) |
| P-2 | `as_of` = `report_date` with no +1: `side="right"` already gives the one-session lag (U8) | DECIDED |
| P-3 | Full history from 2005 | Proposed (same read cost; better coverage than kurry) |
| P-4 | Derived tables recomputed in full; legacy renamed `*_legacy`, dropped in P7 after confirmation | Proposed |
| P-5 | One DAG task `extract-earnings-calls` replaces download and ingest; the pool `scrape` is removed if no other task uses it | Proposed |
| P-6 | Dataset repo and path are module constants of the extractor (single consumer). Tunables go in `configs/data.yml` `earnings_calls`: `lookback_days: 45`, `read_workers: 4`, `reconcile_days: 7` | Proposed |
| P-7 | Parallel reads, single-thread writes; table created before threads start (known cold-table race) | Proposed |
| C-1 | Text cleaning before FinBERT and embeddings, in the shared split library: repair the sentence splitter (decimals, abbreviations); Q&A sentiment text = management answers only; courtesy and backchannel sentences removed anywhere; safe-harbor, IR boilerplate and the participant roster removed from prepared remarks (regex precision ≥ 18/20); self-introductions and addressed names removed; hedges kept | DECIDED (user, 2026-10-02, after the noise research) |
| C-2 | Minimum cleaned turn length of 10 words for embedded questions and answers | DECIDED (user) |
| C-3 | Tone drift over time: deferred; recorded as a TODO in P8 | DECIDED (user) |
| C-4 | FinBERT full scoring is run by the user; the agent runs everything else | DECIDED (user) |
| R-1 | AC-001 floor re-baselined: tickers with a call per calendar quarter ≥ 97 % of the in-scope tickers that quarter (≈ 472 of 487) | DECIDED (user, after P9 F-003) |
| R-2 | AC-003 accepted: a full build takes 18.6 min (bound by DB writes); no-op ≤ 4 s; the first real incremental is timed from the nightly run | DECIDED (user, after P9 F-004) |
| R-3 | AC-006 re-baselined: ≥ 60 % of **management** words kept in prepared remarks plus answers (analyst text is excluded by design, C-1); re-measured after the F-016 fix | DECIDED (user) |
| R-4 | P9 findings F-001, F-016 and F-002 are fixed before the user's rebuild; both cache tags move to `speaker-clean-v2` | DECIDED (orchestrator, within the approved scope) |
| P-8 | Role logic moves from `split_turns` steps 2a/2b into `src/utils/earnings_call_split.label_turns`; the embeddings module imports it | Proposed |

## 5. Traceability

| AC | Checks |
|---|---|
| AC-001, AC-002 | P5 validator; parity script |
| AC-003 | P3 no-op unit test; P5 timed runs |
| AC-004 | P3 re-issue test |
| AC-005, AC-007 | P1 fixtures; P4 embedding tests and grep |
| AC-006 | P1 sample run; P5 validator over all calls |
| AC-008 | P4 contract and PIT tests; P5 coverage vs legacy |
| AC-009 | P6 and P7 grep plus full pytest vs P0 |
| AC-010 | P8 OK lint |

## 6. Write scope

**New:**

- `src/utils/earnings_call_split.py`
- `src/data_extract/utils/behavioral/fetch_earnings_call_transcripts.py`
- `tests/utils/test_speaker_turn_split.py`
- `tests/fixtures/earnings_calls/*.json`
- `tests/data_extract/behavioral/test_earnings_call_transcripts.py`

**Edit:**

- Aggregation: `earnings_call_features.py` (section reads), `earnings_call_embeddings.py` (call and turn reads; `split_turns` header parsing removed).
- Extract wiring: `step_extract_behavioral.py`, `step_extract_all_data.py`, `src/data_extract/cli.py`, `src/dags/dag_data_extraction.py`.
- Validator: `src/validate/checks/earnings_calls.py`.
- `utils_earnings_call_cache.py`: keep `invalidate`; drop `save_earnings_call_sections`.
- `src/utils/polite_http.py`: host entries.
- Tests:
  - features, embeddings, embedding limits;
  - `tests/validate/test_earnings_calls.py`;
  - `tests/dags/test_extraction_dag.py`;
  - `tests/utils/test_polite_http.py`;
  - `tests/data_store/test_read_equivalence.py`.

**Delete:**

- **ROIC, Fool, kurry:**
  - `fetch_earnings_calls.py`
  - `fetch_roic_transcripts.py`
  - `fetch_hf_transcripts.py`
  - `utils_missing_quarters.py`
  - `utils_split_qa.py`
  - `utils_behavior.py`
  - their 8 test files
- **Wiki:**
  - `fetch_wiki_pageviews.py`
  - `test_wiki_incremental.py`
  - `test_wiki_resolution.py`

**Protected (must not change):** the roster Wikipedia scraper — `prices/fetch_tickers.py`, `utils/universe.py`, `common/{gics,entity_lineage,identity,incremental}.py`, `def14a/fetch.py`, `ssl_setup.py`.

**Risk-zone edits (approved by this gate):**

| Path / effect | Change |
|---|---|
| `schema.py` | Replace the `earnings_call_sections` definition (PK, columns; `date_col=as_of`, `freshness=quarterly`); remove any `wiki_pageviews` trace |
| `sql/schema.sql` | Hand splice: replace only the `earnings_call_sections` block (the generator is lossy); remove any wiki block |
| `constants.py` | Bump `EARNINGS_CALL_SENTIMENT_CACHE_VERSION` to `speaker-v1`. Delete `FOOL_BASE`, `ROIC_*`, `EARNINGS_CALL_REQUEST_PAUSE`, `EARNINGS_CALL_REPORT_GRACE_DAYS`. Keep `EARNINGS_REPORT_TO_QUARTER_LAG_DAYS` and `NO_EARNINGS_CALL_TICKERS`. |
| `configs/data.yml`, `configs/validate.yml` | `earnings_calls` blocks |
| `src/context.py:27` | Comment only |
| Live DB | P5: rename `earnings_call_sections`, `earnings_call_sentiment` and `earning_calls_embedding` to `*_legacy`; create the new table; load; rebuild. P7: drop `*_legacy` **after confirmation**. |
| `data/call_transcripts/` | P7: delete the kurry parquet, Fool HTML and index JSON (local and Airflow volume) **after confirmation** |
| Fingerprint baseline | Only if P5 shows it covers `cube_part_text`; gated |
| `.env` | The user removes `ROIC_API_KEY` (not edited by the agent) |

## 7. Testing strategy

- **Known-truth tests**:
  - split boundaries and roles on 14+ hand-labelled fixtures;
  - row-group pruning and key diff on a tmp parquet.
- **Real data**:
  - the 585-call sample;
  - the live full load;
  - the validator over all calls;
  - coverage and PIT vs legacy.
- **Live-DB check of `as_of` DATE round-trip**: sqlite hides `datetime.date`.
- **Every test prints a sanity conclusion.**
- **Regression**: full pytest vs the P0 baseline. The EC subset baseline is 66 passed, 1 failed; the failure is the wiki test, which P6 removes.

## 8. Risks, rollback, replan triggers

**Risks:**

- HF throttling: back off on 429; `read_workers` is tunable.
- Format drift: the extractor asserts the schema.
- Single source: the validator coverage gate.
- Cost: the P5 gate.
- Airflow running `dev` during cutover: the EC tasks are paused by the user before P5 and unpaused after merge.

**Rollback:**

- Code is on the branch only.
- Data: drop the new table; rename the `*_legacy` tables back; restore the `cube_part_text` snapshot from `_cache`.

**Replan triggers:**

- `ok` < 98.5 % on all calls;
- full build > 30 min;
- FinBERT > 6 h or OpenAI > $15 projected;
- feature-contract test failing;
- fingerprint change.

## 9. Phases

### P0 ✅ Preflight and baseline

**Owner**: sub-agent.

**Steps**:

1. `rtk git worktree add reports/worktrees/earnings-call-extraction-gaps -b harness/earnings-call-extraction-gaps 71a9ecd`.
2. In the worktree, run the full pytest with `-q`; save to `_out/p0-pytest-full.txt`.
3. Save DB counts and sizes of the 3 EC tables and `cube_part_text`, plus the monthly `f_ec_*` non-null counts, to `_out/p0-db.json` (read-only SELECTs).

**Done when**: the files exist.

**Rollback**: remove the worktree.

### P1 ✅ Shared speaker-turn split and roles (REQ-004; AC-005, AC-006, AC-007 fixtures)

**Owner**: sub-agent. Runs in parallel with P6.

**Files**:

- `src/utils/earnings_call_split.py`
- `tests/utils/test_speaker_turn_split.py`
- `tests/fixtures/earnings_calls/`: 14 calls from `_cache/defeatbeta_sample.parquet` plus AXON 25Q4 and one follow-up case.

**Changes:**

1. Port `_scripts/split_prototype.py`:
   - `clean_paragraphs` produces turns: drop empty and placeholder paragraphs, strip `A - ` prefixes, merge same-speaker paragraphs.
   - `qa_start` requires at least 150 management words, then either a strong operator/host hand-off, or a new non-operator speaker asking a short question that a speaker already seen answers.
2. Move the role rules from `earnings_call_embeddings.split_turns` steps 2a/2b (`:297-332`) into `label_turns(turns, section, mgmt_names)`. Reuse the module's cleaning and pleasantry rules by moving them alongside.
3. Add the two role fixes:
   - a speaker who already asked a question in the call stays a questioner;
   - "up next, we have X" joins the hand-off name pattern.
4. Add `CallSplit(prepared_remarks, qa, turns, status)` with header-free texts.

**RED**: the fixture tests are written first. They fail, because no module exists yet and the regex split is wrong on PLTR, PNW, AXON, JNJ and PSKY.

**Verify**:

- `rtk "$PY" -m pytest tests/utils/test_speaker_turn_split.py -q`: all exact.
- The sample script re-pointed at the library reaches `ok` ≥ 98.5 % and prepared q05 ≥ 0.15; follow-up mislabel and AXON under-split are reported before and after.

**Done when**: tests are green and nothing imports the library yet.

### P2 ✅ Schema (REQ-001) — risk zone

**Owner**: sub-agent.

**Files**: `src/data_store/schema.py`, `sql/schema.sql`.

**Change**: the new `earnings_call_sections` definition (PK `(ticker, quarter, paragraph)`, index on `as_of`).

**Verify**:

- `rtk git diff sql/schema.sql` touches only that block.
- `rtk "$PY" -m pytest tests/data_store -q` shows no new failures.

No live DB change in this phase.

### P3 ✅ Fast extractor (REQ-002, REQ-003; AC-003, AC-004)

**Owner**: sub-agent.

**Files**:

- `fetch_earnings_call_transcripts.py`
- `configs/data.yml`
- `cli.py`: `extract-earnings-calls [-F] [-t]`
- `step_extract_behavioral.py`
- `dag_data_extraction.py` and `test_extraction_dag.py`: one task
- the new test file

**Algorithm**:

1. Resolve the dataset revision SHA and the transcripts-file hash (`spec.json`). If they are unchanged since the last `record_run` and the run is not `-F`, it is a no-op. Bounded discovery: how `run_manifest` stores the extra field.
2. Open via `HfFileSystem` pinned to the SHA, with corporate TLS via `src/utils/ssl_setup.py`; read the footer.
3. Select row groups whose symbol range intersects the roster (dataset form uses `.`).
   - Incremental: also require `max(report_date)` ≥ the stored `max(as_of)` − `lookback_days`.
   - Full re-check every `reconcile_days` or with `-F`.
4. Read the index columns with `read_workers` threads; diff against DB `(ticker, quarter, transcript_id)` to find new and re-issued calls.
5. Read the `transcripts` column only for the row groups that hold those calls; explode to paragraphs; map to repo tickers.
6. Write in a single thread, per batch:
   - for re-issued calls, delete the paragraphs, run `invalidate_earnings_call_derivatives`, then write a pending marker;
   - upsert;
   - `record_run` with the revision.
7. Assert the source schema.

**Tests** (tmp parquet, known truth):

- pruning;
- diff;
- re-issue (AC-004);
- no-op;
- ticker mapping;
- a live smoke test on 2 tickers, skipped when offline.

**Verify**: `rtk "$PY" -m pytest tests/data_extract/behavioral/test_earnings_call_transcripts.py tests/dags/test_extraction_dag.py -q`.

### P4 ✅ Aggregation and validator rewiring (REQ-005, REQ-006, REQ-008; AC-007, AC-008 tests)

**Owner**: sub-agent.

**Files**:

- `earnings_call_features.py`:
  - `_SECTION_COLS`, `_yield_sections_to_score`, the `score_earnings_calls` keys and the `sentiment_kpis_streamed` text reload all read paragraphs per ticker, then `CallSplit` texts.
  - The quality gate is applied unchanged.
- `earnings_call_embeddings.py`:
  - `_section_calls`, `_calls_from_frame`, `_yield_call_texts` and `embed_earnings_calls` take turns from `CallSplit`.
  - Delete the header regexes (`_SPEAKER_COLON`, `_SPEAKER_DASH`, `_SPEAKER_DASH_COLON`, `_NAME_ONLY`, `_INLINE_HDR`) and the parsing in `split_turns`.
- `constants.py`: cache version bump.
- `src/validate/checks/earnings_calls.py` and `configs/validate.yml`.
- Tests:
  - features, embeddings and limits fixtures move to paragraph rows;
  - add a PIT test: a call is not visible on its `as_of` session and is visible on the next session.

**Untouched**: KPI math, `_daily_frame`, `step_cube_text`.

**Verify**:

- `rtk "$PY" -m pytest tests/data_aggregate -k earnings_call -q`
- `rtk "$PY" -m pytest tests/validate/test_earnings_calls.py -q`
- `rtk grep -rn "_SPEAKER_COLON\|_INLINE_HDR" src` returns nothing.

### P4b ✅ Text cleaning (C-1, C-2) — added 2026-10-02

**Owner**: sub-agent. **Why**: the noise research (`01-research.md`, follow-up of 2026-10-02) showed the following in the model inputs:

- broken decimals;
- analyst and operator text in the Q&A sentiment string (30 %);
- boilerplate (5 %) and roster text in prepared remarks;
- one-word turns.

**Files**:

- `src/utils/earnings_call_split.py`
- `src/utils/text_metrics.py` (`clean_earnings_call_text` splitter)
- `src/constants/constants.py` (cache versions)
- `earnings_call_embeddings.py` (cache-version key only)
- the tests

**Changes**: C-1 and C-2. `CallSplit.prepared_remarks` and `CallSplit.qa` become the cleaned sentiment texts. The sentiment cache version is bumped, and the embeddings get a cache-version key so the 72 probe rows are recomputed.

**Verify**:

- the split tests, plus new known-truth tests for decimals, boilerplate, names, management-only Q&A and the 10-word minimum;
- `tests/data_aggregate -k earnings_call`;
- `tests/validate/test_earnings_calls.py`;
- re-run the 2,940-call noise measurement after cleaning: the residual share of each class, gate failures, and decimals intact.

### P5 ✅ Live cutover and rebuild (REQ-007; AC-001, AC-002, AC-003, AC-006, AC-008) — gated data mutation

**Precondition**: the user has paused the EC tasks of the extraction and aggregation DAGs.

**Owner**: sub-agent, with stops at gates.

**Steps**:

1. Snapshot `cube_part_text` to `_cache`.
2. Rename the 3 legacy tables to `*_legacy`.
3. Time `rtk "$PY" -m src data_extract extract-earnings-calls -c ./configs -F`.
4. Re-run: expect a no-op.
5. Run the validator; check parity with the index at the same revision.
6. **Cost probe**: score and embed 200 random calls and extrapolate. **STOP** if the projection exceeds 6 h of FinBERT or $15 of OpenAI.
7. `rtk "$PY" -m src data_aggregate build-text -c ./configs -F`.
8. Check:
   - per-date `f_ec_*` coverage vs the snapshot;
   - the PIT script;
   - the validator split table;
   - fingerprint, gated.
9. Time an incremental extract on the next dataset revision.

**Rollback**: rename the legacy tables back and restore the snapshot.

### P6 ✅ Wikipedia pageviews removal (REQ-010)

**Owner**: sub-agent. Runs in parallel with P1.

**Files**:

- delete `fetch_wiki_pageviews.py`, `test_wiki_incremental.py` and `test_wiki_resolution.py`;
- edit `step_extract_behavioral.py` and `step_extract_all_data.py`;
- edit the CLI and DAG if they reference it;
- edit `polite_http` or test host entries if they are wiki-specific;
- edit the `docker-compose.yml` mention;
- edit `schema.py` only if a trace exists.

**Protected paths**: §6.

**Verify**:

- `rtk grep -rn -i "pageview\|wiki_pageviews\|fetch_wiki" src tests configs` returns nothing.
- `rtk "$PY" -m pytest tests/data_extract tests/dags -q` shows no new failures, and the wiki failure is gone.

### P7 ✅ ROIC, Fool and kurry retirement (REQ-009; AC-009)

**Owner**: sub-agent.

**Steps**:

1. Delete the modules and tests (§6).
2. Delete the constants.
3. Clean the `polite_http`, `docker-compose` and `context.py` comments.
4. Run `rtk grep -rn -i "roic_\|fool\|kurry\|hf_transcripts\|transcript_index\|missing_quarters\|split_prepared_qa\|ROIC_API_KEY" src tests configs`: no product hits (the ROIC *ratio* is excluded).
5. Run the full pytest vs P0.
6. **After explicit user confirmation**: drop the `*_legacy` tables and delete the legacy `data/call_transcripts` files (local and Airflow volume).

### P8 ✅ Docs and memory (AC-010)

**Owner**: sub-agent, using OpenKnowledge MCP.

**Pages**: `data-sources`, `table-catalog`, `live-database`, `modules/data-extract`, `modules/data-aggregate`, `run-the-pipeline`, `configuration`, `data-platform`, `data-access`.

The orchestrator updates the memory notes.

**Verify**: OK lint shows no broken links.

### P9 ✅ Harness Validate (CONDITIONAL after fixes; merged to `dev` as `35be44e`, 2026-10-03)

**Owner**: `$harness-validate`, through sub-agents.

**Produces**:

- domain checks;
- correctness review;
- Ponytail review;
- `06-final.md`;
- `report.html`.

## 10. Completion checklist

- [ ] AC-001 … AC-010 have current evidence.
- [ ] Risk-zone edits stay within §6.
- [ ] Drops and deletions were confirmed.
- [ ] Full pytest shows no new failures vs P0.
- [ ] The Airflow EC tasks were unpaused after merge (user).
- [ ] Every sub-agent was stopped after hand-back.
- [ ] `00-run.md` is current.

## 11. Plan review

Self-review:

- every AC maps to a check;
- data mutation happens only in P5 and P7, after the code phases;
- destructive steps sit behind confirmation;
- the cost gate precedes the rebuild;
- the blast radius on aggregation is about 80 lines, with KPI math and derived schemas untouched (research follow-up 2026-10-02).

An independent plan review has not run yet; it can run as a sub-agent before P1 if wanted.
