# Final — earnings-call extraction gaps (P9 Harness Validate)

**Verdict**: **CONDITIONAL**

The second pass re-validated the fix commit `8161e52` against the re-baselined criteria R-1 … R-4 (`02-plan.md` §4). The three blocking findings are fixed and were verified independently:

- **F-001**: a reconcile now deletes calls that leave the source, and the label a relabel flip superseded.
- **F-016**: executives tagged as questioners fell from 61 to 3 of the 5,241 recent ok calls.
- **F-002**: a re-issued valid call now keeps its pending marker until it is scored.

Under the re-baselined rules, AC-001, AC-003 and AC-006 pass, and no acceptance criterion fails. The extraction itself is unchanged: 33,591 calls, parity exact, clean grain, PIT rule intact. The verdict is CONDITIONAL rather than STRONG for three reasons:

1. **AC-008 cannot be judged yet.** It waits for the user's FinBERT, embedding and cube rebuild under the new `speaker-clean-v2` tags.
2. **AC-001 passes only on the scope R-1 names.** That scope is 487 tickers. The extractor's own scope is 490, because ED, EXPD and NVR are not in `NO_EARNINGS_CALL_TICKERS`, and on 490 tickers 2026Q1 sits at 96.94 %.
3. **The F-016 fix has a measured side effect.** 8 recent ok calls (NFLX 6, DIS 1, MCK 1) now have 0 Q&A exchanges, so they carry no per-turn embedding KPI. F-005 also stays open.

Findings: 1 high, 6 medium, 10 low/info. F-001, F-002 and F-016 are FIXED, and no high finding is open. Simplification: 7 material items, not implemented.

## Scope and identities

| Field | Value |
|---|---|
| Repository | PEA (`stock_pick_strat`) |
| Base | `dev` @ `71a9ecd` |
| Candidate | `harness/earnings-call-extraction-gaps` @ `8161e52` (P9 fix commit on top of `56dea8b`; worktree `reports/worktrees/earnings-call-extraction-gaps`, clean, verified `_out/p9fix2-head.txt`) |
| Commits | `63edc8b` P6, `ede92af` P1, `0b46784` P2, `ad4bab9` + `d887eec` P3, `3ed3585` P4, `166a9f2` P4b, `56dea8b` P7, `8161e52` P9 fixes |
| Wiki (P8) | 12 files modified, uncommitted, in the main tree under `wiki/` |
| Current `dev` | `091278b`; `git merge-tree --write-tree dev 8161e52` → clean, exit 0 (`_out/p9fix2-merge-tree.txt`) |
| Validated | first pass 2026-10-02/03 on `56dea8b`; second pass 2026-10-03 on `8161e52`, fresh read-only context. No code edit, commit or DB write. Every new script and output sits in this run dir. |

## Stage timeline and approvals

| Stage | Report | Notes |
|---|---|---|
| Run | [00-run](00-run.md) | decisions U1–U10 |
| Research | [01-research](01-research.md) | defeatbeta chosen as sole source |
| Spec v3 | [01-spec](01-spec.md) | AC-001 … AC-010 |
| Plan v2 + P4b | [02-plan](02-plan.md) | approved 2026-10-02; C-1…C-4 cleaning decisions; R-1…R-4 re-baselines after the first P9 pass |
| Implementation | [03-implementation](03-implementation.md) | P0–P8 ✅; P5 live cutover done; P9 fix commit `8161e52` |
| Validation | [04-validation-01](04-validation-01.md) | first pass; this file carries the second pass |
| Register | [defects](defects.md) | 17 findings + 7 simplifications (dispositions as of the first pass; current ones are in this file) |
| Review | [05-review-01](05-review-01.md) | correctness + Ponytail |
| Fix plan | [05-simplification-plan](05-simplification-plan.md) | proposed, not implemented |

## Acceptance-criterion matrix

| AC | Implementation evidence | P9 validation evidence | Status | Limitation |
|---|---|---|---|---|
| AC-001 | P5 parity exact (`_out/p5-parity.txt`) | **R-1** (≥ 97 % of in-scope tickers per calendar quarter, listed by the quarter's first day): since 2024Q1 the minimum is **97.54 %** (2026Q1, 475/487) and the maximum 99.38 % (`_out/p9fix2-ac001.txt`) | **PASS** | Holds on the 487-ticker scope that R-1 names. On the extractor's 490-ticker scope (adds ED, EXPD, NVR, which hold no call), 2026Q1 = **96.94 %**, a fail. 41 quarters before 2024 are below 97 %, but R-1 does not bind history. |
| AC-002 | P5 parity 0 `as_of` mismatches | 507/510 = **99.41 %** of events since 2026-06-01 within ±3 days | **PASS** | 3 source gaps (F-013) |
| AC-003 | timed runs `_out/p5-extract-*.log` | **R-2** accepts: full build 18.6 min (1,119 s, DB-write bound), no-op 1.3–3.9 s | **PASS (accepted)** | The first real incremental is timed from the nightly run. Since F-001, a reconcile also reads paragraph 1 of every stored call in scope (33.6 k rows) and deletes any stale call; this is not timed. |
| AC-004 | re-issue test + 3 new F-001 tests | All pass in `tests/data_extract/behavioral` (`_out/p9fix2-pytest-split-validate-extract-dags.txt`); `_scripts/p9fix_orphan_repro.py` re-run gives orphan on relabel flip: **NO**, removed call deleted on reconcile: **YES** | **PASS** | An incremental run never deletes, so a relabel seen first by an incremental can leave two calls on one date for up to `reconcile_days` (7). The new validator grain key `ticker_dates_with_multiple_calls` flags it at score 8. |
| AC-005 | 19 fixtures (3 new: TXN 2025Q1, BMY 2026Q1, JPM 2024Q3) | The fixture tests pass. Independent re-split of the 3 new fixtures: management tagged question is Ilan / Massacesi / Dimon under `56dea8b`, **none** under `8161e52`; no analyst is tagged answer | **PASS** | Fixture level. TXN records a known residual: the CFO, Lizardi, answers in a single exchange and stays tagged question. |
| AC-006 | P5 validator (all history) | Since 2024: ok **98.87 %**, 0 exceptions (33,591 calls in all history: 0 exceptions). **R-3** (management words kept in prepared + answers): q01 **0.906**, q05 0.932, median 0.963; **0** of 5,241 ok calls below 0.60 (all history: 3 of 33,132) (`_out/p9fix2-summary.txt`) | **PASS** | The management set is self-referential: persons that either split version tags prepared or answer. As a strict lower bound, the all-words ratio leaves 39 recent ok calls (0.74 %, down from 85) below 0.60; these are analyst-heavy calls that C-1 excludes by design. |
| AC-007 | grep + fixtures | Header-regex grep shows 0 hits. Live: **3** of 5,241 recent ok calls have a ≥ 1,000-word person tagged question (was 61): VLO and T analysts, and CRM 2027Q1 Patrick Stokes (`_out/p9fix2-summary.txt`, which agrees with `_out/p9fix-question-role-scan.txt`) | **PASS** | Residual: an executive speaking in fewer than 3 exchanges keeps the old rules (for example NFLX 2025Q3 CFO Neumann, 586 words). |
| AC-008 | contract + PIT tests; 3-ticker cube | aggregate subset **30 passed**, including the new F-002 test; live PIT on ABNB/CEG/PLTR holds; calls since 2024 5,298 new vs 5,056 legacy | **DEFERRED** | Per-date coverage needs the user's full rebuild under `speaker-clean-v2` (F-006) |
| AC-009 | P6/P7 grep; P7 full pytest no new failure | `--collect-only` **2,069** collected, exit 0. Subsets: the only failure is P0 baseline `test_dag_chain_is_derived_from_the_registry` | **PASS** | F-014 |
| AC-010 | 12 wiki files (OK MCP) | First-pass OK audit: no problem in the changed pages | **PASS** | F-017 (one stale line); wiki uncommitted; not re-audited for `8161e52` (no wiki change in the fix) |

## Lane summaries

- **Code.**
  - The fix diff touches 12 files (+715/−26) and stays in scope: extractor removal, `_qa_management` role evidence, acknowledgement of scorable markers, and a validator grain key.
  - It edits `src/constants/constants.py`, an ask-first file, to bump both cache tags to `speaker-clean-v2` (R-4).
  - Collection is clean. The EC subsets pass: 108 + 30 passed, with 1 P0 baseline failure.
  - The merge onto current `dev` is clean.
- **Data.**
  - Grain, parity and PIT are unchanged from the first pass (no extractor run since).
  - Re-split of all 33,591 stored calls under both versions: 0 exceptions. The status mix is identical (ok 33,132), and labelled turns change on 198 of 5,241 recent ok calls.
- **Modelling.**
  - The 12-feature contract and the PIT tests pass.
  - The split change invalidates every cached FinBERT score and embedding (all `speaker-clean-v1` today: 7,475 embedded calls, 78 scored calls). The rebuild is the user's.
- **Operations.**
  - The three `*_legacy` tables are gone (0 rows in `pg_class`).
  - `data/call_transcripts/` is **still present** (115 entries; size not re-measured, P7 measured 2.07 GB).
  - Airflow EC tasks stay paused until merge.

## Validation attempts

First pass (`56dea8b`): see `_out/p9-*.txt`; unchanged rows are not repeated. Second pass (`8161e52`):

| Command / query | Exit | Decisive observation | Artifact |
|---|---|---|---|
| `pytest --collect-only -q tests` | 0 | 2,069 collected | `_out/p9fix2-collect.txt` |
| `pytest tests/utils/test_speaker_turn_split.py tests/validate/test_earnings_calls.py tests/data_extract/behavioral tests/dags -q` | 1 | 1 failed (P0 baseline `test_dag_chain_is_derived_from_the_registry`), 108 passed | `_out/p9fix2-pytest-split-validate-extract-dags.txt` |
| `pytest tests/data_aggregate -k "earnings_call or cube_text" -q` | 0 | 30 passed | `_out/p9fix2-pytest-aggregate-ec.txt` |
| `_scripts/p9fix_orphan_repro.py` (re-run) | 0 | orphan on relabel flip NO; removed source calls deleted on reconcile YES | `_out/p9fix-orphan-repro.txt` |
| independent re-split of the TXN/BMY/JPM fixtures, old vs new module | 0 | executives tagged question: 1/1/1 → 0/0/0; analysts tagged answer: 0 | inline, this file (AC-005) |
| `_scripts/p9fix2_split_compare.py` × 8 shards (old module exported from `56dea8b` as `_scripts/p9fix2_split_old.py`) + `_scripts/p9fix2_summary.py` | 0 | F-016 scan 61 → 3; R-3 0 below 0.60; 0-exchange ok calls 8 → 16 since 2024 | `_out/p9fix2-split-compare-*.csv`, `_out/p9fix2-summary.txt` |
| `_scripts/p9fix2_ac001.py` | 0 | R-1 min 97.54 % on 487 tickers; 96.94 % on 490 | `_out/p9fix2-ac001.txt` |
| psql: `*legacy*` relations, cache tags per model | 0 | 0 legacy tables; embedding 7,475 `speaker-clean-v1` + 64 untagged; sentiment 78 `speaker-clean-v1` + 67 `speaker-v1` | inline |
| `git merge-tree --write-tree dev 8161e52` | 0 | clean | `_out/p9fix2-merge-tree.txt` |

Abstentions:

- **Generic nine-check library**: not run, for the same reasons as the first pass.
- **Full suite**: not re-run (user constraint).
- **FinBERT, embeddings, `build-text` and any extractor run**: not run (user constraint).

## Findings and dispositions

| ID | Score | Title | Disposition |
|---|---|---|---|
| F-001 | 7 high | Stored calls leaving the source, or a relabel flip, are never deleted | **FIXED** in `8161e52`. `stale_calls` + `_remove_calls` on reconcile runs only: derivatives are invalidated, then a pending marker is written, then the paragraphs are deleted. A ticker with no source call left is warned about, not deleted. The validator gains `ticker_dates_with_multiple_calls` (score 8). Verified by the repro re-run and 3 new extractor tests plus 1 validator test. |
| F-016 | 6 medium | Executives tagged `question` on live calls | **FIXED** in `8161e52`. `_qa_management` makes a Q&A speaker management when they speak in ≥ 3 exchanges, are never named by a hand-off, carry no analyst label and are not a host; management now outranks the sticky-asker rule. Live: 61 → 3 recent calls (all history 413 → 63). The 3 TXN/BMY/JPM fixtures were added. Side effect: see the residual risks. |
| F-002 | 5 medium | Re-issued valid call permanently skipped if FinBERT is unavailable | **FIXED** in `8161e52`. `acknowledge_earnings_call_invalidations` leaves the markers of still-scorable calls pending. The new test `test_reissued_valid_call_marker_survives_an_engine_less_build` checks engine None → 0 acknowledged, then engine back → both sections scored and a KPI row. |
| F-003 | 5 medium | AC-001 floor vs the scope ceiling | **RESOLVED by R-1**; residual 490-vs-487 scope gap (see AC-001) |
| F-004 | 5 medium | Full build 18.6 min; incremental untimed | **ACCEPTED by R-2** |
| F-005 | 5 medium | Validator does not measure the AC populations | OPEN (not in the fix scope) |
| F-006 | 4 medium | AC-008 and drift unmeasurable until the full rebuild | ACCEPTED-DEFERRED |
| F-007 … F-015, F-017 | 1–3 | Retries of non-`ok` calls; marker ordering; label jumps; paragraph 1 reread; undeclared dependency; zero-turn refresh; 3 source gaps; stale docstrings; `inf` paragraph; stale wiki line | OPEN / INFO |

## Reviews

- **First pass.** The correctness and Ponytail passes are unchanged; see [05-review-01](05-review-01.md).
- **Second-pass review of the fix diff.** No hard-rule violation was found. Points checked:
  - Removal runs only when `is_full`, and `since` is then `None`, so a reconcile always compares against the whole source of its tickers.
  - The crash order (derivatives → marker → paragraphs) leaves the call stored, so the next reconcile repeats the removal.
  - `_scorable_keys` reuses `_valid_calls` and `cleaned_sections`, so it marks exactly the set `score_earnings_calls` would score.
  - Note: `_scorable_keys` loads every paragraph of every ticker with a pending marker. This is cheap today, but it would grow after a large reconcile removal.
- **Ponytail.** S-001 … S-007 remain proposed and not implemented.

## Residual risks and assumptions

- **IR-read side effect of the F-016 fix (measured).** Calls where a host reads the analysts' questions now produce 0 Q&A exchanges:
  - The host is a prepared speaker, so management, and the executives who answer are now management too, so nothing is tagged question.
  - Before the fix, these calls had exchanges only because the executives were mis-tagged as questioners.
  - Ok calls with 0 exchanges went from **8 → 16** since 2024 and **169 → 191** over all history.
  - The calls that flipped are **8** since 2024 (NFLX 2023Q4, 2024Q1, 2024Q2, 2024Q4, 2025Q1 (1 exchange left), 2025Q2; DIS 2026Q2; MCK 2026Q3) and **25** over all history. MCK 2026Q3 is a transcript with shifted speaker labels.
  - Effect: these calls carry no answer turns, so no per-turn embedding KPI. Their FinBERT `qa` text holds the executives' answers plus the host-read question text.
  - The host-read question text in `qa` was already present on IR-read calls before the fix (for example NFLX 2025Q3 splits identically under both versions).
  - This side effect is not registered in [defects](defects.md).
- **Residual F-016.** An executive who answers in fewer than 3 exchanges keeps the old rules (TXN CFO in the fixture, NFLX 2025Q3 CFO with 586 words).
- **Single source.** defeatbeta only, with the licence accepted for personal research (U2).
- **Live table state.**
  - `cube_part_text` holds 3 tickers.
  - Every cached sentiment score and embedding is under an old tag, so all of them are recomputed under `speaker-clean-v2`.
- **Airflow** runs `dev` against the same DB. The EC tasks stay paused until merge.

## Commit and merge ledger

| Commit | Phase | Content |
|---|---|---|
| `63edc8b` | P6 | Wikipedia pageviews removed |
| `ede92af` | P1 | split library + 16 fixtures |
| `0b46784` | P2 | re-schemed `earnings_call_sections` |
| `ad4bab9`, `d887eec` | P3 | defeatbeta extractor; one call per date |
| `3ed3585` | P4 | aggregation + validator rewiring |
| `166a9f2` | P4b | text cleaning |
| `56dea8b` | P7 | ROIC/Fool/kurry retirement |
| `8161e52` | P9 fixes | F-001 stale-call removal, F-016 executive roles (+3 fixtures), F-002 pending marker, cache tags → `speaker-clean-v2` |

No merges; the validator made no commit.

## PR handoff recommendation and user actions

**Recommendation**: the PR can be opened as CONDITIONAL. Conditions to state in the PR:

- AC-008 is deferred to the user's rebuild.
- AC-001 holds on the 487-ticker scope. Either add ED/EXPD/NVR to `NO_EARNINGS_CALL_TICKERS`, or accept that on the extractor scope 2026Q1 = 96.94 %.
- The IR-read side effect (8 recent calls) is known.
- F-005 stays open.

**User actions**, in order:

1. **Confirm the `constants.py` edit.** The cache tags are now `speaker-clean-v2` (R-4); `constants.py` is an ask-first file.
2. **FinBERT**: run per `_out/p5-finbert-command.txt` (`FINBERT_DEVICE=cuda rtk "$PY" "$RUN/_scripts/p5_score_only.py" run` from the worktree). The script reads the tag from constants, so it scores under `yiyanghkust/finbert-tone:speaker-clean-v2`. The file's text and verification SQL still say `speaker-clean-v1`: substitute `v2`. Expect about 33,155 calls.
3. **Embeddings**: run `_scripts/p5_embed_only.py` under `text-embedding-3-small:speaker-clean-v2`. This is a full re-embed: the 7,475 `v1` calls are no longer reusable. P5 projected about $6 for the remainder, so expect somewhat more.
4. **Cube**: `rtk "$PY" -m src data_aggregate build-text -c ./configs -F`, then the checklist in `_out/p5-next.txt`, with `v1` read as `v2`. The validator target must be `-T cube_part_text`. Re-judge the 16 drift findings on the full cube; this closes AC-008.
5. **Merge**, then unpause the EC tasks of the Airflow extraction and aggregation DAGs.
6. **`.env`**: remove `ROIC_API_KEY`.
7. **Finish deleting `data/call_transcripts/`.** It is still present with 115 entries (the same bind mount as Airflow). The three `*_legacy` tables are already dropped.

**Actions not taken by P9**: no code edits, commits, merges, pushes, PR, DB writes, deletions, extractor runs, FinBERT, embeddings, cube builds or full suite.

## Report integrity

`_scripts/p9fix2_report_check.py` → `_out/p9fix2-report-check.txt`. This is the first-pass checker with the candidate set to `8161e52` and the HEAD file set to `p9fix2-head.txt`. It checks:

- one verdict, matching this file;
- the base and candidate revisions, with the candidate equal to the worktree HEAD;
- each of the 10 ACs exactly once;
- the high finding F-001 dispositioned;
- relative links, all existing and inside the run dir;
- the stage index;
- no placeholders and no remote resources;
- finding counts matching.

## Report index

- **Stage reports**:
  - [00-run](00-run.md)
  - [01-research](01-research.md)
  - [01-spec](01-spec.md)
  - [02-plan](02-plan.md)
  - [03-implementation](03-implementation.md)
  - [04-validation-01](04-validation-01.md)
  - [defects](defects.md)
  - [05-review-01](05-review-01.md)
  - [05-simplification-plan](05-simplification-plan.md)
  - this file
  - `report.html`
- **P9 second-pass scripts** (`_scripts/`):
  - `p9fix2_split_compare.py`
  - `p9fix2_split_old.py` (the `56dea8b` split module)
  - `p9fix2_summary.py`
  - `p9fix2_ac001.py`
  - `p9fix2_report_check.py`
  - re-run: `p9fix_orphan_repro.py`
- **P9 second-pass outputs** (`_out/`):
  - `p9fix2-head.txt`
  - `p9fix2-collect.txt`
  - `p9fix2-pytest-*.txt`
  - `p9fix2-split-compare-*.csv` / `.log`
  - `p9fix2-summary.txt`
  - `p9fix2-ac001.txt`
  - `p9fix2-merge-tree.txt`
  - `p9fix2-report-check.txt`
- **Fix-stage evidence** (implementer): `_out/p9fix-*.txt`, `_scripts/p9fix_*.py`.
- **First pass**: `_out/p9-*.txt`, `_scripts/p9_*.py`.
- **Reused evidence**:
  - `_out/p5-parity.txt`
  - `_out/p5-validator-3-cube.txt`
  - `_out/earnings_calls.json`
  - `_out/p5-e2e-checks.txt`
  - `_out/p5-extract-*.log`
  - `_out/p7-pytest-full.txt`
  - `_out/p7-pending-deletions.txt`
  - `_out/p5-finbert-command.txt`
  - `_out/p5-next.txt`
