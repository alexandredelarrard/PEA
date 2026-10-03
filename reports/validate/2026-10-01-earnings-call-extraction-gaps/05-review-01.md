# Review 01 — correctness and Ponytail (P9)

**Diff**: `71a9ecd..56dea8b` (8 commits, 68 files, +4,107/−3,814). Two independent read-only sub-agent passes; each claim below was re-checked against the code by the validator before it was accepted. Findings are registered in [defects](defects.md).

## Correctness pass

| Agent item | Accepted as | Verification by the validator |
|---|---|---|
| Stored calls leaving the source never deleted; relabel flip | **F-001 (high, 7)** | Reproduced with the production pure functions: `_scripts/p9_orphan_repro.py` → `_out/p9-orphan-repro.txt` (two calls on one date after run 2). Live count 0 (`_out/p9-db-checks.txt` §2). |
| Re-issued valid call skipped for good when FinBERT is unavailable | **F-002 (medium, 5)** | Read `utils_earnings_call_cache.py:25-48`, `earnings_call_features.py:162-182`, `step_cube_text.py:84-89`: the acknowledgement runs unconditionally after the cube write; `invalid-handled` is in `done`. |
| Validator does not measure the AC populations | **F-005 (medium, 5)** | Confirmed: P9 had to measure AC-002, AC-006 and the AC-001 floor ad hoc. |
| Non-`ok` calls retried every run | F-007 (low, 3) | `sec_keys` includes every stored call. |
| Zero-turn embedding calls refresh the cube every run | F-012 (low, 2) | Pre-existing; not counted. |
| Crash ordering in `_write_batch` | F-008 (low, 3) | Delete precedes marker save (`:389-394`). |
| Calls without paragraph 1 rewritten every run | F-010 (low, 2) | 0 such calls live. |
| `huggingface_hub` undeclared | F-011 (low, 2) | Not in `pyproject.toml` or `airflow/requirements-airflow.txt`; installed through `transformers`. |
| Stale comments | F-014 (info) | `store.py:380`, `incremental.py:11` (ask-first / protected). |
| `inf` paragraph raises | F-015 (info) | Unreachable from BIGINT. |

**No issue** (agent, spot-checked by the validator): KPI math, `prepare_earnings_call_kpis` and `_daily_frame` (`side="right"`) unchanged — the features diff touches no `searchsorted`/`as_of` window line; the sentiment cache bump and the embedding model tag force a full recompute without mixing (the KPI pass filters `model == EARNINGS_CALL_EMBEDDING_CACHE_MODEL`; the 64 untagged probe calls in the live table are therefore ignored); reads run on threads with one handle each, writes on the calling thread; a cold table forces a full run; no SQL outside `data_store`, no literal table names, no `print`, no late imports, no new sibling cross-imports; DAG has one `extract-earnings-calls` task and no task uses the removed `scrape` pool.

**Merge readiness**: `dev` has moved 4 commits past the base (`191fc65` also deletes the pageviews fetcher; `48d80d9` edits `run_manifest.record_run`). `git merge-tree --write-tree dev harness/earnings-call-extraction-gaps` → clean (`_out/p9-merge-tree.txt`). `record_run` still carries `identity_scope_fingerprints` forward on `dev`, which the extractor's no-op depends on.

## Ponytail (simplicity) pass

All new config keys are read (`fetch_earnings_call_transcripts.py:416/424/440`, `earnings_calls.py:345/350`). Material items, each grep-verified:

| ID | Item | Evidence | Saves |
|---|---|---|---|
| S-001 | `Crawler` class orphaned by this diff (its last src consumer `utils_behavior.py` is deleted) | `grep -rn "Crawler\b\|from src.utils.crawler" src` → no consumer outside `crawler.py`; Google Trends uses only `load_proxy_pool` | ~160 src + ~124 test lines |
| S-002 | `PARAGRAPH_FIELDS` unused | `fetch_earnings_call_transcripts.py:77`, definition only | 1 |
| S-003 | module `logger` unused (logs via `context.log`) | `:68` | 2 |
| S-004 | `clean_turn_text(removed=...)` never passed | `earnings_call_split.py:585-595` | ~10 |
| S-005 | second cleaning path: `assess_earnings_call_sections` re-cleans already-cleaned split output | `text_metrics.py:541-558` | ~18 |
| S-006 | constants placement: `EARNINGS_REPORT_TO_QUARTER_LAG_DAYS`, `EARNINGS_CALL_EMBEDDING_CACHE_VERSION`/`_MODEL` have one src consumer each | repo rule (2+ consumers) | move |
| S-007 | `embed_earnings_calls(model=...)` dead and inconsistent: a non-default model is written under one tag and filtered out by the KPI pass under the constant | `earnings_call_embeddings.py:172,193,262` | ~3 |

Minor (optional): duplicate grain counts that cannot fire under the PK/NOT NULL; `_SPLIT_STATUSES` repeats the `SplitStatus` Literal; two near-identical per-ticker loaders; triple sort and a dual paragraph key kept for fixtures; repeated greeting regex fragment and "speed-only" prefilters with no measured speed-up; `label_turns(names=)` unused; `ExtractSummary.revision` only printed; `HISTORY_START` comment wrong. No new dependency was added for simplicity's sake; `_ordered_parallel` and `_Readers` are justified (no back-pressured ordered helper exists in the repo).

## Outcome

The correctness pass leaves one **high** finding (F-001) open, so closure is blocked. The simplicity items are material but not behaviour-changing; they go to [05-simplification-plan](05-simplification-plan.md) for the user to accept.
