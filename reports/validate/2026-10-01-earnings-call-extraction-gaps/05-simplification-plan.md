# Simplification and fix plan (proposed, pending user acceptance)

P9 ran read-only (user constraint: no code edits). These are the minimal patches P9 recommends before the PR. None is implemented. After any of them lands, the affected tests, the validator and both reviews must re-run.

## Correctness (blocking or recommended)

| Order | Finding | Scope | Minimal patch | Fails before / passes after |
|---|---|---|---|---|
| 1 | F-001 (high) | `fetch_earnings_call_transcripts.py` (`_stored_calls`, `diff_calls`, `extract_earnings_calls`), `checks/earnings_calls.py` | Return a third "gone" frame from the diff on full/reconcile runs (stored in-scope keys absent from the source) and delete them with derivative invalidation and a pending marker; add `calls_per_ticker_as_of > 1` to the validator grain block. | `_scripts/p9_orphan_repro.py` prints YES today; a two-snapshot test in `test_earnings_call_transcripts.py` |
| 1b | F-016 (medium) | `earnings_call_split._label`, `_is_operator`, fixtures | Do not label `q` from `boundary` when the preceding turn is a management redirect; seed management from Q&A answerers; let answer evidence beat the sticky-asker rule. | `_scripts/p9_question_role_scan.py` (61 calls ≥ 1,000 words today); new TXN 2025Q1 and BMY 2026Q1 fixtures |
| 2 | F-002 | `earnings_call_features.acknowledge_earnings_call_invalidations` or the `done` set | Acknowledge only markers whose call was scored or split non-`ok`. | unit test: re-issue → engine None run → engine run scores the call |
| 3 | F-008 | `_write_batch` | Save markers before the deletes. | crash-injection unit test |
| 4 | F-005 | `checks/earnings_calls.py`, `configs/validate.yml` (ask-first) | Add since-2026-06 event match rate, since-2024 ok rate/q05, the 60 % turn-word rule, and a finding on the quarter floor. | validator test on fixture rows |
| 5 | F-011 | `pyproject.toml`, `airflow/requirements-airflow.txt` | Declare `huggingface_hub`. | import check in the Airflow image |

## Simplification (material, behaviour-neutral)

| ID | Patch |
|---|---|
| S-001 | Delete `Crawler`/`_mask` and `tests/utils/test_crawler.py`; keep `load_proxy_pool`; then drop `polite_http.get_text`/`get_json` if still unused. |
| S-002, S-003 | Delete `PARAGRAPH_FIELDS`, `logger` and `import logging`. |
| S-004 | Drop `clean_turn_text(removed=)` and the `drop()` helper. |
| S-005 | `assess_earnings_call_sections` counts words on its input; delete `clean_earnings_call_text`; update the two tests. |
| S-006 | Move the three single-consumer constants into their consumers (constants.py is ask-first). |
| S-007 | Drop `embed_earnings_calls(model=)`; use `EARNINGS_CALL_EMBEDDING_CACHE_MODEL`. |

Verification after the patches: `tests/utils/test_speaker_turn_split.py`, `tests/data_aggregate -k "earnings_call or cube_text"`, `tests/validate/test_earnings_calls.py`, `tests/data_extract/behavioral`, `tests/dags`, `tests/utils`, plus `pytest --collect-only`.
