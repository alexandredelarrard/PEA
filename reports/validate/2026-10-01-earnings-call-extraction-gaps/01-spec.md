# Spec v3 — defeatbeta raw transcripts in `earnings_call_sections`, split in aggregation

**Status**: DRAFT v3 — all user decisions closed 2026-10-02; awaiting approval.
**Research**: 01-research.md (§1–7, defeatbeta follow-up, option-B contract follow-up) · **Base**: `dev` @ `71a9ecd`.

## User decisions

| ID | Decision |
|---|---|
| U1 | The sole transcript source is HF `defeatbeta/yahoo-finance-data`, `data/US/stock_earning_call_transcripts.parquet`. It replaces ROIC AI, Motley Fool and kurry. |
| U2 | Personal, non-commercial research, so the licence note is accepted. |
| U3 | Extraction stores the **raw transcript only**. Aggregation reads it and rebuilds the texts and turns that sentiment and embeddings use today. |
| U4 | The section split must be improved: the split defect drove the false "missing quarters". |
| U5 | Extraction must be as fast as possible; refactor the EC extract steps. |
| U6 | Retire all ROIC, Fool and kurry code after the new path is validated. |
| U7 | **Same table count.** Drop the old `earnings_call_sections` and re-create `earnings_call_sections` with the raw-transcript schema. |
| U8 | **One-day lag.** A call is usable one day after the real call day, with no extra lag. The aggregation rule (`_daily_frame`, `searchsorted(as_of, side="right")`) already starts at the first session after `as_of`, so `as_of` stores the real call date and no further day is added. |
| U9 | **Wikipedia pageviews.** Delete the fetcher, its table and every dependency. The table is already absent from the registry and the DB; the fetcher and its test are broken. The S&P 500 roster scraper, which also reads Wikipedia, stays. |
| U10 | **Features.** The 12 `f_ec_*` features are unchanged. New EC features (question-only distance, prepared-vs-Q&A embedding distance, prepared tone) are **deferred**. Out of scope: new features, model retraining, feature testing beyond contract parity. |

## Outcome

EC extraction runs fast and incremental, from defeatbeta into the re-schemed `earnings_call_sections`. The current aggregation flow (FinBERT sentiment, text metrics, embeddings, the 12 `f_ec_*` features and `cube_part_text`) runs unchanged in contract on top of it, fed by a better speaker-turn split.

## Evidence (measured; see research)

- **Coverage**: 496/500 roster tickers; 33,676 calls from 2005-10 to 2026-09-30; 481–489 tickers per calendar quarter in 2024–2026; about one daily build.
- **Call date**: `report_date` is the call date, date only (equal to the ROIC call date for 1,477/1,487 calls).
- **Read cost**: the file is symbol-sorted; 551 of 1,195 row groups touch the roster; footer 1.5–3 s; whole row groups about 0.86 s each.
- **Split**: the speaker-turn prototype is valid on 99.1 % of 585 calls, versus 97.4 % for the regex split.
- **Turns**: direct roles give 99.1 % of calls with questions and answers, and 0 question turns by prepared speakers. They match the legacy turn counts (median ratio 1.00) and fix legacy in 8 of the 10 worst disagreements.
- **Role defects to fix**:
  - follow-up questions labelled as answers: 9.1 % of answer turns;
  - "Up next, we have X" hand-offs not recognised: 40 of 585 calls, all AXON calls among them.
- **Unchanged under this design**:
  - all KPI math;
  - the `earnings_call_sentiment`, `earning_calls_embedding` and `cube_part_text` schemas;
  - `_daily_frame`.
- **Key collision**: 23 of 526 overlapping calls hold a different call under the same `(ticker, quarter)` label in legacy, so derivatives must be fully rebuilt.

## Requirements

- **REQ-001 — `earnings_call_sections` re-schemed.**
  - One row per source paragraph: PK `(ticker, quarter, paragraph)`.
  - Columns: `as_of DATE` (the real call date = `report_date`), `transcript_id INT`, `speaker TEXT`, `content TEXT`.
  - `quarter` is the source fiscal label; `ticker` is in repo form (`BRK.B` becomes `BRK-B`).
  - Raw as published: no cleaning or splitting at extract time.
- **REQ-002 — Fast incremental extract.**
  - An unchanged dataset revision is a no-op.
  - Otherwise read only roster row groups, pruned on incremental runs by `report_date` statistics, with a periodic or `-F` full reconcile.
  - Reads are pinned to one revision; writes are single-threaded and resume-safe per batch.
- **REQ-003 — Re-issue.** A changed `transcript_id` replaces the call's paragraphs and invalidates its sentiment and embedding rows.
- **REQ-004 — Shared split.** A pure split in `src/utils/earnings_call_split.py` maps paragraphs to cleaned turns, the Q&A start, role-labelled turns and header-free `prepared_remarks`/`qa` texts.
  - Role labels: question / answer / prepared, with `person`, `exchange_idx` and `answer_idx` as today.
  - Status: `ok` / `no_qa` / `no_prepared` / `empty`.
  - It never depends on an `Operator` line opening the call, on intro previews, or on regex speaker headers.
  - It includes the two role fixes.
- **REQ-005 — Aggregation parity.** Sentiment, text metrics and embeddings read the new table through the split and produce the same row contracts as today.
  - Embeddings get turns `{section, tag, person, text, exchange_idx, answer_idx}`.
  - Sentiment gets the two section texts plus `as_of`, passing `assess_earnings_call_sections`.
  - Non-`ok` statuses are deterministic, never retried, and follow today's quality-gate behaviour.
- **REQ-006 — PIT.** `as_of` = the real call date. A call is invisible on its own date and visible from the next trading session, for 66 sessions. This is today's rule, now with a correct date (Fool URL dates are gone).
- **REQ-007 — Rebuild.** Recompute all sentiment and embeddings (cache version bumped) and `cube_part_text` in full from the new source. Legacy tables are renamed `*_legacy` until validation passes, then dropped after confirmation.
- **REQ-008 — Validator.** `src/validate/checks/earnings_calls.py` reads the new schema and reports:
  - per-calendar-quarter roster coverage and its chart;
  - split-status distribution;
  - section-share quantiles;
  - `as_of` versus `earnings_surprises`.
- **REQ-009 — Retirement.** Remove all ROIC, Fool, kurry, gap-logic, `transcript_index` and Fool-crawler code, tests, constants, caches, DAG tasks and docs.
- **REQ-010 — Wikipedia pageviews removal.** Remove `fetch_wiki_pageviews.py`, its tests (`test_wiki_incremental.py`, `test_wiki_resolution.py`), and its calls from the Step, `step_extract_all_data`, CLI and DAG. Also remove any registry, DDL, config and doc trace. Keep the roster scraper.

## Acceptance criteria

| ID | Criterion | Evidence |
|---|---|---|
| AC-001 | Roster calls in the table equal the roster calls in the pinned revision (per-ticker counts). Each calendar quarter 2024Q1–2026Q3 has ≥ 485 tickers. Per-year ticker counts for 2005–2026 equal the dataset index. | validator CSV and chart |
| AC-002 | `as_of` = `report_date` for 100 % of calls; ≥ 97 % of `earnings_surprises` events since 2026-06-01 have a call within ±3 days. | validator |
| AC-003 | Same-revision re-run writes 0 rows in ≤ 15 s; full build ≤ 15 min; incremental ≤ 3 min. | timed runs in 03-implementation |
| AC-004 | A re-issued `transcript_id` replaces the paragraphs (no orphans) and deletes the call's sentiment and embedding rows. | unit test |
| AC-005 | 14 hand-labelled fixtures give the exact Q&A start and status. AXON 25Q4 has no more than 5 answers per question; a known analyst follow-up is labelled `question`. | `tests/utils/test_speaker_turn_split.py` |
| AC-006 | Over all roster calls since 2024: `ok` ≥ 98.5 %; non-`ok` calls have a status, never an exception; prepared share q05 ≥ 0.15; no `ok` call has prepared + qa < 60 % of turn words. | validator |
| AC-007 | Embedding turns come from the split with no header regexes (grep). On fixtures, 100 % of analyst turns are `question` and no management turn is `question`. | pytest and grep |
| AC-008 | After rebuild:<br>• the 12-feature contract test passes;<br>• a PIT test shows a call is not visible on its `as_of` session and is visible on the next session;<br>• per-date `f_ec_*` non-null coverage over 2024–2026 is ≥ legacy. | pytest and validator |
| AC-009 | No product or test reference to the retired sources or to `wiki_pageviews` remains. The roster Wikipedia scraper is untouched. The full pytest suite has no new failures against the P0 baseline (the wiki failure disappears). | grep and pytest |
| AC-010 | The wiki pages describe the new path only. | OK lint and page list |

## Data invariants

- **Grain**: `(ticker, quarter, paragraph)`, one call per `(ticker, quarter)`.
- **PIT**: `as_of` = call date; visibility starts at the next session.
- **Determinism**: derived rows are a function of `(transcript_id, split version)`.
- **Idempotency and resume**: an unchanged source writes nothing; a crash loses at most one batch.

## Risks and permissions

**Risk-zone edits, approved through the plan:**

- `schema.py` and `sql/schema.sql` (hand-spliced);
- `constants.py`;
- `configs/`;
- the `context.py` comment;
- `data/` caches;
- live DB: rename, create and drop;
- the fingerprint baseline, if it is affected.

**Operational risk:** the nightly Airflow DAGs run the `dev` code against the same DB. During the live cutover (P5) the EC tasks of the extraction and aggregation DAGs must be paused until the branch is merged. Otherwise `dev` code would hit the new schema. User action required.

**Other risks:**

- FinBERT and OpenAI rebuild cost (gated);
- single-source dependency;
- models trained on the old features (retraining deferred).

## Deferred

- New EC features: question-only distance, prepared-vs-Q&A embedding distance, prepared tone.
- Model retraining.
- Historical-membership universe.
- `stock_officers` role oracle. Trigger: `ok` < 98.5 %.
