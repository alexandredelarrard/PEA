# Defects register — earnings-call-extraction-gaps (P9)

**Candidate**: `harness/earnings-call-extraction-gaps` @ `56dea8b` (base `71a9ecd`). **Validated**: 2026-10-02/03. Read-only pass: no fix branch was opened (user constraint), so every finding is **OPEN — recommended fix** unless marked otherwise.

Score bands: 9–10 critical, 7–8 high, 4–6 medium, 1–3 info/low.

| ID | Score | Severity | AC / scope | Title | Status |
|---|---|---|---|---|---|
| F-001 | 7 | high | AC-001, AC-004, REQ-003, grain invariant | Stored calls that leave the source (relabel flip, removed call) are never deleted | OPEN |
| F-002 | 5 | medium | REQ-003, REQ-005 | A valid re-issued call is permanently skipped if FinBERT is unavailable on the next `build-text` | OPEN |
| F-003 | 5 | medium | AC-001 | Per-calendar-quarter floor ≥ 485 not met (474–484); scope ceiling is 487 | OPEN — user decision |
| F-004 | 5 | medium | AC-003 | Full build 18.6 min > 15 min; real incremental never timed | OPEN — user decision |
| F-005 | 5 | medium | AC-001, AC-002, AC-006 | The validator does not measure the AC populations (since-2026-06 match rate, since-2024 split, 60 % rule, quarter floor) | OPEN |
| F-016 | 6 | medium | AC-007, REQ-004, AC-006 | Executives who speak only in the Q&A (and CEOs on Q&A-only calls) are tagged `question` on real calls: 61 ok calls since 2024 (1.16 %) carry a ≥ 1,000-word management voice as questioner | OPEN |
| F-006 | 4 | medium | AC-008 | Per-date `f_ec_*` coverage vs legacy and the 16 drift findings are unmeasurable until the full rebuild | ACCEPTED-DEFERRED (user decision) |
| F-007 | 3 | low | REQ-005 | Non-`ok` / gate-invalid calls re-read, re-split and FinBERT re-loaded every run | OPEN (pre-existing pattern) |
| F-008 | 3 | low | REQ-003 | `_write_batch` deletes before writing the refresh marker | OPEN |
| F-009 | 3 | info | features | Provider fiscal-label jump (2025Q3 → 2026Q4) on 8–10 retailers nulls one call's delta/distance features | INFO |
| F-010 | 2 | low | AC-003 | Calls missing paragraph 1 look new on every changed run (0 today) | OPEN |
| F-011 | 2 | low | ops | `huggingface_hub` is used directly but only installed transitively via `transformers` | OPEN |
| F-012 | 2 | low | perf | Zero-turn embedding calls refresh the cube tail every run (pre-existing, uncounted) | OPEN |
| F-013 | 2 | info | source | 3 of 510 `earnings_surprises` events since 2026-06-01 have no call in the source (PGR, AES, TECH) | INFO |
| F-014 | 1 | info | AC-009 | Stale docstrings name deleted code (`store.py:380`, `common/incremental.py:11`, `polite_http.py:21`) | INFO (ask-first paths) |
| F-017 | 2 | info | AC-010 | `wiki/reference/live-database.md:54` describes a `wiki_pageviews` table that is no longer in the DB (`to_regclass` NULL) | OPEN (wiki) |
| F-015 | 1 | info | REQ-004 | `_paragraph_order` raises `OverflowError` on `inf` (unreachable from a BIGINT column) | INFO |
| S-001..S-007 | — | material simplification | Ponytail | See [05-review-01](05-review-01.md) and [05-simplification-plan](05-simplification-plan.md) | OPEN |

## F-001 — orphan stored calls (high, 7)

- **Observed**: `_scripts/p9_orphan_repro.py` → `_out/p9-orphan-repro.txt`. Run 1 stores `2020Q4 @ 2020-03-01` (two labels on one date, no later call → highest ordinal kept). Run 2 adds `2020Q1 @ 2020-06-01`; `one_call_per_date` now keeps `2019Q4 @ 2020-03-01`, which arrives as **new**; `2020Q4` is never compared. Table after run 2 holds **two calls on 2020-03-01**. A call removed from the source is likewise never deleted.
- **Live today**: 0 instances (`_out/p9-db-checks.txt`: `multi_call_per_date = 0`; P5 parity exact at revision `cd41c0ac`). The defect is latent.
- **Expected**: grain invariant "one call per (ticker, quarter)" and "stored table = deduplicated source at the pinned revision" (AC-001 parity) hold after every run, not only after a cold load.
- **Mechanism**: `fetch_earnings_call_transcripts.py:369-382` — `_stored_calls` inner-merges stored rows onto the source calls, so stored keys absent from the source are invisible; `diff_calls` (`:261-276`) has no "gone" output; nothing in `extract_earnings_calls` deletes them, on `-F` or reconcile either. `one_call_per_date` (`:236-258`) can flip its choice once a later call arrives.
- **Impact**: a duplicated call on one date flows into sentiment, embeddings and `_daily_frame`; consecutive-quarter deltas break for that ticker; silent (the validator has no calls-per-(ticker, as_of) check).
- **Fix**: on full / reconcile runs, delete in-scope stored keys missing from `calls` (paragraphs + `invalidate_earnings_call_derivatives` + pending marker); treat a stored key with the same `(ticker, as_of)` under another quarter as a re-key; add `calls_per_ticker_as_of > 1` to the validator grain block with a finding.
- **Verify**: the repro script prints `orphan on relabel flip -> NO`; a two-snapshot unit test in `test_earnings_call_transcripts.py` fails before and passes after.

## F-002 — re-issued valid call skipped for good (medium, 5)

- **Observed (code)**: `_write_batch` (`fetch_earnings_call_transcripts.py:393`) writes `invalid-pending` markers for **every** re-issue (the old code did so only for invalid replacements). If `get_sentiment_engine` returns None (`earnings_call_features.py:179-182`), the step still writes the cube and `acknowledge_earnings_call_invalidations` (`step_cube_text.py:84,89`) turns the marker into `invalid-handled`, which `done` includes (`earnings_call_features.py:166-171`). The marker shares the sentiment PK, so the call never re-enters `todo_keys`.
- **Impact**: one engine-less run after a re-issue permanently drops that call's sentiment; silent. Requires the model to be unavailable (offline HF / missing torch), so the probability is low.
- **Fix**: acknowledge only markers whose call was scored or split non-`ok`; or exclude `invalid-handled` from `done` for calls whose split is `ok`.
- **Verify**: unit test — re-issue, run with engine None, run with engine → the call is scored.

## F-003 — AC-001 quarter floor (medium, 5)

- **Observed**: distinct tickers per calendar quarter of the call date, 2024Q1–2026Q3 = 474, 479, 481, 484, 483, 480, 481, 481, 475, 480, 479 (`_out/p9-db-checks.txt`); validator reporting-quarter metric 464–477 valid / 467–480 with a call (`_out/earnings_calls.json`). Floor 485 is never met. 03-implementation did not report this criterion.
- **Mechanism**: scope = `load_universe_tickers` (490 = 500 roster − 9 `INSUFFICIENT_HISTORY_TICKERS` − BRK-B); ED, EXPD and NVR hold no calls → ceiling **487**. The spec floor was set from the full-roster index (481–489 of 500). Data equals the source exactly (P5 parity), so this is a threshold/scope mismatch, not missing data. The 9 excluded spin-offs (GEHC, GEV, HONA, KVUE, Q, SNDK, SOLV, VLTO, FDXF) had 79 legacy calls since 2024 and have no `prices`, so no cube impact.
- **Fix**: user re-baselines the floor on the in-scope population (e.g. ≥ 97 % of 487) or widens the extract scope to the roster.

## F-004 — AC-003 timing (medium, 5)

- **Observed**: full `-F` 1,119 s wall (1,091 s DB writes) > 900 s; no-op 1.3–3.9 s in-process (CLI wall 23.6–53.5 s with interpreter start under load); incremental timed only as a no-op (no new revision appeared). `_out/p5-extract-*.log`, `_out/p3-timing.txt`.
- **Fix**: user accepts the 18.6 min one-off (below the 30 min replan trigger) and times the first real incremental: `time rtk "$PY" -m src data_extract extract-earnings-calls -c ./configs`.

## F-005 — validator misses the AC populations (medium, 5)

- `src/validate/checks/earnings_calls.py:213-237, 261, 322`: `as_of_vs_release_days` is all-history and per call, not "share of events since 2026-06-01 matched ±3 d"; `split_ok_rate` and q05 are all-history, not since 2024; the 60 % turn-word rule is absent; the per-quarter floor raises no finding. P9 measured them ad hoc (`_out/p9-db-checks.txt`, `_out/p9-split-since-2024.txt`).
- **Fix**: add the four metrics with findings and thresholds in `configs/validate.yml`.

## F-006 — AC-008 deferred (medium, 4, accepted-deferred)

- The live `cube_part_text` holds only ABNB, PLTR, CEG (3,780 rows). The 16 score-7 drift findings (`_out/p5-validator-3-cube.txt`) are judged artefacts: `train_end` = 2022-01-01, so the train window holds only PLTR (from 2020-11) and ABNB (from 2021-02) with too few calls for `*_vs_hist` (missingness Δ −0.85 is the history warm-up) and PSI over two tickers' values is not a distribution test. P9 agrees with that reasoning, but it is not evidence of soundness: re-run after the user's full rebuild.
- **Proxy measured**: calls since 2024 — new 5,298 vs legacy sentiment 5,056; 109 legacy calls with no new call within 7 days on in-scope tickers, sampled ones are mostly wrong legacy dates (e.g. 2025-11-27 Fool publish dates).

## F-016 — management tagged as questioner on real calls (medium, 6)

- **Observed**: `_scripts/p9_question_role_scan.py` → `_out/p9-question-role-scan.txt` / `.csv` (exit 0) over the 5,241 ok calls since 2024: a question-tagged person speaks ≥ 800 words in 81 calls (1.55 %), ≥ 1,000 in 61 (1.16 %), ≥ 2,500 in 11; question words exceed answer words in 45 calls. Every one of the top 25 is an executive, e.g. TXN 2025Q1 Haviv Ilan (CEO) 3,861 words tagged question, answers left 191 words; BMY 2026Q1 Cristian Massacesi (CMO) 3,500; JPM 2024Q3 Jamie Dimon 2,709; CRM 2026Q2 Marc Benioff 3,098; PEP 2025Q3 Ramon Laguarta; NFLX Greg Peters on 4 calls.
- **Expected**: REQ-004 / AC-007 — no management turn is `question`. The 16 fixtures pass; the defect only shows on live calls.
- **Mechanism** (traced on BMY 2026Q1, `_out/p9-turn-roles.txt`): in `earnings_call_split._label` (`:687-700`) the management set is the **prepared-remarks speakers only**; a management redirect such as "Thanks for the question, Asad. Cristian, you want to start" is classified by `_is_operator` as a logistics turn (`True`, measured), which sets `boundary`; the next speaker, an executive absent from the prepared remarks, is then labelled `q`; if the turn contains a question mark the sticky-asker rule (spec role fix #1) keeps them a questioner for the rest of the call. Q&A-only formats (PEP, HSY, FANG, NFLX, DASH: prepared < 500 words on 140 ok calls) leave the management set nearly empty, so the CEO is exposed from the first answer. A related qa_start defect: DHI 2024Q2 starts the Q&A at the CFO's prepared turn 4 of 141, so prepared = 171 words.
- **Impact**: on those calls the Q&A sentiment text (`CallSplit.qa`, answers only) loses most management text, and the embedding roles behind `f_ec_qa_qq_distance` / `f_ec_qa_coherence_mean` are wrong; silent. Breadth ~1–2 % of recent calls.
- **Fix**: never label a speaker `q` from `boundary` alone when the preceding turn is a management redirect; seed the management set from Q&A speakers who answer (long, no-question turns following an analyst) and let the answer evidence override the sticky-asker rule; add a fixture for TXN 2025Q1 and BMY 2026Q1.
- **Verify**: the scan's ≥ 1,000-word count drops to the analyst-only residue (inspect the top 25); the new fixtures fail before and pass after.

## F-007 … F-015, F-017

- **F-007**: `earnings_call_features.py:154,169` — `sec_keys` = every call × tags; non-`ok` / gate-invalid calls are never written, so each run loads FinBERT and re-splits those tickers (500 such calls today). Fix: a handled marker for deterministic non-`ok` results.
- **F-008**: `fetch_earnings_call_transcripts.py:389-394` — paragraphs and derivatives deleted before the marker is saved; a crash in between loses the cube refresh. Fix: save markers first.
- **F-009**: DB shows `2025Q3 → 2026Q4` label jumps for DG, DLTR, HD, LOW, LULU, TGT, ULTA, WSM (`_out/p9-db-checks.txt`); the consecutive-quarter rule nulls one call's `tone_delta`/`length_delta`/distances. Provider labelling; no action unless measured material.
- **F-010**: `_stored_calls` reads paragraph 1 only; a call without it is rewritten each changed run. 0 such calls today.
- **F-011**: declare `huggingface_hub` in `pyproject.toml` and `airflow/requirements-airflow.txt`.
- **F-012**: `earnings_call_embeddings.py:205-212` — `earliest` set before `if not turns: continue`.
- **F-013**: PGR 2026-07-15, AES 2026-08-04, TECH 2026-08-12: no call within ±10 days in the source.
- **F-014**: docstrings in ask-first / protected files.
- **F-015**: catch `OverflowError` in `_paragraph_order`.
- **F-017**: `docker exec pea_db psql -c "select to_regclass('wiki_pageviews')"` → NULL; reword the line as history or drop it.

**Totals**: 1 high (F-001), 6 medium (F-002, F-003, F-004, F-005, F-006, F-016), 10 low/info (F-007 … F-015, F-017); 7 material simplifications (S-001 … S-007).
