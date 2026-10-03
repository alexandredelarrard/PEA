# Validation 01 — independent domain validation (P9)

**Candidate**: worktree `reports/worktrees/earnings-call-extraction-gaps`, branch `harness/earnings-call-extraction-gaps`, HEAD `56dea8b` (`_out/p9-head.txt`); base `71a9ecd`. **Run**: 2026-10-02 21:49 → 22:xx local, fresh validator context.
**Constraints (user)**: no full test suite (P7 ran it: `_out/p7-pytest-full.txt`), no FinBERT/embedding/`build-text`, no code edits, no DB writes. The P5 validator run on `cube_part_text` is reused, not re-run.

## Lanes

Code (targeted tests, collection, grep, review), data (read-only SQL over the live tables, pure-function replays), modelling (PIT and feature contract only; retraining is out of scope).

## Commands and exits

| # | Command (worktree root unless noted) | Exit | Decisive observation | Artifact |
|---|---|---|---|---|
| 1 | `rtk "$PY" -m pytest --collect-only -q tests` | 0 | 2,052 collected = P7 total (7 + 2,019 + 26) | `_out/p9-collect.txt` |
| 2 | `pytest tests/utils/test_speaker_turn_split.py tests/validate/test_earnings_calls.py -q` | 0 | 77 passed | `_out/p9-pytest-split-validate.txt` |
| 3 | `pytest tests/data_aggregate -k "earnings_call or cube_text" -q` | 0 | 29 passed | `_out/p9-pytest-aggregate-ec.txt` |
| 4 | `pytest tests/data_extract/behavioral tests/dags tests/utils tests/data_store -q` | 1 | 1 failed, 169 passed, 4 skipped; the failure is P0 baseline item 1 (`test_dag_chain_is_derived_from_the_registry`, unrelated literal-chain assertion) | `_out/p9-pytest-extract-dags-utils-store.txt` |
| 5 | grep for header regexes, retired sources, protected paths | — | header regexes: 0 src hits; retired sources: only 2 docstrings in ask-first/protected files; protected roster scraper files: empty diff | `_out/p9-grep.txt` |
| 6 | `docker exec -i pea_db psql … < _scripts/p9_db_checks.sql` | 0 | grain clean; AC-001 quarters 474–484; AC-002 99.41 %; scope ceiling 487 | `_out/p9-db-checks.txt` |
| 7 | `PYTHONPATH=. python _scripts/p9_orphan_repro.py` | 0 | orphan on relabel flip: YES | `_out/p9-orphan-repro.txt` |
| 8 | `PYTHONPATH=. python _scripts/p9_split_since_2024.py` | see AC-006 | split over every call since 2024 | `_out/p9-split-since-2024.txt` |
| 9 | `git merge-tree --write-tree --name-only dev harness/earnings-call-extraction-gaps` (main tree) | 0 | clean merge onto current `dev` (`191fc65`) | `_out/p9-merge-tree.txt` |
| 10 | OK `audit` on `wiki/` (main tree) | — | 3 dead links, all in `modules/application-and-scripts.md` (`scripts/dod/*`), not touched by P8; OK `lint` ran no rule family | (tool output, recorded here) |
| reuse | P5 `validate -T cube_part_text` | 1 | 16 score-7 drift findings on a 3-ticker cube; split ok 98.63 %; grain clean | `_out/p5-validator-3-cube.txt`, `_out/earnings_calls.json` |
| reuse | P5 parity at revision `cd41c0ac` | — | 33,591 = 33,591, 0 only-in-source, 0 only-in-DB, 0 `as_of` mismatches | `_out/p5-parity.txt` |

The generic nine-check `src/validate` library was **not** re-run on `earnings_call_sections`: the table is a raw text table (no numeric features) whose grain/coverage checks the domain check `earnings_calls` already performs, and the cube it feeds holds only the 3-ticker sample. That is an accepted abstention until the user's full rebuild; it prevents any claim about `cube_part_text` distributions.

## Data lane

- **Scope** (`_out/p9-db-checks.txt` §1): `earnings_call_sections` 2,804,060 rows, 33,591 calls, 487 tickers, 2005-10-13 → 2026-10-01. `earnings_call_sentiment` 290 rows / 145 calls (probe + sample only). `cube_part_text` 3,780 rows, 3 tickers. Legacy tables still present (10.3 GB).
- **Grain** (§2): 0 calls with >1 `as_of`, 0 missing paragraph 1, 0 non-contiguous, 0 multi-`transcript_id`, 0 calls sharing a (ticker, `as_of`); 2,867 calls with NULL `transcript_id` (source property, handled by the date-change re-issue rule).
- **Coverage**: parity with the source is exact; per-quarter tickers 474–484 vs floor 485 (F-003); 13 roster tickers have no call: BRK-B, ED, EXPD, NVR (no calls anywhere) and 9 `INSUFFICIENT_HISTORY_TICKERS` spin-offs (no prices).
- **PIT**: `_daily_frame` and `side="right"` unchanged in the diff; the PIT unit test passes (`test_call_is_invisible_on_its_as_of_session_and_visible_on_the_next`); live sample: ABNB 2025-08-06 call, tone 0.4137 on the `as_of` session → 0.5857 on 2025-08-07; CEG 2025-05-06 NULL → 0.3543 next session (`_out/p5-e2e-checks.txt` §3). `as_of` = source `report_date` for 100 % (parity). AC-002: 507/510 events since 2026-06-01 within ±3 days.
- **Idempotency**: no-op on unchanged fingerprint (3 runs, 0 rows); exact parity after the full load; **but** the incremental/reconcile path cannot remove a stored key that leaves the source (F-001, reproduced).
- **Drift**: the 16 P5 drift findings are judged sample artefacts (F-006): `train_end` 2022-01-01 leaves PLTR and ABNB with a handful of calls in the train window, so `*_vs_hist` missingness Δ −0.85 is history warm-up and PSI over two tickers is not a distribution test. The reasoning holds; it is not evidence of soundness. Re-run after the full cube rebuild.
- **Coverage vs legacy proxy** (§6): calls since 2024 — new 5,298 vs legacy 5,056. Legacy calls with no new call within 7 days: 79 out of scope (the 9 spin-offs) and 109 in scope; sampled in-scope cases are legacy date errors (Fool publish dates such as 2025-11-27 for ABNB/APTV/GOOGL/DDOG 2025Q3, whose real calls are present on 2025-11-06/10-30/10-29/11-06).

## AC-006 (since 2024)

`_scripts/p9_split_since_2024.py` → `_out/p9-split-since-2024.txt` (exit 0, 17.6 min), production `split_call` over every stored call dated ≥ 2024-01-01:

| Measure | Result | Target | Status |
|---|---|---|---|
| calls / exceptions | 5,301 / **0** | never an exception | pass |
| status | ok 5,241, no_qa 32, no_prepared 28 → **98.87 %** | ≥ 98.5 % | pass |
| prepared share q05 / q50 / q95 | **0.166** / 0.410 / 0.624 | q05 ≥ 0.15 | pass |
| (prepared + qa) / cleaned turn words | q01 0.561, q05 0.665; **85 ok calls (1.62 %) < 0.60** | none < 0.60 | **fail as written** |

The 60 % rule predates user decision C-1 (Q&A sentiment text = management answers only), so question text now leaves `qa` by design. Decomposition of 6 of the 85 (`_scripts/p9_low_coverage_examples.py` → `_out/p9-low-coverage-examples.txt`): adding question-turn words lifts every one to 0.63–0.91. But the per-turn roles (`_scripts/p9_turn_roles.py` → `_out/p9-turn-roles.txt`) show that part of that "question" text is **management mislabelled as question** — see F-016 below.

## F-016 — real-data role check (AC-007)

`_scripts/p9_question_role_scan.py` → `_out/p9-question-role-scan.txt` (exit 0, 14.6 min): over 5,241 ok calls since 2024, 61 (1.16 %) have a question-tagged person with ≥ 1,000 words and 45 have more question than answer words; all top-25 suspects are executives (TXN Haviv Ilan 3,861 words, BMY Cristian Massacesi 3,500, CRM Marc Benioff 3,098, JPM Jamie Dimon 2,709). Mechanism traced on BMY 2026Q1: a management redirect line is classified as logistics (`_is_operator` → True), the boundary labels the next, non-prepared executive `q`, and the sticky-asker rule keeps it. AC-007 holds on the 16 fixtures but not on live calls. Registered as F-016 (medium, 6).

## Verdict of this attempt

Measured; one high (F-001) and six medium findings open; AC-008 abstains until the user's full rebuild. Not ready for PR without the fixes or explicit user acceptance — see [06-final](06-final.md).
