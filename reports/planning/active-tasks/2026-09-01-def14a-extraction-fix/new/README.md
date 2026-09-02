# `new/` — Phase-6 Step 1 output

## What is here, and what is NOT

| file | status |
|---|---|
| `sec_def14a.parquet` | **COMPLETE and valid.** 85 rows, 25 columns, all 23 baseline tickers, 2023+. Built by `scripts/def14a_phase6_runner.py --ecd`, which is deterministic — no LLM, no spend. |
| everything else | **ABSENT ON PURPOSE.** The proxy (`--proxy`) and vote (`--votes`) sub-runs were stopped before completing; you are doing the DEF 14A extraction yourself. |

The proxy/vote parquets that briefly existed here were a 22-row smoke set, and were **deleted
rather than left in place**: they were produced by each ticker's OLDEST filing only, and — worse
— through the folder-index bug described below, so every fill rate computed off them was wrong.
A stale artifact that reads like a validation run is more dangerous than a missing one.

`COMPARISON.md` therefore reports the LLM-path gates as `-`/PENDING. That is honest: those
gates have no measurement, not a failing one.

## To produce the missing side

```bash
"$PY" scripts/def14a_phase6_runner.py -c ./configs --proxy --workers 12   # ~656 calls, ~$11, ~100 min
"$PY" scripts/def14a_phase6_runner.py -c ./configs --votes --workers 12   # ~320 calls, ~$2
"$PY" scripts/compare_def14a_baseline.py                                  # regenerates COMPARISON.md
```

Both write **only parquet** — no DB writes at all — so running them cannot touch live data.
Run `--proxy` before `--votes`: the vote role map reads the proxy child tables out of this
directory (see below).

## Two measured facts worth keeping

**The pre-2001 folder-index trap.** 88 of the 656 cached filings carry
`primaryDocument == ""`, so `_doc_url` builds a bare *directory* URL and Phase 0's `cache_htm`
holds a **10,217-char EDGAR directory listing** instead of the proxy. The real document is the
`.txt` sibling (159,203 chars on A's 2000 filing). Reading `cache_htm` for those filings fed the
model a folder index, and they came back with **0-2 non-null fields** — which reads as
"pre-2001 does not extract" when the cause is opening the wrong file. `_filings()` now prefers
`cache_txt` when `pre_2001_empty_primary` is set. Verified on A 2000-01-13: **2 non-null fields
-> 27**, plus `company_name` "Agilent Technologies", `board_size` 6, CEO "Edward W. Barnholt",
auditor "PricewaterhouseCoopers LLP", 5 NEOs, and 5 exec-comp / 11 ownership / 6 director rows.
Production was never affected — it uses Phase 1's fixed `_doc_url`.

**The role map cannot be measured from the database yet.** `_role_source` joins to
`def14a_llm` / `def14a_executive_comp` / `def14a_director_comp`, and the last two DO NOT EXIST
in Postgres until the cutover creates them. A DB-sourced map returned nothing and the run
reported **96.8% `unmatched`**, which says everything about the missing tables and nothing about
the join. `--votes` therefore builds its role frames from *this directory* via
`_parquet_role_source`, which holds exactly what the cutover is about to write.
