# Post-run checklist — what waits until the walk finishes

The full `main.py` walk started 2026-08-27 00:06 and runs **until tomorrow**. Verification for
this refactor is scoped to the 8-ticker sample (Phase 0). Everything below is deliberately
**outside** the refactor's gate: it either needs the walk's output, or it writes to data the walk
is still writing.

Tick these off in a separate session, and record the results in the Phase 7 DoD report.

## Before touching anything (already done)

- [x] `data/extraction_manifest.json` copied to `manifest-snapshot.json` — 17 entries, 3.6 KB.
      The in-flight run will overwrite the live file on completion, and Phase 5 needs those
      entries to seed `extraction_run`.

## When will the walk actually finish?

Measured, not assumed:

- **Fetch**: 402 tickers walked by 12:05 from a 00:06 start = ~34 tickers/h, against a ~491-ticker
  universe. Fetch alone lands around **14:45 today**.
- **Replay**: it runs immediately after, single-threaded, and one real filer's full-history replay
  measured **323.69 s** (MCD, E=69). 491 tickers = **~44 hours**. `build_ticker` replays every
  event on every run and only then filters to new `as_of`s, so already-published tickers cost full
  price.
- So the run plausibly finishes around **2026-08-29**, not tomorrow. Check
  `SELECT count(distinct ticker) FROM fundamentals_history_sec;` to track it — it was **54** at
  12:05.

That gap is worth knowing before planning around "tomorrow". It is also the single strongest
argument for Phase 4.

## Blocked on the walk finishing

### 1. Coverage measurement, then the `cols` remediation decision (deferred item D-2)

- [ ] Zero-fact detector:
  ```sql
  SELECT t.ticker FROM sp500_tickers t
  LEFT JOIN (SELECT DISTINCT ticker FROM fundamentals_facts) f USING (ticker)
  WHERE f.ticker IS NULL;
  ```
  Confirmed still failing at 12:05 today: **NEM, MO, AIZ have 0 rows**, against 402 tickers /
  2.44 M rows present.
- [ ] Weaker signal: tickers whose `max(filing_date)` sits far behind the roster median.
- [ ] **Then decide**: targeted refetch of the affected tickers, or a wider rebuild. This is
      decision 3 and it was explicitly deferred until this number exists.

### 2. The ledger seed and JSON cutover (Phase 5)

- [ ] Run `scripts/seed_extraction_run.py` against the **post-run** JSON.
- [ ] Assert `manifest_window` returns the same `(since, is_full_rescan)` pair from the DB as
      from the JSON, for every table. Record the pairs.
- [ ] Only then delete `data/extraction_manifest.json` and the `.gitignore:120` entry.
- [ ] **Precondition**: no fetcher may run between the code cutover and this seed. The sample
      replays never touch `manifest_window` or `record_run`, so the window is safe if nothing
      else is launched.

### 3. Full-universe replay acceptance (Phase 4)

The walk's own replay publishes history for the remaining tickers under the **old** code
(`fundamentals_history_sec` had only **54 tickers** at 12:05). Once it has:

- [ ] Non-rebuild `build_fundamentals_history` over the full universe with the refactored code.
      Expected: **0 rows appended, 0 drift raised**. `diff_against_stored` compares against
      numbers published by the old code, which makes this the strongest available check — far
      stronger than any sample.
- [ ] Record wall clock at `workers=1` and at the configured width, plus peak RSS.
- [ ] If drift *is* raised: it is a real finding. Do not pass `--rebuild-history` to make it go
      away — read the diff, find which phase moved the cell, and revert that item.

### 4. The wide edge-case check (Phase 0's excluded 12)

The sample carries 8 tickers. These 12 were named by the research as distinct edge cases and left
out on purpose:

| Ticker(s) | Edge |
|---|---|
| GS | Q1-2013 tagged `-> 03-30` and `-> 03-31` |
| AMT | one annual fact per field -> the shared-calendar fix (69 collisions) |
| AXP, JPM | bank revenue basis, filer-specific (AXP yes, JPM no) |
| AMZN | 20x split-adjusted share block |
| XOM | 36 % frozen `revenueGrowth` under the legacy staircase |
| EOG | fiscal-2012 share basis, 2-for-1 split at 1.996–2.003 |
| ALL, GILD, SPGI, GPC, ZBH, JCI, SJM | the real Q4s at 1.03–2.59x that calibrate `max_opposite_sign_q4_ratio` |

- [ ] Replay-equality on these too, at full history, once the full-universe check above is clean.
      They are cheap by then — the code is faster and the harness already exists.

### 5. Re-run the restatement census over the whole table (Phase 3.5)

- [ ] The in-phase census is scoped to the 8 sample tickers and marked **indicative**. Re-run it
      over all ~491 tickers and update `restatement-census.md`'s recommendation on the vintage
      redesign (deferred item D-3) with the real distribution.

### 6. `pension_facts` (Phase 5.4)

- [ ] `fetch_financial_statements` is now on the Step chain, so `main.py` starts writing
      `pension_facts`. Confirm the download budget is acceptable **before** the next full run —
      it shares the cache dir with `fetch_financial_notes`, already the heaviest download in the
      repo (~300–450 MB per set, ~26 GB back-fill at `notes_years_history=15`).

## Not blocked, but out of scope by decision

See [deferred.md](deferred.md) — D-1 (the cube's dead workforce block, a **live bug**), D-4 (SEC
retry/backoff), D-5 (the 49 GB cache), D-6 (constants outside `fundamentals/`), D-7 (OpenAI/FRED/
Sharadar clients), D-9 through D-13.

**D-1 is the one worth scheduling next.** `employee_growth`, `revenue_per_employee` and the whole
workforce panel are silently empty in the cube today, and therefore in every model trained on it.
