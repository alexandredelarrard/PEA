# Phase 0 — Baseline snapshot + comparison harness ⬜

**Goal**: freeze today's tables as an immutable baseline artifact for the 23 chosen tickers, cache
their raw filings to disk, and build the script that prints the before/after comparison. Nothing
about the extraction changes in this phase.

**Why first**: the DEF 14A and 8-K backfills are still running (`sec_def14a` moved 26 → 151 tickers
between the research pass and this plan). A baseline read at cutover time would be a different
population than the baseline read today, and the comparison would be meaningless.

---

## The 23 tickers (D9)

**13 named-defect tickers**, each pinned to the defect it demonstrates:

| ticker | defect it proves is fixed |
|---|---|
| `A` | `<br>`-stacked multi-year cell → `salary` = 1.000e19, `year` = `200420032002` |
| `AMAT` | worst offender: `total` = 3.527e23, wrong in 9 consecutive filings 2017-2026 |
| `PG` | ownership `shares` 10× via a `<sup>` footnote digit — 12/12 rows across 6 proxies; also flips audit-fee scale between its own filings |
| `SBUX` | ECD zero-matrix → `peo_total_comp` and `peo_actually_paid_comp` = 0.0 for FY2023/24/25; `net_income` tagged in $ millions (1856.4) |
| `BA` | co-PEO year: Ortberg kept, **Calhoun dropped** (CAP −23,875,735) |
| `NKE` | co-PEO year: Hill kept, Donahoe dropped |
| `CAT` | director-comp `stock_awards` NULL under a `Restricted Stock Units` header; `pension_change == total` duplication |
| `PFE` | director-comp `stock_awards` NULL on 13/13 rows under a header reading exactly `Stock Awards ($)` |
| `GE` | `$`-in-its-own-`<td>` desync (3 of 4 numeric columns dropped); CEO typed `5pct_holder`; a literal `Total` row typed `director_officer` |
| `T` | zero voting proposals — writes `Management Proposals:` with rows `1.`-`8.`, never "Proposal 1" |
| `XOM` | `FOR` fabricated onto two shareholder proposals the board opposes; fee/ownership content on an image page |
| `JPM` | say-on-pay 31% (a real revolt the 0.50 floor deletes); address-as-holder; SCT invisible to a classifier without block separators |
| `AAPL` | `peo_name` = display text `"Mr. Cook"`; 26 `ecd:PeoName` facts of which only 5 are `PeoMember`; recommendation cell is a JPEG |

**10 random**, drawn from the `def14a_llm` ticker set with a fixed seed so the run is reproducible:

```python
RANDOM_SEED = 20260901          # never change; the baseline is keyed to it
DEFECT_TICKERS = ["A", "AMAT", "PG", "SBUX", "BA", "NKE", "CAT", "PFE",
                  "GE", "T", "XOM", "JPM", "AAPL"]
# 10 more sampled from sorted(def14a_llm.ticker.unique()) minus DEFECT_TICKERS
```

Draw from `def14a_llm` (497 tickers), **not** `sp500_tickers` — a ticker with no baseline rows
gives nothing to compare against. Write the resolved 23 to the snapshot manifest so Phase 6 uses
the identical list even if `def14a_llm` has grown.

---

## Changes

### 1. `scripts/def14a_baseline.py` (new) — snapshot

- [ ] Resolve the 23 tickers (13 pinned + 10 seeded draw); persist to
      `reports/planning/active-tasks/2026-09-01-def14a-extraction-fix/baseline/manifest.json`
      with the snapshot UTC timestamp and each table's row count at snapshot time.
- [ ] For each of the 6 tables, read **only** the 23 tickers and write one parquet per table to
      `baseline/`:

      def14a_llm, sec_def14a, sec_def14a_executive_comp, sec_def14a_director_comp,
      sec_def14a_ownership, sec_def14a_votes
      + sec_8k rows where item = '5.07'  (for Phase 5's baseline: rows = 0 by construction)

- [ ] All reads via `context.store.load(table, where={"ticker": TICKERS}, columns=...)`.
      Never unprojected, never `pd.read_sql` (AGENTS.md hard rule).
- [ ] Print the snapshot summary: per table, rows / tickers / accessions / date range.

```python
# shape only -- the store call is the contract
df = context.store.load(Tables.def14a_llm, where={"ticker": tickers})
df.to_parquet(out_dir / "def14a_llm.parquet", index=False)
```

### 2. `scripts/def14a_baseline.py` — filing cache

- [ ] For each of the 23 tickers, `list_filings(context, cik, DEF14A_FORMS, years=31)` and
      `sec_get` each `doc_url`, writing the raw bytes to
      `data/cache/def14a_probe/{ticker}_{filing_date}_{accession}.htm`.
- [ ] Use the existing rate-limited `sec_get`; ~23 × 26 ≈ **600 requests, ~70 s** at 9 req/s.
- [ ] Skip a file that already exists, so the cache is idempotent and every later phase re-reads
      from disk with **zero** network and zero LLM cost. This is what makes Phase 2's recall
      harness cheap enough to run on every edit.
- [ ] Also cache the pre-2001 `<accession>.txt` variant for filings whose `primaryDocument` is `""`
      — Phase 1 needs both to prove the fix.

### 3. `scripts/compare_def14a_baseline.py` (new) — the comparison

- [ ] Loads `baseline/*.parquet` and `new/*.parquet` and prints one metric table. Fill rate is
      computed **per field over the same accession set** in both, so a coverage change cannot
      masquerade as a quality change.
- [ ] Defect-specific assertions, printed as PASS/FAIL rows rather than raising, so one regression
      does not hide the rest:

| check | baseline expectation | target |
|---|---|---|
| exec-comp rows with any component > $1e9 | > 0 | **0** |
| ownership rows where `shares/(percent × shares_out)` ≈ 10 | > 0 (PG 12) | **0** |
| accessions with `n_neos == 1`, 2012+ | ~21.7% | ≪ that |
| `say_on_pay_support_pct < 0.50` rows | dropped by the floor | **present** |
| pre-2001 rows fully NULL | 401 / 422 | ≪ that |
| `peo_total_comp == 0.0` (SBUX FY23-25) | 3 rows | **0** |
| distinct PEO names per (ticker, fiscal year) for BA/NKE | 1 | **2** |
| mean carve payload chars | ~50,300 | ≤ 40,000 |

- [ ] Output written to `reports/planning/active-tasks/2026-09-01-def14a-extraction-fix/COMPARISON.md`
      as well as stdout, so Phase 6 has an artifact to attach.

### 4. Model sanity check before the schema is committed (D11)

- [ ] On **3 cached filings** (AAPL 2026, JPM 2026, CAT 2026), run today's
      `prepare_def14a_sections` → `LLMExtractor` under both `gpt-4o-mini` and `gpt-5-mini` and
      print a field-by-field diff.
- [ ] Purpose: the research's per-field correctness numbers are `gpt-4o-mini`. If `gpt-5-mini`
      differs materially on the fields Phase 3 expands, that is worth knowing before writing the
      Pydantic contract, not after the full rerun.
- [ ] 6 calls total. Record the outcome in this file.

---

## Verification

- [ ] Run: `"$PY" scripts/def14a_baseline.py -c ./configs`
      → 7 parquet files + `manifest.json` exist; printed row counts match a direct
      `psql` count for the same 23 tickers.
- [ ] Run: `"$PY" scripts/def14a_baseline.py -c ./configs` **again**
      → zero new HTTP requests (cache hit on every filing), parquet unchanged.
- [ ] Run: `"$PY" scripts/compare_def14a_baseline.py --baseline-only`
      → prints the baseline metric table with the `new` column empty. This is the "before" picture
      quoted in Phase 6.
- [ ] Confirm the cache holds ≥ 1 pre-2001 filing with an empty `primaryDocument` (needed by
      Phase 1). `A`, `GE` and `T` all have them.
- [ ] Confirm no DB writes occurred: `manifest.json` row counts re-read after the run are unchanged.

## Rollback

Nothing to roll back — this phase only reads the DB and writes new files under `scripts/`,
`data/cache/` and the plan directory.

## Notes

- The snapshot is the **only** rollback for Phase 6's truncation. Do not delete `baseline/`.
- `sec_8k` item 5.07 baseline is rows = 0 by construction (nothing parses them today). It is
  snapshotted anyway so Phase 5 can show the `item_text` corpus it worked from and prove the
  guard's "emit nothing" cases were genuinely empty rather than skipped.
- Per the repo's Postgres DATE round-trip trap: parquet-cached harnesses hide the
  `DATE → datetime.date` bug class entirely. Phase 6 re-runs the defect assertions **against
  Postgres** after the real rerun, not only against parquet.
