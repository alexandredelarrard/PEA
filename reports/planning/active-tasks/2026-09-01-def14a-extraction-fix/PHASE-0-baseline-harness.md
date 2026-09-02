# Phase 0 — Baseline snapshot + comparison harness ✅

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

- [x] Resolve the 23 tickers (13 pinned + 10 seeded draw); persist to
      `reports/planning/active-tasks/2026-09-01-def14a-extraction-fix/baseline/manifest.json`
      with the snapshot UTC timestamp and each table's row count at snapshot time.
- [x] For each of the 6 tables, read **only** the 23 tickers and write one parquet per table to
      `baseline/`:

      def14a_llm, sec_def14a, sec_def14a_executive_comp, sec_def14a_director_comp,
      sec_def14a_ownership, sec_def14a_votes
      + sec_8k rows where item = '5.07'  (for Phase 5's baseline: rows = 0 by construction)

- [x] All reads via `context.store.load(table, where={"ticker": TICKERS}, columns=...)`.
      Never unprojected, never `pd.read_sql` (AGENTS.md hard rule).
- [x] Print the snapshot summary: per table, rows / tickers / accessions / date range.

```python
# shape only -- the store call is the contract
df = context.store.load(Tables.def14a_llm, where={"ticker": tickers})
df.to_parquet(out_dir / "def14a_llm.parquet", index=False)
```

### 2. `scripts/def14a_baseline.py` — filing cache

- [x] For each of the 23 tickers, `list_filings(context, cik, DEF14A_FORMS, years=31)` and
      `sec_get` each `doc_url`, writing the raw bytes to
      `data/cache/def14a_probe/{ticker}_{filing_date}_{accession}.htm`.
- [x] Use the existing rate-limited `sec_get`; ~23 × 26 ≈ **600 requests, ~70 s** at 9 req/s.
- [x] Skip a file that already exists, so the cache is idempotent and every later phase re-reads
      from disk with **zero** network and zero LLM cost. This is what makes Phase 2's recall
      harness cheap enough to run on every edit.
- [x] Also cache the pre-2001 `<accession>.txt` variant for filings whose `primaryDocument` is `""`
      — Phase 1 needs both to prove the fix.

### 3. `scripts/compare_def14a_baseline.py` (new) — the comparison

- [x] Loads `baseline/*.parquet` and `new/*.parquet` and prints one metric table. Fill rate is
      computed **per field over the same accession set** in both, so a coverage change cannot
      masquerade as a quality change.
- [x] Defect-specific assertions, printed as PASS/FAIL rows rather than raising, so one regression
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
| mean carve payload chars | ~50,300 (measured 51,198) | ≤ 45,000 (re-set in Phase 2; was 40,000) |

- [x] Output written to `reports/planning/active-tasks/2026-09-01-def14a-extraction-fix/COMPARISON.md`
      as well as stdout, so Phase 6 has an artifact to attach.

### 4. Model sanity check before the schema is committed (D11)

- [x] On **3 cached filings** (AAPL 2026, JPM 2026, CAT 2026), run today's
      `prepare_def14a_sections` → `LLMExtractor` under both `gpt-4o-mini` and `gpt-5-mini` and
      print a field-by-field diff.
- [x] Purpose: the research's per-field correctness numbers are `gpt-4o-mini`. If `gpt-5-mini`
      differs materially on the fields Phase 3 expands, that is worth knowing before writing the
      Pydantic contract, not after the full rerun.
- [x] 6 calls total. Record the outcome in this file.

#### RESULT — 33 of 51 watched fields agree; the 18 that differ all matter

**Production really does run `gpt-5-mini`.** Both call sites pass
`config.data_extract.llm_model` (`cli.py:418`, `step_extract_structure.py:35`), so the
`gpt-4o-mini` default inside `fetch_def14a_llm` is dead — as the plan assumed. Confirmed against
the data, not just the code: the stored rows for these three filings carry gpt-5-mini's answers
(CAT/JPM `n_neos=6`, AAPL `auditor_fees=34,277,000`, `insider_ownership_pct` NaN on all three),
not gpt-4o-mini's.

| field | `gpt-4o-mini` | `gpt-5-mini` | reading |
|---|---|---|---|
| `n_neos` | **1 on all 3** | **6 on all 3** | gpt-5-mini reads the SCT far better |
| `auditor_fees_usd` (AAPL) | 34,277 | **34,277,000** | gpt-5-mini already applies the "(in thousands)" note |
| `insider_ownership_pct` | 4e-06 / 0.087 / 0.15 | **None on all 3** | 4o-mini FABRICATES (AAPL 4e-06, CAT 15% are both absurd); gpt-5-mini correctly refuses |
| `majority_voting_for_directors` | True | False (JPM/CAT) | both are guesses — the degenerate inferred-FALSE field |
| `board_size`, `n_directors`, `n_women_directors` | 13 / 8 | 11 / 10 | the bios are TRUNCATED, so neither is reliable |

**Four consequences for the later phases:**

1. **`n_neos == 1` is run-to-run VARIANCE, not a model limitation.** AAPL 2026 stores `n_neos=1`
   from a production gpt-5-mini run, and the same model on the same carve returned **6** here.
   The anchor lands marginally, so the answer flips between runs. This *strengthens* Phase 2 —
   handing the model an explicit TSV of the SCT removes the variance, it does not merely raise
   recall — and it means G5 cannot be closed by a prompt change.
2. **The carve payload is 51,198 chars on ALL THREE filings — byte-identical.** Every section
   hits its cap on every filing, exactly as the research measured (mean 50,300). Phase 2's payload
   reduction is measured against 51,198, and `board_size` disagreeing between models is a
   *truncation* symptom, not a comprehension one.
3. **`insider_ownership_pct` will get WORSE before it gets better.** gpt-5-mini refuses to invent
   it, so its 57.3% fill is honest and the fix is Phase 2 actually delivering the ownership table —
   the Phase 1 dual-class prompt edit alone cannot move it.
4. **No reason to revisit the Pydantic contract.** gpt-5-mini is better or equal on every field
   Phase 3 expands, and its failures are recall (carve) rather than comprehension.

Probe kept at `scratchpad/model_probe.py` (not committed — a one-off measurement, and the numbers
that matter are recorded above).

---

## DISCOVERY — the pre-2001 `_doc_url` bug has TWO shapes, not one

Building the cache surfaced a second failure mode the research did not separate. Of the 663
filings cached for the 23 tickers:

| shape | n | `doc_url` behaviour | covered by Phase 1's planned fix? |
|---|---|---|---|
| `primaryDocument == ""` | **88** | returns HTTP **200** and a ~10,230-byte EDGAR **folder index** (1,487 chars of text, contains neither "proxy" nor "annual meeting") | **yes** — empty → `<accession>.txt` |
| `primaryDocument == "0001.txt"` | **7** | **404s** — the named document is not in the archive | **NO** — `primary_doc` is truthy, so the fallback never fires |

The 7 are AEE / GE / NKE / PEG / REG ×2 / T, all filed 2000-08 → 2001-03. They produce *no row
at all* today (`sec_get` raises, `_process_filing` swallows it), so they are invisible in the
"401 of 422 fully NULL" count — they are a **separate, unmeasured** loss on top of it.

Measured fix for both shapes — `<accession>.txt` is the right bytes either way:

| ticker | `<accession>.txt` | folder index |
|---|---|---|
| AEE | 65,478 B → 53,661 chars, proxy+meeting ✓ | 10,241 B → 1,488 chars ✗ |
| GE | 144,271 B → 125,326 chars ✓ | 10,226 B → 1,485 chars ✗ |
| NKE | 159,114 B → 141,936 chars ✓ | 10,234 B → 1,487 chars ✗ |
| T | 180,800 B → 165,380 chars ✓ | 10,234 B → 1,487 chars ✗ |

**Phase 1 therefore needs both halves** (deviation from the plan as written, recorded there):
`_doc_url` handles the empty case, and `list_filings` additionally emits a `txt_url` column so a
consumer can fall back when the primary document GET **fails**. A 404-triggered fallback is the
general form and needs no date cutoff or `"0001.txt"` hardcode.

## Verification

- [x] Run: `"$PY" scripts/def14a_baseline.py -c ./configs`
      → 7 parquet files + `filings.parquet` + `manifest.json` exist; `def14a_llm` = **445 rows**,
      matching `psql count(*)` for the same 23 tickers exactly.
- [x] Run: `"$PY" scripts/def14a_baseline.py -c ./configs` **again**
      → **0 downloaded, 744 already on disk**. Idempotent; parquet unchanged.
- [x] Run: `"$PY" scripts/compare_def14a_baseline.py --baseline-only`
      → prints the baseline metric table, `new` column empty, **0 FAIL / 14 PENDING**.
- [x] Confirm the cache holds ≥ 1 pre-2001 filing with an empty `primaryDocument` — **88 of them**,
      and both variants (`.htm` folder index + `.txt` full submission) are cached for each.
- [x] Confirm no DB writes occurred — the script only calls `store.load`.

### Baseline numbers (the "before" picture Phase 6 quotes)

| gate | baseline | matches research? |
|---|---|---|
| G1 exec-comp rows > $1e9 | **70** (23 tickers) | ✓ 109 DB-wide |
| G2 ownership footnote-digit rows | **0** | ✗ **not demonstrable** — see caveat below |
| G3 `peo_total_comp == 0.0` | 0 | SBUX has no `sec_def14a` rows yet |
| G4 PEOs for BA/NKE | BA=1 | ✓ (NKE has no rows yet) |
| G5 % 2012+ with `n_neos == 1` | **21.07%** | ✓ 21.7% |
| G6 `auditor_name` fill | **2.20%** | ✓ 2.05% |
| G7 post-2008 proxies w/ director comp | 22.52% | edgar path only; LLM path is 0 |
| G8 sub-0.50 say-on-pay surviving the drop | **0** | ✓ the floor deletes all 3 |
| G9 pre-2001 rows empty | **92.31%** | ✓ 401/422 = 95% |
| G10 mean carve payload | **51,198** | ✓ 50,300 — and identical on all 3 probed filings |
| G12 `pct_female_directors` fill | 92.13% | — |

**Three caveats now recorded in `COMPARISON.md` itself:**

- **G2 has no baseline evidence.** The edgar HTML backfill had reached only 6 of the 23 tickers
  (A, AAPL, AMAT, BA, ECL, EOG) at snapshot time, and **PG — the ticker carrying the
  footnote-digit defect — is not among them**. The gate still guards the new side.
- **G8 must be measured THROUGH `drop_implausible_def14a`.** `def14a_llm` keeps all 3 revolts
  (GE, JPM, TDG); the cube's 0.50 floor deletes them. Measured on the raw table the gate reads 3
  on both sides and proves nothing.
- **G9 must exclude the four inferred-FALSE provision flags.** Today's prompt orders the model to
  return FALSE for `poison_pill` / `classified_board` / `dual_class_shares` / `majority_voting`,
  so a row extracted from a 10 KB folder index still looks populated: **17 of 26** pre-2001
  baseline rows carry *nothing else*. A naive all-null test reports 34.6% instead of 92.3%.

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
