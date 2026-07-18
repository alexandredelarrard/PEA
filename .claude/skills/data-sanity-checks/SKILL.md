---
name: data-sanity-checks
description: >-
  Audit the pipeline's PostgreSQL tables for unexpected missing values and column
  inconsistencies on a random sample per table, then prove whether each gap is a
  DATA issue (genuinely absent at the source) or an EXTRACTION bug (present at the
  source but dropped by our code) and iterate on a fix. Use when the user asks to
  sanity-check the data, verify the extracted/aggregated tables, or investigate
  missing / NaN values.
---

# Data sanity checks

Goal: for a random sample of **each created table**, confirm missing values are as
expected and columns are consistent; and for anything suspicious, **prove at the root
source** whether it's a data issue or an extraction issue, then fix the extraction and
re-verify. Fixing is iterative — measure, triage, trace, fix, re-measure.

## 0. Preconditions
- The Postgres container is up (`docker compose up -d db` from `stock_pick_strat/`).
- Run everything from `stock_pick_strat/` with the project env (`poetry run python ...`),
  so `./configs`, `.env`, and `src` resolve.

## 1. Measure — run the checker
```bash
cd stock_pick_strat
poetry run python "../.claude/skills/data-sanity-checks/table_health.py"
# subset / tuning:  --tables fundamentals_history,cube   --sample 300   --flag-threshold 0.5
```
It prints, per table: PK integrity (null/duplicate), `date_col` coverage, and per-column
null-rate on the sample; then a **TRIAGE WORKLIST** of flagged columns (all-null, high
null-rate, PK/date problems). Work that list.

## 2. Triage — is the gap expected?
For each flagged column, first check it against **Known-legitimate sparsity** below. If it
matches, it's expected → note it and move on. Otherwise treat it as suspicious and go to §3.

Hard failures are never "expected" — always investigate:
- PK nulls or duplicate PK groups (upsert/merge or COPY bug),
- NULLs in a `date_col`,
- a column **all-null on the full table** (nothing is landing → almost always extraction),
- a column that used to populate and now doesn't.

## 3. Prove it at the root source (data vs extraction)
Pick 2-3 sampled tickers that have the value missing, and inspect the **raw source** for
that table (map below). The verdict:
- **Absent at source** → DATA issue: the value genuinely doesn't exist. Confirm it's on the
  expected-sparsity list (add it if it's a newly-understood legitimate gap) — no code change.
- **Present at source but NULL/dropped in the DB** → EXTRACTION bug: fix our code (§4).

Source map (all raw sources are on disk or re-queryable; find exact paths via `context.paths`):

| Table | Root source to check |
|---|---|
| `fundamentals_history` | Cached SEC **companyfacts JSON** under `data/sec_bulk_cache/` — search the `us-gaap`/`dei` tags for the concept. If the tag exists there but the column is NULL → a missing/!coalesced XBRL tag. |
| `fundamentals_snapshot` | yfinance `.info` (`marketCap`, `forwardPE`) — re-query the ticker; `forwardPE` is legitimately absent without analyst coverage. |
| `prices` / `dividends` | yfinance — re-download the ticker over the window. |
| `short_interest` | FINRA RegSHO daily files. |
| `institutional_holdings` | Cached **13F zips** under `data/sec_bulk_cache/form13f/` (SUBMISSION/INFOTABLE tsv). call/put/debt/other are `0`, not NULL, for pure long-equity rows. |
| `def14a_llm` | The cached **DEF 14A filing text** + the Pydantic schema — did the proxy disclose it (data) or did the LLM/flatten miss it (extraction)? |
| `macro` | FRED series. |
| `google_trends` / `wiki_pageviews` | pytrends / Wikimedia API — many tickers legitimately have thin coverage. |
| `cube` / `cube_signal` / `predictions` | Derived — trace back to the input table + the builder in `src/data_aggregate/` (feature warmup, peer availability, join keys). |

## 4. Fix the extraction bug (only for proven extraction issues)
Follow the repo conventions (see `CLAUDE.md`) — do **not** touch risk-zone files
(`src/context.py`, `src/utils/step.py`, `src/constants/*`, `configs/*`, `src/data_store/*`,
`sql/schema.sql`, `data/`) without approval.
- **Missing XBRL concept** → add the alternative tag to the fetcher's candidate list and
  **coalesce** (union candidates per period; don't take the first present) in
  `src/data_extract/utils/fundamentals/`. Derive a concept only when no filer tags it.
- **Reconciliation gap** (13F, CUSIP↔ticker) → fix the identifier join, never the free-text name.
- **Parser/flatten gap** (DEF 14A) → fix the field mapping / `_flatten`.
- New tag names, thresholds, or keys go in `src/constants/constants.py`; booleans persist as
  numeric 1.0/0.0 flags.

## 5. Re-verify and iterate
Re-run the relevant fetcher for the affected tickers (it's per-entity + incremental, so this
is cheap), then re-run the checker on just that table:
```bash
poetry run python "../.claude/skills/data-sanity-checks/table_health.py" --tables <table>
```
Confirm the column now populates for the tickers that have the value at source. Repeat §2-§5
until the TRIAGE worklist holds only expected sparsity. Add/adjust a test alongside any fix
(synthetic known-truth for the math + a real-data coverage check against the cached source).

## 6. Conclude (mandatory)
Print a sanity-check conclusion stating, per table checked: what was verified, which gaps are
DATA (expected, with why) vs EXTRACTION (fixed, with what changed), and that PK/consistency held.
Per `CLAUDE.md`, the work is not done without this printed conclusion.

---

## Known-legitimate sparsity (expected — not bugs)
- **Sector-specific fundamentals** are NULL outside their sector: bank NII/NIM, insurance
  premiums/claims/combined ratio, REIT FFO/rental income, energy DD&A/production. A column
  populated only for its GICS group is correct.
- **TTM columns** need ~4 quarters of warmup (`min_periods`), so the earliest `as_of` rows per
  ticker are NULL by design.
- **`fundamentals_snapshot` and `institutional_holdings` accrue point-in-time going forward**
  (no clean back-history from yfinance/13F), so early history is sparse and the most recent 13F
  quarter lags ~45 days.
- **`forwardPE`** is absent for names without analyst coverage.
- **`cube` peer-relative / forward / sector features** are NULL before feature warmup, for
  sole-member peer groups, and until the snapshot has accrued — expected, not a join bug.
- **Attention data** (`google_trends`, `wiki_pageviews`) has genuinely thin coverage for some tickers.
