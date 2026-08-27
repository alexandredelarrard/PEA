# Phase 5 — Documentation and definition of done ⬜

**Goal**: leave the repo's docs telling the truth about a two-source fundamentals stack, and write
the DATA DoD report.

**Prerequisite**: phase 4 verified; the override register adjudicated.
**Read first**: [README.md](README.md) — especially "Known risks, accepted".

---

## Doc changes

### `AGENTS.md` — **cap 70 lines**, so this is a rewrite of existing lines, not an append

- [ ] The code map's `data_extract/` line gains `fundamentals_sharadar/`.
- [ ] Add one line making the two-table split unmissable, because a future agent reading
      "fundamentals_history" will otherwise assume it is the SEC table it has always been:
      `fundamentals_history` = Sharadar-first merged; `fundamentals_history_sec` = the SEC replay.
- [ ] Propose the wording before editing — `AGENTS.md` is a declared risk zone and has a hard cap.

### `docs/data_sources.md`

A new Sharadar section. The quirks that cost a day each if undocumented:

- [ ] Direct API only; **never** `data.nasdaq.com`, never `nasdaqdatalink` / `quandl` with this key.
- [ ] The filing-date column is **`date`**, not `datekey` (the Nasdaq channel's name, and the
      research doc's notation throughout).
- [ ] **`from` defaults to "1 year ago"**, `limit` to 10000, `sort` to `date.desc` — always explicit.
- [ ] **`fields=` silently drops an unavailable field** with no warning.
- [ ] **403 means not entitled**, not throttled — and `polite_http.http_get` retries 403 four times
      by default, so it must be called with `retries=0`.
- [ ] Only **8 columns are USD-converted**; everything else is the filer's reporting currency while
      `marketcap`/`price` are always USD. We assert USD and refuse non-USD filers (D20).
- [ ] Money columns are **actual units** in SF1 but **USD millions** in the `daily` table — a 10⁶
      factor between two tables in the same subscription.
- [ ] Ratio columns are **decimal fractions**, not percentages, despite the 2019 dictionary typing
      them `%`. `evebit` is `bigint` and returns integer-truncated.
- [ ] **`de` is liabilities/equity**, not debt/equity, despite its name.
- [ ] `capex` and the `ncf*` legs are stored **negative**; the repo's `capex` is `non_negative`.
- [ ] `lastupdated` is a **per-ticker reprocessing stamp**, not a per-row change stamp.
- [ ] **Only AR\* is point-in-time.** MR\* rows mutate in place and are not stored.
- [ ] **Quarterly dimensions are US-domestic-only** — ADR (form 20) and Canadian (form 40) filers
      have no ARQ/MRQ at all. Relevant the moment the universe widens past the S&P 500.
- [ ] SF1 covers the **primary share class only**.
- [ ] The measured entitlement of the current key (29 DJIA tickers, ~5y, no bulk).

### `docs/data_schema.md`

- [ ] Rows for `fundamentals_sharadar`, `sharadar_tickers`, `sharadar_actions`, `sharadar_sp500`.
- [ ] `fundamentals_history` rewritten: new column set, new grain semantics, **and the explicit note
      that the 4 amendment/provenance columns are gone by decision**.
- [ ] `fundamentals_history_sec` documented as the SEC replay — **still reason-coded, still the only
      table `src/validate/` looks at, and still the sole owner of `is_amendment` /
      `amended_fiscal_end` / `amended_fields`**, which are SEC reconciliation columns and do not
      cross into the merged table.

### `docs/database.md`

- [ ] Refresh the fundamentals section with measured row counts, ticker counts and date ranges from
      the phase-4 verification query.

### `docs/architecture.md`

- [ ] `StepExtractFundamentalsSharadar` in the extraction stage, before `StepExtractFundamentals`.

### `README.md` (repo root)

- [ ] One line if it enumerates data sources.

---

## The Simplify code

Run the agent via `/simplify` to refactor the code where it can. 
Use the diff for all the commits done on the branch feature/sharadar-fundamentals. 
EFFICIENCY-focused code quality review step which reviews  an in-progress refactor in the Python repo. Get the diff with: `git diff`. 
YOUR ANGLE — EFFICIENCY / CONSISTENCY: Flag wasted work the diff introduces. (sub agent to run it) 

---

## Final verification

```bash
PY="$HOME/AppData/Local/pypoetry/Cache/virtualenvs/stock-pick-strat-lkf53h9P-py3.13/Scripts/python.exe"
rtk "$PY" -m pytest tests/data_extract -v -s
rtk "$PY" -m pytest tests/test_no_sql_outside_data_store.py -v -s
```

- [ ] All new tests pass and each prints a sanity conclusion.
- [ ] No `sqlalchemy` / `pd.read_sql` / `to_sql` / `store.engine` outside `src/data_store/`.
- [ ] No table-name string literals — every call passes a `Tables.<name>` object.
- [ ] No `print()` in `src/`; `self._log` in steps, `context.log` in helpers.
- [ ] Full type annotations, imports at top, no cross-imports between `src/` subfolders.
- [ ] `AGENTS.md` is still ≤ 70 lines.
- [ ] The DoD report exists at `reports/<YYYY-MM-DD>/<slug>__DATA.md`.

---

## What this plan deliberately leaves broken

State these in the report rather than fixing them — they are out of scope by your decision, and
three of them were **already broken before Sharadar existed**:

- `src/data_aggregate/` is not updated. The cube will need work; that is your later task.
- The GICS sector gate fails closed (it reads `sector`/`industry_group` off a frame that no longer
  has them), so every sector-gated KPI is all-NaN — **already true today**.
- The employee panel reads `employees` off the wrong frame — **already true today**.
- `revenueGrowth` / `earningsGrowth` have no implementing mechanism — **already true today**.
- The aggregate fingerprint baseline is pinned to a pre-rebuild 237-column parquet and **will not
  notice any of this**.
