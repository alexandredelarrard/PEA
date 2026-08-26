# Sharadar Integration — Plan Index

**Date Created**: 2026-08-26
**Planning Phase**: 2 of 3 (FIC workflow)
**Based on Research**: [2026-08-26-sharadar-fundamentals.md](../../../research/financial-data/2026-08-26-sharadar-fundamentals.md)
**Spec**: [specs/2026-08-26/leverage-shadar-v1.md](../../../../specs/2026-08-26/leverage-shadar-v1.md)
**Next Phase**: Implementation (`/implement`)

Read this file first, then the phase file you are working on. **One phase per session.**
Phases are consecutive: do not start phase N+1 until phase N's verification block passes.

| phase | file | goal | depends on |
|---|---|---|---|
| ⬜ 0 | *(user-owned, see below)* | rename `fundamentals_history` → `fundamentals_history_sec` | — |
| ✅ 1 | [phase-1-extract.md](phase-1-extract.md) | 4 Sharadar tables + fetcher + step + CLI, real rows on DJIA-**30** | phase 0 |
| ⬜ 2 | [phase-2-diagnostics.md](phase-2-diagnostics.md) | measure the acceptance gates **from the DB**; decide the per-field zero rule | phase 1 |
| ⬜ 3 | [phase-3-field-map.md](phase-3-field-map.md) | the 112 → repo-camelCase map + basis translations + TTM build | phase 2 |
| ⬜ 4 | [phase-4-merge.md](phase-4-merge.md) | merged `fundamentals_history` + gap check + override register | phase 3 |
| ⬜ 5 | [phase-5-docs-dod.md](phase-5-docs-dod.md) | docs, AGENTS.md, DoD report | phase 4 |

---

## Scope, stated in one place

**This plan is exactly three things:**

1. **Extract Sharadar** and ingest its tables into Postgres.
2. **Build the merged TTM `fundamentals_history`** — the conjunction of the Sharadar and SEC
   extraction schemes — as the strongest available fundamentals dataset.
3. Leave that table ready for `data_aggregate` and the models to consume **later**.

**This plan is explicitly NOT:**

- ⚠ **Not an extension of the SEC check scheme to Sharadar.** `src/validate/`, its 35-check
  `CHECK_REGISTRY`, the `fundamentals_check` / `fundamentals_check_run` / `fundamentals_check_fix`
  tables and the triage agents **stay pointed at SEC data only**. Nothing in this plan registers a
  check, writes a `fundamentals_check` row, or re-points a validator. Phase 2 is a **standalone
  diagnostic** — it measures three acceptance gates so you can decide whether to buy the Full tier,
  and it produces the per-field zero rule that phase 3 needs. It is not the validator and must not
  be wired into it.
- **Not a `data_aggregate` update.** That is a separate plan with its own scope. No aggregation file
  is touched here, and the cube is not expected to stay green.
- **Not an S&P 500 build.** The free tier cannot do it. Everything is proven on the **DOW** and
  parameterised so the scale-up is a config change.

---

## Phase 0 — the rename, which YOU are doing by hand

You said you would do this yourself. **It is not a global find-replace, and getting it wrong is the
single most expensive mistake available in this plan.** 103 references across 30 `src/` files split
into two groups that must move in *opposite* directions.

### Group A → rename to `fundamentals_history_sec` (the SEC producer and its validator)

| area | files |
|---|---|
| producer | `src/data_extract/utils/fundamentals/build_history.py`, `fetch_fundamentals_sec.py`, `kpi_catalogue.py` |
| step + CLI | `src/data_extract/transformers/step_extract_fundamentals.py`, `src/data_extract/cli.py` |
| validator | all of `src/validate/fundamentals/` — `validator.py`, `substrate.py`, `scope.py`, `clusters.py`, `report.py`, `__init__.py`, `checks/tier1_value.py`, `checks/tier2_series.py` |
| external validators | `src/validate/external/tiingo_comparison.py`, `yahoo_comparison.py`, `src/validate/cli.py` |
| tests | `tests/data_extract/test_build_history.py`, `tests/data_extract/test_amendment_grain.py` |
| registry | `src/data_store/schema.py` — rename the existing `Table` entry |

### Group B → stays `fundamentals_history` (it will read the NEW merged table)

**All 12 `src/data_aggregate/` files**, unchanged: `step_cube_extras.py`,
`step_cube_fundamentals.py`, `step_cube_target.py`, `utils/common/capital.py`, `pit.py`,
`sources.py`, `utils/extras/governance_features.py`, `utils/fundamentals/dividend_features.py`,
`employee_features.py`, `fundamental_features.py`, `utils/target/factors.py`.

⚠ Two files reference **both** meanings and must be read line by line, not swept:
`src/constants/constants.py` and `src/data_extract/utils/structure/def14a_validate.py`.

`fundamentals_reason_codes` keeps pointing at the SEC table (D24), so its `Table` entry needs a
clarifying comment, not a rename.

### The DDL, on the live volume

`sql/schema.sql` is a Postgres **initdb** script mounted at
[docker-compose.yml:64](../../../../docker-compose.yml#L64) — it runs only on an empty data
directory and **is never applied to your long-lived volume**. So the file edit and the live rename
are two separate actions:

```bash
MSYS_NO_PATHCONV=1 docker exec pea_db psql -U alexandre -d pea \
  -c 'ALTER TABLE "fundamentals_history" RENAME TO "fundamentals_history_sec";'
```

plus renaming each `ix_fundamentals_history_*` index, and editing the `-- [extract]` block in
`sql/schema.sql` so a fresh DB matches.

**Tell the implementing agent when phase 0 is done** — phase 1 assumes `fundamentals_history` is a
free name.

---

## Locked decisions

Decided in the grill-me interview on 2026-08-26. **Do not re-litigate any of it during
implementation.** If a decision turns out to be wrong, stop and say so; do not silently choose
differently.

### Vendor, channel, entitlement

| # | decision |
|---|---|
| D1 | **Sharadar Direct**, `https://api.sharadar.com/v1.0`. Personal-use licence, accepted knowingly. Never `data.nasdaq.com`, never the `nasdaqdatalink` / `quandl` libraries. |
| D2 | The Direct API's filing-date column is **`date`**, not `datekey`. `fiscalperiod` **is** present. Both differ from Nasdaq Data Link and from the research doc's notation. |
| D3 | Build and prove on the **~5 years** the current key grants. History depth is a **config knob**, never a hard-coded window, so upgrading to the Full tier is a config change and not a code change. |
| D4 | Universe is an **argument**. Everything runs on the 29 entitled DJIA tickers today and on the S&P 500 the day the key is upgraded. |
| D5 | Ingest **`fundamentals` (SF1) + `tickers` + `actions` + `sp500`**. Not `events`, not `daily`. |
| D6 | Spelling is **`sharadar`** everywhere — table, folder, CLI, constants, env var. |

### Measured entitlement (probed 2026-08-26; authoritative over the research doc)

- **29 of 30 DJIA tickers** entitled; `DOW` denied; every non-DJIA ticker returns **HTTP 403**.
- History **~5 years** — AAPL ARQ earliest `date` = `2021-10-29`, 20 ARQ rows.
- **`bulk/fundamentals?years=5|10|full` → HTTP 404.** Not entitled. Do not build against it.
- `tickers`, `actions`, `sp500`, `events`, `daily` all return rows.
- SF1 is **112 columns**, confirmed from the response header.
- ⚠ The key in `.env` is currently named **`SHARDAR_API_KEY`** (missing the second `a`).

### Storage shape

| # | decision |
|---|---|
| D7 | `fundamentals_sharadar` holds **all 112 columns as delivered**. A mapping mistake must be re-derivable without refetching. |
| D8 | Dimensions stored: **ARQ + ARY + ART only**. MRQ/MRY/MRT mutate in place and would break `diff_against_stored`. |
| D9 | Three separate vendor-shaped side tables: **`sharadar_tickers`, `sharadar_actions`, `sharadar_sp500`**. |
| D10 | Code lives at **`src/data_extract/utils/fundamentals_sharadar/`** plus `transformers/step_extract_fundamentals_sharadar.py`. |
| D11 | **New sibling step** in `StepExtractAllData`, running **before** `StepExtractFundamentals`. |
| D12 | **Per-ticker row API only.** No bulk path, now or later. |
| D13 | Resume via **`store.max_date_by(ticker)` → an explicit `date.gte`**. `-F/--full` re-pulls the whole configured window. |

### The merged table

| # | decision |
|---|---|
| D14 | **Field-block precedence.** Sharadar owns a declared set of columns for all history; SEC owns the rest. No column ever switches source mid-series. |
| D15 | `fundamentals_history` = Sharadar's mapped columns **+ 15 SEC columns + PK**. **No source column, and none of `is_amendment` / `amended_fiscal_end` / `amended_fields`** — those are *pure SEC reconciliation* columns, they stay on `fundamentals_history_sec`, and Sharadar has no amendment events so they would be permanently null here. |
| D16 | Sharadar columns are renamed to **repo camelCase via the field map**; where no repo counterpart exists, Sharadar's own name is kept. |
| D17 | **TTM for duration fields, instantaneous for stock fields** — the current contract, unchanged. TTM is the repo's 4-discrete-quarter sum built from ARQ, **not** Sharadar's `ART`. |
| D18 | **15 SEC-owned columns**: `goodwill`, `intangiblesExGoodwill`, `ppeGross`, `accumulatedDepreciation`, `minorityInterest`, `operatingLeaseLiability`, `financeLeaseLiability`, the 6 regime top-line legs (`premiumsEarned`, `netInterestIncome`, `noninterestIncome`, `netInvestmentIncome`, `realizedInvestmentGains`, `rentalIncome`), `employees`, **and `regime`**. |
| D18b | **`regime` is carried from SEC** into the merged table (decided 2026-08-26). It is the filing's resolution regime, stamped per filing by the SEC facts layer; Sharadar has no regime concept. It is *not* derived from GICS — that would silently change what the column means. Coverage is the SEC roster's, like every other SEC-owned column. |
| D19 | Join on **`ticker`**. The 3 CIK-cutover tickers (APA, GOOGL, ETN) get an explicit continuity test. |
| D20 | **Assert USD**; refuse to write a non-USD filer and log it loudly. |
| D21 | Vendor ratios (`pe`, `roe`, `de`, `ev`, `marketcap`, …) live in the **raw table only** and never reach `fundamentals_history`. |

### Basis forks

| field | decision |
|---|---|
| `ebitda` | **Top-down**: `opinc + depamor`. EBITDA has no accounting standard — SEC Reg S-K Item 10(e) defines it bottom-up purely for non-GAAP reconciliation. Top-down is the operating-business measure and the repo's existing basis. |
| `cash` | **`cashneq + investmentsc`**. Restricted cash absent and accepted. |
| `stockholdersEquity` | **Two explicit columns**: `stockholdersEquity` (Sharadar parent-only, universal, one basis) and `stockholdersEquityInclNci` (parent + SEC `minorityInterest`, 54 tickers). |
| `totalRevenue` (banks) | **Not a blanket rule.** A measured gap check proposes; you adjudicate case by case; the decision lands in an override register. JPM matches exactly; AXP is 6.6–8.1% low. |
| `netIncome` | ← **`consolinc`** (incl. NCI), not `netinc`. Measured on JPM's 11 dates. |
| `sharesOutstanding` | ← **`sharesbas`**, cross-checked against SEC on the overlap. Whether it sums share classes is undocumented — phase 2 measures it. |
| `freeCashflow` | ← **`fcf`**. Measured: `capex` is **negative** and `fcf == ncfo + capex` **exactly**. No reconstruction needed. |

### Governance

| # | decision |
|---|---|
| D22 | Override register: **machine-proposed, human-approved** JSON. The merge only *reads* it and is fully deterministic. |
| D23 | Gap check threshold: **3% relative AND an absolute floor**, on all shared value fields. |
| D24 | **Reason codes stay with the SEC table only.** `fundamentals_reason_codes` keeps pointing at `fundamentals_history_sec`. ⚠ Accepted consequence: `unexplained_null` stops being a universal zero-ceiling gate — the merged table's instrument is the gap check instead. |
| D25 | **The validator stays on the SEC table.** `src/validate/` is not re-pointed at Sharadar in this plan. |
| D26 | **The SEC pipeline is unchanged** — still all 60 fields, triage continues, simply second in precedence. |
| D27 | `sharadar_sp500` is **ingested but `src/utils/universe.py` is not touched.** The survivorship-bias fix is a separate task. |
| D28 | Acceptance gates for the Full-tier purchase decision: **completeness**, **no implausible quarters**, **per-field zero-fill prevalence**. |
| D29 | Sequencing: **extract first, measure from the DB.** |

---

## Measured facts that override the research doc

The research doc was written before the key existed. These were measured on 2026-08-26 against the
live API and supersede it.

1. **`fcf == ncfo + capex`, exactly**, on AAPL ARQ. `capex` is stored **negative**
   (`-2,455,000,000`). So are `ncfi`, `ncff`, `ncfdiv`, `ncfcommon`, `ncfinv`, `ncfdebt`.
   Sign conventions are as-filed, not absolute values.

2. **ΣARQ == ARY to the cent** — AAPL FY2024 and FY2025, CAT FY2024 and FY2025, JPM FY2024 and
   FY2025, all at `+0.000%`. This **confirms the research's §4.3 prediction**: the spec's acceptance
   check #3 (Q4 = FY − 9M) is *tautological* on Sharadar. It can never fail, therefore it can never
   inform you. Phase 2 replaces it per D28.

3. **Zero-fill prevalence, ARQ, 14 DJIA tickers, 279 rows:**

   | field | zeros | field | zeros | field | zeros |
   |---|---|---|---|---|---|
   | `deposits` | 71.3% | `intexp` | **25.4%** | `depamor` | 7.2% |
   | `rnd` | 52.3% | `cor` | 21.5% | `divyield` | 4.7% |
   | `inventory` | 35.8% | `capex` | 14.3% | `dps` | 4.3% |
   | `sbcomp` | 28.7% | `debtc` | 8.0% | `revenue`, `debt`, `sgna`, `receivables`, `cashneq`, `accoci`, `debtnc` | **0.0%** |

   Most of these zeros are *correctly* "not applicable" — a bank has no inventory, a retailer has no
   R&D. **But `intexp = 0` for JPM and GS is provably false**; banks have enormous interest expense.
   This is exactly why the rule is per-field and measured, not global.

4. **`fields=` silently drops an unavailable field** rather than erroring — a typo yields a missing
   column and no warning. Every response header must be validated against the expected set.

5. **`lastupdated` is a per-ticker reprocessing stamp**, not a per-row change stamp (AAPL
   2026-07-31 vs GS 2026-08-04). Do not use it as a global watermark.

---

## Out of scope

- **`src/data_aggregate/`** — you said you will fix it later. This plan touches no aggregation file
  and does not attempt to keep the cube green.
- **`src/validate/`** — not re-pointed at Sharadar (D25).
- **`src/utils/universe.py`** and the survivorship-bias fix (D27).
- **The bulk-zip ingestion path** (D12) — not entitled, and permanently out of scope.
- **`events` and `daily` tables** (D5).
- **The Russell 1000 universe.** Sharadar has no Russell membership table of any kind; the word
  "Russell" appears nowhere in their documentation. Sourcing it is a separate task.
- **Re-fixing the 1,947 open SEC validation clusters** — the SEC pipeline runs as it does today (D26).

---

## Known risks, accepted

| risk | why it is accepted / how it is mitigated |
|---|---|
| `unexplained_null` stops gating the merged table (D24) | Deliberate. The gap check replaces it as the instrument. State it in the DoD report rather than discovering it later. |
| SEC-owned columns are populated for **54 tickers** and NULL for everything else | Structural, not a bug. Stated up front and shown in the phase-4 coverage report. |
| `stockholdersEquityInclNci` exists for 54 tickers only | Which is why it is a *second* column rather than a modification of the first — one basis per column is preserved. |
| Personal-use licence (D1) | Your call, made knowingly. Their §8 restricts publishing conclusions about the data. |
| `ensure_table` infers types from the FIRST frame | Mitigated in phase 1 by hard-casting every value column to `float64` before the first write. An all-`None` object column silently becomes `TEXT` and every later ticker's number is then stored as a string — measured live on `minorityInterest`. |
| `ensure_table` is check-then-create with **no lock** | Threaded writers on a cold table race the `CREATE` and losers silently lose rows. Phase 1 serialises the first write per table, as [edgar_driver.py:94-107](../../../../src/data_extract/utils/common/edgar_driver.py#L94-L107) already does. |
| No documented rate limits | Grepped across all Sharadar docs: zero hits for rate limit / throttle / 429 / concurrent. Phase 1 stays conservative and single-threaded until measured. |

---

## Success criteria for the whole plan

- [ ] `python -m src data_extract fundamentals-sharadar` populates 4 tables with real rows for 29 DJIA tickers.
- [ ] Re-running it is idempotent and pulls only new rows.
- [ ] The phase-2 diagnostic answers all three acceptance gates (D28) with printed numbers.
- [ ] `fundamentals_history` is rebuilt Sharadar-first with the ~14 SEC columns merged in.
- [ ] Every basis fork in the table above is implemented as decided, with a test pinning it.
- [ ] The override register exists, is human-approved, and the merge is deterministic given it.
- [ ] Zero `sqlalchemy` / `pd.read_sql` / `to_sql` outside `src/data_store/` (a test enforces this).
- [ ] Every new test prints a sanity-check conclusion.
- [ ] A DATA DoD report is written.
