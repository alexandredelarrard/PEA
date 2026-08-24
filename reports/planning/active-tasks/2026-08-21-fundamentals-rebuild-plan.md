# Implementation Plan: Fundamentals extraction rebuild

**Date Created**: 2026-08-21
**Planning Phase**: 2 of 3 (FIC Workflow)
**Based on Research**: [2026-08-21-fundamentals-extraction.md](../../research/financial-data/2026-08-21-fundamentals-extraction.md)
**Request**: [specs/2026-08-21_plan_how to extract_Edgartools.md](../../../specs/2026-08-21_plan_how%20to%20extract_Edgartools.md)
**Companion**: [reports/adhoc/fields.md](../../adhoc/fields.md) — every removed field, with concepts,
coverage and its original evidence note, as a prioritised rebuild menu
**Next Phase**: Implementation (`/implement`)

> ## ⚠ THIS FILE IS NOW THE RECORD, NOT THE PLAN
>
> **The active plan is [2026-08-23-fundamentals-rebuild-plan-v2.md](2026-08-23-fundamentals-rebuild-plan-v2.md).**
> Start there. It carries the one-page state of Phases 1-4b, the re-assigned register of all 27
> deferred items, the new **Phase 4c** (deferred resolution-layer fixes), the new **Phase 5b**
> (the validator toolkit, which absorbs this file's Phase 7), and a Phase 9 rewritten around the
> in-sample 26 / out-of-sample 26 rosters.
>
> **What this file is still for**, and why it was not collapsed into the summary: it holds the
> *measured evidence* behind every number v2 quotes — the §Implementation log for Phases 1-4b,
> and the design detail for **Phase 5** (the publication-event grain, the 40-field contract with
> its authorities) and **Phase 6** (the 171-column removal mechanics) that v2 deliberately does
> not restate.
>
> **Superseded and removed from this file** (read v2 instead): Phase 7 → v2 §Phase 5b · Phase 9
> and Phase 10 · Testing Strategy · Risk Mitigation · Success Criteria · Estimated Effort · Open
> items. Phase 4b's option A/B/C question is **closed — A was implemented** (see §Phase 4b —
> implemented).

---

## Overview

Delete the current 7,268-line fundamentals extraction stack and rebuild it around one architectural
change: **stop resolving KPIs by priority-ordered candidate lists, and drive roll-ups from the
filer's own XBRL calculation linkbase**, with the tag list demoted to a fallback.

The research measured that candidate concepts inside one field's list are substitutes only
**30-56%** of the time — because FASB's own element definitions make several of them a superset and
its subset, or two disjoint legs of a total. No amount of tag-list tuning fixes that; it is the
wrong resolution primitive.

Scope confirmed with the user during planning:

| # | Decision | Choice |
|---|---|---|
| 1 | Breadth | **Strict Tiers 1-3** (~39 KPIs) + the calculation inputs those definitions require |
| 2 | Resolution | **`xbrl.calculation_linkbase()`**, tag list as fallback |
| 3 | Substrate | **Per-filing `filing.xbrl()`** (current) — no FSDS/FSNDS ingest |
| 4 | Debt | `totalDebt` = gross debt (ex-lease) **+ finance leases + operating leases** |
| 5 | Cash | cash + equivalents + **restricted cash** + **short-term investments** |
| 6 | Vintage | **Append-only facts; `fundamentals_history` on a publication-event grain** (see §Phase 5.0) |
| 7 | Zero-vs-missing | Compustat-style **`_DC` reason code** per value + "combined into" destination |
| 8 | Validator | **Flag-only + hard-impossible nulling**, two separate layers |
| 9 | `epsDiluted` | **`netIncome_ttm / dilutedShares_ttm`** — not a sum of four quarterly EPS |
| 10 | `researchAndDevelopment` | **Re-added** as a 40th tiered field, regime-gated, one basis universe-wide (§5.2) |
| 11 | Reason codes | **Long side table** `fundamentals_reason_codes`, not 40 `_DC` companion columns |
| 12 | `fundamentals_audit.py` | **Deleted.** The external-ground-truth check moves *inside* `FundamentalsValidator`; callable from several entry points |

### Verified during planning (not inherited from the research)

- `xbrl.calculation_linkbase()` exists in the installed **edgartools 5.44.1** with the documented
  columns. On JPM's latest 10-K: **465 arcs, `menucat=="Statements"` → 108**, and
  `RevenuesNetOfInterestExpense ← +1 InterestIncomeExpenseNet, +1 NoninterestIncome` resolves exactly
  as §3.2 claims. Timing: **`xbrl()` 3.7 s, `calculation_linkbase()` 0.0 s** — the linkbase is
  *free* on the substrate we already pay for. This is what makes decision #2 affordable under #3.
  (The method's own docstring says `menucat` is `'S'`/`'D'`/`'N'`; that is wrong — the real values
  come from FilingSummary.xml and are `Statements`/`Details`/`None`. Filter on `"Statements"`.)
- `edgar/ttm/calculator.py::_derive_q4_from_fy` derives **Q4 = FY − YTD_9M first**, and only falls
  back to `FY − (Q1+Q2+Q3)`. This repo has **only the fallback**. That is the single highest-value
  thing to steal (see Phase 4).
- Live DB: `fundamentals_facts_legacy` = **7,776,870 rows**, `fundamentals_history_legacy` =
  **27,602 rows**. The rename kept the 7.8M raw tag-agnostic audit dump; the 2.37M-row old *pipeline
  output* is gone. Consequence: we keep an excellent offline audit substrate **and** a
  239-column regression baseline, but cannot diff against the old facts-layer output.
- Column census of `sql/schema.sql`'s `fundamentals_history`: **239 columns, 143 consumed
  downstream, 96 read by nothing.** Of the 143: 46 tiered, 14 computed in-code, **78 extra extracted
  fields**. Strict Tiers 1-3 drops those 78 → Phase 6 exists to repair the consumers.
- `data/gaps/` and `data/sec_bulk_cache/` **do not exist**, which darkens ~28 existing tests.

---

## Current State Analysis

### What gets deleted (7,268 lines)

| File | Lines | Why |
|---|---|---|
| `fetch_fundamentals_edgar.py` | 1,232 | priority-list resolver; 8 hardcoded threads; PK cannot upsert the live table |
| `fundamentals_tags.py` | 1,433 | 213 fields / 376 concepts / 6 dicts; ~80% evidence commentary; belongs in `configs/*.json` |
| `fundamentals_periods.py` | 1,140 | good engineering, wrong entry point — the survivable parts are re-homed, not kept in place |
| `fetch_fundamentals.py` | 906 | the dead companyfacts-JSON path; only `_spine_grid`/`_assemble_base`/`_derive_history` still called |
| `fundamentals_derive.py` | 198 | thin adapter over the above |
| `fundamentals_employees.py` | 180 | 10-K body-text headcount — **re-homed, not deleted** (`employees` is Tier 2) |

Untouched (unrelated to the fundamentals tables): `fetch_earnings_surprises.py`,
`fetch_financial_statements.py` (→ `pension_facts`), `fetch_financial_notes.py` (→
`notes_num`/`notes_text`).

### The five defects being designed out

1. **Candidate lists treated as substitutes.** Revenue: only **52 of 443 tickers (12%)** use one
   concept across history; 56% use three or more; two thirds of switches have no overlap period.
2. **`shortTermDebt` discards a real component.** `LongTermDebtCurrent` and `ShortTermBorrowings`
   are disjoint legs whose sum is `DebtCurrent`. **2,017 cells across 111 tickers** tag both with no
   total; the discarded leg is the larger one **54.4%** of the time.
3. **The Q4 footing check is vacuous.** All 203,798 Q4 rows are derived as `FY−(Q1+Q2+Q3)`, so
   `Q1+Q2+Q3+Q4==FY` passes 99.73% *by construction*.
4. **The TTM staircase.** **1,622 of 26,242 consecutive pairs (6.2%)** have `totalRevenue` exactly
   unchanged (APA 100%, XOM 36%) → `revenueGrowth` is exactly 0 for three quarters in four.
5. **Expected absence and coverage regression look identical.** 78 tickers legitimately never file
   `AssetsCurrent` (Reg S-X Rules 9-03 / 7-03; `17 CFR 210.1-02(bb)(1)(i)`). A null count cannot
   tell that apart from a broken extractor.

---

## Desired End State

- `fundamentals_facts` — long, append-only, accession-grain. One row per
  `(ticker, accession, field, fiscal_year, fiscal_period, duration_type)`, carrying the resolution
  provenance (`resolution_method`, `source_concept`, `roll_up_children`, `dc_code`).
- `fundamentals_history` — wide, `(ticker, as_of)`, **~71 columns** (40 tiered KPIs + 13 calculation
  inputs + 10 derived + 5 keys + `regime` + provenance), down from 239. **One row per publication
  event** (§5.0): an amendment appends a row at its own filing date and never edits the original, so
  a daily training set contains only values that were genuinely knowable on their own date.
- `fundamentals_quality` — the validator's report table. Flag-only.
- A nightly incremental run measured in **minutes**, and a full rebuild measured in **hours**.
- Every KPI carries a written definition in `configs/fundamentals/fundamentals_kpis.json`, sourced from FASB
  element documentation / Reg S-X / ASC — never from a generic source.

---

## Out of Scope

- **SEC bulk FSDS / FSNDS ingest** (`cal.tsv`, `pre.txt`, `iprx`/`durp`/`datp`) — decision #3 keeps
  the per-filing substrate. Revisit only if the full-rebuild wall-clock becomes binding.
- **Tier 4 as scored KPIs.** 6 of the 7 Tier-4 fields survive as *calculation inputs* because the
  chosen `totalRevenue` definition literally cannot be computed for banks/insurers/REITs without
  them. `researchAndDevelopment` is the one deliberate loss — see **Known gaps**.
- **A vintaged `fundamentals_history`** (decision #6 chose latest-known).
- Nareit FFO/AFFO/EBITDAre, CET1, PV-10/SMOG, reserve data — none has a us-gaap element; all need a
  separate source.
- Leases as a *scored* KPI family. `operatingLeaseLiability` reaches 486 tickers but cov is only
  **.513** (ASC-842-era only) — it enters `totalDebt` per decision #4 and carries a regime flag for
  the structural FY2019 discontinuity, but is not z-scored on its own.
- The `epsDiluted` TTM convention is **flagged, not silently changed** — see **Open items**.

---

## Part A — Demolition

### Phase 1: Baseline capture, then delete ✅

**Goal**: make the rebuild falsifiable before removing the thing it replaces.

> **DONE 2026-08-21** on branch `feature/fundamentals-rebuild`. See §Implementation log
> for what the plan did not anticipate.

**Changes**:

1. Baseline snapshot (throwaway script under the scratchpad, not committed):
   - [x] Dump `fundamentals_history_legacy` to parquet — the 239-column, 27,602-row regression
         baseline. Every Phase 9 comparison is against this.
   - [x] Record per-field non-null counts and per-ticker row counts.
   - [x] Keep `fundamentals_facts_legacy` (7.8M rows) in place — it is the offline audit substrate
         Phase 7's thresholds get calibrated on, at zero network cost.

2. Delete, in dependency order:
   - [x] `src/data_extract/utils/fundamentals/fetch_fundamentals.py`
   - [x] `src/data_extract/utils/fundamentals/fetch_fundamentals_edgar.py`
   - [x] `src/data_extract/utils/fundamentals/fundamentals_derive.py`
   - [x] `src/data_extract/utils/fundamentals/fundamentals_tags.py`
   - [x] `src/data_extract/utils/fundamentals/fundamentals_periods.py`
   - [x] `src/utils/fundamentals_tag_ledger.py` (its `detect_tag_switch_breaks` idea is reborn in
         Phase 7 as a check, not a standalone CSV writer)
   - [x] `src/validate/fundamentals_validation.py`, `analyze_history.py`,
         `run_fundamentals_integration_report.py` (rebuilt in Phase 7)
   - [x] `src/validate/fundamentals_audit.py` — **delete** (decision #12). Its Tiingo→Yahoo
         ground-truth comparison is not lost: the *check logic* moves inside `FundamentalsValidator`
         as an external-source check (Phase 7), so there is one place that knows how to judge a
         value. `src/utils/tiingo_comparison.py` and `yahoo_comparison.py` **survive as I/O
         adapters** — they fetch, they do not judge — with their field lists trimmed in Phase 6.
   - [x] `tests/utils/test_fundamentals_audit.py` (102 lines) — deleted with it; its intent moves to
         Phase 7's validator test.

3. Delete the tests bound to deleted code (~5,100 lines):
   - [x] `test_fetch_fundamentals.py`, `test_fetch_fundamentals_edgartools.py` (1,329),
         `test_fundamentals_amendments.py`, `test_fundamentals_coverage_gaps.py` (716),
         `test_fundamentals_diagnostics.py`, `test_fundamentals_expanded.py`,
         `test_fundamentals_fiscal_calendar.py`, `test_fundamentals_fiscal_period.py` (610),
         `test_fundamentals_missing_fix.py`, `test_fundamentals_plausibility.py`,
         `test_fundamentals_reconstruction.py`, `test_fundamentals_sector_coverage.py`,
         `tests/utils/test_fundamentals_tag_ledger.py`
   - [x] **Keep and re-point**: `test_fundamentals_point_in_time.py` (deliberately red — it is the
         acceptance criterion for the leak fix), `test_step_extract_fundamentals.py`,
         `test_fundamentals_employees.py`, `tests/utils/test_fundamentals_audit.py`,
         `tests/data_aggregate/test_fundamental_features.py` (Phase 6 rewrites its expectations).

4. `src/constants/constants.py` — remove only what becomes unreferenced:
   - [ ] `FUNDAMENTALS_FACTS RECONCILIATION` block (:577-600), `FUNDAMENTALS_DISCONTINUITY_MIN/MAX`
         (:641-642), the `XBRL TAG-SWITCH LEDGER` block (:645+),
         `FUNDAMENTALS_FINDINGS_RANKED_FILENAME` (:759) — *if and only if* Phase 7 does not re-adopt
         them. Sequence the constants pass **after** Phase 7, not here.
   - [x] `FUNDAMENTALS_FORMS` stays (still the form list).
   - [ ] Risk zone → propose the diff and get approval before editing.

**Verification**:
- [x] `rtk grep -rn "fundamentals_tags\|fundamentals_periods\|fetch_fundamentals" src/ tests/` returns
      nothing outside the new modules.
- [x] `"$PY" -m pytest tests/ -q --collect-only` collects with zero import errors.
- [x] The baseline parquet exists and has 27,602 rows × 239 columns.

**Rollback**: everything is in git; `fundamentals_*_legacy` are untouched in Postgres.

---

## Implementation log

### Phase 1 — done 2026-08-21

**Measured, matching the plan exactly**: baseline parquet `27,602 rows × 239 columns`
(491 tickers, `as_of` 2009-07-31 .. 2026-08-10, 2 all-null fields, 68 fields >90% covered);
`fundamentals_facts_legacy` = **7,776,870** rows, left untouched. Artifacts in the session
scratchpad: `fundamentals_history_legacy.parquet`, `baseline_field_nonnull.csv`,
`baseline_ticker_rows.csv`.

**Deleted**: 10 `src/` modules + 19 test files = **~12,900 lines**. Test collection
**969 → 720, zero import errors**.

**Open item #1 CLOSED — edgartools 5.51.0.** `poetry.lock` had *already* been resolved to
5.51.0 (with `httpxthrottlecache` 0.3.5 → 0.6.1, the HTTP-cache fix); only the venv was
stale at 5.44.1. Installed the locked versions and **re-confirmed the plan's gating
measurement on 5.51.0**:

| claim | result |
|---|---|
| `calculation_linkbase()` exists, documented columns | yes — `concept, concept_taxonomy, parent_concept, parent_taxonomy, weight, role_uri, role_short, menucat, is_abstract, label` |
| JPM latest 10-K arc count | **465** (accession `0001628280-26-008131`, FY2025, filed 2026-02-13) |
| `menucat` census | `Details` 353, **`Statements` 108**, NaN 4 |
| `RevenuesNetOfInterestExpense` roll-up | **+1.0 `NoninterestIncome`, +1.0 `InterestIncomeExpenseNet`** |
| cost | `xbrl()` **5.20 s**, `calculation_linkbase()` **0.006 s** (0.1%) |

Decision #2's premise holds on the upgraded library. Two corrections for Phase 3:
- The columns are **`parent_concept` / `concept`**, not `parent`/`child`.
- **`weight` sign census on the Statements slice: 84 × `+1.0`, 24 × `-1.0`.** 22% of arcs
  are contra-accounts, so "preserve the weight sign" is load-bearing, not defensive.

#### What the plan did not anticipate

1. **`detect_level_outliers` had to be re-homed, not deleted.** It lived in
   `analyze_history.py` (deleted) but `tiingo_comparison.py` / `yahoo_comparison.py` — which
   the plan keeps — both import it. Moved to **`src/utils/outliers.py`**, beside the MAD
   kernel it already shares, with `LEVEL_OUTLIER_COLUMNS` declared once. Phase 7's
   `level_outlier` check imports it from there, which is the "one implementation, many
   callers" split decision #12 asks for. `detect_source_tag_misalignment` was NOT ported —
   Phase 7 rebuilds it as the `tag_switch_break` check.
2. **`data/sec_bulk_cache` holds 0 companyfacts JSON files.** So `build_ticker_history`'s
   callers were already permanently skipping — which confirms `fetch_fundamentals.py` was
   dead, but the *module-level imports* still broke collection. Consequences:
   - `tests/data_aggregate/test_composites_config.py::real_panel` **re-pointed at the real
     `fundamentals_history` table** (projected `store.load`) instead of rebuilding from
     companyfacts JSON. Strictly better: it now exercises the input the cube actually reads,
     and it starts working the moment Phase 5 populates the table.
   - **`pipeline_fingerprint.compute()`, `pipeline_fingerprint_baseline.json` (262 KB) and
     `test_refactor_regression.py` retired** (user-approved). That guard could never run
     again on any machine, and its baseline asserted an unreproducible
     `extract.fundamentals_history` key. `aggregate_fingerprint.py` supersedes it and keeps
     importing `frame_digest`, so `pipeline_fingerprint.py` survives as the shared kernel
     (`frame_digest` + `synthetic_prices`, 109 → ~50 lines). **§6.4's baseline is therefore
     `aggregate_fingerprint_baseline.json` only.**
3. **Two more test files had to go than §6.3 listed**, both because their imports *and*
   their substrate were gone: `test_capital_and_restatements.py` (456 lines — §6.3 already
   called it "largely deletable"; the restatement half is owed to Phase 8's
   `test_amendment_grain.py`) and `test_sector_gates_and_tags.py` (357 — §6.3 says
   "rewrite", so Phase 6 owes it back). Also deleted beyond the plan's list:
   `test_fundamentals_edgartools_integration.py` (imported 4 deleted modules) and
   `test_analyze_history.py` (its intent → Phase 7's `level_outlier` / `tag_switch_break`).
4. **The plan contradicted itself on `tests/utils/test_fundamentals_audit.py`** — listed
   both as deleted (item 2) and as "keep and re-point" (item 3). Followed decision #12 and
   **deleted** it; item 3's mention is a pre-decision-#12 leftover.
5. **Wiring temporarily disconnected, with markers.** `FORM_REGISTRY["fundamentals"]`, the
   `fundamentals` CLI command and `StepExtractFundamentals`' first two calls are removed
   and comment-marked with the phase that re-adds them (3 / 5 / 5).
   `test_form_registry.py::test_registry_forms_match_constants` was hand-listing four
   registry keys — made **data-driven** (`EXPECTED_FORMS`) and it now prints which expected
   keys are unregistered, so the gap stays visible instead of passing silently.
6. **Tests kept but gated on the phase that will satisfy them**, via
   `pytest.importorskip` pointing at the future module — so they turn green by *arrival*
   rather than needing to be remembered: `test_fundamentals_employees.py` checks 1 and 4
   (→ `periods.py`, `build_history.py`), and
   `test_fundamentals_point_in_time.py::test_as_of_never_precedes_fiscal_end_unit`
   (→ `build_history.py`). Checks 2/3 of the employees test stay live.
7. **`EMPLOYEES_FIELD` re-homed** from `fundamentals_tags.py` into
   `fundamentals_employees.py`, which is headcount's only producer.
8. **Deferred as the plan sequences them**: the `constants.py` pass (after Phase 7) and the
   stale `schema.py` / `sql/schema.sql` comments (Phase 3). These are the only two places a
   deleted module name still appears, both in prose.

#### Phase 1 verification

- [x] No import of a deleted module anywhere under `src/`, `tests/`, `scripts/`.
- [x] `pytest tests/ --collect-only` → **720 collected, 0 errors**.
- [x] `pytest tests/data_extract tests/utils tests/data_store` → **294 passed, 16 skipped**.
- [x] Baseline parquet exists: 27,602 × 239.

### Phase 2 — done 2026-08-21

**The census matches the contract exactly**: 11 Tier-1 + 12 Tier-2 + **17** Tier-3
(16 + `researchAndDevelopment`, carried as `tier: 3` + `regime_gated` so nothing special-cases
a fourth tier value) + 13 calculation inputs = **53 fields, 40 scored, 49 extracted**.

#### ✅ 2026-08-21, second pass — all 17 UNVERIFIED fields CLOSED

A [second research pass](../../research/financial-data/2026-08-21-fundamentals-extraction-part2.md)
closed every one against FASB's own 2025 taxonomy (`us-gaap-doc-2025.xml`,
`us-gaap-ref-2025.xml`, `us-gaap-lab-2025.xml`, `us-gaap-2025.xsd`) and eCFR Reg S-X.
**`authority: "UNVERIFIED"` now appears nowhere in the catalogue**: 43 of 53 fields cite a primary
source directly, 10 inherit one, 0 are placeholders. Configs moved to
**`configs/fundamentals/*.json`**.

A third citation grade was added, because "sourced" turned out not to be binary:
**`authority_caveat`** marks a VERIFIED field with one weaker sub-claim — typically an ASC
paragraph whose *number* is primary (FASB's reference linkbase) while its *prose* could only be
read in a secondary reproduction, `asc.fasb.org` being login-walled (403). 7 fields carry one:
`stockholdersEquity`, `dilutedShares`, `stockBasedComp`, `ppeNet`, `goodwill`,
`intangiblesExGoodwill`, `accountsReceivable`. `EXPECTED_UNVERIFIED` in the test is now an empty
frozenset, so the test's job flips from *reporting* the gap to *guarding against regressing* into it.

**Three of the research's own conclusions changed once measured against the tag ledger**:

1. **The `incomeTaxExpense` "bug" is not a bug.** The research called
   `IncomeTaxExpenseBenefitContinuingOperations` a tag that "does not exist anywhere in the 2025
   FASB taxonomy" and flagged it implementation-blocking. Correct *for 2025* — but the ledger shows
   **276 undimensioned facts across 82 tickers, window 2011-06-30 .. 2013-12-31**. It is a
   **deprecated pre-2014 element**, not a typo, and since the rebuild reads history back to 2009 it
   is a real second candidate for those tickers' early years. **Kept**, with the window recorded, so
   nobody deletes it for "not existing". Your "drop if never used" condition did not fire. Also
   recorded: the similarly-named 2025 elements that are *not* substitutes —
   `FederalIncomeTaxExpenseBenefitContinuingOperations` (839 facts / 97 tickers) is a jurisdictional
   leg, `…AdjustmentOfDeferredTaxAssetLiability` (289 / 73) is a reconciliation item.
2. **`interestExpense`'s bank roll-up needed different elements than the research proposed.** It
   suggested `InterestExpenseDeposits + InterestExpenseBorrowings` and explicitly asked for a
   coverage measurement first. Measured on the 14 covered banks: **`InterestExpenseBorrowings`
   reaches only 2 of 14** — that roll-up fails for 12 of them. The elements filers actually use for
   Rule 9-04 captions 7 and 8 are `InterestExpenseShortTermBorrowings` (10/14) and
   `InterestExpenseLongTermDebt` (11/14), beside `InterestExpenseDeposits` for caption 6 (13/14).
3. **`Assets` is present for 442 of 442, not 441 of 441** — the register's denominator was stale
   (see below).

#### Your four decisions, as implemented

| # | decision | what shipped |
|---|---|---|
| 1 | **`ppeNet`** → option (b): net out `FinanceLeaseRightOfUseAsset` where separately tagged | Implemented **with a linkbase condition**, because the tag alone is not enough: a filer that shows the ROU asset as its *own* balance-sheet line also tags it, and subtracting there removes an amount that was never inside `ppeNet`. So the adjustment applies only where the linkbase does **not** declare the ROU asset a *sibling* of `PropertyPlantAndEquipmentNet` — parent/child means folded in, sibling means separate. Exactly what the linkbase-driven architecture is for. **Measured: 182 of the 417 `ppeNet` tickers (44%) tag it, so the fix is detectable for 44% and invisible for 56%** — the field gets closer to one basis without ever becoming exactly one basis, and ASC-842-era only (2017+), so pre-2019 needs no adjustment at all. |
| 2 | **`interestExpense`** → option (b): drop the generic elements, build from Reg S-X 9-04 captions | Bank override added: `total_concept = InterestExpenseOperating` (caption 9, "Total interest expense"), legs = Deposits + ShortTermBorrowings + LongTermDebt (captions 6/7/8), `InterestExpenseDebt` in `never_use`. **One deviation, stated plainly: `InterestExpense` is retained as the *second* fallback for banks.** Dropping it literally would blank **2011-2024 for all 14 banks** — the two elements *partition* the timeline with no overlap (`InterestExpenseOperating` is 14/14 but only from 2024-06-30, 188 facts; `InterestExpense` is 14/14 from 2011-09-30 to 2024-06-30, 1,064 facts). That is a clean taxonomy handover, not a coverage preference, so it is never *preferred* — only reached where caption 9 does not exist. |
| 3 | **`incomeTaxExpense`** → check the ledger, drop if never used | Checked; it *was* used (see above). Retained and documented. |
| 4 | **AVB, EA, EQR** → fine to be absent | Recorded as `_unclassifiable_tickers` with your decision. They have facts but no `sp500_tickers` row, so no regime can be assigned. **One consequence worth enforcing in code**: a regime join must *skip* an unmatched ticker, never default it to `industrial` — defaulting would add three unclassified names to the 340-ticker industrial denominator and shift every industrial rate in the register. |

#### Register + roster corrections applied

- **Scope 441 → 442.** The README figure was off by one against the file's own `by_regime` block,
  which was always right and sums exactly to 442 (14+16+6+19+28+18+1+340).
- **`bank.capex`'s "intermittent" label replaced with the measured three-way split.** Not one
  pattern but three: **always** (AXP, C, COF, FITB, HBAN — 5 banks, every covered year, not
  intermittent at all) · **sporadic** (CFG 1/13 years, KEY 3/16, TFC 6/16) · **never** (BAC, JPM,
  MTB, PNC, RF, SYF — the 6 that make the 0.43 rate). `expected_absent` stays `true` — the
  mixed-TTM risk in the sporadic group is real and silent — but a downstream consumer should now
  read AXP/C/COF/FITB/HBAN capex as usable and CFG/KEY/TFC as genuinely spotty.
- **`force_regime._why`: 37 → 36 live tickers** (6+9+9+12); prose-only slip, the ticker lists were
  right.
- **GICS Financials resolves to FIVE destinations, not four** — `hybrid` (BRK-B) is a legitimate
  fifth bucket. Reconciles exactly: 16+17+6+36+1 = 76.
- **`PLD` added to the `real_estate` roster** (Industrial REITs), which had 23 entries against a
  live membership of 24 — in a file whose own README calls the lists "the LIVE membership, not an
  illustration". Fittingly, Industrial REITs is the exact sub-industry the first research flagged
  as the one a contiguous-code-range scan drops.
- The register's **arithmetic was independently reproduced cell-for-cell** by the second pass
  (bank capex 0.43/6, utility+energy currentAssets 0.00/0, insurer SG&A 0.69/11 with the exact
  AFL/AIG/AIZ/HIG/TRV list, real_estate opInc 0.21/4 as ARE/CPT/DOC/O, industrial R&D 0.44/150).
  Only the prose was wrong.

#### Historical record — the original 17 UNVERIFIED fields

26 cite a primary source directly, 10 **inherit** one, **17 do not have one**:
`accountsPayable` · `accountsReceivable` · `accumulatedDepreciation` · `dilutedShares` ·
`goodwill` · `incomeTaxExpense` · `intangiblesExGoodwill` · `interestExpense` · `inventory` ·
`minorityInterest` · `ppeGross` · `ppeNet` · `pretaxIncome` · `retainedEarnings` ·
`sellingGeneralAdmin` · `stockBasedComp` · `stockholdersEquity`.

All 17 are now closed (see above). Kept here because the *reason* they were open is worth
remembering: the first research established verbatim authority for the fields it *investigated*
(revenue, debt, capex, D&A, cash, R&D, the required-vs-elective captions) and never touched the
ordinary Tier-2/3 balance-sheet lines. Rather than invent plausible citations, each carried an
`authority_note` naming the document that would close it — and the prediction that
"`us-gaap-doc-2025.xml` closes most of them in one pass" proved exactly right. Two of the notes'
hypotheses were *wrong* in a way only the primary source could show: `inventory`'s guessed ASC
330-10-50 tie does not exist (the operative citation is Reg S-X 5-02(6)(c) itself), and
`interestExpense`'s note treated the element family as a plain superset/subset problem when the
real finding is that `InterestExpense` has **no Reg-S-X anchor at all**. Marking them UNVERIFIED
rather than guessing is what made both discoverable.

Two authority mechanisms were added because the plan's binary (sourced / UNVERIFIED) did not fit:
- **`authority_inherits_from`** — a tier-0 calculation input has no primary source *of its own*;
  it exists because another field's cited definition requires it. Naming that field is the
  honest citation, and the test walks the chain. 10 fields. Two of them (`effectiveTaxRate`,
  `basicShares`) inherit from an UNVERIFIED parent, and the test prints that explicitly.
- **`override_reason`** on an exception cell — see `bank.capex` below.

#### Substrate finding that changes Phases 7 and 9

**`fundamentals_facts_legacy` covers only 445 of the 500 universe tickers, and the 58 with no
facts at all are alphabetically clustered** — essentially everything from U/V/W onward (UNH,
UNP, UPS, USB, V, VLO, VRT, VZ, WFC, WMT, WY …) plus recent additions (GEHC, GEV, KVUE, RDDT,
SOLV, VLTO). That is a **truncated backfill**, not genuine absence. Consequences:

- **Phase 7**: calibrating thresholds on this substrate is still right, but the sample is
  alphabetically biased and has never seen the U-W tail. Say so when reporting a threshold.
- **Phase 9**: four slice tickers — **USB, VLO, VRT and V** — have *no* legacy facts, so there
  is no baseline to regress them against. Expect it; do not read it as a rebuild failure.
- It also invalidated my first absence measurement, which counted those 58 as "absent" and
  reported bank `currentAssets` absence as 16/16 including two tickers with no facts at all.

#### The expected-absence register is measured per REGIME, not per GICS sector

The research measured absence by GICS *sector*; the regimes are finer, and "Financials" spans
bank + insurer + broker-dealer + the forced-industrial fee businesses, so the sector matrix
could not be translated. Re-measured directly off the 7.8M-fact dump, scoped to the 441 covered
tickers (share of a regime's tickers with **no** undimensioned fact for the concept, ever):

| regime | n | currentAssets | grossProfit | COGS | opInc | inventory | SG&A | R&D | capex(PP&E) |
|---|---|---|---|---|---|---|---|---|---|
| bank | 14 | **1.00** | 1.00 | .93 | .93 | **1.00** | 1.00 | **1.00** | .43 |
| insurer | 16 | **1.00** | .94 | .94 | .75 | **1.00** | .69 | **1.00** | .56 |
| broker_dealer | 6 | .83 | 1.00 | 1.00 | 1.00 | 1.00 | .83 | .83 | .50 |
| real_estate | 19 | .84 | .79 | .63 | .21 | .89 | .95 | **1.00** | .74 |
| utility | 28 | **.00** | .89 | .64 | .07 | .54 | **.96** | .93 | .18 |
| energy | 18 | **.00** | .78 | .28 | .22 | .17 | .56 | .67 | .44 |
| industrial | 340 | .05 | .36 | .16 | .11 | .28 | .30 | .44 | .16 |

Four results worth stating:
- **`Assets` is present for 441 of 441 tickers, in every regime** — the strongest possible
  confirmation that `totalAssets` is the one balance-sheet total with no regime variant.
- **bank and insurer are 100% absent for `currentAssets`, `currentLiabilities`, `inventory` and
  `R&D`.** Exactly what `17 CFR 210.1-02(bb)(1)(i)` predicts. Perfectly structural.
- **utility and energy are 0% absent for `currentAssets`** — they file *classified* balance
  sheets. So they must NOT be grouped with banks/insurers for that exception. The plan's field
  list implied a coarser grouping; this corrects it. What a utility *does* lack is the Rule 5-03
  SG&A caption (.96) — its income statement runs on O&M and fuel.
- **`bank.capex` is only .43 absent, not ~1.0.** This **refines the plan**: over full history 8
  of 14 covered banks *do* tag `PaymentsToAcquirePropertyPlantAndEquipment` at some point, so
  the plan's "not reliably reconstructible" is about **intermittency, not absence** — and the
  absence *rate* is the wrong statistic. Kept `expected_absent: true` with a written
  `override_reason`, because an intermittently-tagged capex yields a TTM that silently mixes
  tagged and untagged quarters: freeCashflow would be **wrong** rather than missing. Bank FCF
  stays null by design.

#### Two corrections to the plan's regime traps

1. **There are no mortgage REITs in the current S&P 500.** The plan flags "mortgage REITs are
   40204010 under Financials" as a trap; verified against the live roster, that sub-industry has
   **zero members**, so it cannot mis-route anything today. The rule is encoded anyway (→
   `broker_dealer`, since a mortgage REIT holds a levered securities portfolio, not property) so
   a future index addition is handled.
2. **`sp500_tickers` carries `sub_industry`**, so the traps are encoded against the *stored
   names* rather than GICS code ranges — which also fixes the plan's own warning that the
   equity-REIT block is not contiguous. **7 sub-industries force a regime, covering 44 live
   tickers**: Insurance Brokers (6), Financial Exchanges & Data (9), Transaction & Payment
   Processing (9), Asset Management & Custody Banks (12), Telecom Tower REITs (3), Data Center
   REITs (2), Timber REITs (1) — all → `industrial`. `Asset Management & Custody Banks` is the
   dangerous one: BNY and STT genuinely do take deposits, but they share a sub-industry with
   BLK and TROW. Routed to Article 5 as the safer uniform default, with the **role-URI branch
   still overriding** whenever those two ship a deposit-based role — which is precisely why role
   URI is checked *first*. Also added beyond the plan: `Consumer Finance` (AXP, COF, SYF) → bank,
   and `Investment Banking & Brokerage` → broker_dealer.

#### Other deviations

- **`_derived_columns` block added to the KPI JSON.** The 10 derived history columns are part of
  the contract (a `feeds` reference must be able to name one, and Phase 5 needs the formulas in
  one place) but are *not* catalogue fields — nothing resolves them against a concept. The
  block also records the 4 dropped derived fields and the `insufficient_quarters` TTM rule.
- **`revenueGrowth`/`earningsGrowth` are declared on a 365-DAY `as_of` offset**, not a 4-row
  offset, per §6.1's `infer_yoy_periods` finding — encoded now so Phase 5/6 cannot forget.
- **The three JSON filenames are still module constants in `kpi_catalogue.py`**, not
  `constants.py`. The plan directs them to `constants.py`, which is a named risk zone; folded
  into the single batched risk-zone approval at the start of Phase 3.
- **No local `us-gaap-doc-2025.xml`**, and a filing's `element_catalog` exposes *labels only* —
  no `documentation` string, and `balance` is `None` even for `us-gaap_Assets`. So authority
  strings could not be auto-sourced. **This also matters for Phase 7**: the `sign_convention`
  check's `TAG.crdr` sign oracle is **not** available from the per-filing substrate and needs
  the taxonomy download or the FSNDS `cal.tsv` `negative` column.

#### Phase 2 verification

- [x] 8/8 tests pass; every entry has tier/kind/sign/unit/definition/authority.
- [x] Tier census asserted == 11/12/17/13; duplicate top-level keys checked in the file **text**
      (a duplicate JSON key is silently dropped by the parser, not raised).
- [x] Cross-references resolve; all 13 inputs declare what they `feed`; all 4 derived/ratio
      fields carry a formula and no concept list; all 49 extracted fields have a resolution route.
- [x] The measured traps are asserted to be *in the contract*, not just in the plan:
      `shortTermDebt` two-leg sum, the bank revenue roll-up + gross-interest ban, capex's
      superset + MAA's IPR&D element banned for both capex and R&D, `epsDiluted` derived with
      the as-reported tag as cross-check only, `depAmort` aggregate, non-additive fields.
- [x] Suite collects: **725 tests, 0 errors** (720 − 3 retired guard + 8 new).

---

### Phase 3 — done 2026-08-22

Batched risk-zone approval obtained for all four items: `schema.py` + `sql/schema.sql`,
`configs/fundamentals/fundamentals_kpis.json`, `constants.py`, `configs/configs.yml`.
Also settled: **derived quarters stay in memory**; `fundamentals_facts` is strictly as-filed,
so the publication-event grain and the no-leakage property are provable rather than asserted.

#### The plan's three resolution routes had to become six, plus a non-route

| `Resolution.method` | when | vs the plan |
|---|---|---|
| `linkbase_total` | a catalogue candidate is reported AND the filer's linkbase declares it | as planned |
| **`linkbase_root`** | **no candidate reported — discover the top node structurally** | **new; see below** |
| `linkbase_sum` | the declared legs are reported, no total | as planned |
| **`field_sum`** | **composed of other CATALOGUE FIELDS** | **new**; `totalDebt` and `ppeNet` declare `roll_up.sum` over field names, not concepts, so they cannot resolve against a concept at all |
| **`tag_primary`** | **the catalogue's TOP-PRIORITY reported concept, with no linkbase arc for it** | **new — split out of `tag_fallback`; see the gate below** |
| `tag_fallback` | every linkbase route failed **and** the winner was not the top-priority concept | the plan's meaning, now narrowed to it |
| **`unresolved`** | **no value by any route; `dc_code` says why** | **new**; not a route, and excluded from any route rate |

#### The APA root cause is sharper than the research recorded — and it kills the tag list outright

The research said `RevenuesAndOther` is "absent from `fundamentals_tags.py`". Measured on the
live 10-K, the reason is stronger: **`RevenuesAndOther` is a COMPANY EXTENSION,
`apa:RevenuesAndOther`** — $9.220B FY2025, declared in APA's own linkbase as the +1.0 child of
the pretax node. Meanwhile `us-gaap:Revenues` exists in the same filing with **84 facts, every
one of them dimensioned** (segment detail), i.e. zero undimensioned facts.

So APA could never have been fixed by adding a tag: the tag is company-private. That is the
single strongest available argument for decision #2, and it was not available before Phase 3.
**DTE is the same failure with a different signature**: it declares
`RegulatedAndUnregulatedOperatingRevenue` as a *parentless root* of its income-statement role,
because its `OperatingIncomeLoss` arc set carries only the `-1 CostsAndExpenses` side — a
permitted filer omission. Route 2 handles both.

#### A general "climb to the parent" was tried, measured, and REMOVED

The obvious reading of "walk from the filer's declared parent" is to climb from a matched
candidate while the parent is a pure `+1` aggregation and is not another field's declared
total. Both conditions verified individually on real filings (XOM's revenue stops at the
netting pretax node; `ppeNet` needs the second condition to stop at `Assets`). **Together they
are still nowhere near sufficient** — measured over 9 tickers the climb produced:

| field | climbed to | should be |
|---|---|---|
| `cash` | `AssetsCurrent` | `CashAndCashEquivalentsAtCarryingValue` |
| `shortTermDebt` | `LiabilitiesCurrent` / `LiabilitiesAndStockholdersEquity` | `DebtCurrent` |
| `ppeNet` | `AssetsNoncurrent` | `PropertyPlantAndEquipmentNet` |
| `netIncome` | `ComprehensiveIncomeNetOfTax` | `ProfitLoss` |

The catalogue names ~60 concepts against a balance sheet of hundreds, so it cannot supply a
dense enough boundary — the whole upper balance sheet is unclaimed and purely additive.
**The climb was deleted and replaced by opt-in structural discovery on `totalRevenue` alone**,
which is the only field with the problem (the research measured 56% of tickers using 3+ revenue
concepts; balance-sheet concepts are stable). Keeping the general climb would have swapped one
silent-wrong-number mechanism for another.

#### Three real bugs the tests caught, all ordering or namespace

1. **A roll-up LEG could satisfy `linkbase_total`.** With `DebtCurrent` unreported,
   `LongTermDebtCurrent` won route 1 and `ShortTermBorrowings` was never added — **the exact
   §2.3 defect, reintroduced through the back door.** Legs are now excluded from the total
   route and stay eligible only as a last resort.
2. **`dei:` candidates never matched.** The catalogue namespaces cover-page tags while the
   linkbase and the reported-concept set are keyed bare, so
   `dei:EntityCommonStockSharesOutstanding` — the **only summable share tag for a multi-class
   issuer** — was skipped on 5 of 6 filings in favour of `CommonStockSharesOutstanding`, a
   single class. This is the multi-class NULL defect; it now has a regression test.
3. **Linkbase presence outranked candidate priority.** XOM declares
   `CommonStockSharesOutstanding` under `CommonStockSharesIssued`, so route 1 preferred it over
   the higher-priority cover-page tag. Priority now dominates; the linkbase decides *between*
   totals and discovers unnameable ones, it does not demote the catalogue's first choice.

#### Measured: resolution across the six regimes

Every regime resolves to the concept its own filer declares, on the right template:

| ticker | regime | route | concept | FY2025 revenue |
|---|---|---|---|---|
| XOM | energy | `linkbase_total` | `us-gaap:Revenues` | $332.238B |
| APA | energy | **`linkbase_root`** | **`apa:RevenuesAndOther`** | **$9.220B** (was 0) |
| JPM | bank | `linkbase_total` | `us-gaap:RevenuesNetOfInterestExpense` | $182.447B |
| DTE | utility | **`linkbase_root`** | `us-gaap:RegulatedAndUnregulatedOperatingRevenue` | $15.814B |
| MAA | real_estate | `linkbase_total` | `us-gaap:Revenues` | $2.209B |
| MET | insurer | `linkbase_total` | `us-gaap:Revenues` | $77.084B |

JPM CY2024 = **$177.556B**, matching the research's independently measured figure exactly.

**Fallback rate — the plan's ~20% architecture gate. The first measurement was against a
badly-specified metric, and the fix was to split the label, not to argue for exclusions.**

Pooled into one `tag_fallback` label the rate read **27.8%** — over the gate. Defending it
required arguing that two populations "should not count": cover-page items (`sharesOutstanding`
is a `dei:` tag, and a us-gaap *calculation* linkbase describes face statements, so it can never
contain one) and reason-coded absences (`totalLiabilities` is elective — Rule 5-02 has no "Total
liabilities" caption — so APA and DTE simply never tag it). Both exclusions are genuinely
correct, but **a metric that needs them argued for is the wrong metric.**

Split three ways, the gate applies literally and needs no exclusions at all. Tier-1 fields over
the six regime filings: `linkbase_total` 31 · `tag_primary` 12 · `field_sum` 6 · `linkbase_root`
2 · `tag_fallback` **1** · `unresolved` 2 (not a route). **Genuine `tag_fallback` = 1.9% of the
52 resolved** — an order of magnitude under the 20% gate.

`tag_primary` means the catalogue's *first choice* was taken and the filer's linkbase simply has
no arc for it. That is expected and benign: a calculation arc exists **only** where a filer
declares a total-and-components relationship, so a leaf (`goodwill`, `inventory`) or a cover-page
tag can never have one. It is not evidence about this design; `tag_fallback` is.

#### Entity scoping, measured

MAA: 5,324 facts → **567 consolidated (10.6%)**; 315 `LegalEntityAxis` facts under
`maa:LimitedPartnershipMember` (a company extension, so no member deny-list could catch it);
all `xbrli:identifier` values are the parent's CIK, confirming the identifier is useless as a
discriminator. Share count after scoping = **116,901,020**, the parent's.
Southern Company: 11,425 facts → **813**, with **8,638** `LegalEntityAxis` occurrences (the
research measured 3,579 — it has grown) and again a single identifier.

#### Deviations from the plan's Phase-3 text

- **`fetch_fundamentals_sec.py` is 241 lines, not "< 350"** — because it reuses the existing
  `run_edgar_fetch` driver rather than reimplementing the walk. That driver already provides
  the manifest window, the accession dedup set, the thread pool **and the `store.ensure_table`
  create-lock the plan asks for**, so the fix for that race is inherited rather than rewritten.
- **Resume is accession-based, not `max_date_by`.** `existing_filings`' docstring records that
  a per-ticker max-date cutoff was tried and reverted: it never re-checks a scanned range, so a
  filing missed by a prior bug stays missing forever. The plan's `max_date_by` suggestion is
  superseded by the repo's own settled convention.
- **`Context` exposes no config directory**, so the catalogue loads from its `./configs`
  default (the same value the CLI defaults to). Adding one means touching `context.py`, a risk
  zone, for no functional gain.
- **A `fundamentals-facts` CLI command ships now**; the combined `fundamentals` command still
  waits for Phase 5's history build, as the plan sequences it.
- **`Catalogue.regime_for_sub_industry` was reading only one GICS level.** The regimes config
  declares membership at whichever level is natural — `bank`/`insurer` by sub-industry,
  `real_estate` by industry group, `utility`/`energy` by **sector** — so reading only
  `sub_industry` returned None for *every* energy, utility and REIT ticker, which then fell
  through to the `industrial` default and would have read a utility against Rule 5-03. Replaced
  by `regime_for_gics(sector, industry_group, sub_industry)`, most-specific level first. A
  Phase-2 gap that only extraction could surface.

#### Carried into Phase 5

**`totalLiabilities` is `not_disclosed` for APA and DTE.** Rule 5-02 has no "Total liabilities"
caption, so `us-gaap:Liabilities` is elective and these filers omit it. The plan already
specifies the repair — *"where untagged, `totalAssets − equityInclNCI`"* — but that is a
cross-field derivation, so it belongs to the history layer, not the facts layer. Flagged rather
than left silent.

#### Phase 3 verification

- [x] Synthetic known-truth: declared total beats disagreeing legs; a partial leg set emits
      nothing; `-1.0` weights preserved; `never_use` refuses MAA's IPR&D-tagged capex; bank
      capex is `not_applicable` even when the tag is present.
- [x] Real data, six regimes: each resolves to its filer's own declared concept, on the right
      regime template. **APA non-zero and sourced from an extension total.**
- [x] MAA's LP excluded by axis; share count is the parent's.
- [x] `sql/schema.sql`'s DDL is now internally consistent — all six PK columns are declared
      (previously `field` and `duration_type` were named in the PK and absent from the column
      list).
- [x] `tests/data_extract/test_linkbase_resolution.py` **13 passed**,
      `test_entity_scope.py` **8 passed**. Suite collects **746 tests, 0 errors** (725 + 21).

---

### Phase 3b — robustness sweep (26 tickers x full 2011-2026 history) ✅

**Why.** Phase 3's evidence was 6 filings, ~3 annual periods each, **zero quarters**, nothing
before FY2023. Every failure mode that would actually invalidate the architecture is temporal.

**Method.** Every 10-K and 10-Q filed since 2011-01-01, all 49 extracted fields resolved per
filing, ledger cached to parquet. Amendments excluded — the amendment grain is Phase 5's
acceptance test, and a Part-III-only 10-K/A carries cover-page facts alone, which would inflate
`unresolved` with rows that say nothing about resolution quality. (edgartools' `form=`
prefix-matches, so `/A` must be filtered explicitly — found the hard way in the smoke test.)

**The roster.** 26 tickers, each buying a distinct edge case, none for coverage padding:

| ticker | regime | what it is here to prove |
|---|---|---|
| AAPL | industrial | Sep FY; the ASC-606 concept switch |
| CSCO | industrial | 52/53-week FY; the 2017 53-week Q4 |
| KR | industrial | Jan fiscal year-end |
| XOM | energy | frozen-TTM baseline (36%) |
| **APA** | energy | **extension revenue total; the 0-revenue chain** |
| EOG | energy | per-company capex elements |
| VLO | energy | the D&A tie-break (~200x on Valero) |
| JPM | bank | `RevenuesNetOfInterestExpense` |
| BAC | bank | the FY2023 restatement trap (98,581 → 102,769) |
| MTB | bank | 28% frozen TTM |
| USB | bank | no legacy facts — the Phase 9 blind spot |
| MET | insurer | the LDTI 2021-01-01 break |
| PGR | insurer | P&C; tagged ratio concepts |
| AFL | insurer | third insurer |
| MAA | real_estate | Up-C `LegalEntityAxis` extension member; IPR&D-tagged capex |
| SPG | real_estate | unclassified balance sheet |
| AMT | **industrial** | Telecom Tower REIT → industrial, the GICS regime trap |
| DTE | utility | **parentless revenue root** |
| SO | utility | six registrant CIKs |
| NEE | utility | `RegulatoryAssets` frame absent |
| **ETN** | industrial | **`totalRevenue == 0`** (16 legacy rows) |
| **VRT** | industrial | **`totalRevenue == 0`** (5 legacy rows) |
| SWKS | industrial | FY2020 tags a 370-day **and** a 97-day fact as `fp='FY'` |
| BRK-B | hybrid | hybrid regime; multi-class; no `AssetsCurrent` |
| **GS** | broker_dealer | **the only broker-dealer** — a regime the plan's own Phase 9 slice never exercises |
| META | industrial | edgartools #691: 0 undimensioned shares-outstanding facts 2012-2026 |

Two additions beyond the plan's Phase 9 slice: **GS**, because `broker_dealer` is one of the
eight regimes and the 32-ticker slice contains none; and **META**, the known upstream
shares-outstanding bug, because `sharesOutstanding` is where a namespace defect was just fixed.
**SMCI/ADM dropped** (amendments belong to Phase 5) and **EQR dropped** (no `sp500_tickers` row,
so no regime can be assigned — see the unclassifiable-ticker rule).

**Cost.** ~1,700 filings. Committed as `tests/data_extract/test_linkbase_history.py`, gated:
`FUNDAMENTALS_HISTORY_SWEEP=full` runs all 26, otherwise a 3-ticker subset (APA / DTE / SWKS —
the extension total, the parentless root, the fiscal-calendar edge) keeps the default suite fast.

**Findings — swept 2026-08-22: 26 tickers, 1,544 filings, 144,190 ledger rows.**
Full audit: [2026-08-22-phase3b-resolution-audit.md](../../research/financial-data/2026-08-22-phase3b-resolution-audit.md).

**The numbers are NOT right.** Five defects, none findable from Phase 3's 6-filing snapshot:

| # | defect | blast radius |
|---|---|---|
| 1 | **`menucat` is null on every pre-2015 filing**, so `menucat=="Statements"` discards the WHOLE linkbase | **418 of 1,544 filings (27.1%)**; linkbase share **0.9% in 2011-2014** vs ~70% from 2016 — the rebuild is switched OFF for four of fifteen years |
| 2 | `discover_root`'s parentless-root fallback accepts any all-positive root | **74 revenue rows** on `Assets`, `LiabilitiesAndStockholdersEquity`, cash-flow totals, `ComprehensiveIncomeNetOfTax`, `NoninterestExpense` — APA yields **revenue of −$467M** |
| 3 | a filer tagging `Revenues = 0` still wins the priority walk | **ETN 14 rows, VRT 24 rows still zero — the plan's headline acceptance criterion is UNMET** |
| 4 | bank revenue lands on a post-provision / single-leg basis | MTB 110 rows on `InterestIncomeExpenseAfterProvisionForLoanLoss`, 6 on `NoninterestIncome` |
| 5 | `shortTermDebt` subtracts lease legs unconditionally | **158 negatives** across 10 tickers, worst −$893M |

Plus, as coverage rather than resolver defects: `totalLiabilities` never resolves for APA /
DTE / EOG / ETN / VLO (elective tag, Tier-1 hole); META has no `sharesOutstanding`
(edgartools #691, confirmed live); and **APA carries only 22 filings, none before 2021** — a
CIK change to Apache Corp that a by-ticker walk reports silently.

**What holds**: the APA extension-total repair (62 rows, $4.308-12.132B); one stable regime
per ticker across 15 years, with AMT correctly `industrial` and BRK-B `hybrid`; quarters
finally present (21,553 `quarterly` + 8,420 `ytd6` + 7,953 `ytd9`) against ZERO in the Phase 3
measurement; only 4 `other`-shaped durations in 144k rows; and 10-Q linkbases are **not** worse
than 10-K's (52.0% vs 47.0% `linkbase_total`), refuting the main quarterly worry.

---

## Part B — The rebuild

### Phase 2: The KPI catalogue ✅

> **DONE 2026-08-21.** `configs/fundamentals/fundamentals_kpis.json` (53 fields),
> `configs/fundamentals/fundamentals_regimes.json` (8 regimes), `configs/fundamentals/fundamentals_exceptions.json`
> (measured per-regime absence register), `src/data_extract/utils/fundamentals/kpi_catalogue.py`,
> `tests/data_extract/test_kpi_catalogue.py` — **8 tests, all passing.** See §Implementation log.

**Goal**: one machine-readable, human-auditable file that *is* the contract. No tag list in Python.

**Changes**:

1. `configs/fundamentals/fundamentals_kpis.json` — new. One entry per field:
   ```json
   {
     "shortTermDebt": {
       "tier": 2, "kind": "instant", "sign": "non_negative", "unit": "USD",
       "definition": "Gross debt due within one year, EXCLUDING lease obligations.",
       "authority": "FASB us-gaap-doc-2025: DebtCurrent = 'Amount of debt AND lease obligation, classified as current.' LongTermDebtCurrent and ShortTermBorrowings are disjoint legs whose sum is DebtCurrent.",
       "roll_up": {"sum": ["LongTermDebtCurrent", "ShortTermBorrowings"]},
       "total_concept": "DebtCurrent",
       "total_adjustment": {"subtract": ["FinanceLeaseLiabilityCurrent", "OperatingLeaseLiabilityCurrent"]},
       "fallback_concepts": ["LongTermDebtCurrent", "ShortTermBorrowings"],
       "regimes": {"bank": {"add": ["AdvancesFromFederalHomeLoanBanksCurrent"]}},
       "components": ["longTermDebtCurrentOnly", "shortTermBorrowingsOnly"]
     }
   }
   ```
   - [x] `roll_up` is what the calculation linkbase is *checked against*, not a substitute for it.
   - [x] `authority` is mandatory and must quote a primary source. Reviewable by a human.
   - [x] Anything the research could not establish gets `"authority": "UNVERIFIED"` and is surfaced
         by a schema test — no silent guesses.

2. `configs/fundamentals/fundamentals_regimes.json` — new. The regime → template map, driven by the
   **statement role URI** the filer actually used, with GICS only as a tiebreak:
   - [x] `sfp-dbo` 108000 → bank · `sfp-ibo` 108200 → insurer · `sfp-sbo` 112000 → broker-dealer ·
         `sfp-clreo` 110000 / `sfp-ucreo` 110200 → real estate · `soi-int` 132001 · `soi-ins` 136000
         · `soi-reit` 145000 · `scf-dbo` 160000.
   - [x] The four verified GICS traps encoded as explicit exceptions: **mortgage REITs are 40204010
         under Financials**; the equity-REIT run is **not contiguous** (`601025` Industrial REITs);
         **Insurance Brokers 40301010** (MMC/AON/AJG) and **Financial Exchanges 40203040** /
         **Payments 40201060** (ICE/CME/V/MA) are Article 5 fee businesses — **do not route GICS 40
         to a bank template**; tower/data-center/timber REITs file like industrials.
   - [x] Hybrids (BRK) get `regime: "hybrid"` and are excluded from regime-relative scoring rather
         than forced into a template.

3. `configs/fundamentals/fundamentals_exceptions.json` — new. The "expected absence" register the user asked
   for: `(regime, field) -> expected_absent`, so Phase 7 can tell structural absence from
   regression. Seeded from §2.8's measured matrix and justified by
   `17 CFR 210.1-02(bb)(1)(i)`.

4. `src/data_extract/utils/fundamentals/kpi_catalogue.py` — new, small. Loads + validates the three
   JSONs once, exposes typed accessors. Constants (paths, the JSON filenames) → `constants.py`.

**Verification**:
- [x] `test_kpi_catalogue.py`: every entry has tier/kind/sign/definition/authority; every concept
      named is a real us-gaap 2025 element; no field appears in two tiers; no `UNVERIFIED` authority
      without a matching entry in **Open items**.
- [x] Print the field count per tier and assert 40 tiered (11+12+16+1 R&D) + 13 inputs.

**Estimated effort**: 1-1.5 days (the definitions are the work, not the plumbing).

---

### Phase 3: The facts layer — linkbase-driven resolution ✅

**Goal**: `fundamentals_facts`, resolved from the filer's own declared roll-up.

**Changes**:

> **DONE 2026-08-22.** `xbrl_linkbase.py`, `entity_scope.py`, `fetch_fundamentals_sec.py`
> (241 lines), `Tables.fundamentals_facts` + a repaired `sql/schema.sql` block,
> `tests/data_extract/test_linkbase_resolution.py` (13 tests) and `test_entity_scope.py`
> (8 tests) - **21 tests, all passing.** See §Implementation log.

1. `src/data_extract/utils/fundamentals/xbrl_linkbase.py` — new. The core primitive.
   - [x] `statement_arcs(xbrl) -> DataFrame` — `calculation_linkbase()` filtered to
         `menucat == "Statements"`. **Preserve the `weight` sign** (`-1.0` is a real contra-account
         rollup, not noise).
   - [x] **Shipped as `resolve_field`, with FIVE routes, not three** (`linkbase_root` and
         `field_sum` added; the generic parent-walk was measured to over-climb and was
         REMOVED -- see the log). Walk from the filer's declared parent.
         Three outcomes, all recorded in `resolution_method`:
         `linkbase_total` (the filer declares a total; use it) →
         `linkbase_sum` (no total, but the leaf children are declared; sum them with weights) →
         `tag_fallback` (no linkbase for this concept; use `fallback_concepts` in priority order).
   - [x] This is what would have caught **APA**: its linkbase declares `RevenuesAndOther` as the
         pretax revenue parent, a concept absent from the old tag list, so the resolver fell through
         to an element APA tags as literally `$0.00`, every quarter, for 19 rows.

2. `src/data_extract/utils/fundamentals/entity_scope.py` — new. **Filter on the AXIS, not the
   member.**
   - [x] Take dimensionally-unqualified (default-member) facts. A fixed us-gaap member list cannot
         catch a company-extension member — MAA scopes its LP with `maa:LimitedPartnershipMember`,
         and Southern Company carries **six registrant CIKs / 3,579 `LegalEntityAxis` occurrences**
         in one instance with all identifiers = parent.
   - [x] One documented exception hook for regulatory capital (which is *only* reachable
         dimensioned) — declared, not used in this pass.

3. `src/data_extract/utils/fundamentals/fetch_fundamentals_sec.py` — new, replaces the 1,232-line
   fetcher. Target: **< 350 lines**.
   - [~] **DEVIATED: resume is accession-set only, NOT `max_date_by`** -- `existing_filings`
         records that a per-ticker max-date cutoff was tried and reverted (it never re-checks a
         scanned range). Per-ticker filing walk over `FUNDAMENTALS_FORMS`, resuming from
         `store.max_date_by(Tables.fundamentals_facts, "ticker")` → `resume_since`, plus the
         accession set already stored. Never a full read.
   - [x] Worker count from `configs/configs.yml`, **not hardcoded** (the old `DEFAULT_WORKERS = 8`
         was a literal).
   - [x] Inherited from `run_edgar_fetch`'s create-lock. `store.ensure_table` race: the table must be **created before** the thread pool starts —
         check-then-create with no lock silently loses rows on a cold table.
   - [x] Append-only: original and `/A` amendments coexist as separate rows, never overwritten.

4. Schema:
   - [x] `Tables.fundamentals_facts` in `src/data_store/schema.py` — new column set, PK
         `(ticker, accession_number, field, fiscal_year, fiscal_period, duration_type)`.
   - [x] `sql/schema.sql` — rewrite the block. **The current DDL is internally inconsistent**: its PK
         names `field` and `duration_type`, neither of which appears in its own column list
         (`schema.sql:328-359`). Fix that here.
   - [x] Risk zone → propose the DDL diff and get approval. **Granted 2026-08-22** for all four.

**Verification**:
- [x] Known-truth fixture: a synthetic linkbase where the total and the legs disagree → assert
      `linkbase_total` wins and `resolution_method` says so.
- [x] Real-data: JPM resolves `totalRevenue` via `RevenuesNetOfInterestExpense` and **not** via
      `Revenues`; APA resolves via `RevenuesAndOther` and is **not** 0; XOM via `Revenues`;
      MET/MAA/DTE per the §3.2 table. Print the resolved concept per ticker.
- [ ] **STILL OPEN — covered only by a SYNTHETIC fixture so far.** Folded into the Phase 3b
      robustness pass (24 tickers x full 2011-2026 history).
      `shortTermDebt` on the 111 tickers that tag both legs with no total: assert the stored value
      equals the **sum**, and that both components are populated.
- [x] MAA: assert the LP's `LegalEntityAxis` facts are excluded and `sharesOutstanding` is the
      parent's.

**Estimated effort**: 3-4 days.

> **Phase 3b (robustness) IN PROGRESS.** Everything above was measured on **6-7 tickers x the
> LATEST 10-K only** -- ~3 ANNUAL periods each, **zero quarterly periods, nothing before FY2023**,
> 54 resolutions from 6 filings. Far too thin to trust, because every mechanism most likely to
> break resolution is temporal: pre-2013 filings may ship no calculation linkbase at all; ASC 606
> (FY2018) forces the revenue concept to switch mid-history; ASC 842 (FY2019) makes the `ppeNet`
> lease adjustment appear mid-series; and a 10-Q's linkbase is smaller than a 10-K's, while
> quarters are what the entire TTM layer is built from.
>
> Phase 3b sweeps **26 tickers x every 10-K and 10-Q from 2011 to today** (~1,700 filings, all 49
> extracted fields) and asserts the result in `tests/data_extract/test_linkbase_history.py`.
> Roster and findings below.

---

### Phase 3c — fix the defects the 26-ticker sweep found ✅

Audit: [2026-08-22-phase3b-resolution-audit.md](../../research/financial-data/2026-08-22-phase3b-resolution-audit.md).
26 tickers x 1,544 filings x 2011-2026, then a **clarification pass on 2026-08-22** that
re-derived each defect from the filings rather than from the ledger alone. That pass
**overturned three of the five fixes originally proposed here**, and the post-fix re-sweep
then **found four more defects the fixes themselves introduced or exposed** (3c.8). What
changed, and why, is recorded in each item.

**Ordering rule**: 3c.1 first, then re-measure. It changes which arcs every other route
sees, so every downstream number stays provisional until it lands.

---

#### 3c.1 `statement_arcs`: the arc filter is lossy in BOTH eras — **critical** ✅

`menucat` comes from FilingSummary.xml. Two separate failures, not one:

1. **It is `None` for 100% of arcs on 418 of 1,544 filings (27.1%)** — all of 2011 through
   mid-2015. `menucat == "Statements"` yields zero arcs and the resolver silently degrades
   to the tag list for four of fifteen years (linkbase share **0.9% in 2011-14** vs ~70%
   from 2016).
2. **It is also lossy where it IS populated.** Measured on 28 filings: a role-URI test is
   *never* missing an arc menucat keeps (**0 missing, every filing**) and recovers **99
   arcs menucat drops**. Every one of those is a genuine face-statement role that
   FilingSummary failed to categorise — a filer splitting one statement's calculations over
   several roles, where only the first gets categorised:

   | ticker | role the role-URI test recovers | its `menucat` |
   |---|---|---|
   | APA 2022 | `STATEMENTOFCONSOLIDATEDOPERATIONS` | `Uncategorized` |
   | AMT 2026 | `CONSOLIDATEDSTATEMENTSOFOPERATIONS_1` | `NaN` |
   | SO 2026 | `ConsolidatedStatementsofIncomeSouthernCalculation` | `NaN` |
   | SPG 2026 | `StatementConsolidatedStatementsOfOperationsAndComprehensiveIncomeCalc2` | `NaN` |
   | MAA 2026 | `StatementConsolidatedBalanceSheets2` | `NaN` |

   Four of the five are **income-statement** roles. This is very likely part of why revenue
   needed structural discovery at all.

**But role-URI matching is a naming heuristic over filer-authored strings, and the naive
pattern is dangerous.** The audit's first pattern (`disclosure|details|polic|table|
parenthetical|note`) admitted **136 extra arcs on AFL and 203 on MTB**, because:

  * footnote roles end in the SINGULAR `...Detail`, which `details` does not match; and
  * **Reg S-X Schedule I / II parent-company-only condensed statements**
    (`ScheduleIiCondensedFinancialInformationOfRegistrantCondensedBalanceSheets...`) contain
    none of those words. Those are *parent-only* financials that look exactly like face
    statements — admitting them would corrupt consolidated numbers in a way nothing
    downstream would catch.

  That is the answer to "any reason not to use it": yes, and it is the Schedule I/II trap.
  The hardened pattern below was calibrated against `menucat` as ground truth and scores
  **0 missing / 99 extra, all 99 verified legitimate**.

- [x] Filter arcs by `menucat == "Statements"` **OR** a hardened role-URI test — a UNION,
      not a fallback. Each test is lossy in a way the other is not; neither admitted a
      junk arc in the measured sample.
- [x] Hardened exclusion pattern (calibrated, do not loosen):
      `detail | disclosure | polic | parenthetical | schedule | tables?$ | uncategor | highlight`
      — case-insensitive. `schedule` is load-bearing: it is what excludes Reg S-X
      Schedule I/II parent-only statements.
- [x] Record which test admitted each arc (`arc_filter` = `menucat` / `role_uri` / `both`),
      so the two populations stay separable.
- [x] Test: `test_statement_arcs_is_the_union_of_menucat_and_the_role_test`. It also pins
      the direction the measured sample never exercised — a Consolidated Schedule of
      Investments is a FACE statement that `menucat` keeps and the role test drops, which
      is why `menucat` is unioned rather than replaced.

**MEASURED (re-sweep, 26 tickers x 1,552 filings):** linkbase share of valued resolutions

| year | before | after |
|---|---|---|
| 2011 | 0.9% | **73.7%** |
| 2012 | 0.9% | **72.3%** |
| 2013 | 0.9% | **71.2%** |
| 2014 | 0.8% | **69.7%** |
| 2015 | 37.5% | **70.5%** |
| 2022 | 64.1% | **67.6%** |
| 2016-2021, 2023-2026 | 67.4-71.4% | +0.0 to +0.3pp |

Pooled **51.4% -> 69.7%**. `tag_fallback` 6.87% -> 6.84%, well inside the 20% gate. The
role test admits a mean of **88.5 arcs per filing in 2011-2015** where `menucat` admits
10.3, and still adds arcs on **13.3% of modern filings** (144 of 1,085) — confirming the
second failure was real and not an artefact of the small clarification sample.

#### 3c.2 `discover_root`: rank the roots, do not just filter them — **critical** ✅

Measured: **74 revenue rows** on `Assets` (18), `LiabilitiesAndStockholdersEquity` (16),
cash-flow period-increase totals (24), `ComprehensiveIncomeNetOfTax` (14),
`NoninterestExpense` (2). APA yields revenue of **-$467M**.

**The clarification pass found the actual mechanism, and it is not "no constraint".**
`roots_with_children()` returns roots in **arc insertion order** and `discover_root` takes
the first that qualifies. It is a lottery, not a choice. The proof: on DTE's 2020-04-28
10-Q the correct answer `RegulatedAndUnregulatedOperatingRevenue` was present, reported and
all-positive — and lost, because the cash-flow root appeared earlier in the arc list.

**It also found that the fix approved on 2026-08-22 — "require an income-statement role" —
is not implementable as the code stands.** `ArcGraph._role_of` indexes only the `concept`
column, and a parentless root appears ONLY in `parent_concept`. So `role_of()` returns
`None` for **every root this route ever considers**, and a role test would reject all of
them. Indexing the parent side as well populates it correctly and the test then separates
the cases cleanly (verified on all 10 offending filings):

| root | `period_type` | role once parent-indexed | verdict |
|---|---|---|---|
| `Assets`, `LiabilitiesAndStockholdersEquity` | **instant** | ...BalanceSheet / FinancialPosition / FinancialCondition | rejected twice over |
| `CashCash...PeriodIncreaseDecrease...` | duration | ...CashFlows | rejected by role |
| `ComprehensiveIncomeNetOfTax` | duration | ...ComprehensiveIncome | rejected by role |
| `NoninterestExpense` | duration | **...StatementOfIncome / OfEarnings** | **survives — see below** |
| `RegulatedAndUnregulatedOperatingRevenue` (DTE) | duration | ...StatementsOfOperations | **kept, correct** |

- [x] Index `role_uri` from the `parent_concept` side too in `ArcGraph._role_of`. Without
      this nothing else in 3c.2 can work.
      Test: `test_a_parentless_root_has_a_role_at_all`.
- [x] Require the root's `period_type == "duration"`, from the new
      `entity_scope.duration_concepts`. (`balance` was tested as a second axis and
      **rejected**: it is empty for GS's `RevenuesNetOfInterestExpense` and DTE's
      `RegulatedAndUnregulatedOperatingRevenue`, so requiring `credit` would reject the
      correct answers.)
- [x] Require the root's role to be an income-statement role
      (`operations|income|earnings`, and NOT `comprehensive` — see 3c.8 for the third
      exclusion the re-sweep forced).
- [x] **Rank the survivors instead of taking arc order** — the actual defect. Widest
      aggregation first, name as the tie-break, so the same filing always resolves the same
      way. Test: `test_root_discovery_rejects_the_balance_sheet_and_cash_flow_roots`.
- [x] `NoninterestExpense` added to the bank regime's `never_use` (and, after 3c.8, to
      `NOT_A_TOP_LINE`, because GS hits it as a `broker_dealer`). `resolve_field` now
      `bare()`s `never_use` before passing it to `discover_root`, matching `_candidates`.

**MEASURED:** all 74 known-bad rows are gone — `Assets` (18), `LiabilitiesAndStockholdersEquity`
(16), `ComprehensiveIncomeNetOfTax` (14) and the `CashCashEquivalentsRestrictedCash...`
totals (24) no longer appear as `totalRevenue` for any ticker. DTE **gains** rows rather
than losing them (226 -> 300 valued, now on `RegulatedAndUnregulatedOperatingRevenue` /
`Revenues` / `SalesRevenueNet` / `UtilityRevenue` and nothing else).

#### 3c.3 Zero guard — **high, and the acceptance criterion itself was wrong** ✅

43 rows carry `totalRevenue == 0`. The clarification pass asked how a genuine zero is told
from a fake one and found the discriminator is not subtle: **in every case the concept is
zero in EVERY period of the filing** (`vmin == vmax == 0`). There is no
one-bad-quarter-among-good-ones case in the data. So the test is a **filing-level** one and
resolution stays period-agnostic — the design worry recorded on 2026-08-22 does not apply.

More importantly, the three cases do not want the same answer:

| case | what the filing says | right answer |
|---|---|---|
| **ETN** (14 rows) | `Revenues` = 0, no linkbase arc; `SalesRevenueNet` = **$20.9-22.6 bn** sits right there, undimensioned | skip the zero, take `SalesRevenueNet` |
| **APA** (4 rows) | `RevenueFromContractWithCustomerIncludingAssessedTax` = 0 via **`linkbase_total`**; `apa:RevenuesAndOther` = **$596M-3,874M** in the same filing | skip the zero, let route 2 find the extension |
| **VRT** (24 rows) | `Revenues` = 0 is the **only** revenue-like fact in the filing; the largest duration facts are `ProceedsFromIssuanceInitialPublicOffering` $690M and G&A of $123k | **KEEP the zero — it is correct** |

**VRT's zeros are not a defect.** Those 2018-2020 filings are the *GS Acquisition Holdings*
blank-cheque shell, pre-merger with Vertiv: trust-account dividend income, a $690M IPO, and
genuinely no revenue. The plan's headline acceptance criterion "zero zero-revenue rows" is
therefore **itself wrong** and has been restated.

Note APA's zero arrives by `linkbase_total`, so the guard **cannot** be confined to the
priority walk as previously written — the filer declared the zero concept in its own
linkbase. Applying it to `available` instead covers every route at once:

- [x] `entity_scope.zero_only_concepts` computes, once per filing, the concepts that are
      exactly 0 in every period they report. `resolve_field` resolves with those withheld.
- [x] If the field is then unresolvable, it **re-resolves with them restored** and sets
      `Resolution.zero_only_retained`, surfaced in the `adjustment` JSON column
      (`adjustment::jsonb ? 'zero_only_retained'`). No schema change, fully auditable.
      Route 3 deliberately still reads the full `available`: a genuinely-zero LEG
      contributes 0 to a weighted sum and must not break it.
- [x] Restated acceptance criterion: *no zero-revenue row unless the filer reports no other
      non-zero revenue concept anywhere in the filing.*
      Test: `test_a_zero_in_every_period_loses_to_a_real_number_but_survives_alone`.

**MEASURED:** APA 4 zeros -> **0** (all 71 rows now `apa:RevenuesAndOther`, min $596M).
ETN 14 -> **0**. `zero_only_retained` fires on 105 valued rows (0.10%), 6 tickers, 7 fields,
and **not one of them is non-zero** — the flag never fires spuriously. VRT is covered by
3c.8: the guard worked, but a non-operating item then won the vacated slot.

#### 3c.4 Bank revenue basis — **high** ✅

MTB resolves `totalRevenue` to `InterestIncomeExpenseAfterProvisionForLoanLoss` (110 rows)
and `NoninterestIncome` alone (6). The first is net interest income **after** the credit
provision — a different basis from Rule 9-04 caption 10, not comparable with JPM's
`RevenuesNetOfInterestExpense`; the second is one leg of two.

- [x] Added both, plus `NoninterestExpense` (3c.2), to the bank regime's `never_use` with
      the Reg S-X reasoning. `NoninterestIncome` stays a child of the regime's `roll_up.sum`
      — `never_use` filters candidates and discovered roots, not roll-up legs — so banning
      it as a standalone total is exactly what pushes MTB onto the correct two-leg basis.
- [x] Re-measured MTB against JPM / BAC / USB.

**MEASURED:** MTB's 110 post-provision rows and 6 single-leg rows are gone. 96 of them now
resolve by `linkbase_sum` on `InterestIncomeExpenseNet + NoninterestIncome` — the Rule 9-04
basis — moving median revenue from **$1.588 bn to $2.335 bn**, i.e. the previous number was
understating the bank's top line by ~32%.

#### 3c.5 `shortTermDebt` lease subtraction — **medium** ✅ (rule changed twice; see 3c.8)

**158 negative values across 10 tickers** (worst -$893M). Verified in the clarification
pass: **all 158 carry a lease subtraction and not one negative occurs without one** — so
lease subtraction is the sole cause, with no second mechanism hiding behind it.

**But that same check killed the fix originally proposed here.** "Only subtract from
`DebtCurrent`" is wrong: `DebtCurrent` accounts for **51 of the 158** negatives on its own
(9% of its 546 subtracted rows), and it is the concept that most *should* be safe. The
resolved concept is not the discriminator:

| resolved concept | rows with a subtraction | negatives | % |
|---|---|---|---|
| `us-gaap:LongTermDebtCurrent` | 594 | 86 | 14% |
| `us-gaap:DebtCurrent` | 546 | 51 | 9% |
| `us-gaap:ShortTermBorrowings` | 862 | 17 | 2% |

- [x] Gave `shortTermDebt` an `_only_when` condition — the structural mechanism `ppeNet`
      uses, not a concept whitelist.
- [x] **Then measured it and found the mechanism itself was unsound** — see 3c.8. The rule
      is now `_only_when_test: declared_descendant`, requiring POSITIVE evidence.
- [x] Eight further sign violations exist outside `shortTermDebt` and are **unrelated to
      leases**: MTB 2011 `accumulatedDepreciation` (x2, filer tags it negative), DTE 2011
      `longTermDebtCurrentOnly` (x2), SWKS/VRT `interestExpense`, BAC 2018 `stockBasedComp`,
      AMT 2021 `longTermDebt`. Record; these are filer sign conventions, handled by the
      Phase 7 validator, not here.

#### 3c.6 Coverage checks, not fixes — **medium/low** ✅

**`totalLiabilities` — verified NOT a resolver defect, and NOT a missing tag list.** The
field has exactly one candidate, `us-gaap:Liabilities`, and the clarification pass checked
the filings directly:

  * **APA / EOG / ETN / VLO** report **zero `Liabilities` facts of any kind**. Reg S-X Rule
    5-02 contains no "Total liabilities" caption, so tagging it is elective and its absence
    is not a filing defect.
  * **DTE** reports 4 `Liabilities` facts and **all 4 are dimensioned** (registrant-scoped).
    `entity_scope` correctly drops them; none is a consolidated total.

  So no tag work is warranted. Deferred to the end of the plan, after the missing-tag work,
  exactly as instructed.

- [x] Deferred to **Phase 5**, where the derivation is already specified. Both inputs are
      confirmed present on all five tickers.
- [x] ~~Fix the derivation before relying on it.~~ **Withdrawn — the claim was wrong.** The
      clarification pass reported that `"derived_fallback": "totalAssets - stockholdersEquity"`
      uses parent-only equity and so overstates liabilities by the NCI. It does not: the
      formula names the catalogue FIELD `stockholdersEquity`, whose own concept order is
      `StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest` first and
      `StockholdersEquity` only as a fallback. The reading was of the field's NAME, not its
      concept list. The residual exposure is confined to filers that tag only the parent-only
      concept while reporting NCI, and that is a Phase 5 validator check, not a formula bug.
- [x] **`sharesOutstanding` absent for META** — edgartools #691, confirmed live and still
      the only Tier-1 hole besides `totalLiabilities` after the re-sweep. Upstream; record,
      do not work around.
- [x] **Filing-count truncation check** (detect and report only). VRT 33, META 57, all
      others 62-63. **KR is repaired**: 54 -> 62, the 8 filings the since-fixed
      `fiscal_period` bug was dropping.
- [ ] **RE-REGISTRATION breaks the ticker walk, and it is not the one-off the audit called
      it** (verified live 2026-08-22). `Company(ticker)` resolves through SEC's
      `company_tickers.json`, which maps a ticker only to the CURRENT registrant, so a
      predecessor's decade is invisible — no error, no gap signal, just a short list:

      | ticker | listed CIK | first | predecessor CIK | predecessor window |
      |---|---|---|---|---|
      | APA | 1841666 APA Corp | 2021-05 | **6769 Apache Corp** | 2011-02 -> **2024-11** |
      | GOOGL | 1652044 Alphabet | 2015-10 | **1288776 Google Inc** | 2011-02 -> 2016-02 |
      | ETN | 1551182 Eaton plc | 2012-11 | Eaton Corp (Ohio) | Irish domestication |
      | VRT | 1674101 | 2018-08 | none | genuine 2018 SPAC listing, NOT a defect |
      | CVS | 64803 | 2011-02 | same CIK | RENAME only, no break |

      A **rename** is harmless (CVS Caremark -> CVS Health, Facebook -> Meta keep their
      CIK). A **re-registration** — new holding company, foreign domestication — does not.

      **And the repair is not a union of the two CIKs.** Apache Corp kept filing its own
      10-K/10-Q at 4 per year through 2024-11-07, because it has registered public debt and
      therefore remains a reporting registrant. 2021-2024 is **double-covered**, and Apache
      Corp's statements are the SUBSIDIARY's, not the listed parent's. Concatenating would
      duplicate ~15 filings and mix two entities; the rule has to be a dated cutover at the
      reorganisation (2021-03-01 for APA, 2015-10-02 for Alphabet).

      Universe-layer work, out of scope for Phase 3c — but it belongs in the plan as a
      named gap, not as an APA footnote, because GOOGL shows it recurs and the out-of-sample
      roster will reproduce it.

#### 3c.8 What the re-sweep found — the fixes' own defects — **critical** ✅

Four defects that could only appear *after* 3c.1-3c.5 landed. Three of them are 3c.1 doing
its job: making 418 previously-unreadable filings readable also made them resolvable
**wrongly**, which is exactly why the ordering rule put 3c.1 first.

- [x] **A bank's cash-flow statement matched the income-statement role test.** The FASB
      standard role is `StatementOfCashFlowsIndirectDepositBasedOperations` — it contains
      **"Operations"**. MTB stored **27 rows** of `CashAndCashEquivalentsPeriodIncreaseDecrease`
      as revenue and USB another **9** (`CashPeriodIncreaseDecrease`,
      `NetCashProvidedByUsedInOperatingActivities`), all in the 2012-2016 filings 3c.1 had
      just unlocked. `NOT_INCOME_STATEMENT_ROLE` now excludes `cash[\s_-]*flow` as well as
      `comprehensive`.
- [x] **`NOT_A_TOP_LINE` was missing three non-operating items** that sit under the pretax
      subtotal, each measured:
      `InvestmentIncomeDividend` (VRT 2018-2020, **17 rows** — the SPAC shell's trust
      dividends won the slot the zero guard had correctly vacated, which is why VRT's
      correct zero did not survive 3c.3);
      `ForeignCurrencyTransactionGainLossUnrealized` (ETN 2012, 2 rows, revenue of **-$149**
      for a $16 bn company);
      `NoninterestExpense` (GS 2019, 4 rows — regime-independent, since GS is
      `broker_dealer` and the bank `never_use` never applied to it).
- [x] **`_only_when` read SILENCE as evidence.** `has_sibling` returns False both for
      "declared elsewhere" and for "the linkbase says nothing at all", and the first
      version of 3c.5 treated the second as licence to subtract. That left **75 of the 127
      surviving negatives** on the `tag_primary` / `tag_fallback` routes, where the resolved
      concept is not in the linkbase at all. A structural condition can no longer be
      satisfied by a concept the structure does not mention.
- [x] **The sibling test is VACUOUS on this roster, `ppeNet` included.** Probed 31 filings
      spanning every (ticker, concept, route) combination that still subtracted: the lease
      leg is `LEG-NOT-IN-LINKBASE` or `TOTAL-NOT-IN-LINKBASE` in **every single case** and a
      declared descendant in **none**. So `ppeNet`'s flagship "the linkbase distinguishes
      folded-in from separately-presented" has never actually discriminated anything here —
      all 128 of its adjustments fire on silence.

      The fix is **not** to make both fields stricter, because the two standards point
      opposite ways:

      | field | standard | what silence means | test |
      |---|---|---|---|
      | `ppeNet` | ASC 842-20-45-4 **permits** folding the finance-lease ROU asset into PP&E | weak evidence FOR containment | `not_a_declared_sibling` (unchanged) |
      | `shortTermDebt` | ASC 842-20-45-1 **requires** operating lease liabilities to be presented separately | evidence AGAINST containment | `declared_descendant` (new) |

      `total_adjustment._only_when_test` now names the test; the default is the weaker one,
      so `ppeNet` keeps its measured behaviour (128 adjustments, **0** negatives) untouched.
- [ ] **Carry to Phase 7**: `ppeNet`'s adjustment is unguarded in practice. It is not
      producing bad values today, but the guard the catalogue claims for it does not fire,
      and that should be stated in the validator rather than assumed away here.

#### 3c.7 Re-sweep and the before/after the audit could not settle — **the real acceptance test** ✅

> Its one open item — *"fold the surviving numbers into `test_linkbase_history.py` as
> assertions"* — is now **v2's Phase 4c.5**.

- [x] Re-ran the 26-ticker x 2011-2026 sweep after 3c.1-3c.5 (25.6 min, 1,552 filings,
      145,158 rows), then again after 3c.8.
- [x] Resolution mix **by year**: 2011-2014 moves from 0.9% to 69.7-73.7% linkbase. Full
      table in 3c.1.
- [x] The **modern** era moved too, as 3c.1 predicted: 2022 +3.5pp and the role test adds
      arcs on 13.3% of 2016+ filings.
- [x] **The 2011-2014 before/after comparison — the question the audit could not settle.**
      108,586 rows join on the same (ticker, accession, field, period) key. For fiscal years
      **<= 2014, the route changed on 63.8% of rows and the VALUE agreed on 99.86% of them**
      (>= 2015: route changed 1.52%, value agreed 99.23%). **This is the evidence that the
      architecture change is an improvement rather than merely a different answer**: where
      the tag list happened to be right, reading the filer's own roll-up confirms it, and
      the 0.14% that moved is where it was not.

      The 648 material disagreements are all accounted for and all in the intended
      direction: MTB's bank basis (96 rows, 3c.4), the lease over-subtraction unwinding
      (226 `totalDebt` + 224 `shortTermDebt` rows, 3c.5), the garbage roots being replaced
      by real revenue (49 rows, 3c.2), the zero guard (32 rows, 3c.3), and one
      `stockholdersEquity` row moving off a zero-only concept.
- [ ] Fold the surviving numbers into `test_linkbase_history.py` as assertions.
- [x] Re-ran KR's shard: 54 -> 62 filings, the `fiscal_period` bug's 8 dropped filings
      recovered.
- [x] **Out-of-sample: 26 tickers the rebuild has never seen.** Every number above is
      measured on a roster chosen BECAUSE it broke things, so it is a fit to known failures.
      The out-of-sample roster is chosen for what the first one misses: **the entire Health
      Care sector** (UNH and CVS run insurance economics inside a non-Financials GICS
      sector; HCA carries negative stockholders' equity), **captive finance** (CAT, DE),
      **the broker/bank boundary** (SCHW, AXP), a second multi-registrant utility (DUK), a
      second data-centre/tower REIT (EQIX) and multi-class shares outside tech (GOOGL).
      Full roster: UNH CVS HCA JNJ LLY TMO MSFT NVDA ORCL WMT COST PG MCD CAT DE BA UNP WFC
      C SCHW AXP CB DUK PLD EQIX GOOGL — zero overlap with the in-sample 26.

#### 3c.9 What the OUT-OF-SAMPLE roster found — three defects that pre-date Phase 3c ✅

26 tickers with zero overlap, 1,611 filings, 160,611 rows. **Every Phase 3c criterion passes
on a roster it was never tuned to** (linkbase 74.7% in 2011-14, `tag_fallback` 6.85%, 0 revenue
values off the income statement, 0 zero-revenue rows, 0 negatives caused by an adjustment,
0 regime flips), and structural discovery generalises to a sector never seen: HCA resolves on
`HealthCareOrganizationPatientServiceRevenue`, PLD on `RealEstateRevenueNet`, DUK on
`RegulatedAndUnregulatedOperatingRevenue` — none of which any tag list names.

But the sweep found three defects that are NOT Phase 3c regressions and that the in-sample
roster could not surface. Ranked:

- [x] **`linkbase_sum` applies the filer's weight relative to the WRONG parent — Tier 3,
      blocking.** FIXED. MSFT `sellingGeneralAdmin` = **-$34.7bn on 159 of 202 rows**. Verified on
      the 2026-07-29 10-K: MSFT declares `SellingAndMarketingExpense` and
      `GeneralAndAdministrativeExpense` as **-1.0 children of `OperatingIncomeLoss`**, while
      the field's own total is `SellingGeneralAndAdministrativeExpense`. The weight describes
      how a leg foots into OPERATING INCOME, not into SG&A, so `_linkbase_weights` flips the
      sign of an expense aggregate. It is the only field affected in either roster (159 of
      1,770 `linkbase_sum` rows on `non_negative` fields), because it is the only one whose
      legs hang off a subtotal that is not its own total.
      **Fix**: honour the filer's weight only when the legs share ONE parent AND that parent
      is the field's declared total (`shortTermDebt` -> `DebtCurrent`, bank `totalRevenue` ->
      `RevenuesNetOfInterestExpense` both still qualify); otherwise +1.0.
- [x] **`_compose` zero-fills a MISSING DEBT LEG — Tier 1, blocking.** FIXED. `totalDebt` sums
      `longTermDebt + shortTermDebt + financeLeaseLiability + operatingLeaseLiability` and
      treats a missing component as zero. The docstring justifies it -- *"totalDebt must not
      vanish because a filer has no finance leases"* -- which is right for a missing LEASE leg
      and wrong for a missing DEBT leg. Measured: **213 of 2,655 in-sample rows (8.0%) and 29
      of 3,096 out-of-sample (0.9%) are a LEASE LIABILITY reported as total debt** — BRK-B
      $4.9-6.3bn, GS $2.1-2.4bn, META $7.6-16.7bn, AFL, MAA, PGR, ORCL, LLY, MCD, NVDA, CAT.
      PGR traces the whole failure: correct at $1.9-2.7bn 2011-2016, NULL for 2017-2018, then
      $179-211M for 2019-2021 once `longTermDebt` stops resolving and the sum silently becomes
      the operating lease liability. Every one of those numbers looks like data.
      **Fix**: `roll_up` needs required-vs-optional components. A sum with no debt leg must be
      NULL with a reason code.
- [x] **`employees` is in the XBRL extractor's field list but is not an XBRL field.** FIXED. Its spec
      says `"source": "text:10-K"`, `"annual_only": true` — it comes from
      `fundamentals_employees.py`. It resolves 0 times on all 52 tickers and emits one
      reason-coded row per filing, ~1,600 wasted rows per sweep. Remove it from
      `extracted_fields`.

**VERIFIED 2026-08-22 on BOTH rosters** (full re-sweep, 36.9 min in-sample + 29.1 min out).
`ppeNet` was found to share `totalDebt`'s defect and was fixed with it.

| check | in-sample before -> after | out-of-sample before -> after |
|---|---|---|
| `employees` rows emitted | 1,552 -> **0** | 1,611 -> **0** |
| `linkbase_sum` negatives on `non_negative` fields | 2 -> 2 (as-filed) | **159 -> 0** |
| MSFT `sellingGeneralAdmin` | n/a | **-$34.7bn -> +$3.7 to +$34.7bn**, 0 negative |
| `totalDebt` composed without a debt leg | **213 -> 0** | **29 -> 0** |
| `ppeNet` composed from one leg | **88 -> 0** | **8 -> 0** |
| valued rows lost | totalDebt -213, ppeNet -88, **nothing else** | totalDebt -29, ppeNet -8, **nothing else** |

The cost is EXACTLY the wrong rows and no others -- no collateral damage to any of the
other 46 fields on either roster. All five Phase 3c acceptance criteria still PASS on both.

Out of sample **every Tier-1 field is now at 100% median quarter coverage**, `totalDebt`
included (26/26 tickers). In-sample `totalDebt` drops to **23/26 tickers**: AFL, BRK-B and
MAA lose it entirely, because `longTermDebt` never resolves for them and the field now
refuses to substitute a lease liability. That is the intended trade -- a reason-coded NULL
instead of a confident wrong number -- and it isolates the real remaining gap:

- [ ] **`longTermDebt` concept coverage is the last Tier-1 hole.** 23/26 in-sample, median
      94.9% of quarters, and its absence is the sole cause of every `incomplete_roll_up` on
      `totalDebt` (117 filings / 7 tickers in-sample, 10 / 2 out). AFL, BRK-B and MAA are
      an insurer, a hybrid holding company and an Up-C REIT -- three unclassified-balance-
      sheet filers that almost certainly tag debt under a concept the candidate list does
      not name. Widening it is the natural next step and the obvious first target for the
      validation layer.

Two further observations, for Phase 7 rather than here:

- [ ] **AXP revenue is post-provision for 2011-2018** (91 rows on
      `TotalRevenuesNetOfInterestExpenseAfterProvisionsForLosses`), then switches to the
      ASC-606 element — a basis break mid-history, the same class as MTB's. ~~AXP routes to
      `industrial` (GICS "Transaction & Payment Processing Services"), so the bank regime's
      `never_use` never applies to it. Whether a card lender should route to `bank` is a
      regime-router question, not a resolver one.~~
      **CORRECTED 2026-08-23 — this sentence is wrong.** `sp500_tickers` gives AXP
      `sub_industry = 'Consumer Finance'`, which `fundamentals_regimes.json:23-28` maps to
      **`bank`** (and whose `consumer_finance_note` already names AXP as the confirmed case).
      "Transaction & Payment Processing Services" is **V and MA**, not AXP. So AXP *is* on the
      bank template and the bank `never_use` *does* apply to it — the concept was simply never
      added to that list. Not a regime-router question at all: one `never_use` entry, exactly as
      3c.4 did for MTB. → **v2 Phase 4c.7**.
- [ ] **`totalLiabilities` is systemic, not roster-specific.** Six more tickers with zero
      coverage out of sample (DUK, LLY, MCD, ORCL, TMO, WMT) on top of the in-sample five,
      confirming Reg S-X 5-02's elective caption affects ~20% of issuers. Phase 5's
      `totalAssets - stockholdersEquity` derivation is load-bearing, not a nicety.
- [ ] **The regime-exception register does not scope the balance-sheet detail fields**
      (`accountsPayable`, `accountsReceivable`, `ppeGross`, `accumulatedDepreciation`,
      `intangiblesExGoodwill`, `minorityInterest`). Banks, brokers, insurers and REITs run
      unclassified balance sheets and have no such caption, but nothing declares that, so a
      coverage gate cannot separate "structurally absent" from "regression" — 346 of the
      in-sample "holes" are structural. Declare them before Phase 7 builds a gate on them.

**Estimated effort**: 1.5-2 days + ~25 min per re-sweep. Raised from 1-1.5: 3c.2 needed an
`ArcGraph` change and a ranking rule, 3c.3 the two-pass resolve, and 3c.8 was not foreseen
at all.

**Risk zones touched**: `configs/fundamentals/fundamentals_kpis.json` — `totalRevenue`
bank `never_use` (3c.2/3c.4) and `shortTermDebt` `total_adjustment._only_when` +
`_only_when_test` (3c.5/3c.8). Both approved 2026-08-22. `ppeNet` deliberately NOT touched.

---

### Phase 4: The period engine ✅

**Goal**: discrete quarters that are actually discrete, and a Q4 that is not a tautology.

**Changes**:

1. `src/data_extract/utils/fundamentals/periods.py` — new, rewritten from
   `fundamentals_periods.py`. Target: **< 400 lines** (from 1,140).

   **Steal edgartools' Q4 ladder** (`edgar/ttm/calculator.py:680-802`), which this repo does not
   have:
   - [ ] **Primary: `Q4 = FY − YTD_9M`.** One as-reported YTD number instead of three derived
         quarters — strictly fewer error sources, and it works for the cash-flow concepts ASC 230
         only requires cumulatively.
   - [ ] **Fallback: `Q4 = FY − (Q1+Q2+Q3)`** (today's only path).
   - [ ] Select inputs **by calendar period, not by `fiscal_period` label** — the SEC tags
         comparative facts in a re-filing with the *filing's* label, so one calendar quarter can
         appear as Q1, Q2 *and* Q3 across successive 10-Qs (edgartools GH #848).
   - [ ] Port `_is_additive_concept`: refuse instants, share counts, ratios and per-share units.
   - [ ] Same ladder for `Q2 = YTD6 − Q1` and `Q3 = YTD9 − YTD6`.

   **Keep, verbatim in intent — these are right and were expensively learned**:
   - [ ] FY anchor must be annual-**shaped**, not merely FY-**labelled** (Skyworks FY2020 tags both
         a 370-day and a 97-day fact as `fp='FY'`).
   - [ ] Relative chronological rank, never a day-count divisor — calendar quarters are 90/91/92
         days, and a 52/53-week issuer's fiscal Q1 can be ~112 days.
   - [ ] Fiscal years keyed off the issuer's own 10-K period-ends (immune to 52/53-week drift).
   - [ ] `_fy_matches_quarterly_run_rate` uses **sum of |quarters|**, not |sum of quarters|.
   - [ ] Never trust a native Q4 label.
   - [ ] The `YTDn` → quarter label remap (cash-flow concepts arrive as `YTD3/6/9/12`).
   - [ ] Provenance travels on the value's own row, never re-joined.
   - [ ] The anchor-concept tie-break (worth ~200× on Valero: its true D&A is
         `DepreciationAmortizationAndAccretionNet` at $2,405M/yr tagged YTD-only, while a $47M/yr
         `DepreciationAndAmortization` line carries a discrete-quarter context).

2. **The staircase fix** (defect #4). Today's annual fallback `ttm_a → <field>_ann.ffill(limit=4)`
   freezes 6.2% of consecutive pairs.
   - [ ] A TTM is emitted **only** from four discrete quarters. Where they do not exist, the TTM is
         **NULL with `dc_code = 'insufficient_quarters'`** — not a carried-forward annual.
   - [ ] XOM pre-2018 therefore goes NULL rather than repeating 420.8 for four filings, and
         `revenueGrowth` stops being exactly 0 for three quarters in four.
   - [ ] This *reduces* coverage on purpose. Quantify the loss in Phase 9 and report it.

**Verification**:
- [x] Known-truth: synthetic YTD ladder where `FY−YTD9` and `FY−(Q1+Q2+Q3)` disagree → assert the
      YTD9 path wins and is recorded.
- [x] Real-data: **AAPL Q4-2025 revenue = 102.466 B** (the edgartools-verified figure).
      Skyworks FY2020 → **956.8M, matching the filer's published figure exactly** (the plan
      expected 956.7M).
- [x] Frozen-TTM detector: **0** exactly-repeated consecutive `totalRevenue` pairs for
      APA / XOM / ETN / MTB. Roster-wide rate below. (TROW is in neither roster.)
*(`test_fundamentals_point_in_time.py` moves to Phase 5 — under the §5.0 event grain the PIT
invariant is a property of the history build, not of the period engine.)*

**Estimated effort**: 2-3 days.

#### Phase 4 — done 2026-08-22

`src/data_extract/utils/fundamentals/periods.py` (**710 lines, 328 of them code** --
over the plan's <400 target, which counted lines rather than logic; the deleted engine was
1,140 and this repo's comment density is deliberate) and `tests/data_extract/test_periods_q4.py`
(**20 tests: 16 synthetic known-truth + 4 real-data**, all passing). `period_shape` and the
duration bands MOVED here from `fetch_fundamentals_sec.py`, which now imports them — the
period vocabulary belongs to the period engine.

**The two headline defects are gone**, measured on 26 tickers × 1,552 filings × 2011-2026:

| | legacy | now |
|---|---|---|
| Q4 rows derived by the tautological `FY − (Q1+Q2+Q3)` | **203,798 of 203,798 (100%)** | **3 of 4,489 (0.07%)** |
| Q4 rows the footing check can genuinely test | 0 | **4,486 (99.9%)** |
| frozen consecutive `totalRevenue` TTM pairs | **6.2%** (APA 100%, XOM 36%) | **0.33%** (5 of 1,532) |
| frozen pairs, all fields | — | 0.66% |
| TTM emitted | always, by carrying an annual forward | **93.6%**, the rest reason-coded |

The 5 surviving frozen revenue pairs are all genuine: NEE 2 and USB 1 are figures the filer
rounds to three significant digits and that genuinely repeat; VRT 2 are the SPAC shell's
correct zeros.

**Independent validation the plan did not ask for, and the strongest evidence available.**
Sum the four derived discrete quarters and compare against the filer's OWN annual fact —
which the engine never reads when it derives Q1-Q3, so agreement is a real test rather than
an identity. **3,698 comparable (ticker, field, fiscal year-end) points: 97.9% agree within
0.5%, median error 0.0000%, p95 0.05%.** The 47 that disagree by >2% are concentrated in
`incomeTaxExpense` / `pretaxIncome` / `operatingIncome`, only 2 involve a concept switch,
and they are restatement and basis change — they belong to Phase 7's `q4_footing` and
`cross_identity` checks. A related measurement worth keeping: **4.53% of annual windows (234
of 5,166) move by more than 2% between their first and last filing.** Risk 5 is real and
larger than the plan assumed.

#### Five things the plan's Phase-4 text did not anticipate

1. **The subtraction is CROSS-FILING.** The FY fact comes from the 10-K and the YTD9 from
   the Q3 10-Q — two separate `resolve_field` calls that can land on different concepts. The
   legacy engine met this and relaxed strict tag-matching to a scale test after strictness
   made 107 real quarters underivable. Ported, but **gated on an actual concept comparison**,
   which the new `source_concept` column makes possible and the old code could not do:
   measured, 125 of 18,650 quarters (0.67%) cross a concept switch and are flagged. The
   scale test itself was rewritten to compare **per-day rates** — a count-based annualisation
   (`× 4 / len(parts)`) is dimensionally wrong the moment a leg is itself cumulative, which
   it always is here.
2. **A share count must be differenced in SHARE-DAYS, not refused.** The plan says *"refuse
   share counts"*; so does edgartools' `_is_additive_concept`. Measured, refusing them leaves
   `dilutedShares_ttm` computable at **129 of 1,532 points (8%)** — filers never publish a
   discrete Q4 weighted average, so a four-quarter run never closes — and decision #9 defines
   `epsDiluted` as `netIncome_ttm / dilutedShares_ttm`. Refusing does not protect Tier-2 EPS,
   it deletes it. `average × days` IS additive (it is share-days outstanding), so the ladder
   runs in that space and converts back: **epsDiluted goes from 8% to 87.8% computable.**
   What decision #9 forbids is summing four quarterly *EPS* figures — a ratio of two flows
   whose denominator moves — which is a different thing from a time-average of a stock, and
   ratios never reach here anyway (`build_periods` walks only `kind == "duration"`).
   Validated against ground truth: at every fiscal year end where both exist, the share-day
   derivation agrees with the filer's own annual weighted average within 0.5% on **97.3% of
   710 points, median error 0.0052%**.
3. **`period_days` is not additive, and that cost 1%.** `(end - start).days` under-counts by
   a day and the legs do not foot: a calendar year reads 365 while its nine-month and Q4 legs
   read 273 + 91 = 364. Multiplying and dividing by it loses a day at every junction.
   Counting both endpoints (274 + 92 = 366) improved the share-count median error **50×,
   from 0.2756% to 0.0052%**.
4. **A stock split makes a trailing window mix two unit bases.** A split retroactively
   rescales every prior share count, and the four quarters of a window come from four
   different filings. Measured: **45 `dilutedShares` windows across 8 tickers** (AAPL 7:1
   2014 and 4:1 2020, NEE 4:1 2020, AFL 2:1 2018, EOG 2:1 2014, KR 2:1 2015, VRT's SPAC
   merger, and BRK-B whose A and B classes differ by 246,000×). AAPL's FY2012 derived Q4 read
   **24.3 billion shares**. Refused with a new `split_basis_mismatch` code rather than
   repaired — picking a basis is guessing which one the consumer wants, and the number it
   feeds is wrong by an exact integer factor and looks entirely plausible. 24 TTM rows.
5. **The fiscal calendar belongs to the TICKER, and had three separate bugs.** Built per
   *field* it has one bucket for a field with one annual fact, so AMT's `interestExpense` put
   2015, 2016 and 2017 all in FY2017 and produced four Q1s. A gap in the annual facts
   swallowed three years (MAA's 2013 quarters landed in FY2017). And ranking quarters
   chronologically from the END mislabelled the year a filer is still inside — AAPL's three
   post-10-K quarters came out Q2/Q3/Q4 instead of Q1/Q2/Q3, and those are exactly the
   quarters a model trades on. Fixed by sharing one calendar per ticker, interpolating
   missing years, extrapolating one year forward, and positioning each quarter against **that
   year's own start and length**. Collisions **69 → 0**; quarters labelled **99.3% → 100%**.
   Kroger's 16/12/12/12-week retail calendar labels correctly with no special case.

Also found, and fixed here because the shape is the same: **the same quarter tagged twice a
day apart survived deduplication.** Filers nudge the boundary day between filings — GS tags
Q1-2013 as both `→ 03-30` and `→ 03-31`, KR ships three variants of the quarter ending
2011-11-05. A window's identity is now its END within `_SAME_PERIOD_DAYS = 7`, an order of
magnitude below the smallest real gap between two quarter ends on either roster (KR's 82).

#### `instant_stock` — an obligation Phase 1 left for this phase

`test_fundamentals_employees.py::test_employee_fact_row_is_a_year_end_instant` was gated on
`pytest.importorskip("...periods")` so it would *turn green by arrival*. It did the opposite:
the module arrived and the test started failing on a missing `instant_stock`. Implemented —
an instant tagged with the fiscal YEAR is the year-END snapshot and belongs in the Q4 slot,
and the discriminator is the **absence of a `period_start`**, not the label, because a
duration field legitimately has both an `FY` and a `Q4` flavour and must never be relabelled
that way. `build_periods` now returns `(quarters, ttm, instants)` rather than leaving
instants to a second call: a history built from the first two alone is missing every
balance-sheet level and would still look complete.

#### Phase 4 — the three questions, measured on BOTH rosters (2026-08-22)

52 tickers, 3,163 filings, 40,687 discrete quarters. Asked because "the ladder works" is a
claim, not evidence.

##### 1. Is every filing and every quarter there?

| | in-sample | out-of-sample |
|---|---|---|
| filings per ticker per year | 4.06 – 4.19 (median 4.08) | 4.00 – 4.10 (median 4.08) |
| tickers with < 58 filings 2011-2026 | **APA, ETN, META, VRT** | **GOOGL** |
| revenue quarters present, between each ticker's own first and last | **1,649 / 1,670 (98.74%)** | **1,724 / 1,733 (99.48%)** |
| tickers with a COMPLETE quarter grid | **23 / 26** | **24 / 26** |

Every short filer is already a known, explained case: APA and GOOGL are the re-registration
gap (§3c.6), ETN the 2012 Irish domestication, VRT the genuine 2018 SPAC listing, META the
2012 IPO. **No ticker is short for an unexplained reason**, and every other ticker files at
4.0-4.2 filings a year, i.e. nothing is being dropped by the walk.

The 30 missing quarters are 5 gaps, all in the FACTS layer rather than the period engine —
the filer tagged no revenue concept that survived entity scoping in those quarters:
**MAA 17** (2013Q4-2015Q2, the Up-C `LegalEntityAxis` era), **JNJ 8** (2015 and 2020),
**GS 3** (2019), **DE 1**, **VRT 1**.

##### 2. Proof that the Q4 derivation is right — a hold-out with ground truth

Take every (ticker, field, fiscal year) where the filer published **all three** of the FY
fact, the YTD9 fact **and its own discrete Q4**. The engine prefers the as-reported quarter
and never derives these, so forcing the derivation there is a genuine hold-out.

`FY − YTD9 == reported Q4` is an **identity** whenever the filer's own three numbers are
mutually consistent, so the result splits cleanly:

| | in-sample | out-of-sample |
|---|---|---|
| hold-out cases | **591** | **752** |
| the filer's own three numbers FOOT | 553 (94.5%) | 694 (92.4%) |
| → derived value **exact to the dollar** | **74.86%** | **81.70%** |
| → within 0.1% | 88.25% | 94.24% |
| → **within 1%** | **98.73%** | **98.99%** |
| the filer's own numbers do NOT foot | 32 (5.5%) | 57 (7.6%) |
| → we match | 3.1% | 0.0% |
| **filer consistent AND we are wrong by >1%** | **7** | **7** |

The 14 residual cases are all **the filer's own rounding measured against a small Q4
denominator** — NEE fiscal 2017 is typical: a $2M gap on a $5,173M year is 0.04% of the
year and 1.1% of the $186M quarter. Largest residual 4.8%.

Where the filer does *not* foot, no method can match both its Q4 and its FY; those are
reclassifications between the 10-Q and the 10-K (VLO fiscal 2012 operating income: YTD9
$2,426M + Q4 $1,584M = $4,010M against a published FY of $5,044M).

**Footing on what the engine actually emits**, additive fields only, restricted to the years
whose Q4 is *not* the identity and so **cannot pass by construction**:

| | in-sample | out-of-sample |
|---|---|---|
| independently-derived complete years tested | **3,698** | **4,133** |
| Q1+Q2+Q3+Q4 foots to the filer's FY within 0.5% | **97.94%** | **97.6%** |

##### 3. Edge cases, field by field

The per-field census covers, for all 22 duration fields: share derived, share on the
tautological route, negatives on a non-negative field, day-count outliers, concept switches,
hold-out accuracy, footing rate and TTM null rate.

**Negative values on a `non_negative` field — the sharpest defect class:**

| | in-sample | out-of-sample |
|---|---|---|
| negative quarters | **1 of 18,648** | **6 of 22,039** |
| …of which the engine caused | **0** | **0** |

All 7 are `as_reported` — the filer's own tagged sign. Boeing tags `CostOfRevenue` negative
in its 2009 comparatives (4 quarters, −$13.7 to −$14.5bn), Citigroup once in 2020, PLD's
2011 `ShareBasedCompensation` once, VRT's `InterestExpense` once. These are Phase 7's
`sign_convention` check (Debreceny et al. 2010: debit/credit treatment is the *dominant*
cause of XBRL arithmetic failure), not the period engine's.

**Everything else, on both rosters:**

- **tautological Q4 route**: ≤ 0.16% of any field's quarters; 3 rows in-sample, 10 out.
- **footing**: every field ≥ 93.5%; the weakest are `totalRevenue` (96.4% / 93.5%),
  `operatingIncome` (96.1% / 95.2%) and `incomeTaxExpense` (96.7% / 96.3%) — the same three
  the restatement analysis flags, and the same three the hold-out flags.
- **day-count outliers**: the 15-18 quarters per field over 100 days are Kroger's 16-week
  fiscal Q1 (111 days) and Skyworks' 97-day fiscal-2020 Q4, both correct and both correctly
  labelled. None under 80 days except 9 `costOfRevenue` out of sample.
- **concept switches**: 0.67% in-sample, 0.71% out — each one flagged on its own row.
- **TTM null rate**: 4.5-10% for every field except out-of-sample
  `realizedInvestmentGains` (20.5%, an event-driven insurer line that is genuinely absent
  in most quarters).

**How much of this is "no edge cases" versus "no edge cases we can see"?** The honest
boundary: this proves the *period arithmetic*, on 52 tickers spanning 8 regimes, 3 fiscal
calendars (calendar, September, 52/53-week, and Kroger's 4-4-5), the ASC-606 and ASC-842
cutovers, 6 stock splits, a SPAC merger, a foreign domestication and two re-registrations.
It does **not** prove the concept resolution underneath it — `capex` resolving for 15 of 26
in-sample tickers is a Phase-3 hole this measurement surfaced but cannot fix — and it cannot
see a field that resolves consistently to the *wrong* concept in every period, because
nothing internal contradicts it. That is what Phase 7's external-source check is for.

#### Carried forward from Phase 4

> **Phase 4 closed 2026-08-23.** Every item below is now either done or DELIBERATELY
> deferred to the phase that owns it, so nothing is outstanding *in this phase*: `capex`,
> the `totalRevenue` sign rationale, the `depAmort` periodicity case and the hold-out proof
> were all closed by Phase 4b; the missing reason code on a REFUSED QUARTER belongs to
> Phase 5's `fundamentals_reason_codes` (adding a second mechanism here would give it two
> sources of truth); and the three weakest fields are Phase 7's priority.

- [x] **`capex` never resolves for 9 non-bank tickers** *(CLOSED by Phase 4b, option A)* — APA, CB, DTE, EOG, MAA, MET, NEE,
      PLD and AFL are all `unresolved` / `not_disclosed`, while only BAC/C/JPM/MTB/USB/WFC are
      correctly `not_applicable`. (This item originally said **7**; re-measuring on the
      out-of-sample roster added CB and PLD.) A Phase-3-class **concept-coverage** hole, not a
      period-engine one, and it kills Tier-1 `freeCashflow` for an E&P and two utilities — the
      regimes where capex matters most. **→ RESEARCHED, MEASURED AND DESIGNED IN §Phase 4b;
      awaiting the user's option A/B/C decision.**
- [x] **`totalRevenue` is `sign: "any"` in the catalogue** *(CLOSED: written into the
      catalogue as `totalRevenue.sign_rationale`, measured 0/12,960 figure included)*, so the sharpest derived-quarter
      guard — a negative value on a non-negative field is arithmetically impossible — does
      **not** protect the top line. **→ ANSWERED IN §Phase 4b.8**: measured at 0 negatives in
      12,960 as-filed rows, so the usual "an insurer books realized losses" justification is
      NOT what supports it; the real reason is that the guard protects a DERIVED quarter.
      Remaining work is to write it into the catalogue as a `sign_rationale` key.
- [x] **`depAmort` is annual-only for AFL and CSCO** (84 facts, zero quarterly or YTD), so no
      discrete quarter exists and `ebitda` is annual-only for them. Correct behaviour, not a
      defect. **→ §Phase 4b.9**, and note that chasing this uncovered a much larger hole the
      item concealed: `depAmort` **never resolves at all** for 9 of 52 tickers (CB, GOOGL, MET,
      MSFT, ORCL, PGR, SWKS, UNP, USB), killing Tier-1 `ebitda` for all nine. See §Phase 4b.0.
      **BOTH CLOSED by Phase 4b**: the periodicity case is recorded in
      `fundamentals_exceptions.json`'s new `by_ticker_periodicity` block (which also
      carries the newly-found CSCO route-1 note-level defect as an explicit open
      finding), and 8 of the 9 total-absence tickers now resolve. CB stays
      `not_disclosed`, correctly.
- [ ] **A REFUSED QUARTER carries no reason code.** When `_derived` rejects a subtraction --
      the sign guard, the per-day scale test -- it returns None and the window simply has no
      row, so the null is unexplained. That is a hole against the plan's "zero unexplained
      nulls" criterion. Deliberately left to **Phase 5**, whose `fundamentals_reason_codes`
      is the one table that records why a value is absent; adding a second mechanism here
      would give it two sources of truth.
- [x] Re-run the period engine on the OUT-OF-SAMPLE roster -- done, see the three questions
      above. Every Phase-4 result holds on 26 tickers it was never tuned to.
- [x] Fold the hold-out proof into `test_periods_q4.py` as a standing assertion. **DONE in
      Phase 4b** as `test_the_q4_derivation_survives_a_holdout_against_the_filers_own_quarter`:
      asserts >80% of the filers' own trios foot and >95% of those derive within 1%.
- [ ] **`totalRevenue`, `operatingIncome` and `incomeTaxExpense` are the three weakest
      fields on every independent measure** -- the hold-out, the footing check and the
      restatement rate all name the same three. Phase 7 should treat them as the priority
      for `q4_footing` and `cross_identity`.

**Risk zones touched**: `configs/configs.yml` — a new `data_extract.fundamentals_periods`
block carrying the two derived-quarter guards (`max_opposite_sign_q4_ratio: 3.0`,
`q4_tag_mismatch_fy_max: 2.0`), both recovered with their measured provenance from the
deleted `fundamentals_tags.py`. **User chose `configs/` over `constants.py` on 2026-08-22.**
They are injected as a `PeriodGuards` dataclass so a known-truth test can state its own, and
one test asserts the config values are the ones every guard test is written against.

---

### Phase 4b — the concept-coverage holes: `capex`, `depAmort`, and the leaf-sum route ✅

> **STATUS: IMPLEMENTED 2026-08-23 as OPTION A.** The research below (§4b.0-4b.11) is the
> design as it stood before implementation and is kept verbatim, including the two figures
> the implementation later corrected — see *Phase 4b — implemented* at the end of the
> section for what was built, the four deviations each measurement forced, and the ground
> truth. §4b.4's two refuted designs remain refuted; do not re-propose them.

**Goal**: a Tier-1 `freeCashflow` and `ebitda` that exist for the regimes where they matter,
without inventing either one.

#### 4b.0 Why this section exists — both Phase-4 carry-forwards were UNDERSTATED

Phase 4 logged two coverage items from the in-sample roster. Re-measured across **both**
26-ticker rosters (52 tickers, 3,163 filings, 302,467 fact rows), both were larger than
recorded:

| the Phase-4 carry-forward said | measured on all 52 |
|---|---|
| `capex` never resolves for **7** non-bank tickers | **9** — APA, CB, DTE, EOG, MAA, MET, NEE, PLD, AFL. Out-of-sample adds **CB and PLD**, neither of which the in-sample roster could see. |
| `depAmort` is **annual-only for AFL and CSCO** | `depAmort` **NEVER RESOLVES AT ALL** for **9 of 52** — CB, GOOGL, MET, MSFT, ORCL, PGR, SWKS, UNP, USB. The AFL/CSCO annual-only case is real but *separate* and much milder. |

Measured coverage, `value IS NOT NULL`, both rosters pooled:

| field | tickers with ANY value | never resolves |
|---|---|---|
| `capex` | 37 / 52 | AFL APA **BAC C** CB DTE EOG **JPM** MAA MET **MTB** NEE PLD **USB WFC** |
| `depAmort` | 43 / 52 | CB GOOGL MET MSFT ORCL PGR SWKS UNP USB |
| `longTermDebt` | 49 / 52 | AFL BRK-B MAA *(already logged in §Phase 3c, line ~1306)* |

The six bolded names are banks and are **correct** — `capex.regimes.bank.dc_code =
not_applicable`. The other nine are not. `dc_code` on the null `capex` rows splits exactly:
**672 `not_disclosed`** (the real hole) against **378 `not_applicable`** (the banks, by design).

**Downstream cost.** `freeCashflow` = `operatingCashFlow − capex` and `ebitda` =
`operatingIncome + depAmort` are both **Tier 1**. So two Tier-1 fields are null for ~17% of
the roster each, concentrated in exactly the regimes a reader would most want them for: an
E&P, two utilities, three REIT/insurers, and — for `ebitda` — MSFT, GOOGL, ORCL and UNP.

#### 4b.1 Root cause — ONE cause, two fields

Neither hole is a period-engine problem and neither is a missing *value*. In every case the
filer reports the number; the resolver cannot name the concept it sits under. Three distinct
mechanisms, in descending order of how much of the hole they explain:

1. **The catalogue's leaf list is too narrow, and route 3 is all-or-nothing.**
   `capex.roll_up.sum` is exactly `[PaymentsToAcquirePropertyPlantAndEquipment,
   PaymentsToAcquireSoftware, PaymentsToAcquireIntangibleAssets]`, and `_linkbase_weights`
   refuses unless **every** declared child is reported (`xbrl_linkbase.py:812`,
   `if not all(child in available for child in children): return ()`).
   That rule is *correct* for `shortTermDebt` — two disjoint legs a filer almost always tags
   together — and *wrong* for capex, where the FASB roll-up under
   `PaymentsToAcquireProductiveAssets` has ~7 members (PP&E + Software + Intangibles +
   MineralRights + CryptoAsset + EquipmentOnLease + Other) and every filer reports a
   different subset. No filer on either roster reports all three declared children, so route
   3 has **never once fired for capex**.

2. **The capex line has an industry-specific standard element the list does not name.**
   Measured across 17 tickers × 941 filings × 19,001 investing-node arc rows:

   | standard concept | tickers using it | filings |
   |---|---|---|
   | `PaymentsToAcquirePropertyPlantAndEquipment` | AMT DTE DUK EQIX PGR SO VLO XOM | 366 |
   | `PaymentsForCapitalImprovements` | MAA PLD VLO | 137 |
   | `PaymentsToAcquireOtherPropertyPlantAndEquipment` | DUK EOG EQIX MAA SO | 121 |
   | `PaymentsToAcquireRealEstateAndRealEstateJointVentures` | MET SPG | 106 |
   | `PaymentsToAcquireRealEstate` | EQIX MAA PLD | 91 |
   | `PaymentsToAcquireProductiveAssets` | AMT EQIX SPG VLO | 71 |
   | `PaymentsToAcquireOilAndGasPropertyAndEquipment` | EOG VLO | 66 |
   | `PaymentsForProceedsFromNuclearFuel` | NEE | 63 |
   | `PaymentsToDevelopRealEstateAssets` | MAA PLD | 57 |
   | `PaymentsToExploreAndDevelopOilAndGasProperties` | APA | 22 |
   | `PaymentsForProceedsFromProductiveAssets` | SPG | 21 |
   | `PaymentsToAcquireInProcessResearchAndDevelopment` | MAA | 19 |
   | `PaymentsToAcquireAndDevelopRealEstate` | PLD | 16 |
   | `PaymentsToAcquireOtherProductiveAssets` | VLO | 7 |
   | `PaymentsToAcquireOilAndGasProperty` | APA | 6 |
   | `PaymentsToAcquireOtherRealEstate` | MAA | 5 |

   Only the first and sixth are in the catalogue today. Note the trap: **the two elements MAA
   actually uses for its recurring and development capex are the two entries in
   `capex.never_use`** — `PaymentsForCapitalImprovements` and
   `PaymentsToAcquireInProcessResearchAndDevelopment`. Those bans are correct *as bans on a
   standalone TOTAL* (the §2 research measured a name-keyed extractor booking a REIT as an
   R&D spender) and must not simply be deleted; admitting them as **legs of a sum** is a
   different claim and has to be written as one. There is precedent in this catalogue: bank
   `totalRevenue` bans `NoninterestIncome` as a total while keeping it a leg of the regime
   roll-up.

3. **The capex or D&A line is a COMPANY EXTENSION**, which no candidate list can ever name.
   Same failure mode as `apa:RevenuesAndOther` (which is why `totalRevenue` got
   `linkbase_root_discovery`), and it is what makes DTE, NEE, MSFT and SWKS unreachable.

For `depAmort` the mechanism is (1) + (3) only: the nine blind tickers tag `Depreciation` and
`AmortizationOfIntangibleAssets` as **separate cash-flow lines** and never the aggregate
`DepreciationDepletionAndAmortization`; two of the nine use an extension.

#### 4b.2 The decisive evidence — read the LINKBASE, not the fact set

Children of the standard anchor node in each filer's own calculation linkbase, latest 10-K.
This is the most important table in the section: it is what makes the fix safe.

**`capex` — children of `NetCashProvidedByUsedInInvestingActivities` (weight −1.0):**

| ticker | leaves | FY2025 |
|---|---|---|
| APA | `us-gaap:PaymentsToExploreAndDevelopOilAndGasProperties` + `apa:` leasehold / proved / gathering legs | $2,740M + legs |
| EOG | `us-gaap:PaymentsToAcquireOilAndGasPropertyAndEquipment` + `us-gaap:PaymentsToAcquireOtherPropertyPlantAndEquipment` | $6,115M + $479M |
| MAA | `us-gaap:PaymentsForCapitalImprovements` + `us-gaap:PaymentsToAcquireInProcessResearchAndDevelopment` + `maa:PaymentsToAcquireRealEstateAndOtherAssets` | $360M + $272M + $133M |
| **DTE** | `dte:PlantAndEquipmentExpendituresUtility` + `dte:PlantAndEquipmentExpendituresNonUtility` | $4,343M + $86M = **$4,429M** |
| **NEE** | `nee:CapitalExpendituresOfFPL` · `nee:CapitalExpendituresOfPublicUtility` · `nee:IndependentPowerInvestments` · `us-gaap:PaymentsForProceedsFromNuclearFuel` · `nee:OtherCapitalExpenditures` | |
| MET | none — the investing section is entirely portfolio flows (`PaymentsToAcquireAvailableForSaleSecuritiesDebt` $77.6bn). `PaymentsToAcquireRealEstateAndRealEstateJointVentures` $633M is an **invested asset**, not PP&E capex | — |
| AFL | **nothing at all**, across 155 filings | — |

**DTE's number is independently confirmed by the filer's own arithmetic**:
`dte:PaymentsToAcquireProductiveAssetsIncludingPaymentsToAcquireBusinessesNetOfCashAcquired`
= $4,639M undimensioned, minus `us-gaap:PaymentsToAcquireBusinessesNetOfCashAcquired` $210M,
= **$4,429M exactly**. The leaf sum is right.

> **A trap worth naming.** DTE *does* tag `us-gaap:PaymentsToAcquirePropertyPlantAndEquipment`
> at $3,686M — but **only dimensioned to `dte:DTEElectricMember`**, the subsidiary registrant.
> `entity_scope.consolidated_facts` correctly drops it. Anyone "fixing" capex by relaxing the
> dimensional filter would store the SUBSIDIARY's capex as the consolidated group's, 17% low,
> and it would look entirely plausible. Do not do that.

**`depAmort` — children of `NetCashProvidedByUsedInOperatingActivities` (weight +1.0):**

| ticker | leaves | FY2025 |
|---|---|---|
| GOOGL | `us-gaap:Depreciation` | $21,136M |
| **MSFT** | `msft:DepreciationAmortizationAndOther` *(extension)* | $38,534M |
| ORCL | `us-gaap:Depreciation` + `us-gaap:AmortizationOfIntangibleAssets` | $7,623M + $3,010M |
| UNP | `us-gaap:Depreciation` | $2,465M |
| USB | `us-gaap:DepreciationNonproduction` + `us-gaap:AmortizationOfIntangibleAssets` | $382M + $636M |
| PGR | `us-gaap:Depreciation` | $313M |
| MET | `us-gaap:OtherDepreciationAndAmortization` | $753M |
| **SWKS** | `us-gaap:Depreciation` + `swks:AmortizationOfIntangibleAssetsIncludingInventoryStepUp` *(extension)* + `us-gaap:AmortizationOfFinancingCostsAndDiscounts` | $388M + $226M + $4M |
| CB | no depreciation line exists — only `PresentValueOfFutureInsuranceProfitsAmortizationExpense1` and a **−1.0** accretion contra | — |
| *AAPL* | `us-gaap:DepreciationDepletionAndAmortization` — **route 1 already wins** | $11,698M |
| *VLO* | `us-gaap:DepreciationAmortizationAndAccretionNet` — **route 1 already wins** | $3,158M |

Two structural facts fall straight out, and both become guards in §4b.5:
- **PGR, MET and CB each carry a `−1.0` `AccretionAmortizationOfDiscountsAndPremiumsInvestments`
  contra** in the same node. The weight sign excludes it; a name filter would not.
- **CB genuinely has no D&A line.** `not_disclosed` is the correct answer for CB and must stay
  one. This fix targets 8 of the 9, not 9.

#### 4b.3 A methodological finding worth keeping: `companyfacts` is BLIND to extensions

`https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json` publishes **no
company-extension taxonomy at all**. Verified directly: DTE's and NEE's companyfacts carry
only `dei`, `us-gaap` and `ffd` — so their **entire capital programme is invisible there**,
and a companyfacts-based audit reports them capex-blind when they tag it in every 10-K.

This compounds the already-recorded limitation that companyfacts also drops *dimensioned*
facts (§Phase 3, entity scoping). Together: **companyfacts can prove a concept is PRESENT and
can never prove one is ABSENT.** Any coverage claim in Phases 7 and 9 must be measured off
`filing.xbrl()`, not off companyfacts or `frames`.

#### 4b.4 TWO DESIGNS WERE MEASURED AND REFUTED — do not re-propose them

**Refuted #1 — "sum the name-matching D&A / capex leaves the filer reports."**
Ground-truth test: take every (ticker, fiscal year) where a filer publishes **both** the
aggregate `DepreciationDepletionAndAmortization` **and** the separate legs, and ask whether
the leg sum reproduces the aggregate. **84 points; only 14 matched within 2%.**

| ticker | year | aggregate | name-keyed leg sum | error |
|---|---|---|---|---|
| AAPL | 2025 | $11,698M | $8,000M | **−31.6%** |
| DTE | 2025 | $1,841M | $1,291M | **−29.9%** |
| VLO | 2015 | $1,842M | $1,300M | **−29.4%** |
| AMT | 2013 | $800M | $766M | −4.3% |
| EQIX | 2013-2024 | — | — | **0.00%, twelve consecutive years** |

The cause: **AAPL's `us-gaap:Depreciation` is a PP&E-NOTE disclosure, not its cash-flow
line.** A name cannot tell a note figure from a statement line; the filer's calculation
linkbase can. That is exactly why the guard in §4b.5 is *"the leaf must be declared in a
cash-flow-statement role"*, and why AAPL and VLO are not at risk in practice — route 1 finds
their aggregate first and the leaf route never runs for them.

**Refuted #2 — "a negative-weight EXTENSION child of the investing node is capex."**
The only candidate rule that would have reached DTE and NEE structurally. Measured across
17 tickers × 941 filings, it admits, among others:

- `apa:EquityMethodInvestmentContribution` — up to $501M, an investment
- `nee:PurchasesOfSecuritiesInSpecialUseFunds` — $1,367-2,600M, securities
- `nee:PurchasesOfOtherSecurities` — securities
- `dte:ConsolidationOfVIES` — a consolidation entry
- `dte:RefundsToSynfuelPartners`, `dte:DistributionsToContributionsFromAffiliatesForNotesReceivables`
- `eog:ChangesInComponentsOfWorkingCapitalAssociatedWithInvestingActivities` — working capital

The inverse framing ("everything that is not a named non-capex standard concept") fails
identically, on the same rows.

**Conclusion, and it is the crux of the whole section:** there is no structural rule that
identifies a company-extension capex leaf. Extensions must be **declared per filer**, or the
filer stays `not_disclosed`. There is no third answer.

#### 4b.5 The proposed change, file by file

**(a) `src/data_extract/utils/fundamentals/xbrl_linkbase.py` — one new route, ~40 lines.**

Insert `STATEMENT_LEAF_SUM = "statement_leaf_sum"` between route 3 (`linkbase_sum`) and
route 4 (`field_sum`) in `_resolve_once`. It fires only when the field declares
`roll_up.any_of` and route 3 has not already answered.

```
roll_up.any_of : [[group A alternatives], [group B alternatives], ...]

  within a group -> take the FIRST reported concept, never two
                    (stops Depreciation + DepreciationNonproduction double-counting)
  across groups  -> sum
  fire when      -> at least one group hits
```

Three guards, each earned by a measurement above and each of which must become a named test:

1. **Statement-role guard.** A leaf is admissible only if the filer declares it in a
   cash-flow-statement role in `statement_arcs`. Without it, AAPL's note-level `Depreciation`
   is a −31.6% answer (§4b.4).
2. **Weight-sign guard.** Honour the declared sign and drop a leaf whose weight opposes the
   field's direction. Without it, PGR/MET/CB's `−1.0` accretion contra lands inside D&A.
3. **Partial-leaf guard.** If the anchor node carries sibling leaves the catalogue cannot
   classify, do **not** emit a partial sum — reason-code it. This is the `shortTermDebt`
   discipline (`_linkbase_weights`' own docstring: *"a partial sum is not the total"*), and
   §4b.6 measures what it costs.

Provenance: reuse `Resolution.children` so `roll_up_children` records exactly which leaves
were summed — `source_concept` alone cannot express a multi-leaf answer.

**(b) `src/data_extract/utils/fundamentals/fetch_fundamentals_sec.py` — `_materialise`.**
Its `LINKBASE_SUM` branch intersects period keys across legs (*"A `linkbase_sum` emits a
period ONLY where every leg is reported for that same period"*, `:118-124`). The new route
needs the same treatment, but over the leaves actually chosen for that period rather than
over a fixed list. ~10 lines.

**(c) `configs/fundamentals/fundamentals_kpis.json` — config only, no code.**
- `capex`: a per-regime `roll_up.any_of` for `energy`, `real_estate`, `utility` and
  `industrial`, populated from the §4b.1 census.
- `capex.regimes.insurer`: `dc_code: not_applicable`, **matching the bank precedent** — AFL
  tags nothing, MET tags only portfolio flows, and the absence register already measures
  insurer capex at `0.56` absent, worse than the `0.43` that earned banks the same treatment.
  Note the cost: **PGR and CB currently resolve capex and would stop.** That is the trade the
  bank cell already documents — an intermittently-tagged capex yields a TTM that silently
  mixes tagged and untagged quarters, so FCF would be *wrong* rather than missing. Record it
  in `fundamentals_exceptions.json` as an `override_reason` carrying the per-ticker three-way
  split, exactly as `bank.capex` does.
- `depAmort`: `roll_up.any_of` = [[`Depreciation`, `DepreciationNonproduction`,
  `DepreciationPremisesAndEquipment`], [`AmortizationOfIntangibleAssets`,
  `FiniteLivedIntangibleAssetsAmortizationExpense`], [`OtherDepreciationAndAmortization`]].
  Keep `DepreciationDepletionAndAmortization` as `total_concept` so route 1 keeps winning for
  AAPL/VLO/CSCO and the new route is genuinely a fallback.
- `totalRevenue`: **record the `sign: "any"` decision** (§4b.8).

**(d) `configs/fundamentals/fundamentals_exceptions.json` + `kpi_catalogue.py` — ONLY under
option A.** A `by_ticker` block declaring each extension leaf with its measured evidence,
plus a `FieldSpec.filer_leaves(ticker, field)` accessor (~25 lines) feeding the SAME route
from (a). No new route and no new config file.

**(e) Tests.** `tests/data_extract/test_linkbase_resolution.py` gains synthetic known-truth
fixtures for the three guards; a new real-data test asserts DTE FY2025 capex = $4,429M
against the filer's own $4,639M − $210M identity, and that AAPL `depAmort` still resolves to
$11,698M by route 1 rather than $8,000M by the leaf route.

#### 4b.6 THE OPEN DECISION — the user must choose A, B or C

The crux is the **partial-leaf** case. A filer that splits capex or D&A across standard *and*
extension elements gets a silently understated number if only the standard part is summed, so
guard (3) refuses it — which means option B fixes far fewer tickers than the §4b.1 census
suggests:

| ticker | standard-only leaf sum | truth | error if the guard is relaxed |
|---|---|---|---|
| SWKS `depAmort` | $392M | ~$618M | **−37%** |
| MAA `capex` | misses `maa:PaymentsToAcquireRealEstateAndOtherAssets` in **60 of 63** filings | | |
| APA `capex` | misses the `apa:` leasehold / proved legs in **21 of 22** filings | | |
| PLD `capex` | extension leaves present in **33** filings | | |

| option | scope | what it actually fixes | effort |
|---|---|---|---|
| **A** *(recommended)* | route + standard leaves + per-filer extension register | `capex`: APA DTE EOG MAA NEE PLD SPG · `depAmort`: 8 of 9 · insurer → `not_applicable` | ~2 days + re-sweep |
| **B** | route + standard leaves only, partial-leaf guard strict | `capex`: EOG · `depAmort`: GOOGL ORCL UNP USB PGR MET. APA/MAA/PLD/SWKS/DTE/NEE/MSFT stay null | ~1 day + re-sweep |
| **C** | change nothing; carry all of it to Phase 7 | — | 0 |

**A is recommended** because the per-filer register is not a nice-to-have for DTE and NEE —
it is what makes the *standard-leaf* half safe for APA, MAA, PLD and SWKS. Under B, guard (3)
nulls most of the tickers the widened list was meant to fix, so B buys much less than it looks
like it does.

#### 4b.7 Acceptance — the six-field sanity check, both rosters, every quarter

Run after implementation, on **both** 26-ticker rosters, 2011→today. The user asked for
exactly these six fields; `capex`/`depAmort` are the inputs and `freeCashflow`/`ebitda` the
Tier-1 outputs, so both layers must be reported.

- [ ] **Coverage delta table** per field × ticker: tickers with ≥1 value, and median % of
      quarters covered, **before vs after**. No field may LOSE a ticker except insurer `capex`
      (PGR, CB) — which must be a declared `not_applicable`, not a silent null.
- [ ] **Every remaining null carries a reason code.** Zero rows with
      `value IS NULL AND dc_code IS NULL`. CB `depAmort` and AFL/MET `capex` must read
      `not_disclosed` / `not_applicable`, never `unresolved`.
- [ ] **Ground truth on the new route**: DTE FY2025 `capex` = $4,429M (= the filer's own
      $4,639M − $210M); EOG = $6,594M; APA = $2,740M + legs; ORCL FY2026 `depAmort` =
      $10,633M; USB = $1,018M.
- [ ] **The AAPL/VLO regression guard**: `depAmort` still resolves by route 1 to $11,698M /
      $3,158M and **not** to the note-level leaf sum. This is the §4b.4 defect — assert it.
- [ ] **`freeCashflow` and `ebitda` sanity**: sign, magnitude against revenue, and no quarter
      where `|ebitda| > |totalRevenue|` without an explanation. Report the count per ticker.
- [ ] **Continuity through time**: for each newly-fixed ticker print the annual series
      2011→2025 and check for a level step at the point the resolution route changes. A route
      change mid-history is a `tag_switch_break` candidate for Phase 7, not a silent basis
      change.
- [ ] **`longTermDebt`** is in the user's six but is a *different* hole (AFL, BRK-B, MAA —
      unclassified-balance-sheet filers tagging debt under an unnamed concept, §Phase 3c line
      ~1306). It is **not** fixed by this section. Re-measure it here anyway, so the report is
      honest about what moved and what did not.
- [ ] **`totalRevenue`** must be unchanged by this section — assert the coverage table is
      identical before and after, so any movement is a bug in the new route's gating.

#### 4b.8 `totalRevenue`'s `sign: "any"` — the recorded decision (closes a Phase-4 item)

**Measured**: `totalRevenue` is negative in **0 of 12,960 as-filed rows** across both rosters,
including five insurers (MET, AFL, PGR, CB) and BRK-B. So the standing justification — *"an
insurer's top line carries realized investment losses"* — is **not evidenced at the as-filed
annual or quarterly grain** on this roster.

It is still the right setting, for a different and better reason: the guard protects a
**DERIVED** quarter, and `Q4 = FY − YTD9` on a restated year can legitimately go negative for
an insurer whose realized losses land in Q4. The legacy `NON_NEGATIVE_FLOW_FIELDS` included
revenue and would have nulled that quarter.

- [ ] Write this into the `totalRevenue` catalogue entry as an explicit `sign_rationale` key
      so it is a recorded decision rather than an accident, and include the measured 0/12,960
      figure. Phase 7's `sign_convention` check owns the as-filed side.

#### 4b.9 `depAmort` annual-only for AFL and CSCO — separate, and correct (closes a Phase-4 item)

Distinct from the nine-ticker hole above, and **not** a defect. AFL and CSCO tag
`DepreciationDepletionAndAmortization` on the ANNUAL window only — 48 and 36 annual facts,
**zero quarterly, ytd6 or ytd9** — so no discrete quarter exists and `ebitda` is annual-only
for them.

- [ ] Record in `fundamentals_exceptions.json` so Phase 7's `coverage_quarters` gate reads it
      as structural rather than as a regression. This is a *periodicity* exception, not an
      *absence* one, and the register has no key for that today — it needs one.

#### 4b.10 Reproducing every number above

Interpreter and environment:
```
PY="$HOME/AppData/Local/pypoetry/Cache/virtualenvs/stock-pick-strat-lkf53h9P-py3.13/Scripts/python.exe"
# .env line 3 holds SEC_USER_AGENT, required by edgar.set_identity
```

Scratchpad scripts, written 2026-08-22/23 under
`…/74f72906-bc7d-4152-b182-1f0740f86830/scratchpad/`:

| script | what it produces |
|---|---|
| `rosters.py` | the two 26-ticker rosters, `IN_SAMPLE` / `OUT_SAMPLE`, with GICS triples |
| `cx_facts.py` | companyfacts probe, name-filtered to investing payments. Caches to `p5/cf/*.json` (~74 MB, 18 tickers) |
| `cx_wide.py` | every concept with an annual USD duration fact, ranked — how the DTE/NEE extension blindness was found |
| `cx_xbrl2.py` | per-fact detail from `filing.xbrl()` **with the `dim_*` columns** — how DTE's subsidiary-only `PaymentsToAcquirePropertyPlantAndEquipment` was caught |
| `cx_link.py` | children of the investing node from `statement_arcs`, latest 10-K |
| `cx_hist.py` | **the evidence base**: every investing-node child arc + value, 17 tickers × 2011-2026 → `p5/hist/*.parquet` (941 filings, 19,001 rows, ~25 min on 4 workers) |
| `da_link.py` | children of the operating-activities node — the `depAmort` equivalent |
| `p4_in/shards`, `p4_oos/shards` | the Phase-4 facts ledgers (52 parquet, 302,467 rows) every coverage figure here is computed from |

Two gotchas that cost time and will cost it again:
- `df.itertuples()` **renames any column whose name starts with `_`** to a positional
  `_1` / `_9`. Name working columns `bare_name`, not `_bare`.
- `statement_arcs` returns the child in a column called **`concept`** (with
  `concept_taxonomy`), not `child_concept`; the parent is `parent_concept`.

#### 4b.11 What this section supersedes

- The Phase-4 carry-forward *"`capex` never resolves for 7 non-bank tickers"* → **9**, §4b.0.
- The Phase-4 carry-forward *"`depAmort` is annual-only for AFL and CSCO"* → that case is real
  and is §4b.9, but it concealed a **9-ticker total-absence** hole, §4b.0.
- `real_estate.capex`'s note in `fundamentals_exceptions.json` — *"a REIT's real capex IS
  reconstructible, it is simply tagged with real-estate-specific elements … the linkbase
  branch should find it"* — is **confirmed correct** by §4b.1, and is what this section
  implements. Leave `expected_absent: false` there.

#### Phase 4b — implemented 2026-08-23, OPTION A

**The user asked for Phase 4 onward "especially phase 4b" without naming an option, so
option A was taken — the one §4b.6 recommends, and the only one that makes the standard-leaf
half safe for APA, MAA, PLD and SWKS.** Under B, guard 3 nulls most of the tickers the
widened list was meant to fix.

**Files changed** (5 source + 3 config/test, ~250 lines of new logic):

| file | change |
|---|---|
| `src/data_extract/utils/fundamentals/xbrl_linkbase.py` | route 3b `statement_leaf_sum` (`_leaf_sum`, ~90 lines incl. rationale), `ArcGraph._children_with_role` + `children_on_role`, `ANCHOR_ROLES`, `PARTIAL_LEAF_SUM`, `_roll_up`, `dc_code_when_absent` in the terminal `unresolved` |
| `src/data_extract/utils/fundamentals/kpi_catalogue.py` | `Catalogue.filer_leaves`, `Catalogue.periodicity_shapes`, `by_ticker` / `by_ticker_periodicity` loading + contradiction and evidence validation |
| `src/data_extract/utils/fundamentals/fetch_fundamentals_sec.py` | `_materialise` handles route 3b on the same period intersection; `ticker=` threaded into `resolve_field` |
| `configs/fundamentals/fundamentals_kpis.json` | `capex.roll_up.any_of` at field level + `energy`/`utility`/`real_estate`/`insurer` regimes; `depAmort.roll_up.any_of`; `totalRevenue.sign_rationale` |
| `configs/fundamentals/fundamentals_exceptions.json` | `by_ticker` register (17 filers) + `by_ticker_periodicity` (AFL, CSCO) |
| `tests/data_extract/test_leaf_sum_resolution.py` | **new, 25 tests**: 16 synthetic guard tests + 9 real-data ground-truth tests |
| `tests/data_extract/test_periods_q4.py` | the Phase-4 **hold-out proof** folded in as a standing assertion (closes a Phase-4 carry-forward) |
| `tests/data_extract/test_linkbase_history.py` | passes `ticker=` so the 26-ticker sweep measures the production pipeline, register included |

##### The route, and why the guards are where they are

`roll_up.any_of` is a list of GROUPS: the first reported alternative *within* a group, summed
*across* groups, firing when at least one group hits. It sits between routes 3 and 4, so a
declared total still wins and AAPL/VLO keep their aggregates.

**Every grouping was set by measuring CO-OCCURRENCE, not by reading element names.** Two
concepts that never appear in the same filing are era or naming variants of one line and
share a group; two that co-occur are disjoint legs and get separate groups. The measurement
is decisive rather than suggestive — the counts land on exact filing totals:

| filer | group | counts |
|---|---|---|
| DTE | `PlantAndEquipmentExpendituresUtility` ∪ `PaymentsToAcquirePropertyPlantAndEquipmentUtility` | 60 + 3 = **63 = its filing count**, 0 co-occurrence |
| DTE | `...Utility` vs `...NonUtility` | co-occur in **all 60** → two legs |
| NEE | `CapitalExpendituresOfPublicUtility` ∪ `OfFPL` ∪ `OfFPLSegment` | 41 + 18 + 4 = **63**, 0 co-occurrence |
| NEE | `OtherCapitalExpenditures` ∪ `ProceedsPaymentsFromOtherCapitalExpenditures` | 49 + 14 = **63**, 0 co-occurrence |
| GOOGL | `Depreciation` ∪ 2 `goog:` spellings | 11 + 26 + 7 = **44**, 0 co-occurrence |
| SWKS | `AmortizationOfIntangibleAssets` ∪ `OtherDepreciationAndAmortization` ∪ `swks:...IncludingInventoryStepUp` | 4 + 13 + 17, 0 co-occurrence, and the middle two carry the **identical $296M** |
| MAA | `PaymentsToAcquireInProcessResearchAndDevelopment` ∪ `PaymentsToDevelopRealEstateAssets` | 19 + 12, 0 co-occurrence |
| EOG | oil-and-gas PP&E vs other PP&E | co-occur in **all 53** → two legs |

##### Four deviations from §4b.5's text, each forced by a measurement

1. **`insurer.capex` gets `dc_code_when_absent`, NOT the bank `dc_code`.** §4b.5(c)
   proposed the bank treatment and noted "PGR and CB currently resolve capex and would
   stop". Re-measured: **CB does not resolve capex at all** (63 filings, 0 values), and
   **PGR resolves it in 63 of 63** — one concept throughout, $65-364M, on annual, ytd9,
   ytd6 and quarterly windows alike. So the stated justification (*"an intermittently-tagged
   capex yields a TTM that silently mixes tagged and untagged quarters"*) is **false for the
   only insurer it would have cost**. A new key attaches the code to the ABSENCE instead:
   AFL/MET/CB read `not_applicable`, PGR keeps a complete real series. Strictly better than
   both the plan's option and a per-ticker override, and it generalises to the ~450 unswept
   tickers.
2. **`OtherDepreciationAndAmortization` joins the amortisation group** instead of being its
   own. §4b.5(c) put it in a third group; SWKS's co-occurrence table shows it is an era
   variant of `AmortizationOfIntangibleAssets` carrying the identical $296M, so a separate
   group would double-count the moment a filer tagged both. MET, whose only D&A line it is,
   is unaffected either way.
3. **`AmortizationOfFinancingCostsAndDiscounts` is NOT a D&A leaf.** §4b.2 listed it among
   SWKS's leaves ($4M). It is amortisation of deferred financing costs and debt discount,
   which ASC 835-30 puts in interest expense. Excluding it costs SWKS $4-6M and is what
   keeps **CB correctly `not_disclosed`** — admitting it would manufacture a $694M D&A line
   for a filer that has none.
4. **A refusal DECLINES; it does not terminate.** §4b.5 says guard 3 should "reason-code
   it". Implemented as a fall-through: the refusal is carried and only becomes a `dc_code`
   if routes 4 and 5 also fail. Without this, every filer parking an unrelated extension in
   the node would LOSE a value it already had — XOM (`xom:AdditionalInvestmentsAndAdvances`,
   45 filings), DUK (`duk:PurchasesOfEmissionAllowances`), EOG, GOOGL. This is what makes
   guard 3 safe to leave strict, and it is why the register lists `not_leaves` for filers
   that need no leaf at all.

##### Two of the plan's own figures were wrong, and both mixed fiscal years

- **§4b.7 says "ORCL FY2026 `depAmort` = $10,633M"**, which is FY2026's `Depreciation`
  ($7,623M) plus **FY2024's** `AmortizationOfIntangibleAssets` ($3,010M). ORCL's FY2026
  amortisation is $1,671M; the internally consistent figure is **$9,294M**.
- **§4b.6 says "SWKS `depAmort` … truth ~$618M"**, which is **FY2023** ($387.8M + $225.9M).
  FY2025 is $278.7M + $184.3M = **$463.0M**. The −37% partial-leaf claim itself is correct
  and reproduces exactly on FY2023.

##### Ground truth, verified live against the latest 10-K of each filer

| filer | field | route | value | check |
|---|---|---|---|---|
| **DTE** | capex | `statement_leaf_sum` on 2 `dte:` legs | **$4,429.0M** | = the filer's own $4,639M − $210M, **to the dollar** |
| **EOG** | capex | 2 us-gaap legs | **$6,594.0M** | = §4b.7's target exactly |
| **APA** | capex | 3 legs (1 us-gaap + 2 `apa:`) | $2,766.0M | §4b.7's "$2,740M + legs" |
| **MAA** | capex | 3 legs (2 us-gaap + 1 `maa:`) | $765.7M | = §4b.2's $360 + $272 + $133 |
| **NEE** | capex | 4 legs (1 us-gaap + 3 `nee:`) | $24,606M | FPL + NEER + other + nuclear fuel |
| **PLD** | capex | 4 legs (3 us-gaap + 1 `pld:`) | $5,471.3M | |
| **MSFT** | depAmort | 1 `msft:` leg | **$38,534M** | = §4b.7's target exactly |
| **GOOGL** | depAmort | `us-gaap:Depreciation` | **$21,136M** | = §4b.7's target exactly |
| **USB** | depAmort | 2 us-gaap legs | **$1,018M** (FY2023) | = §4b.7's $382 + $636 exactly |
| **ORCL** | depAmort | 2 us-gaap legs | $9,271M | see the correction above |
| **SWKS** | depAmort | 1 us-gaap + 1 `swks:` | $463.0M | see the correction above |
| **AAPL** | depAmort | **`linkbase_total`** | **$11,698M** | the §4b.4 REGRESSION GUARD: the note-level `Depreciation` is $8,000M, −31.6% |
| **VLO** | depAmort | **`linkbase_total`** | **$3,158M** | same guard; its note-level `Depreciation` is $2,300M |
| **PGR** | capex | `statement_leaf_sum` | kept | insurer that DOES tag capex |
| AFL / MET / CB | capex | `unresolved` | — | **`not_applicable`**, was `not_disclosed` |
| CB | depAmort | `unresolved` | — | **`not_disclosed`**, as §4b.2 requires |

AAPL's protection turned out to be double: its note-level `us-gaap:Depreciation` is **not a
child of the operating-activities node at all**, so guard 1 excludes it even before route 1
wins. The synthetic test asserts the guard directly, because a real filing cannot be made to
put the same concept on two roles on demand.

##### Three findings this work surfaced that it does NOT fix

1. **VLO's capex went dark in 2023.** Neither `PaymentsToAcquireProductiveAssets` nor
   `PaymentsToAcquirePropertyPlantAndEquipment` is reported undimensioned in any VLO filing
   from 2023-07 onward — both are DECLARED in its investing node and neither is tagged at
   consolidated scope. 21 of its 63 filings have no capex, and this was **already true
   before Phase 4b** (verified against the before-ledger row for accession
   `0001628280-26-011499`). A Phase-7 coverage finding, not a regression.
2. **CSCO's route-1 `depAmort` is almost certainly the PP&E-NOTE figure.** Its cash-flow D&A
   line is `csco:DepreciationAmortizationAndOther` at **$2,811M** in 45 of 48 node-carrying
   filings, while the annual `DepreciationDepletionAndAmortization` that wins route 1 is
   **$1,200M** — 2.3× smaller, and the same defect class as AAPL's note-level
   `Depreciation` but arriving through route 1 where route 3b cannot see it. Registering the
   extension was considered and **rejected**: route 1 keeps winning in the 12 filings that
   tag the note figure, so the series would carry two bases 2.3× apart and every growth rate
   across the boundary would be wrong. Fixing it needs a **route-1 statement-role test**,
   which is a different change. Recorded in `by_ticker_periodicity.CSCO` as
   `open_finding_do_not_silence`.
3. **`_values_by_period` picks the LAST duplicate, including a rounded one.** ORCL's FY2026
   10-K tags `us-gaap:Depreciation` twice for the same period — $7,623M and $7,600M — and
   "later duplicates win" takes $7,600M, a 0.3% haircut. Pre-existing and route-independent
   (it affects every route), but visible here for the first time because route 3b sums two
   legs and the arithmetic is checkable.

##### Carried forward from Phase 4b

- [ ] The register covers **17 of ~500 tickers**, and only those whose extensions were
      actually swept. Phase 9 must extend it as the roster widens; an unregistered filer
      keeps whatever the tag routes give it, which is correct-but-partial rather than wrong.
- [ ] `NEE`'s `PaymentsToAcquireBusinessesGrossAndRelatedCapitalExpenditures` (8 filings,
      $5,165M) is excluded as a MIXED acquisition-plus-capex line. That understates NEE's
      2018-19 capex by up to $5.2bn in those 8 filings and is the largest known cost of any
      decision in the register.
- [ ] `PLD`'s `pld:AcquisitionOfPropertiesNetOfCash` (4 filings) is tagged NEGATIVE
      (−$1,025M) and excluded; those 2011-12 filings understate.
- [ ] A route-1 **statement-role test** would close finding 2 above and is the natural
      Phase-7 companion to guard 1.

#### 4b.12 Acceptance — both rosters, measured 2026-08-23

Every number below is produced by `accept4b.py`, `per_ticker_2pc.py`, `explain.py` and
`defect_census.py` over the **output of `filing_rows`**, so this measures production rows and
not a reduced copy. BEFORE is the Phase-4 ledger (52 shards, 302,467 rows, swept 2026-08-22);
AFTER is the same sweep re-run against route 3b.

##### The three populations the request asks about

| | in-sample | out-of-sample |
|---|---|---|
| rows before / after | 143,490 / 144,131 | 158,977 / 159,513 |
| **(a) value NOW EXISTS where none did** | **1,150** (capex 597, depAmort 553) | **917** (capex 199, depAmort 718) |
| **(b) value existed and was WRONG, now corrected** | **116** (capex only; AAPL 29, SWKS 57, VLO 30) | **232** (capex only; DE 139, CAT 58, BA 35) |
| **(c) absence now CODED, was an unexplained null** | **188** (capex 126 → `not_applicable`, depAmort 62 → `partial_leaf_sum`) | **102** (capex 63 → `not_applicable`, capex 14 + depAmort 25 → `partial_leaf_sum`) |
| values LOST | **0** | 31 (see below) |
| tickers gained, capex | 15 → **20** (APA, DTE, EOG, MAA, NEE) | 22 → **23** (PLD) |
| tickers gained, depAmort | 22 → **26** (MET, PGR, SWKS, USB) | 21 → **26** (CB, GOOGL, MSFT, ORCL, UNP) |
| tickers lost, any field | **none** | **none** |

**Zero unexplained nulls on both rosters** — a row with no value now always carries a
`dc_code`. Out-of-sample went 394 → 0, in-sample 639 → 0. The in-sample figure needed the
ledger re-swept (39.2 min, 2026-08-23) because the original sweep straddled the `_compose`
fix; the re-sweep is byte-identical in row count (144,131) with **0 value differences** and
exactly 639 `dc_code` changes — 638 to `not_disclosed`, 1 to `no_usable_period` (JPM 2011
`pretaxIncome`, where `reported_concepts` matched bare and `_values_by_period` matched
namespaced). Confirming that the two ledgers differ ONLY in the reason codes is what makes
every value in this section safe to quote from either sweep.

`depAmort` has **0** changed values on both rosters: route 3b only ever fired where nothing
was there, which is what a fallback ordered behind the declared total should do. All of (b)
is `capex`, and all of it is `tag_fallback` → `statement_leaf_sum` — a single narrow leg being
replaced by the filer's own complete set. Median change +11.1% in-sample, +119.7%
out-of-sample; the largest are DE fiscal 2016 (+3.6×, $644.4M → $2,955.1M, adding
`PaymentsToAcquireEquipmentOnLease` — Deere's captive-finance lease fleet) and VLO fiscal
2017 Q1 (+1.14×, $279M → $596M, the basis inconsistency described in §4b.5).

The **31 lost out-of-sample values** are BA (23) and EQIX (8), all `tag_fallback` on
`PaymentsToAcquirePropertyPlantAndEquipment`. Cause: route 3b materialises on the
**intersection** of its legs' periods, so a filing that tags PP&E for a window while its
intangibles leg is tagged only annually loses that window. No filing lost capex outright
(BA: 63 of 63 filings carry a value before and after; EQIX 57 → 60), but BA's fiscal-2011
**annual** point is now absent, which is a gap in a series rather than a narrower number in
it. Recorded below as a carry-forward, because the obvious repair — sum whatever legs are
present — reintroduces exactly the mixed-basis inconsistency route 3b exists to prevent.

##### Are the calculations strong? The footing test, both rosters

Sum the four derived discrete quarters of a fiscal year and compare with the filer's **own
annual fact**. The engine never reads that annual when deriving Q1–Q3, so agreement is a test
rather than an identity.

| | in-sample | out-of-sample |
|---|---|---|
| complete (ticker, field, fiscal-year) points | 1,596 | 1,800 |
| within 2% of the filer's own annual | **99.12%** | **98.78%** |
| median relative error | 0.000000% | 0.000000% |
| p95 relative error | 0.0056% | 0.0042% |
| points involving a DERIVED quarter | 1,310 (82.1%) — **99.85%** within 2% | 1,467 (81.5%) — **99.73%** within 2% |
| points entirely AS-REPORTED | 286 — 95.80% within 2% | 333 — 93.39% within 2% |

The derived quarters are **more** accurate than the as-reported ones on both rosters. That
inversion is the strongest single result here: the Q4 = FY − YTD9 ladder is not the weak link,
and the residual sits in the filers' own re-presented figures.

Per field, within 2%: capex **100.0% / 100.0%**, operatingCashFlow **100.0% / 100.0%**,
depAmort 99.37% / 99.20%, totalRevenue 98.73% / 96.62%, operatingIncome 96.94% / 96.89%.

##### Every failure over 2%, attributed by mechanism

A four-quarter sum can miss the annual for exactly two reasons, and they are distinguishable
without external data: a **derivation error** leaves at least one quarter with
`basis != as_reported`, while a **restatement** leaves all four as-filed and the four then
foot to the FIRST-FILED annual, the gap equalling the annual's own movement between vintages.

| | in-sample (14) | out-of-sample (22) |
|---|---|---|
| restatement (ASC 205-20 reclassification) | **12** | **14** |
| derivation error | 2 (SPG depAmort 2013 = 4.09%, AMT depAmort 2023 = 2.67%) | 4 (MSFT depAmort 2026 and 2025, EQIX totalRevenue 2023, EQIX depAmort 2011 — largest 12.75%, on a fiscal year that is not yet complete) |
| reference itself defective | 0 | 4 (PG ×2, PLD ×2 — the first-filed annual is a note-level fact, so neither reference is clean) |

The restatement cases are unambiguous. VLO operating income fiscal 2012: first-filed
$4,010M, **equal to our four quarters to the dollar**, last-restated $5,044M (+25.8%). DTE
2019 $1,707M → $1,430M. SPG revenue 2012 $4,880M → $4,256M. MET revenue 2015 $69,951M →
$61,343M. Every one has at least 2 annual vintages and 0 derived quarters. This is the same
population as Phase 4's independently-measured *"4.53% of annual windows move by more than
2% between their first and last filing"*.

##### The independent proof of route 3b: leaf sum vs declared total

Where two filings report the same (ticker, field, period) and one resolves by a declared
TOTAL while the other resolves by `statement_leaf_sum`, the two numbers come from disjoint
evidence — the filer's own aggregate against the sum of its own leaves.

| | in-sample | out-of-sample |
|---|---|---|
| comparable points / tickers | 89 / 13 | 94 / 13 |
| exact to the dollar | 76.40% | 78.72% |
| within 2% | 78.65% | 80.85% |
| median relative error | 0.000000% | 0.000000% |

The disagreements are **not** route-3b errors. In every case the leaf sum is the correct
figure and the declared total is a narrow or misused line:

- **MCD capex** — leaf sum **$2,741.7M** for fiscal 2018, which is MCD's own
  `PaymentsToAcquirePropertyPlantAndEquipment` tag for that window, against
  `PaymentsToAcquireProductiveAssets` **$101.7M** from the two earlier vintages of the same
  year. Both numbers are the filer's, for the same period, and they differ by 27×: MCD uses
  the "productive assets" concept for a much narrower investing line than its name implies.
- **VLO capex** — `PaymentsToAcquireProductiveAssets` $577M against a leaf sum of $864M; §4b.5
  proves the identity PP&E + CapImp + VIE capex to the dollar for fiscal 2016.
- **MTB depAmort** — $91.5M declared against $138.4M of leaves.
- **AAPL capex** fiscal 2013 — $8,165M declared against $9,076M including the intangibles leg,
  which the catalogue's own cited FASB definition requires.

##### The residual, as a number rather than a caveat

`statement_leaf_sum` carries a statement-role guard. Routes 1, 5 and 6 do not, and that is now
the dominant remaining defect class. Four shapes, screened across both rosters:

| defect | in-sample | out-of-sample | status |
|---|---|---|---|
| **D1** annual mislabelled as a quarter, annual ALSO present | 0 | 5 rows, ORCL `totalRevenue`, FY2018/19/21/22 | **FIXED** — `_drop_annual_masquerading_as_quarter` |
| **D1b** annual present ONLY under a quarterly window | 0 | 1, ORCL `totalRevenue` FY2020 | carried forward |
| **D2** latest annual vintage off-scale (<0.5× or >2× the largest vintage) | 0 | 3 — MCD `depAmort` 2018 and 2019 ($214.8M vs $1,482.0M), PG `totalRevenue` 2012 ($28,400M vs $83,680M) | carried forward |
| **D3** consistently-narrow declared total (invisible to any cross-vintage test) | VLO, MTB, AAPL, SWKS | MCD capex (64 rows, 10 fiscal years, ~12× low), CAT capex (84 rows) | partially fixed; see below |

D1 is fixed. The guard refuses a `quarterly`-shaped fact when (1) an annual fact ends within 7
days of it, (2) the two values agree to 0.1%, and (3) an interim cumulative inside the same
year is materially non-zero — condition 3 being what keeps a legitimate Q4-equals-FY series
(nine months of zero) alive. Effect: out-of-sample footing **98.56% → 98.78%**, ORCL revenue
from 4 failing fiscal years to **0**, all 15 years now footing to within 0.003%, and Q4 fiscal
2022 reading $11,840M against Oracle's published $11.8bn. In-sample: **unchanged at 99.12%**,
so no collateral damage. Three tests in `test_periods_q4.py` cover the masquerade, the
legitimate Q4-equals-FY case, and an ordinary Q4.

D3 is the important residual. Route 3b is a **fallback**, so where a filer declares a total
that is narrower than the field's definition, route 1/5/6 keeps winning and route 3b never
runs — leaving two eras on different bases. 14 (ticker, field) pairs carry a mixed basis on
each roster; 4 in-sample and 2 out-of-sample disagree by more than 2%. **MCD capex is the
worst case found anywhere: fiscal 2008-2017 reads $77-186M on the narrow concept while
2018-2025 reads $1,641-3,365M on the leaf sum, so the series steps 35.6× across the single
2017→2018 boundary where route 3b takes over** (median leaf-vs-total gap on comparable
points: 12.0×). No cross-vintage test can see it, because MCD tags the same narrow concept
consistently in every filing of the earlier era.

##### Carried forward from 4b.12

- [ ] **A statement-role test on routes 1, 5 and 6.** This single change closes D1b, D2, D3,
      the CSCO route-1 note-level defect from §4b.11, and the AMT `LongTermDebt` defect
      ($1.9M note-level against $21,127M). Confirmed instances now number **8 tickers**
      (AMT, AAPL, CSCO, MCD, PG, plus the three D2/D3 rows). It is deliberately NOT done here:
      it reorders resolution for all 52 tickers and the plan's own guidance is not to ship an
      unmeasured design into this field.
- [ ] **Route 3b's period intersection** costs 31 out-of-sample values and one BA annual
      point. Summing only the legs that are present would recover them at the cost of a mixed
      basis; needs measuring before choosing.
- [ ] **MCD capex** needs either a `never_use` entry for its misuse of
      `PaymentsToAcquireProductiveAssets` or the route-1 statement-role test. Until then its
      pre-2018 capex is unusable and its 2017→2018 growth rate is meaningless.
- [ ] **ORCL fiscal 2020 `totalRevenue`** has no annual fact at all — all three vintages stamp
      the full-year `us-gaap:Revenues` with a Q4 window, so `FY − YTD9` cannot run and Q4
      reads $39,068M instead of about $10,439M. Repair is to reclassify the fact's duration,
      which is inference over a filer's own tagging and wants its own decision.
- [ ] `longTermDebt` **completeness is not measurable by the quarter grid.** Many filers tag it
      only in 10-Ks, and a company with no long-term debt legitimately has no fact (AAPL
      pre-2013, META pre-2022, SWKS pre-2020, MET, PGR, GS, JPM, SPG). Its cross-vintage
      agreement is 17/23 in-sample and 17/26 out-of-sample; the disagreements are concept
      switching between `LongTermDebt`, `LongTermDebtAndCapitalLeaseObligations` and
      `LongTermDebtNoncurrent` — three genuinely different bases. Note that
      `us-gaap:LongTermDebt` sits at priority 2 in the fallback list but INCLUDES the current
      portion, contradicting the field's own definition. That is a **Phase-3 catalogue** issue,
      not a Phase-4b one, and UNP shows it on 16 of 16 comparable quarter-ends.

---

### Phase 5: The history layer ⬜

**Goal**: `fundamentals_history`, ~71 columns, one regime column, one reason-code side table, on a
**publication-event grain**.

---

#### 5.0 The grain: one row per publication event

**This supersedes the earlier "latest-known rebuild" reading of decision #6.** That design leaked,
and the leak is measurable.

**The problem.** A `10-Q/A` or `10-K/A` restates a quarter that was already filed. Under a
latest-known rebuild, `_resolve_latest_per_period` picks the amendment's value and writes it back
onto the row keyed by the **original** filing date. A model reading that row believes the corrected
number was public months before it existed.

Measured on `fundamentals_facts_legacy` (7.8M facts, 25,279 filings):

| | |
|---|---|
| amendments | **246** (`10-K/A` 138, `10-Q/A` 108) = **0.97%** of filings, 91 + 83 tickers |
| carrying **<10 facts** — Part III / cover-page only | **88 (36%)** |
| carrying 10-99 facts | 7 (3%) |
| carrying 100-399 facts | 113 (46%) |
| carrying **400+** facts — full restatement | **38 (15%)** |
| lag original → amendment | avg **89 d**, min 0, **max 921 d** |
| landing **> 90 days** after the original | **57 of 217 (26%)** |

A quarter of all amendments arrive more than a full quarter late, and the worst is **2.5 years**.
That is the exact failure `test_fundamentals_point_in_time.py` is deliberately red on.

**The rule.** `fundamentals_history` is keyed on **when information became public**, not on the
period it describes. `as_of` is a filing date — an original's or an amendment's.

- [ ] **A row is emitted for every (ticker, date) on which ≥1 extracted value became newly public.**
      Originals always qualify.
- [ ] **An amendment emits a row only if it actually changes ≥1 extracted value.** This is what
      discards the 88 Part III / cover-only amendments — and it is stricter than a fact-count
      threshold, because an amendment can re-tag 200 facts to identical values and still be a no-op.
- [ ] **The row is a complete snapshot.** Every column carries its latest-known value as of that
      date; unamended columns are identical to the previous row. The row is self-contained, so a
      plain `asof` merge works with no reconstruction.
- [ ] **Rows are immutable once written.** The nightly job only ever appends.
- [ ] **Same-day collapse**: group publication events by `(ticker, date)`, never by accession. Two
      filings on one day — including an amendment for a different quarter — produce **one** row
      reflecting both.

**Row-count impact: +158 rows on 27,602 (+0.6%).** Essentially free.

**This is NOT the "full vintaged history" the user declined.** That option added `knowledge_date`
to the PK and appended a complete vintage on **every nightly rebuild** (~365/year regardless of
filings), forcing every consumer to pick a vintage. This keeps the PK at **`(ticker, as_of)`**, adds
rows only when a **filing** makes new information public (~4-6/ticker/year), and downstream
`PitFrames` forward-fill works **unchanged** — an amendment row is simply a genuine new observation.
The two decisions are compatible; this one is strictly cheaper.

**Two consequences worth stating plainly, because they are not obvious:**

1. **"The value that was edited" propagates further than one cell.** `fundamentals_history` stores
   **TTM levels**, not discrete quarters. Restating Q1-2024 revenue changes the TTM for Q1, Q2, Q3
   *and* Q4 2024, and `revenueGrowth = pct_change(4)` moves as well. That is mechanically correct,
   not a violation of the intent: the amendment row must show the world **recomputed with the
   restated quarter**. What stays frozen is the **earlier rows**, which keep their as-filed values
   forever. That is where the no-leakage property lives.

2. **`fiscal_end` must stay monotone in `as_of`.** An amendment to Q1-2024 filed 2024-08-20, after
   Q2-2024 was filed 2024-05-01, gets `fiscal_end = 2024-06-30` — the latest period **known** at
   that date — not 2024-03-31. Otherwise `fiscal_end` runs backwards as `as_of` advances and breaks
   any consumer assuming order. The restated period is recorded separately in
   `amended_fiscal_end` so the information is not lost.

**Provenance columns this adds**: `publication_form` (`10-K`/`10-Q`/`10-K/A`/`10-Q/A`) ·
`is_amendment` · `amended_fiscal_end` (the period restated, NULL on originals) · `amended_fields`
(the list that actually changed).

**Operational corollary — a code change is not a publication event.** Re-deriving history after a
bug fix rewrites numbers under an already-trained model, which is precisely the vintage instability
the research measured elsewhere (earnings sign flips ~14% across vintages; ~50% of anomalies change
inference). So: nightly **appends only**; a code change requires an explicit, logged, versioned
`--rebuild` — never a silent overwrite.

**This makes the PIT invariant structural.** No row can carry a value that postdates its own
`as_of`, by construction rather than by guard. `test_fundamentals_point_in_time.py` going green is
therefore the defining acceptance test of this phase, not a Phase 4 side effect.

---

#### 5.1 Fields

**Field list — the contract.** 40 tiered KPIs, each with its definition in
`configs/fundamentals/fundamentals_kpis.json`:

**Tier 1 (11) — every ticker, every regime**

| field | definition | authority / trap |
|---|---|---|
| `totalAssets` | `Assets`, consolidated incl. NCI | the one BS total with no regime variant |
| `totalLiabilities` | `Liabilities`; where untagged, `totalAssets − equityInclNCI` | Rule 5-02 has **no** "Total liabilities" caption → elective. The fallback is correct by construction |
| `stockholdersEquity` | **incl-NCI** consolidated basis | `…IncludingPortionAttributableToNoncontrollingInterest`, else `StockholdersEquity + MinorityInterest` |
| `cash` | **decision #5**: cash + equivalents + restricted + short-term investments | ASU 2016-18 `230-10-45-4` makes the restricted-inclusive total the CF anchor (public FY2018, **retrospective**). Bank: `CashAndDueFromBanks` is the *first line*, not the total |
| `totalDebt` | **decision #4**: gross ex-lease debt + finance leases + operating leases | both legs must be on the **same** basis (§2.3). FASB BC264 says operating leases are *not* debt-like — the user has chosen the S&P/Moody's treatment over Fitch's. Pre-FY2019 operating leases are off-BS → `regime_break` flag |
| `sharesOutstanding` | `dei:EntityCommonStockSharesOutstanding` (cover page) | the **only** summable tag for multi-class |
| `totalRevenue` | regime-dependent, see below | the ASC-606 cutover (2018-01-31) is a dated universe-wide taxonomy event |
| `netIncome` | `ProfitLoss` preferred (consolidated) | FASB's own `scf-indir` linkbase has exactly one arc into OCF: `← +1 ProfitLoss`. **A validated existing decision — do not "fix" it** |
| `operatingCashFlow` | `NetCashProvidedByUsedInOperatingActivities` | |
| `ebitda` | **derived, non-GAAP** = `operatingIncome + depAmort` | no us-gaap element. SEC Item 10(e)(1)(ii)(A) names EBITDA by exception |
| `freeCashflow` | **derived, non-GAAP** = `operatingCashFlow − capex` | C&DI 102.07: *"does not have a uniform definition"*; it is a **liquidity** measure — **never per-share** |

`totalRevenue` by regime (§5.2/§5.3/§5.4):
- **Industrial / E&P / utility**: the filer's own linkbase revenue root (`Revenues`,
  `RevenuesAndOther`, `RegulatedAndUnregulatedOperatingRevenue`, …).
- **Bank**: `coalesce(Revenues, InterestIncomeExpenseNet + NoninterestIncome, InterestIncomeExpenseNet)`.
  **Never fall back to `InterestAndDividendIncomeOperating`** — that is *gross* interest income and
  would inflate revenue by the entire interest-expense leg. Note `Revenues` for a bank already means
  *net* revenue (JPM CY2024 = $177,556M) — the same tag silently switches basis by regime.
- **Insurer**: `PremiumsEarnedNet + NetInvestmentIncome + RealizedInvestmentGainsLosses`. The ASC-606
  element captures **3.2%** of MetLife — preferring it understates a life insurer by ~30×.
- **REIT**: `Revenues`, else `OperatingLeaseLeaseIncome`.

**Tier 2 (12)**: `pretaxIncome` · `incomeTaxExpense` · `longTermDebt` (**ex-lease**; if only
`LongTermDebtAndCapitalLeaseObligations` is tagged, subtract the noncurrent finance-lease leg) ·
`shortTermDebt` (**the §2.3 fix**: `LongTermDebtCurrent + ShortTermBorrowings`) · `dilutedShares` ·
`epsDiluted` (**decision #9 — computed as `netIncome_ttm / dilutedShares_ttm`, never a sum of four
quarterly EPS**; EPS is not additive, and four summed quarters drift from annual EPS as the share
count moves. edgartools hit the same non-additivity from the other side: its naive Q4-EPS derivation
matched AAPL in only **5 of 17 years** (GH #690). The as-reported `EarningsPerShareDiluted` is still
*extracted* into `fundamentals_facts` and used by the validator as an independent cross-check on the
computed value — it is simply not the published number) · `depAmort`
(`DepreciationDepletionAndAmortization`, the FASB-declared *aggregate*; **not**
`DepreciationAndAmortization`, officially non-production only) · `retainedEarnings` ·
`stockBasedComp` · `effectiveTaxRate` (ratio — instant path, never TTM-summed) · `capex` (see below)
· `employees` (10-K body text; no XBRL concept).

`capex`: `PaymentsToAcquireProductiveAssets` is officially a **superset** of
`PaymentsToAcquirePropertyPlantAndEquipment` — *"property, plant and equipment (capital
expenditures), software, and other intangible assets"* — which is why they disagree **37.1%** of the
time while sitting at priority 1 and 0 of the same list. Resolution: the filer's linkbase capex
total, else the weighted sum of its declared leaf payment elements, else PP&E. Two traps encoded:
MAA tags **$272M of development capex as `PaymentsToAcquireInProcessResearchAndDevelopment`**, and
`230-10-45-13(c)` pushes seller-financed principal into *financing*, so this is **not** gross PP&E
additions. Bank capex is **not reliably reconstructible** (JPM/BAC/USB do not appear in the
`PaymentsToAcquirePropertyPlantAndEquipment` CY2024 frame at all) → `dc_code = 'not_applicable'`,
and bank FCF is therefore null by design.

**Tier 3 (16)**: `ppeNet` · `interestExpense` · `goodwill` · `operatingIncome` · `ppeGross` ·
`accumulatedDepreciation` · `intangiblesExGoodwill` · `currentAssets` · `currentLiabilities` ·
`sellingGeneralAdmin` · `accountsReceivable` · `accountsPayable` · `grossProfit` · `costOfRevenue` ·
`inventory` · `minorityInterest`.

Three authoritative notes that change how these are validated:
- **`GrossProfit` and `OperatingIncomeLoss` were never required line items, for anyone.** Run against
  Rule 5-03's text: `"gross profit"` → **0 occurrences**; `"operating income"` → **1**, inside
  caption 7 "Non-operating income". Their absence is **not** a filing defect. Reconstruct from
  captions 1-9.
- **`CostOfRevenue` *is* required** — captions 1 and 2 are a matched five-way pair and 5-03(b)
  enforces the pairing.
- **`AssetsCurrent`/`LiabilitiesCurrent` are "when appropriate"** (Rule 5-02 captions 9 and 21).
  `17 CFR 210.1-02(bb)(1)(i)` is the citable authority for *"specialized industries in which
  classified balance sheets are normally not presented"*. → structural flag, never a null count.

**Tier 3R (1) — `researchAndDevelopment`, re-added by decision #10, regime-gated.**

Coverage 225/491 — universal only in Info Tech (65/72), Materials (17/26), Health Care (42/57).
Compustat's own `XRD` definition ends *"This item is not available for banks"*. So:
`regime_gated: true`, **never universe-z-scored**, and `dc_code = 'not_applicable_for_regime'`
outside the R&D-intensive cohorts.

The user's requirement — *consistent between tickers and time* — is not free here. Measured on the
7.8M-fact dump:

| | |
|---|---|
| `ResearchAndDevelopmentExpense` vs `…ExcludingAcquiredInProcessCost`, both tagged, same period | 21 pairs / 4 tickers |
| agree within 1% | **0.0%** |
| off by >50% | 23.8% |
| mean ratio (aggregate ÷ ex-IPR&D) | **1.675** |
| tickers using exactly **one** concept across full history | **198 of 212 (93%)** |
| ticker-years with both co-tagged (a reconcilable overlap) | **1.0%** |
| undimensioned facts | `…Expense` **10,429** vs `…ExcludingAcquiredInProcessCost` **1,284** |

Read: the two concepts are **superset and subset** — the aggregate includes acquired in-process R&D
write-offs, the other excludes them — and they are decisively **not** substitutes (0% agreement is
worse than capex's 42.9%). But R&D is **well-behaved over time**: 93% of tickers never switch. The
danger is therefore cross-ticker basis mixing, not within-ticker drift.

Rules, all encoded in the catalogue entry:
- [ ] **One basis universe-wide: `ResearchAndDevelopmentExpense`** (the aggregate). Chosen over the
      economically-cleaner ex-IPR&D basis purely on coverage — 8× more facts — since mandating the
      narrow concept would collapse the field to ~4% of filers.
- [ ] **Record the basis per row.** Where a filer tags only `…ExcludingAcquiredInProcessCost`, use it
      and stamp `dc_code = 'basis_ex_iprd'`. Do **not** silently coalesce the two as today's
      priority list does — that is the exact defect this rebuild exists to remove.
- [ ] **Do not attempt an IPR&D bridge.** `ResearchAndDevelopmentInProcess` (130 facts) was co-tagged
      in **0 of the 21** overlap pairs, so the two bases cannot be reconciled from a third element.
- [ ] **Never coalesce `ResearchAndDevelopmentExpenseSoftwareExcludingAcquiredInProcessCost`**
      (532 facts) into this field — ASC 985-20 software R&D is a different measure.
- [ ] **Never let `PaymentsToAcquireInProcessResearchAndDevelopment` in** — it is an *investing cash
      outflow*, and it is the element MAA uses to tag **$272M of multifamily development capex**. A
      name-keyed R&D extractor books a REIT as an R&D spender.
- [ ] The 14 switchers get a `tag_switch_break` flag from Phase 7; with 1.0% co-tagging there is no
      overlap year to reconcile against, so a switch is an unreconcilable level step, exactly as for
      revenue and capex.
- [ ] **Known residual, flagged not fixed**: capitalized software (ASC 350-40 / 985-20) depresses
      reported R&D expense, so a capitalizing filer looks like it spends less than an economically
      identical expensing one. This is *not* recoverable from the R&D tag. `capitalizedSoftware` is
      in the removed set — re-add it as a `tier: 0` companion if this ever matters for the factor.

**Calculation inputs (13)** — carried because the *chosen definitions* require them, not as scored
KPIs. Excluded from z-scores and from peer ranks; `tier: 0` in the catalogue:
`longTermDebtCurrentOnly` · `shortTermBorrowingsOnly` · `financeLeaseLiability` ·
`operatingLeaseLiability` (→ decision #4) · `shortTermInvestments` · `restrictedCash` (→ decision
#5) · `netInterestIncome` · `noninterestIncome` (→ bank `totalRevenue`) · `premiumsEarned` ·
`netInvestmentIncome` · `realizedInvestmentGains` (→ insurer `totalRevenue`) · `rentalIncome` (→
REIT `totalRevenue`) · `basicShares` (→ `optionOverhang`).

**Derived (10)**, kept from `_derive_history`: `grossMargins` · `operatingMargins` ·
`profitMargins` · `returnOnEquity` · `debtToEquity` · `revenue_q` · `netIncome_q` · `revenueGrowth`
· `earningsGrowth` · `optionOverhang`.
**Dropped (4)** — their inputs leave with the 78: `totalAssetsExLease` · `nonServicePensionCost` ·
`exciseTaxAdjustment` · `debtMaturity5yTotal`. Consumers repaired in Phase 6.

**Changes**:

1. `src/data_extract/utils/fundamentals/build_history.py` — new. Long → wide. Target **< 300 lines**.
   - [ ] **Driven by the publication-event ladder of §5.0**: enumerate each ticker's distinct filing
         dates, and for each one rebuild the full snapshot from facts with `filing_date <= that
         date`. This *is* the `as_of_cutoff` replay the old `derive_fundamentals_history` already
         supported — promoted from an audit-only debug path to the production build loop.
   - [ ] `as_of` is **no longer computed**. `_assemble_base`'s median-of-spine heuristic disappears:
         under the event grain `as_of` simply *is* the filing date. Simpler and more correct.
   - [ ] The two leak-guard passes survive as a **redundant** check — under the event grain they can
         no longer fire, which is exactly what makes them a good assertion.
   - [ ] Emit a row only when the snapshot differs from the previous one (the no-op amendment rule).
   - [ ] Reads `fundamentals_facts` **projected and filtered** (`columns=`/`where=`), never
         `SELECT *` per ticker as `_derive_history` does today. Note the replay is O(filings) per
         ticker, so the per-ticker facts frame is loaded **once** and sliced in memory — never
         re-queried per event.
   - [ ] Adds `regime` (from `configs/fundamentals/fundamentals_regimes.json`) and the §5.0 provenance columns to
         every row.

2. `fundamentals_reason_codes` — new long side table
   `(ticker, as_of, field, dc_code, combined_into)`.
   - **Deviation from the spec, stated explicitly**: the user asked for a Compustat-style `_DC`
     *companion column per value*. That is 39 extra columns and a ~50% wider table, almost entirely
     NULL. A sparse long table carries identical information (including the "combined into"
     destination) at a fraction of the width and joins in one line. **Reversible if the user
     prefers the literal companion-column form.**
   - Codes: `not_disclosed` · `not_applicable_for_regime` · `combined_into` (with destination) ·
     `failed_hard_guard` · `insufficient_quarters` · `regime_break`.
   - Compustat's own caveat is worth inheriting as an expectation: only ~1.2% of missing `XRD` is
     coded there, so most blanks stay unexplained even at a vendor.

3. `src/data_extract/utils/fundamentals/employees.py` — re-homed from `fundamentals_employees.py`,
   unchanged in behaviour (`employees` is Tier 2).

4. Schema + wiring:
   - [ ] `Tables.fundamentals_history` new column set; `Tables.fundamentals_reason_codes`;
         `Tables.fundamentals_quality` (Phase 7). `sql/schema.sql` rewritten for all three.
   - [ ] `step_extract_fundamentals.py` and `cli.py` re-pointed at the new entry points.

**Verification**:
- [ ] Column count is exactly as contracted; every column traces to a catalogue entry.
- [ ] **The PIT invariant**: for every row, every contributing fact has
      `filing_date <= as_of`. `test_fundamentals_point_in_time.py` goes green.
- [ ] **Amendment round-trip, on a real restating ticker**: assert the original row keeps its
      as-filed value *unchanged*, a new row exists at the amendment's own `filing_date`, and only the
      amended field (plus the TTM/growth columns whose window contains it) differs. Print both rows
      side by side — this is the sanity-check conclusion for the phase.
- [ ] **No-op amendments emit nothing**: assert the 88 <10-fact amendments produce **0** rows.
- [ ] `fiscal_end` is monotone non-decreasing in `as_of` for every ticker.
- [ ] Same-day collapse: no `(ticker, as_of)` duplicate; a ticker filing twice in one day gets one row.
- [ ] Idempotency: running the build twice appends **0** rows the second time.
- [ ] Bank/insurer/REIT top lines are non-null for the tickers §2.9 measured as broken (the 11 banks
      with NII but no noninterest income, the 6 insurers with premiums but no NII, the 17 with
      neither leg).

**Estimated effort**: 2-3 days (the event-ladder replay adds ~0.5 day over a period-grain build).

---

### Phase 6: Downstream repair ⬜

**Goal**: `data_aggregate` compiles and produces a cube against a 68-column table instead of 239,
with every reference to a removed column gone and every test that asserted on one either repaired or
deleted.

This phase exists **because** of the strict-Tiers-1-3 choice. It is not optional and it is not small.
Measured:

| | n |
|---|---|
| `fundamentals_history` columns today | 239 |
| kept (11 T1 + 12 T2 + 16 T3 + 1 T3R + 13 inputs + 10 derived + 5 keys) | **68** |
| **removed** | **171** |
| …of which **consumed downstream today** | **78** |
| …of which read by nothing | 93 |
| `SECTOR_KPI_COLS` that lose ≥1 input | **27 of 57** (29 before R&D was re-added) |
| test files that break | **8 (1,878 lines)** |

Every removed field — with its us-gaap concepts, measured ticker coverage and the original evidence
note — is catalogued in **[reports/adhoc/fields.md](../../adhoc/fields.md)**, organised as a rebuild
menu with a priority ranking and an add-one-back recipe.

---

#### 6.0 Column-removal mechanics

Deleting the columns is not just a DDL edit — the names are referenced in six places and a miss in
any of them is a silent `KeyError` at cube build time.

- [ ] `sql/schema.sql` — drop the 171 columns from the `fundamentals_history` DDL. **Risk zone**:
      propose the diff, get approval.
- [ ] `src/data_store/schema.py` — `Tables.fundamentals_history`: `read_columns` /
      `optional_columns` if the new spec declares a projection.
- [ ] `src/constants/constants.py` — `SECTOR_KPI_SCOPE` (:506) enumerates sector→KPI membership and
      references removed KPI names; the sector/industry blocks at :479-525 need the same pass.
      **Risk zone.**
- [ ] `src/data_aggregate/utils/common/sources.py` and `pit.py` — projection lists.
- [ ] `configs/build_cube.yml` (17.6 KB) — grep for removed field and KPI names; feature toggles and
      any explicit column lists live here.
- [ ] **The live table**: the rebuild writes to a fresh `fundamentals_history`;
      `fundamentals_history_legacy` keeps all 239 columns untouched as the reference. Do not
      `ALTER TABLE DROP COLUMN` anything.

Guard against the silent-miss failure mode:
- [ ] Add a test that asserts **no name in the removed set appears anywhere under `src/` or
      `configs/`**. That single test replaces manual grepping and stays useful after this phase.

---

#### 6.1 Builders

1. `src/data_aggregate/utils/fundamentals/sector_features.py` (392 lines) — the hardest hit.
   **Measured: 27 of the 57 `SECTOR_KPI_COLS` lose an input.** (29 before decision #10 re-added `researchAndDevelopment`, which revives `rd_capitalized_roic` — the Damodaran capitalized-R&D construction at `_capitalized_rd` — and `capex_to_rate_base`.)

   | dead KPI | removed input(s) |
   |---|---|
   | `payout_ratio`, `buyback_intensity`, `reinvestment_rate`, `ffo_payout`, `affo_dividend_coverage` | `dividendsPaid`, `buybacks` |
   | `efficiency_ratio`, `bank_roa`, `bank_operating_margin`, `provision_rate` | `noninterestExpense`, `provisionForCreditLosses` |
   | `loan_to_deposit`, `loan_growth`, `npl_ratio`, `net_charge_off_rate`, `deposit_stickiness` | `loans`, `deposits`, `depositsDomestic`, `nonaccrualLoans`, `netChargeOffs` |
   | `aoci_to_equity`, `reserve_coverage_velocity` | `accumulatedOCI`, `htmSecurities`, `htmSecuritiesFairValue`, `htmUnrealizedLoss`, `allowanceCreditLosses`, `tier1CapitalRatio` |
   | `loss_ratio`, `premium_growth`, `float_growth` | `claimsIncurred`, `dacAmortization`, `premiumsWritten`, `insuranceReserves` |
   | `rental_margin`, `investment_income_ratio` | `straightLineRent`, `aboveBelowMarketLeaseAmort`, `gainOnDispositions`, `realEstateImpairment` |
   | `exploration_intensity`, `ddna_intensity`, `property_overvaluation_cushion` | `explorationExpense`, `depletionDDA`, `oilGasPropertyNet` |
   | `deferred_rev_intensity`, `rpo_coverage`, `patent_cliff`, `bad_debt_intensity` | `deferredRevenue`, `remainingPerformanceObligation`, `regulatoryAssets`, `amortizationIntangibles`, `provisionDoubtfulAccounts` |

   *Attribution at block boundaries is approximate (two independent scans agree on the count of 29
   and on the input set; confirm each row at implementation).* The 30 survivors are the universal
   ones — `effective_tax_rate`, `accruals_ratio`, `asset_turnover`, `capex_intensity`, `roic`,
   `earnings_quality`, DSO/DIO/DPO, `combined_ratio`, `ffo_margin`, `net_debt_to_ebitdare`, … —
   because their inputs are all Tier 1-3 or calculation inputs.
   - [ ] Decide per KPI: delete, or re-add its input as a catalogue `tier: 0` entry. **Enumerate the
         list and get the user's call before deleting** — this is signal, not plumbing.
   - [ ] Whatever is deleted, remove the name from `SECTOR_KPI_COLS` **and** from
         `constants.SECTOR_KPI_SCOPE`, or the availability gate will look for a column that no
         builder emits.

2. `fundamental_features.py` (1,606 lines) — **44 removed fields referenced, the largest single
   consumer.** Loses `changeInReceivables/Payables/Inventory` → **DSO / DPO / DIO /
   cash-conversion-cycle and the Beneish M-score all break**; plus the special-items family
   (`impairment`, `restructuring`, `goodwillImpairment`, `litigationExpense`, `discontinuedOps`,
   `unusualItems`, `gainOnSaleGeneric`, `bargainPurchaseGain`), the tax family (`deferredTaxAssets`,
   `valuationAllowance`, `unrecognizedTaxBenefits`, `incomeTaxesPaid`), and
   `capitalizedSoftware`, `pensionDeficit`, `preferredEquity`, `remainingPerformanceObligation`,
   `reportableSegments`, `researchAndDevelopment`.
3. `dividend_features.py` (146) — loses `dividendsPaid` (and `buybacks`/`dividendsPerShare` via the
   panel). The per-share ex-date history (`Tables.dividends`) survives, so **yield survives**; payout
   ratio, FCF coverage and buyback yield do not.
4. `capital.py` (175) — 5 removed fields: `operatingLeaseRouAsset`, `marketableSecuritiesCurrent`,
   `assetRetirementObligation`, `pensionDeficit`, `totalAssetsExLease`. `net_debt` follows the new
   `totalDebt` (the lease legs are folded into it by decision #4).
4b. **`src/utils/tiingo_comparison.py` (10 removed fields) and `yahoo_comparison.py` (4)** —
   the external ground-truth comparators, easy to forget because they are not cube code. Both are
   consumed by `fundamentals_audit.py`, which Phase 1 recommends keeping.
4c. `src/validate/analyze_history.py` (10) and `fundamentals_validation.py` (7) — both are deleted
   in Phase 1 and rebuilt in Phase 7, so their references die with them. Listed here only so the
   grep-guard test in §6.0 does not surprise anyone.
5. `factors.py` — the accruals family follows `netIncome`/`operatingCashFlow` (both Tier 1) and is
   fine; check for `changeIn*` use.

5b. **`infer_yoy_periods` breaks on the amendment grain — found while checking §5.0 downstream.**
   [pit.py:93-106](../../../src/data_aggregate/utils/common/pit.py#L93-L106) infers "how many rows
   make one year" from the pooled median `as_of` gap (→ 4) and growth is then `pct_change(N)`, a
   **row** offset. A ticker with an amendment has 5 rows in that year, so 4 rows back is ~9 months,
   not 12 — a seasonality contamination in exactly the tickers this rebuild is trying to handle
   better. The pooled median itself is safe (158 short gaps in ~27,000, clamped and rounded).
   - [ ] Fix: compute growth on a **365-day `asof` offset**, not a row offset. More correct
         regardless of cadence, and it makes the growth columns immune to filing-frequency changes.
   - [ ] Everything else in `PitFrames` is unaffected — it pivots on `as_of`, sorts, and
         forward-fills onto the trading index, so an amendment row is simply a new observation that
         takes effect on its own date. **Verified by reading `pit.py`, not assumed.**
6. `employee_features.py`, `governance_features.py`, `step_cube_extras.py`, `step_cube_target.py`,
   `pit.py`, `sources.py` — projection lists updated.

---

#### 6.2 `cube_part_fundamentals` shrinks

The part's own column set is not declared anywhere — it is whatever the five builders emit, merged
by `PanelMerger`. So it shrinks implicitly, and `write_part`'s `COLUMNS_CHANGED` path will fire and
force a `full=True` rebuild ([step_cube_fundamentals.py:100-102](../../../src/data_aggregate/transformers/step_cube_fundamentals.py#L100-L102)).

- [ ] **Expect and allow one full cube rebuild.** This is designed behaviour, not a failure — but it
      is not incremental, so budget the wall-clock.
- [ ] Print the feature-name delta (before vs after) as an artifact. Anything that disappears without
      appearing in the §6.1 casualty list is an unintended loss and must be chased.
- [ ] Downstream of the cube: `configs/models.yml` / `modellling.yml` feature lists, and any strategy
      config naming a dead feature. A model config referencing a vanished feature fails at train
      time, long after this phase.

---

#### 6.3 Test fallout — 8 files, 1,878 lines

These are `data_aggregate` / `utils` tests, **distinct from the ~5,100 lines of `data_extract` tests
Phase 1 deletes**. They assert on removed fields, so they break at collection or on the first
assertion. Measured by grepping the removed set across `tests/`:

| test file | lines | removed fields referenced | disposition |
|---|---|---|---|
| `data_aggregate/test_capital_and_restatements.py` | 456 | **35** | mostly a lease/debt/restatement-detail test → **largely deletable**; salvage the restatement half into Phase 5's `test_amendment_grain.py`, which supersedes it |
| `data_aggregate/test_sector_gates_and_tags.py` | 357 | 6 | **rewrite** — the availability-gating logic is still right, only the KPI list changes |
| `data_aggregate/test_quality_features.py` | 350 | 10 | **rewrite or delete** with the special-items family; depends on the §6.1 keep/drop call |
| `data_aggregate/test_sector_features.py` | 303 | 18 | **rewrite** down to the 28 surviving KPIs |
| `data_aggregate/test_dividend_features.py` | 162 | 1 (`dividendsPaid`) | **repair** — yield survives; drop the payout/coverage assertions |
| `utils/test_yahoo_comparison.py` | 165 | 1 (`dividendsPerShare`) | **repair** — one field off the comparison list |
| `data_aggregate/test_new_factor_features.py` | 85 | 13 | **delete** — it is almost entirely bank/insurer/REIT factor inputs |
| `data_aggregate/test_fundamental_features.py` | 819 | 1 (`researchAndDevelopment`) | **rewrite** to the new contract (already in the plan); the single reference is trivial, the rewrite is not |

- [ ] Do this **after** the §6.1 keep/drop decision, not before — every "re-add" removes work here.
- [ ] Any test deleted rather than rewritten must have its *intent* recorded in
      [reports/adhoc/fields.md](../../adhoc/fields.md) beside the field it covered, so re-adding the
      field brings its check back with it.

---

#### 6.4 The fingerprint baseline

`tests/data_aggregate/aggregate_fingerprint_baseline.json` **will change** — it carries an
`input.fundamentals_slice` key plus per-label fingerprints, all of which move when the input table
and the feature set change.

- [ ] Regeneration is tightly gated (`docs/testing.md`, and it is a named risk zone in
      `coding_standard.md`) → propose, justify with the before/after diff, get approval.
- [ ] Regenerate **once**, at the end of Phase 6 — not incrementally per builder, or the baseline
      stops being evidence of anything.

---

**Verification**:
- [ ] The §6.0 grep-guard test passes: no removed name anywhere under `src/` or `configs/`.
- [ ] `"$PY" -m src data_aggregate build-cube` completes on the 32-ticker slice.
- [ ] Feature-count delta printed explicitly: N before, M after, with the dropped list, and every
      dropped name reconciled against the §6.1 casualty table.
- [ ] Full `"$PY" -m pytest tests/data_aggregate -q` green.
- [ ] Fingerprint baseline regenerated once, with the diff attached to the approval request.

**Estimated effort**: 4-5 days (raised from 3-4 — the 1,878 lines of test fallout in §6.3 were not
costed in the first draft). **This is the largest single risk in the plan** — see Risks.

---

### Phase 7: The validator — SUPERSEDED

**Absorbed into [v2's Phase 5b, the validator toolkit](2026-08-23-fundamentals-rebuild-plan-v2.md).**
Everything specified here survives there, plus the four-tier waterfall from the
[methodology research](../../research/financial-data/2026-08-22-fundamentals-validator-methodology.md),
the fourteen deferred items this file logged as "carry to Phase 7", and the historical-defect
corpus the validator has to re-find. The number 7 is retired so no cross-reference goes stale.

---

### Phase 8: Tests and documentation ⬜

**Changes**:

1. Tests, following `docs/testing.md`'s split — **real data by default; synthetic known-truth only
   for parsing/derivation math, always paired with a real-data coverage check**:
   - [ ] `tests/data_extract/test_kpi_catalogue.py` (schema + authority completeness)
   - [ ] `tests/data_extract/test_linkbase_resolution.py` (synthetic + the 6 real regimes)
   - [ ] `tests/data_extract/test_periods_q4.py` (synthetic ladder + AAPL/Skyworks real)
   - [ ] `tests/data_extract/test_entity_scope.py` (MAA, Southern)
   - [ ] `tests/validate/test_fundamentals_validator.py` (planted violations)
   - [ ] `tests/data_extract/test_amendment_grain.py` — **new, the §5.0 acceptance test**. Synthetic
         known-truth: an original + a restating amendment 400 days later → assert exactly 2 rows, the
         original unchanged, the amendment at its own date, only the amended field + its dependent
         TTM/growth columns different, `fiscal_end` monotone. Paired with the real-data check on
         SMCI/ADM/JPM per `docs/testing.md`'s rule.
   - [ ] `tests/data_extract/test_fundamentals_point_in_time.py` — re-pointed, must go **green**
   - [ ] Every test **prints a sanity-check conclusion** (AGENTS.md hard rule)
2. Docs, moving with the code:
   - [ ] `docs/data_schema.md:75-80` — the fundamentals section: new PKs, new column counts, the two
         new tables. The "239 columns" line becomes wrong the moment Phase 5 lands. **Also rewrite
         the `fundamentals_history` grain sentence**: it currently says "derived from
         `fundamentals_facts`", which no longer conveys that `as_of` is a *publication event* and
         that amendments append rather than overwrite. That distinction is the table's whole
         contract with the modelling layer and must be stated where a reader will hit it first.
   - [ ] `docs/data_sources.md:16-20,105-108` — new module names; the tag-ledger bullet now points
         at the validator.
   - [ ] `docs/architecture.md:46-47,102,111` — `validate/` contents, extraction order, the cube row.
   - [ ] `docs/config.md` — the three new JSON config files and why they are JSON not YAML.
   - [ ] `docs/database.md` — what is actually populated after the rebuild.
   - [ ] `AGENTS.md` — **capped at 70 lines** (currently 70). Only if a line genuinely must change,
         and only by removing one. Propose first.
   - [ ] `README.md` if the CLI surface changed.

**Estimated effort**: 1.5-2 days.

---

### Phase 9 / Phase 10 / Testing Strategy / Risk Mitigation — SUPERSEDED

Read [v2](2026-08-23-fundamentals-rebuild-plan-v2.md). The 32-ticker ad-hoc slice specified here is **retired**: Phase 9 now runs on the
**in-sample 26** and **out-of-sample 26** rosters that every Phase 3c/4/4b number is already
measured on, driven by the validator's own report, with SMCI/ADM carried as a separate
amendment pair. v2 also re-states the testing strategy and the risk register for the phases
that remain.

---

## Nightly-operation notes

The whole design is shaped by "extractions run nightly, a bot reads the output and takes positions":

- **Resume, never rescan.** `store.max_date_by(Tables.fundamentals_facts, "ticker")` → `resume_since`,
  plus the stored accession set, with `manifest_full_rescan_days` for periodic self-heal.
- **Append-only facts** means a nightly re-run is idempotent and an amendment lands as its own row at
  its own filing date — no retroactive rewrite of a number the bot already traded on.
- **`fundamentals_history` is append-only on the publication-event grain** (§5.0). A `10-Q/A` filed
  921 days after the original becomes consumable on **day 921**, not retroactively on day 0. The bot
  can therefore train on daily data whose every value was genuinely knowable on its own date, which
  is the whole point.
- **Nightly appends; it never rewrites.** A code change is not a publication event — re-deriving
  history after a bug fix silently moves numbers under an already-trained model. Rebuilds are
  explicit (`--rebuild`), logged and versioned.
- **`test_fundamentals_point_in_time.py` going green is the Phase 5 acceptance gate**, and under the
  event grain it is structural rather than guard-enforced: no row can carry a value that postdates
  its own `as_of`.
- **The validator runs in the same nightly step** and writes `fundamentals_quality`. A `severity`
  above threshold should be visible before the model trades, not discovered later.

---

## Success Criteria / Estimated Effort — SUPERSEDED

Read [v2](2026-08-23-fundamentals-rebuild-plan-v2.md).

## Resolved — decisions 9-12 (2026-08-21)

1. ~~`epsDiluted` TTM convention~~ → **`netIncome_ttm / dilutedShares_ttm`** (decision #9). The
   as-reported tag is still extracted and used by the validator as an independent cross-check.
2. ~~The Phase 6 per-KPI keep/drop list~~ → **already answered by "strict Tiers 1-3".** Flagging this
   as an open call was redundant: the rule is mechanical — *if a KPI's input field is not in the kept
   set, the KPI is deleted.* There is no third option short of re-adding the input, which is what
   decision #10 does for R&D. **Risk 1 is downgraded accordingly**: what remains is execution
   (27 KPIs, 8 test files), not a decision.
3. ~~`researchAndDevelopment`~~ → **re-added** as Tier 3R, regime-gated, with the cross-ticker basis
   rules in Phase 5 (decision #10).
4. ~~Reason codes~~ → **long side table** `fundamentals_reason_codes` (decision #11).
5. ~~`fundamentals_audit.py`~~ → **deleted**; check logic folded into `FundamentalsValidator`,
   invoked from step / CLI / tests (decision #12).

## Open items — SUPERSEDED

Open item 1 (the edgartools upgrade) closed 2026-08-21 on **5.51.0**. Open item 3 (Phase 4b's
A/B/C question) closed 2026-08-23 — **option A was implemented**, see §Phase 4b — implemented.
Open item 2 (`capitalizedSoftware` as a tier-0 companion) is carried in v2 §B.3. The four
genuinely open decisions were all **taken on 2026-08-23** — see
[v2 §Decisions taken](2026-08-23-fundamentals-rebuild-plan-v2.md): a new
`fundamentals_cik_cutover.json`; AXP onto one Rule 9-04 basis (it is already in the `bank`
regime — **this file's §3c.9 is wrong about that**); ORCL FY2020's Q4-windowed annual refused as
`ambiguous_duration`; DQC/Arelle deferred with a revisit trigger.

## Known gaps carried forward from research

These were not closed and remain open; none blocks implementation, all affect interpretation:
Compustat's exact `CAPX` include/exclude list (so any capex factor ported from a paper is
approximate); whether Compustat `COGS` includes `DP` and whether `XSGA` includes `XRD`; Compustat's
`REVT` under `INDFMT=FS` for a bank. Also: `us-gaap-doc-2025.xml` holds 14,899 documentation labels
against 17,326 schema elements — every definition quoted above was read from an entry that *was*
present, but **absence from that file is not evidence of absence from the taxonomy**.

And two corrections to keep from resurfacing: **Du, Huddart & Jiang (2023) "Lost in standardization"
is RETRACTED** — do not cite its magnitudes. The **8× rent multiple was Moody's, not S&P**, and it
varied by sector, not region.
