# Implementation Plan: Fundamentals extraction rebuild

**Date Created**: 2026-08-21
**Planning Phase**: 2 of 3 (FIC Workflow)
**Based on Research**: [2026-08-21-fundamentals-extraction.md](../../research/financial-data/2026-08-21-fundamentals-extraction.md)
**Request**: [specs/2026-08-21_plan_how to extract_Edgartools.md](../../../specs/2026-08-21_plan_how%20to%20extract_Edgartools.md)
**Companion**: [reports/adhoc/fields.md](../../adhoc/fields.md) — every removed field, with concepts,
coverage and its original evidence note, as a prioritised rebuild menu
**Next Phase**: Implementation (`/implement`)

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
   - [ ] Dump `fundamentals_history_legacy` to parquet — the 239-column, 27,602-row regression
         baseline. Every Phase 9 comparison is against this.
   - [ ] Record per-field non-null counts and per-ticker row counts.
   - [ ] Keep `fundamentals_facts_legacy` (7.8M rows) in place — it is the offline audit substrate
         Phase 7's thresholds get calibrated on, at zero network cost.

2. Delete, in dependency order:
   - [ ] `src/data_extract/utils/fundamentals/fetch_fundamentals.py`
   - [ ] `src/data_extract/utils/fundamentals/fetch_fundamentals_edgar.py`
   - [ ] `src/data_extract/utils/fundamentals/fundamentals_derive.py`
   - [ ] `src/data_extract/utils/fundamentals/fundamentals_tags.py`
   - [ ] `src/data_extract/utils/fundamentals/fundamentals_periods.py`
   - [ ] `src/utils/fundamentals_tag_ledger.py` (its `detect_tag_switch_breaks` idea is reborn in
         Phase 7 as a check, not a standalone CSV writer)
   - [ ] `src/validate/fundamentals_validation.py`, `analyze_history.py`,
         `run_fundamentals_integration_report.py` (rebuilt in Phase 7)
   - [ ] `src/validate/fundamentals_audit.py` — **delete** (decision #12). Its Tiingo→Yahoo
         ground-truth comparison is not lost: the *check logic* moves inside `FundamentalsValidator`
         as an external-source check (Phase 7), so there is one place that knows how to judge a
         value. `src/utils/tiingo_comparison.py` and `yahoo_comparison.py` **survive as I/O
         adapters** — they fetch, they do not judge — with their field lists trimmed in Phase 6.
   - [ ] `tests/utils/test_fundamentals_audit.py` (102 lines) — deleted with it; its intent moves to
         Phase 7's validator test.

3. Delete the tests bound to deleted code (~5,100 lines):
   - [ ] `test_fetch_fundamentals.py`, `test_fetch_fundamentals_edgartools.py` (1,329),
         `test_fundamentals_amendments.py`, `test_fundamentals_coverage_gaps.py` (716),
         `test_fundamentals_diagnostics.py`, `test_fundamentals_expanded.py`,
         `test_fundamentals_fiscal_calendar.py`, `test_fundamentals_fiscal_period.py` (610),
         `test_fundamentals_missing_fix.py`, `test_fundamentals_plausibility.py`,
         `test_fundamentals_reconstruction.py`, `test_fundamentals_sector_coverage.py`,
         `tests/utils/test_fundamentals_tag_ledger.py`
   - [ ] **Keep and re-point**: `test_fundamentals_point_in_time.py` (deliberately red — it is the
         acceptance criterion for the leak fix), `test_step_extract_fundamentals.py`,
         `test_fundamentals_employees.py`, `tests/utils/test_fundamentals_audit.py`,
         `tests/data_aggregate/test_fundamental_features.py` (Phase 6 rewrites its expectations).

4. `src/constants/constants.py` — remove only what becomes unreferenced:
   - [ ] `FUNDAMENTALS_FACTS RECONCILIATION` block (:577-600), `FUNDAMENTALS_DISCONTINUITY_MIN/MAX`
         (:641-642), the `XBRL TAG-SWITCH LEDGER` block (:645+),
         `FUNDAMENTALS_FINDINGS_RANKED_FILENAME` (:759) — *if and only if* Phase 7 does not re-adopt
         them. Sequence the constants pass **after** Phase 7, not here.
   - [ ] `FUNDAMENTALS_FORMS` stays (still the form list).
   - [ ] Risk zone → propose the diff and get approval before editing.

**Verification**:
- [ ] `rtk grep -rn "fundamentals_tags\|fundamentals_periods\|fetch_fundamentals" src/ tests/` returns
      nothing outside the new modules.
- [ ] `"$PY" -m pytest tests/ -q --collect-only` collects with zero import errors.
- [ ] The baseline parquet exists and has 27,602 rows × 239 columns.

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
   - [ ] `roll_up` is what the calculation linkbase is *checked against*, not a substitute for it.
   - [ ] `authority` is mandatory and must quote a primary source. Reviewable by a human.
   - [ ] Anything the research could not establish gets `"authority": "UNVERIFIED"` and is surfaced
         by a schema test — no silent guesses.

2. `configs/fundamentals/fundamentals_regimes.json` — new. The regime → template map, driven by the
   **statement role URI** the filer actually used, with GICS only as a tiebreak:
   - [ ] `sfp-dbo` 108000 → bank · `sfp-ibo` 108200 → insurer · `sfp-sbo` 112000 → broker-dealer ·
         `sfp-clreo` 110000 / `sfp-ucreo` 110200 → real estate · `soi-int` 132001 · `soi-ins` 136000
         · `soi-reit` 145000 · `scf-dbo` 160000.
   - [ ] The four verified GICS traps encoded as explicit exceptions: **mortgage REITs are 40204010
         under Financials**; the equity-REIT run is **not contiguous** (`601025` Industrial REITs);
         **Insurance Brokers 40301010** (MMC/AON/AJG) and **Financial Exchanges 40203040** /
         **Payments 40201060** (ICE/CME/V/MA) are Article 5 fee businesses — **do not route GICS 40
         to a bank template**; tower/data-center/timber REITs file like industrials.
   - [ ] Hybrids (BRK) get `regime: "hybrid"` and are excluded from regime-relative scoring rather
         than forced into a template.

3. `configs/fundamentals/fundamentals_exceptions.json` — new. The "expected absence" register the user asked
   for: `(regime, field) -> expected_absent`, so Phase 7 can tell structural absence from
   regression. Seeded from §2.8's measured matrix and justified by
   `17 CFR 210.1-02(bb)(1)(i)`.

4. `src/data_extract/utils/fundamentals/kpi_catalogue.py` — new, small. Loads + validates the three
   JSONs once, exposes typed accessors. Constants (paths, the JSON filenames) → `constants.py`.

**Verification**:
- [ ] `test_kpi_catalogue.py`: every entry has tier/kind/sign/definition/authority; every concept
      named is a real us-gaap 2025 element; no field appears in two tiers; no `UNVERIFIED` authority
      without a matching entry in **Open items**.
- [ ] Print the field count per tier and assert 40 tiered (11+12+16+1 R&D) + 13 inputs.

**Estimated effort**: 1-1.5 days (the definitions are the work, not the plumbing).

---

### Phase 3: The facts layer — linkbase-driven resolution ⬜

**Goal**: `fundamentals_facts`, resolved from the filer's own declared roll-up.

**Changes**:

1. `src/data_extract/utils/fundamentals/xbrl_linkbase.py` — new. The core primitive.
   - [ ] `statement_arcs(xbrl) -> DataFrame` — `calculation_linkbase()` filtered to
         `menucat == "Statements"`. **Preserve the `weight` sign** (`-1.0` is a real contra-account
         rollup, not noise).
   - [ ] `resolve_total(arcs, field_spec) -> Resolution` — walk from the filer's declared parent.
         Three outcomes, all recorded in `resolution_method`:
         `linkbase_total` (the filer declares a total; use it) →
         `linkbase_sum` (no total, but the leaf children are declared; sum them with weights) →
         `tag_fallback` (no linkbase for this concept; use `fallback_concepts` in priority order).
   - [ ] This is what would have caught **APA**: its linkbase declares `RevenuesAndOther` as the
         pretax revenue parent, a concept absent from the old tag list, so the resolver fell through
         to an element APA tags as literally `$0.00`, every quarter, for 19 rows.

2. `src/data_extract/utils/fundamentals/entity_scope.py` — new. **Filter on the AXIS, not the
   member.**
   - [ ] Take dimensionally-unqualified (default-member) facts. A fixed us-gaap member list cannot
         catch a company-extension member — MAA scopes its LP with `maa:LimitedPartnershipMember`,
         and Southern Company carries **six registrant CIKs / 3,579 `LegalEntityAxis` occurrences**
         in one instance with all identifiers = parent.
   - [ ] One documented exception hook for regulatory capital (which is *only* reachable
         dimensioned) — declared, not used in this pass.

3. `src/data_extract/utils/fundamentals/fetch_fundamentals_sec.py` — new, replaces the 1,232-line
   fetcher. Target: **< 350 lines**.
   - [ ] Per-ticker filing walk over `FUNDAMENTALS_FORMS`, resuming from
         `store.max_date_by(Tables.fundamentals_facts, "ticker")` → `resume_since`, plus the
         accession set already stored. Never a full read.
   - [ ] Worker count from `configs/configs.yml`, **not hardcoded** (the old `DEFAULT_WORKERS = 8`
         was a literal).
   - [ ] `store.ensure_table` race: the table must be **created before** the thread pool starts —
         check-then-create with no lock silently loses rows on a cold table.
   - [ ] Append-only: original and `/A` amendments coexist as separate rows, never overwritten.

4. Schema:
   - [ ] `Tables.fundamentals_facts` in `src/data_store/schema.py` — new column set, PK
         `(ticker, accession_number, field, fiscal_year, fiscal_period, duration_type)`.
   - [ ] `sql/schema.sql` — rewrite the block. **The current DDL is internally inconsistent**: its PK
         names `field` and `duration_type`, neither of which appears in its own column list
         (`schema.sql:328-359`). Fix that here.
   - [ ] Risk zone → propose the DDL diff and get approval.

**Verification**:
- [ ] Known-truth fixture: a synthetic linkbase where the total and the legs disagree → assert
      `linkbase_total` wins and `resolution_method` says so.
- [ ] Real-data: JPM resolves `totalRevenue` via `RevenuesNetOfInterestExpense` and **not** via
      `Revenues`; APA resolves via `RevenuesAndOther` and is **not** 0; XOM via `Revenues`;
      MET/MAA/DTE per the §3.2 table. Print the resolved concept per ticker.
- [ ] `shortTermDebt` on the 111 tickers that tag both legs with no total: assert the stored value
      equals the **sum**, and that both components are populated.
- [ ] MAA: assert the LP's `LegalEntityAxis` facts are excluded and `sharesOutstanding` is the
      parent's.

**Estimated effort**: 3-4 days.

---

### Phase 4: The period engine ⬜

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
- [ ] Known-truth: synthetic YTD ladder where `FY−YTD9` and `FY−(Q1+Q2+Q3)` disagree → assert the
      YTD9 path wins and is recorded.
- [ ] Real-data: **AAPL Q4-2025 revenue = 102.466 B** (the edgartools-verified figure).
      Skyworks FY2020 → 956.7M vs the filer's published 956.8M.
- [ ] Frozen-TTM detector on the rebuilt table: assert **0** exactly-repeated consecutive
      `totalRevenue` pairs for APA/XOM/ETN/MTB/TROW, and report the universe-wide rate vs the 6.2%
      baseline.
*(`test_fundamentals_point_in_time.py` moves to Phase 5 — under the §5.0 event grain the PIT
invariant is a property of the history build, not of the period engine.)*

**Estimated effort**: 2-3 days.

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

### Phase 7: The validator ⬜

**Goal**: a dedicated class, run after `fundamentals_history` is built, writing a report table.

**Changes**:

1. `src/validate/fundamentals_validator.py` — new. One class, `FundamentalsValidator`, run from the
   step after the history build. Writes `fundamentals_quality`
   `(run_date, ticker, field, fiscal_year, fiscal_period, check, severity, observed, expected, detail)`.

   **Layer 1 — hard guards (the only thing allowed to mutate)**: physically-impossible values only —
   negative `Assets`/`Revenue`/share counts, `|EPS| > 1000`, a ratio outside `[-1, 1]`. Nulls the
   value and stamps `dc_code = 'failed_hard_guard'`. Deliberately *narrower* than today's 13
   mutating rules: the 2026-08 audit measured **745 correct rows nulled by over-strict Q4 guards**.

   **Layer 2 — flag-only checks.** Never mutate. The user's three checks, plus what makes them real:

   | check | rule | why |
   |---|---|---|
   | `coverage_universe` | all **490** universe tickers have ≥1 row | the user's check 1 |
   | `coverage_quarters` | every ticker's quarters are contiguous to the last expected filing date at run time | the user's check 1 |
   | `coverage_field` | every Tier 1-3 field present for every ticker-quarter **unless** `configs/fundamentals/fundamentals_exceptions.json` says the regime excuses it | this is the check that stops structural absence looking like a regression |
   | `q4_footing` | `Q1+Q2+Q3+Q4 == FY` — **only on years where Q4 was NOT derived from that identity** | today's version passes 99.73% *by construction* and validates nothing. An identity must not take a derived quantity as an input. Where Q4 came from `FY−YTD9`, the footing against `Q1+Q2+Q3` is a genuine independent test |
   | `level_outlier` | MAD modified-z > **3.5** (Iglewicz & Hoaglin) on QoQ log-change, per (ticker, field), min 8 quarters | flag-only, severity by magnitude |
   | `tag_switch_break` | a `source_concept`/`resolution_method` change coinciding with a level step | the kink explainer, reborn as a check |
   | `cross_identity` | `Assets == Liabilities + Equity`; `GrossProfit == Revenue − COGS`; `NetIncome` ties the IS to the CF bridge | XBRL US DQC families `0004/0009/0011/0128` |
   | `sign_convention` | `TAG.crdr` is the sign oracle | Debreceny et al. (2010): the **dominant** cause of XBRL arithmetic failure is debit/credit treatment. DQC `0013/0014/0015/0174` |
   | `scale` | order-of-magnitude jump vs the field's own history | DQC `0095/0139/0192/0222` |
   | `pit_leak` | no row carries a fact with `filing_date > as_of`; no historical row changed value since the last run | §5.0. Structurally impossible under the event grain — which is exactly why it belongs here as a **standing assertion**: if it ever fires, the append-only contract has been broken by a code change, and the bot is training on numbers it could not have traded on |

   - [ ] **Calibrate every threshold offline on `fundamentals_facts_legacy` (7.8M rows) before
         wiring it in.** Zero network cost, and it is the substrate the research's own measurements
         came from. A threshold that fires on 40% of rows is not a check.

2. **The external-source check lives inside the validator** (decision #12). `fundamentals_audit.py`
   is deleted; `FundamentalsValidator.check_external_sources()` becomes the one place that judges a
   value against Tiingo/Yahoo, writing `check='external_disagreement'` rows like any other check.
   - [ ] `tiingo_comparison.py` / `yahoo_comparison.py` stay as **fetch-only adapters** — they
         return a comparable frame and hold the URL templates; all ranking, bucketing and verdict
         logic moves into the validator. That is the "one implementation, many callers" split the
         user asked for.
   - [ ] Calibrate the disagreement threshold against Boritz & No (2020), which is the reason to run
         this check *and* to distrust it: as-reported XBRL matches the 10-K to within **0.01%**
         while aggregators disagree at **6.5-7.7%**. A disagreement is evidence about the
         *aggregator* at least as often as about us, so this check is `severity: info` by default
         and never gates anything.

3. **Callable from several entry points, implemented once.** The class is the only implementation;
   these are thin wrappers over it:
   - [ ] `StepExtractFundamentals` — runs it nightly after the history build.
   - [ ] `"$PY" -m src validate fundamentals [-t TICKER]` — a real CLI command, not a `__main__`
         block. None of the current validators has one.
   - [ ] Tests instantiate the class directly against a frame — no DB, no CLI.

**Verification**:
- [ ] Synthetic frame with one planted violation per check → each fires exactly once, nothing else does.
- [ ] Run on the 30-ticker slice; print the finding count by check and severity.
- [ ] Assert the hard-guard layer nulled **0** values on a clean known-good ticker (AAPL).

**Estimated effort**: 2-3 days.

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

### Phase 9: The 30-ticker acceptance rebuild ⬜

**Goal**: the plan's stated endpoint. Prove both tables end-to-end before touching 490 tickers.

**Ticker slice** — chosen to hit every regime and every known trap, not at random:

| regime | tickers | what it proves |
|---|---|---|
| industrial / tech | AAPL, MSFT, CSCO, TGT, KR, COST | 52/53-week fiscal years; the Cisco-2017 53-week Q4 bug |
| bank | JPM, USB, BAC, TFC, MTB | `RevenuesNetOfInterestExpense`; deposits-are-not-debt; capex `not_applicable` |
| insurer | MET, PGR, AFL | the 3.2% ASC-606 slice; the LDTI 2021-01-01 break |
| REIT | MAA, SPG, AMT, EQR | Up-C `LegalEntityAxis`; MAA's IPR&D-tagged capex; AMT files like an industrial |
| E&P / energy | XOM, APA, EOG, DVN, VLO | the APA `RevenuesAndOther` zero-revenue chain; Valero's D&A tie-break |
| utility | SO, DTE, NEE | six registrant CIKs; AFUDC-equity; regulatory assets |
| known-broken | ETN, VRT, SWKS, BRK-B | `totalRevenue == 0`; Skyworks FY2020; the hybrid |
| **amendment / restatement** | **SMCI, ADM** | the §5.0 publication-event grain. Measured: SMCI 4 amendments / 846 facts (the 2024 late-filing + auditor episode — the hardest adversarial case available), ADM 4 / 1,280. **JPM (1 `10-Q/A`, 761 facts) and MTB (2, 925) are already in the slice above**, so bank amendments are covered too |

Slice is **32 tickers**, not 30 — SMCI and ADM were added specifically to exercise §5.0, because
without a substantive restater the amendment grain is untested. Trim them if you want the literal 30.

**Acceptance criteria** — every one must be *measured and printed*, not asserted by eye:
- [ ] Both tables build from empty; row counts and column counts reported.
- [ ] **The amendment ladder, printed end to end for SMCI and ADM**: the original row, its
      as-filed value, the amendment row at its own `filing_date`, and the delta — proving the
      original was never retroactively edited. This is the headline artifact of the rebuild.
- [ ] **PIT invariant holds on every row** of the slice; `test_fundamentals_point_in_time.py` green.
- [ ] **APA / ETN / VRT `totalRevenue` is non-zero and non-null** for every quarter it should exist.
- [ ] **Frozen-TTM rate = 0** on this slice (baseline: APA 100%, XOM 36%).
- [ ] **`shortTermDebt` equals the sum of both legs** wherever both are tagged with no total.
- [ ] **`q4_footing` runs on a non-empty set** — i.e. some Q4s came from `FY−YTD9` and are genuinely
      testable. If the set is empty, the check is still vacuous and Phase 4 is not done.
- [ ] Every Tier 1-3 field is either populated or carries a `dc_code`. **Zero unexplained nulls.**
- [ ] Regression vs the Phase 1 baseline parquet: for the fields present in both, report the
      distribution of relative differences and **explain every case beyond 1%**. Differences are
      expected — several are the point — but each needs a named cause.
- [ ] Wall-clock for 30 tickers recorded, and the 490-ticker full rebuild and nightly incremental
      extrapolated from it (see Risks for the current estimate).

**Estimated effort**: 1 day + rebuild wall-clock.

---

### Phase 10: EFFICIENCY review ⬜

**Goal**: the user's requested final quality gate.

- [ ] Spawn a sub-agent with: *"Review this in-progress refactor in the Python repo. Get the diff
      with `git diff HEAD -- <path>`. ANGLE — EFFICIENCY: flag wasted work the diff introduces."*
      Run it per new module (`fetch_fundamentals_sec.py`, `xbrl_linkbase.py`, `periods.py`,
      `build_history.py`, `fundamentals_validator.py`).
- [ ] Specific things to point it at, because they are the ones that bite nightly: repeated
      `store.load` of the same slice; unprojected reads; per-ticker frame concatenation in a loop;
      recomputing the linkbase per field instead of per filing; `apply`/`iterrows` where a vectorised
      merge exists (the old `build_tag_frames` got this right and is worth matching).
- [ ] Then `/simplify` on the same diff.

**Estimated effort**: 0.5 day.

---

## Testing Strategy

**Unit (synthetic known-truth)** — parsing and derivation math only: the Q4/YTD ladder, linkbase
weight arithmetic, roll-up sums, ratio formulas, the reason-code state machine.

**Integration (real data)** — the 30-ticker slice: resolution per regime, entity scoping, coverage,
the validator end to end. Per `docs/testing.md`, every synthetic fixture is **paired with a
real-data coverage check** — the fixture proves the formula, the coverage check proves it fires on
real filings.

**Regression** — the Phase 1 baseline parquet. Not "no change" (change is the goal) but "every
change has a named cause".

**Manual** — read three filings by hand (one bank, one REIT, one E&P) and tie the stored numbers to
the PDF. Nothing else catches a plausible-but-wrong number.

---

## Risk Mitigation

1. **Phase 6 is a bigger job than Phase 3 — but it is now execution, not a decision.** Dropping 78
   consumed columns touches ~2,700 lines of cube builders plus 1,878 lines of tests, and deletes
   real signal (Beneish, cash-conversion cycle, combined ratio, payout ratio).
   *Downgraded from "largest single risk" to "largest single job"*: "strict Tiers 1-3" already
   determines every case mechanically — input not kept ⇒ KPI deleted — so there is no sign-off gate
   blocking the work. Any later change of heart is a one-line `tier: 0` catalogue add plus a
   coverage check, per [reports/adhoc/fields.md](../../adhoc/fields.md), not a redesign.
   *Remaining risk is a silent miss*, mitigated by the §6.0 grep-guard test.

2. **The full rebuild is long.** ~490 tickers × ~60 filings ≈ 30k `filing.xbrl()` calls at 4-10 s
   ≈ **33-83 h single-threaded, 4-10 h at 8 workers**.
   *Mitigation*: this is a one-time backfill, not the nightly path. Nightly is incremental —
   ~5-8 filings universe-wide on a quiet night, ~20-80 at earnings peak, i.e. **minutes**. The
   resume path (`max_date_by` + stored accession set) is therefore load-bearing and gets its own
   test. Checkpoint per ticker so an interrupted backfill resumes.

3. **Never kill the backfill by image name.** A previous multi-hour SEC download was destroyed by a
   blanket `python.exe` kill. Kill by PID only.

4. **The linkbase is not universal.** Older filings and some filers have no calculation linkbase.
   *Mitigation*: that is exactly what `resolution_method = 'tag_fallback'` is for. Measure the
   fallback rate in Phase 9 — if it exceeds ~20%, the architecture's premise needs re-examining
   before the full rebuild.

5. **Restatement disagreement is real and will show up.** `us-gaap:Revenues` for BAC FY2023 is
   98,581M as filed but **102,769M** as re-presented in the FY2025 10-K, and the `frames` API returns
   the restated figure.
   *Mitigation*: we read per-filing, so we get as-filed — which is the correct basis for a trading
   model. Record it so nobody "fixes" it toward frames.

6. **Coverage will drop, on purpose** (the staircase fix nulls carried-forward TTMs).
   *Mitigation*: quantify it in Phase 9 and report it as a headline number. A coverage drop that is
   not explained is a bug; this one is a design choice.

7. **Risk-zone files**: `constants.py`, `data_store/`, `sql/schema.sql`, `configs/`, and the
   aggregate fingerprint baseline are all touched. Every one gets a proposed diff and approval first.

**Rollback**: all deletions are in git. `fundamentals_facts_legacy` / `fundamentals_history_legacy`
stay in Postgres untouched until the user explicitly retires them — the new tables are new names, so
rollback is repointing `schema.py`, not restoring data.

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

## Success Criteria

- [ ] Old stack deleted; no dangling imports; test suite collects clean.
- [ ] `fundamentals_facts` + `fundamentals_history` + `fundamentals_reason_codes` +
      `fundamentals_quality` build end-to-end for the 30-ticker slice.
- [ ] The user's three checks are all **non-vacuous** and pass, with `q4_footing` running on a
      non-empty independent set.
- [ ] **No leakage**: every row's contributing facts have `filing_date <= as_of`; amendments append
      at their own filing date; originals are byte-identical before and after a restatement lands.
- [ ] Every Tier 1-3 value is populated or reason-coded. Zero unexplained nulls.
- [ ] APA/ETN/VRT revenue fixed; frozen-TTM rate 0 on the slice; `shortTermDebt` sums both legs.
- [ ] No removed field name survives anywhere under `src/` or `configs/` (the §6.0 guard test);
      `tests/data_aggregate` green; fingerprint baseline regenerated once, with approval.
- [ ] New code is materially smaller than the 7,268 lines removed (target < 1,800).
- [ ] Docs in sync; every test prints its sanity-check conclusion.
- [ ] EFFICIENCY review + `/simplify` applied.

## Estimated Effort

| Phase | Estimate |
|---|---|
| 1 Demolition | 0.5 day |
| 2 KPI catalogue | 1-1.5 days |
| 3 Facts layer | 3-4 days |
| 4 Period engine | 2-3 days |
| 5 History layer | 2-3 days |
| 6 Downstream repair | 4-5 days |
| 7 Validator | 2-3 days |
| 8 Tests + docs | 1.5-2 days |
| 9 30-ticker rebuild | 1 day + wall-clock |
| 10 EFFICIENCY review | 0.5 day |
| **Total** | **18-24 days** |

---

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

## Open items

1. ~~The edgartools upgrade~~ → **CLOSED 2026-08-21.** venv now on **5.51.0** /
   `httpxthrottlecache` 0.6.1 (the lock had already been resolved; only the venv was stale).
   The JPM linkbase measurement decision #2 rests on **reproduced exactly** — 465 arcs,
   `menucat=="Statements"` → 108, the two-leg bank revenue roll-up at weight +1.0 each,
   linkbase 0.006 s against `xbrl()` 5.20 s. Full table in §Implementation log.
2. **`capitalizedSoftware` as a `tier: 0` companion to R&D** — see the residual flagged in Phase 5.
   Only worth it if the R&D factor turns out to be sensitive to capitalisation policy.

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
