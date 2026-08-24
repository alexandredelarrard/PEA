# Implementation Plan v2: Fundamentals rebuild — Phases 4c → 10

**Date Created**: 2026-08-23
**Planning Phase**: 2 of 3 (FIC Workflow)
**Supersedes**: the *forward-looking* half of
[2026-08-21-fundamentals-rebuild-plan.md](2026-08-21-fundamentals-rebuild-plan.md) (v1, 3,144
lines). That file stays as the **implementation record** for Phases 1-4b — every measured number
quoted here is traceable to it and none of it is repeated at length.
**Based on research**:
[2026-08-22-fundamentals-validator-methodology.md](../../research/financial-data/2026-08-22-fundamentals-validator-methodology.md)
(how SEC / XBRL-US DQC / Compustat actually validate — the design input for Phase 5b) ·
[2026-08-22-validation-numbers-sec.md](../../research/financial-data/2026-08-22-validation-numbers-sec.md)
(the verification pass; its §4 is four of the deferred items below) ·
[2026-08-22-phase3b-resolution-audit.md](../../research/financial-data/2026-08-22-phase3b-resolution-audit.md)
**Original request**: [specs/2026-8-22/…validation_numbers_sec.md](../../../specs/2026-8-22/2026-08-21_research_validation_numbers_sec.md)
**Next Phase**: Implementation (`/implement`)

---

## Overview — what this revision changes

Three things, all of them asked for:

1. **Phases 1-4b are compressed to one page** (§A). They are done and measured; the narrative
   belongs in the record, not in the plan.
2. **Every deferred `[ ]` from Phases 1-4b is collected into one register and re-assigned**
   (§B). There are **25** of them, scattered across nine sub-sections of v1, several logged as
   "carry to Phase 7" against a Phase 7 that did not yet exist as a design. **6** are
   resolution-layer defects that must land *before* the history layer (→ new **Phase 4c**);
   **3** belong to the history layer; **12** become named validator checks or
   documented-not-fixed residuals; **4** are decisions only the user can make.
3. **A validator TOOLKIT is inserted between Phase 5 and Phase 6** (new **Phase 5b**), and it
   **absorbs and supersedes v1's Phase 7**. Two validators would have been absurd. Phase 9's
   acceptance is then rewritten to *be* the toolkit's own report, run on the **in-sample 26**
   and the **out-of-sample 26** rosters.
4. **All four open decisions were taken on 2026-08-23** and are written into the phases that own
   them (§B.4). Answering the AXP one **found an error in v1**: §3c.9 says *"AXP routes to
   `industrial` (GICS 'Transaction & Payment Processing Services')"* — it does not.
   `sp500_tickers` gives AXP `sub_industry = 'Consumer Finance'`, which
   `fundamentals_regimes.json:23-28` maps to **`bank`** already. "Transaction & Payment
   Processing Services" is V and MA. So the regime the decision asked for is the regime AXP has;
   the missing piece is one `never_use` entry, and the work is a quarter of what the question
   implied. Recorded because the wrong sentence would otherwise have driven a regime re-route
   with real side effects (capex → `not_applicable`, four expected-absent cells).

Execution order after this revision:
**4c → 5 → 5b → 6 → 8 → 9 → 10.** (Phase 7's number is retired; §Phase 5b says why.)

### The one structural argument this revision rests on

Every acceptance number in Phases 3b, 4 and 4b was produced by **uncommitted scratchpad
scripts** — `accept4b.py`, `defect_census.py`, `explain.py`, `per_ticker_2pc.py`, `rosters.py`,
`cx_*.py`, `da_link.py` (v1 §4b.10 lists them). They live under a session directory that will
not survive, so **not one figure in v1 is reproducible today by anyone else**, and Phase 9 needs
exactly those measurements again on the same 52 tickers.

That is what makes Phase 5b a *toolkit* rather than a check pass: its first job is to absorb
those scripts into committed, tested code so that "are these numbers right?" is a command, not
an archaeology exercise. It is also why the validator moves **before** the downstream repair —
Phase 6 rebuilds the whole cube on top of `fundamentals_history`, and building a cube on numbers
nobody has measured is how the previous stack got here.

---

## Part A — Where the rebuild stands (Phases 1-4b, all ✅)

Full detail: [v1](2026-08-21-fundamentals-rebuild-plan.md) §Implementation log. Current tree:
**810 tests collected, 0 errors**; 5 new fundamentals modules (`xbrl_linkbase.py` 1,072 ·
`periods.py` 764 · `kpi_catalogue.py` 490 · `fetch_fundamentals_sec.py` 367 · `entity_scope.py`
202 = **2,895 lines** against 7,268 deleted) and 5 new test files (2,218 lines).

| Phase | What shipped | The numbers that matter |
|---|---|---|
| **1** Demolition | 10 `src/` modules + 19 test files deleted (~12,900 lines). `detect_level_outliers` re-homed to `src/utils/outliers.py`. `pipeline_fingerprint` guard retired. Baseline captured. | baseline parquet **27,602 × 239**; `fundamentals_facts_legacy` **7,776,870** rows left in place; collection 969 → 720, 0 errors |
| **2** KPI catalogue | `configs/fundamentals/{fundamentals_kpis,fundamentals_regimes,fundamentals_exceptions}.json` + `kpi_catalogue.py`. Load-time schema enforcement. | **53 fields** = 11 T1 + 12 T2 + 17 T3 + 13 T0; **40 scored, 49 extracted**; `authority: UNVERIFIED` appears **0 times**; absence register measured per **regime** over 442 tickers |
| **3** Facts layer | `xbrl_linkbase.py` (resolution), `entity_scope.py` (filter on the AXIS), `fetch_fundamentals_sec.py`, `Tables.fundamentals_facts`. **Six routes**, not three: `linkbase_total` · `linkbase_root` · `linkbase_sum` · `statement_leaf_sum` (3b) · `field_sum` · `tag_primary` · `tag_fallback`, plus `unresolved`. | genuine `tag_fallback` **6.84%** vs the ~20% architecture gate; APA resolves on `apa:RevenuesAndOther` **$9.220B** (was 0) — a **company extension**, so no tag list could ever have reached it; MAA 5,324 facts → **567** consolidated |
| **3b/3c** Robustness | 26-ticker × 2011-2026 sweep found **5 defects Phase 3's 6-filing snapshot could not see**, the re-sweep found **4 more the fixes themselves created**. `menucat` ∪ hardened role-URI; root ranking; zero-guard; bank revenue basis; conditional lease subtraction. | linkbase share 2011-14 **0.9% → 73.7%**; pooled **51.4% → 69.7%**; for FY ≤ 2014 the route changed on **63.8%** of rows and the **value agreed on 99.86%** — the evidence that the architecture change is an *improvement*, not merely a different answer |
| **3c.9** Out-of-sample | A second 26-ticker roster with **zero overlap** passed every Phase-3c criterion untuned, and found 3 defects that pre-dated 3c: MSFT SG&A **−$34.7bn** (wrong-parent weight), `totalDebt` **zero-filling a missing debt leg** (213 + 29 rows reading a lease liability as total debt), `employees` in the XBRL field list. All fixed; cost was **exactly** the wrong rows, 0 collateral. | structural discovery generalised to a never-seen sector: HCA on `HealthCareOrganizationPatientServiceRevenue`, PLD on `RealEstateRevenueNet`, DUK on `RegulatedAndUnregulatedOperatingRevenue` |
| **4** Period engine | `periods.py` — Q4 = **FY − YTD9** first; ticker-level fiscal calendar; share counts differenced in **share-days**; `instant_stock`; window identity within 7 days. | tautological Q4 **100% → 0.07%**; frozen TTM **6.2% → 0.33%**; `epsDiluted` computable **8% → 87.8%**; four derived quarters foot to the filer's own annual within 0.5% on **97.9%** of 3,698 independent points |
| **4b** Coverage holes | Route **3b `statement_leaf_sum`** (`roll_up.any_of`, groups set by measured **co-occurrence**, not element names) + a per-filer extension register (`by_ticker`, 17 filers) + `by_ticker_periodicity`. Option A. | `capex` tickers 15 → **20** / 22 → **23**; `depAmort` 22 → **26** / 21 → **26**; **1,150 + 917** values now exist where none did; **116 + 232** were wrong and are corrected; **0** lost in-sample; DTE capex **$4,429.0M** = the filer's own $4,639M − $210M *to the dollar* |

**Two properties worth restating because everything downstream leans on them**: the facts table
is **strictly as-filed** (derived quarters stay in memory), which is what makes §5.0's
publication-event grain provable rather than asserted; and the **derived quarters are more
accurate than the as-reported ones** on both rosters (99.85% vs 95.80% within 2%), so the Q4
ladder is not the weak link — the filers' own re-presented figures are.

---

## Part B — The deferred-item register

This is the "re-distillate" pass. Every unchecked box in v1's Phases 1-4b, its blast radius as
measured, and the phase that now **owns** it. Nothing here is new work discovered in this
planning pass — it is v1's own backlog, made addressable.

### B.1 → Phase 4c (resolution layer; must land before Phase 5 writes history)

| # | item | first logged | measured blast radius |
|---|---|---|---|
| 1 | **A statement-role test on routes 1, 5 and 6.** A note-level fact can win where a statement line exists. | v1 §4b.12 | **8 tickers** confirmed: AMT `longTermDebt` **$1.9M** note-level vs **$21,127M**; CSCO `depAmort` $1,200M vs $2,811M (**2.3×**, 45 filings); MCD `capex` ~**12×** low, 64 rows over 10 fiscal years; PG `totalRevenue` $28,400M vs $83,680M; AAPL/VLO protected only by luck. Closes D1b, D2, D3 and the §4b.11 CSCO finding in one change. |
| 2 | `longTermDebt`'s candidate list **contradicts its own definition** — `us-gaap:LongTermDebt` sits at priority 2 and **includes the current portion**. | v1 §4b.12 | UNP on **16 of 16** comparable quarter-ends; cross-vintage agreement only 17/23 and 17/26, the disagreements all concept-switching between three genuinely different bases |
| 3 | `longTermDebt` **never resolves** for AFL, BRK-B, MAA → the last Tier-1 hole; sole cause of every `incomplete_roll_up` on `totalDebt`. | v1 §3c.9 | 3 tickers lose `totalDebt` entirely; 117 filings / 7 tickers in-sample, 10 / 2 out |
| 4 | `_values_by_period` picks the **LAST duplicate**, including a rounded one. | v1 §4b.11 | ORCL FY2026 `Depreciation` $7,623M vs $7,600M — a 0.3% haircut; route-independent, so it affects every field |
| 5 | **The exception register does not scope the 6 balance-sheet detail fields** (`accountsPayable`, `accountsReceivable`, `ppeGross`, `accumulatedDepreciation`, `intangiblesExGoodwill`, `minorityInterest`). | v1 §3c.9 · research §4.1 | **346** in-sample "holes" are structural. Config-only, authority already established. **Blocks** Phase 5b's `coverage_field` check. |
| 6 | Fold the surviving 3c numbers into `test_linkbase_history.py` as standing assertions. | v1 §3c.7 | the sweep is currently a measurement, not a guard |

### B.2 → Phase 5 (history layer)

| # | item | first logged | why it lands here |
|---|---|---|---|
| 7 | **A refused quarter carries no reason code.** `_derived` returns `None` and the window simply has no row. | v1 §Phase 4 carry-forward | breaks the "zero unexplained nulls" criterion. `fundamentals_reason_codes` is the one table that records why a value is absent — adding a second mechanism in `periods.py` would give it two sources of truth |
| 8 | **`totalLiabilities`'s derived fallback** (`totalAssets − stockholdersEquity`) is specified, not implemented. | v1 §3c.6 · research §4.2 | **11 tickers, ~20% of the swept universe** at zero coverage (APA DTE EOG ETN VLO + DUK LLY MCD ORCL TMO WMT). Reg S-X 5-02 has no "Total liabilities" caption, so this is systemic and the derivation is load-bearing, not a nicety. Cross-field ⇒ history layer |
| 9 | Route 3b's **period intersection** costs values. | v1 §4b.12 | 31 out-of-sample values (BA 23, EQIX 8) and BA's fiscal-2011 **annual** point. The obvious repair reintroduces the mixed basis route 3b exists to prevent → keep strict, prove it is reason-coded |

### B.3 → Phase 5b (become named validator checks; documented, not "fixed")

| # | item | first logged | becomes |
|---|---|---|---|
| 10 | `ppeNet`'s lease adjustment is **unguarded in practice** — all 128 of its adjustments fire on *silence*, never on positive evidence. | v1 §3c.8 | `adjustment_unguarded` (Tier 1). Not producing bad values today (0 negatives), but the catalogue claims a guard that never discriminates. State it, don't assume it away |
| 11 | `totalRevenue`, `operatingIncome`, `incomeTaxExpense` are **the three weakest fields on every independent measure** — hold-out, footing and restatement rate all name the same three. | v1 §Phase 4 | the priority set for `q4_footing` and `cross_identity`; their thresholds get their own calibration row |
| 12 | **MCD capex** steps **35.6×** across the single 2017→2018 boundary where route 3b takes over. No cross-vintage test can see it — MCD tags the same narrow concept consistently in the earlier era. | v1 §4b.12 | the motivating case for the new **`basis_step`** check (Tier 2). Fixed by item 1; the check is what stops the next one |
| 13 | **VLO capex went dark in 2023** — neither concept tagged undimensioned in any filing from 2023-07 on, 21 of 63 filings. Pre-dates Phase 4b. | v1 §4b.11 | `coverage_field` + `basis_step` finding |
| 14 | **NEE's mixed acquisition-plus-capex line** excluded — understates 2018-19 capex by up to **$5.2bn** across 8 filings. The largest known cost of any register decision. | v1 §4b carry-forward | `register_cost` (Tier 1, `severity: info`) — a declared, quantified understatement, surfaced every run rather than living in a report |
| 15 | **PLD's `pld:AcquisitionOfPropertiesNetOfCash`** is tagged **negative** (−$1,025M) and excluded; 4 filings understate. | v1 §4b carry-forward | same |
| 16 | The `by_ticker` extension register covers **17 of ~500** tickers. | v1 §4b carry-forward | `register_coverage` — an unregistered filer keeps whatever the tag routes give it, which is *correct-but-partial*, not wrong. The check names which filers are running partial |
| 17 | `sign_convention`'s `TAG.crdr` **sign oracle is not available** from the per-filing substrate (`element_catalog` exposes labels only; `balance` is `None` even for `us-gaap:Assets`). | v1 §Phase 2 | a **named dependency** on Phase 5b, not a check: needs either the `us-gaap-*-2025.xml` taxonomy download or the FSNDS `cal.tsv` `negative` column. Decide in 5b.0 |
| 18 | `longTermDebt` **completeness is not measurable by the quarter grid** — many filers tag it only in 10-Ks, and a company with no long-term debt legitimately has no fact (AAPL pre-2013, META pre-2022, SWKS pre-2020, MET, PGR, GS, JPM, SPG). | v1 §4b.12 | `coverage_field` must read a **periodicity** register for this field, exactly as `depAmort` does for AFL/CSCO. Otherwise the check fires on 8 correct tickers |
| 19 | `capitalizedSoftware` as a `tier: 0` companion to R&D — capitalising filers look like they spend less. | v1 Open item 2 | recorded as a known residual; only worth it if the R&D factor proves sensitive |
| 20 | The `constants.py` pass — `FUNDAMENTALS_FACTS RECONCILIATION`, `FUNDAMENTALS_DISCONTINUITY_MIN/MAX`, the `XBRL TAG-SWITCH LEDGER` block, `FUNDAMENTALS_FINDINGS_RANKED_FILENAME`. | v1 §Phase 1 | v1 deliberately sequenced this **after** the validator, because the validator decides whether it re-adopts them. Now due at the **end of 5b**, as one proposed diff |
| 21 | `AGENTS.md`'s code map still names `fundamentals_audit`, `fundamentals_validation`, `analyze_history` — and `src/validate/` **does not exist**; Phase 1 deleted the folder. | found this pass | 5b recreates `src/validate/`; the one-line map fix goes with it (AGENTS.md is capped at 70 lines — replace, never add) |

### B.4 → DECIDED 2026-08-23 (was open; no default was taken)

| # | question | decision | lands in |
|---|---|---|---|
| 22 | **Re-registration / CIK cutover.** `Company(ticker)` resolves only the *current* registrant, so a predecessor's decade is invisible with no error and no gap signal: APA misses **2011-02 → 2021-05** (Apache Corp, CIK 6769), GOOGL 2011-2015 (Google Inc, CIK 1288776), ETN the 2012 Irish domestication. The repair is **not** a union of CIKs — Apache Corp kept filing its own 10-K/10-Q through 2024-11 (it retains registered public debt), so 2021-2024 is double-covered by two *different legal entities*; concatenating would duplicate ~15 filings and mix a subsidiary's statements with the parent's. | **A new config JSON** — `configs/fundamentals/fundamentals_cik_cutover.json`, beside the other three fundamentals configs: hand-curated, reviewable, one **dated cutover** per ticker with its evidence. Chosen over a `sp500_tickers` column because that table is rebuilt from Wikipedia and would silently overwrite curated data on every roster refresh. | **4c.6** |
| 23 | **AXP's revenue basis.** Post-provision for 2011-2018 (**91 rows** on `TotalRevenuesNetOfInterestExpenseAfterProvisionsForLosses`), then the ASC-606 element — a mid-history basis break, the same class as MTB's. | **One Rule 9-04 basis, through the bank regime.** ⚠ But **AXP is already routed to `bank`**: `sp500_tickers` gives it `sub_industry = 'Consumer Finance'`, which `fundamentals_regimes.json:23-28` maps to `bank`, with a `consumer_finance_note` that already names AXP as the confirmed case. v1 §3c.9's *"AXP routes to `industrial`"* is **wrong** — that sub-industry is V and MA. So this needs **only the `never_use` entry**, exactly as 3c.4 did for MTB: no regime re-route, and none of its side effects. | **4c.7** |
| 24 | **ORCL fiscal 2020 `totalRevenue`** — all three vintages stamp the full-year `us-gaap:Revenues` with a **Q4 window**, so `FY − YTD9` cannot run and Q4 reads **$39,068M** instead of about $10,439M. D1 (an annual masquerading as a quarter *while a real annual also exists*) is fixed; this is D1b, where the annual exists **only** under the quarterly window, so there is nothing to compare it against. | **Refuse the fact**, `dc_code = 'ambiguous_duration'`. Declining to guess is not inference; reclassifying would be, and the only available test for *"is this really the year?"* uses the very quarters being derived — circular. A $39bn Q4 propagates into four TTM windows, `revenueGrowth`, and every peer z-score built on them. | **4c.8** |
| 25 | **Tier 0 — the XBRL-US DQC ruleset (196 rules, v30.0.3) via Arelle, per filing?** The one mechanism that is provenance-independent *by construction*, and the only one that can see an error in the **filer's own calculation linkbase**, upstream of everything we read. | **Defer.** Tier 3 already buys genuine provenance independence *inside the filer's own disjoint numbers* — `leaf_vs_total` (89/94 comparable points), `cross_vintage`, the Q4 hold-out (591/752 cases) — at zero new dependency and zero backfill cost. DQC stays a designed, costed option with a **named revisit trigger** (5b.0). | **5b.0** |

### B.5 Closed by this pass — do not re-open

- v1 §3c.6's claim that `totalAssets − stockholdersEquity` **overstates liabilities by the NCI**
  was **withdrawn**: the formula names the catalogue *field* `stockholdersEquity`, whose concept
  order puts `…IncludingPortionAttributableToNoncontrollingInterest` first. The reading was of
  the field's **name**, not its concept list.
- v1's acceptance criterion *"zero zero-revenue rows"* was itself **wrong**. VRT's 2018-2020
  zeros are the GS Acquisition Holdings blank-cheque shell, pre-merger: a $690M IPO, trust
  dividend income, and genuinely no revenue. Restated criterion: *no zero-revenue row unless the
  filer reports no other non-zero revenue concept anywhere in the filing.*
- v1 §4b.6's *"SWKS `depAmort` truth ~$618M"* and §4b.7's *"ORCL FY2026 = $10,633M"* both
  **mixed fiscal years**. Correct: SWKS FY2025 **$463.0M**, ORCL FY2026 **$9,294M**.
- **Two designs were measured and REFUTED** — summing the name-matching D&A/capex leaves (84
  ground-truth points, only 14 matched within 2%; AAPL off by −31.6% because its
  `us-gaap:Depreciation` is a PP&E-**note** disclosure) and "a negative-weight extension child
  of the investing node is capex" (admits `nee:PurchasesOfOtherSecurities`,
  `dte:ConsolidationOfVIES`, `apa:EquityMethodInvestmentContribution`). Do not re-propose them.
- **`companyfacts` can prove a concept is PRESENT and can never prove one is ABSENT** — it
  publishes no company-extension taxonomy at all (DTE's and NEE's entire capital programme is
  invisible there) *and* drops dimensioned facts. Every coverage claim in Phases 5b and 9 must
  be measured off `filing.xbrl()`.
- **Do not relax the dimensional filter to "fix" capex.** DTE tags
  `us-gaap:PaymentsToAcquirePropertyPlantAndEquipment` at $3,686M **only** dimensioned to
  `dte:DTEElectricMember`. Admitting it stores the *subsidiary's* capex as the group's, 17% low,
  and it looks entirely plausible.

---

## Part C — The remaining phases

### Phase 4c: the deferred resolution-layer fixes ✅

> **Closed 2026-08-23.** 30 of 34 items done; 3 carried deviations (marked inline);
> 4 items deliberately left open and moved to **§B.6**, which sequences them AFTER the
> Phase 5b validator, because each is an edge case whose correct treatment is a
> per-value judgement the validator exists to make.

**Goal**: close B.1 before any history row is published.

**Why this cannot wait for Phase 5b.** `fundamentals_history` stores **TTM levels and YoY
growth**, and §5.0 makes it **append-only**: once rows are published on the publication-event
grain, a later resolution fix needs an explicit, logged, versioned `--rebuild` rather than a
silent overwrite (v1's own operational corollary — *a code change is not a publication event*).
A field carrying two bases across a boundary produces a growth feature that is **pure
artefact**: MCD capex steps **35.6×** at 2017→2018, CSCO `depAmort` runs **2.3×** apart. Fixing
the basis after the cube is trained is the expensive order.

**4c.1 The statement-role test on routes 1, 5 and 6** — the single change that closes five
logged defects.

- [x] Route 3b already carries this guard, and that is exactly why it is the safe route. Extend
      it to `linkbase_total`, `tag_primary` and `tag_fallback`.
- [x] **Formulate it so silence is not evidence.** §3c.8's most expensive lesson was `_only_when`
      reading "the linkbase says nothing" as licence to act. A leaf like `goodwill` or a
      `dei:` cover-page tag can *never* have a calculation arc, and route 5 is its normal home.
      So the rule is: **reject only when the concept IS declared in the linkbase and EVERY role
      it is declared on is a non-statement role.** Undeclared ⇒ unaffected.
- [x] Reuse `NON_STATEMENT_ROLE` / the hardened 3c.1 pattern
      (`detail|disclosure|polic|parenthetical|schedule|tables?$|uncategor|highlight`). Do not
      loosen it — `schedule` is what excludes Reg S-X Schedule I/II parent-only statements,
      which look exactly like face statements and would corrupt consolidated numbers silently.
- [x] Record the rejection on the row (`role_rejected` in the `adjustment` JSON), so the
      population stays separable and Phase 5b can count it.

**Verification** — this reorders resolution for all 52 tickers, so it gets the 3c.1 treatment:
- [x] Synthetic known-truth: one concept declared on a note role only, one declared on both, one
      not declared at all → assert reject / keep / keep.
- [x] Real-data ground truth, all five confirmed instances: AMT `longTermDebt` **$21,127M** not
      $1.9M · CSCO `depAmort` **$2,811M** not $1,200M · MCD `capex` on the leaf sum for the
      **whole** history, with the 2017→2018 step **gone** · PG `totalRevenue` $83,680M · AAPL
      `depAmort` still **$11,698M** and VLO still **$3,158M** (the §4b.4 regression guard).
      **→ DEVIATION:** MEASURED, RESULT REFUTED THE PREMISE. 0 of the 5 named instances reproduced; the real mechanism is an UNDECLARED coarse tag beating the filer's declared statement lines. Ground truth re-established on CSCO depAmort FY2025 = $2,811M. See log §4c.1.
- [x] Both-roster before/after, on the same join key v1 §3c.7 used: report **route changed %**
      and **value agreed %** per year, and account for every material disagreement by name.
      Expect movement *only* where the winning concept is note-only.

**4c.2 `longTermDebt`** — the catalogue is wrong about its own field.

- [x] Demote or remove `us-gaap:LongTermDebt` (priority 2), which **includes the current
      portion** against a field defined ex-current. Where it is the only concept available,
      either subtract the current leg or reason-code it — measure both before choosing.
- [x] Same treatment for `LongTermDebtAndCapitalLeaseObligations` (subtract the noncurrent
      finance-lease leg, as the Tier-2 spec already says) vs `LongTermDebtNoncurrent`.
- [ ] Widen for AFL, BRK-B, MAA — an insurer, a hybrid holding company and an Up-C REIT, i.e.
      three unclassified-balance-sheet filers tagging debt under a concept the list does not
      name. Read the **linkbase**, not the fact set (§4b.2's method), before adding anything.
- [ ] Declare the legitimate-absence set as a **periodicity/absence** register entry (item 18),
      or `coverage_field` will fire on 8 correct tickers.
- [x] Risk zone: `configs/fundamentals/fundamentals_kpis.json` → propose the diff.

**4c.3 `_values_by_period`'s duplicate rule.**
- [x] Replace "the last duplicate wins" with a deterministic choice. Recommended: prefer the
      fact with the **finer `decimals`** (ORCL's $7,623M over $7,600M), and always emit the
      disagreement for Phase 5b's `duplicate_fact` check.
- [x] Measure the fire rate on both rosters first — it is route-independent, so it touches every
      field, and a change here moves numbers everywhere.

**4c.4 The exception register widens** (item 5) — config only, no code.
- [x] Add the six balance-sheet detail fields to the `bank` / `insurer` / `broker_dealer` /
      `real_estate` `by_regime` blocks, following the exact pattern `currentAssets` already
      uses. Same mechanism (Reg S-X 5-02 "when appropriate"), same authority
      (`17 CFR 210.1-02(bb)(1)(i)`), already cited in the file for the fields it does cover.
      **→ DEVIATION:** FIVE fields written (accountsPayable, accountsReceivable, ppeGross, accumulatedDepreciation, intangiblesExGoodwill), not six -- the plan never enumerated the sixth. 31 cells, 10 with expected_absent=true. The register's completion is a B.6 item: a cell only means something once the validator's coverage_field check consumes it.
- [x] **Measure each cell off `filing.xbrl()`**, not companyfacts, and scope the query — the
      legacy substrate covers only **445 of 500** tickers and is missing the whole U-W tail, so
      an unscoped rate lies.
- [x] `fundamentals_exceptions.json` is hand-formatted; a `json.dumps` round-trip reformats all
      545 lines. Use a validated emitter or a text splice.

**4c.5 Standing assertions** (item 6) — fold the 3c/4/4b numbers into
`test_linkbase_history.py` and `test_leaf_sum_resolution.py` so the sweep guards instead of
merely measuring.

**4c.6 The CIK cutover table** (decision 22) — a new config plus a dated walk.

This is the largest of the three decision items, because the config is the easy half.

- [x] `configs/fundamentals/fundamentals_cik_cutover.json` — one entry per ticker:
      `cutover_date`, `predecessor_cik`, `successor_cik`, `kind`
      (`reorganisation` / `domestication` / `rename`), and an `evidence` string. Seeded with the
      three confirmed cases: **APA** (6769 → 1841666, 2021-03-01), **GOOGL** (1288776 → 1652044,
      2015-10-02), **ETN** (Eaton Corp Ohio → 1551182, the 2012 Irish domestication).
- [x] **A `rename` is not a cutover and must be recorded as such.** CVS Caremark → CVS Health
      and Facebook → Meta keep their CIK and need no entry at all. Encoding them as cutovers
      would double-walk one CIK. The config's schema should make the distinction explicit so the
      next person does not add one.
- [x] `fetch_fundamentals_sec.py` / the edgar driver: where a ticker has an entry, walk **both**
      CIKs and keep predecessor filings **strictly before** `cutover_date` and successor filings
      **on or after** it. This is the whole point — Apache Corp filed 4×/year through
      **2024-11-07** as a *subsidiary* with its own registered debt, so a union would duplicate
      ~15 filings and blend two legal entities' statements.
- [x] Accession dedup already exists (the resume path is accession-based, not `max_date_by`), so
      a duplicate cannot survive the store — but assert it, because a silent duplicate would
      double a period's facts and every downstream sum with it.
- [x] `entity_scope` needs no change: pre-reorganisation, the predecessor **is** the consolidated
      parent, so undimensioned facts are the right scope on both sides of the boundary.
- [ ] Load-time validation in `kpi_catalogue.py` (or a sibling loader): every ticker named exists
      in the universe; `predecessor_cik != successor_cik`; `cutover_date` falls inside the
      predecessor's own filing window. A typo here silently deletes history.
- [x] **Every APA number measured so far will move, and that is expected.** APA currently carries
      **22 filings, none before 2021**; with the cutover it should reach the ~62 every other
      in-sample ticker has. Its pre-2021 Apache Corp filings may resolve `totalRevenue` on
      concepts the 2021+ APA Corp filings never use, so the `apa:RevenuesAndOther` result (71
      rows, $596M-12,132M) becomes a *subset* of a longer series. Re-baseline, do not treat the
      change as a regression.
- [ ] Phase 5b's `filing_continuity` check becomes the enforcement: after this lands, a
      truncation it reports is either a **missing cutover entry** or one of the legitimate
      short-history cases (VRT's 2018 SPAC listing, META's 2012 IPO). Those two are declared, so
      anything else is a work item.
- [x] Risk zone: `configs/` → propose the diff.

**4c.7 AXP's revenue basis** (decision 23) — one `never_use` entry, then measure.

- [x] Add `TotalRevenuesNetOfInterestExpenseAfterProvisionsForLosses` to the **bank** regime's
      `totalRevenue.never_use`, with the Reg S-X 9-04 reasoning, exactly as 3c.4 did for MTB's
      `InterestIncomeExpenseAfterProvisionForLoanLoss`. **No regime change** — AXP is already
      `bank` (§B.4 item 23).
- [x] **Then measure, because banning a concept is not the same as gaining a better one.** 3c.4
      worked because MTB tags both Rule 9-04 legs: 96 of its 110 post-provision rows moved onto
      `InterestIncomeExpenseNet + NoninterestIncome` and median revenue rose $1.588bn → $2.335bn.
      If AXP does **not** tag both legs, this turns 91 rows into reason-coded nulls instead of
      into a comparable basis. Check the legs first, on the existing out-of-sample sweep.
- [x] If the legs are absent: report it and **stop** — a documented basis break beats a
      three-quarters-empty top line for a $60bn-revenue issuer. That is a measurement outcome,
      not a re-litigation of the decision.
- [x] Also check COF and SYF, the other two Consumer Finance names, for the same concept. If they
      carry it too, this is a cohort fix rather than an AXP one.

**4c.8 ORCL's Q4-windowed annual** (decision 24) — refuse, do not reclassify.

- [x] Extend `periods.py`'s D1 guard (`_drop_annual_masquerading_as_quarter`) to the D1b case: a
      `quarterly`-shaped fact whose duration is annual-length while **no** annual fact exists for
      that fiscal year. Refuse it with `dc_code = 'ambiguous_duration'`; the code surfaces through
      Phase 5's `fundamentals_reason_codes`.
- [x] **Gate it on the window LENGTH, not on the value**, so the rule cannot be circular. A
      ~365-day fact tagged into a quarterly slot is a tagging defect regardless of what it
      contains; a value-based test would need the quarters being derived.
      **→ DEVIATION:** GATE IS NOT THE WINDOW LENGTH -- that was impossible. ORCL's fact is genuinely 91 days, so there is no length anomaly to detect. Gated on the filer's own as-filed nine-month cumulative instead, which is still non-circular. See log §4c.8.
- [x] Keep D1's condition-3 protection intact — a legitimate Q4-equals-FY series (nine months of
      zero) must survive, which is what stops this guard eating a real quarter.
- [x] Measure the fire rate on both rosters. Known population is **1 row** (ORCL FY2020); if it
      fires more widely, the length band is wrong.

**Estimated effort**: 3-3.5 days + ~35 min per roster re-sweep (in-sample 36.9 min,
out-of-sample 29.1 min measured). Raised from 2-2.5 by 4c.6-4c.8; 4c.6's dated walk is ~1 day on
its own and touches the fetcher, not just a config.

**Risk zones touched**: `fundamentals_kpis.json`, `fundamentals_exceptions.json`, **the new
`fundamentals_cik_cutover.json`** → one batched proposed diff, as Phase 3 did.

---

### Phase 5: The history layer ⬜

Unchanged from v1 §Phase 5 in design — **`fundamentals_history` on the publication-event grain**
(§5.0), ~71 columns, `fundamentals_reason_codes` as a long side table (decision #11). That
section is not restated here; read it in v1. What follows is only what this revision **adds**.

**The grain, in one paragraph, because everything else depends on it.** `as_of` is a *filing
date*, not a period end. A row is emitted for every `(ticker, date)` on which ≥1 extracted value
became newly public; an amendment emits a row **only if it changes a value** (which discards the
88 Part-III/cover-only amendments — stricter than a fact-count threshold, since an amendment can
re-tag 200 facts to identical values); rows are **immutable** once written. Measured cost:
**+158 rows on 27,602 (+0.6%)**. This is what makes the PIT invariant structural rather than
guard-enforced — a `10-Q/A` landing **921 days** late becomes consumable on day 921, and 26% of
amendments land more than a full quarter after the original.

**Added by the register:**

- [ ] **Item 7 — every refused quarter gets a reason code.** `periods.py::_derived` currently
      returns `None` on a sign-guard or scale-test rejection and the window silently has no row.
      Emit the refusal to `fundamentals_reason_codes` from the **history build**, not from
      `periods.py`, so the table stays the single source of truth. Codes already earned:
      `insufficient_quarters`, `split_basis_mismatch` (45 windows / 8 tickers — AAPL 7:1 and
      4:1, BRK-B's 246,000× class ratio), `partial_leaf_sum`, `incomplete_roll_up`,
      `zero_only_retained`, and `ambiguous_duration` (decision 24, emitted by 4c.8).
- [ ] **Item 8 — implement `totalLiabilities = totalAssets − stockholdersEquity`** where
      `us-gaap:Liabilities` is untagged. Both inputs confirmed present on all 11 affected
      tickers. Stamp the row `resolution_method = 'derived_identity'` so it never reads as a
      resolved fact, and make it a `cross_identity` input in Phase 5b rather than a silent fill.
- [ ] **Item 9** — keep route 3b's strict period intersection; assert the 31 affected
      out-of-sample values are **reason-coded**, not silent, and print BA's fiscal-2011 gap as a
      named cost.

**Verification** (v1's list, plus):
- [ ] `test_fundamentals_point_in_time.py` goes **green** — the defining acceptance test of this
      phase, deliberately red since Phase 1.
- [ ] **Amendment round-trip on SMCI and ADM** (the two substantive restaters; SMCI is the 2024
      late-filing/auditor episode, the hardest adversarial case available): the original row
      keeps its as-filed value *unchanged*, a new row exists at the amendment's own filing date,
      and only the amended field plus the TTM/growth columns whose window contains it differ.
      Print both rows side by side.
- [ ] `fiscal_end` monotone non-decreasing in `as_of` for every ticker (an amendment to Q1-2024
      filed after Q2-2024 carries `fiscal_end = 2024-06-30`, with the restated period in
      `amended_fiscal_end`).
- [ ] Idempotency: a second run appends **0** rows.
- [ ] **Zero rows with `value IS NULL AND dc_code IS NULL`** — this is the gate item 7 exists to
      make passable, and Phase 5b's `unexplained_null` is its standing form.

**Estimated effort**: 2.5-3 days.

---

### Phase 5b: The fundamentals validator TOOLKIT ⬜ — NEW

> **This absorbs and supersedes v1's Phase 7.** Two validator phases would have been absurd;
> the number 7 is retired to avoid a stale cross-reference. Everything v1 §Phase 7 specified is
> here, plus the toolkit framing, the four-tier waterfall from the methodology research, and the
> fourteen register items from §B.3.

**Goal**: one committed, reusable toolkit that answers *"are these numbers right, for these
tickers?"* — run automatically after every build, and on demand for any ticker set. It is the
thing that makes Phase 9 reproducible and the nightly run trustworthy.

#### 5b.0 The design, and the three precedents it rests on

The research
([methodology](../../research/financial-data/2026-08-22-fundamentals-validator-methodology.md))
found one pattern shared by every entity examined, and it is not the obvious one:

> **Nobody validates by re-deriving the value and checking it matches itself, and nobody claims
> 100%.** Every one of them runs a **cheap, automated, deterministic layer first, on data
> independent of how the number was derived**, and reserves expensive review for whatever that
> layer flags. None blocks or silently corrects on a statistical flag; they escalate it.

Three concrete precedents, each of which settles a design question:

- **The SEC itself chose warn-over-reject.** An EDGAR submission whose own calculation linkbase
  does not foot produces a **warning, not a rejection**; the filing is accepted and the warning
  stays visible downstream. **Rule 6.6.30** exists specifically for sign inversion
  (*"Invert the sign of a numeric fact whose element has an `xbrli:balance` value that is
  inconsistent with the reporting concept being reported"*). That is direct precedent for
  flag-only, from the strictest party in the chain.
- **XBRL-US's DQC had to learn that a naive footing check over-fires.** DQC_0118's own
  documentation: *"Calculation inconsistencies reported to XBRL filers can be overwhelming as
  many don't represent real errors, so validation rules filter out false inconsistencies."*
  Same lesson this repo drew independently from `menucat` and the zero guard. **This is why
  calibration is a deliverable of this phase, not a follow-up.**
- **Compustat runs >2,500 checks per company *and* mandatory analyst review of every report,
  forever.** So: *"100%" is not reached by better automation — it is reached by automation
  narrowing what a human has to look at, applied continuously.* A validator that tries to
  replace the human step is solving a harder problem than any vendor has solved. Its job is to
  produce a **short, ranked, explained** list.

And one framing from Bloomberg's data-governance material worth borrowing: quality is graded
(*essential / sufficient / best-in-class*), not pass/fail. Hence severities, and only `critical`
gates.

**Decision to take at 5b.0** (register item 17): the `sign_convention` check needs a
debit/credit oracle that the per-filing substrate **does not expose** — `element_catalog` gives
labels only, and `balance` is `None` even for `us-gaap:Assets`. Either download
`us-gaap-*-2025.xml` once and cache it, or read the FSNDS `cal.tsv` `negative` column. Pick one
before writing the check, or it ships as a stub. Recommended: the taxonomy download, cached
under `data/` — it is a one-off, it also unblocks any future authority work, and it does not
introduce the bulk-dataset ingest that decision #3 kept out of scope.

**Decision 25 (Tier 0 / DQC via Arelle) — DECIDED 2026-08-23: DEFER.** Build Tiers 1-4 only.
The reasoning, recorded so the option stays live rather than forgotten: Tier 3 already obtains
genuine provenance independence *from inside the filer's own data* — `leaf_vs_total` plays the
filer's declared aggregate against the sum of its own leaves, `cross_vintage` plays one filing
against another, and the Q4 hold-out plays FY and YTD9 against a Q4 the engine never reads — at
**zero new dependency and zero backfill cost**. DQC buys one thing those cannot: a defect in the
**filer's own calculation linkbase**, upstream of everything we read.

- [ ] State the deferral in the phase report, with its **revisit trigger**, rather than leaving it
      implied: *adopt Tier 0 if Tier 3's checks converge (i.e. `leaf_vs_total` and
      `cross_vintage` stop disagreeing) while Phase 9 or the full-universe run still surfaces
      wrong values that no check explains.* That is the observable signature of a defect class
      living upstream of our resolver, and it is the only condition under which the dependency
      earns its cost.
- [ ] If it is ever adopted, the cheapest first step is the **sampled offline audit** (~300
      filings stratified by regime × era, compared against our Tier 1-3 output on the same rows),
      not a per-filing pipeline tier. That answers *"does DQC see anything we cannot?"* as a
      number, for ~1 day and no nightly cost.

#### 5b.1 Shape

```
src/validate/                       -- the folder Phase 1 deleted; recreated here
  fundamentals_validator.py         FundamentalsValidator: the ONE implementation
  fundamentals_checks/
    __init__.py                     CHECK_REGISTRY -- never hand-list what a registry drives
    tier1_value.py                  deterministic, per resolved value      (vectorised)
    tier2_series.py                 statistical, per (ticker, field) series
    tier3_internal.py               cross-vintage / cross-route / hold-out  <- the key tier
    tier4_external.py               Tiingo / Yahoo, severity=info, never gates
  fundamentals_report.py            the markdown + printed renderer Phase 9 pastes
```

**Two layers, exactly as decision #8 set them:**

- **Layer A — hard guards. The only thing allowed to mutate.** Physically-impossible values
  only: a negative `Assets`, a negative share count, `|EPS| > 1000`, a ratio outside `[-1, 1]`.
  Nulls the value and stamps `dc_code = 'failed_hard_guard'`. Deliberately **narrower** than the
  old stack's 13 mutating rules — the 2026-08 audit measured **745 correct rows nulled by
  over-strict Q4 guards**. A guard that nulls a real number is worse than the check it replaces.
- **Layer B — flag only.** Never mutates. Writes `fundamentals_quality`.

**Entry points — one implementation, thin wrappers** (decision #12's "one place that knows how
to judge a value"):
- [ ] `StepExtractFundamentals` — runs after the history build, every night.
- [ ] `"$PY" -m src validate fundamentals [-t TICKER] [--roster in_sample|out_of_sample|all]
      [--since DATE] [--tier 1,2,3] [--report PATH]` — a **real CLI command**, not a `__main__`
      block. None of the old validators had one.
- [ ] Tests instantiate the class against a frame — no DB, no CLI.
- [ ] `tiingo_comparison.py` / `yahoo_comparison.py` stay **fetch-only adapters**: they return a
      comparable frame and hold the URL templates. All ranking, bucketing and verdict logic
      lives in the validator.

**Rosters stop living in a scratchpad.** `configs/fundamentals/fundamentals_rosters.json` —
`in_sample` (26), `out_of_sample` (26), `amendment_pair` (SMCI, ADM), each entry carrying *why
that ticker is on the list*, which is the property that made both rosters useful. Risk zone
(`configs/`) → propose the diff.

#### 5b.2 The checks

**Tier 1 — per-value, deterministic, provenance-independent, vectorised over the whole table.**
Cheap enough to run on every row every night.

| check | rule | earned by |
|---|---|---|
| `unexplained_null` | **0** rows with `value IS NULL AND dc_code IS NULL` | the user's "null if the value is not existing at all"; register item 7 |
| `coverage_universe` | every universe ticker has ≥1 row | user check 1 |
| `coverage_quarters` | each ticker's quarter grid is contiguous to the last expected filing date at run time | user check 1. Baseline to beat: revenue quarters present **98.74% / 99.48%**; the 30 known gaps are 5 named cases (MAA 17 in the Up-C era, JNJ 8, GS 3, DE 1, VRT 1) |
| `coverage_field` | a Tier 1-3 field is expected for every ticker-quarter **unless** the register excuses it — by regime (`by_regime`), by filer periodicity (`by_ticker_periodicity`), or by declared absence | **the check that stops structural absence looking like a regression.** Blocked on 4c.4 (the 6 detail fields, 346 holes) and item 18 (`longTermDebt`'s 10-K-only filers) |
| `expected_absent_drift` | a value **present** where the register says `expected_absent` is equally a finding | the register is measured, so it decays; PGR resolving capex when the insurer cell said it would not is exactly how §4b's deviation 1 was caught |
| `q4_footing` | `Q1+Q2+Q3+Q4 == FY`, **only on years whose Q4 was not derived from that identity** | user check 3. The old version passed **99.73% by construction** and validated nothing. Now **99.9%** of Q4 rows are genuinely testable |
| `cross_identity` | `Assets == Liabilities + Equity`; `GrossProfit == Revenue − COGS`; `NetIncome` ties IS to the CF bridge | DQC families 0004/0009/0011/0126/0128. Must treat a `derived_identity` `totalLiabilities` as an input, never as independent evidence |
| `sign_convention` | a `non_negative` field carrying a negative as-filed value; oracle = the taxonomy `balance` attribute | Debreceny et al. (2010): debit/credit treatment is the **dominant** cause of XBRL arithmetic failure. Known population: **7 rows** across both rosters, all `as_reported` (BA `CostOfRevenue` ×4 in its 2009 comparatives, C once in 2020, PLD, VRT), plus 8 more found in 3c.5 (MTB `accumulatedDepreciation`, DTE, SWKS, BAC, AMT) |
| `adjustment_unguarded` | an adjustment fired on **silence** rather than positive evidence | item 10: all 128 `ppeNet` lease adjustments. 0 bad values today, but the catalogue claims a guard that never discriminates |
| `register_cost` / `register_coverage` | the declared, quantified cost of each register exclusion; and which filers run on a partial register | items 14, 15, 16. NEE up to **$5.2bn** over 8 filings; 17 of ~500 filers registered. `severity: info`, but visible every run |
| `filing_continuity` | filings-per-ticker-per-year against the measured 4.0-4.2 band; a short series must map to a **named** cause — a `fundamentals_cik_cutover.json` entry, or one of the two declared legitimate cases | **this check is the enforcement mechanism for decision 22.** Once 4c.6 lands, the only acceptable short histories are VRT's 2018 SPAC listing and META's 2012 IPO; APA, GOOGL and ETN must be repaired by cutover, and anything else the check finds is a **missing cutover entry** — i.e. a work item, not a fact about the filer |
| `pit_leak` | no row carries a fact with `filing_date > as_of`; no historical row changed value since the last run | **structurally impossible** under §5.0 — which is exactly why it belongs here as a standing assertion. If it ever fires, an append-only contract was broken by a code change and the bot is training on numbers it could not have traded on |

**Tier 2 — per-series, statistical. Produces *candidates*, never verdicts.**

| check | rule | earned by |
|---|---|---|
| `level_outlier` | MAD modified-z > **3.5** (Iglewicz & Hoaglin) on QoQ log change, per `(ticker, field)`, min 8 quarters | reuse `src/utils/outliers.py` (Phase 1 re-homed it beside the MAD kernel precisely so there is one implementation) |
| `frozen_series` | exactly-repeated consecutive TTM values | the staircase defect. **6.2% → 0.33%**; the 5 survivors are genuine (NEE/USB three-significant-digit rounding, VRT's correct zeros) |
| `tag_switch_break` | a `source_concept` / `resolution_method` change coinciding with a level step | **user check 2**, and independently the SEC's own named review category: *"you have used different XBRL elements to tag the same reported line item from period to period."* Known base rate: **0.67% / 0.71%** of quarters |
| **`basis_step`** | a level step at the exact boundary where `resolution_method` changes — a *route* change, not a concept change | **the most valuable check the register earns.** MCD capex **35.6×** at 2017→2018; CSCO `depAmort` two bases 2.3× apart; VLO capex dark from 2023. **No cross-vintage test can see any of them** — the filer tags the same narrow concept consistently in the earlier era, so only the route boundary betrays it (items 12, 13) |
| `scale` | order-of-magnitude jump vs the field's own history | DQC 0091/0095/0103/0139/0157 |
| `periodicity` | a field with only annual facts for a ticker | AFL/CSCO `depAmort` (48 and 36 annual facts, **zero** quarterly/ytd) is **correct**, not a regression — read `by_ticker_periodicity` |

**Tier 3 — cross-vintage, cross-route, hold-out. The provenance-independent tier that needs no
external data**, because it plays the filer's own disjoint numbers against each other. This is
where most of the real evidence lives, and it is what makes deferring Tier 0 defensible.

| check | rule | standing baseline to hold |
|---|---|---|
| `holdout_q4` | where a filer published the FY fact, the YTD9 fact **and** its own discrete Q4, force the derivation and compare — the engine never derives these, so it is a genuine hold-out | 591 / 752 cases; **98.73% / 98.99%** within 1% where the filer's own three numbers foot. Already a standing test; the toolkit makes it a *reportable* number |
| `annual_footing` | the four derived quarters vs the filer's own annual, restricted to years whose Q4 is not the identity | **99.12% / 98.78%** within 2%, median error 0.000000%. Derived quarters beat as-reported ones (99.85% vs 95.80%) |
| `leaf_vs_total` | the same `(ticker, field, period)` resolved by `statement_leaf_sum` in one vintage and by a declared total in another — two disjoint pieces of the filer's own evidence | 89 / 94 comparable points, **76.40% / 78.72%** exact. Every disagreement found so far is the *total* being narrow (MCD 27×, VLO, MTB, AAPL FY2013 $8,165M vs $9,076M) — which is `basis_step`'s corroboration |
| `cross_vintage` | the same `(ticker, field, period)` across filings. **Restatement and defect are distinguishable without external data**: a derivation error leaves ≥1 quarter with `basis != as_reported`, while a restatement leaves all four as-filed and they foot to the **first-filed** annual | 12 of 14 in-sample and 14 of 22 out-of-sample failures over 2% are restatements. VLO operating income FY2012: first-filed $4,010M = our four quarters **to the dollar**, last-restated $5,044M. Base rate: **4.53% of annual windows move >2%** between first and last filing |
| `derived_vs_asreported` | `epsDiluted` computed vs the filer's `EarningsPerShareDiluted`; the share-day `dilutedShares` derivation vs the filer's own annual weighted average | decision #9's designed cross-check. Share-day baseline: **97.3% of 710 points** within 0.5%, median error 0.0052% |
| `duplicate_fact` | one filing tagging the same `(concept, period)` twice with different values | item 4 / 4c.3. ORCL $7,623M vs $7,600M |
| `restatement_ledger` | record, never repair | BAC FY2023 `Revenues` is **98,581M as filed** and **102,769M** as re-presented in the FY2025 10-K, and the `frames` API returns the restated figure. We read per-filing, so we get as-filed — the correct basis for a trading model. **Record it so nobody "fixes" it toward frames.** |

**Tier 4 — external corroboration. `severity: info`, and it never gates anything.**

- [ ] Tiingo / Yahoo, run **only** on what Tiers 1-3 flagged — mirroring Compustat's
      automate-first-escalate-the-residual pattern, and because it is the slowest and
      rate-limited path.
- [ ] Calibrate against Boritz & No (2020), which is the reason to run this check *and* to
      distrust it: as-reported XBRL matches the 10-K to within **0.01%** while aggregators
      disagree at **6.5-7.7%**. **A disagreement is evidence about the aggregator at least as
      often as about us.**

#### 5b.3 Calibration — a deliverable, not a follow-up

DQC_0118's own documentation is the warning: a naive footing check *"overwhelms"*. So:

- [ ] Calibrate every threshold **offline**, before wiring anything in, on the two swept ledgers
      (52 tickers, ~304k rows, both eras) and on `fundamentals_facts_legacy` (7.8M rows, zero
      network cost).
- [ ] **Print a fire-rate table per check.** A check firing on >2% of rows without a named
      mechanism is a threshold bug, not a finding. Report it as such.
- [ ] **Every threshold quoted off the legacy substrate carries its caveat**: that dump covers
      **445 of 500** tickers and is missing the entire U-W tail (UNH, UNP, UPS, USB, V, VLO,
      VRT, VZ, WFC, WMT …) plus recent additions. It is alphabetically biased and has never seen
      that tail. Scope every rate query or it lies.
- [ ] **Filer-sophistication as a prior** — the *Accounting Review* finding (smaller filers,
      first-year adopters and in-house taggers have systematically higher error rates; high
      custom-tag rates correlate with lower comparability and more comment letters) suggests
      Tier-2 thresholds could be filer-aware. **Recorded as a refinement, not built here** — our
      universe is the S&P 500, i.e. the low-error end of that distribution.

#### 5b.4 The acceptance test that matters: re-find the bugs we already fixed

A validator that cannot re-detect the defects this rebuild spent three phases finding is not
validated. The toolkit's own test corpus is the archive.

| historical defect | must be | check |
|---|---|---|
| MSFT `sellingGeneralAdmin` −$34.7bn (wrong-parent weight) | **silent now** — regression guard | `sign_convention` |
| `totalDebt` = a lease liability (BRK-B $4.9-6.3bn, META $7.6-16.7bn, PGR $179-211M) | **silent now** | `cross_identity`, `unexplained_null` |
| APA `totalRevenue` = 0 / −$467M | **silent now** | `basis_step`, `level_outlier` |
| MTB revenue on a post-provision basis (110 rows, ~32% understated) | **silent now** | `tag_switch_break` |
| MCD capex 35.6× step at 2017→2018 | **flagged `critical`** before 4c.1, **silent** after | `basis_step` |
| CSCO `depAmort` 2.3× two-basis series | same | `basis_step` |
| AMT `longTermDebt` $1.9M note-level | same | `scale`, `basis_step` |
| ORCL FY2020 Q4 revenue $39,068M | **flagged** | `q4_footing`, `scale` |
| BAC FY2023 revenue 98,581 → 102,769 | **flagged `info`**, never repaired | `restatement_ledger` |
| AAPL FY2012 derived Q4 = 24.3bn shares (the 7:1 split) | **reason-coded, not flagged** | `split_basis_mismatch` |

**Verification**:
- [ ] Synthetic frame with **one planted violation per check** → each fires exactly once and
      nothing else does. This is the standard; a check that cannot be planted cannot be trusted.
- [ ] The table above, reproduced as a real-data test.
- [ ] **0 findings at severity ≥ high on a known-clean ticker (AAPL)** other than the ones named
      here.
- [ ] Layer A nulled **0** values on AAPL.
- [ ] Every test prints its sanity-check conclusion (AGENTS.md hard rule).

**Also due at the end of this phase** (v1 sequenced them here deliberately):
- [ ] Item 20 — the `constants.py` pass, as one proposed diff, now that the validator has
      decided what it re-adopts.
- [ ] Item 21 — `src/validate/` exists again; fix `AGENTS.md`'s code-map line **by replacing
      it**, not adding (cap 70 lines).
- [ ] `Tables.fundamentals_quality` + `Tables.fundamentals_reason_codes` in `schema.py` and
      `sql/schema.sql`. Risk zone → propose the DDL.

**Estimated effort**: 4-5 days. Raised well above v1's 2-3 for Phase 7: this now absorbs ~15
scratchpad scripts into tested code, adds Tier 3 (six checks that did not exist as a design),
and owns 14 register items.

---

### Phase 6: Downstream repair ⬜

Unchanged in substance from v1 §Phase 6 — read it there. Summary of the size, because it is the
largest single **job** in the plan (downgraded from largest *risk*: "strict Tiers 1-3" decides
every case mechanically, so there is no sign-off gate blocking the work):

| | n |
|---|---|
| `fundamentals_history` columns before → after | **239 → 68** |
| removed | 171, of which **78 are consumed downstream today** |
| `SECTOR_KPI_COLS` losing ≥1 input | **27 of 57** |
| test files that break | **8 (1,878 lines)** |

The three things that actually bite:
- [ ] The **§6.0 grep-guard test** — no removed name survives anywhere under `src/` or
      `configs/`. A miss is a silent `KeyError` at cube-build time, and the names live in six
      places (`sql/schema.sql`, `schema.py`, `constants.SECTOR_KPI_SCOPE`, `sources.py`,
      `pit.py`, `configs/build_cube.yml`).
- [ ] **`infer_yoy_periods` breaks on the amendment grain.** `pit.py` infers "rows per year" from
      the pooled median `as_of` gap (→ 4) and growth is `pct_change(N)`, a **row** offset. A
      ticker with an amendment has 5 rows that year, so 4 rows back is ~9 months — seasonality
      contamination in exactly the tickers this rebuild handles better. Fix: a **365-day `asof`
      offset**, already declared that way in the KPI JSON so Phase 5/6 cannot forget.
- [ ] `write_part`'s `COLUMNS_CHANGED` path fires → **one full cube rebuild**, by design. Budget
      the wall-clock, print the feature-name delta, and reconcile every disappearance against
      the §6.1 casualty table.
- [ ] The **aggregate fingerprint baseline** regenerates **once**, at the end, with the diff
      attached to the approval request. Not incrementally per builder, or it stops being
      evidence.

**Now runs after the validator**, which is the only ordering change: the cube is built on numbers
that have been measured, and Phase 5b's report is the input to the §6.1 keep-or-re-add call on
the 27 dead sector KPIs.

**Estimated effort**: 4-5 days.

---

### Phase 8: Tests and documentation ⬜

Per v1, plus what this revision creates. Following `docs/testing.md` — real data by default,
synthetic known-truth only for parsing/derivation math, always paired with a real-data coverage
check.

- [ ] `tests/validate/test_fundamentals_validator.py` — planted violations, one per check, plus
      the §5b.4 historical-defect corpus.
- [ ] `tests/data_extract/test_amendment_grain.py` — the §5.0 acceptance test (synthetic original
      + a restating amendment 400 days later → exactly 2 rows, original unchanged, `fiscal_end`
      monotone), paired with the real SMCI/ADM/JPM check.
- [ ] `tests/data_extract/test_statement_role_routes.py` — 4c.1's three-way synthetic
      (note-only / both / undeclared) plus the five real ground-truth instances.
- [ ] Docs, moving with the code: `docs/data_schema.md` (the "239 columns" line is wrong the
      moment Phase 5 lands; **also rewrite the grain sentence** — that `as_of` is a *publication
      event* and amendments append rather than overwrite is the table's whole contract with the
      modelling layer), `docs/data_sources.md`, `docs/architecture.md` (the `validate/` row),
      `docs/config.md` (now **four** fundamentals JSONs), `docs/database.md`, `AGENTS.md`
      (replace one line, cap 70), `README.md` if the CLI surface moved.

**Estimated effort**: 1.5-2 days.

---

### Phase 9: Acceptance — in-sample 26, then out-of-sample 26, driven by the validator ⬜

**Goal**: the plan's stated endpoint, restructured around the two rosters and expressed entirely
as **the validator's own report**. v1's 32-ticker ad-hoc slice is retired: the two 26-ticker
rosters already exist, every Phase 3c/4/4b number is measured on them, and comparability across
phases is worth more than a fresh slice.

**Why two rosters and not one, stated plainly.** The in-sample 26 was **chosen because it broke
things**, so every rule was tuned on it — a pass there proves *consistency*, not
generalisation. The out-of-sample 26 has **zero overlap** and was never tuned; a finding there is
a genuine generalisation failure. This distinction has already paid for itself twice: §3c.9 found
three pre-existing defects (MSFT SG&A, `totalDebt` zero-fill, `employees`) that the in-sample
roster **could not surface**, and §4b.0 found that both Phase-4 carry-forwards were understated
because CB and PLD are out-of-sample only.

#### 9.1 In-sample (26) — build and verify the rules work

| ticker | regime | what it is here to prove |
|---|---|---|
| AAPL · CSCO · KR | industrial | Sep FY / 52-53-week + the 2017 53-week Q4 / Jan year-end, 16-week retail Q1 |
| XOM · APA · EOG · VLO | energy | frozen-TTM baseline · extension revenue total · per-company capex · the ~200× D&A tie-break |
| JPM · BAC · MTB · USB | bank | `RevenuesNetOfInterestExpense` · the FY2023 restatement · 28% frozen TTM · no legacy facts |
| MET · PGR · AFL | insurer | the LDTI 2021 break · tagged ratio concepts · the insurer that tags nothing |
| MAA · SPG | real_estate | Up-C `LegalEntityAxis` extension member, IPR&D-tagged capex · unclassified balance sheet |
| AMT | **industrial** | Telecom Tower REIT → industrial: the GICS regime trap |
| DTE · SO · NEE | utility | parentless revenue root · six registrant CIKs · absent frames |
| ETN · VRT · SWKS | industrial | the two `totalRevenue == 0` cases (one a real SPAC shell) · the 370-day-and-97-day `fp='FY'` |
| BRK-B | hybrid | multi-class, no `AssetsCurrent` |
| GS | broker_dealer | **the only broker-dealer** — a regime v1's own slice never exercised |
| META | industrial | edgartools #691: 0 undimensioned shares-outstanding facts |

- [ ] Build both tables from empty for these 26. Report row and column counts.
- [ ] **Run the validator; the report IS the acceptance artifact.** Findings by check × severity,
      and **every `critical` and `high` finding either zero or explained by name**. "Explained"
      means a named mechanism, not a plausible story — the standard §4b.12 set when it attributed
      all 36 footing failures to restatement / derivation error / defective reference.
- [ ] The user's **three original checks**, each reported as a number:
      **(1)** every quarter of every ticker present to today — baseline **98.74%**, and every gap
      one of the 5 named cases;
      **(2)** no kink from a definition or tag change — `tag_switch_break` + `basis_step` counts,
      each attributed;
      **(3)** `Q1+Q2+Q3+Q4 == FY` on a **non-empty independent set** — baseline **99.12%** within
      2%. *If the set is empty the check is still vacuous and Phase 4 is not done.*
- [ ] **Zero unexplained nulls**: `value IS NULL AND dc_code IS NULL` → 0.
- [ ] **The amendment ladder printed end to end** for the amendment pair (SMCI, ADM): the
      original row, its as-filed value, the amendment row at its own filing date, and the delta —
      proving the original was never retroactively edited. **This is the headline artifact of the
      whole rebuild.**
- [ ] Regression vs the Phase 1 baseline parquet, for the fields present in both: distribution of
      relative differences, and **every case beyond 1% explained**. Differences are expected —
      several are the point — but each needs a named cause. Note **USB, VLO, VRT and V have no
      legacy facts at all**, so there is no baseline to regress them against; expect it, do not
      read it as a failure.
- [ ] **Quantify the coverage the staircase fix deliberately costs** and report it as a headline
      number. A TTM is emitted only from four discrete quarters — measured **93.6%** emitted, the
      rest reason-coded. An unexplained coverage drop is a bug; this one is a design choice.

#### 9.2 Out-of-sample (26) — verify it generalises, then fix the edge cases

Roster (zero overlap, chosen for what the first one misses): **UNH CVS HCA JNJ LLY TMO MSFT NVDA
ORCL WMT COST PG MCD CAT DE BA UNP WFC C SCHW AXP CB DUK PLD EQIX GOOGL** — the entire Health
Care sector (UNH and CVS run insurance economics inside a non-Financials GICS sector; HCA carries
negative stockholders' equity), captive finance (CAT, DE), the broker/bank boundary (SCHW, AXP),
a second multi-registrant utility (DUK), a second data-centre/tower REIT (EQIX), and multi-class
shares outside tech (GOOGL).

- [ ] Build and run the **identical** validator report. Same checks, same thresholds, **no
      re-tuning before measuring** — the point is to see what breaks.
- [ ] Compare the two reports side by side, per check: fire rate in-sample vs out-of-sample. A
      check that fires 10× more often out of sample is either a genuine generalisation failure or
      a threshold fitted to the in-sample roster. **Both are findings; distinguish them.**
- [ ] Fix each out-of-sample finding, then **re-run BOTH rosters**. This is not optional
      ceremony: §3c.8 is the precedent — four defects were created *by* the 3c.1-3c.5 fixes and
      were only visible on the re-sweep. A fix measured on one roster is unmeasured.
- [ ] The out-of-sample-only cases already known, which must be in the report and attributed:
      MCD capex (the 35.6× step, fixed by 4c.1), ORCL FY2020 revenue (decision 24), PG's
      note-level annual, MSFT `depAmort` derivation residual, EQIX FY2023 revenue, the BA/EQIX 31
      values lost to route 3b's period intersection, GOOGL's re-registration gap.
- [ ] **Extend the `by_ticker` extension register** for any out-of-sample filer whose capex or
      D&A is a company extension. This is the register-widening path Phase 9 owns (item 16), and
      it is the only mechanism that reaches an extension leaf — there is **no structural rule**
      that identifies one (§4b.4 measured and refuted the two candidates).
- [ ] **Extend `fundamentals_cik_cutover.json`** for every truncation `filing_continuity` reports
      on this roster. GOOGL is already known and seeded in 4c.6; the check exists precisely so the
      rest surface here rather than in the full-universe run. Both registers now widen with the
      roster, and both are Phase 9's standing obligation.

#### 9.3 Then, and only then: the full universe

- [ ] Extrapolate the wall-clock from the 52-ticker runs. Current estimate: ~490 tickers × ~60
      filings ≈ **30k `filing.xbrl()` calls** at 1.4-5.8 s ≈ **33-83 h single-threaded, 4-10 h at
      8 workers**. One-time backfill; nightly is incremental (~5-8 filings on a quiet night,
      ~20-80 at earnings peak) — **minutes**.
- [ ] Checkpoint per ticker so an interrupted backfill resumes. The resume path (accession set,
      not `max_date_by`) is load-bearing and has its own test.
- [ ] **Never kill the backfill by image name.** A previous multi-hour SEC download was destroyed
      by a blanket `python.exe` kill. **Kill by PID only.**
- [ ] Run the validator on the full universe and report the finding count by check. The **~448
      never-swept tickers** are where the register's 17-filer coverage will show; expect
      `register_coverage` findings, and treat them as a work queue rather than a failure.

#### 9.4 Optional — a random cold roster (recommended)

The out-of-sample 26 is a **designed stress set**, not a random sample: it was picked for
specific hard properties. So it measures *robustness to known-hard shapes*, not the expected
error rate on an arbitrary ticker.

- [ ] Draw **26 tickers at random** from the ~448 never-swept names, seed recorded, and run the
      same report. That is the only honest estimate of what the remaining universe looks like,
      and it costs one sweep (~30 min).

**Estimated effort**: 2 days + rebuild wall-clock (raised from v1's 1 day: two rosters, a
mandatory re-run of both after every out-of-sample fix, and register-widening work).

---

### Phase 10: EFFICIENCY review ⬜

- [ ] Per new module (`fetch_fundamentals_sec.py`, `xbrl_linkbase.py`, `periods.py`,
      `build_history.py`, `fundamentals_validator.py`): spawn a sub-agent with *"ANGLE —
      EFFICIENCY: flag wasted work the diff introduces"*, over `git diff HEAD -- <path>`.
- [ ] Point it at the ones that bite nightly: repeated `store.load` of the same slice;
      unprojected reads; per-ticker frame concatenation in a loop; recomputing the linkbase per
      **field** instead of per **filing**; `apply`/`iterrows` where a vectorised merge exists.
      Add one specific to this phase: **the validator must not re-read `fundamentals_history` per
      check** — load once, projected, and pass the frame down the tiers.
- [ ] Then `/simplify` on the same diff.

**Estimated effort**: 0.5 day.

---

## Testing Strategy

- **Synthetic known-truth** — parsing and derivation math only: the Q4/YTD ladder, linkbase
  weight arithmetic, roll-up and leaf-sum guards, the statement-role three-way, the reason-code
  state machine, one planted violation per validator check.
- **Real data** — the two 26-ticker rosters: resolution per regime, entity scoping, coverage, the
  validator end to end. Every synthetic fixture is **paired** with a real-data coverage check —
  the fixture proves the formula, the coverage check proves it fires on real filings.
- **Regression** — the Phase 1 baseline parquet. Not "no change" (change is the goal) but "every
  change has a named cause".
- **Manual** — read three filings by hand (one bank, one REIT, one E&P) and tie the stored
  numbers to the PDF. Nothing else catches a plausible-but-wrong number, which is this domain's
  characteristic failure.

---

## Risk Mitigation

1. **4c.1 reorders resolution for all 52 tickers.** It is the same class of change as 3c.1,
   which improved 2011-2014 from 0.9% to 73.7% linkbase *and* created four new defects visible
   only on the re-sweep. *Mitigation*: the 3c.1 protocol — before/after on the same join key,
   route-changed % and value-agreed % by year, every material disagreement named — plus a
   mandatory re-sweep of **both** rosters.
2. **The CIK cutover can double-count instead of extending** (4c.6). Apache Corp filed 4×/year
   as a *subsidiary* through 2024-11-07, so a union of the two CIKs blends two legal entities
   over 2021-2024 and would look like a fuller history rather than a corrupted one — the
   dangerous direction. *Mitigation*: the walk is a **dated cutover**, not a union; accession
   dedup is asserted rather than assumed; and load-time validation requires `cutover_date` to
   fall inside the predecessor's own filing window, so a typo cannot silently delete or overlap
   a decade. Expect every APA baseline to move (22 → ~62 filings) and re-baseline deliberately.
3. **Phase 5's append-only grain makes late fixes expensive.** Once history is published, a
   resolution fix needs an explicit versioned `--rebuild`, because re-deriving history under an
   already-trained model is exactly the vintage instability the research measured. *Mitigation*:
   Phase 4c exists, and it comes first.
4. **A validator that over-fires is worse than no validator.** DQC's own experience. *Mitigation*:
   §5b.3 makes calibration a deliverable with a printed fire-rate table, and Layer A stays
   deliberately narrow — the precedent is 745 correct rows nulled by over-strict Q4 guards.
5. **Phase 6 is a silent-miss risk**, not a decision risk. *Mitigation*: the §6.0 grep-guard test.
6. **Restatement disagreement is real and will show up** (BAC FY2023: 98,581 → 102,769).
   *Mitigation*: we read per-filing so we get as-filed, which is the correct basis for a trading
   model. `restatement_ledger` records it so nobody "fixes" it toward `frames`.
7. **Coverage drops on purpose.** *Mitigation*: quantify and headline it (§9.1).
8. **Risk zones**: `constants.py`, `data_store/`, `sql/schema.sql`, `configs/`, and the aggregate
   fingerprint baseline are all touched. Every one gets a proposed diff and approval first —
   batched per phase, as Phase 3 did.
9. **`fundamentals_exceptions.json` is hand-formatted**; a `json.dumps` round-trip reformats all
   545 lines into an unreviewable diff. Validated emitter or text splice only.

**Rollback**: all deletions are in git. `fundamentals_facts_legacy` (7.8M rows) and
`fundamentals_history_legacy` (27,602 × 239) stay in Postgres untouched until explicitly retired
— the new tables are new *names*, so rollback is repointing `schema.py`, not restoring data.

---

## Success Criteria

- [ ] Phase 4c: the five statement-role ground-truth instances correct; both rosters re-swept
      with every value change named; MCD's 2017→2018 step gone.
- [ ] Phase 4c's decisions land and are **measured**, not merely coded: **APA reaches ~62 filings**
      (from 22) with **no duplicated accession** across the cutover boundary; **AXP** resolves on
      one Rule 9-04 basis for its whole history **or** the missing legs are reported and the ban
      withdrawn; **ORCL FY2020**'s $39,068M Q4 is `ambiguous_duration`, and that guard fires on
      that row and no other.
- [ ] `fundamentals_facts` + `fundamentals_history` + `fundamentals_reason_codes` +
      `fundamentals_quality` build end to end for **both** 26-ticker rosters.
- [ ] The user's three checks are **non-vacuous** and pass, with `q4_footing` running on a
      non-empty independent set.
- [ ] **No leakage**: every row's contributing facts have `filing_date <= as_of`; amendments
      append at their own filing date; originals are byte-identical before and after a
      restatement lands. `test_fundamentals_point_in_time.py` green.
- [ ] **Zero unexplained nulls** on both rosters.
- [ ] The validator **re-finds every historical defect** in the §5b.4 table and stays silent on
      every fixed one.
- [ ] Out-of-sample fire rates are within a stated factor of in-sample, per check, or the
      divergence is attributed.
- [ ] No removed field name survives under `src/` or `configs/`; `tests/data_aggregate` green;
      fingerprint baseline regenerated **once**, with approval.
- [ ] New code stays materially smaller than the 7,268 lines removed (2,895 today + ~900 for
      Phase 5 and ~800 for Phase 5b ≈ **4,600** — under target, and the validator is new
      capability rather than replaced volume).
- [ ] Docs in sync; every test prints its sanity-check conclusion; EFFICIENCY review +
      `/simplify` applied.

## Estimated Effort

| Phase | Estimate | Note |
|---|---|---|
| 1-4b | **done** | ~11 days spent |
| **4c** Deferred resolution fixes | **3-3.5 d** + 2 × ~35 min sweeps | new; raised from 2-2.5 by the four decisions — 4c.6's dated CIK walk is ~1 d and touches the fetcher, not just a config |
| 5 History layer | 2.5-3 d | +0.5 d for register items 7-9 |
| **5b** Validator toolkit | **4-5 d** | absorbs v1 Phase 7 (2-3 d) + ~15 scratchpad scripts + Tier 3 |
| 6 Downstream repair | 4-5 d | unchanged |
| 8 Tests + docs | 1.5-2 d | unchanged |
| 9 Acceptance, both rosters | **2 d** + wall-clock | raised from 1 d |
| 10 EFFICIENCY review | 0.5 d | unchanged |
| **Remaining total** | **18-21 days** + backfill wall-clock | |

---

## Decisions taken — 2026-08-23

All four open items are **closed**. Recorded here as well as in §B.4 so the plan reads
self-contained, and so a later reader can see what was chosen *against*.

| # | decision | chosen | rejected, and why |
|---|---|---|---|
| 22 | CIK cutover table | **a new `configs/fundamentals/fundamentals_cik_cutover.json`** | a `sp500_tickers` column — that table is rebuilt from Wikipedia and would silently overwrite curated data; and leaving it unrepaired — costs APA ~10 years and GOOGL ~5 permanently |
| 23 | AXP's revenue basis | **one Rule 9-04 basis via the bank regime** — which AXP is already in, so this reduces to a single `never_use` entry | leaving it as a documented card-issuer break. Note the plan was **wrong** about AXP's current regime (§B.4), which is why the work is a quarter of what the question implied |
| 24 | ORCL FY2020's Q4-windowed annual | **refuse it**, `dc_code = 'ambiguous_duration'` | reclassifying the duration — circular, the only test uses the quarters being derived; and storing $39,068M and flagging it — flag-only never mutates, so the cube consumes it before anyone reads the report |
| 25 | Tier 0, DQC via Arelle per filing | **defer**, with a named revisit trigger in 5b.0 | adopting now — +~2 d, a new dependency, a re-costed 30k-filing backfill. A sampled offline audit is kept as the cheap first step **if** the trigger fires |

**Two implementation defaults taken — not user decisions**, flagged so they can be overridden
before code is written:

1. `longTermDebt`'s priority-2 `us-gaap:LongTermDebt` (4c.2) — demote it, or subtract the current
   leg where it is the only concept available. **The measurement decides**: if subtracting works
   on the filers that need it, subtract; otherwise reason-code. No preference imposed here.
2. `sign_convention`'s debit/credit oracle (5b.0) — **cache the `us-gaap-*-2025.xml` taxonomy**
   under `data/`. A one-off download that also unblocks any future authority work, and it avoids
   the FSNDS bulk-dataset ingest that decision #3 deliberately kept out of scope.


## Known gaps carried from research — interpretation, not blockers

Compustat's exact `CAPX` include/exclude list (so any capex factor ported from a paper is
approximate); whether Compustat `COGS` includes `DP` and `XSGA` includes `XRD`; Compustat's
`REVT` under `INDFMT=FS` for a bank. `us-gaap-doc-2025.xml` holds 14,899 documentation labels
against 17,326 schema elements — **absence from that file is not evidence of absence from the
taxonomy**. And two corrections to keep from resurfacing: **Du, Huddart & Jiang (2023) "Lost in
standardization" is RETRACTED** — do not cite its magnitudes; the **8× rent multiple was
Moody's, not S&P**, and it varied by sector, not region.

---

## Implementation log — Phase 4c (started 2026-08-23)

**Execution order actually used**: the code-only items first (4c.1, 4c.3, 4c.8), then 4c.6
(config + fetcher), with 4c.2 / 4c.4 / 4c.5 / 4c.7 held until the both-roster sweep lands,
because all four are explicitly *measure-first*.

### New committed instruments (the fix for §"the one structural argument")

| file | what it is |
|---|---|
| `scripts/sweep_fundamentals_resolution.py` | the 52-ticker sweep as a **command**. One network pass, **two resolutions** per filing (`prefer_structure` on/off) off one `filing.xbrl()`, so 4c.1's before/after needs one sweep rather than two. Per-ticker parquet ⇒ resumable. `--limit N` + a shell loop is **mandatory**: an all-52 single process reached **14.7 GB RSS** before it was killed, because edgartools' per-filing caches are never released inside a process. |
| `scripts/report_fundamentals_sweep.py` | the offline report over those ledgers: route mix vs the 20% gate, the 4c.1 before/after on the fact-identity join key, 4c.1's two censuses, 4c.3's duplicate census, 4c.4's regime × field coverage, 4c.2's `longTermDebt` basis census, 4c.7's AXP legs. |
| `configs/fundamentals/fundamentals_rosters.json` | the two 26-ticker rosters + the amendment pair, **each ticker carrying why it is on the list**. Validated: 26 / 26 / 2, zero overlap, every ticker has a reason. Approved as a risk-zone change and pulled forward from Phase 5b because 4c's own acceptance needs it. |

### 4c.1 — the plan's premise was wrong, and the repair is a different rule

Implemented as specified **and measured**: `is_note_only` (declared, and every declared role a
non-statement role; silence never counts) on routes 1 and 5, with a relaxation pass so a
note-only concept that is the filer's *whole* answer is kept and flagged `role_only_retained`
rather than nulled. It fires **0 times on 13 swept tickers**, and **none of the five
"confirmed instances" reproduces**:

| case | plan says | measured on the latest 10-K |
|---|---|---|
| CSCO `depAmort` | note-role arc, 2.3× low | `us-gaap:DepreciationDepletionAndAmortization` = **exactly $700,000,000, `decimals=-8`, in FY2023, FY2024 AND FY2025** — the same rounded narrative figure three years running. It has **no calculation arc at all** (`tag_primary`), so no role test can see it. The truth is `csco:DepreciationAmortizationAndOther` **$2,811M**, `decimals=-6` — a **company extension**, so no tag list can name it either. |
| AAPL `depAmort` | regression guard | same shape, already correct: `us-gaap:Depreciation` $8.0bn at `decimals=-8` (the note figure) loses on priority to `DepreciationDepletionAndAmortization` $11,698M at `-6`, **declared** on `.../CONSOLIDATEDSTATEMENTSOFCASHFLOWS`. |
| AMT `longTermDebt` | $1.9M note-level | neither `LongTermDebtNoncurrent` nor `LongTermDebt` is reported undimensioned; it resolves to `LongTermDebtAndCapitalLeaseObligations`, **declared on the balance sheet**. Its `depAmort` shows the same `decimals=-8` pattern (`us-gaap:Depreciation` $1.1bn) and already loses to the declared `DepreciationAmortizationAndAccretionNet` $2,041.6M. |
| MCD `capex` | ~12× low | already `statement_leaf_sum`; `capex`'s priority-2 concept is a `roll_up.sum` **leg**, so route 1 never takes it. Nothing withheld. |
| PG `totalRevenue` | note-level $28,400M | `us-gaap:Revenues` via `linkbase_total`. Its `depAmort` candidate is declared on the cash-flow statement. |

**So the discriminator in every measured case is DECLAREDNESS (and, corroborating it,
`decimals`) — never a note-role URI.** Decision taken 2026-08-23: reformulate 4c.1 as a
declaredness test, keeping the role test as its second, narrower half.

- [x] **`is_note_only`** — routes 1 and 5, positive evidence only, no coverage cost. Kept as a
      standing guard against a class that does occur in principle; 0 fires today.
- [x] **The declaredness test** — where a route-1 candidate carries **no calculation arc
      anywhere** and the filer **does** declare this field's own statement lines under its
      anchor node, route 3b wins. Recorded per row as `undeclared_rejected`.
- [x] **Scope is two fields.** Only `capex` and `depAmort` declare `roll_up.any_of`, so route
      3b cannot fire for anything else — which is what makes this safe against the XOM case
      that set Phase 3's *"candidate priority dominates linkbase presence"* rule
      (`sharesOutstanding` has no `any_of`; a named test asserts it).
- [x] `Resolution` carries `role_rejected`, `role_only_retained`, `undeclared_rejected`; all
      three ride the `adjustment` JSON, so `adjustment::jsonb ? 'undeclared_rejected'` finds
      every row 4c.1 reordered. No schema change to a risk-zone table.
- [x] `resolve_field(..., prefer_structure=False)` is the **measurement seam** that makes the
      before/after one sweep instead of two. Production never passes False.
- [x] `tests/data_extract/test_statement_role_routes.py` — **9 tests**: the note-role
      three-way (reject / keep-on-both / keep-undeclared) plus the no-coverage-cost case; the
      declaredness three-way (undeclared loses / declared keeps winning / no-`any_of` field is
      out of scope); and two real-data tests over six live 10-Ks.
- [ ] **CSCO still reads $700M** until it gets a `by_ticker` extension-register entry —
      measured: `leaves: [["DepreciationAmortizationAndOther"]]` plus
      `not_leaves: ["AccountsReceivableAndFinancingReceivableCreditLossExpenseReversal"]`
      (the unclassified extension sibling in the FY2025 node, without which route 3b refuses
      with `partial_leaf_sum`). Verified to produce the cash-flow line on the 2016 and 2020
      10-Ks. **Blocked on the batched `configs/` approval.**
- [ ] Both-roster before/after per fiscal year — **blocked on the clean sweep**.

### 4c.3 — the duplicate rule, and the primitive the real 4c.1 mechanism needed

- [x] "Last duplicate wins" → **the finer `decimals` wins**, with `_precision` handling `INF`,
      negative scales and absent values (absent ranks lowest, so a fact that declares its
      precision always beats one that does not).
- [x] Every disagreement is **recorded** as `duplicate_fact` on the surviving period —
      concept, both values, both `decimals` — whether or not the tie-break changed the answer.
      Identical re-tagging (the common case) is not a disagreement and is not flagged.
- [x] The ledger is **unioned across legs** in `_materialise` and across components in
      `_compose`; taking only the first leg's copy silently loses a duplicate in the second.
- [ ] Fire rate on both rosters — **blocked on the clean sweep** (0 on the 13-ticker partial,
      but those parquets predate the change).

### 4c.6 — the CIK cutover: DONE and verified against live EDGAR

- [x] `configs/fundamentals/fundamentals_cik_cutover.json` — APA `6769 → 1841666` @2021-03-01,
      GOOGL `1288776 → 1652044` @2015-10-02, ETN `31277 → 1551182` @2012-11-01. Approved.
- [x] `cutover_date` is defined as **the first date on which the successor is the reporting
      registrant**, which is what makes the two walks disjoint by construction. ETN's date is
      2012-11-01 rather than the transaction's late-November close, because the successor's
      first filing is 2012-11-14 and the predecessor's last is 2012-10-31 — the only boundary
      that is both gapless and non-overlapping.
- [x] `src/data_extract/utils/fundamentals/cik_cutover.py` — loader with strict validation:
      `kind: rename` **rejected by name** (a rename keeps its CIK, so an entry would walk one
      CIK twice), `predecessor_cik == successor_cik` rejected after zero-padding, empty
      `evidence` rejected.
- [x] `build_ticker_fundamentals` walks both CIKs on the date split and stamps each row with
      **the registrant that actually filed it**. Dedup is asserted, not assumed: a cutover that
      loses accessions in dedup raises.
- [x] `tests/data_extract/test_cik_cutover.py` — **9 tests**, all green on live EDGAR.

**Two measured facts worth keeping:**
- APA: **15** Apache Corp 10-K/10-Q filings dated 2021-05-07 → 2024-11-07 are correctly
  **dropped**. A union would have blended a subsidiary's statements into the parent's.
- GOOGL: the two CIK indexes are **NOT disjoint** — `0001652044-16-000012` and
  `0001193125-16-520367` appear under both, because Google Inc stayed a **co-registrant** on
  Alphabet's first 10-K. Both are post-cutover, so the dated walk takes each exactly once.
  This is the plan's warning made concrete, and the test asserts disjointness of what is
  **kept** rather than of the indexes.

### 4c.8 — ORCL's Q4-windowed annual: DONE, and the plan's prescribed gate was impossible

- [x] The plan says *"gate it on the window LENGTH, not on the value"* — *"a ~365-day fact
      tagged into a quarterly slot"*. **Measurement refutes the premise**: ORCL's fact is
      `period_start=2020-03-01, period_end=2020-05-31`, **91 days**, so `duration_type` is
      `quarterly` and there is no length anomaly to see. The plan's description of the
      mechanism ("stamp the full-year figure with a Q4 window") is right; its prescribed test
      contradicts it.
- [x] **The non-circular evidence that IS available is the filer's own nine-month
      cumulative** — an as-filed fact, not a derived quarter, so the rule stays inside decision
      24's constraint. D1b fires when: no annual fact ends within 7 days of the quarter's end
      (i.e. D1 declined to judge it), AND a contiguous `ytd9` exists, is >1% of the quarter
      (the same materiality floor as D1's condition 3), AND is **smaller** than the "quarter".
      A fourth quarter exceeding the nine months before it would have to be more than three
      quarters of the year.
- [x] `AMBIGUOUS_DURATION`; refusals surface through an optional `refusals` list on
      `quarterize` / `build_periods` so Phase 5 writes them to `fundamentals_reason_codes` and
      that table stays the single source of truth.
- [x] **A bug the change exposed**: `_drop_annual_masquerading_as_quarter` short-circuited on
      `annual.empty`, which disables D1b on precisely the frames it exists to judge.
- [x] Measured on real ORCL filings: **exactly one period refused** (FY2020, 3 facts, one per
      vintage), and fiscal 2018 / 2019 / 2021 / 2022 all keep an `fy_minus_ytd9` Q4 of
      **$11.0-11.8bn** — D1 already handles those, because each has a real annual fact. The
      fire rate is the plan's stated population exactly.
- [x] **The fixture window is load-bearing and is documented as such.** The annual fact that
      lets D1 handle fiscal 2021 and 2022 arrives in the FY2023/FY2024 10-Ks; truncate the
      window at 2022 and D1b fires on three years instead of one. `quarterize` is handed the
      ticker's whole stored history, so a narrow fixture tests a different question.
- [x] 4 tests (3 synthetic + 1 real); `tests/data_extract/test_periods_q4.py` now **28 green**.

### Still open in Phase 4c

| item | why it is waiting |
|---|---|
| **4c.2** `longTermDebt` | *"measure both before choosing"* — the basis census is section 6 of the report |
| **4c.4** the six detail fields | must be measured off `filing.xbrl()` per regime — section 5 |
| **4c.5** standing assertions | needs the post-4c numbers to assert on |
| **4c.7** AXP | *"check the legs first"* — section 7. GICS **verified**: AXP is `Consumer Finance` ⇒ `bank` already, so §B.4 item 23's correction to v1 is confirmed and only a `never_use` entry is in scope |
| CSCO `by_ticker` entry | batched into the next `configs/` approval |

**Test state**: 810 → **832** collected, 0 errors.

---

## Implementation log — Phase 4c, part 2 (2026-08-23)

### 4c.2 — `longTermDebt`: the priority order now follows the balance sheet's own shape

The plan offered "demote **or** remove". Removing it was measured and rejected: `us-gaap:LongTermDebt`
is the only debt concept 103 rows tag at all, so removal converts them to nulls for no accounting gain.
Demoting it is free. The change is entirely in `fundamentals_kpis.json`:

| regime class | `total_concept` | `fallback_concepts` order |
|---|---|---|
| classified (industrial, energy, utility, hybrid) | `LongTermDebtNoncurrent` | Noncurrent -> AndCapitalLease -> **LongTermDebt last** |
| unclassified (bank, insurer, broker_dealer, real_estate) | **`LongTermDebt`** | **LongTermDebt first** -> Noncurrent -> AndCapitalLease |

**Why the split, not a blanket rule.** FASB defines `LongTermDebt` as the total *including* the current
portion. On a classified sheet that is a **different basis**, so it must be a last resort. On an
unclassified sheet -- Reg S-X Article 9 / 7 / 12, and `17 CFR 210.1-02(bb)(1)(i)` for REITs -- there is
nothing current to exclude, so it *is* the noncurrent figure and putting it first removes the
oscillation rather than causing it.

**Measured, 52-ticker x 2011-2026 sweep:**

- classified: `LongTermDebt` won **488 of 4,151** rows. Demotion moves **385 (79%)** onto a declared
  noncurrent line. `LongTermDebtAndCapitalLeaseObligations`, now ahead of it, reaches this field's
  exact basis via the pre-existing lease subtraction, so the promoted route is *better*, not merely
  different.
- unclassified: it already won **1,326** rows against 78 and 92 for the other two, so the override
  makes the priority match the accounting instead of leaving it to the filer's tagging habit.
- residual: **103 rows, 3 tickers** (EQIX 60, SWKS 42, ORCL 1) where it is the only concept tagged.

**No subtraction was added, and this is the measured reason.** The obvious "subtract the current
portion" fix does not work: SWKS and ORCL never tag `LongTermDebtCurrent` at all -- their current leg
is `us-gaap:DebtCurrent`, which is *all* current debt (revolver, commercial paper), not the current
maturities of long-term debt. Subtracting it would manufacture a wrong number. EQIX exposes a current
leg on only 4 of its rows, at 7.8%. So the residual carries a 4-8% wider basis, disclosed in config,
which is the honest answer and far short of the x11,545 class of error this item was opened for.

### 4c.4 — the absence register, on a second and explicitly-labelled substrate

31 cells written across all 8 regimes, **10** with `expected_absent: true`:

| field | true for | measured |
|---|---|---|
| `accountsPayable` | bank, broker_dealer, insurer, real_estate | 7/7, 2/2, 4/4, 3/3 |
| `accountsReceivable` | bank, broker_dealer, insurer, real_estate | 7/7, 2/2, 4/4, 3/3 |
| `ppeGross` | real_estate | 3/3 |
| `accumulatedDepreciation` | real_estate | 3/3 |

The mechanism is verbatim, not analogical: the concepts are literally `AccountsPayableCurrent` and
`AccountsReceivableNetCurrent`, so `17 CFR 210.1-02(bb)(1)(i)` applies word for word. For real estate
`ppeGross`/`accumulatedDepreciation`, the balances are disclosed under the real-estate concept family
(`RealEstateInvestmentPropertyAtCost`), so the absence is a tagging-family fact.

Every other cell is `false` **with its measured rate recorded**, including two that a coverage-driven
reflex would have silenced:

- `intangiblesExGoodwill` / real_estate is **3/3 absent and deliberately left `false`** -- REITs
  commonly *do* carry acquired in-place-lease intangibles, so 3-of-3 on n=3 is at least as likely a
  resolution gap as a disclosure fact. Silencing it would hide the defect the register exists to find.
- hybrid is 100% absent on all five fields but **n=1**, so nothing is silenced from it.

New cells carry `"_basis": "sweep_2026_08_23"` and their own `_n_basis`, and the README now states
that the sweep basis asks a *different question* from the legacy basis ("does the field resolve" vs
"does a fact exist") on a much smaller denominator -- the two rates must never be averaged or compared.

### 4c.5 — the standing assertions

Two criteria added to `test_linkbase_history.py`, written as criteria rather than as the counts this
roster happens to produce:

- `test_the_debt_basis_never_switches_to_the_current_inclusive_element` -- no classified filing may mix
  the current-inclusive element with a noncurrent line. Asserts the *ordering*, not disuse, so the
  103-row last-resort path stays legal.
- `test_noncurrent_debt_never_steps_by_an_order_of_magnitude` -- the boundary must be invisible in the
  series, on latest-vintage values per balance date (differencing two vintages measures restatement).
  Ceiling 10x; pre-4c.2 AMT stepped x11,545.

Green on the FAST_SUBSET, which includes SWKS (a residual ticker) and DTE (tags all three concepts).

### Verification status at close of Phase 4c

| gate | result |
|---|---|
| `tag_fallback` share | **5.71%** vs the 20% ceiling |
| 4c.1 route change | 0.127% of rows; value agreed to the dollar on 99.925%; 128 lost / 6 gained, all route-3b period intersection; `totalAssets`/`goodwill`/`restrictedCash` provably untouched |
| 4c.8 D1b fire rate | **1** distinct (ticker, field, period) -- ORCL FY2020, 3 vintages. Quarter basis mix: 68.18% as-reported, 21.10% `fy_minus_ytd9`, 5.35%/5.34% the two ladder steps, 0.03% the `fy_minus_q1q2q3` fallback |
| Q1+Q2+Q3+Q4 vs FY | 8,793 complete four-quarter years; **1,318 independent** (non-tautological) points at **94.61% within 2%**, 80.12% exact to the dollar |
| test suite | 311 passed / 10 skipped on `tests/data_extract`, plus 13 in the two files `-x` had cut short. One unrelated failure: `test_short_interest_resume` (SEC fetch returned 0 rows; nothing in `utils/prices/` was touched) |

---

## §B.6 — Phase 4c residue: edge cases to close AFTER the validator

**Sequencing decision (2026-08-23, user).** These are not skipped work. Each is a case where the
correct answer is a per-value judgement, and Phase 5b's validator is the instrument that makes such a
judgement testable instead of a guess. Attempting them now would mean hand-reconciling individual
filings; doing them after 5b means each is a named check with a calibrated threshold.

| # | item | what is known now | why it waits for the validator |
|---|---|---|---|
| B.6.1 | `longTermDebt` widening for **AFL, BRK-B, MAA** (was 4c.2 item 3) | MAA: `us-gaap:NotesPayable` $5,405M = its own declared `SecuredDebt` $360M + `UnsecuredDebt` $5,045M. AFL: `us-gaap:DebtAndCapitalLeaseObligations`, a declared +1.0 child of `Liabilities`. BRK-B: segment-dimensioned only -- a declared absence, since §B.5 forbids relaxing the dimensional filter | Both candidates are **total** debt lines. Adding them to `longTermDebt` would double-count inside `totalDebt`'s roll-up, and the check that catches double-counting is 5b's `roll_up_consistency` |
| B.6.2 | the absence-register cell for that set (was 4c.2 item 4) | ties to B.6.1 | a cell asserting "legitimately absent" is only meaningful once `coverage_field` consumes it |
| B.6.3 | `intangiblesExGoodwill` / real_estate | 3/3 absent, left `false` on purpose (see log above) | needs re-measurement on the full roster; n=3 cannot distinguish a resolution gap from a disclosure fact |
| B.6.4 | completing the register for the remaining instant fields | 17 instant fields still have no cell (`cash`, `goodwill`, `restrictedCash`, `ppeNet`, `stockholdersEquity`, `totalLiabilities`, ...) | same reason as B.6.2 -- write the cells the validator's findings actually demand, not a speculative sweep |
| B.6.5 | cutover load-time validation, 3rd rule (was 4c.6) | `predecessor_cik != successor_cik`, `kind`, date parseability and non-empty `evidence` are enforced in `load_cutovers`. Missing: **ticker exists in the universe** (needs the roster at load time, which `load_cutovers` has no handle on) and **`cutover_date` inside the predecessor's filing window** (needs network) | the window rule is exactly 5b's `filing_continuity` check; the universe rule is a loader-signature change worth doing once, alongside it |
| B.6.6 | the 128 rows 4c.1 costs, with **no `dc_code`** | EQIX capex 40, EQIX depAmort 40, SCHW cash 34, NEE ppeNet 8, VRT depAmort 6. Cause: `filing_rows` only reason-codes a field with **no periods at all**, so a route-3b partial period intersection drops silently. This is the plan's own **item 9** -- estimated at 31, actually 128 | a silent null is precisely what the validator must surface. Fixing the reason-coding first and then re-measuring is one change; guessing at it now is two |
| B.6.7 | 538 duplicate-fact disagreements (4c.3 ledger) | worst BA `operatingCashFlow` 45%, WMT `shortTermDebt` 10.7%. The `decimals` tiebreak now resolves them deterministically and records both sides | the ledger exists so 5b can threshold it. Which disagreements are *material* is a calibration question, not a resolution one |
| B.6.8 | both-roster re-sweep after the 4c.2 / 4c.4 config change (Risk 1) | the 4c.2 change is validated by the two new standing tests; the ledgers in `data/fundamentals_sweep/` were built pre-change and are now stale for `longTermDebt` only | a full re-sweep is ~2h of network and 14.7 GB RSS. Better spent once, driven by the validator, than twice |
