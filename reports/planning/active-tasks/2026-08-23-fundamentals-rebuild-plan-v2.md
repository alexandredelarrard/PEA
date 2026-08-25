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

5. **Phase 5 is now fully specified** (§Phase 5, added 2026-08-23 after a planning interview).
   It had been a 48-line stub deferring to v1; the interview established that **neither file
   alone was implementable** — v1 held the same-day collapse rule, the complete-snapshot
   property, the `build_history.py` spec and the reason-code tuple; v2 held three register items
   and five codes v1 never named; and the two listed *different* reason codes, so the real code
   set existed nowhere. Fourteen decisions were taken (§5.9), three of them **user challenges to
   the plan as written**: `totalLiabilities` should read the filer's own liability legs before
   falling back to an identity (§5.1); a >1-year-late amendment is unlearnable and should not
   emit a row (decision 34); and the rebuild model is delete-and-refetch, not a versioned
   overwrite (§5.6). The interview also **resolved the 68-vs-71 column contradiction** and found
   that Phase 5 **could not pass its own zero-null gate** as sequenced (decision 26).

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
| 8 | **`totalLiabilities`'s derived fallback** (`totalAssets − stockholdersEquity`) is specified, not implemented. | v1 §3c.6 · research §4.2 | **11 tickers, ~20% of the swept universe** at zero coverage (APA DTE EOG ETN VLO + DUK LLY MCD ORCL TMO WMT). Reg S-X 5-02 has no "Total liabilities" caption, so this is systemic and the derivation is load-bearing, not a nicety. **⚠ RE-SCOPED 2026-08-23 (decision 30)**: the identity is demoted to a *fallback*. §5.1 reads the filer's own liability legs first — a **facts**-layer change, measured before written |
| 9 | Route 3b's **period intersection** costs values. | v1 §4b.12 | Estimated 31 out-of-sample values (BA 23, EQIX 8) and BA's fiscal-2011 **annual** point; **measured 128** (B.6.6). The obvious repair reintroduces the mixed basis route 3b exists to prevent → keep strict, prove it is reason-coded. **⚠ This item and B.6.6 are the SAME item, double-assigned.** Resolved to Phase 5 (decision 26) because the zero-null gate cannot pass while it is open |

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

### Phase 5: The history layer ✅ — SPECIFIED 2026-08-23, DELIVERED and VERIFIED 2026-08-24

> **Reading this section.** Everything below is the SPECIFICATION as written on 2026-08-23; its
> `- [ ]` boxes are left as written rather than back-ticked, because measurement during delivery
> **refuted four of them** and back-ticking would hide that. What was actually built, and the
> numbers it was accepted on, are in the implementation log at the end of this file. The
> departures, each measured before it was made:
>
> 1. **§5.1 "add `roll_up.any_of` and let route 3b sum them" — NOT DONE, refuted.** The spec
>    itself required reporting the refusal rate first, and that measurement killed it: **0 of 44**
>    sampled 10-Ks declare a `Liabilities` total, 44 of 44 declare a leg-set, but route 3b
>    ENUMERATES legs and drops an unlisted us-gaap sibling **silently** -- and the leg-sets vary by
>    filer AND by year (EOG, TMO, MCD all change between 2023 and 2024). A Tier-1 balance-sheet
>    total short by a caption, which is the `shortTermDebt` defect this rebuild exists to remove.
>    The identity won on the merits, exactly as the spec's own escape clause allowed.
> 2. **§5.1 "stamp the identity path `resolution_method = 'derived_identity'`" — DONE ELSEWHERE.**
>    Putting a derived number in `fundamentals_facts` would contradict that table's documented
>    contract ("every row carries a number the filer actually tagged"), which the
>    publication-event grain rests on. The identity lives in the HISTORY layer and is stamped in
>    `fundamentals_reason_codes` instead, so the cell never reads as resolved evidence. The 5b
>    `cross_identity` obligation is unchanged.
> 3. **§5.1 "reason-code `not_disclosed` only when the legs refuse AND the equity basis is ex-NCI
>    AND `minorityInterest` did not resolve" — SUPERSEDED, and this was the right call.** That
>    rule refuses whenever NCI is NULL, which conflates "not tagged" with "genuinely zero" and cost
>    MCD its entire `totalLiabilities` history (assets $59.92bn, equity **-$1.02bn**, so $60.94bn of
>    liabilities left NULL). Delivered as a four-step ladder instead -- **tagged NCI -> NCI DEDUCED
>    from the filer's two equity bases -> assumed zero, only where the filer has never tagged one,
>    under its own `derived_identity_nci_assumed_zero` code -> refuse.** `totalLiabilities` NULL
>    fell **210 -> 38**, and all 38 survivors are missing `stockholdersEquity` outright, so nothing
>    refuses for want of an inference any more.
> 4. **§5.2's contract is 69 columns, not 68.** `fiscal_quarter` was added on user request
>    mid-phase, and the 60 value columns were reordered into STATEMENT order.
>
> **A full adjudication of every `- [ ]` below -- CLOSED / DIFFERENT / DO NOT DO / STILL
> OPEN -- is in `## PHASE 5a FINAL STATUS` at the end of this file. Read that before
> starting 5b.**
>
> One spec item was dropped as unnecessary rather than superseded: the `totalLiabilities`
> `derived_fallback` config key. The basis is now queryable per row in
> `fundamentals_reason_codes`, which a config key would have restated without making measurable.

> **Why this section is now long.** v2 previously said *"unchanged from v1 §Phase 5 in design —
> read it in v1."* The planning interview of 2026-08-23 established that **neither file alone is
> implementable**: v1 holds the same-day collapse rule, the "row is a complete snapshot"
> property, the `build_history.py` implementation spec and the reason-code tuple, none of which
> v2 mentioned; v2 holds three register items and five reason codes v1 never named. The two files
> also listed **different** reason codes, so the real code set existed in neither. This section
> is therefore **self-contained and supersedes v1 §Phase 5** wherever they differ. Fourteen
> decisions were taken; they are tabulated at §5.9.

**Goal**: `fundamentals_history` on the publication-event grain, **68 columns** *(specified as 68; **69 as delivered** -- `fiscal_quarter` was added on user request mid-phase, see the implementation log)*, plus two new
side tables (`fundamentals_reason_codes`, `fundamentals_employees`), built and validated on the
**52 roster tickers**. The full universe is Phase 9.3's job, deliberately — a resolution bug
found at 5b then costs a rebuild of 52 tickers, not 500.

#### 5.0 The grain — one row per publication event

`as_of` is a **filing date**, not a period end. The five rules, restated in full because
everything downstream leans on them:

1. **A row is emitted for every `(ticker, date)` on which ≥1 extracted value became newly
   public.** Originals always qualify.
2. **An amendment emits a row only if it changes ≥1 extracted value *and* lands ≤365 days after
   the original** (→ decision 34). The value test discards the **88** Part-III/cover-only
   amendments and is stricter than a fact-count threshold, because an amendment can re-tag 200
   facts to identical values and still be a no-op.
3. **The row is a complete snapshot.** Every column carries its latest-known value as of that
   date; unamended columns are identical to the previous row. The row is self-contained, so a
   plain `asof` merge works with **no reconstruction**. This is the property `PitFrames` depends
   on and the reason §5.3's reason codes are dense rather than sparse.
4. **Rows are immutable once written.** The nightly job only ever appends; see §5.6 for how that
   is *enforced* rather than asserted.
5. **Same-day collapse**: group publication events by `(ticker, date)`, never by accession. Two
   filings on one day produce **one** row reflecting both.

**Both 10-K and 10-Q filing dates are publication events, and the 10-K's FY fact is load-bearing**
(decision 36). Phase 4's primary Q4 route is `FY − YTD9`, and the FY fact exists only in the
10-K; it is what drove tautological Q4 from **100% → 0.07%** and `epsDiluted` computability from
**8% → 87.8%**. `fundamentals_facts` already keys on `accession_number`, so a 10-Q and a 10-K
filed the same day are already two distinct fact sets — nothing there needs changing.

**Measured amendment population** (`fundamentals_facts_legacy`, 7.8M facts / 25,279 filings):
246 amendments (`10-K/A` 138, `10-Q/A` 108) = **0.97%** of filings across 91 + 83 tickers;
**88 (36%)** carry <10 facts, 113 (46%) carry 100-399, **38 (15%)** carry 400+; lag avg **89 d**,
max **921 d**; **57 of 217 (26%)** land >90 days after the original. v1's row-count estimate was
**+158 on 27,602 (+0.6%)** = 246 − 88. **That figure must be re-measured**, because decision 34
adds a >365-day cutoff that v1's arithmetic did not have.

**Two consequences worth stating plainly, because they are not obvious:**

1. **A restated value propagates further than one cell.** The table stores **TTM levels**.
   Restating Q1-2024 revenue moves the TTM for Q1, Q2, Q3 *and* Q4 2024. That is mechanically
   correct: the amendment row must show the world **recomputed with the restated quarter**. What
   stays frozen is the **earlier rows**, which keep their as-filed values forever. *That is where
   the no-leakage property lives* — expect this to be re-litigated as a bug during
   implementation, and do not.
2. **`fiscal_end` must stay monotone non-decreasing in `as_of`.** An amendment to Q1-2024 filed
   2024-08-20, after Q2-2024 was filed 2024-05-01, carries `fiscal_end = 2024-06-30` — the latest
   period *known* at that date. The restated period goes in `amended_fiscal_end`.

**This is NOT the full vintaged history the user declined.** That option put `knowledge_date` in
the PK and appended a complete vintage on every nightly rebuild (~365/year regardless of
filings). This keeps the PK at **`(ticker, as_of)`** and adds rows only when a *filing* makes new
information public (~4-6/ticker/year). Strictly cheaper, and `PitFrames` forward-fill works
unchanged.

**Same-day provenance resolves by precedence** (decision 37), so every column stays scalar and
queryable: `publication_form` = the highest-precedence form present
(`10-K` > `10-K/A` > `10-Q` > `10-Q/A`); `is_amendment` = OR; `amended_fiscal_end` = the latest
restated period; `amended_fields` = the union. Accession-level detail is always recoverable from
`fundamentals_facts`.
- [ ] Measure and report how many `(ticker, date)` pairs actually collapse. If it is a handful,
      say so; if it is common, the precedence rule needs its own test.

#### 5.1 `totalLiabilities` — measure the linkbase FIRST, before any history row exists

Register item 8 specified `totalLiabilities = totalAssets − stockholdersEquity`. The interview
rejected that as the *primary* route (decision 30): the field currently declares **one** concept
(`total_concept: Liabilities`, `fallback_concepts: ["Liabilities"]`) and **no roll-up at all**, so
the filer's own liability legs — which Reg S-X 5-02 *does* prescribe as captions 21-31, unlike the
absent "Total liabilities" caption — have never been read. Summing them is the filer's own
evidence; the identity is an inference.

**This is a facts-layer change and it reopens Phase 4c's closed resolution layer**, so it comes
first and it is measure-first, exactly as 4c.2 and 4c.4 were.

- [ ] Read the **calculation linkbase** off `filing.xbrl()` — never companyfacts (§B.5: it
      publishes no company-extension taxonomy and drops dimensioned facts) — for the **11**
      tickers at zero coverage: **APA DTE EOG ETN VLO DUK LLY MCD ORCL TMO WMT**. Question:
      under `LiabilitiesAndStockholdersEquity`, does the filer declare a liability leg-set?
- [ ] **Do not write the config from the Reg S-X caption list.** §B.5 measured and refuted that
      style of reasoning twice (name-matching D&A leaves: 14 of 84 within 2%; "a negative-weight
      extension child of the investing node is capex"). Element names are not evidence of what a
      filer declares.
- [ ] Where the legs are declared, add `roll_up.any_of` and let **route 3b** sum them. Note the
      known risk: route 3b's **strict period intersection** refuses the whole sum if any leg is
      missing in a period, and it already costs 128 rows on two *short*-legged fields (B.6.6). A
      liability leg-set is longer, so **report the refusal rate before adopting it** — if it is
      worse than the identity's coverage, the identity wins on the merits.
- [ ] Fall back to the identity where the legs refuse, with the **NCI bridge**:
      `totalAssets − (equity if the equity row's own `source_concept` is the incl-NCI concept
      else equity + minorityInterest)`. §B.5 withdrew the NCI objection on the grounds that the
      *concept order* puts incl-NCI first — but priority-first is not the same as what actually
      won for a given filing, so read the row, not the list.
- [ ] Stamp the identity path `resolution_method = 'derived_identity'` so it never reads as a
      resolved fact, and make it a `cross_identity` **input** in 5b, never independent evidence.
- [ ] Reason-code `not_disclosed` only when the legs refuse **and** the equity basis is ex-NCI
      **and** `minorityInterest` did not resolve.
- [ ] **Pair the required re-sweep with B.6.8's outstanding one** so ~2h of network and the
      14.7 GB RSS `--limit N` loop are spent **once**, not twice. Risk zone
      (`fundamentals_kpis.json`) → propose the diff.

#### 5.2 The column contract — exactly 68, enumerated *(specified as 68; **69 as delivered** -- `fiscal_quarter` was added on user request mid-phase, see the implementation log)*

v1 said "~71" twice and "68" once; **neither number had an enumeration**, yet *"column count is
exactly as contracted"* is a verification item. The discrepancy is now resolved: v1's 68 =
53 catalogue + 10 derived + 5 keys, where the 5 keys are the legacy table's own
`ticker, as_of, fiscal_end, sector, industry_group`; "~71" was that same set plus `regime` plus
provenance (which is 4 columns, hence 73 — the tilde was hiding an unfinished count).

**The contract, after decisions 31-33 and 35:**

| family | n | members |
|---|---|---|
| catalogue | **52** | 11 T1 + 12 T2 + 16 T3 + 1 T3R + 13 tier-0 inputs, **less `employees`** (→ §5.4) |
| derived | **8** | `grossMargins` `operatingMargins` `profitMargins` `returnOnEquity` `debtToEquity` `revenue_q` `netIncome_q` `optionOverhang` |
| keys | **3** | `ticker` `as_of` `fiscal_end` |
| regime | **1** | from `fundamentals_regimes.json`; already stamped per filing on `fundamentals_facts`, so take it from there |
| provenance | **4** | `publication_form` `is_amendment` `amended_fiscal_end` `amended_fields` |
| **total** | **68** | |

- [ ] **One column per field, bare name, TTM basis for flows** (decision 31): a `kind: duration`
      field's column holds the **TTM**; a `kind: instant` field's column holds the latest
      instant. This matches the legacy naming convention exactly (`totalRevenue` *was* the TTM),
      so `build_cube.yml` and `SECTOR_KPI_SCOPE` need no renaming. The `_ttm` suffix in the KPI
      JSON's `_derived_columns` prose is naming the *concept*, not a column.
- [ ] **`ebitda_q` and `freeCashflow_q` are dropped.** The legacy table had four `_q` columns; the
      contract keeps two. This was an unflagged casualty of v1's derived-10 and is now a declared
      one — reconcile it in Phase 6's §6.1 casualty table.
- [ ] **`sector` and `industry_group` leave the table** (decision 32). They are a slowly-changing
      dimension joinable from `sp500_tickers`; carrying them duplicates it inside a
      point-in-time table. ⚠ **Note the residual honestly**: `regime` stays, and it is derived
      from `sub_industry` from that same non-vintaged roster, so the look-ahead this removes from
      two columns is still present in one. It is accepted because `regime` drives resolution and
      cannot be joined at cube time. **Phase 6 grows** — 9 `data_aggregate` modules read
      `sector`/`industry_group` off this frame today, and `constants.py:516` documents the old
      behaviour.
- [ ] `revenueGrowth` and `earningsGrowth` leave the table (decision 33, §5.7).
- [ ] `capexGlobal` is not in the catalogue and not in the derived-8 → dropped. Confirm against
      §6.1.
- [ ] Every column traces to a catalogue entry, a derived formula, or this table. Assert the
      count.

#### 5.3 `fundamentals_reason_codes` — dense, and the code set written down once

`(ticker, as_of, field, dc_code, combined_into)` — a long side table, per decision #11, joining
`fundamentals_history` on `(ticker, as_of)`. v1's stated deviation from the user's spec stands and
is **reversible**: the user asked for a Compustat-style `_DC` companion column per value, which is
39 extra columns and a ~50% wider table, almost entirely NULL.

**Population is DENSE** (decision 29): one row per null-or-qualified cell, **at every publication
event**. So APA's `totalLiabilities` gets a row at all ~62 of its filing dates. Cost is
~150-200k rows universe-wide, mostly repetition. Bought with it: the zero-unexplained-nulls gate
is a one-line `LEFT JOIN` on `(ticker, as_of, field)`, 5b's `unexplained_null` needs no
reconstruction, and *"was this null explained on 2019-05-02?"* stays a lookup. A sparse
state-change grain would have contradicted §5.0 rule 3, which is the whole contract.

**The code set is the UNION of v1's list, v2's list and what `periods.py` emits today.** It
existed in no single place before now.

| code | meaning | emitted by | status |
|---|---|---|---|
| `insufficient_quarters` | fewer than 4 discrete contiguous quarters | `periods.py` | ✅ exists |
| `split_basis_mismatch` | share-count basis break across a split (45 windows / 8 tickers — AAPL 7:1 and 4:1, BRK-B's 246,000× class ratio) | `periods.py` | ✅ exists |
| `ambiguous_duration` | 4c.8's D1b — an annual masquerading as a quarter with no annual to compare against (ORCL FY2020) | `periods.py` | ✅ exists |
| `not_disclosed` | the filer tags nothing for this field | history build | new |
| `not_applicable_for_regime` | `fundamentals_exceptions.json` says the field cannot exist in this regime | history build | new |
| `not_applicable` | structural, not regime-table-driven: bank `capex`, and therefore bank `freeCashflow` | history build | new |
| `combined_into` | folded into another field; carries the destination in `combined_into` | history build | new |
| `unresolved` | resolution found no route (v1's `no_usable_period` folds in here) | facts layer, surfaced by history | new |
| `partial_leaf_sum` | route 3b refused — a declared leaf was missing | facts layer | new |
| `incomplete_roll_up` | a roll-up leg was missing | facts layer | new |
| `period_intersection_partial` | **B.6.6** — route 3b's strict period intersection dropped this period. 128 rows today, silently | facts layer (`filing_rows`) | new, §5.7 |
| `zero_only_retained` | the only facts available were zero | facts layer | new |
| `regime_break` | a definitional discontinuity (ASC 842, ASU 2016-18, LDTI 2021) | history build | new |
| `basis_ex_iprd` | R&D on the `…ExcludingAcquiredInProcessCost` basis | facts layer | new |
| `failed_hard_guard` | 5b Layer A nulled a physically-impossible value | Phase 5b | reserved |

- [ ] Refusals reach this table from the **history build**, not from `periods.py` — `periods.py`
      already exposes its `refusals` out-parameter (4c.8) and that is the only mechanism. Two
      sources of truth for "why is this absent" is the failure mode the table exists to prevent.
- [ ] Inherit Compustat's caveat as an *expectation*, not a target: only ~1.2% of missing `XRD`
      is coded there. Most blanks stay unexplained even at a vendor — but ours must not, because
      §5.8's gate is zero.

#### 5.4 `fundamentals_employees` — a third table, because the source is text

`employees` is a Tier-2 catalogue field with `source: "text:10-K"` and `annual_only: true`; it was
removed from the XBRL field list in 3c.9, and `fundamentals_employees.py` still exists in the
tree. Decision 35 moves it out of the wide table.

- [ ] `fundamentals_employees (ticker, as_of, employees)`. Keeps a text-parsed number out of an
      otherwise entirely XBRL-sourced table, and a parse failure can no longer fail the history
      build.
- [ ] `employees` **stays in the KPI catalogue** (it has a tier and an authority) but must leave
      `Catalogue.all_column_names`' history contract — a `kpi_catalogue.py` change, with a test.
- [ ] **Phase 6 must repoint `employee_features.py`**, which reads headcount off the
      `fundamentals_history` frame today (`employee_features.py:5,79-93`), and
      `step_cube_fundamentals.py:175`'s docstring. Add to §6.1.
- [ ] Risk zone: `schema.py` + `sql/schema.sql` DDL → propose it with the other two.

#### 5.5 `build_history.py`

New: `src/data_extract/utils/fundamentals/build_history.py`. **Target < 300 lines.**

- [ ] **The API contract already exists in the tree.**
      `tests/data_extract/test_fundamentals_point_in_time.py:108-123` does
      `importorskip("…build_history")` and calls
      `build_ticker_history(ticker, facts) -> DataFrame` with `as_of` and `fiscal_end` columns.
      Honour that signature — the test turns green by *arrival*.
- [ ] Driven by the publication-event ladder: enumerate each ticker's distinct filing dates and
      for each rebuild the full snapshot from facts with `filing_date <= that date`. This is the
      `as_of_cutoff` replay the old `derive_fundamentals_history` already supported, promoted
      from an audit-only debug path to the production loop.
- [ ] **`as_of` is no longer computed.** `_assemble_base`'s median-of-spine heuristic disappears:
      under the event grain `as_of` simply *is* the filing date. Simpler and more correct.
- [ ] The two old leak-guard passes survive as a **redundant** assertion — under this grain they
      can no longer fire, which is exactly what makes them a good test.
- [ ] Reads `fundamentals_facts` **projected and filtered** (`columns=` / `where=`), never
      `SELECT *` per ticker. The replay is O(filings) per ticker, so load the per-ticker facts
      frame **once** and slice it in memory — never re-query per event. (Phase 10 names this
      explicitly.)
- [ ] `step_extract_fundamentals.py:35-38` is the hook point, and `cli.py:148-151` already
      documents the combined `fundamentals` command as deliberately absent until now. Note
      `tests/data_extract/test_step_extract_fundamentals.py` **pins the call order**, so
      inserting the history build changes that test.
- [ ] Regime gating (`researchAndDevelopment`, `currentAssets`, `currentLiabilities`,
      `grossProfit`) is applied **here**, reading `fundamentals_exceptions.json`, and emits
      `not_applicable_for_regime`. The facts layer stays regime-agnostic about *absence*.

#### 5.6 Immutability, and the rebuild path you will actually use

**The store has no append-only primitive.** `store.save` is `INSERT … ON CONFLICT DO UPDATE` on
the registry PK `(ticker, as_of)` (`store.py:533-543`), so "rows are immutable once written" was
asserted in v1 and **unenforceable in the code**. A re-run after a resolution change would
silently overwrite history — precisely the failure §5.0 exists to prevent.

- [ ] **Recompute, diff, raise** (decision 28). Load the ticker's existing history projected,
      rebuild every event in memory, and if any already-stored row would change, **ABORT and
      print the diff** — unless a rebuild flag is passed. Makes immutability provable, gives 5b's
      `pit_leak` check for free, and catches resolution drift the moment it appears. Cost is a
      ~62-row read per ticker per night.
- [ ] Compare values **exactly**, not with a tolerance. If it ever trips on floating-point noise
      we want to know that, rather than to have pre-forgiven it.
- [ ] **Two rebuild flags, ticker-level** (decision 27), matching the two bug classes:
      `--rebuild-history -t APA` deletes that ticker's `fundamentals_history` rows and rebuilds
      from **existing facts** — no network, for a bug in the history layer;
      `--rebuild -t APA` deletes from **both** tables and refetches all ~62 filings — for a bug
      in the resolution layer. `store.delete(table, where)` already exists
      (`store.py:586-603`), and the fetcher's accession-set resume path makes a deleted ticker
      look exactly like a never-fetched one, which is the user's stated mental model.
- [ ] **No `build_version` column.** The rebuild *is* the version: a deleted-and-refetched ticker
      is current by construction, and the diff-and-raise guard is what tells you which tickers to
      delete. Do not resurrect `pipeline_fingerprint`.
- [ ] Idempotency: a second run appends **0** rows and raises nothing.
- [ ] ⚠ **Once Phase 6 trains a cube on this table, a delete-and-rebuild is still allowed but
      stops being free** — it re-derives numbers under a trained model, which is the vintage
      instability the research measured (earnings sign flips ~14% across vintages; ~50% of
      anomalies change inference). Log every rebuild in the phase report from Phase 6 onward.

#### 5.7 The register items, re-scoped

- [ ] **Item 7 — every refused quarter gets a reason code.** `periods.py::_derived` returns
      `None` on a sign-guard or scale-test rejection and the window silently has no row. Route it
      through the `refusals` out-parameter to §5.3's table.
- [ ] **Item 9 + B.6.6, pulled into Phase 5** (decision 26). These were the same item,
      double-assigned: item 9 sat in Phase 5, B.6.6 sequenced it after 5b. Phase 5 **cannot pass
      its own zero-unexplained-nulls gate** while it is open. Fix `filing_rows` so a route-3b
      partial period intersection emits `period_intersection_partial` instead of dropping, then
      re-measure — the estimate was 31 values, the measurement is **128** (EQIX capex 40, EQIX
      depAmort 40, SCHW cash 34, NEE ppeNet 8, VRT depAmort 6).
- [ ] **The worse half of B.6.6, stated because a null-gate can never catch it**: SCHW `cash` is
      an **instant** field, so a dropped period does not produce a null in the snapshot — it
      produces a **stale forward-filled value**, silently. Under §5.0 rule 3 carrying forward is
      correct *behaviour*; what was missing is any record that the field's own newest period was
      refused. Emit the code on the `(ticker, as_of, field)` where the refusal happened, so the
      stale carry is visible.
- [ ] Keep route 3b's **strict** period intersection. The obvious repair reintroduces the mixed
      basis the route exists to prevent. Print BA's fiscal-2011 annual gap as a named cost.
- [ ] **Growth leaves this table** (decision 33). `revenueGrowth` / `earningsGrowth` are computed
      by `pit.py` at cube time on a **fixed 365-day `as_of` offset**. This is the larger fix:
      `infer_yoy_periods` is not used by two columns — it feeds
      `fiscal_change_to_daily(…, periods=yoy_periods)` across many cube features
      (`fundamental_features.py:515-516,653`, `governance_features.py:95`), all of which inherit
      the row-offset bug. Replacing the inference with 365 days fixes **all** of them. → §6.1.

#### 5.8 Verification

- [ ] `test_fundamentals_point_in_time.py` goes **green** — the defining acceptance test of this
      phase, deliberately red since Phase 1. All four of its tests, including the
      `importorskip`-gated unit test.
- [x] **Column count is exactly 69** (68 specified + `fiscal_quarter`), and every column traces to §5.2's table. Asserted by `Catalogue.history_columns`, by `build_ticker`, by `test_build_history` and by gate 2 against the LIVE table.
- [ ] **Amendment round-trip on SMCI and ADM** (the two substantive restaters; SMCI is the 2024
      late-filing/auditor episode, the hardest adversarial case available): the original row
      keeps its as-filed value **unchanged**, a new row exists at the amendment's own filing
      date, and only the amended field plus the TTM columns whose window contains it differ.
      **Print both rows side by side** — this is the phase's sanity-check conclusion.
- [ ] **No-op amendments emit nothing**: the 88 <10-fact amendments produce **0** rows.
- [ ] **The >365-day cutoff fires on the population it should.** Measure how many amendments
      exceed it (v1 measured 26% beyond 90 days and a 921-day max, but never counted beyond 365),
      and re-derive the total row-count delta — v1's +158 predates this rule.
- [ ] `fiscal_end` monotone non-decreasing in `as_of` for every ticker.
- [ ] Same-day collapse: no `(ticker, as_of)` duplicate; report how many pairs collapsed.
- [ ] Idempotency: a second run appends **0** rows.
- [ ] **Zero rows with a NULL value and no reason code** — the `LEFT JOIN` against §5.3. This is
      the gate that items 7 / 9 / B.6.6 exist to make passable, and 5b's `unexplained_null` is
      its standing form.
- [ ] Bank / insurer / REIT top lines non-null for the tickers §2.9 measured as broken (the 11
      banks with NII but no noninterest income, the 6 insurers with premiums but no NII, the 17
      with neither leg).
- [ ] `tests/data_aggregate/test_composites_config.py::real_panel` starts working the moment this
      table is populated — a green-by-arrival dependency nobody will otherwise rediscover.
- [ ] `tests/data_extract/test_amendment_grain.py` — synthetic: an original + a restating
      amendment **400 days later** → under decision 34 that is now **beyond** the cutoff, so
      assert **1** row, not 2. Add a ≤365-day twin that asserts 2. Paired with the real
      SMCI/ADM/JPM check.

#### 5.9 Decisions taken 2026-08-23 (planning interview)

| # | question | decision | rejected, and why |
|---|---|---|---|
| 26 | Phase 5's zero-null gate vs B.6.6's 128 silent nulls, sequenced after 5b | **Pull the `filing_rows` reason-coding fix into Phase 5** and re-measure | a named-exceptions list — that is the mechanism by which the previous stack accumulated silent holes; and demoting the gate to 5b, which would publish append-only rows before anything was measured |
| 27 | rebuild granularity | **Two ticker-level flags**: `--rebuild-history` (facts kept, no network) and `--rebuild` (both tables, refetch) | one flag always refetching — every history-layer fix would cost a full re-download; and period-level granularity — deleting one quarter invalidates every later snapshot, so you must compute the dependency closure |
| 28 | how immutability is enforced given an upsert-only store | **Recompute all, diff against stored, raise on any change** unless a rebuild flag is passed | build-only-newer-events — drift becomes undetectable; and an `ON CONFLICT DO NOTHING` primitive — immutability without detection, which hides drift just as well as overwriting it |
| 29 | reason-code row population | **Dense** — one row per null/qualified cell at every event | sparse state-change — contradicts §5.0 rule 3 and turns the null gate into a window function; scored-fields-only — a tier-0 input going dark is what corrupts `totalDebt` and `cash` |
| 30 | `totalLiabilities` — why deduce it at all? *(user challenge)* | **A ladder: linkbase leg-sum first, identity + NCI bridge as fallback**, measured before written | leg-sum only — route 3b's strict intersection on a long leg-set may cost more than the identity; identity only, as v1 specified — it never reads the filer's own liability captions, which Reg S-X *does* prescribe |
| 31 | column basis and naming | **TTM for flows, bare names, `revenue_q` + `netIncome_q` only** | `X` + `X_q` for every flow (~108 columns); a long side table for discrete quarters — a third table for something no consumer has asked for. **`ebitda_q` / `freeCashflow_q` die** |
| 32 | `sector` / `industry_group` | **Dropped** — join from `sp500_tickers` | keeping them; note the residual: `regime` stays and carries the same non-vintaged-roster look-ahead |
| 33 | who computes growth | **`pit.py`, at cube time, offset fixed to 365 days** | Phase 5 computing it — which would fix 2 columns and leave `infer_yoy_periods`' row-offset bug in every other cube growth feature |
| 34 | a >1-year-late amendment changes nothing consumable *(user framing)* | **≤365 days late ⇒ emit a row and let the TTM rebuild at that `as_of`; >365 days ⇒ no row.** A stock has long since passed a 2-year-old restatement, so there is nothing for a long/short model to learn from it | emitting always — pays row count for an unlearnable event; emitting never — loses the restatement-as-signal case entirely. Note the cutoff is a clean proxy: a quarter stays inside some live TTM window for ~12 months, so ≤365 days ≈ "still changes a published value" |
| 35 | `employees` (text-sourced, annual-only) | **Its own table** `fundamentals_employees` | in the wide table — a text parse failure would fail the history build; NULL-and-deferred — a Tier-2 field ships empty |
| 36 | do 10-K filings feed the history build? *(clarification)* | **Yes.** 10-K filing dates are publication events and the FY fact is the primary Q4 input | quarterly-only — deletes the `FY − YTD9` ladder and reverts Q4 to the label Phase 4 measured as tautological on 100% of rows |
| 37 | same-day collapse provenance | **Scalar with a precedence rule** | a pipe-joined `publication_form` — silent form-filter bugs; `(ticker, as_of, accession)` PK — breaks the `PitFrames` pivot |
| 38 | the 68-vs-71 column contradiction | **68, enumerated at §5.2.** Both v1 figures now reconcile: 68 = 53 catalogue + 10 derived + 5 legacy keys; "~71" was that plus `regime` plus a provenance count that was never finished (it is 4, so 73) | leaving it as "~71" — *"column count is exactly as contracted"* is a verification item and there was no contract |
| 39 | build scope | **The 52 roster tickers**; full universe stays at Phase 9.3 | full universe now — 33-83h single-threaded spent *before* the validator exists, so a bug found at 5b costs a 500-ticker rebuild instead of a 52-ticker one |

**Estimated effort**: **4-5 days** (raised from 2.5-3). §5.1's measure-first pass and its re-sweep
are ~1 day on their own and touch the facts layer; B.6.6 adds ~0.5; the third table, the
diff-and-raise guard and the two rebuild flags add the rest.

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
- [ ] `Tables.fundamentals_quality` in `schema.py` and `sql/schema.sql`. Risk zone → propose the
      DDL. (`fundamentals_reason_codes` and `fundamentals_employees` are **Phase 5's** DDL now —
      §5.3, §5.4 — so 5b only adds the one table.)

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
- [ ] **`infer_yoy_periods` breaks on the amendment grain — and it is bigger than two columns.**
      `pit.py:93-106` infers "rows per year" from the pooled median `as_of` gap (→ 4) and growth
      is `pct_change(N)`, a **row** offset. A ticker with an amendment has 5 rows that year, so 4
      rows back is ~9 months — seasonality contamination in exactly the tickers this rebuild
      handles better. Fix: a **365-day `asof` offset**, already declared that way in the KPI JSON
      so Phase 5/6 cannot forget. **Verified 2026-08-23**: the function is *not* used by
      `revenueGrowth`/`earningsGrowth` alone — it feeds
      `fiscal_change_to_daily(…, periods=yoy_periods)` across many cube features
      (`fundamental_features.py:515-516,653,923`, `governance_features.py:95`), all of which
      inherit the bug. Fixing it here fixes all of them.
- [ ] **Phase 6 now OWNS `revenueGrowth` and `earningsGrowth`** (Phase 5 decision 33). They leave
      the wide table and are computed at cube time on the fixed 365-day offset. Prove
      `infer_yoy_periods` has no remaining consumer before deleting it.
- [ ] **Four new casualties from Phase 5's decisions, to be reconciled in the §6.1 table**:
      `sector` + `industry_group` leave `fundamentals_history` (decision 32 — 9 `data_aggregate`
      modules read them off this frame today; join from `sp500_tickers`, and fix
      `constants.py:516`); `employees` moves to its own table (decision 35 — repoint
      `employee_features.py:5,79-93` and `step_cube_fundamentals.py:175`); `ebitda_q` and
      `freeCashflow_q` are dropped (decision 31). The 239 → 68 arithmetic is unchanged; its
      **composition** is not.
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
- [ ] `tests/data_extract/test_amendment_grain.py` — the §5.0 acceptance test. ⚠ **Phase 5
      decision 34 changed this fixture**: an amendment 400 days later is now **beyond** the
      365-day cutoff, so the assertion is **1** row, not 2. Add a ≤365-day twin asserting 2 rows,
      the original unchanged, and `fiscal_end` monotone. Paired with the real SMCI/ADM/JPM check.
- [ ] `tests/data_extract/test_statement_role_routes.py` — 4c.1's three-way synthetic
      (note-only / both / undeclared) plus the five real ground-truth instances.
- [ ] Docs, moving with the code: `docs/data_schema.md` (the "239 columns" line becomes **68**
      the moment Phase 5 lands, and there are now **five** fundamentals tables — `facts`,
      `history`, `reason_codes`, `employees`, `quality`; **also rewrite the grain sentence** —
      that `as_of` is a *publication event*, that amendments append rather than overwrite, and
      that an amendment **>365 days late emits no row**, is the table's whole contract with the
      modelling layer), `docs/data_sources.md`, `docs/architecture.md` (the `validate/` row),
      `docs/config.md` (now **four** fundamentals JSONs), `docs/database.md`, `docs/runbook.md`
      (the two rebuild flags of §5.6 — this is the command the user will actually run between
      sessions), `AGENTS.md` (replace one line, cap 70), `README.md` if the CLI surface moved.

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
   resolution fix needs an explicit rebuild, because re-deriving history under an already-trained
   model is exactly the vintage instability the research measured. *Mitigation*: Phase 4c exists
   and comes first; §5.6's diff-and-raise guard makes drift **loud** rather than silent (the
   store is upsert-only, so without it immutability was unenforceable); the two ticker-level
   rebuild flags make a fix cheap **while Phase 6 has not yet trained a cube** — which is
   precisely why the build scope is 52 tickers, not 500 (decision 39).
3b. **§5.1 reopens the facts layer that Phase 4c just closed.** The `totalLiabilities` leg-sum is
   a new `roll_up.any_of` on a field that has never had one, and route 3b's strict period
   intersection may cost more coverage than the identity it replaces. *Mitigation*: measure
   before writing (never from the Reg S-X caption list — §B.5 refuted that reasoning twice),
   report the refusal rate, and **let the identity win on the merits if it is better**. The
   required re-sweep is paired with B.6.8's so it is spent once.
3c. **Dropping `sector`/`industry_group` grows Phase 6** — 9 `data_aggregate` modules read them
   off this frame. *Mitigation*: the §6.0 grep-guard test already covers exactly this miss class;
   add both names to it explicitly.
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
- [x] Phase 5: `fundamentals_history` is **exactly 69 columns** (§5.2 + `fiscal_quarter`), every one traced;
      `totalLiabilities` resolves for the 11 zero-coverage tickers on the **filer's own liability
      legs** where they are declared, and the identity fallback is stamped `derived_identity`
      wherever it is used; **zero** rows carry a NULL value with no reason code; a re-run raises
      nothing and appends nothing.
- [ ] `fundamentals_facts` + `fundamentals_history` + `fundamentals_reason_codes` +
      `fundamentals_employees` + `fundamentals_quality` build end to end for **both** 26-ticker
      rosters.
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
| **5** History layer | **4-5 d** + 1 re-sweep | raised from 2.5-3 by the 2026-08-23 interview: §5.1's measure-first `totalLiabilities` pass touches the **facts** layer (~1 d + the re-sweep, paired with B.6.8's), B.6.6 pulled forward (+0.5 d), plus a third table, the diff-and-raise guard and two rebuild flags |
| **5b** Validator toolkit | **4-5 d** | absorbs v1 Phase 7 (2-3 d) + ~15 scratchpad scripts + Tier 3 |
| 6 Downstream repair | 4-5 d | unchanged |
| 8 Tests + docs | 1.5-2 d | unchanged |
| 9 Acceptance, both rosters | **2 d** + wall-clock | raised from 1 d |
| 10 EFFICIENCY review | 0.5 d | unchanged |
| **Remaining total** | **19.5-23 days** + backfill wall-clock | |

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

**Decisions 26-39 are the Phase 5 planning interview** (same date) and live at **§5.9**, beside
the phase they govern. Three of them are user challenges that changed the design rather than
merely settling it: `totalLiabilities` reads the filer's own liability legs before falling back to
an identity; a >365-day-late amendment emits no row; and the rebuild model is delete-and-refetch,
not a versioned overwrite.

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
| ~~B.6.6~~ | ~~the 128 rows 4c.1 costs, with **no `dc_code`**~~ **→ MOVED TO PHASE 5 (decision 26)** | EQIX capex 40, EQIX depAmort 40, SCHW cash 34, NEE ppeNet 8, VRT depAmort 6. Cause: `filing_rows` only reason-codes a field with **no periods at all**, so a route-3b partial period intersection drops silently. This is the plan's own **item 9** -- estimated at 31, actually 128 | **The deferral was wrong.** Phase 5's gate is *zero rows with a NULL value and no reason code*, so it cannot pass while this is open. Worse: SCHW `cash` is an **instant** field, so the drop produces a **stale forward-filled value**, not a null — which no null-gate can ever catch. Now §5.7 |
| B.6.7 | 538 duplicate-fact disagreements (4c.3 ledger) | worst BA `operatingCashFlow` 45%, WMT `shortTermDebt` 10.7%. The `decimals` tiebreak now resolves them deterministically and records both sides | the ledger exists so 5b can threshold it. Which disagreements are *material* is a calibration question, not a resolution one |
| B.6.8 | both-roster re-sweep after the 4c.2 / 4c.4 config change (Risk 1) | the 4c.2 change is validated by the two new standing tests; the ledgers in `data/fundamentals_sweep/` were built pre-change and are now stale for `longTermDebt` only | a full re-sweep is ~2h of network and 14.7 GB RSS. Better spent once than twice — **and §5.1's `totalLiabilities` leg-sum now needs one too, so PAIR THEM**: one sweep covers 4c.2, 4c.4 and 5.1 |

---

## Implementation log — Phase 5 (2026-08-23/24)

**Execution order actually used**: the code-only register items first (7, 9/B.6.6), then the
vocabulary and the column contract, then `build_history.py`, then the risk-zone batch (approved
2026-08-23), then §5.1's measurement, then the network backfill. §5.1 came *after* the code
because its own instruction is measure-first and its measurement needed no history rows.

### New committed instruments

| file | what it is |
|---|---|
| `src/data_extract/utils/fundamentals/reason_codes.py` | the `dc_code` vocabulary, in ONE place for the first time. Existing codes are **imported** from `periods.py` / `xbrl_linkbase.py` rather than restated, so each still has exactly one definition; the history-layer codes are declared here. `ALL_CODES` (18) is asserted against every row the build writes, and `IS_QUALIFIER` (5) separates "the value is present but off-basis" from "the cell is null" — a distinction the null-gate needs and a single flat list cannot express. |
| `src/data_extract/utils/fundamentals/build_history.py` | the publication-event replay. `build_ticker_history(ticker, facts)` (the signature `test_fundamentals_point_in_time.py` has pinned since Phase 1) and `build_ticker(...) -> TickerHistory(history, reason_codes)`; `carry_latest_known` (the second pinned API, from `test_fundamentals_employees.py`); `publication_events`; `diff_against_stored`; `build_fundamentals_history(context, tickers, *, rebuild_history=False)`. |
| `scripts/measure_total_liabilities_legs.py` | §5.1's measurement as a command: the calculation linkbase under `LiabilitiesAndStockholdersEquity` per filing → declared `Liabilities` total?, leg-set, route-3b refusal rate. Documented in the runbook. |
| `tests/data_extract/test_build_history.py` | **16** tests: the 69-column enumeration and the statement ORDER, `fiscal_quarter` on every row (incl. a September filer, where a calendar-month rule labels December Q4 instead of Q1), code-vs-config formula cross-check, the closed vocabulary, the grain, same-day collapse, no-op amendments, the append-only diff (including the `datetime.date` round-trip), zero-unexplained-nulls on real AAPL facts, and the `totalLiabilities` identity on APA + WMT plus both NCI branches. |
| `tests/data_extract/test_amendment_grain.py` | 3 tests: the ≤365-day round-trip printed side by side, the >365-day cutoff with its exact boundary (365 admits, 366 refuses), and value-test-not-fact-count. |

### §5.2 — the column contract, reconciled to 68 and enumerated (**69 as delivered**)

`Catalogue.history_columns` IS the contract now, and `build_ticker` asserts its length. The
plan's own arithmetic reconciles exactly: **3** keys + **52** catalogue fields (53 less
`employees`) + **8** derived + `regime` + **4** provenance = **68**. `CUBE_TIME_COLUMNS`
subtracts `revenueGrowth`/`earningsGrowth` in code as well as in the config, so the contract
cannot regrow by a config edit alone.

- [x] One column per field, bare name, TTM for flows. `_FORMULAS` is the single implementation
      of all ten computed columns and a test parses the config's own `derived_from` prose back
      out and compares — the config stays the contract, the code stays the implementation.
- [x] `ebitda_q` / `freeCashflow_q` / `capexGlobal` dropped; `sector` / `industry_group` dropped
      (decision 32). The `regime` residual is stated in the DDL comment, not hidden.
- [x] `employees` left `history_columns` via a `FieldSpec.is_text_sourced` property (`source`
      starts with `text`) rather than a hardcoded name, so the rule is the reason.

### §5.5 / §5.6 — what the replay actually does, and the one performance surprise

- [x] `as_of` is the filing date; `_assemble_base`'s median-of-spine heuristic is gone with the
      old builder. **`fiscal_end` is capped at `as_of`** — a definition the plan did not state
      and the pinned test forced: ROP's fixture files 2020-12-31 numbers on 2020-11-02, and
      without the cap `fiscal_end > as_of` and the look-ahead assertion fires. Capped, the row
      reports the newest period it actually could (Q3), and monotonicity is structural.
- [x] **Instants align on `as_of`, not on `fiscal_end`.** The cover-page `dei` share count is
      dated at the filing, *after* the period it accompanies, and it is the only summable count
      for a multi-class issuer — capping instants at `fiscal_end` would delete
      `sharesOutstanding` for the current period on every filer.
- [x] A refused TTM stays NULL with its code and is **not** forward-filled; an instant IS
      carried forward, because that is what "latest known value" means for a level. The two
      halves of §5.0 rule 3, and they are not the same rule.
- [x] `diff_against_stored` + `--rebuild-history` / `--rebuild` (decisions 27, 28). Compared
      exactly, no tolerance. The `datetime.date` round-trip is handled and tested — a DATE
      column returns as `datetime.date`, which never equals a `Timestamp`, so an unguarded
      comparison would report every stored row as drifted.
- [x] `build_ticker` pins the date and text dtypes. An all-NaT `amended_fiscal_end` infers
      `datetime64[s]` while a populated one is `[us]`, so two builds of one ticker compared
      unequal on dtype alone — and forgiving a dtype is one step from forgiving a value.
- [x] **Every value column is cast to `float64`, and the guard immediately earned its keep.**
      `sql/schema.sql` is applied only when Postgres INITIALISES a volume, so on this
      long-lived one `store.save` created the table from the FIRST frame via `ensure_table`'s
      dtype inference. VRT resolves neither `minorityInterest` nor `restrictedCash`, so both
      became **TEXT**, and APA's real numbers then came back as the string `'1997000000.0'`.
      `diff_against_stored` flagged 116 cells across 51 rows on the very next run — a genuine
      corruption reported as drift, which is exactly what decision 28 was for. Fixed on both
      sides: the frame now declares `float64` for all 60 value columns even when a ticker
      populates none of them, and the three tables were dropped and recreated from the
      committed DDL. A named test asserts the dtypes. **Worth carrying forward: on an existing
      volume, `sql/schema.sql` is documentation until someone applies it.**
- [x] **⚠ PERFORMANCE was the real risk, and it was not in the plan.** The first working replay
      took **~14 minutes per ticker** (≈12 h for 52). Profiled, the cost was not the algorithm:
      `_drop_annual_masquerading_as_quarter` filtered three DataFrames *per quarterly row*, and
      every filter inside the period engine fancy-took all 27 columns — half of them
      Arrow-backed strings, at 98,646 `pyarrow.compute.take` calls per `build_periods`. Three
      changes, none of which touch the arithmetic: a `PERIOD_COLUMNS` projection with the string
      columns off Arrow; a positional `iloc[:n]` slice off a filing-date-sorted frame instead of
      a boolean mask per event; and that function rewritten over numpy views. **14 min → 2.4 min
      per ticker**, with `test_periods_q4.py` green throughout. Phase 10 should still look at it.

### §5.1 — `totalLiabilities`: the leg-sum is the filer's own evidence, and route 3b still cannot sum it

Measured on the 11 zero-coverage tickers × 4 10-Ks each (44 filings):

| finding | number |
|---|---|
| filings declaring a `Liabilities` TOTAL under the balance-sheet root | **0 of 44** |
| filings declaring a liability LEG-SET | **44 of 44** |
| raw route-3b refusal rate (strict period intersection) | **68% (30 of 44)** |
| …caused by a single element, `us-gaap:CommitmentsAndContingencies` | Reg S-X 5-02(24), declared under the root as a footnote POINTER, never given a value; appears in 8 of 11 tickers |
| refusal rate with that element excluded | **4.5% (2 of 44)** — DUK FY2023 `NotesPayableRelatedPartiesNoncurrent`, MCD FY2024 `OperatingLeaseLiabilityNoncurrent`, both genuine partials |
| leg-set size, and its stability | 2 (ETN) to 7 (MCD); **EOG, TMO and MCD each CHANGE theirs between 2023 and 2024** |

So decision 30's premise is confirmed — the legs really are the filer's own evidence and the
`Liabilities` concept is absent, not merely unlucky. **But route 3b cannot safely sum them**, and
this is what decided it: `_leaf_sum` admits only concepts the catalogue ENUMERATES in
`roll_up.any_of`; an unlisted sibling is refused when it is a company extension and **silently
dropped when it is us-gaap**. Because the leg-sets vary by filer *and* by year, any enumeration is
a union that cannot be shown complete — and an incomplete union yields a Tier-1 balance-sheet
total short by a caption, which looks entirely plausible and passes every level check. That is the
`shortTermDebt` defect this rebuild exists to remove, re-created on a bigger field, and §B.5
refuted "element names are evidence of what a filer declares" twice already.

- [x] The identity wins on the merits, exactly as the plan's own instruction allowed for
      (*"if it is worse than the identity's coverage, the identity wins"*).
- [x] **Two deviations from §5.1, both deliberate.** (a) It is computed in the HISTORY layer, not
      the facts layer: `fundamentals_facts` is documented as strictly as-filed — "every row
      carries a number the filer actually tagged" — and `resolution_method = 'derived_identity'`
      would contradict the property the publication-event grain rests on. (b) The stamp is
      therefore `dc_code = derived_identity` in `fundamentals_reason_codes`, a QUALIFIER, so the
      cell never reads as resolved evidence and 5b's `cross_identity` can treat it as an input.
- [x] **The NCI bridge reads the ROW, not the priority list** — and that mattered: both APA and
      WMT resolve equity on `StockholdersEquityIncludingPortionAttributableToNoncontrolling-`
      `Interest`, so adding `minorityInterest` would have *understated* liabilities by the whole
      of it. An ex-NCI basis with no `minorityInterest` resolved is REFUSED rather than
      overstated. Both branches have a named test.
- [x] Measured effect: APA **0 → 70 of 70** rows, WMT **0 → 69 of 69**, every one stamped.
      WMT's latest foots to the filed balance sheet ($289,607M − $100,682M = $188,925M).
- [ ] No `configs/` change was made for this field. A `derived_fallback` documentation key on
      `totalLiabilities`, carrying the measurement above as its evidence, is the right home for
      the decision and is **proposed for the next batched config approval** — outside the scope
      the user approved (which was the `_cube_time_columns` move only).

### §5.4 — `fundamentals_employees`, with a producer

- [x] The table, and `employees` out of `history_columns` with a test.
- [x] **Wired, not just declared.** `fundamentals_employees.py` existed in the tree with no
      caller, so the table would have shipped empty. `build_ticker_fundamentals` now parses the
      headcount out of every 10-K/10-K-A it already has open and returns a second frame; the
      continuity guard is seeded from the stored table and grows through the walk, so each 10-K
      is judged against every earlier one exactly as a full-history pass would judge it.
- [x] Verified live: AAPL 166,000 @2025-10-31, XOM 58,000, EOG 3,400, APA 1,791, and over the
      full backfill MSFT 94,000 -> 228,000, ORCL 115,000 -> 164,000, DE 56,800 -> 83,000 across
      15 years each — all correct against the filings.
- [ ] **⚠ 19 of 54 tickers currently have NO headcount rows, and the cause is mine, not the
      code's.** Recreating the three tables from the committed DDL (see the dtype item above)
      happened at 00:49, *after* backfill chunks 1-3 had already written their 81 + 78 + 76
      employee rows — so the drop took them with it. The facts walk is accession-resumed, so a
      plain re-run parses nothing: those tickers need `fundamentals --rebuild -t <19>`, which
      is ~30 min of network and is exactly what that flag exists for. Deferred until the
      history replay finishes rather than raced against it. The 35 tickers fetched after the
      recreation carry 474 rows and are correct.

### §5.7 — the register items

- [x] **Item 7.** `_derived` records every refusal through the `refusals` out-parameter with the
      window it would have produced and the value it refused. Two new codes were needed and the
      reason is written down: `insufficient_quarters` means the window was not there, and these
      mean it WAS and the arithmetic was refused. A non-additive scale refusal keeps the older,
      more specific `split_basis_mismatch`. Fires on AAPL: 102 `split_basis_mismatch`,
      1 `derived_basis_mismatch`.
- [x] **Item 9 / B.6.6.** `_materialise` now RETURNS the periods the strict intersection drops
      and `rows_from_xbrl` emits a value-less row per dropped period carrying
      `period_intersection_partial` — for every field, including the ones that resolved, which
      is the whole of the item. Disjointness from the resolved periods is asserted rather than
      assumed. `_row` gained an explicit `dc_code` because a resolution that succeeded has no
      code of its own while the PERIOD does; `decimals` no longer stringifies a NaN into `"nan"`.
- [x] `basis_ex_iprd` now actually fires: `dc_code_on_fallback` was declared in the config and
      read by nothing. It rides a new `Resolution.basis_qualifier` into the `adjustment` JSON
      rather than `dc_code`, because `Resolution.resolved` is defined as "no dc_code" and a
      qualifier on that column would have turned every ex-IPR&D row into an absence.
- [x] Route 3b's strict period intersection kept strict, as instructed.
- [x] Growth left the table; the `infer_yoy_periods` repair stays Phase 6 §6.1.

### §5.8 — verification status

| gate | result |
|---|---|
| `test_fundamentals_point_in_time.py` unit test | **GREEN** — red since Phase 1. Both the normal and the ROP early-filing fixture yield lag ≥ 0 |
| `test_fundamentals_employees.py::carried_forward` | **GREEN** by arrival (`carry_latest_known`) |
| column count exactly 69, every column traceable, in STATEMENT order | **PASS**, enumerated by family in the test's own output, and gate 2 checks the stored table's columns *in order* |
| amendment round-trip, ≤365 days | **PASS** — original row byte-identical, TTM 1000 → 1050 only at the amendment's own `as_of`, `fiscal_end` stays 2023-12-31 while `amended_fiscal_end` carries 2023-03-31 |
| no-op amendments emit nothing | **PASS** — a 4-fact amendment re-tagging identical values → 0 rows; a 1-fact 0.5% restatement → 1 row |
| the >365-day cutoff | **PASS**, and the boundary is exact: +365 d admits, +366 d refuses |
| `fiscal_end` monotone non-decreasing; `as_of ≥ fiscal_end` | **PASS**, structural (asserted in `_assert_grain` on every build) |
| same-day collapse, no `(ticker, as_of)` duplicate | **PASS** — 5 accessions on 4 dates → 4 events, `publication_form` = `10-K` by precedence |
| idempotency / append-only | **PASS** — `diff_against_stored(h, h)` empty; one tampered cell → exactly 1 finding |
| **zero rows with a NULL value and no reason code** | **PASS on real AAPL facts**: 69 events × 60 value columns = 4,140 cells, 924 nulls (22.3%), 1,035 reason-code rows, **0 unexplained** |
| AAPL value spot-check | FY2025 revenue **$416.2bn**, total assets **$359.2bn**, `epsDiluted` **7.46** — all match the filed 10-K |
| filing lag `as_of − fiscal_end` (AAPL, 69 events) | median **32 d**, min 25, max 37, **0 rows beyond 200 d**. Was: median 401 d for ATO and lags out to 1,884 d |
| `tests/data_store` + `tests/data_aggregate` | 227 passed / 11 skipped / **9 failed — all 9 fail identically on a stashed tree**, i.e. they pre-date Phase 5 and are Phase 6's backlog (the cube reads a `fundamentals_history` the rebuild emptied) |
| `test_periods_q4.py` offline subset | 23 passed, unchanged through the numpy rewrite |
| `test_step_extract_fundamentals.py` | updated and green: the history replay is pinned IMMEDIATELY after the facts walk, and the adjacency is now the property under test |

### Also landed, because the backfill could not run without it

- [x] **`-F/--full` on `fundamentals-facts` / `fundamentals`.** The run manifest's incremental
      test is "did the ticker universe change size since the last run?", so two consecutive
      6-ticker chunks read as a repeat of one run and the second gets `since = last run`.
      Measured: the first chunked backfill wrote 31,540 rows on chunk 1 and **0** on chunks 2-9.
      Chunking is mandatory (edgartools never releases its per-filing caches; the sweep reached
      14.7 GB RSS on an all-52 process), so the flag is the fix, not a convenience. `-F/--full`
      is already the convention in `data_aggregate`'s CLI. Documented in the runbook.

### Was open in Phase 5 — every item CLOSED 2026-08-24

| item | outcome |
|---|---|
| the 52-ticker backfill | **DONE.** 54 tickers, **317,036 facts** in 9 chunks of 6 with `-F` (~7 min/chunk), then a **4h39m** history replay (6.4 min/ticker measured, not the 2.5 estimated from AAPL -- multi-CIK filers and the banks are far heavier) |
| SMCI / ADM amendment round-trip on REAL filings | **DONE, and each for a different reason.** SMCI 2019-05-17: three 10-Q/A at **737-921 days** past their originals, refused by the 365-day cutoff, so the row is the 10-K only. ADM 2024-11-18: three amendments all **inside** the window (251d / 202d / 111d) but they moved **0** of the 52 fields -- ADM's Nov-2024 restatement was of *intersegment* disclosures, so consolidated totals genuinely did not change, and rule 2 dropped them as no-ops. A value-MOVING amendment printed side by side: SPG 2016-01-13 restating `fiscal_end` 2015-09-30, netIncome **1,949M -> 2,155M** (+10.6%), assets +207M, equity +207M, NCI +30M -- while the 2015-11-04 row keeps its as-filed numbers, which IS the no-leakage property |
| the >365-day population count, and the row-count delta vs v1's +158 | **DONE.** 36 amendment accessions (1.09% of filings), **3 of 36 (8%) beyond 365 days**; lag mean 124d, max 921d. Row delta **+21 on 3,246 original rows = +0.65%**, against v1's predicted +0.6% on 27,602 -- and v1 predicted that BEFORE the cutoff existed |
| bank / insurer / REIT top lines non-null | **DONE, 100% in ALL EIGHT regimes** for revenue, equity, assets AND liabilities. `totalLiabilities` was the last hold-out (energy 75%, industrial 93%) and the NCI ladder below closed it |
| same-day-collapse fire rate on real filings | **DONE.** 9 of 3,273 `(ticker, date)` pairs carry >1 accession (0.27%), max **4** on one day (ADM 2024-11-18, SMCI 2019-05-17 and 2019-12-19). The precedence rule is load-bearing, not theoretical |
| `totalLiabilities` `derived_fallback` config key | **NOT NEEDED, dropped.** The fallback is documented where it is implemented (`_total_liabilities_identity`) and, more usefully, is now *visible in the data*: `derived_identity` vs `derived_identity_nci_assumed_zero` in `fundamentals_reason_codes` says per row which basis was used. A config key would have restated it without making it queryable |

### Contract changes after the first backfill (2026-08-24)

Three defects and one request, all applied together so ONE rebuild covers them.

**1. `fundamentals_facts` had the wrong primary key** — the phase's most consequential finding.
The PK was `(ticker, accession_number, field, duration_type, fiscal_year, fiscal_period)`, i.e.
keyed on the fiscal LABELS. A single filing legitimately carries the same label pair more than
once: AAPL's FY2025 10-K reports FY2023, FY2024 **and** FY2025 annual revenue, all tagged
`fp='FY'` for their own year. Measured over the 294,898-row backfill:

| | |
|---|---|
| rows silently lost to the PK | **18,604 of 337,190 (5.5%)** |
| collisions carrying **2+ different values** | **16,340** |
| of those, ANNUAL facts | **1,522** |
| rows lost under the new PK | **3 (0.001%)** |

The visible symptom was `totalRevenue` NULL for 6 of 29 industrials, and AAPL's 2024-11-01 row
reading 385,603 where FY2024 was 391,035 — the previous year's number surviving the dedup. The
PK is now `(ticker, accession_number, field, duration_type, period_end)`: `period_end` IS the
measurement's identity, the fiscal labels are payload. `fetch_fundamentals_sec._period_end`
guarantees the column is never NULL (a reason-coded row falls back to `period_of_report`, then
to `filing_date`).

**2. `build_history._VALUE_KEY` inherited the same defect.** Amendment detection compared
as-filed values on `(field, fiscal_year, fiscal_period, duration_type)` and
`drop_duplicates(keep='last')` on it, so the same collision collapsed three measurements to one
and would report an amendment as a no-op or the reverse depending on which row survived. Now
keyed on `(field, duration_type, period_end)`, matching the fact identity.

**3. `fiscal_quarter`, requested: "add in fundamentals_history if this is Q1,Q2,Q3,Q4 even if
this is in TTM".** Q1-Q4 on EVERY row, including the TTM and instant ones — a TTM spans four
quarters but the row still reports *as of* one of them, and a filer's Q4 is not its Q1.
Implemented as `periods.fiscal_quarter_of_end`, the mirror of `label_fiscal_periods` for a
caller that has only an end date, sharing `_fiscal_bounds` with it so the two cannot disagree.
Labelled off the filer's OWN year ends **as visible at that event**, not a global calendar, so
it stays point-in-time and works for the 52/53-week and non-December filers. Spot-checked
against AAPL's real year ends: 2023-12-30 → Q1, 2024-03-30 → Q2, 2024-06-29 → Q3,
2024-09-28 → Q4. Nullable `Int64` → `BIGINT`, so a `WHERE fiscal_quarter = 3` is not a float
comparison and a pre-first-10-K event stays NULL rather than becoming 0.

**4. Column ORDER, requested: "revenue → operating revenue / cost → net revenue then debt /
assets, the shares (concepts ordered)".** The table was in the tier-then-name order the fields
are RESOLVED in, which reads as noise: `basicShares` first, `totalRevenue` twenty-four columns
later, `costOfRevenue` in a different tier from the revenue it is subtracted from. The 60 value
columns are now `HISTORY_STATEMENT_ORDER` — revenue (general top line, then the regime-specific
ones that replace it) → cost of sales → gross → operating expense → operating result → below
the line → bottom line → the two single-quarter slices → cash flow → assets → liabilities and
debt → equity → share counts, with each ratio immediately after the line it is computed from so
a reader can check it in place. Declared, not derived; `history_columns` asserts the list
against the catalogue, so a new field in the JSON fails loudly rather than being appended to the
end of the table. **Resolution order is unchanged** — the field loop still walks
`history_fields` tier-first, which the NCI bridge depends on.

The contract is therefore **69** columns, not 68: 4 keys + 52 catalogue fields + 8 derived +
`regime` + 4 provenance.

**`scripts/recreate_fundamentals_tables.py`**, also landed because the rebuild could not run
without it. `sql/schema.sql` is applied only when Postgres INITIALISES a volume; on a live one
`store.save` creates a missing table from `ensure_table`'s dtype inference over the first frame
it is handed — which is how an all-None column became TEXT and every later ticker's number was
stored as `'1997000000.0'`. A PK or column-contract change has to be applied deliberately, and
this is the deliberate application: `--dry-run` prints the row counts and the exact SQL, `--yes`
executes it.

### `not_disclosed` anatomy, and a register change PROPOSED then WITHDRAWN (2026-08-24)

Question from the user: *"Do I get the reason why value is missing in fundamentals_facts? How
can I know it is missing for a good reason beside the generic rules in the jsons in config?"*

**What is stored.** `fundamentals_facts` carries the reason on a value-less stub row, one per
(field, filing): `dc_code`, plus `resolution_method='unresolved'`. But every column that would
constitute EVIDENCE is NULL on that row -- `source_concept`, `roll_up_children`, `root_anchor`,
`role_uri`. A resolved row carries all four. So an absence is recorded as a VERDICT with no
supporting evidence, and `not_disclosed` means "the resolver walked this filing's calculation
linkbase and found no concept it recognises" -- a statement about our concept MAP, not about the
filing. It cannot distinguish "the filer has no such line" from "the filer tagged it under a
name we do not know", and the register JSONs cannot settle it either: `expected_absent` is an
asserted rule, not a filing.

**Anatomy of AAPL's 837 `not_disclosed` codes**, measured:

| population | codes | verified against |
|---|---|---|
| lines Apple structurally does not have (8 fields x 69 events) | 552 | bank/insurer/REIT top lines + NCI |
| early-history absence | 133 | `totalDebt` NULL for exactly 16 events, last null **2013-04-24**, first non-null **2013-07-24** -- Apple's first bond issue was **30 April 2013** |
| the line did not exist under the standard yet | 130 | `operatingLeaseLiability` first appears **2020-01-29**, Apple's ASC 842 adoption; `restrictedCash` 2021-01-28 |
| `debtToEquity`, derived, null because its input is | 16 | by construction |
| **mis-coded** | **5** | valued facts present, TTM unassemblable -> fixed, see below |

**Fix applied (code only).** `build_history._has_valued_fact` + a new fallback: when the filer
DID tag a number and the trailing-twelve window still could not be assembled, the code is now
`insufficient_quarters`, not `not_disclosed` -- because the filer did disclose it. AAPL:
`not_disclosed` 837 -> 832, `insufficient_quarters` 79 -> 84, nulls unchanged at 924, still
**0 unexplained**. Both real-filer tests re-run green.

**Register change proposed, approved, then WITHDRAWN on measurement.** The proposal was to mark
the 7 cross-regime top lines + `minorityInterest` `expected_absent` for `industrial`, relabelling
552 of AAPL's codes `not_applicable_for_regime`. `scripts/audit_absence_evidence.py` (new) refuted
it: across the 27 industrial filers only `noninterestIncome` is 0/27. The others are MIXED and
the values are REAL, filer-tagged and material:

| field | industrial filers | concept the FILER used | median |
|---|---|---|---|
| `premiumsEarned` | UNH, CVS, DE (3/27) | `us-gaap:PremiumsEarnedNet` | $72bn / $34bn / $248M |
| `rentalIncome` | AMT, CAT, BA, CSCO (4/27) | `us-gaap:OperatingLeaseLeaseIncome` | $3.5bn / $549M / ... |
| `netInterestIncome` | WMT, CVS, ETN (3/27) | `us-gaap:InterestIncomeExpenseNet` | -$990M / -$270M / -$72M |
| `netInvestmentIncome` | CVS (1/27) | `us-gaap:NetInvestmentIncome` | $518M |

UnitedHealth earns premiums, American Tower earns rent, Cat Financial leases equipment; a GICS
sub-industry of `industrial` does not remove those lines from their income statements. None of
these fields is `regime_gated`, so the edit would not have DELETED values -- but the claim would
have been false, and it would have masked the one useful signal (a NULL where peers resolve it).

Withdrawn entirely rather than narrowed to the 1 well-evidenced cell, because (a) the industrial
block's own `_authority` says *"Nothing is structurally excused here ... EVERY absence is a
finding"*, so the edit contradicts the block's declared policy, and (b) the payoff had collapsed
from 552 codes to 69, about 0.1% of the run's reason codes. Arguing around a written register
rule for 0.1% is the same move that nearly asserted UNH has no premiums.

**The finding that outlives the proposal: the register's `by_regime` blocks never covered the
cross-regime top lines at all, for ANY regime.** `premiumsEarned`, `netInterestIncome`,
`noninterestIncome`, `netInvestmentIncome`, `realizedInvestmentGains`, `rentalIncome` and
`shortTermBorrowingsOnly` appear in no block -- which is why the audit reports 7 unregistered
structural fields for `energy` and 5 for `utility`. The register was built for the
Article-5-vs-others question and this whole family fell outside it. Closing it honestly needs
more than the 4 filers those regimes have in the roster; **Phase 5b**.

**`scripts/audit_absence_evidence.py`** is the durable answer to the user's question, and it uses
no rules -- only stored facts. Three verdicts per (regime, field): STRUCTURAL (0% of the regime's
filers ever resolve it -> absence is a property of the regime), UNIVERSAL (all resolve it -> a
NULL is a named DEFECT in our extraction), MIXED (only the filing settles it -> the validator's
work queue). Industrial: 16 UNIVERSAL / 31 MIXED / 1 STRUCTURAL. MIXED being the largest is the
honest headline: for most fields no config rule can tell you whether an absence is legitimate,
and the peer comparison tells you WHICH cells need a human to open the filing.

### Phase 5 VERIFIED on the rebuilt data (2026-08-24, 16:43) -- all eight gates PASS

Facts backfill: **317,036 rows / 54 tickers** (was 294,898 under the label PK -- **+22,138,
+7.5%**), 252,001 of them valued. Employees **745 rows / 54 tickers** (was 474 / 35, so the
19-ticker headcount gap closed as a side effect of `-F`). History replay 4h39m.

| gate | result |
|---|---|
| 1. grain | 3,267 rows / 54 tickers / 2009-07-31 -> 2026-08-10; duplicate `(ticker, as_of)` **0**, `fiscal_end` backwards **0**, look-ahead leaks **0** |
| 2. contract | **69 stored = 69 contracted, same columns in the same order** |
| 3. unexplained nulls | 196,020 cells, 71,857 null (36.7%), 76,004 codes, **UNEXPLAINED 0** |
| 4. filing lag | median **34d**, p90 55d, beyond 200d **1 of 3,267 (0.03%)** |
| 5. amendments | 21 rows / 14 tickers; 36 accessions (1.09% of filings); **3 of 36 (8%) beyond 365d, refused by decision 34**; row delta +0.65% vs v1's predicted +0.6% |
| 6. same-day collapse | 9 of 3,273 pairs carry >1 accession, max **4** |
| 7. regime top lines | revenue / equity / assets / **liabilities 100% in ALL EIGHT regimes** |
| 8. code mix | 16 of 19 codes firing, **none outside the declared vocabulary** |

**The 686-day lag outlier is a real filing, not a defect.** SMCI, `as_of=2019-05-17`,
`fiscal_end=2017-06-30`, form 10-K, not an amendment: Super Micro missed filings through its
accounting review and was delisted from Nasdaq in 2018, filing the FY2017 10-K in May 2019. The
same date is also gate 6's 4-accession same-day collapse -- it filed its catch-up reports
together. Two independent gates describing one real event is evidence the grain is faithful.

### The NCI ladder: `totalLiabilities` NULL 210 -> 38

Gate 7 initially showed `totalLiabilities` NULL for EOG (energy) and LLY / MCD (industrial). All
three tag no `Liabilities` total at all, so all three depend on the assets - equity identity, and
it was refusing whenever equity resolved on the EX-NCI element with `minorityInterest` NULL. MCD
made the cost visible: assets $59.92bn, equity **-$1.02bn** (McDonald's has run negative equity
since its buyback programme), i.e. $60.94bn of liabilities the guard declined to state.

Two rounds of user challenge improved this twice:

1. *"Is it impossible to deduce minorityInterest?"* -- **No.** Where a filer tags equity on BOTH
   bases at one `period_end`, the difference IS the NCI: two filed facts, no inference. Now
   implemented (`_deduced_nci`) and carried forward like any other instant.
2. My "must stay refused" table was **wrong**: computed on LIFETIME NCI facts while the code
   tests point-in-time. TMO has 38 valued NCI facts but files its first on 2022-02-24 against a
   history opening 2011-11-04 -- a decade of events with no NCI to know. Only a filer disclosing
   NCI in its FIRST filing (LLY, ETN) is refused throughout.

Resulting precedence, weakest last: **tagged NCI -> deduced NCI -> assumed zero (own code) ->
refuse.**

| | rows | tickers |
|---|---|---|
| `derived_identity` (tagged or deduced NCI -- evidence) | **749** | 17 |
| `derived_identity_nci_assumed_zero` (the assumption) | **152** | 6 |

83% of derived cells are evidence-backed, and the two are separable because the assumption got
its own code (user's choice over folding it into `derived_identity`). The deduction also BOUNDS
what was ever at stake: EOG's two bases agree **to the dollar** on 6 of 7 overlap dates, and
TMO's differ by $8-47M against $27-39bn of equity (**0.02-0.12%**).

**Every one of the 210 original NULLs is now accounted for**: 172 recovered, and all 38 remaining
are missing `stockholdersEquity` (29 also missing `totalAssets`) -- **0 have both inputs
present**. Nothing refuses for want of an inference any more; what is left has no inputs.

`ALL_CODES` is 19, `IS_QUALIFIER` 6.

### Phase 5 deliverables — the files, as shipped

**New modules**

| file | what it is |
|---|---|
| `src/data_extract/utils/fundamentals/build_history.py` | the phase. `fundamentals_facts` -> `fundamentals_history` + `fundamentals_reason_codes` on the publication-event grain. Public API: `build_ticker_history`, `build_ticker`, `publication_events`, `carry_latest_known`, `diff_against_stored`, `build_fundamentals_history`, `facts_frame_from_companyfacts` |
| `src/data_extract/utils/fundamentals/reason_codes.py` | the closed `dc_code` vocabulary, **19 codes**, 6 of them qualifiers. Imports the codes that already existed rather than restating them, so each has exactly one definition |

**New scripts**

| file | what it answers |
|---|---|
| `scripts/verify_fundamentals_history.py` | the eight §5.8 gates over the LIVE tables, read-only, non-zero exit on failure -- so it can gate a rebuild rather than describe one |
| `scripts/recreate_fundamentals_tables.py` | DROP + CREATE the four tables from `sql/schema.sql`. Needed because that file runs only at volume INIT; on a live volume `store.save` infers DDL from the first frame, which is how an all-None column once became TEXT. `--dry-run` prints counts and SQL, `--yes` applies |
| `scripts/audit_absence_evidence.py` | **is a NULL missing for a good reason?** Three verdicts per (regime, field) from stored facts alone, no rules: STRUCTURAL (0% of the regime's filers resolve it), UNIVERSAL (all do -> a NULL is a named DEFECT), MIXED (only the filing can settle it). Industrial: 16 / 31 / 1 |
| `scripts/measure_total_liabilities_legs.py` | the §5.1 measurement that chose the identity over route 3b |

**New tests** — `tests/data_extract/test_build_history.py` (16) and `test_amendment_grain.py` (3);
with `test_step_extract_fundamentals.py`, **21 passed**. `test_periods_q4.py` + `test_amendment_grain.py`
**32 passed** across the numpy and `_fiscal_bounds` rewrites.

**Modified** — `periods.py` (new `fiscal_quarter_of_end` + shared `_fiscal_bounds`; numpy rewrite of
`_drop_annual_masquerading_as_quarter`, which was >50% of the engine's cost: **14 min/ticker ->
2.4**), `kpi_catalogue.py` (`HISTORY_STATEMENT_ORDER`, `history_columns` as THE contract),
`fetch_fundamentals_sec.py` (`_period_end`, refusal stubs, employees wired), `xbrl_linkbase.py`
(`basis_qualifier`), `schema.py` + `sql/schema.sql` (the 69-column DDL, the facts PK, two new
tables), `cli.py` (`-F/--full`, `fundamentals-history`, `fundamentals --rebuild`),
`step_extract_fundamentals.py`, `edgar_driver.py`, and `docs/{data_schema,database,runbook}.md`.

**Report** — `reports/2026-08-24/fundamentals-rebuild-phase5__DATA.md`. Gates 1 pass / 1 fail /
3 n/a; **the D4 FAIL is the invocation, not the data** -- one `--expect-through` was applied across
a quarterly and an annual table, and `fundamentals_employees` is annual (max `as_of` 2026-07-29 is
the latest 10-K; the three quarterly tables all reach 2026-08-10). Explained in its §5, and
re-running it per-table is listed as a next action.

### What Phase 5 did NOT do

- **The 9 `tests/data_aggregate` failures are still failing.** They pre-date this phase (verified
  identical on a stashed tree) and are **Phase 6**. The cube will not build against this table
  until they are fixed.
- **Scope is 54 tickers, not 500.** Deliberate; widening is Phase 9's acceptance step. Every
  coverage rate in the report is against a 54-ticker denominator.
- **An ABSENCE still carries no evidence.** A resolved fact row stores `source_concept` /
  `role_uri` / `roll_up_children`; an unresolved one stores NULL in all of them, so
  `not_disclosed` (68% of codes) is the resolver's verdict about our concept MAP and not a
  checkable claim about the filing. Recording what the filer DID tag on the relevant statement is
  the durable fix, and it is Phase 5b.
- **The register's `by_regime` blocks never covered the cross-regime top-line family** for any
  regime. Phase 5b, and it needs more than the 4 filers `energy` / `utility` / `insurer` have here.

---

## PHASE 5a FINAL STATUS — read this before starting 5b

Every `- [ ]` in the Phase 5 body above is adjudicated here. Four buckets: **CLOSED** (built,
verified, do not revisit), **DIFFERENT** (built another way -- the spec line is wrong, this is
right), **DO NOT DO** (measured and refuted -- attempting it in 5b would re-introduce a defect),
**STILL OPEN** (real remaining work, with an owner).

### A. CLOSED — working, verified, do not revisit

| what | evidence it is closed |
|---|---|
| the publication-event grain (§5.0, 5 rules) | 3,267 rows / 54 tickers; **0** duplicate `(ticker, as_of)`, **0** `fiscal_end` regressions, **0** look-ahead leaks. `as_of` is no longer computed -- the median-of-spine heuristic is gone |
| **`test_fundamentals_point_in_time.py` — the defining acceptance test — is GREEN, 4/4** | Was 3/4 on 2026-08-24: `test_one_row_per_ticker_fiscal_period` asserted one row per `(ticker, fiscal_end)`, which is the OLD grain and demanded the table discard either the original or the restatement. Rewritten as `test_the_grain_is_one_row_per_publication_event_and_every_repeat_is_explained`: `(ticker, as_of)` unique (**0** duplicates) AND every repeated `fiscal_end` explained by an amendment or a declared `fundamentals_cik_cutover.json` boundary — **17 of 17 groups explained, 0 unexplained**. Monotone `as_of` 0 violations; lag median 34d, 1 row beyond 200d |
| `as_of` is a filing date; no leak | gate 1, and structurally via `_latest_period_known` capping `fiscal_end` at `as_of` |
| the 69-column contract, in statement order | gate 2 checks the STORED table's columns *in order* against `Catalogue.history_columns`; `build_ticker` asserts the length; a test asserts the family breakdown and the statement relations |
| `fiscal_quarter` on every row | Q1-Q4 incl. TTM and instant rows; correct for 52/53-week and non-December filers (AAPL: December = **Q1**, September = **Q4**). 11 NULLs, all pre-first-annual-filing |
| one column per field, bare name, TTM for flows | decision 31, delivered; `totalRevenue` always was the TTM so no downstream rename |
| `ebitda_q` / `freeCashflow_q` / `capexGlobal` dropped | asserted absent by `test_build_history` |
| `sector` / `industry_group` out (decision 32) | asserted absent; `regime` stays because it drives RESOLUTION |
| `revenueGrowth` / `earningsGrowth` out (decision 33) | `CUBE_TIME_COLUMNS`, subtracted defensively as well as excluded in config |
| the dense reason-code table + closed vocabulary | **19** codes, 6 qualifiers, single definition each; 76,004 rows; gate 8 finds **none** outside the set |
| **zero unexplained nulls** (items 7, 9, B.6.6) | 196,020 cells, 71,857 null, **0** unexplained. THE gate the phase was accepted on |
| every refused quarter gets a code (item 7) | `periods.py::_derived` records refusals; the history build writes them, so `periods.py` stays pure |
| `fundamentals_employees` side table | 745 rows / **54 of 54** tickers. Text-parsed headcount cannot fail the wide build |
| `employees` stays in the catalogue, leaves the wide table | `Catalogue.side_table_fields == ["employees"]` |
| append-only, enforced not asserted | `diff_against_stored`, exact comparison, no tolerance. A second run appends **0** rows and raises nothing |
| two rebuild flags, ticker-level (decision 27) | `--rebuild-history` (local) and `fundamentals --rebuild` (deletes all four, refetches). No `build_version` column -- the rebuild IS the version |
| regime gating | `_gate`; `not_applicable_for_regime` fires 14,592 times / 25 tickers |
| same-day collapse by `(ticker, date)` + form precedence | 9 of 3,273 pairs, max **4**; measured, not theoretical |
| the >365-day amendment cutoff (decision 34) | **3 of 36 accessions (8%)** refused; boundary exact (+365 admits, +366 refuses) |
| no-op amendments emit nothing | ADM 2024-11-18: three amendments **inside** the window moved **0** fields (its restatement was intersegment-only) -> no row. Rule 2 working on a real filer |
| the amendment round-trip | SPG 2016-01-13 restates `fiscal_end` 2015-09-30: netIncome **1,949M -> 2,155M**, and the 2015-11-04 row keeps its as-filed numbers |
| route 3b keeps its STRICT period intersection | kept; `period_intersection_partial` fires 205 times / 19 tickers instead of silently mixing bases |
| calc linkbase read off `filing.xbrl()`, never companyfacts | unchanged from Phase 4; companyfacts drops dimensioned facts |
| the DDL / schema risk zone | applied under explicit approval; `scripts/recreate_fundamentals_tables.py` makes it repeatable |
| filing lag sane | median **34d**, p90 55d, **1 of 3,267** beyond 200d -- and that one is real (SMCI's delinquent FY2017 10-K, filed 686 days late during its delisting) |
| `totalLiabilities` coverage | **100% in all eight regimes**; NULL fell 210 -> 38 and all 38 lack `stockholdersEquity` outright |

### B. DIFFERENT from the spec — the spec line is wrong, this is right

| spec said | delivered instead, and why |
|---|---|
| §5.1 stamp `resolution_method = 'derived_identity'` in **`fundamentals_facts`** | Stamped in **`fundamentals_reason_codes`** from the HISTORY layer. A derived number in the facts table would break its documented as-filed contract ("every row carries a number the filer actually tagged"), which the publication-event grain rests on. 5b's `cross_identity` obligation is unchanged: a row carrying this code is an INPUT, never corroboration |
| §5.1 reason-code `not_disclosed` when legs refuse **and** basis is ex-NCI **and** NCI did not resolve | A **four-step ladder**: tagged NCI -> NCI **deduced** from the filer's two equity bases (`incl - ex`, two filed facts) -> assumed zero **only** where the filer never tagged one, under its own `derived_identity_nci_assumed_zero` -> refuse. The spec's rule conflated "not tagged" with "zero" and cost MCD its entire history ($60.94bn of liabilities left NULL). 749 evidence-backed cells vs 152 on the assumption |
| §5.2 **68** columns | **69** -- `fiscal_quarter` added on user request; the 60 value columns reordered into statement order |
| the `totalLiabilities` `derived_fallback` config key | **Dropped as unnecessary.** The basis is now queryable per row in `fundamentals_reason_codes`; a config key would restate it without making it measurable |

### C. DO NOT DO — measured and refuted. Attempting these in 5b re-introduces a defect

1. **Do NOT add `roll_up.any_of` for `totalLiabilities` and let route 3b sum the legs.** Measured on
   44 10-Ks: **0 of 44** declare a `Liabilities` total, 44 of 44 declare a leg-set, and the raw
   route-3b refusal rate is 68% -- caused entirely by `us-gaap:CommitmentsAndContingencies`, a Reg
   S-X 5-02(24) footnote pointer, dropping to 4.5% once excluded. But route 3b **enumerates** legs,
   an unlisted us-gaap sibling is dropped **silently**, and leg-sets vary by filer AND year (EOG,
   TMO, MCD all change between 2023 and 2024). The failure mode is a Tier-1 balance-sheet total
   short by a caption that looks entirely plausible -- the `shortTermDebt` defect this rebuild
   exists to remove. The identity wins on the merits, as §5.1's own escape clause allowed.
2. **Do NOT write the absence register from the Reg S-X caption list.** §B.5 refuted "element names
   are evidence of what a filer declares" twice; a caption list is that reasoning a third time.
3. **Do NOT mark the cross-regime top lines `expected_absent` for `industrial`.** Proposed,
   approved, then WITHDRAWN on measurement: of 27 industrial filers only `noninterestIncome` is
   0/27. `premiumsEarned` is real for **UNH ($72bn), CVS ($34bn), DE ($248M)**, `rentalIncome` for
   **AMT ($3.5bn), CAT ($549M), BA, CSCO**, `netInterestIncome` for **WMT (-$990M), CVS, ETN**. A
   GICS sub-industry of `industrial` does not remove those lines from an income statement.
4. **Do NOT assume a NULL `minorityInterest` is zero as a blanket rule.** Correct only where the
   filer has never tagged one, tested **point-in-time**. LLY (134 valued NCI facts) and ETN (120)
   disclose from their first filing and must stay refused; TMO looks refusable on lifetime counts
   but files its first NCI on 2022-02-24 against a history opening 2011-11-04.
5. **Do NOT relax route 3b's strict period intersection.** The obvious repair reintroduces mixed
   bases.
6. **Do NOT trust `(fiscal_year, fiscal_period)` as a fact identity.** A single filing legitimately
   carries the same label pair more than once (AAPL's FY2025 10-K reports FY2023, FY2024 and
   FY2025 annual revenue). Keying on the labels silently dropped **18,604 of 337,190 rows (5.5%)**,
   with 16,340 collisions carrying 2+ different values. `period_end` is the identity.

### D. STILL OPEN in Phase 5a — the real remainder

| # | item | detail | size |
|---|---|---|---|
| **5a-2** | **ETN dual-registrant row carries the SHELL's numbers** — a VALUE defect, not a grain one | Two registrants each filed a 10-Q for period 2012-09-30 during Eaton's Nov-2012 Irish redomicile: cik `0000031277` (Eaton Corp, Ohio) on 2012-10-31 with assets **18,800M**, and cik `0001551182` (Eaton Corp plc) on 2012-11-14 with assets **5M** — the newly-formed holdco's shell balance sheet. Both filings are real publication events, so **two rows is the correct grain** (and `cutover_filings` splits registrants by FILING date, disjointly, exactly as designed). What is wrong is the second row's VALUES: the series steps 18,800M -> 5M, which a model reads as the company evaporating. The register has no rule for a PERIOD that straddles the cutover, and the successor's first filing reports on an entity that was not yet the operating group. Scope is exactly **1 ticker, 1 period, 2 rows of 3,267 (0.06%)**; APA and GOOGL have **0** dual-filed periods, so a fix keyed on `period_of_report` at a declared boundary touches nothing else. Candidate rule: at a cutover, prefer the PREDECESSOR's filing for a `period_of_report` that ends before `cutover_date` | small and precisely bounded |
| **5a-3** | **§2.9's broken-ticker population cannot be fully closed at this scope** | §5.8 asks for bank/insurer/REIT top lines non-null for "the 11 banks with NII but no noninterest income, the 6 insurers with premiums but no NII, the 17 with neither leg". That population was measured on the **442-ticker legacy substrate**; the roster here is 54, and on it every regime is at **100%**. So this is closed FOR THE ROSTER and re-opens at Phase 9 scope. Do not read the 100% as covering the named 34 tickers | re-measure at Phase 9 |
| **5a-4** | **B.6.6's "worse half" needs re-verification before closing** | The spec says SCHW `cash` is refused with no null for a gate to find, so it needs a qualifier of its own. In the rebuilt table SCHW `cash` is **60/60 non-null** carrying only `regime_break` x4, and no `period_intersection_partial` fires on SCHW at all. Either the original condition no longer occurs or the qualifier is not firing where it should -- **verify the condition still exists before ticking or removing this** | one measurement |
| **5a-5** | **Compustat's XRD caveat was never recorded as an expectation** | §5.3 asks that "only ~1.2% of missing `XRD` is genuinely missing" be inherited as an *expectation*, not a target. It is not written down anywhere in the tree | documentation only |
| **5a-6** | **The DoD report's D4 gate reads FAIL** | Not a data gap: one `--expect-through` was applied across a quarterly and an annual table, and `fundamentals_employees` is annual (max `as_of` 2026-07-29 is the latest 10-K; the three quarterly tables all reach 2026-08-10). Re-run `data_profile.py` splitting the annual table out | one command |
| **5a-7** | **§5.1's "pair the re-sweep with B.6.8's outstanding one" — unverified** | A full re-sweep DID happen (the PK rebuild, 317,036 facts). Whether B.6.8's outstanding sweep item was covered by it was never checked, and the point of the instruction was to spend the ~2h of network **once** | one check |

**Nothing in section D blocks 5b from starting.** 5a-1 (the acceptance test) was FIXED on
2026-08-24 and moved to section A. **5a-2 is now the only substantive item left**, and it is one
ticker, one period, two rows.

### E. Explicitly HANDED ON — not Phase 5a's work

| to | item |
|---|---|
| **Phase 6** | the **9 `tests/data_aggregate` failures** (verified identical on a stashed tree, so they pre-date this phase). Includes `test_composites_config.py::real_panel`, which §5.8 hoped would start working and does not |
| **Phase 6** | repoint `employee_features.py` at `fundamentals_employees` |
| **Phase 6** | reconcile the declared casualties (`ebitda_q`, `freeCashflow_q`, `capexGlobal`) and move `revenueGrowth` / `earningsGrowth` into `pit.py` on a **365-day** `as_of` offset (a 4-ROW offset is ~9 months once an amendment row exists) |
| **Phase 6** | ⚠ once a cube is trained on this table, a delete-and-rebuild is still allowed but must be **logged** -- it re-derives numbers a model has already seen |
| **Phase 5b** | register the **cross-regime top-line family** per regime. `scripts/audit_absence_evidence.py` reports 7 unregistered structural fields for `energy` and 5 for `utility`, but those regimes have only **4 filers each** here -- 0-of-4 is not evidence. Needs Phase 9 scope |
| **Phase 5b** | the **MIXED population**: 31 of 48 industrial fields resolve for some filers and not others, so no config rule can decide whether an absence is legitimate. This is the validator's real work queue |
| **Phase 5b** | **give an ABSENCE the evidence a presence has.** A resolved fact row stores `source_concept` / `role_uri` / `roll_up_children`; an unresolved one stores NULL in all of them, so `not_disclosed` (68% of codes) is a verdict about our concept MAP, not a checkable claim about the filing. Recording what the filer DID tag on the relevant statement is the durable fix |
| **Phase 5b** | `cross_identity` must treat `derived_identity` and `derived_identity_nci_assumed_zero` rows as INPUTS, never as corroboration |
| **Phase 9** | widen 54 -> full roster; every rate in the DoD report is against a 54-ticker denominator and must be re-measured |
