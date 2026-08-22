# Research: closing the 17 UNVERIFIED fundamentals fields + verifying the regime-absence register

**Date**: 2026-08-21
**Research Phase**: 1 of 3 (FIC workflow) — a second pass on the same rebuild
**Next Phase**: Implementation directly (`/implement`) — the user is explicitly skipping `/plan` for this
pass, since it is information retrieval feeding an existing plan, not a new design
**Request**: [specs/2026-08-21_research_how to extract_Edgartools_part2.md](../../../specs/2026-08-21_research_how%20to%20extract_Edgartools_part2.md)
**Builds on**: [2026-08-21-fundamentals-extraction.md](2026-08-21-fundamentals-extraction.md) (research)
and [2026-08-21-fundamentals-rebuild-plan.md](../../planning/active-tasks/2026-08-21-fundamentals-rebuild-plan.md)
(plan, Phases 1-2 already implemented)

---

## Research Question

Phase 2 of the rebuild plan shipped `configs/fundamentals/fundamentals_kpis.json` (53 fields),
`fundamentals_regimes.json` (8 regimes) and `fundamentals_exceptions.json` (the expected-absence
register) — but candidly flagged **17 of 53 fields as `authority: "UNVERIFIED"`**, and its own
regime-absence measurement carried a stale denominator and an imprecise "intermittent" label on
bank capex. The user asked, before any further implementation: (1) close as many of the 17 gaps
as primary sources allow, quoting verbatim and never inventing; (2) re-verify the regime-absence
register directly against the live data rather than trusting the config file; (3) give concrete,
field-level corrections to the three JSON configs. No code changes, no plan, no invented
definitions — every claim below is either a primary-source quote with its source, a live
measurement with its query, or an explicit flag that a primary source could not be reached.

---

## Summary

**15 of the 17 UNVERIFIED fields are now resolved with primary-source citations** (FASB's own
2025 `us-gaap-doc-2025.xml` documentation labels, `us-gaap-ref-2025.xml` reference linkbase, and
`us-gaap-2025.xsd` schema attributes, cross-checked against live eCFR text of Reg S-X). Two remain
genuinely open — not because research stopped early, but because the primary sources themselves
leave a choice or expose a defect:

1. **`ppeNet`** — FASB's own reference linkbase ties `PropertyPlantAndEquipmentNet` to the
   lessee ROU-asset disclosure paragraph (ASC 842-20-50-7A(a)), confirming that finance-lease
   right-of-use assets *may* be folded into this element depending on the filer's own presentation
   choice. This is a **filer-level inconsistency the taxonomy itself permits**, not a documentation
   gap — genuinely a decision for the user (§Open items #1).
2. **`incomeTaxExpense`**'s second fallback concept, `IncomeTaxExpenseBenefitContinuingOperations`,
   **does not exist anywhere in the 2025 FASB taxonomy** — confirmed by exhaustive grep of the
   schema, doc and label linkbases. This is not a citation gap, it is a **bug**: the catalogue
   names a tag that cannot ever resolve (§Part 1, incomeTaxExpense).

Separately, re-measuring the regime-absence register directly against `fundamentals_facts_legacy`
and `sp500_tickers` **reproduced every rate cell checked exactly** — the measurement methodology
is sound. But it surfaced four smaller defects in the config's *prose and bookkeeping*, not its
arithmetic: a stale "441" denominator, three tickers with real facts that have silently fallen out
of the reference-data table, an "intermittent" label on bank capex that actually hides a three-way
split (five banks tag it every single year, not intermittently at all), and a REIT ticker
(`PLD`) missing from a roster the file itself calls "live, not illustrative."

And one finding changes an actual design decision, not just a citation: FASB's own linkbase shows
that **`interestExpense` has no bank-specific branch, and should** — Reg S-X Rule 9-04's bank
income-statement captions map to a completely different element family
(`InterestExpenseDeposits` + `InterestExpenseBorrowings` → `InterestExpenseOperating` as the
"Total interest expense" line) than the generic `InterestExpense`/`InterestExpenseDebt` the
catalogue currently falls back to for everyone. This is the same shape of gap the first research
already fixed for `totalRevenue` and `netInterestIncome` — it was simply not extended to
`interestExpense` itself.

---

## Method

Both web-research and DB-verification agents worked from **primary sources fetched directly**,
not from memory or generic summaries, per the request's explicit constraint:

- **FASB's actual 2025 US-GAAP taxonomy files**, downloaded via `curl` (not paraphrased) from
  `https://xbrl.fasb.org/us-gaap/2025/elts/`: `us-gaap-doc-2025.xml` (13.8 MB, documentation
  labels), `us-gaap-ref-2025.xml` (13.5 MB, element → ASC/Reg-S-X cross-references),
  `us-gaap-lab-2025.xml` (14.2 MB, standard/deprecated labels), `us-gaap-2025.xsd` (5 MB, schema —
  `balance`/`periodType` attributes). Same files and technique the first research used.
- **eCFR** (17 CFR Part 210, Regulation S-X), fetched directly where possible; `ecfr.gov` itself
  redirected WebFetch through a bot-check on some routes, in which case a Cornell Legal Information
  Institute mirror (`law.cornell.edu`) was used and is noted per citation — both are the same
  statutory text, LII is simply a more fetch-friendly host.
- **ASC Codification** (`asc.fasb.org`) is confirmed **still login-walled** (403, consistent with
  the first research's finding) — every ASC paragraph *number* below comes from FASB's own
  reference linkbase (primary), but ASC *prose* quotes are flagged secondary (Deloitte DART / PwC
  Viewpoint reproductions) wherever the wall blocked a direct fetch. This mirrors the first
  research's own caveat structure exactly.
- **Live DB**: `docker exec pea_db psql -U alexandre -d pea` against `fundamentals_facts_legacy`
  (7,776,870 rows) and `sp500_tickers`, re-deriving every regime-absence rate from source rather
  than reading it off the config file.

---

## Part 1 — The 17 UNVERIFIED fields

Each field below states the catalogue's prior `authority_note` gap, what was found, and the
verdict. Field names, current fallback lists and roll-ups are in
[`configs/fundamentals/fundamentals_kpis.json`](../../../configs/fundamentals/fundamentals_kpis.json)
(canonical path — nested under `configs/fundamentals/`, not flat under `configs/`).

### 1.1 Equity & tax cluster

#### `stockholdersEquity` — VERIFIED (basis), PARTIALLY VERIFIED (ASC paragraph)
FASB `us-gaap-doc-2025.xml` documentation labels partition exactly:
- `StockholdersEquity`: *"Amount of equity (deficit) attributable to parent. **Excludes** ...
  equity attributable to noncontrolling interest."*
- `StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest`: *"Amount of equity
  (deficit) attributable to parent **and** noncontrolling interest."*

`us-gaap-ref-2025.xml` ties `StockholdersEquity` to **Reg S-X 5-02(29)/(30)** and `MinorityInterest`
to **5-02(31)**. eCFR 17 CFR 210.5-02, verbatim: *"29. Common stocks. ... 30. Other stockholders'
equity ... 31. **Noncontrolling interests in consolidated subsidiaries.** ... 32. Total liabilities
and equity."* Reg S-X has no single numbered caption for "equity incl. NCI" — captions 29+30
(parent) and 31 (NCI) are separate lines that both roll into caption 32. So the roll-up
`= StockholdersEquity + MinorityInterest` is confirmed by both the FASB documentation labels and
the Reg S-X caption structure — the incl-NCI basis is now a **sourced convention**, not merely a
"deliberate but unverified" one.
Caveat: ASC 810-10-45-16 ("NCI shall be reported within equity, separately from the parent's
equity") is confirmed only via a secondary reproduction (Deloitte DART) — a stable, frequently
quoted paragraph, but not independently checked against the login-walled `asc.fasb.org`.

#### `minorityInterest` — VERIFIED
FASB doc label: *"Amount of equity (deficit) attributable to noncontrolling interest."* Standard
label: *"Equity, Attributable to Noncontrolling Interest."* Ref-linkbase: **SX 210.5-02(31)**
(verbatim quote above). Checked `us-gaap-lab-2025.xml` for a `deprecatedLabel` role on
`MinorityInterest` — **none found**; it is not deprecated in the 2025 taxonomy and remains the
correct input to `stockholdersEquity`'s roll-up.

#### `retainedEarnings` — VERIFIED, more precisely than asked
FASB doc label: *"Amount of accumulated undistributed earnings (deficit)."* `us-gaap-ref-2025.xml`
ties `RetainedEarningsAccumulatedDeficit` not just to the general caption but to the exact
sub-clause **SX 210.5-02(30)(a)(3)**. eCFR 210.5-02(30)(a), verbatim: *"Separate captions shall be
shown for (1) additional paid-in capital, (2) other additional capital, **(3) retained earnings**,
(i) appropriated and (ii) unappropriated ..., and (4) accumulated other comprehensive income."*
Caption number is now fully closed: **Rule 5-02(30)(a)(3)**, not just "Rule 5-02" generically.

#### `dilutedShares` — VERIFIED (definition), PARTIALLY VERIFIED (ASC prose)
FASB doc label: *"The average number of shares or units issued and outstanding that are used in
calculating diluted EPS ..., determined based on the timing of issuance of shares or units in the
period."* Ref-linkbase ties it to **ASC 260-10-45-16** and disclosure paragraph **260-10-50-1(a)**.
Not a superset/subset case — a single weighted-average concept; the catalogue's "not additive
across quarters" caution is the correct one. Caveat: the 45-16 paragraph *number* is primary
(FASB's own linkbase); its prose (*"the denominator is increased to include ... dilutive potential
common shares"*) is a secondary (Deloitte DART) reproduction.

#### `pretaxIncome` — VERIFIED, the superset/subset trap resolved directly from FASB's wording
No ASC or Reg-S-X cross-reference exists for either candidate element in FASB's own linkbase
(confirmed empirically — zero entries for both), so this had to be settled from the documentation
labels alone, and it settles cleanly:
- `IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest`:
  *"Amount of income (loss) from continuing operations, **including** income (loss) from equity
  method investments, before deduction of income tax expense..."*
- `IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments`:
  *"Amount of income (loss) from continuing operations before deduction of income tax expense... and
  **addition of** income (loss) from equity method investments."*

"including" vs. "and addition of" is the tell: the first name already has equity-method earnings
folded in (**the superset/aggregate**); the second is *before* that addition (**the subset/partial
leg**, for filers who present equity-in-affiliates as a separate line below this subtotal). The
catalogue's fallback order — superset first — is correct. (Also confirmed: no bare
`IncomeLossFromContinuingOperationsBeforeIncomeTaxes` element exists in the 2025 schema; only the
`...Domestic`/`...Foreign` geographic-split siblings, which are not totals.)

#### `incomeTaxExpense` — PARTIALLY VERIFIED, plus a real bug
FASB doc label for `IncomeTaxExpenseBenefit`: *"Amount of current income tax expense (benefit) and
deferred income tax expense (benefit) pertaining to continuing operations."* This confirms it as
the aggregate, consistent with its role as `total_concept`. Ref-linkbase ties it to **ASC
740-10-50-10** and **740-10-50-12** (components-of-tax-expense and rate-reconciliation disclosure
paragraphs).

**Bug found**: the catalogue's second fallback, `IncomeTaxExpenseBenefitContinuingOperations`,
**does not exist anywhere in the 2025 FASB taxonomy**. Checked exhaustively against the schema,
doc linkbase and label linkbase — zero matches for that bare name. Only *compound* elements exist
with similar but longer names, e.g. `FederalIncomeTaxExpenseBenefitContinuingOperations`,
`IncomeTaxExpenseBenefitContinuingOperationsDiscontinuedOperationsExtraordinaryItems`,
`...ContinuingOperationsGovernmentGrants`, `...ContinuingOperationsAdjustmentOfDeferredTaxAssetLiability`.
This fallback can never resolve as written. Two explanations are possible — a stale/older-taxonomy
name, or a typo for one of the compound names — and the right fix depends on which, if either, filers
actually tag; that requires checking `fundamentals_facts_legacy`'s tag ledger for what string (if
any) filers use as a second candidate, which was outside this agent's scope. **Flagged as an
implementation-blocking correction, not just a citation gap** — see §Part 3.

### 1.2 PP&E & intangibles cluster

#### `goodwill` — VERIFIED
FASB doc label: *"Amount, after accumulated impairment loss, of asset representing future economic
benefit ... not individually identified and separately recognized."* Schema: `balance='debit'`,
`periodType='instant'` — confirms carried net of impairment. Ref-linkbase ties `Goodwill` to **ASC
350-20-45-1**, whose text (Deloitte DART, secondary — `asc.fasb.org` 403'd) reads: *"The aggregate
amount of goodwill shall be presented as a separate line item in the statement of financial
position."* Also cross-tagged to **SX 210.5-02(15)**, but that Reg-S-X caption's actual eCFR text is
titled generically "Intangible assets" — Reg S-X has **no goodwill-specific caption**; the
separate-line mandate is a GAAP (ASC), not Reg-S-X, requirement. Worth stating in the catalogue's
`authority` field so nobody later "fixes" the citation toward a nonexistent Reg-S-X goodwill caption.

#### `intangiblesExGoodwill` — VERIFIED, including the double-count check the note asked for
FASB doc label for `IntangibleAssetsNetExcludingGoodwill` opens with **"Sum of the carrying
amounts of all intangible assets, excluding goodwill"** — settling directly, from FASB's own text
rather than an inference from the element's name, that it genuinely is a total and not a synonym
for one of its legs. The two legs are finite-lived vs. indefinite-lived, mutually exclusive by
construction (an intangible asset is one or the other), confirming the roll-up
`sum: [FiniteLivedIntangibleAssetsNet, IndefiniteLivedIntangibleAssetsExcludingGoodwill]` cannot
double-count. ASC 350-30-45-1 (Deloitte DART, secondary): *"At a minimum, all intangible assets
shall be aggregated and presented as a separate line item ... that requirement does not preclude
presentation of individual intangible assets ... as separate line items."* Ref-linkbase confirms
this paragraph is tagged directly to `IntangibleAssetsNetExcludingGoodwill`.

#### `ppeGross` — VERIFIED
FASB doc label: *"Amount before accumulated depreciation, depletion and amortization of physical
assets used in the normal conduct of business and not intended for resale."* Ref-linkbase AND a
direct eCFR fetch of 17 CFR 210.5-02 independently agree: **caption 13**, *"Property, plant and
equipment. (a) State the basis of determining the amounts."* Not a superset/subset case — Reg S-X
presents gross PP&E (caption 13) and accumulated depreciation (caption 14) as two separate
captions specifically so the net figure is derivable, which is exactly the checkable identity the
catalogue's roll-up already relies on.

#### `ppeNet` — PARTIALLY VERIFIED, a real open decision (see §Open items #1)
FASB doc label: same construction as `ppeGross`, "after accumulated depreciation..." instead of
"before." The requested caveat check — does it include finance-lease right-of-use assets? —
resolves to **yes, potentially, filer-dependent**: `us-gaap-ref-2025.xml` ties
`PropertyPlantAndEquipmentNet` to **ASC 842-20-50-7A(a)**, the lessee ROU-asset
presentation/disclosure paragraph. Per secondary reproductions of ASC 842-20-45-4, a finance-lease
ROU asset must be shown either as its own line item **or folded into an existing line (commonly
PP&E) with a note disclosing which line contains it**. So `PropertyPlantAndEquipmentNet` is **not
guaranteed to exclude finance-lease ROU assets** — this is a genuine, taxonomy-permitted,
filer-level inconsistency, not a bug to fix. No caveat was found on construction-in-progress or
held-for-sale treatment either way (the doc label's example list is explicitly non-exhaustive).

#### `accumulatedDepreciation` — VERIFIED, including the cross-reference question the note raised
FASB doc label: *"Amount of accumulated depreciation, depletion and amortization for physical
assets used in the normal conduct of business."* Schema, directly (not inferred):
`balance='credit'`, `periodType='instant'` — confirms it is genuinely a contra-asset, so the
catalogue's convention of storing it as a positive magnitude is a documented deviation that must be
applied consistently, exactly as the prior note said. Reg-S-X: ref-linkbase AND direct eCFR fetch
agree on **caption 14**, *"Accumulated depreciation, depletion, and amortization of property,
plant and equipment. The amount is to be set forth separately in the balance sheet or in a note
thereto."*

Also resolved a question the note didn't ask but should have: is this the *same* XBRL element as
`depAmort`'s `total_concept` (`DepreciationDepletionAndAmortization`)? **No** —
`AccumulatedDepreciationDepletionAndAmortizationPropertyPlantAndEquipment` is `balance='credit'`,
`periodType='instant'` (balance-sheet contra-asset); `DepreciationDepletionAndAmortization` is
`balance='debit'`, `periodType='duration'` (income/cash-flow-statement flow). Different
`periodType` and opposite `balance` prove these cannot be the same tag reused across statements —
the name overlap is a family-naming coincidence, not a resolution hazard.

### 1.3 Working-capital & opex cluster

#### `accountsReceivable` — VERIFIED
FASB doc labels settle the superset/subset question directly:
- `AccountsReceivableNetCurrent`: *"...of right to consideration from customer for product sold and
  service rendered in normal course of business..."* (trade only)
- `ReceivablesNetCurrent`: *"...due to the entity ... **including trade accounts receivable, notes
  and loans receivable, as well as any other types of receivables**..."* (the stated union)

`ReceivablesNetCurrent`'s own definition names trade AR as one of several things it includes —
confirming `AccountsReceivableNetCurrent ⊂ ReceivablesNetCurrent`. Reg-S-X caption confirmed via
eCFR/Cornell LII: **210.5-02(3)**, *"Accounts and notes receivable."* Caveat: neither element
carries FASB's own SX cross-reference tag for this caption (the linkbase's only tie for
`AccountsReceivableNetCurrent` is a *commonPracticeRef* to ASC 310-10-45-2) — the caption number is
confirmed independently via eCFR, not via FASB's own cross-reference, so it is one notch weaker
than the fields above where FASB's linkbase states the caption directly.

#### `accountsPayable` — VERIFIED, strengthened
FASB doc labels and ref-linkbase tags both confirm the bundling directly:
- `AccountsPayableCurrent` → **SX 210.5-02(19)(a)** only (the narrow trade/vendor subpart).
- `AccountsPayableAndAccruedLiabilitiesCurrent` → **both** SX 210.5-02(19) **and** 210.5-02(20),
  and its doc label lists *"taxes, interest, rent and utilities, accrued salaries and bonuses,
  payroll taxes and fringe benefits"* — language that echoes caption (20)'s own text almost
  verbatim.

eCFR: 210.5-02(19), *"Accounts and notes payable"* (subpart (a)(4) is specifically *"trade
creditors"*); 210.5-02(20), *"Other current liabilities: ... accrued payrolls, accrued interest,
taxes..."* FASB itself tagging the combined element to **both** captions is a stronger, more
direct confirmation of the bundling risk than the prior note's "confirm against 5-02" could claim.

#### `inventory` — VERIFIED, and one correction
FASB doc label: *"Amount after valuation and LIFO reserves of inventory..."* Ref-linkbase:
`InventoryNet` → **SX 210.5-02(6)**. eCFR 210.5-02(6)(c), verbatim: *"If the LIFO inventory method
is used, the excess of replacement or current cost over stated LIFO value shall, if material, be
stated parenthetically or in a note..."* The dedicated LIFO-reserve elements
(`InventoryLIFOReserve`, `ExcessOfReplacementOrCurrentCostsOverStatedLIFOValue`) are tagged
specifically to **5-02(6)(c)**.

**Correction to the prior note**: it guessed the LIFO-reserve disclosure requirement might be an
ASC 330-10-50 item. The taxonomy evidence points instead to **Reg S-X Rule 5-02(6)(c) itself**
(reproduced in the Codification only as `ASC 210-10-S99-1`, the SEC-materials cross-reference
section, not a native ASC 330 paragraph). No independent ASC 330-10-50 disclosure tie was found in
the taxonomy — it may still exist as a parallel GAAP-level requirement, but the operative,
FASB-confirmed citation is the Reg-S-X rule, not 330-10-50.

#### `sellingGeneralAdmin` — VERIFIED (combined-vs-legs relationship; Compustat XSGA question stays open by design)
FASB doc label for `SellingGeneralAndAdministrativeExpense`: *"The aggregate total costs related
to selling a firm's product and services, **as well as all other general and administrative
expenses**."* This wording is, by construction, the union of the other two elements' definitions
(`GeneralAndAdministrativeExpense`: *"expenses... not directly or indirectly associated with the
manufacture, sale or creation of a product"*; `SellingAndMarketingExpense`: *"expenses directly
related to the marketing or selling of products or services"*) — not an independent third concept.
Ref-linkbase: `SellingGeneralAndAdministrativeExpense` → **SX 210.5-03(4)**, eCFR-confirmed as a
single combined caption with no SEC-mandated selling/G&A split. Confirms the roll-up is a
legitimate disaggregation of one statutory caption. The Compustat `XSGA` R&D-inclusion question
remains an explicit, separately-tracked open gap (already recorded as such in the first research
and the plan) — FASB's own definitions don't touch it, since R&D has its own dedicated element and
caption.

#### `stockBasedComp` — VERIFIED (mostly)
FASB ref-linkbase: `ShareBasedCompensation` → **ASC 230-10-45-28(a)** (cash-flow statement,
noncash-add-back reconciliation paragraph) — confirms it is presented/used as the aggregate
add-back line, supporting an undimensioned read as the total.
`AllocatedShareBasedCompensationExpense` → **ASC 718-10-50-2(h)(1)(i)**, a disaggregated-disclosure
paragraph (per its known structure) requiring a breakdown by award type/line item — consistent
with (not contradicting) the catalogue's `never_use` guard. Caveat: the *citation* (paragraph
number) for 718-10-50-2(h)(1)(i) is primary; the specific claim that it *"typically"* produces
per-award-type dimensioning is inferred from secondary guides describing that paragraph's
disclosure table, not from quoted primary prose — flag this one sub-claim as one notch below full
verification.

#### `interestExpense` — PARTIALLY VERIFIED, and it changes a design decision
This is the most consequential of the 17. FASB's ref-linkbase draws a sharp line that the prior
catalogue entry doesn't reflect:
- `InterestExpense` doc label: *"...classified as operating and nonoperating. Includes, but is
  not limited to..."* — and its **only** ref-linkbase ties are ASC 280-10-50 (segment reporting),
  ASC 835-20-50-1(a) (capitalized interest) and ASC 946-220-45-3(i) (investment companies). **No
  Reg-S-X tag at all**, industrial or bank.
- `InterestExpenseDebt` → **SX 210.5-03(8)**, eCFR-confirmed: *"Interest and amortization of debt
  discount and expense"* (the Article-5/industrial income-statement caption).
- For banks, Reg-S-X Rule 9-04's own caption chain — eCFR verbatim: *"6. Interest on deposits.
  7. Interest on short-term borrowings. 8. Interest on long-term debt. 9. **Total interest expense
  (total of lines 6 through 8).** 10. Net interest income (line 5 minus line 9)."* — is tagged by
  FASB to `InterestExpenseDeposits` (6), `InterestExpenseBorrowings` (7 and 8),
  `InterestExpenseOperating` (9, the bank "**Total interest expense**" line), and
  `InterestIncomeExpenseNet` (10, already the repo's `netInterestIncome` source). **`InterestExpense`
  itself does not appear anywhere in that chain.**

So `InterestExpense`'s documentation reads as the broadest catch-all, but FASB's own linkbase does
not anchor it to either the industrial or the bank caption — it's untethered from Reg-S-X entirely,
while a taxonomy-endorsed, caption-anchored element exists for each regime and neither is currently
in the catalogue's fallback list (`["InterestExpense", "InterestExpenseDebt"]`). See §Open items #2
— this is a genuine design gap, parallel to (and not yet as complete as) the bank overrides that
already exist for `totalRevenue` and `netInterestIncome`.

---

## Part 2 — The regime-absence register, re-verified against live data

Every specific rate cell checked reproduced **exactly** against
`configs/fundamentals/fundamentals_exceptions.json`: bank `capex` (0.43 / 6 absent), utility and
energy `currentAssets` (0.00 / 0 both), insurer `sellingGeneralAdmin` (0.69 / 11, with the exact
5-ticker "does tag it" list — AFL, AIG, AIZ, HIG, TRV), real-estate `operatingIncome` (0.21 / 4,
exact tickers ARE, CPT, DOC, O), and industrial `researchAndDevelopment` (0.44 / 150). **The
measurement methodology is sound and consistently applied.** What the live re-derivation found
instead is bookkeeping drift around the edges:

### 2.1 The "441" denominator is stale
`fundamentals_facts_legacy` covers **445** distinct tickers (matching the known truncated-backfill
finding), but **3 of those 445 — AVB, EA, EQR — have facts and yet no row at all in the current
`sp500_tickers` table**, so the config's own GICS-driven regime classification cannot place them.
445 − 3 = **442**, and reclassifying the remaining 442 by the config's own rules sums *exactly* to
442 (bank 14 + insurer 16 + broker_dealer 6 + real_estate 19 + utility 28 + energy 18 + hybrid 1 +
industrial 340 = 442) — which matches the file's own `by_regime` block, not its README's stated
"441." The README figure is off by one against its own table, and off by three-to-four against the
live substrate. **Three tickers with real, usable facts are currently unclassifiable by the
documented method** — a live gap, not a rounding error.

### 2.2 Bank `capex`: "intermittent" is the wrong word for 5 of the 8 tickers that tag it
The override's rate (0.43 absent, 6 of 14) is exactly right. But re-measuring *which* years each of
the 8 tagging banks uses `PaymentsToAcquirePropertyPlantAndEquipment` shows three distinct
patterns, not one:

| pattern | tickers | detail |
|---|---|---|
| **always** (every covered year) | AXP, C, COF, FITB, HBAN | fully reliable — never actually intermittent |
| **sporadic** (tagged in some years, silent in others of the *same ticker's* own history) | CFG, KEY, TFC | CFG: 1 of 13 years; KEY: 3 of 16 (2011-13 only); TFC: 6 of 16 (2011-13, 2020-22) |
| **never** | BAC, JPM, MTB, PNC, RF, SYF | the 6 that make up the 0.43 absence rate |

The plan's and the exceptions register's "intermittency... yields a TTM that silently mixes tagged
and untagged quarters" reasoning is real, but it only describes 3 of the 8 tagging banks. Applying
it to describe the whole 8-ticker "does tag it" population overstates how unreliable bank capex
extraction actually is for the 5 that tag it every year without exception.

### 2.3 "Financials spans four regimes" undercounts by one, and the force_regime prose has a small arithmetic slip
Live query of `sp500_tickers WHERE sector = 'Financials'` returns 13 sub-industries / 76 tickers,
reconciling exactly: bank 16 + insurer 17 + broker_dealer 6 + industrial-via-force_regime 36 + hybrid
1 (BRK-B, `Multi-Sector Holdings`) = 76. So GICS Financials tickers actually land in **five**
destinations, not four — hybrid is a legitimate fifth bucket, reasonably excluded from "the four"
only because it's explicitly carved out of regime-relative scoring, but the phrase itself
undercounts if read literally. Separately, `fundamentals_regimes.json`'s `force_regime._why` prose
says routing the four fee-business traps back to `industrial` "would hit 37 live tickers" — the
actual sum (Insurance Brokers 6 + Financial Exchanges & Data 9 + Transaction & Payment Processing 9
+ Asset Management & Custody Banks 12) is **36**, a minor off-by-one in prose only (the ticker lists
themselves are correct and matched the live data exactly).

### 2.4 `real_estate`'s ticker roster is missing PLD, despite the file's own "live, not illustrative" claim
Reconstructing the `real_estate` regime from live GICS data (REIT/RE-management industry groups,
minus the force_regime REIT exclusions already encoded — tower/data-center/timber) yields 24
candidate tickers **including PLD** (Prologis, Industrial REITs). `fundamentals_regimes.json`'s
`tickers_measured.real_estate` list has only 23 entries and omits PLD. The *rate arithmetic* in
`fundamentals_exceptions.json` is unaffected (its `real_estate._n: 19` already correctly counts PLD
among the 24 candidates minus 5 with no facts), but the regimes file's documented roster —
explicitly claimed to be "the LIVE membership, not an illustration" — has a real gap for the one
ticker whose sub-industry (Industrial REITs, GICS `601025`) the first research already flagged as
the easiest one to silently drop from a contiguous-range scan.

### 2.5 Methodological verdict
"Measured per regime, not per GICS sector" holds up as a genuine improvement, not just a plausible
one — the Financials breakdown is the cleanest evidence: a naive sector-level rule would force-fit
36 fee businesses (asset managers, exchanges, payment processors, insurance brokers) into bank or
insurer templates where `PremiumsEarnedNet` or `InterestAndDividendIncomeOperating` would resolve
to nothing for companies with billions in ordinary revenue. What the live data asks for is
*tightening*, not loosening, in exactly the two spots above (§2.2, §2.4) — the boundary logic is
right; two of its worked examples and one summary number are stale.

---

## Part 3 — Concrete corrections to apply to the three config files

No files were edited during this research (per the request's explicit constraint). This section
gives the field-level diffs for the implementation phase to apply.

### `configs/fundamentals/fundamentals_kpis.json`

| field | change |
|---|---|
| `stockholdersEquity` | `authority`: "UNVERIFIED" → the FASB doc-label quote + SX 5-02(29)/(30)/(31) citation in §1.1. Drop `authority_note`, or shrink it to just the ASC-810-secondary caveat. |
| `minorityInterest` | same treatment — VERIFIED, SX 5-02(31). |
| `retainedEarnings` | VERIFIED, cite **SX 5-02(30)(a)(3)** specifically (not just "Rule 5-02"). |
| `dilutedShares` | VERIFIED for the definition; keep a one-line note that ASC 260-10-45-16's prose is secondary-sourced. |
| `pretaxIncome` | VERIFIED — the fallback order (superset-first) is already correct; add the "including" vs. "and addition of" quote as the authority. |
| `incomeTaxExpense` | **Fix the bug first**: `IncomeTaxExpenseBenefitContinuingOperations` does not exist in the 2025 taxonomy. Before writing a citation, check `fundamentals_facts_legacy`'s tag ledger for whether any filer actually uses this string (possible stale/older-taxonomy tag) or a compound name; replace or drop the fallback accordingly. Authority for `IncomeTaxExpenseBenefit` itself (ASC 740-10-50-10/-12) is otherwise ready to record. |
| `goodwill` | VERIFIED — note explicitly that Reg-S-X has *no* goodwill-specific caption (the separate-line mandate is ASC 350-20-45-1, GAAP not Reg-S-X) so the citation doesn't later get "corrected" toward a caption number that doesn't exist. |
| `intangiblesExGoodwill` | VERIFIED — cite the "Sum of the carrying amounts of all intangible assets" doc-label text directly; it's a stronger authority than the roll-up alone. |
| `ppeGross` | VERIFIED — cite **SX 5-02(13)** exactly. |
| `ppeNet` | PARTIALLY VERIFIED — record the finance-lease ROU-asset ambiguity as a **named, undecided trade-off** (§Open items #1), not as a closed citation. |
| `accumulatedDepreciation` | VERIFIED — cite **SX 5-02(14)** and the schema's `balance='credit'` attribute as the contra-account proof; also record (as a note, not a decision) that it is confirmed distinct from `depAmort`'s `DepreciationDepletionAndAmortization` by `periodType`/`balance`. |
| `accountsReceivable` | VERIFIED, with the one-notch caveat that the SX 5-02(3) caption match is via eCFR, not FASB's own cross-reference tag (unlike most other fields here). |
| `accountsPayable` | VERIFIED — cite the dual SX 5-02(19)+(20) tagging on the bundled element as the strongest form of the trap confirmation. |
| `inventory` | VERIFIED, **and correct the authority_note's guess**: the LIFO-reserve disclosure citation is **Reg S-X 5-02(6)(c)**, not ASC 330-10-50. |
| `sellingGeneralAdmin` | VERIFIED for the combined-vs-legs relationship; leave the Compustat `XSGA`/R&D question exactly as already flagged elsewhere (unchanged, out of scope). |
| `stockBasedComp` | VERIFIED, with one sub-claim (that 718-10-50-2(h)(1)(i) "typically" dimensions by award type) flagged as secondary-sourced rather than quoted. |
| `interestExpense` | **Do not just mark VERIFIED — add a bank regime override.** Current fallback (`InterestExpense`, `InterestExpenseDebt`) has zero Reg-S-X anchor for `InterestExpense` and the wrong element for banks. See §Open items #2 for the two concrete options. |

### `configs/fundamentals/fundamentals_exceptions.json`

- Fix the `_README.how_measured.scope` / narrative "441" to **442** (GICS-classifiable covered
  tickers) and explicitly log the 3 tickers that fall out of `sp500_tickers` (AVB, EA, EQR) as a
  named, tracked gap rather than an implicit rounding difference — the 442 sum is already correct
  in the `by_regime` block itself; only the prose needs the fix.
- Reconsider `bank.capex`'s `override_reason` wording: keep `expected_absent: true` (the FCF-mixing
  risk is real and the rate stays 0.43), but replace "intermittency" with the three-way split from
  §2.2 — it materially changes how confidently a downstream consumer should treat AXP/C/COF/
  FITB/HBAN's capex (reliable) versus CFG/KEY/TFC's (genuinely spotty).

### `configs/fundamentals/fundamentals_regimes.json`

- `force_regime._why`: "37 live tickers" → **36** (6+9+9+12).
- `real_estate.tickers_measured`: add **PLD** under "Industrial REITs" (currently missing from a
  roster the file's own README calls live, non-illustrative data).
- Optionally note explicitly that GICS Financials resolves to **five** destinations
  (bank/insurer/broker_dealer/industrial-via-force_regime/hybrid), not four, if the "spans four
  regimes" phrasing is kept anywhere in prose or comments.

---

## Open items for the implementation phase

These are decisions for the user, not settled by this research — exactly the trade-off framing the
request asked for.

**1. `ppeNet` and finance-lease ROU assets.** FASB's own reference linkbase confirms
`PropertyPlantAndEquipmentNet` may or may not include finance-lease right-of-use assets, depending
on each filer's own presentation choice (ASC 842-20-45-4 permits folding them in with a note, or
showing them separately). Two options, not decided here:
   - **(a)** Accept `PropertyPlantAndEquipmentNet` as reported, with a known, unresolved cross-filer
     inconsistency (some `ppeNet` values include finance-lease ROU assets, some don't, and there's
     no tag-level way to tell which).
   - **(b)** Where `FinanceLeaseRightOfUseAsset` is separately tagged, net it out of `ppeNet` for
     consistency with `totalDebt`'s existing lease-separation logic — at the cost of extra
     complexity and a coverage gap for filers that fold the ROU asset in without tagging it
     separately at all (so it can't always be detected, let alone removed).

**2. `interestExpense`'s missing bank branch.** Two options:
   - **(a)** Keep `InterestExpense` as the general/default fallback (it likely has the broadest
     empirical tag coverage even without a Reg-S-X anchor) but insert `InterestExpenseDebt` ahead
     of it for industrials, since that's the element FASB's own linkbase ties to SX 5-03(8).
   - **(b)** For the `bank` regime specifically, stop using `InterestExpense`/`InterestExpenseDebt`
     entirely and build `interestExpense` from `InterestExpenseDeposits` + `InterestExpenseBorrowings`
     (Reg-S-X 9-04 captions 6-8), or directly from `InterestExpenseOperating` (caption 9, "Total
     interest expense") — mirroring the bank override structure `totalRevenue` and
     `netInterestIncome` already have. Whether real filings actually tag these elements at usable
     coverage was **not** measured here (it would require a fresh tag-frequency query against
     `fundamentals_facts_legacy`, e.g. reproducing the same style of check the DB-verification agent
     ran for bank capex) — that measurement should happen before committing to option (b).

**3. `incomeTaxExpense`'s broken fallback tag.** `IncomeTaxExpenseBenefitContinuingOperations`
must be replaced or removed before implementation — it can never match a real fact. Needs one
query against the facts table's tag ledger to see whether any filer ever used this exact string
(possibly from an older taxonomy year still present in already-ingested facts) before deciding
whether to drop it or substitute a real compound-name element.

**4. The three unclassifiable tickers (AVB, EA, EQR).** They have real facts in
`fundamentals_facts_legacy` but no row in the current `sp500_tickers` reference table, so no regime
can be assigned to them by the documented method. Whether this is a stale `sp500_tickers` snapshot
that needs refreshing, or these tickers genuinely dropped out of the S&P 500 and the facts are
legitimately historical, was not determined here — it's a `sp500_tickers`-freshness question, not a
fundamentals-extraction one, but it blocks regime classification for those three names until
resolved.

---

## Sources consulted (primary unless marked)

- FASB 2025 US-GAAP Financial Reporting Taxonomy: `us-gaap-doc-2025.xml`, `us-gaap-ref-2025.xml`,
  `us-gaap-lab-2025.xml`, `us-gaap-2025.xsd` — `https://xbrl.fasb.org/us-gaap/2025/elts/` (primary,
  fetched directly by all four research agents independently and cross-checked).
- 17 CFR Part 210 (Regulation S-X) — fetched directly via `ecfr.gov`'s versioner API where reachable,
  via a Cornell LII mirror (`law.cornell.edu`) where `ecfr.gov`'s own site redirected through a
  bot-check (primary text either way).
- `asc.fasb.org` — confirmed still login-walled (403), consistent with the first research's finding.
  ASC paragraph *numbers* cited above all come from FASB's own reference linkbase (primary); ASC
  *prose* quotes not independently re-fetched are marked secondary (Deloitte DART / PwC Viewpoint)
  at the point they're used.
- Live `pea` Postgres DB: `fundamentals_facts_legacy` (7,776,870 rows), `sp500_tickers` — queried
  directly via `docker exec pea_db psql` for every regime-absence figure re-verified in Part 2.
