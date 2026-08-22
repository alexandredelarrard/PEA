# Fields removed by the fundamentals rebuild

**Date**: 2026-08-21
**Companion to**: [2026-08-21-fundamentals-rebuild-plan.md](../planning/active-tasks/2026-08-21-fundamentals-rebuild-plan.md)
**Purpose**: a rebuild menu. Every field the rebuild drops, with its us-gaap concept(s), measured
coverage and the evidence note that justified it originally — so any of them can be added back one
at a time, with a check in between.

---

## The numbers

`sql/schema.sql`'s `fundamentals_history` has **239 columns**. The rebuild keeps **68**:

| kept | n |
|---|---|
| Tier 1 universals | 11 |
| Tier 2 near-universal | 12 |
| Tier 3 regime-conditional | 16 |
| Tier 3R — `researchAndDevelopment`, regime-gated (re-added, decision #10) | 1 |
| Calculation inputs (`tier: 0`) | 13 |
| Derived in code | 10 |
| Keys (`ticker`, `as_of`, `fiscal_end`, `sector`, `industry_group`) | 5 |

**171 columns are removed.** Of those:

- **164 are XBRL-extracted**, 7 derived in code.
- **78 are consumed downstream today** — marked **⬤** in the tables below. These are the ones whose
  removal breaks something; the other 93 are read by nothing.
- **61 carry an evidence note** in `fundamentals_tags.py` worth preserving. That note is reproduced
  verbatim in the last column — it is usually the whole reason the field exists, and it is the thing
  that would be lost if these files were simply deleted.

Coverage is `tickers with ≥1 non-null value / 491`, measured on `fundamentals_history_legacy`
(27,602 rows) on 2026-08-21.

---

## How to add one back

Adding a field back is a **four-line change plus a check**, by design — the catalogue is the
contract, so nothing is hardcoded:

1. Add an entry to `configs/fundamentals/fundamentals_kpis.json` with `tier`, `kind`, `sign`, `definition`,
   `authority`, and either a `roll_up` or `fallback_concepts` (both are in the table below).
2. Add the column to `sql/schema.sql` and to the `Tables.fundamentals_history` spec.
3. Re-run the facts + history build for the 32-ticker slice.
4. **The check**: assert the field's ticker coverage is at least the number in the table below. If
   it comes back materially lower, the new linkbase-driven resolver disagrees with the old tag-list
   resolver — investigate before promoting it, because one of the two is wrong and the old one has
   a measured 30-56% substitutability problem.

**Do not add a field back just because a cube builder referenced it.** The research measured that
109 of the old 239 columns were read by nothing, and several of the consumed ones were consumed only
to reconstruct a total that the filer already declares. Check whether the linkbase gives you the
total directly before rebuilding its legs.

---

## Priority guidance

Ranked by signal-per-unit-of-work, not by coverage:

| rank | group | why |
|---|---|---|
| 1 | **B. Shareholder return** (3 fields) | `dividendsPaid` + `buybacks` restore payout ratio, buyback intensity, FCF coverage and total-yield. Cheapest real signal on this list — three well-behaved cash-flow tags, high coverage, no regime logic |
| 2 | **A. Working capital** (3 of 7) | `changeInReceivables/Payables/Inventory` are the whole DSO/DPO/DIO + cash-conversion-cycle + Beneish M-score family. Caveat: all three are on edgartools' own drop-list, and the accrual-vs-balance-sheet-delta distinction matters |
| 3 | **G. Special items** (9 fields) | `impairment`, `restructuring`, `litigationExpense`, `discontinuedOps` drive the earnings-quality / normalisation features. Individually episodic (`CHARGE_FLOWS` 0-fill), so coverage looks low but is correct |
| — | ~~M. `researchAndDevelopment`~~ | **RE-ADDED** by decision #10 as Tier 3R. See the plan's Phase 5 for the cross-ticker basis rules: the aggregate and ex-IPR&D concepts agree **0%** of the time (mean ratio 1.675), so one basis is mandated universe-wide and the alternative is reason-coded, never coalesced |
| 5 | **C. Banks** (15 fields) | restores NIM, efficiency ratio, provision rate, loan-to-deposit, credit metrics. Only worth it if Financials are traded as their own cohort. Note `tier1CapitalRatio` is partly unreachable — CET1 is dimensioned by `LegalEntityAxis` and `companyfacts` drops it |
| 6 | **D. Insurers / E. REITs** | combined ratio and FFO. Both need non-taxonomy definitions (Nareit FFO, Reg S-X Rule 7-04) and both cover <100 tickers. High plumbing, narrow cohort |
| 7 | **I / J / K / O detail** | tax, pension, share-count and intangible detail. Mostly feeds one or two features each |
| 8 | **H. Lease & debt ladders** | 34 fields, and the ones that matter are already folded into the new `totalDebt`. The maturity ladders were read by almost nothing — only `debtMaturity1y` and derived `debtMaturity5yTotal` |
| — | **anything not consumed today** | leave removed unless a new feature needs it |

---

## The fields

**⬤** = consumed downstream today. Blank = read by nothing.

### A. Working capital & earnings quality

*7 fields, 6 currently consumed downstream.*

| field | kind | tickers | us-gaap concept(s) | what it is / why it was there |
|---|---|---|---|---|
| `changeInPayables` ⬤ | flow | 432/491 | `IncreaseDecreaseInAccountsPayableTrade`, `IncreaseDecreaseInAccountsPayable`, `IncreaseDecreaseInAccountsPayableAndAccruedLiabilities`, `IncreaseDecreaseInAccountsPayableAndOtherOperatingLiabilities`, `IncreaseDecreaseInOtherAccountsPayableAndAccruedLiabilities` |  |
| `changeInReceivables` ⬤ | flow | 424/491 | `IncreaseDecreaseInAccountsReceivable`, `IncreaseDecreaseInReceivables`, `IncreaseDecreaseInAccountsAndOtherReceivables`, `IncreaseDecreaseInAccountsAndNotesReceivable`, `IncreaseDecreaseInAccountsReceivableAndOtherOperatingAssets` | filers tag the working-capital change under many element names; coalesce the common variants (generic `...InReceivables`, `...AndOtherReceivables`, and the combined payables-and-accrued-liabilities line) so it isn't null for most. |
| `allowanceDoubtfulAccounts` ⬤ | instant | 330/491 | `AllowanceForDoubtfulAccountsReceivableCurrent` |  |
| `changeInInventory` ⬤ | flow | 299/491 | `IncreaseDecreaseInInventories` |  |
| `provisionDoubtfulAccounts` ⬤ | flow | 217/491 | `ProvisionForDoubtfulAccounts` | trade-receivable bad-debt expense (NON-lending): a revenue-quality signal in its own right (rising = the firm is booking sales it cannot collect), kept OUT of the bank provision pool above. |
| `lifoReserve` ⬤ | instant | 81/491 | `InventoryLIFOReserve`, `ExcessOfReplacementOrCurrentCostsOverStatedLIFOValue` | ---- LIFO -> FIFO normalization (retail / industrial / refining) ---- FIFO inventory = LIFO inventory + LIFO reserve; FIFO COGS = LIFO COGS - the reserve's increase. Without it a LIFO filer's inventory days, GMROI and gross margin are not comparable to its FIFO peers. |
| `inventoryFifoReported` | instant | 15/491 | `FIFOInventoryAmount` | inventory already stated at FIFO by the filer -> the FIFO target directly, used to fill the normalization where the LIFO-basis line is missing. |

### B. Shareholder return

*4 fields, 3 currently consumed downstream.*

| field | kind | tickers | us-gaap concept(s) | what it is / why it was there |
|---|---|---|---|---|
| `dividendsPaid` ⬤ | flow | 491/491 | `PaymentsOfDividendsCommonStock`, `PaymentsOfDividends` |  |
| `buybacks` ⬤ | flow | 491/491 | `PaymentsForRepurchaseOfCommonStock` |  |
| `equityIssuance` | flow | 491/491 | `ProceedsFromIssuanceOfCommonStock` | `...AndEmployeeShareRepurchases` variant was dropped (0 of 498 filers). |
| `dividendsPerShare` ⬤ | flow | 392/491 | `CommonStockDividendsPerShareDeclared`, `CommonStockDividendsPerShareCashPaid` |  |

### C. Banks

*15 fields, 12 currently consumed downstream.*

| field | kind | tickers | us-gaap concept(s) | what it is / why it was there |
|---|---|---|---|---|
| `investmentSecurities` | instant | 322/491 | `AvailableForSaleSecuritiesDebtSecurities`, `AvailableForSaleSecurities`, `TradingSecurities`, `EquitySecuritiesFvNiCurrentAndNoncurrent` | **read by nothing today** |
| `allowanceCreditLosses` ⬤ | instant | 104/491 | `FinancingReceivableAllowanceForCreditLosses`, `LoansAndLeasesReceivableAllowance` |  |
| `provisionForCreditLosses` ⬤ | flow | 88/491 | `ProvisionForLoanLossesExpensed`, `ProvisionForLoanLeaseAndOtherLosses`, `ProvisionForLoanAndLeaseLosses`, `FinancingReceivableExcludingAccruedInterestCreditLossExpenseReversal` | LENDING credit-loss provision only. `ProvisionForDoubtfulAccounts` was removed: it is the TRADE-receivable bad-debt expense (48% of filers, of which only 43 are Financials), so coalescing it here populated a "bank" provision for 178 non-banks and fed `bank_operating_margin`. It now has its own field below. `ProvisionForCreditLossExpenseReversal` is not an element (0 of 498) -> replaced by the real |
| `netChargeOffs` ⬤ | flow | 77/491 | `FinancingReceivableExcludingAccruedInterestAllowanceForCreditLossWriteoffAfterRecovery`, `FinancingReceivableAllowanceForCreditLossWriteoffAfterRecovery`, `AllowanceForLoanAndLeaseLossesWriteoffsNet`, `FinancingReceivableAllowanceForCreditLossesWriteOffs`, `FinancingReceivableExcludingAccruedInterestAllowanceForCreditLossWriteoff`, `AllowanceForLoanAndLeaseLossesWriteOffs` | net loan charge-offs (write-offs, net of recoveries) -> realized credit losses (B3). Duration flow -> TTM-summed. NET-of-recovery tags first, gross write-offs as the fallback. The two names previously listed first do not exist in us-gaap (0 of 498 filers: `...CreditLossesWriteoffAfterRecovery`, `...CreditLossesWriteoff`), which left the column 0.6% populated -- 2.3% even within Financials. The liv |
| `htmSecurities` ⬤ | instant | 69/491 | `HeldToMaturitySecurities`, `DebtSecuritiesHeldToMaturityExcludingAccruedInterestAfterAllowanceForCreditLoss`, `DebtSecuritiesHeldToMaturityAmortizedCostAfterAllowanceForCreditLoss` | held-to-maturity book (amortized cost) + its footnote FAIR VALUE -> the OFF- balance-sheet unrealized loss = amortized - fair value (B1, the SVB blow-up). The CECL-era amortized-cost elements are added because the pre-2020 name alone left the amortized-cost leg (15% of Financials) THINNER than the fair-value leg (32%), so the difference was NaN for most banks. `...AmortizedCostAfterAllowance- ForC |
| `htmSecuritiesFairValue` ⬤ | instant | 65/491 | `HeldToMaturitySecuritiesFairValue` |  |
| `loans` ⬤ | instant | 50/491 | `LoansAndLeasesReceivableNetReportedAmount`, `FinancingReceivableExcludingAccruedInterestBeforeAllowanceForCreditLoss` | ---- Banks ---- |
| `nonaccrualLoans` ⬤ | instant | 44/491 | `FinancingReceivableRecordedInvestmentNonaccrualStatus`, `FinancingReceivableExcludingAccruedInterestNonaccrual` | non-performing (nonaccrual) loans -> forward credit-quality (B3). The CECL-era name is `...ExcludingAccruedInterestNonaccrual`; the `...NonaccrualStatus` suffix variant previously listed is not an element (0 of 498 filers). |
| `htmUnrealizedLoss` ⬤ | instant | 43/491 | `HeldToMaturitySecuritiesAccumulatedUnrecognizedHoldingLoss` | the unrecognized HTM holding LOSS as disclosed directly (42 filers) -- the same quantity as amortized-cost minus fair-value but available without needing both legs, so it fills the gap where only one of them is tagged. |
| `deposits` ⬤ | instant | 34/491 | `Deposits`, `InterestBearingDepositLiabilities` |  |
| `interestIncomeBank` | flow | 32/491 | `InterestAndDividendIncomeOperating` | ---- Banks ---- |
| `noninterestExpense` ⬤ | flow | 26/491 | `NoninterestExpense` |  |
| `tier1CapitalRatio` ⬤ | instant | 25/491 | `TierOneRiskBasedCapitalToRiskWeightedAssets`, `CommonEquityTierOneCapitalRatio` | Tier-1 risk-based ratio, falling back to the CET1 ratio (CET1 <= Tier1, so the ">11% = well-capitalised" screen stays conservative). NOTE `CommonEquityTierOneCapitalToRiskWeightedAssets` was dropped: 0 of 498. Modern CET1 is tagged with a legal-entity dimension (holdco vs bank sub) and companyfacts serves only UNDIMENSIONED facts, so it is structurally unavailable from this source -- the column st |
| `interestBearingDepositsInBanks` | instant | 24/491 | `InterestBearingDepositsInBanks` | a bank's OWN deposits held AT other banks (an ASSET -- not the `deposits` liability above). Together with `CashAndDueFromBanks` it reconstructs the cash-and-equivalents total exactly (see `fundamentals_periods.derive_bank_cash`), which is what several banks stop tagging directly mid-history. |
| `depositsDomestic` ⬤ | instant | 3/491 | `DepositsDomestic` |  |

### D. Insurers

*5 fields, 4 currently consumed downstream.*

| field | kind | tickers | us-gaap concept(s) | what it is / why it was there |
|---|---|---|---|---|
| `insuranceReserves` ⬤ | instant | 41/491 | `LiabilityForClaimsAndClaimsAdjustmentExpense`, `LiabilityForFuturePolicyBenefits` | ---- Insurance ---- |
| `claimsIncurred` ⬤ | flow | 31/491 | `PolicyholderBenefitsAndClaimsIncurredNet`, `IncurredClaimsPropertyCasualtyAndLiability` | `...IncurredHomeAndAutoAndOther` dropped (0 of 498 filers). |
| `dacAmortization` ⬤ | flow | 25/491 | `DeferredPolicyAcquisitionCostAmortizationExpense` |  |
| `deferredAcqCosts` | instant | 25/491 | `DeferredPolicyAcquisitionCosts` | singular `DeferredPolicyAcquisitionCost` dropped (0 of 498 filers). |
| `premiumsWritten` ⬤ | flow | 10/491 | `PremiumsWrittenNet` |  |

### E. REITs

*7 fields, 5 currently consumed downstream.*

| field | kind | tickers | us-gaap concept(s) | what it is / why it was there |
|---|---|---|---|---|
| `gainOnDispositions` ⬤ | flow | 491/491 | `GainLossOnSaleOfProperties`, `GainLossOnDispositionOfProperty`, `GainsLossesOnSalesOfInvestmentRealEstate` | `GainLossOnDispositionOfRealEstate` dropped (0 of 498 filers) -- the real-estate disposal gain REITs actually tag is `GainLossOnSaleOfProperties` (11.4%). |
| `realEstateImpairment` ⬤ | flow | 491/491 | `ImpairmentOfRealEstate` | REAL-ESTATE impairment write-down: NAREIT FFO excludes it alongside real-estate D&A and sale gains, so FFO needs it as an ADD-BACK (a charge flow -> 0 in a normal quarter). Without it FFO was understated in every year a REIT wrote a property down. |
| `straightLineRent` ⬤ | flow | 491/491 | `StraightLineRent` | AFFO adjustments beyond capex: non-cash straight-line rent and above/below-market lease amortization (both sparse -- 3.0% / 2.8% -- so a no-op for most REITs, but correct where disclosed). |
| `aboveBelowMarketLeaseAmort` ⬤ | flow | 491/491 | `AmortizationOfAboveAndBelowMarketLeases` |  |
| `depletionDDA` ⬤ | flow | 348/491 | `ResultsOfOperationsDepreciationDepletionAmortizationAndAccretion`, `DepreciationDepletionAndAmortization` | E&P filers report their depletion under the standard DD&A element rather than the oil&gas-supplement one (which is annual-only) -> coalesce the standard tag. |
| `realEstateNet` | instant | 25/491 | `RealEstateInvestmentPropertyNet` | ---- REITs ---- |
| `realEstateGross` | instant | 23/491 | `RealEstateInvestmentPropertyAtCost` | **read by nothing today** |

### F. Energy & utilities

*12 fields, 4 currently consumed downstream.*

| field | kind | tickers | us-gaap concept(s) | what it is / why it was there |
|---|---|---|---|---|
| `exciseTaxAdjustment` ⬤ | derived | 491/491 | *derived in code* |  |
| `regulatoryAssets` ⬤ | instant | 34/491 | `RegulatoryAssets` | ---- Utilities ---- regulatory assets/liabilities -> a consistent TOTAL pool: the combined tag when present (e.g. SO), else current + noncurrent (reconstructed in build_ticker_history so the current portion is never dropped -> needed for regulatoryAssets/totalAssets and clean-asset-base KPIs). Most utilities split the two (NEE/AEP/D/XEL). |
| `regulatoryLiabilities` | instant | 34/491 | `RegulatoryLiabilities` | **read by nothing today** |
| `regulatoryAssetsNoncurrent` | instant | 30/491 | `RegulatoryAssetsNoncurrent` | **read by nothing today** |
| `regulatoryLiabilitiesCurrent` | instant | 30/491 | `RegulatoryLiabilityCurrent` | **read by nothing today** |
| `regulatoryLiabilitiesNoncurrent` | instant | 29/491 | `RegulatoryLiabilityNoncurrent` | **read by nothing today** |
| `regulatoryAssetsCurrent` | instant | 28/491 | `RegulatoryAssetsCurrent` | **read by nothing today** |
| `exciseTaxes` | flow | 24/491 | `ExciseAndSalesTaxes` | ---- gross-to-net revenue correction ---- Excise / sales taxes collected (6.2%): 19.5% of filers tag revenue under the INCLUDING-assessed-tax element, which overstates the top line vs peers (tobacco, beverages, fuel distribution). Netted off in `_derive_history` only for the periods where the EXCLUDING element is absent, so no double deduction. |
| `explorationExpense` ⬤ | flow | 11/491 | `ExplorationExpense`, `ExplorationAbandonmentAndImpairmentExpense`, `ResultsOfOperationsExplorationExpense` | `ExplorationAbandonmentAndDryHoleCosts` dropped (0 of 498 filers). |
| `oilGasRevenue` | flow | 7/491 | `OilAndGasRevenue`, `OilAndGasSalesRevenue` | ---- Energy (oil & gas) ---- |
| `oilGasPropertyNet` ⬤ | instant | 6/491 | `OilAndGasPropertySuccessfulEffortMethodNet`, `OilAndGasPropertyFullCostMethodNet` | ---- Energy ---- |
| `oilGasPropertyGross` | instant | 4/491 | `OilAndGasPropertySuccessfulEffortMethodGross`, `OilAndGasPropertyFullCostMethodGross` | E&P filers usually tag GROSS oil&gas property (+ accumulated DD&A); net is reconstructed as gross - accumulated in build_ticker_history. |

### G. Non-recurring / special items

*9 fields, 9 currently consumed downstream.*

| field | kind | tickers | us-gaap concept(s) | what it is / why it was there |
|---|---|---|---|---|
| `impairment` ⬤ | flow | 491/491 | `AssetImpairmentCharges`, `GoodwillImpairmentLoss`, `ImpairmentOfLongLivedAssetsHeldForUse` |  |
| `restructuring` ⬤ | flow | 491/491 | `RestructuringCharges`, `RestructuringSettlementAndImpairmentProvisions` |  |
| `goodwillImpairment` ⬤ | flow | 491/491 | `GoodwillImpairmentLoss` | ---- non-recurring items: WIDEN the core-earnings normalization pool (#1) and split goodwill impairment out of the blended `impairment` pool (M&A digestion #3). All are event flows -> 0-filled + TTM-summed via CHARGE_FLOWS below. Signs: charges (litigation) are positive expenses (add back); gains / bargain-purchase / net unusual are signed (gain +, removed from core); discontinued ops is net-of-ta |
| `gainOnSaleGeneric` ⬤ | flow | 491/491 | `GainLossOnSaleOfBusiness`, `GainLossOnSaleOfPropertyPlantEquipment`, `GainLossOnDispositionOfAssets` | `GainLossOnDispositionOfAssets` (19.9%) is the GENERIC disposal line many filers tag instead of the business / PP&E specific ones -- coalesced here (one pool, so a filer reporting several never double-counts) rather than into the real-estate `gainOnDispositions` pool, which the core-earnings block sums alongside this one. |
| `litigationExpense` ⬤ | flow | 491/491 | `LitigationSettlementExpense` |  |
| `discontinuedOps` ⬤ | flow | 491/491 | `IncomeLossFromDiscontinuedOperationsNetOfTax` |  |
| `unusualItems` ⬤ | flow | 491/491 | `UnusualOrInfrequentItemNetGainLoss` |  |
| `bargainPurchaseGain` ⬤ | flow | 491/491 | `BusinessCombinationBargainPurchaseGainRecognizedAmount` |  |
| `debtExtinguishment` ⬤ | flow | 491/491 | `GainsLossesOnExtinguishmentOfDebt` |  |

### H. Leases & debt detail

*34 fields, 5 currently consumed downstream.*

| field | kind | tickers | us-gaap concept(s) | what it is / why it was there |
|---|---|---|---|---|
| `operatingLeaseAdditions` | flow | 491/491 | `RightOfUseAssetObtainedInExchangeForOperatingLeaseLiability` | ---- ASC-842 operating-lease flows ---- Operating-lease ADDITIONS (87.3%) are the operating-lease twin of the finance-lease additions already captured in `capexGlobal` -- the bigger number for retail / restaurants / airlines, so leaving it out made capacity investment asymmetric. |
| `financeLeaseAdditions` | flow | 491/491 | `RightOfUseAssetObtainedInExchangeForFinanceLeaseLiability`, `CapitalLeaseObligationsIncurred` | NON-CASH capacity added via FINANCE / capital leases (data centers, equipment) -- absent from the cash-capex line but real capacity investment (huge for MSFT ~$3-9B/q, historically AMZN). ASC-842 `RightOfUseAsset...FinanceLease` (2019+) coalesced with the pre-2019 capital-lease element (era-separated, so no double count). 0-filled (CHARGE) -> 0 when a filer uses none, so `capexGlobal` = cash capex |
| `totalAssetsExLease` ⬤ | derived | 491/491 | *derived in code* |  |
| `debtIssued` | flow | 491/491 | `ProceedsFromIssuanceOfLongTermDebt` | **read by nothing today** |
| `debtRepaid` | flow | 491/491 | `RepaymentsOfLongTermDebt` | **read by nothing today** |
| `leaseMaturity1y` | instant | 486/491 | `LesseeOperatingLeaseLiabilityPaymentsDueNextTwelveMonths`, `LesseeOperatingLeaseLiabilityPaymentsDueNextRollingTwelveMonths`, `OperatingLeasesFutureMinimumPaymentsDueCurrent` | ---- OPERATING-LEASE maturity ladder: the missing half of the refinancing wall ---- `utils/capital.py` already counts leases as DEBT, but only the lease LIABILITY was extracted -- never its maturity profile. So for a retailer, airline or restaurant chain, where the operating-lease ladder IS the wall, `debtMaturity*` described only the bond half. Each rung coalesces BOTH accounting eras, which the  |
| `leaseMaturity2y` | instant | 486/491 | `LesseeOperatingLeaseLiabilityPaymentsDueYearTwo`, `OperatingLeasesFutureMinimumPaymentsDueInTwoYears` | **read by nothing today** |
| `leaseMaturity3y` | instant | 486/491 | `LesseeOperatingLeaseLiabilityPaymentsDueYearThree`, `OperatingLeasesFutureMinimumPaymentsDueInThreeYears` | **read by nothing today** |
| `leaseMaturity4y` | instant | 486/491 | `LesseeOperatingLeaseLiabilityPaymentsDueYearFour`, `OperatingLeasesFutureMinimumPaymentsDueInFourYears` | **read by nothing today** |
| `leaseMaturity5y` | instant | 486/491 | `LesseeOperatingLeaseLiabilityPaymentsDueYearFive`, `OperatingLeasesFutureMinimumPaymentsDueInFiveYears` | **read by nothing today** |
| `leaseMaturityAfter5y` | instant | 485/491 | `LesseeOperatingLeaseLiabilityPaymentsDueAfterYearFive`, `OperatingLeasesFutureMinimumPaymentsDueThereafter` | **read by nothing today** |
| `leaseMaturityTotal` | instant | 479/491 | `LesseeOperatingLeaseLiabilityPaymentsDue`, `OperatingLeasesFutureMinimumPaymentsDue` | **read by nothing today** |
| `operatingLeaseRouAsset` ⬤ | instant | 478/491 | `OperatingLeaseRightOfUseAsset` | The right-of-use ASSET (97.6% -- the single highest-coverage element the extractor was missing). Two jobs: (a) the operating-side twin of the lease liability, which is already treated as debt in EV / leverage, and (b) it is what makes `totalAssets` JUMP at ASC-842 adoption (FY2019), a break that contaminated every assets-denominated ratio (asset growth = the FF CMA factor, asset turnover, gross pr |
| `leaseUndiscountedExcess` | instant | 470/491 | `LesseeOperatingLeaseLiabilityUndiscountedExcessAmount` | undiscounted total minus the recognised liability = the imputed lease INTEREST |
| `operatingLeaseCost` | flow | 448/491 | `OperatingLeaseCost`, `LeaseAndRentalExpense` | **read by nothing today** |
| `longTermDebtTotal` ⬤ | instant | 445/491 | `LongTermDebt`, `LongTermDebtAndCapitalLeaseObligations` | ---- universal balance sheet (cross-sector §B) ---- |
| `operatingLeasePayments` | flow | 435/491 | `OperatingLeasePayments` | cash rent actually PAID on operating leases -- the cash cost behind the ROU asset |
| `debtMaturity5yTotal` ⬤ | derived | 413/491 | *derived in code* |  |
| `debtMaturity2y` | instant | 404/491 | `LongTermDebtMaturitiesRepaymentsOfPrincipalInYearTwo` | **read by nothing today** |
| `debtMaturity3y` | instant | 403/491 | `LongTermDebtMaturitiesRepaymentsOfPrincipalInYearThree` | **read by nothing today** |
| `debtMaturity4y` | instant | 403/491 | `LongTermDebtMaturitiesRepaymentsOfPrincipalInYearFour` | **read by nothing today** |
| `operatingLeaseLiabilityNoncurrent` | instant | 402/491 | `OperatingLeaseLiabilityNoncurrent` | **read by nothing today** |
| `debtMaturity1y` ⬤ | instant | 401/491 | `LongTermDebtMaturitiesRepaymentsOfPrincipalInNextTwelveMonths`, `LongTermDebtMaturitiesRepaymentsOfPrincipalInNextRollingTwelveMonths` | ---- DEBT MATURITY WALL (instant facts, ~81% coverage per year) ---- How much principal comes due each of the next five years. `refinancing_risk` used only `shortTermDebt`, which misses a wall sitting 2-3 years out. "Rolling" variant = a filer disclosing the ladder from the balance-sheet date forward (rolling 12 months) rather than by fixed fiscal year -- a distinct 2019+ us-gaap concept, not an a |
| `debtMaturity5y` | instant | 401/491 | `LongTermDebtMaturitiesRepaymentsOfPrincipalInYearFive` | **read by nothing today** |
| `operatingLeaseLiabilityCurrent` | instant | 396/491 | `OperatingLeaseLiabilityCurrent` | **read by nothing today** |
| `debtMaturityAfter5y` | instant | 343/491 | `LongTermDebtMaturitiesRepaymentsOfPrincipalAfterYearFive` | **read by nothing today** |
| `financeLeaseRouAsset` | instant | 214/491 | `FinanceLeaseRightOfUseAsset` | **read by nothing today** |
| `financeLeaseLiabilityCurrent` | instant | 207/491 | `FinanceLeaseLiabilityCurrent` | **read by nothing today** |
| `financeLeaseLiabilityNoncurrent` | instant | 207/491 | `FinanceLeaseLiabilityNoncurrent` | **read by nothing today** |
| `commercialPaper` | instant | 170/491 | `CommercialPaper` | **read by nothing today** |
| `debtCombined` | instant | 121/491 | `DebtLongtermAndShorttermCombinedAmount` | **read by nothing today** |
| `capitalLeaseObligationCurrent` | instant | 91/491 | `CapitalLeaseObligationsCurrent` | **read by nothing today** |
| `capitalLeaseObligationNoncurrent` | instant | 90/491 | `CapitalLeaseObligationsNoncurrent` | **read by nothing today** |
| `notesPayable` | instant | 67/491 | `NotesPayable` | **read by nothing today** |

### I. Tax detail

*12 fields, 4 currently consumed downstream.*

| field | kind | tickers | us-gaap concept(s) | what it is / why it was there |
|---|---|---|---|---|
| `incomeTaxesPaid` ⬤ | flow | 483/491 | `IncomeTaxesPaidNet`, `IncomeTaxesPaid` | CASH taxes / interest actually paid (92.8% / 89.8%): the gap vs the accrual expense is the classic earnings-quality tell (a low cash tax rate on a high book rate = aggressive deferral), and cash interest is the true debt-service burden. |
| `unrecognizedTaxBenefits` ⬤ | instant | 468/491 | `UnrecognizedTaxBenefits` |  |
| `deferredTaxLiabilities` | instant | 463/491 | `DeferredIncomeTaxLiabilitiesNet`, `DeferredTaxLiabilities` | **read by nothing today** |
| `deferredIncomeTaxExpense` | flow | 459/491 | `DeferredIncomeTaxExpenseBenefit` | **read by nothing today** |
| `valuationAllowance` ⬤ | instant | 459/491 | `DeferredTaxAssetsValuationAllowance` |  |
| `currentFederalTax` | flow | 458/491 | `CurrentFederalTaxExpenseBenefit` | **read by nothing today** |
| `deferredFederalTax` | flow | 456/491 | `DeferredFederalIncomeTaxExpenseBenefit` | **read by nothing today** |
| `deferredTaxAssets` ⬤ | instant | 451/491 | `DeferredTaxAssetsNet`, `DeferredIncomeTaxAssetsNet` | ---- deferred tax / tax-aggressiveness levels ---- |
| `deferredTaxNet` | instant | 424/491 | `DeferredTaxAssetsLiabilitiesNet` | **read by nothing today** |
| `deferredTaxAssetsGross` | instant | 412/491 | `DeferredTaxAssetsGross` | ---- deferred-tax detail (near-universal, previously unmapped) ---- |
| `currentTaxExpense` | flow | 407/491 | `CurrentIncomeTaxExpenseBenefit` | ---- TAX components (currency flows -> correctly TTM-summed) ---------------------- NOTE the effective RATE lives in LATEST_DURATION_TAGS, not here: it is a ratio, and anything in EXTRA_FLOW_TAGS is TTM-summed (`ttm_a`), which would turn a 21% rate into ~84% by adding four quarters together. |
| `currentForeignTax` | flow | 403/491 | `CurrentForeignTaxExpenseBenefit` | **read by nothing today** |

### J. Pension

*8 fields, 2 currently consumed downstream.*

| field | kind | tickers | us-gaap concept(s) | what it is / why it was there |
|---|---|---|---|---|
| `pensionNetPeriodicCost` | flow | 175/491 | `DefinedBenefitPlanNetPeriodicBenefitCost` | ---- DB-pension net-periodic components (footnote, mostly annual) ---- ASU 2017-07 (effective FY2018) forced every NON-service component out of the operating subtotal. Before that they sat inside SG&A / operating income, so a filer's own operating-margin series BREAKS at adoption. `_derive_history` restates the pre-2018 half using non-service = net periodic cost - service cost. |
| `nonServicePensionCost` ⬤ | derived | 175/491 | *derived in code* |  |
| `pensionServiceCost` | flow | 160/491 | `DefinedBenefitPlanServiceCost` | **read by nothing today** |
| `pensionDeficit` ⬤ | instant | 156/491 | `PensionAndOtherPostretirementDefinedBenefitPlansLiabilitiesNoncurrent` | recognized net underfunded DB-pension/OPEB liability (POSITIVE = deficit) -> off- balance-sheet-ish leverage input (#5). NaN when the firm has no DB-plan deficit. |
| `pensionInterestCost` | flow | 156/491 | `DefinedBenefitPlanInterestCost` | **read by nothing today** |
| `pensionAmortPriorService` | flow | 153/491 | `DefinedBenefitPlanAmortizationOfPriorServiceCostCredit` | **read by nothing today** |
| `pensionAmortGainsLosses` | flow | 144/491 | `DefinedBenefitPlanAmortizationOfGainsLosses`, `DefinedBenefitPlanAmortizationOfNetGainsLosses` | **read by nothing today** |
| `pensionExpectedReturn` | flow | 91/491 | `DefinedBenefitPlanExpectedReturnOnPlanAssets` | **read by nothing today** |

### K. Share counts & per-share

*9 fields, 7 currently consumed downstream.*

| field | kind | tickers | us-gaap concept(s) | what it is / why it was there |
|---|---|---|---|---|
| `epsBasic` ⬤ | flow | 483/491 | `EarningsPerShareBasic`, `EarningsPerShareBasicAndDiluted` |  |
| `commonSharesAuthorized` ⬤ | instant | 466/491 | `CommonStockSharesAuthorized` |  |
| `antidilutiveShares` ⬤ | instant | 442/491 | `AntidilutiveSecuritiesExcludedFromComputationOfEarningsPerShareAmount` |  |
| `commonSharesIssued` ⬤ | instant | 437/491 | `CommonStockSharesIssued` | ---- OUTSTANDING ITEMS (share count is a level, not a flow) ---- `sharesOutstanding` alone cannot distinguish a buyback from a share-count restatement; issued vs authorised gives the headroom, and the antidilutive count is the overhang that never reaches diluted EPS. |
| `commonStockValue` | instant | 432/491 | `CommonStockValue` | **read by nothing today** |
| `preferredSharesAuthorized` ⬤ | instant | 398/491 | `PreferredStockSharesAuthorized` |  |
| `treasuryStock` | instant | 330/491 | `TreasuryStockValue`, `TreasuryStockCommonValue` | **read by nothing today** |
| `preferredEquity` ⬤ | instant | 313/491 | `PreferredStockValue`, `PreferredStockValueOutstanding` |  |
| `redeemableNCI` ⬤ | instant | 155/491 | `RedeemableNoncontrollingInterestEquityCarryingAmount`, `TemporaryEquityCarryingAmountAttributableToParent`, `TemporaryEquityCarryingAmount` | MEZZANINE equity: redeemable NCI / temporary equity sits between debt and common and belongs in EV alongside minority interest. |

### L. Cash & investments detail

*7 fields, 2 currently consumed downstream.*

| field | kind | tickers | us-gaap concept(s) | what it is / why it was there |
|---|---|---|---|---|
| `cashInclRestricted` ⬤ | instant | 491/491 | `CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents`, `CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalentsIncludingDisposalGroupAndDiscontinuedOperations` |  |
| `cashPeriodChange` | flow | 487/491 | `CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalentsPeriodIncreaseDecreaseIncludingExchangeRateEffect`, `CashAndCashEquivalentsPeriodIncreaseDecrease` | ---- CASH-FLOW FOOTING: the reported net change in cash -------------------------- Lets the statement be checked as published (operating + investing + financing + FX == net change) instead of trusting three independently-coalesced subtotals. Coalesces BOTH eras: the ASU-2016-18 restricted-cash-inclusive tag (474 tickers, 2015-2026) and the pre-2018 cash-only one (456 tickers, 2006-2022). |
| `marketableSecuritiesCurrent` ⬤ | instant | 189/491 | `MarketableSecuritiesCurrent`, `AvailableForSaleSecuritiesDebtSecuritiesCurrent`, `AvailableForSaleSecuritiesCurrent`, `OtherShortTermInvestments` | CURRENT marketable securities -> part of the non-operating liquid pool netted in EV. Deliberately separate from `investmentSecurities` below: for a bank or insurer the AFS/HTM book is the CORE operating asset (50-80% of assets), not excess cash, so it must never be netted against enterprise value. |
| `restrictedCashNoncurrent` | instant | 138/491 | `RestrictedCashAndCashEquivalentsNoncurrent`, `RestrictedCashNoncurrent` | **read by nothing today** |
| `longTermInvestments` | instant | 117/491 | `LongTermInvestments` | **read by nothing today** |
| `restrictedCashCurrent` | instant | 105/491 | `RestrictedCashCurrent`, `RestrictedCashAndCashEquivalentsAtCarryingValueCurrent` | **read by nothing today** |
| `cashAndShortTermInvestments` | instant | 47/491 | `CashCashEquivalentsAndShortTermInvestments` | **read by nothing today** |

### M. Income-statement detail

*14 fields, 6 currently consumed downstream.*

| field | kind | tickers | us-gaap concept(s) | what it is / why it was there |
|---|---|---|---|---|
| `accumulatedOCI` ⬤ | instant | 483/491 | `AccumulatedOtherComprehensiveIncomeLossNetOfTax` | accumulated OCI (mostly the available-for-sale securities mark) -> a large NEGATIVE value = unrealized securities losses eroding tangible capital (B1 / the 2023 SVB signal). Near-universal tag. |
| `interestPaid` | flow | 476/491 | `InterestPaidNet`, `InterestPaid` | **read by nothing today** |
| `comprehensiveIncome` ⬤ | flow | 476/491 | `ComprehensiveIncomeNetOfTax` |  |
| `otherComprehensiveIncome` | flow | 429/491 | `OtherComprehensiveIncomeLossNetOfTax` | **read by nothing today** |
| `amortizationIntangibles` ⬤ | flow | 414/491 | `AmortizationOfIntangibleAssets` |  |
| `revenuesTotal` | flow | 378/491 | `Revenues` | consolidated `Revenues` line kept separately (it is also in the totalRevenue coalesce, but the ASC-606 contract slice outranks it there) so the Financials top-line rebuild can recover it for asset managers / insurers. |
| `otherNonoperating` ⬤ | flow | 366/491 | `OtherNonoperatingIncomeExpense` |  |
| `revenueExcludingAssessedTax` | flow | 315/491 | `RevenueFromContractWithCustomerExcludingAssessedTax` | **read by nothing today** |
| `nciIncome` ⬤ | flow | 313/491 | `NetIncomeLossAttributableToNoncontrollingInterest` | income attributable to NCI (67.3%): reconciles the incl-NCI (`ProfitLoss`) and excl-NCI (`NetIncomeLoss`) bases the netIncome coalesce mixes across eras. |
| `equityMethodIncome` ⬤ | flow | 294/491 | `IncomeLossFromEquityMethodInvestments` | equity-method income (61%) inflates pre-tax/net income with NO revenue and NO cash; other non-operating (74.7%) is where ASU-2017-07 parked non-service pension cost. Both must come OUT of core operating earnings. |
| `interestIncome` | flow | 244/491 | `InvestmentIncomeInterest` | **read by nothing today** |
| `costsAndExpenses` | flow | 212/491 | `CostsAndExpenses`, `OperatingCostsAndExpenses` | "Total costs and expenses" (incl. DD&A): lets us derive operating income as revenue - this for non-financials that report no OperatingIncomeLoss line (integrated oil pre-restructuring, e.g. COP 2012-2016). `OperatingCosts- AndExpenses` is the same subtotal under the name REITs use (confirmed: REG, 2017 onward) -- REITs almost never tag `OperatingIncomeLoss`, so without it their operating line is u |
| `revenueIncludingAssessedTax` | flow | 61/491 | `RevenueFromContractWithCustomerIncludingAssessedTax` | **read by nothing today** |
| `marketingExpense` | flow | 54/491 | `MarketingAndAdvertisingExpense` | advertising / marketing spend in its own right -- a COMPONENT of SG&A, not a substitute for it (see the `sellingGeneralAdmin` note above). |

### N. Cash-flow detail

*10 fields, 2 currently consumed downstream.*

| field | kind | tickers | us-gaap concept(s) | what it is / why it was there |
|---|---|---|---|---|
| `acquisitions` ⬤ | flow | 491/491 | `PaymentsToAcquireBusinessesNetOfCashAcquired`, `PaymentsToAcquireBusinessesAndInterestInAffiliates` |  |
| `goodwillAcquired` ⬤ | flow | 491/491 | `GoodwillAcquiredDuringPeriod` |  |
| `ebitda_q` | derived | 491/491 | *derived in code* | **read by nothing today** |
| `freeCashflow_q` | derived | 491/491 | *derived in code* | **read by nothing today** |
| `balanceSheetFooting` | instant | 491/491 | `LiabilitiesAndStockholdersEquity` | ---- balance-sheet FOOTING: the reported other side of the identity ---- `LiabilitiesAndStockholdersEquity` is tagged by ALL 498 tickers over 2007-2026 and was unmapped. It is what the filer PUBLISHED as the footing, so it turns the balance-sheet identity from an inference over three separately-coalesced columns into a check against the statement itself -- and it is a fallback for `totalAssets` on |
| `investingCashFlow` | flow | 490/491 | `NetCashProvidedByUsedInInvestingActivities`, `NetCashProvidedByUsedInInvestingActivitiesContinuingOperations` | **read by nothing today** |
| `financingCashFlow` | flow | 490/491 | `NetCashProvidedByUsedInFinancingActivities`, `NetCashProvidedByUsedInFinancingActivitiesContinuingOperations` | **read by nothing today** |
| `capexGlobal` | derived | 470/491 | *derived in code* | **read by nothing today** |
| `otherInvestingCashFlow` | flow | 386/491 | `PaymentsForProceedsFromOtherInvestingActivities` | **read by nothing today** |
| `otherFinancingCashFlow` | flow | 380/491 | `ProceedsFromPaymentsForOtherFinancingActivities` | **read by nothing today** |

### O. Intangibles & other assets

*18 fields, 7 currently consumed downstream.*

| field | kind | tickers | us-gaap concept(s) | what it is / why it was there |
|---|---|---|---|---|
| `reportableSegments` ⬤ | latest | 453/491 | `NumberOfReportableSegments` | conglomerate complexity / breakup-value proxy (92.2%) |
| `intangiblesGross` ⬤ | instant | 418/491 | `FiniteLivedIntangibleAssetsGross`, `IntangibleAssetsGrossExcludingGoodwill` |  |
| `intangibleAmort4y` | instant | 417/491 | `FiniteLivedIntangibleAssetsAmortizationExpenseYearFour` | **read by nothing today** |
| `intangibleAmort2y` | instant | 416/491 | `FiniteLivedIntangibleAssetsAmortizationExpenseYearTwo` | **read by nothing today** |
| `intangibleAmort3y` | instant | 416/491 | `FiniteLivedIntangibleAssetsAmortizationExpenseYearThree` | **read by nothing today** |
| `intangibleAmort1y` | instant | 415/491 | `FiniteLivedIntangibleAssetsAmortizationExpenseNextTwelveMonths` | ---- forward INTANGIBLE-amortisation ladder (a known future earnings drag) ---- |
| `intangiblesAccumAmort` ⬤ | instant | 414/491 | `FiniteLivedIntangibleAssetsAccumulatedAmortization` |  |
| `intangibleAmort5y` | instant | 411/491 | `FiniteLivedIntangibleAssetsAmortizationExpenseYearFive` | **read by nothing today** |
| `otherAssetsNoncurrent` | instant | 406/491 | `OtherAssetsNoncurrent` | the residual buckets a balance sheet needs to foot |
| `otherLiabilitiesNoncurrent` | instant | 394/491 | `OtherLiabilitiesNoncurrent` | **read by nothing today** |
| `deferredRevenue` ⬤ | instant | 368/491 | `ContractWithCustomerLiability`, `DeferredRevenue` | deferred revenue -> a consistent TOTAL pool: the combined tag when present, else current + noncurrent (filers such as CRM report only the split parts) -- reconstructed in build_ticker_history so current-only is never mixed with total. |
| `deferredRevenueCurrent` | instant | 288/491 | `ContractWithCustomerLiabilityCurrent`, `DeferredRevenueCurrent` | **read by nothing today** |
| `remainingPerformanceObligation` ⬤ | instant | 189/491 | `RevenueRemainingPerformanceObligation` |  |
| `deferredRevenueNoncurrent` | instant | 185/491 | `ContractWithCustomerLiabilityNoncurrent`, `DeferredRevenueNoncurrent` | **read by nothing today** |
| `capitalizedSoftware` ⬤ | instant | 125/491 | `CapitalizedComputerSoftwareNet`, `CapitalizedComputerSoftwareGross` | capitalized internal-use / product software -> IT-investment proxy (AI-leverage #4) |
| `assetRetirementObligation` ⬤ | instant | 106/491 | `AssetRetirementObligation` | ---- asset-retirement obligations (energy / utilities / mining) ---- A debt-like decommissioning liability; added to the off-balance-sheet-inclusive leverage pool, not to interest-bearing debt. |
| `aroNoncurrent` | instant | 69/491 | `AssetRetirementObligationsNoncurrent` | **read by nothing today** |
| `aroCurrent` | instant | 40/491 | `AssetRetirementObligationCurrent` | **read by nothing today** |
