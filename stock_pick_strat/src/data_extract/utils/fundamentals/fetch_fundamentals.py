"""
Fetch fundamental data per ticker.

SEC EDGAR `companyfacts` (free, no key) -> genuine ~10-year point-in-time history
at QUARTERLY cadence. This is the one that lets you make size / value / quality
risk-neutral over a real backtest, because every value comes with the FILING DATE
(`filed`), so we key each row on the date the number actually became public -- no
look-ahead. Flow items are stored as trailing-twelve-month (TTM) sums of discrete
quarters (Q1-Q3 from the 10-Qs, Q4 derived as FY - Q1 - Q2 - Q3), refreshed every
quarter. Built by `build_fundamentals_history_sec()`.

(Forward P/E is NOT scraped as a today-only yfinance snapshot any more; it is
reconstructed historically from the earnings-surprise archive in
earnings_features.forward_earnings_yield, and market cap = shares x daily close.)

Output schema (same `fundamentals_history.parquet` the cube reads):
    ticker, as_of (= filing date), fiscal_end,
    totalRevenue, netIncome, grossMargins, operatingMargins, profitMargins,
    returnOnEquity, debtToEquity, ebitda, freeCashflow, operatingCashFlow,
    researchAndDevelopment, revenueGrowth, earningsGrowth, sharesOutstanding,
    stockholdersEquity,
    cash, longTermDebt, shortTermDebt, totalLiabilities, currentAssets,
    currentLiabilities, goodwill, totalAssets           (raw balance-sheet levels),
    sellingGeneralAdmin, stockBasedComp, acquisitions, interestExpense  (TTM flows),
    revenue_q, netIncome_q, ebitda_q, freeCashflow_q  (discrete single-quarter)

The raw levels / extra TTM flows above feed the refined feature families in
fundamental_features.py: distress (net-debt/EBITDA, interest coverage, current
ratio, cash/debt), S&M efficiency (SG&A intensity + operating leverage), M&A
(acquisition intensity, goodwill growth) and stock-based-comp (SBC intensity,
SBC/OCF). All are TTM/point-in-time, keyed on the SEC filing date.

IMPORTANT for size / value: SEC gives shares and equity, not market cap
(market cap needs price). We store `sharesOutstanding`; compute
marketCap = sharesOutstanding * close in the factor layer (see the companion
change to factors.py). Store equity so book/price is available too.

Run:
    python -m data.fetch_fundamentals
"""
import json
from datetime import datetime, timezone

import pandas as pd
from tqdm import tqdm

from src.constants.constants import SEC_COMPANYFACTS_URL
from src.context import Context
from src.data_extract.utils.common.sec_utils import sec_get, load_cik_mapping


# --------------------------------------------------------------------------- #
# SEC XBRL concept tags. Each logical field maps to a list of candidate us-gaap
# (or dei) tags; we take the first one a filer actually uses. Companies tag the
# same economic concept differently, so candidates matter.
# --------------------------------------------------------------------------- #
FLOW_TAGS = {   # income-statement / cash-flow items (duration facts, annual)
    # Base is built on the revenue period-ends, so a gap in revenue-tag coverage
    # truncates the WHOLE ticker after that era. Filers switch the headline-revenue
    # concept over time / by sector: banks -> RevenuesNetOfInterestExpense (~2013),
    # utilities -> RegulatedAndUnregulatedOperatingRevenue (~2014). Coalescing these
    # (lower priority, fill-only) extends the history instead of dropping it.
    "totalRevenue": ["RevenueFromContractWithCustomerExcludingAssessedTax",
                     "RevenueFromContractWithCustomerIncludingAssessedTax",
                     "Revenues", "SalesRevenueNet",
                     # pre-ASC-606 goods/services split (the only revenue tag many
                     # filers used before ~2018, e.g. WDC=Goods, VRSK=Services) ->
                     # recovers otherwise-truncated pre-2017 revenue history.
                     "SalesRevenueGoodsNet", "SalesRevenueServicesNet",
                     "SalesRevenueEnergyServices",                # utilities pre-2016 (e.g. WEC)
                     "RevenuesNetOfInterestExpense",              # banks (net revenue)
                     "RegulatedAndUnregulatedOperatingRevenue",   # utilities
                     "FoodAndBeverageRevenue",                    # restaurants pre-2016 (e.g. CMG)
                     # ---- sector TOTAL top-line tags (pre-ASC-606 / sector-specific),
                     # appended LAST (lowest priority = FILL-ONLY) so the general tags above
                     # always win where present (recent years untouched). ONLY tags that are
                     # the filer's actual TOTAL revenue are included -- rental-only and
                     # oil&gas-only COMPONENTS are deliberately EXCLUDED: they wrongly win a
                     # small value for mixed filers (an asset manager consolidating an oil&gas
                     # portfolio co -> $22M; a self-storage REIT whose income isn't operating-
                     # lease-classified -> $52M). These stay sector-recoverable via `Revenues`.
                     "HealthCareOrganizationRevenue",             # health care total (e.g. DVA pre-2015)
                     "HealthCareOrganizationPatientServiceRevenueLessProvisionForBadDebts",
                     "HealthCareOrganizationPatientServiceRevenue",
                     "RealEstateRevenueNet",                      # REIT total real-estate revenue
                     # pure-play sector TOTAL top-lines a filer tags INSTEAD of `Revenues` in some
                     # eras (verified per (ticker, quarter) against companyfacts as the LARGEST
                     # revenue line = de-facto total). Fill-only (lowest priority) so they only fire
                     # for a pure-play that reports no general revenue tag (a diversified filer that
                     # reports `Revenues` is untouched, and these never win as a component there):
                     "RefiningAndMarketingRevenue",               # refiners (e.g. VLO)
                     "UtilityRevenue", "ElectricUtilityRevenue",  # regulated utilities (e.g. DTE/ETR/AES)
                     "RevenueMineralSales",                       # miners (e.g. NEM/FCX)
                     "GasGatheringTransportationMarketingAndProcessingRevenue"],  # midstream (e.g. TRGP)
    # NetIncomeLoss (to parent) / ProfitLoss (incl. NCI) are the modern tags; many
    # filers used neither before ~2016 (e.g. WAT 2011-2015) and instead tagged only
    # continuing-ops-incl-NCI. That tag shares ProfitLoss's basis (incl. NCI), so
    # coalescing it at LOWER priority (fill-only) recovers otherwise-truncated early
    # netIncome without altering filers that report the primary tags. The net-income-
    # TO-COMMON tag is added conditionally per filer in `_extract_all` (see
    # `_net_income_tags`) because it is net of preferred dividends and would corrupt
    # preferred-paying REIT/bank/insurer history if coalesced unconditionally.
    "netIncome": ["NetIncomeLoss", "ProfitLoss",
                  "IncomeLossFromContinuingOperationsIncludingPortionAttributableToNoncontrollingInterest"],
    "grossProfit": ["GrossProfit"],
    # Filers used CostOfGoodsSold / CostOfServices before the ASC-606-era
    # consolidation onto CostOfGoodsAndServicesSold (e.g. LLY tags the latter only
    # from ~2018; CostOfGoodsSold covers the earlier years). Coalescing recovers
    # the early cost line -> early gross profit -> gross/operating margin + EBITDA.
    "costOfRevenue": ["CostOfGoodsAndServicesSold", "CostOfRevenue",
                      "CostOfGoodsSold", "CostOfServices",
                      # excl-D&A variants: the headline cost line for many services /
                      # utilities (e.g. CTSH, AEP); slightly understate COGS vs incl-D&A
                      # filers, so kept at LOWER priority (fill-only).
                      "CostOfGoodsAndServiceExcludingDepreciationDepletionAndAmortization",
                      "CostOfServicesExcludingDepreciationDepletionAndAmortization",
                      # goods-only excl-D&A variant: the pre-2018 cost line for several
                      # manufacturers / telecom (e.g. PPG, OXY, HWM, T) -> recovers their
                      # otherwise-missing pre-2017 gross margin.
                      "CostOfGoodsSoldExcludingDepreciationDepletionAndAmortization",
                      # real-estate cost of revenue: homebuilders (PHM/DHI cost of homes
                      # sold, pre-2019) and residential REITs (CPT/DOC property operating
                      # cost -> NOI-style margin) tag their COGS here, not under the goods/
                      # services elements. Fill-only, so standard filers are untouched.
                      "CostOfRealEstateRevenue", "CostsOfRealEstateServicesAndLandSales",
                      # restaurant food/beverage cost of sales (company-operated, e.g. CMG).
                      "FoodAndBeverageCostOfSales",
                      # regulated-utility cost of revenue (fuel + purchased power / gas): the
                      # complete electric/energy cost line filers tag instead of the goods/
                      # services elements (e.g. PNW/AES electric, DTE energy, PEG services).
                      "CostOfGoodsSoldElectric", "CostOfGoodsAndServicesEnergyCommoditiesAndServices",
                      "CostOfServicesEnergyServices", "CostOfDomesticRegulatedElectric"],
    "operatingIncome": ["OperatingIncomeLoss", "OperatingAndNonoperatingRevenues", "OperatingLoss", "OperatingIncome"],
    "depAmort": ["DepreciationDepletionAndAmortization",
                 "DepreciationAmortizationAndAccretionNet",
                 "DepreciationAndAmortization",
                 "DepreciationAndAmortizationRealEstate",
                 "Depreciation",   # many REITs (e.g. EQR) tag plain Depreciation only
                 # sector TOTAL D&A embedded in operating expense / COGS — utilities & miners tag
                 # their whole D&A here rather than the cash-flow element (fill-only, lowest
                 # priority, so a filer with the standard tag is untouched): AEP/PPL utilities, PEG,
                 # FCX (mine depletion in COGS ~ its total D&A). Verified as the filer's total D&A.
                 "UtilitiesOperatingExpenseDepreciationAndAmortization",
                 "CostOfGoodsAndServicesSoldDepreciationAndAmortization",
                 "CostOfGoodsSoldDepreciationDepletionAndAmortization"],
    "operatingCashFlow": ["NetCashProvidedByUsedInOperatingActivities",
                          "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment",
              "PaymentsToAcquireProductiveAssets",
              # pure E&P tag capital spend as oil&gas property, not generic PP&E
              # (e.g. EOG, FANG) -> needed for AFFO / FCF / capital-intensity.
              "PaymentsToAcquireOilAndGasProperty",
              "PaymentsToAcquireOilAndGasPropertyAndEquipment",
              # REITs tag recurring/maintenance capex here (not generic PP&E) -> AFFO =
              # FFO - recurring capex. Fill-only (growth capex acquire/develop excluded).
              "PaymentsForCapitalImprovements",
              # operating PP&E capex variants some non-REIT filers tag instead of the generic
              # element (fill-only, verified TOTAL-scale on real data): the "Other PP&E" line
              # (ADP ~$44M/q, EA, LLY ~$478M/q, GRMN) and the machinery line. Deliberately NOT
              # added: REIT growth capex (PaymentsToAcquireRealEstate / ...DevelopRealEstateAssets)
              # -- REITs stay maintenance-only for AFFO; the non-cash accrual
              # `CapitalExpendituresIncurredButNotYetPaid`; insurer investment-real-estate; and
              # `PaymentsForConstructionInProcess` -- a utility CWIP COMPONENT (~$85M vs AEP's true
              # ~$1B+/q), so it would understate total capex.
              "PaymentsToAcquireOtherPropertyPlantAndEquipment",
              "PaymentsToAcquireMachineryAndEquipment"],
    "researchAndDevelopment": ["ResearchAndDevelopmentExpense",
                               "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost"],
    # ---- added for refined features (S&M efficiency, M&A, SBC, distress) ----
    "sellingGeneralAdmin": ["SellingGeneralAndAdministrativeExpense",
                            "GeneralAndAdministrativeExpense",
                            "SellingAndMarketingExpense",
                            "MarketingAndAdvertisingExpense"],
    "stockBasedComp": ["ShareBasedCompensation",
                       "AllocatedShareBasedCompensationExpense"],
    "acquisitions": ["PaymentsToAcquireBusinessesNetOfCashAcquired",
                     "PaymentsToAcquireBusinessesAndInterestInAffiliates"],
    "interestExpense": ["InterestExpense", "InterestAndDebtExpense",
                        "InterestExpenseNonoperating"],
}
STOCK_TAGS = {  # balance-sheet items (instant facts, point-in-time)
    "stockholdersEquity": ["StockholdersEquity",
                           "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "totalLiabilities": ["Liabilities"],
    # many filers (e.g. HD, MU pre-2020) tag long-term debt only under the
    # capital-lease-inclusive element — coalesce it so LTD isn't ~null for them.
    "longTermDebt": ["LongTermDebtNoncurrent", "LongTermDebt",
                     "LongTermDebtAndCapitalLeaseObligations"],
    # clean unrestricted cash first; then bank / plain-`Cash` variants; the
    # restricted-inclusive and cash+ST-investment totals are last-resort fallbacks
    # for insurers / asset managers (e.g. AIG, ALL) that omit the clean tag.
    "cash": ["CashAndCashEquivalentsAtCarryingValue",
             # disc-ops-inclusive variant of the primary cash line (e.g. FISV mid-divestiture) --
             # same concept, so right after the primary.
             "CashAndCashEquivalentsAtCarryingValueIncludingDiscontinuedOperations",
             "CashAndDueFromBanks", "Cash",
             "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
             # disc-ops variant of the restricted-inclusive fallback (e.g. PACCAR) -- kept next to it
             # (last-resort, restricted-inclusive slightly overstates unrestricted cash).
             "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalentsIncludingDisposalGroupAndDiscontinuedOperations",
             "CashCashEquivalentsAndShortTermInvestments",
             # cash-EQUIVALENTS-only line some REITs tag as their cash (e.g. O); fill-only last resort.
             "CashEquivalentsAtCarryingValue"],
    # ---- added for refined features (distress / liquidity, M&A footprint) ----
    "shortTermDebt": ["DebtCurrent", "LongTermDebtCurrent", "ShortTermBorrowings",
                      "CommercialPaper", "OtherShortTermBorrowings"],
    "currentAssets": ["AssetsCurrent"],
    "currentLiabilities": ["LiabilitiesCurrent"],
    "goodwill": ["Goodwill"],
    "totalAssets": ["Assets"],
}
SHARES_TAGS = {  # tried under dei first, then us-gaap
    "sharesOutstanding": ["EntityCommonStockSharesOutstanding",
                          "CommonStockSharesOutstanding", "WeightedAverageNumberOfDilutedSharesOutstanding"],
}

# --------------------------------------------------------------------------- #
# EXPANDED coverage. companyfacts already returns every tag a filer reports, so
# extracting more concepts costs no extra API calls — non-applicable tags are
# simply null for a company. EXTRA_FLOW_TAGS are duration facts (TTM-summed like
# the curated flows); EXTRA_STOCK_TAGS are instant balance-sheet levels. Sector
# line items are mixed in here (banks/insurance/REIT/energy/utility only report
# their own) and are turned into ratios sector-relatively in data_aggregate.
# --------------------------------------------------------------------------- #
EXTRA_FLOW_TAGS = {
    # ---- universal income statement / cash flow (cross-sector §B) ----
    "incomeTaxExpense": ["IncomeTaxExpenseBenefit"],
    # consolidated `Revenues` line kept separately (it is also in the totalRevenue
    # coalesce, but the ASC-606 contract slice outranks it there) so the Financials
    # top-line rebuild can recover it for asset managers / insurers.
    "revenuesTotal": ["Revenues"],
    "pretaxIncome": [
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest"],
    # "Total costs and expenses" (incl. DD&A): lets us derive operating income as
    # revenue - this for non-financials that report no OperatingIncomeLoss line
    # (integrated oil pre-restructuring, e.g. COP 2012-2016).
    "costsAndExpenses": ["CostsAndExpenses"],
    "interestIncome": ["InvestmentIncomeInterest"],
    "amortizationIntangibles": ["AmortizationOfIntangibleAssets"],
    "dividendsPaid": ["PaymentsOfDividendsCommonStock", "PaymentsOfDividends"],
    "buybacks": ["PaymentsForRepurchaseOfCommonStock",
                 "PaymentsForRepurchaseOfCommonStockAndEmployeeShareRepurchases"],
    "equityIssuance": ["ProceedsFromIssuanceOfCommonStock"],
    "debtIssued": ["ProceedsFromIssuanceOfLongTermDebt"],
    "debtRepaid": ["RepaymentsOfLongTermDebt"],
    "investingCashFlow": ["NetCashProvidedByUsedInInvestingActivities",
                          "NetCashProvidedByUsedInInvestingActivitiesContinuingOperations"],
    "financingCashFlow": ["NetCashProvidedByUsedInFinancingActivities",
                          "NetCashProvidedByUsedInFinancingActivitiesContinuingOperations"],
    "changeInInventory": ["IncreaseDecreaseInInventories"],
    # filers tag the working-capital change under many element names; coalesce the
    # common variants (generic `...InReceivables`, `...AndOtherReceivables`, and the
    # combined payables-and-accrued-liabilities line) so it isn't null for most.
    "changeInReceivables": ["IncreaseDecreaseInAccountsReceivable",
                            "IncreaseDecreaseInReceivables",
                            "IncreaseDecreaseInAccountsAndOtherReceivables",
                            "IncreaseDecreaseInAccountsAndNotesReceivable",
                            "IncreaseDecreaseInAccountsReceivableAndOtherOperatingAssets"],
    "changeInPayables": ["IncreaseDecreaseInAccountsPayableTrade", "IncreaseDecreaseInAccountsPayable",
                         "IncreaseDecreaseInAccountsPayableAndAccruedLiabilities",
                         "IncreaseDecreaseInAccountsPayableAndOtherOperatingLiabilities",
                         "IncreaseDecreaseInOtherAccountsPayableAndAccruedLiabilities"],
    "impairment": ["AssetImpairmentCharges", "GoodwillImpairmentLoss",
                   "ImpairmentOfLongLivedAssetsHeldForUse"],
    "restructuring": ["RestructuringCharges", "RestructuringSettlementAndImpairmentProvisions"],
    # ---- non-recurring items: WIDEN the core-earnings normalization pool (#1) and
    # split goodwill impairment out of the blended `impairment` pool (M&A digestion #3).
    # All are event flows -> 0-filled + TTM-summed via CHARGE_FLOWS below. Signs:
    # charges (litigation) are positive expenses (add back); gains / bargain-purchase /
    # net unusual are signed (gain +, removed from core); discontinued ops is net-of-tax
    # income (removed from core directly).
    "goodwillImpairment": ["GoodwillImpairmentLoss"],
    "gainOnSaleGeneric": ["GainLossOnSaleOfBusiness",
                          "GainLossOnSaleOfPropertyPlantEquipment"],
    "litigationExpense": ["LitigationSettlementExpense"],
    "discontinuedOps": ["IncomeLossFromDiscontinuedOperationsNetOfTax"],
    "unusualItems": ["UnusualOrInfrequentItemNetGainLoss"],
    "bargainPurchaseGain": ["BusinessCombinationBargainPurchaseGainRecognizedAmount"],
    # ---- Banks ----
    "interestIncomeBank": ["InterestAndDividendIncomeOperating"],
    "netInterestIncome": ["InterestIncomeExpenseNet", "InterestIncomeExpenseAfterProvisionForLoanLoss"],
    "provisionForCreditLosses": ["ProvisionForLoanLossesExpensed",
                                 "ProvisionForLoanLeaseAndOtherLosses", "ProvisionForDoubtfulAccounts",
                                 # CECL-era element used by many banks from 2020
                                 "ProvisionForLoanAndLeaseLosses", "ProvisionForCreditLossExpenseReversal"],
    "noninterestIncome": ["NoninterestIncome"],
    "noninterestExpense": ["NoninterestExpense"],
    # net loan charge-offs (write-offs, net of recoveries) -> realized credit losses
    # (B3). Duration flow -> TTM-summed. Net tag preferred, gross write-offs fallback.
    "netChargeOffs": ["FinancingReceivableAllowanceForCreditLossesWriteoffAfterRecovery",
                      "FinancingReceivableAllowanceForCreditLossesWriteoff",
                      "AllowanceForLoanAndLeaseLossesWriteOffs",
                      "AllowanceForLoanAndLeaseLossesWriteoffsNet"],
    # ---- Insurance ----
    "premiumsEarned": ["PremiumsEarnedNet", "PremiumsEarnedNetPropertyAndCasualty"],
    "premiumsWritten": ["PremiumsWrittenNet"],
    "claimsIncurred": ["PolicyholderBenefitsAndClaimsIncurredNet",
                       "IncurredClaimsPropertyCasualtyAndLiability",
                       "PolicyholderBenefitsAndClaimsIncurredHomeAndAutoAndOther"],
    "netInvestmentIncome": ["NetInvestmentIncome"],
    "dacAmortization": ["DeferredPolicyAcquisitionCostAmortizationExpense"],
    # ---- REITs ----
    "rentalIncome": ["OperatingLeaseLeaseIncome", "OperatingLeasesIncomeStatementLeaseRevenue",
                     "RealEstateRevenueNet", "LeaseIncome"],
    "gainOnDispositions": ["GainLossOnDispositionOfRealEstate",
                           "GainLossOnSaleOfProperties",
                           "GainLossOnDispositionOfProperty",
                           "GainsLossesOnSalesOfInvestmentRealEstate"],
    # ---- Energy (oil & gas) ----
    "oilGasRevenue": ["OilAndGasRevenue", "OilAndGasSalesRevenue"],  # E&P top line (pure players / pre-ASC-606)
    "explorationExpense": ["ExplorationExpense", "ExplorationAbandonmentAndDryHoleCosts",
                           "ExplorationAbandonmentAndImpairmentExpense",   # DVN/EQT/OXY
                           "ResultsOfOperationsExplorationExpense"],
    # E&P filers report their depletion under the standard DD&A element rather than
    # the oil&gas-supplement one (which is annual-only) -> coalesce the standard tag.
    "depletionDDA": ["ResultsOfOperationsDepreciationDepletionAmortizationAndAccretion",
                     "DepreciationDepletionAndAmortization"],
}

EXTRA_STOCK_TAGS = {
    # ---- universal balance sheet (cross-sector §B) ----
    "longTermDebtTotal": ["LongTermDebt", "LongTermDebtAndCapitalLeaseObligations"],
    "debtCombined": ["DebtLongtermAndShorttermCombinedAmount"],   # single ST+LT total (banks/insurers)
    "notesPayable": ["NotesPayable"],                             # REIT total-debt fallback
    "commercialPaper": ["CommercialPaper"],
    "operatingLeaseLiability": ["OperatingLeaseLiability"],
    "financeLeaseLiability": ["FinanceLeaseLiabilityNoncurrent", "FinanceLeaseLiability"],
    "accountsReceivable": ["AccountsReceivableNetCurrent", "ReceivablesNetCurrent"],
    "inventory": ["InventoryNet"],
    "accountsPayable": ["AccountsPayableCurrent", "AccountsPayableAndAccruedLiabilitiesCurrent"],
    # net PP&E is the common balance-sheet line; gross is reconstructed as
    # net + accumulated depreciation in build_ticker_history for net-only filers.
    "ppeNet": ["PropertyPlantAndEquipmentNet"],
    "ppeGross": ["PropertyPlantAndEquipmentGross"],
    "accumulatedDepreciation": ["AccumulatedDepreciationDepletionAndAmortizationPropertyPlantAndEquipment"],
    "intangiblesExGoodwill": ["IntangibleAssetsNetExcludingGoodwill", "FiniteLivedIntangibleAssetsNet"],
    # capitalized internal-use / product software -> IT-investment proxy (AI-leverage #4)
    "capitalizedSoftware": ["CapitalizedComputerSoftwareNet", "CapitalizedComputerSoftwareGross"],
    # recognized net underfunded DB-pension/OPEB liability (POSITIVE = deficit) -> off-
    # balance-sheet-ish leverage input (#5). NaN when the firm has no DB-plan deficit.
    "pensionDeficit": ["PensionAndOtherPostretirementDefinedBenefitPlansLiabilitiesNoncurrent"],
    "retainedEarnings": ["RetainedEarningsAccumulatedDeficit"],
    "treasuryStock": ["TreasuryStockValue", "TreasuryStockCommonValue"],
    "preferredEquity": ["PreferredStockValue", "PreferredStockValueOutstanding"],
    "minorityInterest": ["MinorityInterest"],
    # accumulated OCI (mostly the available-for-sale securities mark) -> a large
    # NEGATIVE value = unrealized securities losses eroding tangible capital
    # (B1 / the 2023 SVB signal). Near-universal tag.
    "accumulatedOCI": ["AccumulatedOtherComprehensiveIncomeLossNetOfTax"],
    "shortTermInvestments": ["ShortTermInvestments"],
    "longTermInvestments": ["LongTermInvestments"],
    # deferred revenue -> a consistent TOTAL pool: the combined tag when present,
    # else current + noncurrent (filers such as CRM report only the split parts) --
    # reconstructed in build_ticker_history so current-only is never mixed with total.
    "deferredRevenue": ["ContractWithCustomerLiability", "DeferredRevenue"],
    "deferredRevenueCurrent": ["ContractWithCustomerLiabilityCurrent", "DeferredRevenueCurrent"],
    "deferredRevenueNoncurrent": ["ContractWithCustomerLiabilityNoncurrent", "DeferredRevenueNoncurrent"],
    "remainingPerformanceObligation": ["RevenueRemainingPerformanceObligation"],
    # ---- Banks ----
    "loans": ["LoansAndLeasesReceivableNetReportedAmount",
              "FinancingReceivableExcludingAccruedInterestBeforeAllowanceForCreditLoss"],
    "deposits": ["Deposits", "InterestBearingDepositLiabilities"],
    "depositsDomestic": ["DepositsDomestic"],   # deposit-stickiness metric (sparsely tagged)
    "allowanceCreditLosses": ["FinancingReceivableAllowanceForCreditLosses",
                              "LoansAndLeasesReceivableAllowance"],
    # held-to-maturity book (amortized cost) + its footnote FAIR VALUE -> the OFF-
    # balance-sheet unrealized loss = amortized - fair value (B1, the SVB blow-up).
    "htmSecurities": ["HeldToMaturitySecurities",
                      "HeldToMaturitySecuritiesAmortizedCostAfterAllowanceForCreditLoss"],
    "htmSecuritiesFairValue": ["HeldToMaturitySecuritiesFairValue"],
    # non-performing (nonaccrual) loans -> forward credit-quality (B3)
    "nonaccrualLoans": ["FinancingReceivableRecordedInvestmentNonaccrualStatus",
                        "FinancingReceivableExcludingAccruedInterestNonaccrualStatus"],
    # Tier-1 risk-based ratio; fall back to the modern CET1 ratio for banks that report
    # only CET1 (post-2015). Both are capital-adequacy ratios (>11% = well-capitalised);
    # CET1 <= Tier1, so the >11% screen stays conservative.
    "tier1CapitalRatio": ["TierOneRiskBasedCapitalToRiskWeightedAssets",
                          "CommonEquityTierOneCapitalToRiskWeightedAssets",
                          "CommonEquityTierOneCapitalRatio"],
    # ---- Insurance ----
    "insuranceReserves": ["LiabilityForClaimsAndClaimsAdjustmentExpense",
                          "LiabilityForFuturePolicyBenefits"],
    "deferredAcqCosts": ["DeferredPolicyAcquisitionCost", "DeferredPolicyAcquisitionCosts"],
    # ---- REITs ----
    "realEstateNet": ["RealEstateInvestmentPropertyNet"],
    "realEstateGross": ["RealEstateInvestmentPropertyAtCost"],
    # ---- Energy ----
    "oilGasPropertyNet": ["OilAndGasPropertySuccessfulEffortMethodNet",
                          "OilAndGasPropertyFullCostMethodNet"],
    # E&P filers usually tag GROSS oil&gas property (+ accumulated DD&A); net is
    # reconstructed as gross - accumulated in build_ticker_history.
    "oilGasPropertyGross": ["OilAndGasPropertySuccessfulEffortMethodGross",
                            "OilAndGasPropertyFullCostMethodGross"],
    # ---- Utilities ----
    # regulatory assets/liabilities -> a consistent TOTAL pool: the combined tag when
    # present (e.g. SO), else current + noncurrent (reconstructed in build_ticker_history
    # so the current portion is never dropped -> needed for regulatoryAssets/totalAssets
    # and clean-asset-base KPIs). Most utilities split the two (NEE/AEP/D/XEL).
    "regulatoryAssets": ["RegulatoryAssets"],
    "regulatoryAssetsCurrent": ["RegulatoryAssetsCurrent"],
    "regulatoryAssetsNoncurrent": ["RegulatoryAssetsNoncurrent"],
    "regulatoryLiabilities": ["RegulatoryLiabilities"],
    "regulatoryLiabilitiesCurrent": ["RegulatoryLiabilityCurrent"],
    "regulatoryLiabilitiesNoncurrent": ["RegulatoryLiabilityNoncurrent"],
}

# diluted weighted-average shares (duration fact; we take the latest period's
# value point-in-time) -> true per-share + net-issuance signals
DILUTED_SHARES_TAGS = ["WeightedAverageNumberOfDilutedSharesOutstanding",
                       "WeightedAverageNumberOfDilutedSharesOutstandingAdjustment"]

# Event / charge / financing flows where a quarter with NO reported tag means the
# event did not occur (= 0), not "missing" — reported only in periods it happens,
# and often as year-to-date cumulatives that never yield 4 discrete quarters, so a
# plain TTM sum stays NaN forever. These are 0-filled within the ticker's reporting
# span before the TTM so they become usable features (e.g. TTM impairment = 0 in
# normal years, non-zero after a write-off). Continuous operating flows (revenue,
# NII, premiums, …) are NOT here: their absence means sector-N/A or a coverage gap.
CHARGE_FLOWS = {"impairment", "restructuring", "acquisitions", "buybacks",
                "equityIssuance", "debtIssued", "debtRepaid", "dividendsPaid",
                "gainOnDispositions",
                # widened non-recurring pool (see EXTRA_FLOW_TAGS): 0 in a normal
                # quarter, non-zero only when the event occurs.
                "goodwillImpairment", "gainOnSaleGeneric", "litigationExpense",
                "discontinuedOps", "unusualItems", "bargainPurchaseGain"}

ANNUAL_MIN_DAYS, ANNUAL_MAX_DAYS = 340, 380   # accept a fiscal year as ~365d
QUARTER_MIN_DAYS, QUARTER_MAX_DAYS = 80, 100   # accept a fiscal quarter as ~13 weeks
TTM_QUARTERS = 4                               # trailing-twelve-months = 4 quarters
# Gross margin is mathematically <= 1 (a value > 1 implies negative COGS); a value
# below -200% only arises when revenue is truncated / period-mismatched vs the cost
# line (e.g. a REIT whose rental income moved to the ASC-842 lease-income tag). Values
# outside this band are nulled as extraction artifacts rather than shipped as features.
GROSS_MARGIN_MIN, GROSS_MARGIN_MAX = -2.0, 1.0


def _today_iso() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _history_meta_path(context: Context):
    return context.paths["FUNDAMENTALS_HISTORY_PATH"].with_name(
        "fundamentals_history_meta.json",
    )


def _load_history_meta(context: Context) -> dict | None:
    path = _history_meta_path(context)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _save_history_meta(
    context: Context,
    history: pd.DataFrame,
    universe_size: int,
) -> None:
    meta = {
        "last_built": _today_iso(),
        "row_count": len(history),
        "ticker_count": int(history["ticker"].nunique()),
        "universe_size": universe_size,
    }
    _history_meta_path(context).write_text(
        json.dumps(meta, indent=2),
        encoding="utf-8",
    )


def _load_existing_history(context: Context) -> pd.DataFrame | None:
    df = context.store.load("fundamentals_history")
    return None if df.empty else df


def _is_sec_history_up_to_date(context: Context, cik_mapping: pd.DataFrame) -> bool:
    """True when today's run already attempted the full CIK universe."""
    meta = _load_history_meta(context)
    if meta is None or meta.get("last_built") != _today_iso():
        return False
    if not context.store.exists("fundamentals_history"):
        return False
    return meta.get("universe_size", 0) >= len(cik_mapping)


def _tickers_to_process(
    context: Context,
    cik_mapping: pd.DataFrame,
    existing: pd.DataFrame | None,
) -> pd.DataFrame:
    """Return CIK rows that still need SEC history fetched."""
    if existing is None or existing.empty:
        return cik_mapping

    meta = _load_history_meta(context)
    if meta and meta.get("last_built") == _today_iso():
        have = set(existing["ticker"].unique())
        return cik_mapping[~cik_mapping["ticker"].isin(have)]

    # New calendar day (or no meta): refresh all tickers for new filings.
    return cik_mapping


# --------------------------------------------------------------------------- #
# SEC companyfacts fetch (cached)                                             #
# --------------------------------------------------------------------------- #
def _fetch_companyfacts(context: Context, cik: str) -> dict | None:
    cache_dir = context.paths["SEC_BULK_CACHE_DIR"]
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"companyfacts_CIK{cik}.json"

    if context.use_cache and cache.exists():
        try:
            return json.loads(cache.read_text(encoding="utf-8"))
        except Exception:
            pass
    try:
        resp = sec_get(SEC_COMPANYFACTS_URL.format(cik=cik))
        data = resp.json()
        cache.write_text(json.dumps(data), encoding="utf-8")
        return data
    except Exception as e:
        print(f"companyfacts CIK{cik}: failed ({e})")
        return None


# --------------------------------------------------------------------------- #
# Concept extraction                                                          #
# --------------------------------------------------------------------------- #
_CONCEPT_COLS = ["end", "start", "filed", "form", "fp", "val"]


def _extract_concept(section: dict, tag_candidates: list[str]) -> pd.DataFrame:
    """
    Observations for a logical concept as [end, start, filed, form, fp, val],
    COALESCED across all candidate tags rather than taking only the first one.

    Filers split the same economic concept across tags — over time (ASC-606
    revenue: `Revenues` pre-2018, `RevenueFromContractWithCustomer…` after) and
    by scope (`NetIncomeLoss` excl. NCI vs `ProfitLoss` incl. NCI; equity with /
    without NCI). Taking only the first present tag truncated history badly
    (e.g. CVX revenue from 2018, AVGO/CAT net income near-empty, JNJ/UNH ROE
    only a few years). We therefore union every candidate and, PER PERIOD
    (start, end), keep the highest-priority (earliest-listed) candidate that
    reported it — retaining all of that candidate's filings so the downstream
    earliest-disclosure / point-in-time logic is unchanged.
    """
    frames = []
    for prio, tag in enumerate(tag_candidates):
        if tag not in section:
            continue
        units = section[tag].get("units", {})
        unit_key = next((u for u in ("USD", "shares") if u in units),
                        next(iter(units), None))
        if unit_key is None:
            continue
        rows = [{"end": o.get("end"), "start": o.get("start"),
                 "filed": o.get("filed"), "form": o.get("form"),
                 "fp": o.get("fp"), "val": o.get("val"), "_prio": prio}
                for o in units[unit_key]]
        if rows:
            frames.append(pd.DataFrame(rows))
    if not frames:
        return pd.DataFrame(columns=_CONCEPT_COLS)

    df = pd.concat(frames, ignore_index=True)
    for c in ("end", "start", "filed"):
        df[c] = pd.to_datetime(df[c], errors="coerce")
    # per (start, end) period, keep only the best-priority candidate's rows
    df["_min_prio"] = df.groupby(["start", "end"], dropna=False)["_prio"].transform("min")
    df = df[df["_prio"] == df["_min_prio"]]
    return df[_CONCEPT_COLS].reset_index(drop=True)


def _annual_flow(df: pd.DataFrame) -> pd.DataFrame:
    """Keep annual (~365d duration) observations; first disclosure per fiscal end."""
    if df.empty:
        return df
    d = df.dropna(subset=["end", "start", "filed", "val"]).copy()
    dur = (d["end"] - d["start"]).dt.days
    d = d[(dur >= ANNUAL_MIN_DAYS) & (dur <= ANNUAL_MAX_DAYS)]
    if d.empty:
        return d
    # earliest filing that disclosed each fiscal-year-end value
    d = d.sort_values("filed").drop_duplicates(subset=["end"], keep="first")
    return d[["end", "filed", "val"]].sort_values("end").reset_index(drop=True)


def _quarterly_flow(df: pd.DataFrame) -> pd.DataFrame:
    """Discrete quarterly flow observations [end, filed, val].

    XBRL flow facts come in two shapes and this handles both:
      * DISCRETE "three-months-ended" (~90d) facts -> used directly (typical for
        income-statement items, giving Q1-Q3).
      * YEAR-TO-DATE cumulative facts (3M/6M/9M/FY, all sharing the fiscal-year
        `start`) -> typical for CASH-FLOW items; de-cumulated into quarters by
        differencing consecutive period ends within the fiscal year.
    Any fiscal-year end still missing a quarter (pure-discrete filers that never
    file a Q4 10-Q) gets Q4 DERIVED as FY - (Q1 + Q2 + Q3).

    `filed` on each quarter is the filing that made it computable (the later of
    the cumulatives involved), so nothing is stamped before it is public.
    """
    if df.empty:
        return df
    d = df.dropna(subset=["end", "start", "filed", "val"]).copy()
    d["dur"] = (d["end"] - d["start"]).dt.days
    d = d[(d["dur"] >= 45) & (d["dur"] <= ANNUAL_MAX_DAYS)]
    if d.empty:
        return pd.DataFrame(columns=["end", "filed", "val"])
    # earliest filing that disclosed each (start, end) period
    d = d.sort_values("filed").drop_duplicates(subset=["start", "end"], keep="first")

    # De-cumulate within each fiscal-year `start`: discrete = value - previous
    # cumulative (by end); the first (shortest) period is already discrete.
    d = d.sort_values(["start", "end"])
    grp = d.groupby("start", sort=False)
    disc = d["val"].astype(float) - grp["val"].shift(1)
    is_first = grp.cumcount() == 0
    disc = disc.where(~is_first, d["val"])
    implied = (d["end"] - grp["end"].shift(1)).dt.days
    implied = implied.where(~is_first, d["dur"])

    q = d.assign(val=disc, implied=implied)
    # keep quarter-length periods. Upper bound 120d (not 100) so 52/53-week retailers
    # with a 16-week fiscal Q1 (~112d, e.g. KR/COST/TGT) aren't dropped -> their flows
    # now form 4 quarters and TTM populates. 6-month YTD (181d) stays excluded.
    q = q[(q["implied"] >= 75) & (q["implied"] <= 120)]
    q = (q.sort_values("filed").drop_duplicates(subset=["end"], keep="first")
           [["end", "filed", "val"]])

    # Derive Q4 for any fiscal-year end not already covered above.
    a = d[(d["dur"] >= ANNUAL_MIN_DAYS) & (d["dur"] <= ANNUAL_MAX_DAYS)]
    a = (a.sort_values("filed").drop_duplicates(subset=["end"], keep="first")
           [["end", "filed", "val"]])
    q_ends = set(q["end"])
    derived = []
    for _, r in a.iterrows():
        fye = r["end"]
        if fye in q_ends:
            continue
        prior = q[(q["end"] > fye - pd.Timedelta(days=340))
                  & (q["end"] <= fye - pd.Timedelta(days=20))]
        if len(prior) == 3:
            derived.append({
                "end": fye,
                "filed": max(r["filed"], prior["filed"].max()),
                "val": r["val"] - prior["val"].sum(),
            })
    if derived:
        q = pd.concat([q, pd.DataFrame(derived)], ignore_index=True)
    return (q.sort_values("end").drop_duplicates(subset=["end"], keep="first")
              .reset_index(drop=True))


def _instant_stock(df: pd.DataFrame) -> pd.DataFrame:
    """Point-in-time balance items: first disclosure per period end."""
    if df.empty:
        return df
    d = df.dropna(subset=["end", "filed", "val"]).copy()
    d = d.sort_values("filed").drop_duplicates(subset=["end"], keep="first")
    return d[["end", "filed", "val"]].sort_values("end").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Per-ticker fundamentals history builder                                      #
# --------------------------------------------------------------------------- #
# curated (always-checked) concepts + the "spine" whose period-ends define the grid
_CURATED_FLOWS = ["totalRevenue", "netIncome", "grossProfit", "costOfRevenue", "operatingIncome",
                  "depAmort", "operatingCashFlow", "capex", "researchAndDevelopment",
                  "sellingGeneralAdmin", "stockBasedComp", "acquisitions", "interestExpense"]
_CURATED_STOCKS = ["stockholdersEquity", "totalLiabilities", "longTermDebt", "cash",
                   "shortTermDebt", "currentAssets", "currentLiabilities", "goodwill", "totalAssets"]
_SPINE_FLOWS = ("totalRevenue", "netIncome", "operatingCashFlow")
_SPINE_STOCKS = ("totalAssets", "stockholdersEquity", "totalLiabilities")


# net-income-to-common: added to the netIncome coalesce ONLY for filers with no
# material preferred dividends (see `_net_income_tags`).
_NET_INCOME_TO_COMMON_TAG = "NetIncomeLossAvailableToCommonStockholdersBasic"
_TO_COMMON_MIN_OVERLAP = 4     # need this many primary-vs-to-common overlaps to judge
_TO_COMMON_TOL = 0.02          # per-period relative gap treated as "equal"
_TO_COMMON_MIN_MATCH = 0.90    # share of overlap periods that must match to trust it


def _concept_periods(gaap: dict, tag: str) -> dict[tuple, float]:
    """`{(start, end): val}` for one raw us-gaap tag (USD unit), for the preferred-
    dividend safety check — light-weight, no coalescing/date parsing."""
    units = gaap.get(tag, {}).get("units", {})
    uk = "USD" if "USD" in units else next(iter(units), None)
    if uk is None:
        return {}
    return {(o["start"], o["end"]): o["val"] for o in units[uk]
            if o.get("start") and o.get("end") and o.get("val") is not None}


def _net_income_tags(gaap: dict) -> list[str]:
    """netIncome coalesce list, extended with the net-income-to-common tag ONLY when
    it materially equals ProfitLoss/NetIncomeLoss on their overlapping periods — i.e.
    the filer has no meaningful preferred dividends. This recovers pre-2016 netIncome
    for no-preferred filers (e.g. WAT 2014-2015) without contaminating the YTD
    de-cumulation chain of preferred-paying REITs/banks/insurers (WELL, O, SPG, USB,
    VTR, ...), whose to-common figure is net of preferred dividends."""
    tags = list(FLOW_TAGS["netIncome"])
    common = _concept_periods(gaap, _NET_INCOME_TO_COMMON_TAG)
    if not common:
        return tags
    primary: dict[tuple, float] = {}
    for tag in ("NetIncomeLoss", "ProfitLoss"):
        for k, v in _concept_periods(gaap, tag).items():
            primary.setdefault(k, v)
    rel = [abs(common[k] - primary[k]) / abs(primary[k])
           for k in common if k in primary and primary[k]]
    if len(rel) >= _TO_COMMON_MIN_OVERLAP and \
            sum(r <= _TO_COMMON_TOL for r in rel) / len(rel) >= _TO_COMMON_MIN_MATCH:
        tags.append(_NET_INCOME_TO_COMMON_TAG)
    return tags


def _extract_all(gaap: dict, dei: dict) -> tuple[dict, dict, dict, pd.DataFrame, pd.DataFrame]:
    """Raw us-gaap/dei -> per-concept quarterly flows, annual full-year fallbacks,
    instant balance-sheet levels, plus cover-page shares (dei) and diluted shares."""
    flow_tags = {**FLOW_TAGS, **EXTRA_FLOW_TAGS}
    flow_tags["netIncome"] = _net_income_tags(gaap)   # preferred-dividend-guarded fill
    raw = {k: _extract_concept(gaap, tags) for k, tags in flow_tags.items()}
    flows = {k: _quarterly_flow(v) for k, v in raw.items()}
    annuals = {k: _annual_flow(v) for k, v in raw.items()}      # full-year TTM fallback
    stocks = {k: _instant_stock(_extract_concept(gaap, tags))
              for k, tags in {**STOCK_TAGS, **EXTRA_STOCK_TAGS}.items()}
    _sh = SHARES_TAGS["sharesOutstanding"]
    shares = _instant_stock(_extract_concept(dei, _sh) if any(t in dei for t in _sh)
                            else _extract_concept(gaap, _sh))
    diluted = _instant_stock(_extract_concept(gaap, DILUTED_SHARES_TAGS))
    return flows, annuals, stocks, shares, diluted


def _spine_grid(flows: dict, stocks: dict) -> "pd.DatetimeIndex | None":
    """Fiscal-quarter row grid = UNION of period-ends across the core spine concepts,
    so a revenue-tag gap (banks ~2013, utilities ~2014) doesn't truncate the ticker."""
    ends: set = set()
    for k in _SPINE_FLOWS:
        s = flows.get(k)
        if s is not None and not s.empty:
            ends |= set(s["end"])
    for k in _SPINE_STOCKS:
        s = stocks.get(k)
        if s is not None and not s.empty:
            ends |= set(s["end"])
    return pd.DatetimeIndex(sorted(ends)) if ends else None


def _assemble_base(ends, flows, annuals, stocks, shares, diluted) -> pd.DataFrame:
    """Align every concept onto the quarter grid in ONE frame construction (a single
    dict -> one DataFrame, no repeated column inserts -> no pandas fragmentation).
    Exact-end join for flows/annuals/stocks; backward as-of (reindex-ffill) for the
    cover-page shares; balance-sheet levels carried forward across interim quarters;
    `as_of` = latest filing date among a row's concepts (point-in-time / leak-free)."""
    cols: dict[str, object] = {}

    def exact(key, src, filed=True):
        if src is None or src.empty:
            return
        s = src.drop_duplicates("end").set_index("end").reindex(ends)
        cols[key] = s["val"].to_numpy()
        if filed:
            cols[key + "_filed"] = s["filed"].to_numpy()

    def asof(key, src, filed=False):     # value is dated near the filing, not period end
        if src is None or src.empty:
            return
        s = src.drop_duplicates("end").sort_values("end").set_index("end")
        cols[key] = s["val"].reindex(ends, method="ffill").to_numpy()
        if filed:
            cols[key + "_filed"] = s["filed"].reindex(ends, method="ffill").to_numpy()

    flow_keys = _CURATED_FLOWS + list(EXTRA_FLOW_TAGS)
    for key in flow_keys:
        exact(key, flows.get(key))
    for key in flow_keys:                # annual full-year fallback (suffix `_ann`)
        a = annuals.get(key)
        if a is not None and not a.empty:
            cols[key + "_ann"] = a.drop_duplicates("end").set_index("end")["val"].reindex(ends).to_numpy()
    for key in _CURATED_STOCKS + list(EXTRA_STOCK_TAGS):
        exact(key, stocks.get(key))
    asof("sharesOutstanding", shares, filed=True)
    asof("dilutedShares", diluted)

    base = pd.DataFrame(cols, index=ends)
    level_cols = [k for k in (list(STOCK_TAGS) + list(EXTRA_STOCK_TAGS)) if k in base.columns]
    if level_cols:                       # carry point-in-time levels forward (~1y cap)
        base[level_cols] = base[level_cols].ffill(limit=4)
    filed_cols = [c for c in base.columns if c.endswith("_filed")]
    base["as_of"] = base[filed_cols].max(axis=1) if filed_cols else pd.NaT
    base = base[base["as_of"].notna()]
    return base.rename_axis("end").reset_index().sort_values("end").reset_index(drop=True)


class TickerFundamentalsBuilder:
    """Builds one ticker's point-in-time QUARTERLY fundamentals history from its SEC
    companyfacts. Fundamentals are the backbone of the cube, so the build is split
    into small, testable stages — extract -> grid -> assemble -> derive:

        _extract_all    raw us-gaap/dei -> quarterly / annual / instant concept series
        _spine_grid     the fiscal-quarter row grid (union of core period-ends)
        _assemble_base  align concepts onto the grid in ONE frame + point-in-time as_of
        _derive_history TTM sums (annual fallback), margins/ratios, level reconstructions

    Flows are trailing-twelve-month sums of discrete quarters (annual full-year value
    as a fallback); balance-sheet items are the last filed level carried point-in-time;
    `as_of` is the latest filing date among a row's concepts, so nothing leaks.
    """

    def __init__(self, ticker: str, facts: dict, sector: str | None = None,
                 industry_group: str | None = None):
        self.ticker = ticker
        self.sector = sector
        self.industry_group = industry_group
        self._gaap = facts.get("facts", {}).get("us-gaap", {})
        self._dei = facts.get("facts", {}).get("dei", {})

    def build(self) -> pd.DataFrame:
        flows, annuals, stocks, shares, diluted = _extract_all(self._gaap, self._dei)
        ends = _spine_grid(flows, stocks)
        if ends is None:
            return pd.DataFrame()
        base = _assemble_base(ends, flows, annuals, stocks, shares, diluted)
        if base.empty:
            return pd.DataFrame()
        return _derive_history(base, self.ticker, self.sector, self.industry_group)


def build_ticker_history(ticker: str, facts: dict, sector: str | None = None,
                         industry_group: str | None = None) -> pd.DataFrame:
    """One row per FISCAL QUARTER, keyed on the filing date (`as_of`). Public entry
    point — delegates to TickerFundamentalsBuilder (see it for the full pipeline)."""
    return TickerFundamentalsBuilder(ticker, facts, sector, industry_group).build()


# --------------------------------------------------------------------------- #
# Derived history: TTM flows (annual fallback), margins/ratios, reconstructions #
# --------------------------------------------------------------------------- #
def _derive_history(base: pd.DataFrame, ticker: str, sector, industry_group) -> pd.DataFrame:
    """Turn the assembled quarter grid into the output history: TTM flow sums (with
    the annual full-year fallback), margins / ratios, and clean balance-sheet level
    reconstructions. Flows are seasonality-free trailing-twelve-month; balance-sheet
    items are point-in-time; `as_of` was already stamped as the latest filing date."""

    # ---- discrete quarterly numerics ----
    def col(name):
        # a concept a filer never reports isn't a column in the assembled frame -> NaN
        if name in base.columns:
            return pd.to_numeric(base[name], errors="coerce")
        return pd.Series(float("nan"), index=base.index)

    def ttm(s):
        # trailing 12 months = sum of the 4 most recent quarters
        return s.rolling(TTM_QUARTERS, min_periods=TTM_QUARTERS).sum()

    def ttm_a(key, charge=False):
        """TTM of the quarterly series, falling back to the forward-filled ANNUAL
        full-year value where quarters are unavailable (a filer's reported full year
        IS a trailing-twelve-month). Leak-free: the annual value was filed before the
        interim quarters it is carried into. Used for filers that report a flow only
        annually (no de-cumulable interim quarters)."""
        s = col(key)
        if charge:
            s = s.fillna(0.0)
        q = ttm(s)
        acol = key + "_ann"
        if acol in base.columns:
            ann = pd.to_numeric(base[acol], errors="coerce").ffill(limit=4)
            q = q.where(q.notna(), ann)
        return q

    # discrete SINGLE-QUARTER values (before rolling) -> "latest quarter" features
    rev_q = col("totalRevenue")
    ni_q = col("netIncome")
    oi_q = col("operatingIncome")
    da_q = col("depAmort")
    ocf_q = col("operatingCashFlow")
    capex_q = col("capex")
    ebitda_q = oi_q + da_q.fillna(0)
    # bottom-up single-quarter EBITDA for filers without an operating-income line
    ebitda_q = ebitda_q.where(ebitda_q.notna(),
                              ni_q + col("incomeTaxExpense").fillna(0)
                              + col("interestExpense").fillna(0) + da_q.fillna(0))
    fcf_q = ocf_q - capex_q.fillna(0)

    rev_ttm = ttm_a("totalRevenue")
    ni_ttm = ttm_a("netIncome")
    # Financials top line: the ASC-606 contract-revenue element tags only a FEE SLICE for
    # banks / insurers / asset managers, understating revenue many-fold (e.g. FITB $0.5B
    # vs true $8B, MET, AIG) -> nonsense margins. For the Financials sector ONLY, rebuild
    # revenue from its components and take the most complete signal — the fee slice is
    # always a subset, so max() never double-counts: net interest income + noninterest
    # income (banks), premiums earned + net investment income (insurers), or the
    # consolidated `Revenues` line (asset managers). Non-financials are untouched.
    if sector == "Financials":
        nii, noni = ttm_a("netInterestIncome"), ttm_a("noninterestIncome")
        prem, inv = ttm_a("premiumsEarned"), ttm_a("netInvestmentIncome")
        bank_rev = (nii.fillna(0) + noni.fillna(0)).where(nii.notna() | noni.notna())
        insurer_rev = (prem.fillna(0) + inv.fillna(0)).where(prem.notna() | inv.notna())
        rev_ttm = pd.concat([rev_ttm, bank_rev, insurer_rev, ttm_a("revenuesTotal")],
                            axis=1).max(axis=1)
    elif sector == "Real Estate":
        # REITs tag rental income under the operating-LEASE elements (leases are outside
        # ASC-606), so the contract-revenue element the coalesce grabs is only a small FEE
        # slice (e.g. CPT $11M, EXR $52M vs ~$1B of rent). Take the larger of the coalesced
        # total and the rental line: rent is a subset, so max() never double-counts, and
        # REITs that DO tag rent under contract-revenue (e.g. ARE/PLD) keep their higher total.
        rev_ttm = pd.concat([rev_ttm, ttm_a("rentalIncome")], axis=1).max(axis=1)
    elif sector == "Energy":
        # E&P filers tag the top line as oil & gas revenue; integrated majors report a
        # fuller `Revenues`, which max() keeps (oil&gas is a subset -> no double-count).
        # Sector-gated so a non-energy filer consolidating an oil&gas portfolio company
        # (e.g. an asset manager) is untouched.
        rev_ttm = pd.concat([rev_ttm, ttm_a("oilGasRevenue")], axis=1).max(axis=1)
    gp_ttm = ttm_a("grossProfit")
    cor_ttm = ttm_a("costOfRevenue")
    oi_ttm = ttm_a("operatingIncome")
    da_ttm = ttm_a("depAmort")
    ocf_ttm = ttm_a("operatingCashFlow")
    capex_ttm = ttm_a("capex")
    rnd_ttm = ttm_a("researchAndDevelopment")
    sga_ttm = ttm_a("sellingGeneralAdmin")
    sbc_ttm = ttm_a("stockBasedComp")
    acq_ttm = ttm_a("acquisitions", charge=True)     # no acquisition this quarter = 0 (charge flow)
    int_ttm = ttm_a("interestExpense")
    eq = col("stockholdersEquity")          # instant (point-in-time), not summed
    liab = col("totalLiabilities")
    ltd = col("longTermDebt")
    # instant balance-sheet levels carried raw for the distress / M&A features
    cash = col("cash")
    std_debt = col("shortTermDebt")
    cur_a = col("currentAssets")
    cur_l = col("currentLiabilities")
    goodwill = col("goodwill")
    assets = col("totalAssets")

    # Derive total liabilities when a filer doesn't tag `Liabilities` as a single
    # element (e.g. LLY, AMD report only Assets + Equity + LiabilitiesAnd-
    # StockholdersEquity, leaving `Liabilities` absent). Accounting identity:
    # Liabilities = Assets - Equity. Only where both sides exist, so filers that
    # report none of them stay NaN.
    liab = liab.where(liab.notna(), assets - eq)

    # Total interest-bearing debt (universal): the single combined ST+LT tag when a
    # filer reports it (many banks / insurers), else long-term + short-term, else
    # notes payable (REITs). Its OWN pool, so short- and long-term are never merged
    # into longTermDebt.
    _debt_combined = col("debtCombined")
    _lt_st = (ltd.fillna(0) + std_debt.fillna(0)).where(ltd.notna() | std_debt.notna())
    total_debt = _debt_combined.where(_debt_combined.notna(), _lt_st)
    total_debt = total_debt.where(total_debt.notna(), col("notesPayable"))
    # ZERO debt vs MISSING debt — dissociate the two (they were both NaN before). When a filer
    # reports a balance sheet for a period (total assets or equity present) but tags NO interest-
    # bearing debt, debt is 0 on that date, not unknown. Applied PER PERIOD (so a name that is
    # debt-free only part of its history is 0 there, not NaN) and to BOTH legs + the total so the
    # stored columns stay consistent. EXCLUDED for Financials UNLESS debt-free across ALL history:
    # banks / insurers fund via deposits / FHLB advances / repos tagged outside our debt concepts,
    # so a missing debt tag there is not reliably zero and stays NaN.
    bs_present = assets.notna() | eq.notna()
    may_zero = (sector != "Financials") or (total_debt.notna().sum() == 0)
    if may_zero:
        total_debt = total_debt.where(~(bs_present & total_debt.isna()), 0.0)
        ltd = ltd.where(~(bs_present & ltd.isna()), 0.0)
        std_debt = std_debt.where(~(bs_present & std_debt.isna()), 0.0)

    gross_profit_ttm = gp_ttm.where(gp_ttm.notna(), rev_ttm - cor_ttm)
    # Gross margin, with an extraction-artifact guard: a value > 1 (negative COGS) or
    # below -200% only arises from a revenue/cost period-or-scope mismatch (truncated
    # revenue), never as a real gross margin, so it is nulled rather than shipped.
    gross_margin = (gross_profit_ttm / rev_ttm).where(rev_ttm > 0)
    gross_margin = gross_margin.where(
        (gross_margin >= GROSS_MARGIN_MIN) & (gross_margin <= GROSS_MARGIN_MAX))
    # Net margin, with a FINANCIALS-ONLY artifact guard: a bank/insurer/asset-manager net
    # income exceeding ~1.5x net revenue (or a loss beyond -200%) reflects consolidated-
    # fund NCI / one-time attribution (e.g. ARES pre-IPO), not a real margin. The bound
    # holds structurally in-sector, so it is nulled there; other sectors (biotech losses,
    # one-time gains) keep their genuine extreme margins.
    profit_margin = (ni_ttm / rev_ttm).where(rev_ttm > 0)
    if sector == "Financials":
        profit_margin = profit_margin.where((profit_margin >= -2.0) & (profit_margin <= 1.5))
    # Derive operating income when the filer doesn't tag OperatingIncomeLoss
    # (e.g. LLY, CVX, JNJ post-2015): operating income ≈ gross profit − SG&A − R&D.
    # Only where the components exist, so banks (no gross profit) stay NaN.
    oi_derived = gross_profit_ttm - sga_ttm.fillna(0) - rnd_ttm.fillna(0)
    oi_ttm = oi_ttm.where(oi_ttm.notna(), oi_derived)
    # REITs / integrated oil & gas tag no operating-income line and have no gross profit
    # (e.g. O, EQR, DVN, XOM), yet EBITDAre / EBITDAX need operating income -> bottom-up
    # EBIT = pre-tax income + interest expense. Gated to non-financials (banks / insurers
    # have their own operating-income proxies), so their operating margin stays N/A.
    if sector != "Financials":
        # some non-financials report "Total costs and expenses" (incl. DD&A) but no
        # operating-income line (integrated oil pre-restructuring, e.g. COP 2012-2016):
        # operating income = revenue - total operating costs. Preferred over the EBIT
        # proxy below because it excludes non-operating items. Guard: reject an
        # implausibly high (>60%) implied operating margin, which signals INCOMPLETELY
        # tagged costs (e.g. COP's pre-2012 integrated years) rather than real OI.
        oi_cae = rev_ttm - ttm_a("costsAndExpenses")
        oi_cae = oi_cae.where(oi_cae <= 0.60 * rev_ttm)
        oi_ttm = oi_ttm.where(oi_ttm.notna(), oi_cae)
        # last resort: bottom-up EBIT = pre-tax income + interest expense.
        oi_ttm = oi_ttm.where(oi_ttm.notna(), ttm_a("pretaxIncome") + int_ttm.fillna(0))

    # EBITDA = operating income + D&A, with a bottom-up fallback (net income + taxes
    # + interest + D&A) for filers that report no operating-income line, e.g.
    # integrated oil (XOM). All four inputs exist even when OI/gross profit don't.
    ebitda = oi_ttm + da_ttm.fillna(0)
    ebitda = ebitda.where(ebitda.notna(),
                          ni_ttm + ttm_a("incomeTaxExpense").fillna(0)
                          + int_ttm.fillna(0) + da_ttm.fillna(0))

    out = pd.DataFrame({
        "ticker": ticker,
        "as_of": base["as_of"].dt.date.astype(str),
        "fiscal_end": base["end"].dt.date.astype(str),
        "totalRevenue": rev_ttm,
        "netIncome": ni_ttm,
        "grossMargins": gross_margin,
        "operatingMargins": (oi_ttm / rev_ttm).where(rev_ttm > 0),
        "profitMargins": profit_margin,
        # ROE is defined whenever equity is non-zero; negative-equity firms (heavy
        # buybacks, e.g. VRSN/WYNN) get a (negative) ROE rather than being dropped.
        "returnOnEquity": (ni_ttm / eq).where(eq != 0),
        "debtToEquity": (ltd.where(ltd.notna(), liab) / eq).where(eq > 0),
        "ebitda": ebitda,
        # operatingIncome / depAmort / capex are emitted as raw TTM levels (not just
        # folded into margins / FCF) because the sector KPIs consume them directly:
        # EBITDAre = operatingIncome + depAmort (REIT), EBITDAX = + explorationExpense
        # (oil & gas), FFO uses depAmort, AFFO = FFO - capex, bank operating-income proxy.
        "operatingIncome": oi_ttm,
        "depAmort": da_ttm,
        "capex": capex_ttm,
        "freeCashflow": ocf_ttm - capex_ttm.fillna(0),
        "operatingCashFlow": ocf_ttm,
        "researchAndDevelopment": rnd_ttm,
        "stockholdersEquity": eq,
        "sharesOutstanding": col("sharesOutstanding"),
        # ---- raw levels / TTM flows for the refined features ----
        # (distress: cash + debt + current items + interest; S&M: SG&A;
        #  M&A: acquisitions + goodwill + assets; SBC: stockBasedComp)
        "cash": cash,
        "longTermDebt": ltd,
        "shortTermDebt": std_debt,
        "totalDebt": total_debt,
        "totalLiabilities": liab,
        "currentAssets": cur_a,
        "currentLiabilities": cur_l,
        "goodwill": goodwill,
        "totalAssets": assets,
        "sellingGeneralAdmin": sga_ttm,
        "stockBasedComp": sbc_ttm,
        "acquisitions": acq_ttm,
        "interestExpense": int_ttm,
        # discrete single-quarter values -> "latest quarter" momentum features
        "revenue_q": rev_q,
        "netIncome_q": ni_q,
        "ebitda_q": ebitda_q,
        "freeCashflow_q": fcf_q,
    })

    # Year-over-year growth on the TTM series (4 quarters back), so it is a true
    # annual comparison free of seasonality even at quarterly cadence.
    out["revenueGrowth"] = rev_ttm.pct_change(TTM_QUARTERS)
    out["earningsGrowth"] = ni_ttm.pct_change(TTM_QUARTERS)

    # ---- expanded raw coverage (universal §B + sector line items) ----
    # flows -> TTM (seasonality-free); balance-sheet items -> point-in-time level.
    # Assemble all extra columns at once (avoids DataFrame fragmentation).
    extra = {"costOfRevenue": cor_ttm, "grossProfit": gross_profit_ttm,
             "dilutedShares": col("dilutedShares")}
    for key in EXTRA_FLOW_TAGS:
        extra[key] = ttm_a(key, charge=(key in CHARGE_FLOWS))
    for key in EXTRA_STOCK_TAGS:
        extra[key] = col(key)

    # --- clean reconstructions: fill each target level by DERIVING from its own
    #     components, never by coalescing a different pool (which would mix signals) ---
    _accum = extra["accumulatedDepreciation"]
    # gross PP&E for net-only filers:  gross = net + accumulated depreciation
    extra["ppeGross"] = extra["ppeGross"].where(extra["ppeGross"].notna(),
                                                extra["ppeNet"] + _accum)
    # net oil&gas property for E&P filers that tag only gross:  net = gross - accumulated
    extra["oilGasPropertyNet"] = extra["oilGasPropertyNet"].where(
        extra["oilGasPropertyNet"].notna(), extra["oilGasPropertyGross"] - _accum)
    # total deferred revenue: combined tag when present, else current + noncurrent
    # (no double-count: the split is used only where the combined tag is absent)
    _dr_split = extra["deferredRevenueCurrent"].fillna(0) + extra["deferredRevenueNoncurrent"].fillna(0)
    _dr_split = _dr_split.where(extra["deferredRevenueCurrent"].notna()
                               | extra["deferredRevenueNoncurrent"].notna())
    extra["deferredRevenue"] = extra["deferredRevenue"].where(extra["deferredRevenue"].notna(), _dr_split)
    # regulatory assets/liabilities total: combined tag when present, else current +
    # noncurrent (utilities split them) -> the full pool the utility KPIs need.
    for _tot, _cur, _nc in (("regulatoryAssets", "regulatoryAssetsCurrent", "regulatoryAssetsNoncurrent"),
                            ("regulatoryLiabilities", "regulatoryLiabilitiesCurrent", "regulatoryLiabilitiesNoncurrent")):
        _sp = extra[_cur].fillna(0) + extra[_nc].fillna(0)
        _sp = _sp.where(extra[_cur].notna() | extra[_nc].notna())
        extra[_tot] = extra[_tot].where(extra[_tot].notna(), _sp)

    out = pd.concat([out, pd.DataFrame(extra, index=out.index)], axis=1)

    # sector tags so downstream KPIs can be computed / neutralized sector-relatively
    out["sector"] = sector
    out["industry_group"] = industry_group

    return out.copy().reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Full universe historical build                                             #
# --------------------------------------------------------------------------- #
def build_fundamentals_history_sec(context: Context,
                                   cik_mapping: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Build the ~10-year point-in-time fundamentals history for the universe from
    SEC companyfacts and write it to FUNDAMENTALS_HISTORY_PATH.

    Incremental behaviour:
      - If history was already built today for the full universe, load and return.
      - If history exists from today but new tickers appeared, fetch only those.
      - On a new calendar day, refresh all tickers and merge with existing rows.
    """

    if _is_sec_history_up_to_date(context, cik_mapping):
        hist = context.store.load("fundamentals_history")
        print(
            f"SEC fundamentals history already up to date for {_today_iso()} "
            f"— skipping ({len(hist)} rows, {hist['ticker'].nunique()} tickers)"
        )
        return hist

    existing = _load_existing_history(context)
    to_process = _tickers_to_process(context, cik_mapping, existing)

    if to_process.empty and existing is not None and not existing.empty:
        _save_history_meta(context, existing, len(cik_mapping))
        print(
            f"SEC fundamentals history already up to date for {_today_iso()} "
            f"— skipping ({len(existing)} rows)"
        )
        return existing

    years = context.config.data_extract.years_history
    cutoff = pd.Timestamp.today() - pd.DateOffset(years=years)

    new_frames = []
    for _, r in tqdm(to_process.iterrows(), total=len(to_process),
                     desc="Building SEC fundamentals history"):
        cik, ticker = r["cik"], r["ticker"]
        facts = _fetch_companyfacts(context, cik)
        if not facts:
            continue
        hist = build_ticker_history(ticker, facts, r.get("sector"), r.get("industry_group"))
        if not hist.empty:
            new_frames.append(hist)

    parts = [df for df in (existing, *new_frames) if df is not None and not df.empty]
    if not parts:
        raise RuntimeError("No SEC fundamentals built — check CIK mapping / network.")

    out = pd.concat(parts, ignore_index=True)
    out["as_of_dt"] = pd.to_datetime(out["as_of"])
    out = out[out["as_of_dt"] >= cutoff].drop(columns=["as_of_dt"])
    out = out.drop_duplicates(subset=["ticker", "as_of"], keep="last")
    out = out.sort_values(["ticker", "as_of"]).reset_index(drop=True)

    # persist only the newly-built rows; the DB merges on (ticker, as_of)
    new = pd.concat(new_frames, ignore_index=True) if new_frames else pd.DataFrame()
    if not new.empty:
        context.store.save("fundamentals_history", new)
    _save_history_meta(context, out, len(cik_mapping))
    print(f"Saved {len(new)} new SEC fundamental rows "
          f"({out['ticker'].nunique()} tickers) to DB table 'fundamentals_history'")
    return out


# --------------------------------------------------------------------------- #
# Entry point (called by StepExtractAllData)                                  #
# --------------------------------------------------------------------------- #
def fetch_fundamentals(context: Context, tickers: list[str]):
    """
    Build the SEC 10-year fundamentals history (incremental: skips when already
    built today for the full universe; companyfacts JSONs are cached per CIK
    between runs).

    Forward P/E is no longer scraped as a today-only yfinance snapshot -- it is
    reconstructed HISTORICALLY in the feature layer (earnings_features:
    forward_earnings_yield) from the earnings-surprise archive, and market cap is
    shares x daily close in the factor layer.
    """
    cik_mapping = load_cik_mapping(context)
    return build_fundamentals_history_sec(context, cik_mapping)

