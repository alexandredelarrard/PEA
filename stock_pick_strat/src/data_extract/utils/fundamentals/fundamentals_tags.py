"""
XBRL tag-candidate maps shared by BOTH fundamentals extraction paths:
  * fetch_fundamentals.py        (SEC companyfacts JSON, per-CIK aggregate)
  * fetch_fundamentals_edgar.py  (edgartools, per-filing XBRL)

Each logical field maps to an ORDERED list of candidate us-gaap/dei tags. Filers
tag the same economic concept differently across eras and sectors, so callers
must UNION every candidate and, per period, keep the highest-priority (earliest-
listed) tag that reported it — never take the first candidate present (that
truncates history badly; see the coalescing docstrings in the modules above).

Moved here verbatim from fetch_fundamentals.py so both extraction paths share
ONE definition instead of drifting (the edgartools-based WIP file previously
carried its own, much thinner copy — that's the confirmed root cause of the
MAA `capex` gap: its list lacked `PaymentsForCapitalImprovements`, the REIT
maintenance-capex tag already present here).
"""

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
    # `OperatingAndNonoperatingRevenues` / `OperatingLoss` / `OperatingIncome` were
    # dropped: not us-gaap elements (0 of 498 S&P-500 filers report any of them).
    "operatingIncome": ["OperatingIncomeLoss"],
    "depAmort": ["DepreciationDepletionAndAmortization",
                 "DepreciationAmortizationAndAccretionNet",
                 "DepreciationAndAmortization",
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
    # UNRESTRICTED cash only. The restricted-inclusive and cash+ST-investment totals
    # used to be coalesced in here as last-resort fallbacks, which overstated cash for
    # every filer that fell through to them (95.6% of filers report the restricted-
    # inclusive total) and, because the EV also nets `shortTermInvestments`, DOUBLE-
    # subtracted short-term investments wherever the cash+STI total won. They are now
    # separate pools (`cashInclRestricted` / `cashAndShortTermInvestments`) that
    # `_derive_history` nets down to clean cash before anything consumes it.
    "cash": [ # disc-ops-inclusive variant of the primary cash line (e.g. FISV mid-divestiture) --
             # same concept, so right after the primary.
             "CashAndCashEquivalentsAtCarryingValue",
             "CashAndCashEquivalentsAtCarryingValueIncludingDiscontinuedOperations",
             "CashAndDueFromBanks", 
             "Cash",
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
    # ---- reported per-share + tax-cash + below-the-line items (§B tier-1 additions) ----
    # Reported diluted/basic EPS (98.8% / 98.6%) is the cleanest per-share earnings: it is
    # already net of preferred dividends and handles two-class structures, and it is the
    # figure the analyst archive forecasts, so the surprise features reconcile to it.
    # TTM-summed like any flow (4 quarters of EPS = trailing annual EPS).
    "epsDiluted": ["EarningsPerShareDiluted", "EarningsPerShareBasicAndDiluted"],
    "epsBasic": ["EarningsPerShareBasic", "EarningsPerShareBasicAndDiluted"],
    "dividendsPerShare": ["CommonStockDividendsPerShareDeclared", "CommonStockDividendsPerShareCashPaid"],
    # CASH taxes / interest actually paid (92.8% / 89.8%): the gap vs the accrual expense is
    # the classic earnings-quality tell (a low cash tax rate on a high book rate = aggressive
    # deferral), and cash interest is the true debt-service burden.
    "incomeTaxesPaid": ["IncomeTaxesPaidNet", "IncomeTaxesPaid"],
    "deferredIncomeTaxExpense": ["DeferredIncomeTaxExpenseBenefit"],
    "interestPaid": ["InterestPaidNet", "InterestPaid"],
    # ---- CASH-FLOW FOOTING: the reported net change in cash --------------------------
    # Lets the statement be checked as published (operating + investing + financing + FX ==
    # net change) instead of trusting three independently-coalesced subtotals. Coalesces
    # BOTH eras: the ASU-2016-18 restricted-cash-inclusive tag (474 tickers, 2015-2026) and
    # the pre-2018 cash-only one (456 tickers, 2006-2022).
    "cashPeriodChange": [
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalentsPeriodIncreaseDecrease"
        "IncludingExchangeRateEffect",
        "CashAndCashEquivalentsPeriodIncreaseDecrease"],
    "otherInvestingCashFlow": ["PaymentsForProceedsFromOtherInvestingActivities"],
    "otherFinancingCashFlow": ["ProceedsFromPaymentsForOtherFinancingActivities"],
    # cash rent actually PAID on operating leases -- the cash cost behind the ROU asset
    "operatingLeasePayments": ["OperatingLeasePayments"],
    # ---- TAX components (currency flows -> correctly TTM-summed) ----------------------
    # NOTE the effective RATE lives in LATEST_DURATION_TAGS, not here: it is a ratio, and
    # anything in EXTRA_FLOW_TAGS is TTM-summed (`ttm_a`), which would turn a 21% rate into
    # ~84% by adding four quarters together.
    "currentTaxExpense": ["CurrentIncomeTaxExpenseBenefit"],
    "currentFederalTax": ["CurrentFederalTaxExpenseBenefit"],
    "currentForeignTax": ["CurrentForeignTaxExpenseBenefit"],
    "deferredFederalTax": ["DeferredFederalIncomeTaxExpenseBenefit"],
    "otherComprehensiveIncome": ["OtherComprehensiveIncomeLossNetOfTax"],
    # equity-method income (61%) inflates pre-tax/net income with NO revenue and NO cash;
    # other non-operating (74.7%) is where ASU-2017-07 parked non-service pension cost.
    # Both must come OUT of core operating earnings.
    "equityMethodIncome": ["IncomeLossFromEquityMethodInvestments"],
    "otherNonoperating": ["OtherNonoperatingIncomeExpense"],
    "debtExtinguishment": ["GainsLossesOnExtinguishmentOfDebt"],
    # income attributable to NCI (67.3%): reconciles the incl-NCI (`ProfitLoss`) and
    # excl-NCI (`NetIncomeLoss`) bases the netIncome coalesce mixes across eras.
    "nciIncome": ["NetIncomeLossAttributableToNoncontrollingInterest"],
    "comprehensiveIncome": ["ComprehensiveIncomeNetOfTax"],
    "goodwillAcquired": ["GoodwillAcquiredDuringPeriod"],
    # ---- ASC-842 operating-lease flows ----
    # Operating-lease ADDITIONS (87.3%) are the operating-lease twin of the finance-lease
    # additions already captured in `capexGlobal` -- the bigger number for retail /
    # restaurants / airlines, so leaving it out made capacity investment asymmetric.
    "operatingLeaseAdditions": ["RightOfUseAssetObtainedInExchangeForOperatingLeaseLiability"],
    "operatingLeaseCost": ["OperatingLeaseCost", "LeaseAndRentalExpense"],
    # ---- gross-to-net revenue correction ----
    # Excise / sales taxes collected (6.2%): 19.5% of filers tag revenue under the
    # INCLUDING-assessed-tax element, which overstates the top line vs peers (tobacco,
    # beverages, fuel distribution). Netted off in `_derive_history` only for the periods
    # where the EXCLUDING element is absent, so no double deduction.
    "exciseTaxes": ["ExciseAndSalesTaxes"],
    "revenueExcludingAssessedTax": ["RevenueFromContractWithCustomerExcludingAssessedTax"],
    "revenueIncludingAssessedTax": ["RevenueFromContractWithCustomerIncludingAssessedTax"],
    # ---- DB-pension net-periodic components (footnote, mostly annual) ----
    # ASU 2017-07 (effective FY2018) forced every NON-service component out of the
    # operating subtotal. Before that they sat inside SG&A / operating income, so a
    # filer's own operating-margin series BREAKS at adoption. `_derive_history` restates
    # the pre-2018 half using non-service = net periodic cost - service cost.
    "pensionNetPeriodicCost": ["DefinedBenefitPlanNetPeriodicBenefitCost"],
    "pensionServiceCost": ["DefinedBenefitPlanServiceCost"],
    "pensionInterestCost": ["DefinedBenefitPlanInterestCost"],
    "pensionExpectedReturn": ["DefinedBenefitPlanExpectedReturnOnPlanAssets"],
    "pensionAmortPriorService": ["DefinedBenefitPlanAmortizationOfPriorServiceCostCredit"],
    "pensionAmortGainsLosses": ["DefinedBenefitPlanAmortizationOfGainsLosses",
                                "DefinedBenefitPlanAmortizationOfNetGainsLosses"],
    "dividendsPaid": ["PaymentsOfDividendsCommonStock", "PaymentsOfDividends"],
    "buybacks": ["PaymentsForRepurchaseOfCommonStock"],   # 95.6% coverage; the
    # `...AndEmployeeShareRepurchases` variant was dropped (0 of 498 filers).
    "equityIssuance": ["ProceedsFromIssuanceOfCommonStock"],
    "debtIssued": ["ProceedsFromIssuanceOfLongTermDebt"],
    "debtRepaid": ["RepaymentsOfLongTermDebt"],
    # NON-CASH capacity added via FINANCE / capital leases (data centers, equipment) -- absent from
    # the cash-capex line but real capacity investment (huge for MSFT ~$3-9B/q, historically AMZN).
    # ASC-842 `RightOfUseAsset...FinanceLease` (2019+) coalesced with the pre-2019 capital-lease
    # element (era-separated, so no double count). 0-filled (CHARGE) -> 0 when a filer uses none, so
    # `capexGlobal` = cash capex for non-lease filers and cash capex + leases where they do.
    "financeLeaseAdditions": ["RightOfUseAssetObtainedInExchangeForFinanceLeaseLiability",
                              "CapitalLeaseObligationsIncurred"],
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
    # `GainLossOnDispositionOfAssets` (19.9%) is the GENERIC disposal line many filers
    # tag instead of the business / PP&E specific ones -- coalesced here (one pool, so
    # a filer reporting several never double-counts) rather than into the real-estate
    # `gainOnDispositions` pool, which the core-earnings block sums alongside this one.
    "gainOnSaleGeneric": ["GainLossOnSaleOfBusiness",
                          "GainLossOnSaleOfPropertyPlantEquipment",
                          "GainLossOnDispositionOfAssets"],
    "litigationExpense": ["LitigationSettlementExpense"],
    "discontinuedOps": ["IncomeLossFromDiscontinuedOperationsNetOfTax"],
    "unusualItems": ["UnusualOrInfrequentItemNetGainLoss"],
    "bargainPurchaseGain": ["BusinessCombinationBargainPurchaseGainRecognizedAmount"],
    # ---- Banks ----
    "interestIncomeBank": ["InterestAndDividendIncomeOperating"],
    "netInterestIncome": ["InterestIncomeExpenseNet", "InterestIncomeExpenseAfterProvisionForLoanLoss"],
    # LENDING credit-loss provision only. `ProvisionForDoubtfulAccounts` was removed:
    # it is the TRADE-receivable bad-debt expense (48% of filers, of which only 43 are
    # Financials), so coalescing it here populated a "bank" provision for 178 non-banks
    # and fed `bank_operating_margin`. It now has its own field below.
    # `ProvisionForCreditLossExpenseReversal` is not an element (0 of 498) -> replaced
    # by the real CECL-era name.
    "provisionForCreditLosses": ["ProvisionForLoanLossesExpensed",
                                 "ProvisionForLoanLeaseAndOtherLosses",
                                 "ProvisionForLoanAndLeaseLosses",
                                 "FinancingReceivableExcludingAccruedInterestCreditLossExpenseReversal"],
    # trade-receivable bad-debt expense (NON-lending): a revenue-quality signal in its
    # own right (rising = the firm is booking sales it cannot collect), kept OUT of the
    # bank provision pool above.
    "provisionDoubtfulAccounts": ["ProvisionForDoubtfulAccounts"],
    "noninterestIncome": ["NoninterestIncome"],
    "noninterestExpense": ["NoninterestExpense"],
    # net loan charge-offs (write-offs, net of recoveries) -> realized credit losses
    # (B3). Duration flow -> TTM-summed. NET-of-recovery tags first, gross write-offs
    # as the fallback. The two names previously listed first do not exist in us-gaap
    # (0 of 498 filers: `...CreditLossesWriteoffAfterRecovery`, `...CreditLossesWriteoff`),
    # which left the column 0.6% populated -- 2.3% even within Financials. The live CECL
    # element is `FinancingReceivableAllowanceForCreditLossesWriteOffs` (capital "Off",
    # 62 filers).
    "netChargeOffs": ["FinancingReceivableExcludingAccruedInterestAllowanceForCreditLossWriteoffAfterRecovery",
                      "FinancingReceivableAllowanceForCreditLossWriteoffAfterRecovery",
                      "AllowanceForLoanAndLeaseLossesWriteoffsNet",
                      "FinancingReceivableAllowanceForCreditLossesWriteOffs",
                      "FinancingReceivableExcludingAccruedInterestAllowanceForCreditLossWriteoff",
                      "AllowanceForLoanAndLeaseLossesWriteOffs"],
    # ---- Insurance ----
    "premiumsEarned": ["PremiumsEarnedNet", "PremiumsEarnedNetPropertyAndCasualty"],
    "premiumsWritten": ["PremiumsWrittenNet"],
    # `...IncurredHomeAndAutoAndOther` dropped (0 of 498 filers).
    "claimsIncurred": ["PolicyholderBenefitsAndClaimsIncurredNet",
                       "IncurredClaimsPropertyCasualtyAndLiability"],
    "netInvestmentIncome": ["NetInvestmentIncome"],
    "dacAmortization": ["DeferredPolicyAcquisitionCostAmortizationExpense"],
    # ---- Insurance ----
    # Realized investment gains/losses: the third leg of an insurer's GAAP top line (the
    # rule's `PremiumRevenueNet + NetInvestmentIncome + RealizedInvestmentGainsLosses`),
    # and a management-TIMED item that must come out of CORE earnings for every sector
    # (`GainLossOnInvestments` is tagged by 30% of filers, not just insurers).
    "realizedInvestmentGains": ["RealizedInvestmentGainsLosses", "GainLossOnInvestments",
                                "MarketableSecuritiesRealizedGainLoss"],
    # ---- REITs ----
    "rentalIncome": ["OperatingLeaseLeaseIncome", "OperatingLeasesIncomeStatementLeaseRevenue",
                     "RealEstateRevenueNet", "LeaseIncome"],
    # AFFO adjustments beyond capex: non-cash straight-line rent and above/below-market
    # lease amortization (both sparse -- 3.0% / 2.8% -- so a no-op for most REITs, but
    # correct where disclosed).
    "straightLineRent": ["StraightLineRent"],
    "aboveBelowMarketLeaseAmort": ["AmortizationOfAboveAndBelowMarketLeases"],
    # `GainLossOnDispositionOfRealEstate` dropped (0 of 498 filers) -- the real-estate
    # disposal gain REITs actually tag is `GainLossOnSaleOfProperties` (11.4%).
    "gainOnDispositions": ["GainLossOnSaleOfProperties",
                           "GainLossOnDispositionOfProperty",
                           "GainsLossesOnSalesOfInvestmentRealEstate"],
    # REAL-ESTATE impairment write-down: NAREIT FFO excludes it alongside real-estate
    # D&A and sale gains, so FFO needs it as an ADD-BACK (a charge flow -> 0 in a normal
    # quarter). Without it FFO was understated in every year a REIT wrote a property down.
    "realEstateImpairment": ["ImpairmentOfRealEstate"],
    # ---- Energy (oil & gas) ----
    "oilGasRevenue": ["OilAndGasRevenue", "OilAndGasSalesRevenue"],  # E&P top line (pure players / pre-ASC-606)
    # `ExplorationAbandonmentAndDryHoleCosts` dropped (0 of 498 filers).
    "explorationExpense": ["ExplorationExpense",
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
    # ---- ASC-842 leases: BOTH legs of the liability + the offsetting ROU ASSET ----
    # The current portion of the finance-lease liability (41.6%) was missing, understating
    # debt / EV for lease-heavy names; the pre-2019 CAPITAL-lease elements were missing
    # entirely, so there was NO lease debt at all before ASC-842 adoption. Totals are
    # reconstructed in `_derive_history` (combined tag, else current + noncurrent, else
    # the pre-2019 capital-lease legs -- era-separated, so never double counted).
    "operatingLeaseLiability": ["OperatingLeaseLiability"],
    "operatingLeaseLiabilityCurrent": ["OperatingLeaseLiabilityCurrent"],
    "operatingLeaseLiabilityNoncurrent": ["OperatingLeaseLiabilityNoncurrent"],
    "financeLeaseLiability": ["FinanceLeaseLiability"],
    "financeLeaseLiabilityCurrent": ["FinanceLeaseLiabilityCurrent"],
    "financeLeaseLiabilityNoncurrent": ["FinanceLeaseLiabilityNoncurrent"],
    "capitalLeaseObligationCurrent": ["CapitalLeaseObligationsCurrent"],
    "capitalLeaseObligationNoncurrent": ["CapitalLeaseObligationsNoncurrent"],
    # The right-of-use ASSET (97.6% -- the single highest-coverage element the extractor
    # was missing). Two jobs: (a) the operating-side twin of the lease liability, which is
    # already treated as debt in EV / leverage, and (b) it is what makes `totalAssets` JUMP
    # at ASC-842 adoption (FY2019), a break that contaminated every assets-denominated
    # ratio (asset growth = the FF CMA factor, asset turnover, gross profitability,
    # accruals, Altman Z). `_derive_history` derives a break-free `totalAssetsExLease`.
    "operatingLeaseRouAsset": ["OperatingLeaseRightOfUseAsset"],
    "financeLeaseRouAsset": ["FinanceLeaseRightOfUseAsset"],
    # ---- cash quality: the pools netted OUT of clean cash (see the `cash` note) ----
    "restrictedCash": ["RestrictedCashAndCashEquivalents",
                       "RestrictedCashAndCashEquivalentsAtCarryingValue", "RestrictedCash"],
    "restrictedCashCurrent": ["RestrictedCashCurrent",
                              "RestrictedCashAndCashEquivalentsAtCarryingValueCurrent"],
    "restrictedCashNoncurrent": ["RestrictedCashAndCashEquivalentsNoncurrent",
                                 "RestrictedCashNoncurrent"],
    "cashInclRestricted": ["CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
                           "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalentsIncludingDisposalGroupAndDiscontinuedOperations"],
    "cashAndShortTermInvestments": ["CashCashEquivalentsAndShortTermInvestments"],
    # CURRENT marketable securities -> part of the non-operating liquid pool netted in EV.
    # Deliberately separate from `investmentSecurities` below: for a bank or insurer the
    # AFS/HTM book is the CORE operating asset (50-80% of assets), not excess cash, so it
    # must never be netted against enterprise value.
    "marketableSecuritiesCurrent": ["MarketableSecuritiesCurrent",
                                    "AvailableForSaleSecuritiesDebtSecuritiesCurrent",
                                    "AvailableForSaleSecuritiesCurrent",
                                    "OtherShortTermInvestments"],
    "investmentSecurities": ["AvailableForSaleSecuritiesDebtSecurities",
                             "AvailableForSaleSecurities", "TradingSecurities",
                             "EquitySecuritiesFvNiCurrentAndNoncurrent"],
    # MEZZANINE equity: redeemable NCI / temporary equity sits between debt and common and
    # belongs in EV alongside minority interest.
    "redeemableNCI": ["RedeemableNoncontrollingInterestEquityCarryingAmount",
                      "TemporaryEquityCarryingAmountAttributableToParent",
                      "TemporaryEquityCarryingAmount"],
    # ---- LIFO -> FIFO normalization (retail / industrial / refining) ----
    # FIFO inventory = LIFO inventory + LIFO reserve; FIFO COGS = LIFO COGS - the reserve's
    # increase. Without it a LIFO filer's inventory days, GMROI and gross margin are not
    # comparable to its FIFO peers.
    "lifoReserve": ["InventoryLIFOReserve", "ExcessOfReplacementOrCurrentCostsOverStatedLIFOValue"],
    # ---- asset-retirement obligations (energy / utilities / mining) ----
    # A debt-like decommissioning liability; added to the off-balance-sheet-inclusive
    # leverage pool, not to interest-bearing debt.
    "assetRetirementObligation": ["AssetRetirementObligation"],
    "aroCurrent": ["AssetRetirementObligationCurrent"],
    "aroNoncurrent": ["AssetRetirementObligationsNoncurrent"],
    # ---- DEBT MATURITY WALL (instant facts, ~81% coverage per year) ----
    # How much principal comes due each of the next five years. `refinancing_risk` used
    # only `shortTermDebt`, which misses a wall sitting 2-3 years out.
    # "Rolling" variant = a filer disclosing the ladder from the balance-sheet date
    # forward (rolling 12 months) rather than by fixed fiscal year -- a distinct
    # 2019+ us-gaap concept, not an alternate name for the same fact; several filers
    # (confirmed: MAA's operating-lease ladder) use ONLY this variant.
    "debtMaturity1y": ["LongTermDebtMaturitiesRepaymentsOfPrincipalInNextTwelveMonths",
                       "LongTermDebtMaturitiesRepaymentsOfPrincipalInNextRollingTwelveMonths"],
    "debtMaturity2y": ["LongTermDebtMaturitiesRepaymentsOfPrincipalInYearTwo"],
    "debtMaturity3y": ["LongTermDebtMaturitiesRepaymentsOfPrincipalInYearThree"],
    "debtMaturity4y": ["LongTermDebtMaturitiesRepaymentsOfPrincipalInYearFour"],
    "debtMaturity5y": ["LongTermDebtMaturitiesRepaymentsOfPrincipalInYearFive"],
    "debtMaturityAfter5y": ["LongTermDebtMaturitiesRepaymentsOfPrincipalAfterYearFive"],
    # ---- OPERATING-LEASE maturity ladder: the missing half of the refinancing wall ----
    # `utils/capital.py` already counts leases as DEBT, but only the lease LIABILITY was
    # extracted -- never its maturity profile. So for a retailer, airline or restaurant
    # chain, where the operating-lease ladder IS the wall, `debtMaturity*` described only
    # the bond half. Each rung coalesces BOTH accounting eras, which the tag scan measured
    # as cleanly DISJOINT: `OperatingLeasesFutureMinimumPayments*` runs 2009-2021 (406-451
    # tickers) and the ASC-842 `LesseeOperatingLeaseLiabilityPaymentsDue*` runs 2017-2026
    # (468-478 tickers), overlapping only in the 4-5 transition years. Mapping one era
    # alone would put a structural break in every lease-wall feature at FY2019 adoption --
    # exactly the like-for-like failure this pass was looking for. Union-coalesced per
    # period (never first-present), so the transition years take whichever era the filer
    # actually used.
    "leaseMaturity1y": ["LesseeOperatingLeaseLiabilityPaymentsDueNextTwelveMonths",
                        "LesseeOperatingLeaseLiabilityPaymentsDueNextRollingTwelveMonths",
                        "OperatingLeasesFutureMinimumPaymentsDueCurrent"],
    "leaseMaturity2y": ["LesseeOperatingLeaseLiabilityPaymentsDueYearTwo",
                        "OperatingLeasesFutureMinimumPaymentsDueInTwoYears"],
    "leaseMaturity3y": ["LesseeOperatingLeaseLiabilityPaymentsDueYearThree",
                        "OperatingLeasesFutureMinimumPaymentsDueInThreeYears"],
    "leaseMaturity4y": ["LesseeOperatingLeaseLiabilityPaymentsDueYearFour",
                        "OperatingLeasesFutureMinimumPaymentsDueInFourYears"],
    "leaseMaturity5y": ["LesseeOperatingLeaseLiabilityPaymentsDueYearFive",
                        "OperatingLeasesFutureMinimumPaymentsDueInFiveYears"],
    "leaseMaturityAfter5y": ["LesseeOperatingLeaseLiabilityPaymentsDueAfterYearFive",
                             "OperatingLeasesFutureMinimumPaymentsDueThereafter"],
    "leaseMaturityTotal": ["LesseeOperatingLeaseLiabilityPaymentsDue",
                           "OperatingLeasesFutureMinimumPaymentsDue"],
    # undiscounted total minus the recognised liability = the imputed lease INTEREST
    "leaseUndiscountedExcess": ["LesseeOperatingLeaseLiabilityUndiscountedExcessAmount"],
    # ---- balance-sheet FOOTING: the reported other side of the identity ----
    # `LiabilitiesAndStockholdersEquity` is tagged by ALL 498 tickers over 2007-2026 and was
    # unmapped. It is what the filer PUBLISHED as the footing, so it turns the balance-sheet
    # identity from an inference over three separately-coalesced columns into a check
    # against the statement itself -- and it is a fallback for `totalAssets` on the stub
    # filings that report one side only.
    "balanceSheetFooting": ["LiabilitiesAndStockholdersEquity"],
    # the residual buckets a balance sheet needs to foot
    "otherAssetsNoncurrent": ["OtherAssetsNoncurrent"],
    "otherLiabilitiesNoncurrent": ["OtherLiabilitiesNoncurrent"],
    # ---- OUTSTANDING ITEMS (share count is a level, not a flow) ----
    # `sharesOutstanding` alone cannot distinguish a buyback from a share-count restatement;
    # issued vs authorised gives the headroom, and the antidilutive count is the overhang
    # that never reaches diluted EPS.
    "commonSharesIssued": ["CommonStockSharesIssued"],
    "commonSharesAuthorized": ["CommonStockSharesAuthorized"],
    "commonStockValue": ["CommonStockValue"],
    "preferredSharesAuthorized": ["PreferredStockSharesAuthorized"],
    "antidilutiveShares": [
        "AntidilutiveSecuritiesExcludedFromComputationOfEarningsPerShareAmount"],
    # ---- deferred-tax detail (near-universal, previously unmapped) ----
    "deferredTaxAssetsGross": ["DeferredTaxAssetsGross"],
    "deferredTaxNet": ["DeferredTaxAssetsLiabilitiesNet"],
    # ---- forward INTANGIBLE-amortisation ladder (a known future earnings drag) ----
    "intangibleAmort1y": ["FiniteLivedIntangibleAssetsAmortizationExpenseNextTwelveMonths"],
    "intangibleAmort2y": ["FiniteLivedIntangibleAssetsAmortizationExpenseYearTwo"],
    "intangibleAmort3y": ["FiniteLivedIntangibleAssetsAmortizationExpenseYearThree"],
    "intangibleAmort4y": ["FiniteLivedIntangibleAssetsAmortizationExpenseYearFour"],
    "intangibleAmort5y": ["FiniteLivedIntangibleAssetsAmortizationExpenseYearFive"],
    # ---- deferred tax / tax-aggressiveness levels ----
    "deferredTaxAssets": ["DeferredTaxAssetsNet", "DeferredIncomeTaxAssetsNet"],
    "deferredTaxLiabilities": ["DeferredIncomeTaxLiabilitiesNet", "DeferredTaxLiabilities"],
    "valuationAllowance": ["DeferredTaxAssetsValuationAllowance"],
    "unrecognizedTaxBenefits": ["UnrecognizedTaxBenefits"],
    "allowanceDoubtfulAccounts": ["AllowanceForDoubtfulAccountsReceivableCurrent"],
    "intangiblesGross": ["FiniteLivedIntangibleAssetsGross"],
    "intangiblesAccumAmort": ["FiniteLivedIntangibleAssetsAccumulatedAmortization"],
    "accountsReceivable": ["AccountsReceivableNetCurrent", "ReceivablesNetCurrent"],
    # as-reported (LIFO-basis for a LIFO filer) inventory. `LIFOInventoryAmount` is the
    # carrying value LIFO filers tag when they leave `InventoryNet` sparse -- Kroger tags
    # InventoryNet in only 4 filings but the LIFO/FIFO pair in 140 each.
    "inventory": ["InventoryNet", "LIFOInventoryAmount"],
    # inventory already stated at FIFO by the filer -> the FIFO target directly, used to
    # fill the normalization where the LIFO-basis line is missing.
    "inventoryFifoReported": ["FIFOInventoryAmount"],
    "accountsPayable": ["AccountsPayableCurrent", "AccountsPayableAndAccruedLiabilitiesCurrent"],
    # net PP&E is the common balance-sheet line; gross is reconstructed as
    # net + accumulated depreciation in build_ticker_history for net-only filers.
    # ASC-842 combined PP&E + finance-lease ROU tags (103-133 tickers) are the post-2019
    # presentation for filers that fold the finance-lease asset into the PP&E line. Without
    # them those names read a PP&E base that excludes leased capacity, while `capital.py`
    # counts the lease liability as debt -- an asymmetry in leverage and asset-turnover.
    "ppeNet": ["PropertyPlantAndEquipmentNet",
               "PropertyPlantAndEquipmentAndFinanceLeaseRightOfUseAssetAfterAccumulated"
               "DepreciationAndAmortization"],
    "ppeGross": ["PropertyPlantAndEquipmentGross",
                 "PropertyPlantAndEquipmentAndFinanceLeaseRightOfUseAssetBeforeAccumulated"
                 "DepreciationAndAmortization"],
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
    # The CECL-era amortized-cost elements are added because the pre-2020 name alone
    # left the amortized-cost leg (15% of Financials) THINNER than the fair-value leg
    # (32%), so the difference was NaN for most banks. `...AmortizedCostAfterAllowance-
    # ForCreditLoss` without the `DebtSecurities` prefix is not an element (0 of 498).
    "htmSecurities": ["HeldToMaturitySecurities",
                      "DebtSecuritiesHeldToMaturityExcludingAccruedInterestAfterAllowanceForCreditLoss",
                      "DebtSecuritiesHeldToMaturityAmortizedCostAfterAllowanceForCreditLoss"],
    "htmSecuritiesFairValue": ["HeldToMaturitySecuritiesFairValue"],
    # the unrecognized HTM holding LOSS as disclosed directly (42 filers) -- the same
    # quantity as amortized-cost minus fair-value but available without needing both
    # legs, so it fills the gap where only one of them is tagged.
    "htmUnrealizedLoss": ["HeldToMaturitySecuritiesAccumulatedUnrecognizedHoldingLoss"],
    # non-performing (nonaccrual) loans -> forward credit-quality (B3). The CECL-era
    # name is `...ExcludingAccruedInterestNonaccrual`; the `...NonaccrualStatus` suffix
    # variant previously listed is not an element (0 of 498 filers).
    "nonaccrualLoans": ["FinancingReceivableRecordedInvestmentNonaccrualStatus",
                        "FinancingReceivableExcludingAccruedInterestNonaccrual"],
    # Tier-1 risk-based ratio, falling back to the CET1 ratio (CET1 <= Tier1, so the
    # ">11% = well-capitalised" screen stays conservative).
    # NOTE `CommonEquityTierOneCapitalToRiskWeightedAssets` was dropped: 0 of 498. Modern
    # CET1 is tagged with a legal-entity dimension (holdco vs bank sub) and companyfacts
    # serves only UNDIMENSIONED facts, so it is structurally unavailable from this source
    # -- the column stays ~8% of Financials until it is read from the Financial Statement
    # Data Sets (num.tsv keeps `dimn`), which is tracked separately.
    "tier1CapitalRatio": ["TierOneRiskBasedCapitalToRiskWeightedAssets",
                          "CommonEquityTierOneCapitalRatio"],
    # ---- Insurance ----
    "insuranceReserves": ["LiabilityForClaimsAndClaimsAdjustmentExpense",
                          "LiabilityForFuturePolicyBenefits"],
    # singular `DeferredPolicyAcquisitionCost` dropped (0 of 498 filers).
    "deferredAcqCosts": ["DeferredPolicyAcquisitionCosts"],
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
# value point-in-time) -> true per-share + net-issuance signals.
# The `...Adjustment` variant was dropped (0 of 498 filers).
DILUTED_SHARES_TAGS = ["WeightedAverageNumberOfDilutedSharesOutstanding"]

# DURATION facts that must be taken as the LATEST reported value, never TTM-SUMMED:
# a weighted-average share count is already a period figure, and summing four quarters
# of "3 reportable segments" would report 12. Handled by the same as-of path as the
# cover-page share count.
LATEST_DURATION_TAGS: dict[str, list[str]] = {
    "dilutedShares": DILUTED_SHARES_TAGS,
    # basic shares (98.6%): with diluted it gives the OPTION-OVERHANG wedge
    # (diluted - basic) / basic, which net share count hides once buybacks offset SBC.
    "basicShares": ["WeightedAverageNumberOfSharesOutstandingBasic",
                    "WeightedAverageNumberOfShareOutstandingBasicAndDiluted"],
    # conglomerate complexity / breakup-value proxy (92.2%)
    "reportableSegments": ["NumberOfReportableSegments"],
    # Effective tax rate on continuing operations -- tagged by 481 of 500 tickers with
    # 52,268 facts, and previously unmapped entirely. It belongs HERE, not in
    # EXTRA_FLOW_TAGS: it is a RATIO, and every EXTRA_FLOW field is TTM-summed by `ttm_a`,
    # which would add four quarterly rates into a nonsense ~0.84 "rate". Taking the latest
    # reported duration fact keeps it point-in-time and on its natural scale.
    "effectiveTaxRate": ["EffectiveIncomeTaxRateContinuingOperations"],
}

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
                # 0 in a quarter with no property write-down -> FFO's impairment
                # add-back is defined for every REIT quarter, not only the bad ones.
                "realEstateImpairment",
                # widened non-recurring pool (see EXTRA_FLOW_TAGS): 0 in a normal
                # quarter, non-zero only when the event occurs.
                "goodwillImpairment", "gainOnSaleGeneric", "litigationExpense",
                "discontinuedOps", "unusualItems", "bargainPurchaseGain",
                # 0 when a filer funds no capacity via finance leases -> capexGlobal = cash capex
                "financeLeaseAdditions",
                # operating-lease additions: 0 for a filer that signs none in the quarter
                "operatingLeaseAdditions",
                # event gains/charges reported only when they occur (0 otherwise), all of
                # which the core-earnings normalization strips out
                "debtExtinguishment", "realizedInvestmentGains", "goodwillAcquired",
                # REIT AFFO adjustments: absent = no such non-cash item this quarter
                "straightLineRent", "aboveBelowMarketLeaseAmort"}

# ASU 2017-07 (non-service pension cost presented OUTSIDE operating income) is effective
# for fiscal years beginning after 15-Dec-2017, i.e. FY2018. Fiscal ends before this date
# still carry the non-service components inside operating income and are restated.
ASU_2017_07_EFFECTIVE = "2018-01-01"
ANNUAL_MIN_DAYS, ANNUAL_MAX_DAYS = 340, 380   # accept a fiscal year as ~365d
QUARTER_MIN_DAYS, QUARTER_MAX_DAYS = 80, 100   # accept a fiscal quarter as ~13 weeks
# Wider window used specifically when DE-CUMULATING a YTD-cumulative flow into a
# discrete quarter (see fundamentals_periods.py / the old _quarterly_flow): 120d (not
# 100) so 52/53-week retailers with a 16-week fiscal Q1 (~112d, e.g. KR/COST/TGT)
# still yield 4 usable quarters instead of being dropped. 6-month YTD (181d) stays
# excluded either way.
IMPLIED_QUARTER_MIN_DAYS, IMPLIED_QUARTER_MAX_DAYS = 75, 120
TTM_QUARTERS = 4                               # trailing-twelve-months = 4 quarters
# Gross margin is mathematically <= 1 (a value > 1 implies negative COGS); a value
# below -200% only arises when revenue is truncated / period-mismatched vs the cost
# line (e.g. a REIT whose rental income moved to the ASC-842 lease-income tag). Values
# outside this band are nulled as extraction artifacts rather than shipped as features.
GROSS_MARGIN_MIN, GROSS_MARGIN_MAX = -2.0, 1.0
