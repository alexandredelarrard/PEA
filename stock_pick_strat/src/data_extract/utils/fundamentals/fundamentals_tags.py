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
                      "CostOfServicesEnergyServices", "CostOfDomesticRegulatedElectric",
                      # the post-2016 name for the same regulated-utility cost line
                      # (fuel + purchased power + gas): DTE tagged
                      # `CostOfGoodsAndServicesEnergyCommoditiesAndServices` only
                      # through 2015-Q3 and this element from 2017-Q2, so without it
                      # the cost line -- and every margin built on it -- simply ended
                      # mid-history.
                      "UtilitiesOperatingExpenseProductsAndServices"],
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
                 "CostOfGoodsSoldDepreciationDepletionAndAmortization",
                 # the D&A line INSURERS and BANKS tag instead of any of the above
                 # ("other" only in the sense that it sits outside their interest /
                 # benefits lines -- it IS their total D&A). Fill-only, so a filer
                 # using a standard tag is untouched. Without it these filers had no
                 # quarterly D&A at all: MetLife tags it 53 times over 2011-2025
                 # (median $693M/period) and had depAmort only from 2024, Travelers
                 # 25 times over 2011-2015 (median $642M) and had it only from 2018,
                 # M&T 33 times over 2011-2023 (median $165M) and had it only from
                 # 2015.
                 "OtherDepreciationAndAmortization",
                 # Cisco's cash-flow D&A line ($1,902M in its 2026-Q3 10-Q): it tags
                 # `DepreciationDepletionAndAmortization` ANNUALLY in the 10-K but only
                 # this element in its 10-Qs, so depAmort existed as an `annual` row and
                 # was null on every single quarter. A filer EXTENSION rather than a
                 # us-gaap element (`csco:DepreciationAmortizationAndOther`), which
                 # matches because resolution is by bare concept name, namespace-agnostic
                 # (see `build_tag_frames`) -- safe here because the name is specific
                 # enough that another filer coining it means the same line.
                 # Fill-only, lowest priority.
                 "DepreciationAmortizationAndOther"],
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
              "PaymentsToAcquireMachineryAndEquipment",
              # BANKS tag premises-and-equipment spend under the productive-assets
              # elements rather than the generic PP&E one, and only these two appear
              # in their 10-Qs -- so capex was `annual`-only (M&T: 11 annual rows,
              # zero quarters) or entirely absent (Regions). Both carry the standard
              # outflow-positive convention, matching the elements above. Confirmed:
              # M&T $96M and Regions in their 2026-Q1 10-Qs.
              "PaymentsForProceedsFromProductiveAssets",
              "PaymentsToAcquireOtherProductiveAssets"],
    "researchAndDevelopment": ["ResearchAndDevelopmentExpense",
                               "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost"],
    # ---- added for refined features (S&M efficiency, M&A, SBC, distress) ----
    # `MarketingAndAdvertisingExpense` was REMOVED from this coalesce and given
    # its own field (`marketingExpense`, below): advertising spend is a
    # COMPONENT of SG&A, never the whole line, so as a last-resort fallback it
    # populated a tiny "SG&A" for filers that report no SG&A line at all.
    # Confirmed on Citigroup, whose entire 59-quarter `sellingGeneralAdmin`
    # series resolved from it -- a bank reports noninterest expense, not SG&A,
    # so the correct value there is NULL (and `noninterestExpense` already
    # carries the real figure).
    "sellingGeneralAdmin": ["SellingGeneralAndAdministrativeExpense",
                            "GeneralAndAdministrativeExpense",
                            "SellingAndMarketingExpense"],
    "stockBasedComp": ["ShareBasedCompensation",
                       "AllocatedShareBasedCompensationExpense"],
    "acquisitions": ["PaymentsToAcquireBusinessesNetOfCashAcquired",
                     "PaymentsToAcquireBusinessesAndInterestInAffiliates"],
    "interestExpense": ["InterestExpense", "InterestAndDebtExpense",
                        "InterestExpenseNonoperating",
                        # the GROSS interest-on-debt element some filers tag instead of the
                        # generic one (confirmed: ORLY, $62.7M in its 2026-Q1 10-Q, which had
                        # no `InterestExpense` anywhere -> interestExpense was null for its
                        # ENTIRE quarterly history). Deliberately NOT added: the NET elements
                        # (`InterestIncomeExpenseNet`, `InterestIncomeExpense-
                        # NonoperatingNet`), which is all NWSA/PKG/TJX/URI/ZBH tag quarterly
                        # -- they net interest INCOME off and carry the opposite sign
                        # convention, so filling this field from them would silently mix two
                        # different measures across the panel.
                        "InterestExpenseDebt"],
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
    # total (not "other") noncurrent liabilities -- some filers (confirmed: ACN, ADI,
    # ADM) never tag a combined `Liabilities` total at all, only this + currentLiabilities;
    # `fundamentals_periods.derive_missing_total_liabilities` sums the two into
    # `totalLiabilities` whenever the as-reported total is absent for a period.
    "totalLiabilitiesNoncurrent": ["LiabilitiesNoncurrent"],
    "goodwill": ["Goodwill"],
    "totalAssets": ["Assets"],
}
# Logical field name for the point-in-time share count, named here beside its tag
# list (same convention as EMPLOYEES_FIELD) because `fetch_fundamentals_edgar.
# _cover_page_shares_fallback` looks the field up by name and must never drift
# from the key below.
SHARES_OUTSTANDING_FIELD = "sharesOutstanding"
SHARES_TAGS = {
    # STRICTLY POINT-IN-TIME. `WeightedAverageNumberOfDilutedSharesOutstanding` was
    # removed: it is a period AVERAGE on a DILUTED basis -- a different quantity from
    # the share count outstanding on a date -- and it already has its own field
    # (`dilutedShares`, via DILUTED_SHARES_TAGS), so its presence here only ever
    # duplicated it under a name that means something else.
    #
    # It was not a harmless last-resort fallback either. Being a DURATION fact it
    # carries a period_start, while the two counts below are INSTANT facts and do
    # not -- so `build_tag_frames`' priority coalesce, which groups by (field,
    # period_start, period_end), never compared them and emitted BOTH (three rows
    # for one field in a 10-Q: instant + discrete-quarter + YTD). `instant_stock`'s
    # drop_duplicates then kept whichever came first, so the stored measure was
    # decided by frame ordering: 2,452 rows table-wide resolved to the diluted
    # AVERAGE against 2,056 to a genuine point-in-time count, alternating within
    # single tickers' histories. Since market capitalisation is computed from this
    # field, that is a systematic cross-sectional bias, not noise.
    #
    # `CommonStockSharesOutstanding` (balance sheet, dated exactly at period end) is
    # preferred over the cover-page count so the measurement DATE is consistent with
    # every other instant field. The cover-page count backs it up for the filers that
    # tag no balance-sheet share count -- see `_cover_page_shares_fallback`, which is
    # what makes it reachable at all (its context date is the FILING date, so the
    # current-period filter drops it).
    SHARES_OUTSTANDING_FIELD: ["CommonStockSharesOutstanding",
                               "EntityCommonStockSharesOutstanding"],
}
# The cover-page share count is stated "as of" a date a few weeks AFTER the period it
# reports on (SEC filing deadlines: 40 days after quarter end, 60-90 after fiscal year
# end, plus late filings), never before it. Facts outside that forward window are a
# different filing's and are ignored.
COVER_PAGE_SHARES_TAG = "EntityCommonStockSharesOutstanding"
COVER_PAGE_SHARES_MAX_LAG_DAYS = 150

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
    # (integrated oil pre-restructuring, e.g. COP 2012-2016). `OperatingCosts-
    # AndExpenses` is the same subtotal under the name REITs use (confirmed:
    # REG, 2017 onward) -- REITs almost never tag `OperatingIncomeLoss`, so
    # without it their operating line is underivable.
    "costsAndExpenses": ["CostsAndExpenses", "OperatingCostsAndExpenses"],
    # advertising / marketing spend in its own right -- a COMPONENT of SG&A, not
    # a substitute for it (see the `sellingGeneralAdmin` note above).
    "marketingExpense": ["MarketingAndAdvertisingExpense"],
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
    "intangiblesGross": ["FiniteLivedIntangibleAssetsGross",
                         # the equivalent total for a filer that does not split its
                         # intangibles by useful life (confirmed: JPM, whose
                         # intangiblesGross was null for its whole history)
                         "IntangibleAssetsGrossExcludingGoodwill"],
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
    # `ShortTermInvestments` alone left this field null for 18 of 42 audited
    # tickers. The variants below are the SAME balance-sheet line under the name
    # the filer happens to use, all explicitly scoped to CURRENT (or "short-term")
    # so a long-term securities portfolio can never leak in -- confirmed in live
    # 10-Qs: SYK `AvailableForSaleSecuritiesDebtSecuritiesCurrent` $85M (+ its
    # `OtherShortTermInvestments`), CBOE `MarketableSecuritiesCurrent` $114.5M, PKG
    # $146.6M. Deliberately NOT added: bare `MarketableSecurities` (the only element
    # ATO tags, and the only one with no current/noncurrent scope) -- it can equally
    # be a long-term portfolio, and DTE/GLW/JCI/VLO/ORLY tag no short-term
    # securities line at ALL quarterly, so those stay null rather than guessed.
    "shortTermInvestments": ["ShortTermInvestments",
                             "AvailableForSaleSecuritiesDebtSecuritiesCurrent",
                             "AvailableForSaleSecuritiesCurrent",
                             "MarketableSecuritiesCurrent",
                             "ShortTermMarketableDebtSecurities",
                             "OtherShortTermInvestments"],
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
    # a bank's OWN deposits held AT other banks (an ASSET -- not the `deposits`
    # liability above). Together with `CashAndDueFromBanks` it reconstructs the
    # cash-and-equivalents total exactly (see
    # `fundamentals_periods.derive_bank_cash`), which is what several banks stop
    # tagging directly mid-history.
    "interestBearingDepositsInBanks": ["InterestBearingDepositsInBanks"],
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

# --------------------------------------------------------------------------- #
# PARTIAL-TOP-LINE GUARDS. The ASC-606 contract-revenue elements are the highest-
# priority `totalRevenue` candidates because they ARE the whole top line for a
# modern product/services filer -- but they are only a SLICE of it for anyone
# whose revenue mostly falls outside ASC 606, and the fact itself looks perfectly
# valid (undimensioned, correctly tagged, right period). Consumed by
# `fetch_fundamentals_edgar.build_tag_frames`, which excludes the slice so the
# candidate list falls through to the filer's real total.
# --------------------------------------------------------------------------- #
PARTIAL_REVENUE_TAGS = {"RevenueFromContractWithCustomerExcludingAssessedTax",
                        "RevenueFromContractWithCustomerIncludingAssessedTax"}
# the us-gaap whole-company top line the slice is measured against
TOTAL_REVENUE_TAG = "Revenues"
# How many times bigger `Revenues` must be before the contract element is judged
# a fee SLICE rather than the same top line under a different name. Set from the
# measured distribution over 180 extracted tickers, which is cleanly bimodal:
# 253 quarters across 15 tickers sit ABOVE 3x (REITs and insurers whose revenue
# is almost entirely outside ASC 606 -- UDR 210x, MET 37x, HUM 31x, WRB 25x,
# CCI 16x, SBAC 14x), while 113 quarters across 12 tickers sit between 1.05x and
# 1.5x (energy/utility filers -- DVN, OXY, TRGP, D, SRE, DTE, RSG -- where the
# two totals genuinely differ only by a reconciling item). Only the first group
# is a mis-mapping; re-basing the second would change a defensible number and,
# because the ratio there drifts period to period, could switch concept
# mid-history and put a step in the series.
PARTIAL_REVENUE_MATERIALITY = 3.0
# Concepts a bank or insurer tags for its interest/premium top line. When one of
# these DOMINATES the filing (see `FINANCIALS_TOPLINE_DOMINANCE` below), the filer's
# revenue is interest/premium income, so an ASC-606 contract element in the same
# filing is fee income at best -- and, confirmed on RF, sometimes literally $0.
#
# PRESENCE ALONE IS NOT ENOUGH, and assuming it was is a confirmed, severe bug: the
# original rule nulled `totalRevenue` for any filer tagging any of these ANYWHERE in
# a filing, on the stated assumption that they have "NO non-financial usage". Two of
# them plainly do. `InterestIncomeExpenseNet` is tagged by 12 of the 22 non-financial
# filers measured (ZBH, PKG, WAB, BKR, SPGI, ATO, MPC, NWSA, PFE, SJM, SYK, TJX,
# ECHO) -- for an industrial it is NET INTEREST EXPENSE, a small NEGATIVE number
# (Zimmer Biomet 2022-Q2: -$38.8M against $1,781.8M of real revenue) -- and
# `InterestAndDividendIncomeOperating` by EA and O'Reilly. The result was silent
# total loss of the top line for every such filer that tags ONLY the ASC-606 element
# (nothing to fall through to): ZBH and PKG lost 2019+, BKR 2018+, SPGI 2019-2022,
# and WAB lost 2019-2023 and then SPONTANEOUSLY REAPPEARED in 2024 when it stopped
# tagging the marker -- exactly the ticker-by-ticker instability that makes a feature
# unusable in production.
FINANCIALS_TOPLINE_MARKERS = {"InterestAndDividendIncomeOperating", "NoninterestIncome",
                              "InterestIncomeExpenseNet", "PremiumsEarnedNet",
                              "PremiumsEarnedNetPropertyAndCasualty"}
# How many times bigger a marker line must be than the ASC-606 contract element, FOR
# THE SAME PERIOD, before the filer counts as a bank/insurer whose contract element is
# a mere fee slice. Mirrors `PARTIAL_REVENUE_MATERIALITY` (and the `Revenues`-outranks
# test it powers) so both halves of the partial-revenue guard ask the same economic
# question -- "is this element dwarfed by the filer's real top line?" -- instead of
# one asking a question about element NAMES, which is what broke. A genuine bank
# clears it easily (Regions tags the contract element as $0.00 against ~$1.7B of
# quarterly net interest income); an industrial's small, usually negative net
# interest line never does.
FINANCIALS_TOPLINE_DOMINANCE = PARTIAL_REVENUE_MATERIALITY

# Some filers report "Selling and marketing" and "General and administrative" as two
# SEPARATE, ADDITIVE income-statement lines rather than one combined SG&A concept --
# confirmed via the Tiingo cross-check: CRM (Salesforce) tags ONLY
# `GeneralAndAdministrativeExpense` ($632-767M/quarter) and `SellingAndMarketingExpense`
# ($3.1-3.3B/quarter) simultaneously, both undimensioned, for the SAME period, and never
# tags the combined `SellingGeneralAndAdministrativeExpense` concept at all. The normal
# priority coalesce in `fetch_fundamentals_edgar.build_tag_frames` picks G&A alone
# (priority 1 in `sellingGeneralAdmin`'s candidate list below) and never looks at the
# companion (priority 2) once ONE candidate has already answered the period -- correctly
# reproducing "keep the higher-priority candidate" for every OTHER field, but
# understating `sellingGeneralAdmin` by ~5x here (Tiingo's normalized `sga` matched CRM's
# own G&A + S&M SUM to the dollar). Consumed by `build_tag_frames` to ADD the companion
# ONLY when the winning fact is G&A-ONLY -- the combined tag, when present, already wins
# outright via priority and must never be added to twice. Only the confirmed direction
# (G&A wins, S&M companion) is handled; the untested reverse (S&M wins, G&A companion)
# is deliberately left alone until evidence surfaces for it too.
SGA_GA_ONLY_TAG = "GeneralAndAdministrativeExpense"
SGA_SM_COMPANION_TAG = "SellingAndMarketingExpense"

# diluted weighted-average shares (duration fact; we take the latest period's
# value point-in-time) -> true per-share + net-issuance signals.
# The `...Adjustment` variant was dropped (0 of 498 filers).
DILUTED_SHARES_TAGS = ["WeightedAverageNumberOfDilutedSharesOutstanding"]

# Logical field name for the workforce headcount. Named here, beside every other
# field name, because it is referenced from BOTH halves of the split described on
# its entry below (`fundamentals_employees.py` produces the fact,
# `fundamentals_derive.py` consumes it) and must never drift between them.
EMPLOYEES_FIELD = "employees"

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
    # WORKFORCE HEADCOUNT -- the one field here with NO XBRL tag at all, hence the
    # deliberately EMPTY candidate list. There is no GAAP concept for headcount
    # (`dei:EntityNumberOfEmployees` exists but US filers essentially never tag it),
    # so it is parsed out of the 10-K BODY TEXT by `fundamentals_employees.py` and
    # appended as a ready-made fact row. The empty list is what keeps the two halves
    # apart: `fetch_fundamentals_edgar.build_tag_frames` contributes no candidate tag
    # for this field and never looks for one, while EVERY consumer that iterates this
    # dict by KEY picks `employees` up for free -- `_assemble_base`'s latest-value
    # ffill (right for an annually-disclosed level: it carries into the interim
    # quarters instead of leaving them blank), `_derive_history`'s output column, and
    # `fundamentals_derive`'s per-field series. It belongs in THIS dict rather than
    # EXTRA_STOCK_TAGS for exactly that ffill: a stock field is joined on the exact
    # period end, which would leave headcount populated only on Q4 rows.
    EMPLOYEES_FIELD: [],
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

# Flow fields whose quarterly value is economically incapable of being NEGATIVE: a
# top line, a cost/expense line, or a cash amount PAID (all reported as positive
# magnitudes). Used by `fundamentals_periods._q4_is_coherent` as the sharpest
# available test of a derived Q4 -- a negative here is arithmetically impossible, so
# it always means the FY and the quarters measured different things, no magnitude
# heuristic required. Deliberately EXCLUDES every genuinely signed concept (income,
# EPS, cash-flow subtotals, gains/losses, working-capital movements, other
# comprehensive income): a real loss-making or charge-laden fourth quarter is common
# and must never be nulled -- that over-strictness is exactly what this pass fixed.
NON_NEGATIVE_FLOW_FIELDS = {
    # top lines
    "totalRevenue", "revenuesTotal", "revenueExcludingAssessedTax",
    "revenueIncludingAssessedTax", "oilGasRevenue", "rentalIncome", "premiumsEarned",
    "premiumsWritten", "interestIncomeBank", "noninterestIncome",
    # ACCRUAL expense lines -- cumulative by construction, so a negative discrete
    # quarter is not a business event but a concept mismatch
    "costOfRevenue", "sellingGeneralAdmin", "researchAndDevelopment", "marketingExpense",
    "depAmort", "depletionDDA", "amortizationIntangibles", "noninterestExpense",
    "operatingLeaseCost", "capex",
}
# Deliberately EXCLUDED after checking the rebuilt tickers for negatives: CASH and
# event flows go negative for real reasons, so nulling them would delete good data.
# Confirmed as-reported (derived=0) negatives on live filings: `incomeTaxesPaid` (a
# net refund -- VLO 2021-Q2 -$918M, MTB 2016-Q1, TRV 2023-Q1, MET 2020-Q1),
# `acquisitions` (cash acquired exceeding consideration -- MTB 2011-Q2, SPGI 2022-Q1,
# PKG, VLO), `stockBasedComp` (forfeiture reversal -- WAB 2020-Q2), `interestExpense`
# (WAB 2011-2012) and `dacAmortization` (MET 2023 under LDTI). Every financing flow
# (buybacks / dividendsPaid / debtIssued / debtRepaid / equityIssuance) is likewise
# excluded: they are YTD-cumulative and net, so a quarter can legitimately reverse.

# Balance-sheet STOCK fields that are MAGNITUDES, never signed quantities: an amount
# owed, an amount owned, a count of shares. A negative as-reported value for one of
# these is always a filer tagging defect, never a business fact -- so
# `fetch_fundamentals_edgar.build_tag_frames` treats such a fact as INADMISSIBLE,
# which lets the field's candidate coalesce fall through to the next tag (and leaves
# the field NULL when no other candidate reported it). Rejecting rather than
# sign-flipping is deliberate: `abs()` would silently rewrite the filer's number, and
# the negation is not always the whole story (see below).
#
# Confirmed on DTE (the case this set was added for): its FY2011 and FY2012 10-Ks tag
# `us-gaap:DebtCurrent` with a NEGATIVE value (-$355M, -$634M) because the concept is
# used for the "Less amount due within one year" DEDUCTION row of the long-term-debt
# footnote, and the filer baked the presentation sign into the instance document. As
# priority-0 candidate for `shortTermDebt` it won outright, so DTE's stored short-term
# debt was negative for those two years. Both are undimensioned, so no dimension rule
# could catch it, and from FY2013 on the SAME line is tagged positive, so no
# per-filer rule could either -- the sign is the only signal available.
#
# Deliberately EXCLUDED, because a negative is REAL for them and nulling it would
# delete good data: `stockholdersEquity` / `retainedEarnings` / `accumulatedOCI`
# (buyback- or loss-driven deficits), `treasuryStock` and `accumulatedDepreciation`
# (contra-accounts filers commonly tag negative on purpose), `deferredTaxNet`,
# `pensionDeficit`, `lifoReserve`, `leaseUndiscountedExcess` and the `allowance*`
# contra-asset lines.
NON_NEGATIVE_STOCK_FIELDS = {
    # debt & lease obligations
    "shortTermDebt", "longTermDebt", "longTermDebtTotal", "debtCombined", "commercialPaper",
    "notesPayable", "capitalLeaseObligationCurrent", "capitalLeaseObligationNoncurrent",
    "financeLeaseLiability", "financeLeaseLiabilityCurrent", "financeLeaseLiabilityNoncurrent",
    "operatingLeaseLiability", "operatingLeaseLiabilityCurrent", "operatingLeaseLiabilityNoncurrent",
    "debtMaturity1y", "debtMaturity2y", "debtMaturity3y", "debtMaturity4y", "debtMaturity5y",
    "debtMaturityAfter5y",
    # assets
    "totalAssets", "currentAssets", "cash", "cashInclRestricted", "cashAndShortTermInvestments",
    "restrictedCash", "restrictedCashCurrent", "restrictedCashNoncurrent",
    "shortTermInvestments", "longTermInvestments", "marketableSecuritiesCurrent",
    "investmentSecurities", "htmSecurities", "loans", "accountsReceivable", "inventory",
    "goodwill", "intangiblesGross", "intangiblesExGoodwill", "capitalizedSoftware",
    "ppeGross", "ppeNet", "realEstateGross", "realEstateNet",
    "oilGasPropertyGross", "oilGasPropertyNet",
    # liabilities
    "totalLiabilities", "currentLiabilities", "totalLiabilitiesNoncurrent", "accountsPayable",
    "deferredRevenue", "deferredRevenueCurrent", "deferredRevenueNoncurrent",
    "deposits", "depositsDomestic", "insuranceReserves",
    "aroCurrent", "aroNoncurrent", "assetRetirementObligation",
    # share counts
    "sharesOutstanding", "commonSharesIssued", "commonSharesAuthorized",
    "preferredSharesAuthorized", "antidilutiveShares",
}

# A share-count-shaped field reporting a MAGNITUDE below `SHARE_COUNT_MIN_ABS` is a filer
# scale defect, never a real business fact -- no S&P/Dow constituent has a weighted-average
# or point-in-time share count under a million. Confirmed on MCD: its FY2024 10-Qs tag
# `us-gaap:WeightedAverageNumberOfSharesOutstandingBasic`/`...Diluted` as `721.8`/`725.9`
# where the true counts are `721,800,000`/`725,900,000` -- a 1,000,000x scale error baked
# into the raw XBRL instance (the known SEC DQC "shares reported in millions" defect class,
# not an edgartools or extraction parsing bug -- `numeric_value` is never rescaled by
# decimals/scale anywhere in the installed edgartools package). `NON_NEGATIVE_STOCK_FIELDS`
# above only catches the WRONG SIGN, so a positive-but-1,000,000x-too-small fact sails
# through untouched -- and since both `basicShares` and `dilutedShares` are miss-tagged
# identically here, `apply_plausibility_guards`' relative diluted-vs-basic check also never
# fires. `sharesOutstanding` is already in `NON_NEGATIVE_STOCK_FIELDS`, but gets no magnitude
# check at this layer either -- included here too so a same-shaped scale defect on the
# point-in-time count is caught at the same place, not just downstream in
# `apply_plausibility_guards`. `commonSharesIssued` is share-count-shaped for the same reason.
# Applied the same way as `NON_NEGATIVE_STOCK_FIELDS`: reject (never rescale) so the field's
# candidate coalesce falls through -- with exactly one candidate tag each for `basicShares`/
# `dilutedShares` today, a rejected fact correctly ends up NaN rather than a wrong number
# ("null, never guess wrong"). Kept numerically in sync with
# `src/constants/constants.py::SHARES_OUTSTANDING_MIN` (both express the same real-world
# floor; duplicated rather than imported because this module has zero imports by design).
SHARE_COUNT_MAGNITUDE_FIELDS = {
    "basicShares", "dilutedShares", SHARES_OUTSTANDING_FIELD, "commonSharesIssued",
}
SHARE_COUNT_MIN_ABS = 1_000_000.0

# Per-(ticker, field) tag DENY-LIST: bare concept names that must NEVER be admitted for
# that one filer, applied as a pre-filter in `fetch_fundamentals_edgar.build_tag_frames`.
# The escape hatch for a defect that is genuinely one issuer's own, where no global rule
# can express the right answer -- the same role `def14a_validate.py` plays for the proxy
# parser.
#
# DENY, never PIN, and the distinction is the whole design. A pin ("ticker X's field Y is
# always concept Z") freezes the resolution: the moment that filer follows a taxonomy
# migration -- and they all do, measured on the live table 15.6% of (ticker, field) pairs
# already switch concept mid-history for benign reasons (ASC 842, ASU 2016-18 cash, CECL)
# -- the pinned concept stops being reported and the field goes silently NULL. A deny
# only removes a candidate, so everything else still flows through the global priority
# order, and a filer not listed here is untouched.
#
# Add an entry ONLY with the evidence written down beside it (ticker, fiscal years, the
# actual figures), and only after `fundamentals_tag_ledger` has ranked the case -- an
# entry here is the CONCLUSION of a diagnosis, never a shortcut around one.
FIELD_TAG_DENYLIST: dict[str, dict[str, frozenset[str]]] = {
    # DTE never tags `us-gaap:DebtCurrent` on its balance sheet. It uses that concept for
    # the "Less amount due within one year" DEDUCTION row of its LONG-TERM-DEBT FOOTNOTE
    # (statement_role .../LongTermDebtDetails), which is a different measure AND was filed
    # with the presentation sign baked in: -$355M (FY2011) and -$634M (FY2012). As the
    # priority-0 `shortTermDebt` candidate it won outright and stored short-term debt was
    # NEGATIVE for both years. `NON_NEGATIVE_STOCK_FIELDS` already rejects those two on
    # sign, but from FY2013 the SAME footnote row is tagged POSITIVE ($694M / $161M /
    # $465M) and no sign rule can see it -- while DTE's actual balance sheet reports
    # `ShortTermBorrowings` $131M and `LongTermDebtAndCapitalLeaseObligationsCurrent`
    # $898M for that same 2013-12-31 date. The footnote figure is neither line and matches
    # neither leg, so it is denied: every period then resolves the balance-sheet
    # `ShortTermBorrowings`, ending the annual 10-K-vs-10-Q measure swap that made this
    # series alternate concepts every single year from 2011 to 2015.
    "DTE": {"shortTermDebt": frozenset({"DebtCurrent"})},
    # AEP tags `us-gaap:CostOfGoodsAndServicesSold` -- priority-0 in `costOfRevenue`'s
    # candidate list -- with a value that is NOT its consolidated cost of energy: $0 to
    # -$223M every quarter FY2018-FY2023 (e.g. FY2021 Q1 -$172M, FY2022 Q3 -$223M, FY2023
    # Q1-Q3 all under $47M), impossible for a utility with ~$17-19B/year of revenue and a
    # fuel/purchased-power cost line that dominates it. `fundamentals_tag_ledger` scores
    # the FY2024 cutover (this tag -> `CostOfGoodsAndServiceExcludingDepreciationDepletion
    # AndAmortization`, AEP's real cost line per the candidate-list comment above) at a
    # 38,331x pooled-level jump, unique to AEP (n_tickers_same_switch=1) -- not a taxonomy
    # migration, AEP's own mis-tagging. Denying restores the intended fill-only excl-D&A
    # tag from FY2024 on and leaves FY2018-FY2023 correctly NaN (no other candidate in the
    # list is present for AEP those years) rather than the near-zero garbage that was
    # there before.
    "AEP": {"costOfRevenue": frozenset({"CostOfGoodsAndServicesSold"})},
    # Found via the Tiingo cross-check: CAT tags `us-gaap:CostOfRevenue` consistently and
    # correctly EVERY quarter from 2018 on (~$9-11B/quarter, matching its own ~$40B/year
    # revenue) -- except FY2024, FY2025 and FY2025-Q1, where it ALSO tags the priority-0
    # candidate `us-gaap:CostOfGoodsAndServicesSold` at $33M / $27M / $49M, a ~300x
    # understatement that wins the coalesce outright and corrupts the derived Q4 (and
    # every TTM window that includes it) with a huge, wrong swing. Both facts are
    # undimensioned, so no dimension rule could reject the small one, and CAT's own
    # `CostOfRevenue` value for the SAME periods is right there, just lower priority.
    # Denying restores it for exactly the years CAT double-tags, leaving every other
    # year (which never carries `CostOfGoodsAndServicesSold` at all) untouched.
    "CAT": {"costOfRevenue": frozenset({"CostOfGoodsAndServicesSold"})},
    # Found via the Tiingo cross-check, same shape as the CAT entry above: MCD tags
    # BOTH `us-gaap:DepreciationDepletionAndAmortization` (priority-0 in `depAmort`'s
    # candidate list, $99M for 2024-Q1, $99M for 2023-Q1 -- a small, WRONG figure
    # repeating quarter to quarter) AND `us-gaap:DepreciationAndAmortization`
    # (priority-2, $510M for 2024-Q1, $490M for 2023-Q1) undimensioned, for the exact
    # same period. Confirmed against Tiingo's own `depamor`: $510M for 2024-Q1,
    # matching the LOWER-priority tag exactly, not the one that currently wins.
    # McDonald's real total D&A (a company with ~$40B+ gross PP&E) is the ~$500M/
    # quarter figure, not $99M. Denying the small tag falls through to the correct one.
    "MCD": {"depAmort": frozenset({"DepreciationDepletionAndAmortization"})},
}

# How far a derived Q4 whose sign matches NONE of Q1/Q2/Q3 may exceed the largest
# quarter already observed that year, for a SIGNED field. Raised from 1.0 after the
# 1.0 bar was measured rejecting real, as-filed quarters at 1.03-2.6x -- Allstate
# fiscal 2023 (three catastrophe-loss quarters then a +$1,489M Q4, 1.10x), Gilead
# fiscal 2017 (-$3,865M Q4 on the Tax Cuts and Jobs Act writedown, 1.26x), S&P Global
# fiscal 2014 (-$1,169M Q4 on the $1.6B legal settlement, 2.41x), Genuine Parts 2025
# (2.39x), Zimmer Biomet 2018 (2.59x), Johnson Controls 2012 (2.59x), J.M. Smucker
# 2023 (1.99x). A charge-driven quarter is normal and legitimately dwarfs the
# run-rate; magnitude alone cannot distinguish it from a data error, so the
# NON_NEGATIVE_FLOW_FIELDS sign test above is what now does the real work and this
# bar only backstops the signed fields.
MAX_OPPOSITE_SIGN_Q4_RATIO = 3.0
# Ceiling on |FY| / (the quarters' own annualized scale) before a fiscal year whose FY
# fact resolved a DIFFERENT concept than its quarters is treated as two unrelated
# lines. Only an UPPER bound: a lower bound cannot work, because a year of offsetting
# quarters legitimately foots to a small annual figure (see
# `_fy_matches_quarterly_run_rate`), and an FY that is too SMALL because it is the
# wrong concept produces a negative Q4 that the sign test above already rejects.
Q4_TAG_MISMATCH_FY_MAX = 2.0
# A fiscal year's FY anchor is treated as PARTIAL (not the whole year) once this many
# independent non-negative fields report an annual figure BELOW their own Q1-Q3
# cumulative -- see `fundamentals_periods.drop_derived_q4_for_partial_fiscal_years`.
# Set from the measured distribution over the 20 re-extracted tickers, which is
# cleanly separated: the two genuine year-wide cases fail on FIVE fields (S&P Global
# 2012) and SEVEN (Johnson Controls 2012) -- both 2012 divestitures, where the 10-K
# restates the year to continuing operations while the quarters stand as originally
# filed, so no Q4 subtraction is valid. Every other flagged year fails on exactly ONE
# field, which means something different and much more common: that field's FY
# resolved a different concept than its quarters (KeyCorp's D&A in eight separate
# years, United Rentals 2017, Echo 2019/2022) -- already handled per field by the
# sign test. A threshold of 2 was measured catching KeyCorp fiscal 2024 (D&A plus
# noninterest income, both concept mismatches) and needlessly destroying five good
# Q4s that year.
MIN_PARTIAL_FY_FIELDS = 3
# Slack on the FY-vs-nine-months comparison, so ordinary restatement/rounding noise
# between a 10-Q and the later 10-K is not read as a partial anchor.
PARTIAL_FY_TOLERANCE = 0.98

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

# --- fiscal-year calendar reconstruction (fundamentals_periods.
# resolve_fiscal_year_by_filing_calendar) -----------------------------------
# Mean length of a fiscal year in days, used ONLY to extrapolate one step beyond
# the observed 10-K period-end dates (the in-progress fiscal year, which has no
# 10-K yet, and any quarter preceding the earliest 10-K in the history window).
# Inside the observed range the ACTUAL 10-K period ends are used, so a 52/53-week
# filer's drift never accumulates.
FISCAL_YEAR_MEAN_DAYS = 365.25
# Absorbs that drift on the single extrapolated step: a 53-week fiscal year is
# 371d, so "one year later" can overshoot FISCAL_YEAR_MEAN_DAYS by ~6d and must
# still resolve to +1 fiscal year, not +2.
FISCAL_YEAR_EXTRAPOLATION_GRACE_DAYS = 10
# A 10-K's cover page (dei:DocumentFiscalYearFocus) and edgartools' per-fact
# fiscal_year BOTH carry occasional filer typos (confirmed: Cisco's fiscal-2016
# 10-K facts say 2017; J.M. Smucker's fiscal-2015 Q1 cover page says 2014), so the
# fiscal-year LABEL for the whole calendar is voted on across every 10-K rather
# than read off any single one. Below this many independent votes the vote is not
# meaningful and native labels are left untouched ("null, never guess wrong").
MIN_FISCAL_YEAR_LABEL_VOTES = 2

# --- per-filing XBRL retrieval (fetch_fundamentals_edgar._filing_xbrl) ------
# A filing's XBRL instance is fetched over the network and parsed; a failure is
# usually transient (SEC throttling, a truncated download) but was previously
# indistinguishable from "this filing has no XBRL" and so never retried, which
# silently and permanently dropped whole filings (8 of 2,482 measured).
XBRL_PARSE_ATTEMPTS = 3
XBRL_RETRY_BACKOFF_SECONDS = 2.0
# A newly-seen filing does not carry its own fiscal year's OTHER filings, but the
# derived Q4 (FY - Q1 - Q2 - Q3) and the cross-field derivations need them, so
# already-extracted filings reporting within this many days of a new one are
# re-parsed alongside it. Wider than a fiscal year (a 53-week year is 371d) so the
# whole year is always covered, and narrow enough that a routine incremental run
# re-reads only its own year.
FISCAL_YEAR_CONTEXT_DAYS = 400
# Gross margin is mathematically <= 1 (a value > 1 implies negative COGS); a value
# below -200% only arises when revenue is truncated / period-mismatched vs the cost
# line (e.g. a REIT whose rental income moved to the ASC-842 lease-income tag). Values
# outside this band are nulled as extraction artifacts rather than shipped as features.
GROSS_MARGIN_MIN, GROSS_MARGIN_MAX = -2.0, 1.0
