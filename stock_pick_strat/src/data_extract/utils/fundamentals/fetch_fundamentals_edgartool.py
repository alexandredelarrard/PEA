import pandas as pd
from typing import List, Dict,Optional, Any
import numpy as np 
from datetime import datetime, timedelta
from edgar import Company, set_identity

from src.context import Context

# ==============================================================================
# TAG DEFINITIONS & US-GAAP XBRL MAPPING MATRIX
# ==============================================================================

FLOW_TAGS = [
    'totalRevenue', 'netIncome', 'grossProfit', 'costOfRevenue', 'operatingIncome', 
    'depAmort', 'operatingCashFlow', 'capex', 'researchAndDevelopment', 
    'sellingGeneralAdmin', 'stockBasedComp', 'acquisitions', 'interestExpense', 
    'incomeTaxExpense', 'revenuesTotal', 'pretaxIncome', 'costsAndExpenses', 
    'interestIncome', 'amortizationIntangibles', 'epsDiluted', 'epsBasic', 
    'dividendsPerShare', 'incomeTaxesPaid', 'deferredIncomeTaxExpense', 
    'interestPaid', 'equityMethodIncome', 'otherNonoperating', 'debtExtinguishment', 
    'nciIncome', 'comprehensiveIncome', 'goodwillAcquired', 'operatingLeaseAdditions',
    'operatingLeaseCost', 'exciseTaxes', 'revenueExcludingAssessedTax', 
    'revenueIncludingAssessedTax', 'pensionNetPeriodicCost', 'pensionServiceCost', 
    'pensionInterestCost', 'pensionExpectedReturn', 'pensionAmortPriorService', 
    'pensionAmortGainsLosses', 'dividendsPaid', 'buybacks', 'equityIssuance', 
    'debtIssued', 'debtRepaid', 'financeLeaseAdditions', 'investingCashFlow', 
    'financingCashFlow', 'changeInInventory', 'changeInReceivables', 
    'changeInPayables', 'impairment', 'restructuring', 'goodwillImpairment', 
    'gainOnSaleGeneric', 'litigationExpense', 'discontinuedOps', 'unusualItems', 
    'bargainPurchaseGain', 'interestIncomeBank', 'netInterestIncome', 
    'provisionForCreditLosses', 'provisionDoubtfulAccounts', 'noninterestIncome', 
    'noninterestExpense', 'netChargeOffs', 'premiumsEarned', 'premiumsWritten', 
    'claimsIncurred', 'netInvestmentIncome', 'dacAmortization', 'realizedInvestmentGains', 
    'rentalIncome', 'straightLineRent', 'aboveBelowMarketLeaseAmort', 
    'gainOnDispositions', 'realEstateImpairment', 'oilGasRevenue', 
    'explorationExpense', 'depletionDDA'
]

STOCK_TAGS = [
    'stockholdersEquity', 'totalLiabilities', 'longTermDebt', 'cash', 'shortTermDebt', 
    'currentAssets', 'currentLiabilities', 'goodwill', 'totalAssets', 'longTermDebtTotal', 
    'debtCombined', 'notesPayable', 'commercialPaper', 'operatingLeaseLiability', 
    'operatingLeaseLiabilityCurrent', 'operatingLeaseLiabilityNoncurrent', 
    'financeLeaseLiability', 'financeLeaseLiabilityCurrent', 
    'financeLeaseLiabilityNoncurrent', 'capitalLeaseObligationCurrent', 
    'capitalLeaseObligationNoncurrent', 'operatingLeaseRouAsset', 
    'financeLeaseRouAsset', 'restrictedCash', 'restrictedCashCurrent', 
    'restrictedCashNoncurrent', 'cashInclRestricted', 'cashAndShortTermInvestments', 
    'marketableSecuritiesCurrent', 'investmentSecurities', 'redeemableNCI', 'lifoReserve', 
    'assetRetirementObligation', 'aroCurrent', 'aroNoncurrent', 'debtMaturity1y', 
    'debtMaturity2y', 'debtMaturity3y', 'debtMaturity4y', 'debtMaturity5y', 
    'debtMaturityAfter5y', 'deferredTaxAssets', 'deferredTaxLiabilities', 
    'valuationAllowance', 'unrecognizedTaxBenefits', 'allowanceDoubtfulAccounts', 
    'intangiblesGross', 'intangiblesAccumAmort', 'accountsReceivable', 'inventory', 
    'inventoryFifoReported', 'accountsPayable', 'ppeNet', 'ppeGross', 
    'accumulatedDepreciation', 'intangiblesExGoodwill', 'capitalizedSoftware', 
    'pensionDeficit', 'retainedEarnings', 'treasuryStock', 'preferredEquity', 
    'minorityInterest', 'accumulatedOCI', 'shortTermInvestments', 'longTermInvestments', 
    'deferredRevenue', 'deferredRevenueCurrent', 'deferredRevenueNoncurrent', 
    'remainingPerformanceObligation', 'loans', 'deposits', 'depositsDomestic', 
    'allowanceCreditLosses', 'htmSecurities', 'htmSecuritiesFairValue', 
    'htmUnrealizedLoss', 'nonaccrualLoans', 'tier1CapitalRatio', 'insuranceReserves', 
    'deferredAcqCosts', 'realEstateNet', 'realEstateGross', 'oilGasPropertyNet', 
    'oilGasPropertyGross', 'regulatoryAssets', 'regulatoryAssetsCurrent', 
    'regulatoryAssetsNoncurrent', 'regulatoryLiabilities', 
    'regulatoryLiabilitiesCurrent', 'regulatoryLiabilitiesNoncurrent'
]

SHARE_TAGS = ['sharesOutstanding', 'dilutedShares', 'basicShares', 'reportableSegments']

FINANCE_TAGS = FLOW_TAGS + STOCK_TAGS + SHARE_TAGS

# Core mapping dictionary between standardized FINANCE_TAGS and US-GAAP taxonomy concepts
USGAAP_MAP: Dict[str, List[str]] = {
    # =========================================================================
    # FLOW TAGS (Income Statement & Cash Flow Statement)
    # =========================================================================
    'totalRevenue': [
        'Revenues', 'SalesRevenueNet', 'RevenueFromContractWithCustomerExcludingAssessedTax', 
        'RevenueFromContractWithCustomerIncludingAssessedTax', 'TotalRevenuesAndOtherIncome'
    ],
    'netIncome': [
        'NetIncomeLoss', 'ProfitLoss', 'NetIncomeLossAvailableToCommonStockholdersBasic'
    ],
    'grossProfit': [
        'GrossProfit', 'GrossProfitLoss'
    ],
    'costOfRevenue': [
        'CostOfGoodsAndServicesSold', 'CostOfRevenue', 'CostOfGoodsSold', 'CostOfServices'
    ],
    'operatingIncome': [
        'OperatingIncomeLoss'
    ],
    'depAmort': [
        'DepreciationDepletionAndAmortization', 'DepreciationAmortizationAndImpairment', 
        'DepreciationAndAmortization', 'Depreciation', 'AmortizationOfIntangibleAssets'
    ],
    'operatingCashFlow': [
        'NetCashProvidedByUsedInOperatingActivities'
    ],
    'capex': [
        'PaymentsToAcquirePropertyPlantAndEquipment', 'PaymentsToAcquireProductiveAssets', 
        'PaymentsToAcquireSoftware'
    ],
    'researchAndDevelopment': [
        'ResearchAndDevelopmentExpense', 'ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost'
    ],
    'sellingGeneralAdmin': [
        'SellingGeneralAndAdministrativeExpense', 'SellingAndMarketingExpense', 
        'GeneralAndAdministrativeExpense'
    ],
    'stockBasedComp': [
        'ShareBasedCompensation', 'AllocatedShareBasedCompensationExpense'
    ],
    'acquisitions': [
        'PaymentsToAcquireBusinessesNetOfCashAcquired', 'PaymentsToAcquireBusinessesGross'
    ],
    'interestExpense': [
        'InterestExpense', 'InterestAndDebtExpense', 'InterestExpenseNonoperating'
    ],
    'incomeTaxExpense': [
        'IncomeTaxExpenseBenefit'
    ],
    'revenuesTotal': [
        'Revenues', 'SalesRevenueNet', 'TotalRevenuesAndOtherIncome'
    ],
    'pretaxIncome': [
        'IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeTaxes', 
        'IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest'
    ],
    'costsAndExpenses': [
        'CostsAndExpenses', 'OperatingExpenses'
    ],
    'interestIncome': [
        'InterestAndDividendIncomeOperating', 'InterestIncomeOperating', 'InvestmentIncomeInterest'
    ],
    'amortizationIntangibles': [
        'AmortizationOfIntangibleAssets'
    ],
    'epsDiluted': [
        'EarningsPerShareDiluted'
    ],
    'epsBasic': [
        'EarningsPerShareBasic'
    ],
    'dividendsPerShare': [
        'CommonStockDividendsPerShareDeclared', 'CommonStockDividendsPerShareCashPaid'
    ],
    'incomeTaxesPaid': [
        'IncomeTaxesPaidNet', 'IncomeTaxesPaid'
    ],
    'deferredIncomeTaxExpense': [
        'DeferredIncomeTaxExpenseBenefit'
    ],
    'interestPaid': [
        'InterestPaidNet', 'InterestPaid'
    ],
    'equityMethodIncome': [
        'IncomeLossFromEquityMethodInvestments'
    ],
    'otherNonoperating': [
        'OtherNonoperatingIncomeExpense', 'NonoperatingIncomeExpense'
    ],
    'debtExtinguishment': [
        'GainsLossesOnExtinguishmentOfDebt'
    ],
    'nciIncome': [
        'NetIncomeLossAttributableToNoncontrollingInterest'
    ],
    'comprehensiveIncome': [
        'ComprehensiveIncomeNetOfTax'
    ],
    'goodwillAcquired': [
        'GoodwillAcquiredDuringPeriod'
    ],
    'operatingLeaseAdditions': [
        'OperatingLeaseRightOfUseAssetObtainedInExchangeForOperatingLeaseLiability'
    ],
    'operatingLeaseCost': [
        'OperatingLeaseCost'
    ],
    'exciseTaxes': [
        'ExciseAndSalesTaxes'
    ],
    'revenueExcludingAssessedTax': [
        'RevenueFromContractWithCustomerExcludingAssessedTax'
    ],
    'revenueIncludingAssessedTax': [
        'RevenueFromContractWithCustomerIncludingAssessedTax'
    ],
    'pensionNetPeriodicCost': [
        'DefinedBenefitPlanNetPeriodicBenefitCost'
    ],
    'pensionServiceCost': [
        'DefinedBenefitPlanServiceCost'
    ],
    'pensionInterestCost': [
        'DefinedBenefitPlanInterestCost'
    ],
    'pensionExpectedReturn': [
        'DefinedBenefitPlanExpectedReturnOnPlanAssets'
    ],
    'pensionAmortPriorService': [
        'DefinedBenefitPlanAmortizationOfPriorServiceCostCredit'
    ],
    'pensionAmortGainsLosses': [
        'DefinedBenefitPlanAmortizationOfGainsLosses'
    ],
    'dividendsPaid': [
        'PaymentsOfDividends', 'PaymentsOfDividendsCommonStock'
    ],
    'buybacks': [
        'PaymentsForRepurchaseOfCommonStock'
    ],
    'equityIssuance': [
        'ProceedsFromIssuanceOfCommonStock'
    ],
    'debtIssued': [
        'ProceedsFromIssuanceOfLongTermDebt', 'ProceedsFromIssuanceOfDebt'
    ],
    'debtRepaid': [
        'RepaymentsOfLongTermDebt', 'RepaymentsOfDebt'
    ],
    'financeLeaseAdditions': [
        'FinanceLeaseRightOfUseAssetObtainedInExchangeForFinanceLeaseLiability'
    ],
    'investingCashFlow': [
        'NetCashProvidedByUsedInInvestingActivities'
    ],
    'financingCashFlow': [
        'NetCashProvidedByUsedInFinancingActivities'
    ],
    'changeInInventory': [
        'IncreaseDecreaseInInventories'
    ],
    'changeInReceivables': [
        'IncreaseDecreaseInAccountsReceivable'
    ],
    'changeInPayables': [
        'IncreaseDecreaseInAccountsPayable'
    ],
    'impairment': [
        'AssetImpairmentCharges', 'ImpairmentOfLongLivedAssetsToBeDisposedOf'
    ],
    'restructuring': [
        'RestructuringCharges', 'RestructuringAndRelatedCost'
    ],
    'goodwillImpairment': [
        'GoodwillImpairmentLoss'
    ],
    'gainOnSaleGeneric': [
        'GainLossOnSaleOfPropertyPlantEquipment', 'GainLossOnSaleOfAssets'
    ],
    'litigationExpense': [
        'LitigationSettlementExpense'
    ],
    'discontinuedOps': [
        'IncomeLossFromDiscontinuedOperationsNetOfTax'
    ],
    'unusualItems': [
        'OtherCostAndExpenseOperating', 'UnusualOrInfrequentItemNet'
    ],
    'bargainPurchaseGain': [
        'BargainPurchaseGainRecognizedAmount'
    ],

    # Financial Services / Banks
    'interestIncomeBank': ['InterestAndDividendIncomeOperating'],
    'netInterestIncome': ['InterestIncomeExpenseNet'],
    'provisionForCreditLosses': ['ProvisionForCreditLosses'],
    'provisionDoubtfulAccounts': ['ProvisionForDoubtfulAccounts'],
    'noninterestIncome': ['NoninterestIncome'],
    'noninterestExpense': ['NoninterestExpense'],
    'netChargeOffs': ['LoansAndLeasesReceivableNetReportedChargeOffs'],

    # Insurance
    'premiumsEarned': ['PremiumsEarnedNet'],
    'premiumsWritten': ['PremiumsWrittenNet'],
    'claimsIncurred': ['PolicyholderBenefitsAndClaimsIncurredNet'],
    'netInvestmentIncome': ['NetInvestmentIncome'],
    'dacAmortization': ['DeferredPolicyAcquisitionCostAmortizationExpense'],
    'realizedInvestmentGains': ['RealizedInvestmentGainsLosses'],

    # Real Estate / REITs
    'rentalIncome': ['OperatingLeaseLeaseIncome', 'RentalIncomeNonoperating'],
    'straightLineRent': ['StraightLineRentAdjustments'],
    'aboveBelowMarketLeaseAmort': ['AmortizationOfAboveAndBelowMarketLeases'],
    'gainOnDispositions': ['GainLossOnDispositionOfRealEstateDiscontinuedOperations'],
    'realEstateImpairment': ['ImpairmentOfRealEstate'],

    # Energy / Oil & Gas
    'oilGasRevenue': ['OilAndGasRevenue'],
    'explorationExpense': ['ExplorationExpense'],
    'depletionDDA': ['DepreciationDepletionAndAmortization'],


    # =========================================================================
    # STOCK TAGS (Balance Sheet)
    # =========================================================================
    'stockholdersEquity': [
        'StockholdersEquity', 'StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest'
    ],
    'totalLiabilities': [
        'Liabilities'
    ],
    'longTermDebt': [
        'LongTermDebtNoncurrent', 'LongTermDebt'
    ],
    'cash': [
        'CashAndCashEquivalentsAtCarryingValue', 'CashAndCashEquivalents'
    ],
    'shortTermDebt': [
        'ShortTermBorrowings', 'DebtCurrent'
    ],
    'currentAssets': [
        'AssetsCurrent'
    ],
    'currentLiabilities': [
        'LiabilitiesCurrent'
    ],
    'goodwill': [
        'Goodwill'
    ],
    'totalAssets': [
        'Assets'
    ],
    'longTermDebtTotal': [
        'LongTermDebtAndCapitalLeaseObligations', 'LongTermDebt'
    ],
    'debtCombined': [
        'DebtAndCapitalLeaseObligations', 'DebtInstrumentCarryingAmount'
    ],
    'notesPayable': [
        'NotesPayable'
    ],
    'commercialPaper': [
        'CommercialPaper'
    ],
    'operatingLeaseLiability': [
        'OperatingLeaseLiability'
    ],
    'operatingLeaseLiabilityCurrent': [
        'OperatingLeaseLiabilityCurrent'
    ],
    'operatingLeaseLiabilityNoncurrent': [
        'OperatingLeaseLiabilityNoncurrent'
    ],
    'financeLeaseLiability': [
        'FinanceLeaseLiability'
    ],
    'financeLeaseLiabilityCurrent': [
        'FinanceLeaseLiabilityCurrent'
    ],
    'financeLeaseLiabilityNoncurrent': [
        'FinanceLeaseLiabilityNoncurrent'
    ],
    'capitalLeaseObligationCurrent': [
        'CapitalLeaseObligationsCurrent'
    ],
    'capitalLeaseObligationNoncurrent': [
        'CapitalLeaseObligationsNoncurrent'
    ],
    'operatingLeaseRouAsset': [
        'OperatingLeaseRightOfUseAsset'
    ],
    'financeLeaseRouAsset': [
        'FinanceLeaseRightOfUseAsset'
    ],
    'restrictedCash': [
        'RestrictedCashAndCashEquivalentsAtCarryingValue', 'RestrictedCash'
    ],
    'restrictedCashCurrent': [
        'RestrictedCashAndCashEquivalentsCurrent'
    ],
    'restrictedCashNoncurrent': [
        'RestrictedCashAndCashEquivalentsNoncurrent'
    ],
    'cashInclRestricted': [
        'CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents'
    ],
    'cashAndShortTermInvestments': [
        'CashCashEquivalentsAndShortTermInvestments'
    ],
    'marketableSecuritiesCurrent': [
        'MarketableSecuritiesCurrent'
    ],
    'investmentSecurities': [
        'AvailableForSaleSecuritiesDebtSecurities', 'SecuritiesFairValueDisclosure'
    ],
    'redeemableNCI': [
        'RedeemableNoncontrollingInterestEquityCarryingAmount'
    ],
    'lifoReserve': [
        'LIFOReserve'
    ],
    'assetRetirementObligation': [
        'AssetRetirementObligations'
    ],
    'aroCurrent': [
        'AssetRetirementObligationCurrent'
    ],
    'aroNoncurrent': [
        'AssetRetirementObligationNoncurrent'
    ],
    'debtMaturity1y': [
        'LongTermDebtMaturitiesRepaymentsOfPrincipalInNextTwelveMonths'
    ],
    'debtMaturity2y': [
        'LongTermDebtMaturitiesRepaymentsOfPrincipalInYearTwo'
    ],
    'debtMaturity3y': [
        'LongTermDebtMaturitiesRepaymentsOfPrincipalInYearThree'
    ],
    'debtMaturity4y': [
        'LongTermDebtMaturitiesRepaymentsOfPrincipalInYearFour'
    ],
    'debtMaturity5y': [
        'LongTermDebtMaturitiesRepaymentsOfPrincipalInYearFive'
    ],
    'debtMaturityAfter5y': [
        'LongTermDebtMaturitiesRepaymentsOfPrincipalAfterYearFive'
    ],
    'deferredTaxAssets': [
        'DeferredIncomeTaxAssetsNet', 'DeferredTaxAssetsNetCurrent'
    ],
    'deferredTaxLiabilities': [
        'DeferredIncomeTaxLiabilities', 'DeferredTaxLiabilitiesNoncurrent'
    ],
    'valuationAllowance': [
        'DeferredTaxAssetsValuationAllowance'
    ],
    'unrecognizedTaxBenefits': [
        'UnrecognizedTaxBenefits'
    ],
    'allowanceDoubtfulAccounts': [
        'AllowanceForDoubtfulAccountsReceivableCurrent'
    ],
    'intangiblesGross': [
        'FiniteLivedIntangibleAssetsGross', 'IndefiniteLivedIntangibleAssetsExcludingGoodwill'
    ],
    'intangiblesAccumAmort': [
        'FiniteLivedIntangibleAssetsAccumulatedAmortization'
    ],
    'accountsReceivable': [
        'AccountsReceivableNetCurrent', 'AccountsReceivableNet'
    ],
    'inventory': [
        'InventoryNet', 'Inventories'
    ],
    'inventoryFifoReported': [
        'InventoryFIFO'
    ],
    'accountsPayable': [
        'AccountsPayableCurrent', 'AccountsPayable'
    ],
    'ppeNet': [
        'PropertyPlantAndEquipmentNet'
    ],
    'ppeGross': [
        'PropertyPlantAndEquipmentGross'
    ],
    'accumulatedDepreciation': [
        'AccumulatedDepreciationDepletionAndAmortizationPropertyPlantAndEquipment'
    ],
    'intangiblesExGoodwill': [
        'IntangibleAssetsNetExcludingGoodwill'
    ],
    'capitalizedSoftware': [
        'CapitalizedComputerSoftwareNet'
    ],
    'pensionDeficit': [
        'DefinedBenefitPlanObligationInExcessOfPlanAssets'
    ],
    'retainedEarnings': [
        'RetainedEarningsUnappropriated', 'RetainedEarnings'
    ],
    'treasuryStock': [
        'TreasuryStockValue'
    ],
    'preferredEquity': [
        'PreferredStockValue'
    ],
    'minorityInterest': [
        'MinorityInterest', 'NoncontrollingInterest'
    ],
    'accumulatedOCI': [
        'AccumulatedOtherComprehensiveIncomeLossNetOfTax'
    ],
    'shortTermInvestments': [
        'MarketableSecuritiesCurrent', 'ShortTermInvestments'
    ],
    'longTermInvestments': [
        'OtherLongTermInvestments', 'LongTermInvestments'
    ],
    'deferredRevenue': [
        'DeferredRevenue', 'ContractWithCustomerLiability'
    ],
    'deferredRevenueCurrent': [
        'DeferredRevenueCurrent', 'ContractWithCustomerLiabilityCurrent'
    ],
    'deferredRevenueNoncurrent': [
        'DeferredRevenueNoncurrent', 'ContractWithCustomerLiabilityNoncurrent'
    ],
    'remainingPerformanceObligation': [
        'RevenueRemainingPerformanceObligation'
    ],

    # Financial Sector (Banking & Insurance)
    'loans': ['LoansAndLeasesReceivableNetReported'],
    'deposits': ['Deposits'],
    'depositsDomestic': ['DomesticDeposits'],
    'allowanceCreditLosses': ['FinancingReceivableAllowanceForCreditLosses'],
    'htmSecurities': ['HeldToMaturitySecuritiesCarryingReportedAmount'],
    'htmSecuritiesFairValue': ['HeldToMaturitySecuritiesFairValue'],
    'htmUnrealizedLoss': ['HeldToMaturitySecuritiesUnrealizedLosses'],
    'nonaccrualLoans': ['FinancingReceivableOnNonaccrualStatus'],
    'tier1CapitalRatio': ['Tier1RiskBasedCapitalRatio'],
    'insuranceReserves': ['LiabilityForUnpaidClaimsAndClaimAdjustmentExpenses'],
    'deferredAcqCosts': ['DeferredPolicyAcquisitionCosts'],

    # Real Estate & Oil/Gas
    'realEstateNet': ['RealEstateInvestmentPropertyNet'],
    'realEstateGross': ['RealEstateInvestmentPropertyAtCost'],
    'oilGasPropertyNet': ['OilAndGasPropertySuccessfulEffortMethodNet'],
    'oilGasPropertyGross': ['OilAndGasPropertySuccessfulEffortMethodGross'],

    # Utilities
    'regulatoryAssets': ['RegulatoryAssets'],
    'regulatoryAssetsCurrent': ['RegulatoryAssetsCurrent'],
    'regulatoryAssetsNoncurrent': ['RegulatoryAssetsNoncurrent'],
    'regulatoryLiabilities': ['RegulatoryLiabilities'],
    'regulatoryLiabilitiesCurrent': ['RegulatoryLiabilitiesCurrent'],
    'regulatoryLiabilitiesNoncurrent': ['RegulatoryLiabilitiesNoncurrent'],


    # =========================================================================
    # SHARE & SEGMENT TAGS
    # =========================================================================
    'sharesOutstanding': [
        'EntityCommonStockSharesOutstanding', 'CommonStockSharesOutstanding'
    ],
    'dilutedShares': [
        'WeightedAverageNumberOfDilutedSharesOutstanding'
    ],
    'basicShares': [
        'WeightedAverageNumberOfSharesOutstandingBasic'
    ],
    'reportableSegments': [
        'NumberOfReportableSegments'
    ]
}

class FetchFundamentals:
    """Extracts, standardizes, and calculates quarterly financial metrics from EDGAR filings."""

    def __init__(self, context: Any, tickers: List[str]):
        self.tickers = tickers
        self.context = context

        # Set identity as required by SEC EDGAR guidelines
        set_identity("Jane Doe jdoe@example.com")

    def run(self) -> pd.DataFrame:
        """Executes full history load, extraction, and Q4 calculation flow."""
        existing = self._load_existing_history()
        all_results = []

        for ticker in self.tickers:
            self.context.log.info(f"Starting extraction for ticker: {ticker}")
            ticker_df = self.extract_ticker_fundamentals(existing, ticker, filing_types=["10-Q", "10-K"]) #, "10-Q/A", "10-K/A"
            
            if not ticker_df.empty:
                all_results.append(ticker_df)

        if not all_results:
            self.context.log.info("No new financial records extracted.")
            return pd.DataFrame()

        final_df = pd.concat(all_results, ignore_index=True)
        self.upsert_fundamentals(final_df)
        return final_df

    def _load_existing_history(self) -> Optional[pd.DataFrame]:
        """Loads existing ticker and accession number records to prevent duplicate extraction."""
        try:
            df = self.context.store.load("fundamentals_history", columns=['ticker', 'accession_number'])
            return None if df.empty else df.drop_duplicates()
        except Exception:
            return None

    def _filings_done(self, ticker: str, existing: Optional[pd.DataFrame]) -> List[str]:
        """Returns list of already processed accession numbers for a ticker."""
        if existing is None or existing.empty:
            return []
        subset = existing.loc[existing['ticker'] == ticker, 'accession_number']
        return subset.unique().tolist()

    def get_facts_dataframe(self, xb: Any) -> Optional[pd.DataFrame]:
        """Converts edgartools XBRL or FactsView structures safely into a pandas DataFrame."""
        if xb is None:
            return None

        facts_obj = getattr(xb, "facts", None)
        if facts_obj is None and hasattr(xb, "instance"):
            facts_obj = getattr(xb.instance, "facts", None)

        if facts_obj is None:
            return None

        if isinstance(facts_obj, pd.DataFrame):
            return facts_obj
        elif hasattr(facts_obj, "to_dataframe"):
            return facts_obj.to_dataframe()
        elif hasattr(facts_obj, "to_pandas"):
            return facts_obj.to_pandas()
        elif hasattr(facts_obj, "query"):
            return facts_obj.query().to_dataframe()

        return None

    def extract_ticker_fundamentals(
        self, existing: Optional[pd.DataFrame], ticker: str, filing_types: List[str]
    ) -> pd.DataFrame:
        """Extracts standard financial metrics for 10-Q and 10-K filings, deriving Q4 flow metrics."""
        
        company = Company(ticker)
        filings = company.get_filings(form=filing_types)
        filings_done = self._filings_done(ticker, existing)

        records = []
        ytd_flow_tracker: Dict[int, Dict[str, float]] = {}  # Tracks cumulative Q1-Q3 flows per fiscal year

        # Process filings chronologically to build accurate YTD tracking for Q4 deduction
        sorted_filings = sorted(filings,
            key=lambda x: x.filing_date
        )

        for filing in sorted_filings:
            if filing.accession_number in filings_done:
                continue

            try:
                xb = filing.xbrl()
                facts_df = self.get_facts_dataframe(xb)

                if facts_df is None or len(facts_df) == 0:
                    continue

                parsed_values = self._map_facts_to_tags(facts_df)
                
                # Base Metadata
                filing_date = str(filing.filing_date)
                period_of_report = str(filing.period_of_report)
                fiscal_year = int(pd.to_datetime(period_of_report).year)
                form = filing.form.upper()

                if "10-Q" in form:
                    # Record filing metrics directly
                    record = {
                        'ticker': ticker,
                        'as_of_date': filing_date,
                        'fiscal_end': period_of_report,
                        'accession_number': filing.accession_number,
                        **parsed_values
                    }
                    records.append(record)

                    # Update YTD flow tracker for flow tags
                    if fiscal_year not in ytd_flow_tracker:
                        ytd_flow_tracker[fiscal_year] = {tag: 0.0 for tag in FLOW_TAGS}

                    for tag in FLOW_TAGS:
                        val = parsed_values.get(tag, np.nan)
                        if not np.isnan(val):
                            # Accrue or replace YTD depending on filing structure
                            ytd_flow_tracker[fiscal_year][tag] = ytd_flow_tracker[fiscal_year].get(tag, 0.0) + val

                elif "10-K" in form:
                    # Calculate Q4 values by subtracting Q1-Q3 YTD values from 10-K Annual Totals
                    q4_values = parsed_values.copy()

                    if fiscal_year in ytd_flow_tracker:
                        for tag in FLOW_TAGS:
                            fy_val = parsed_values.get(tag, np.nan)
                            ytd_q3_val = ytd_flow_tracker[fiscal_year].get(tag, 0.0)

                            if not np.isnan(fy_val) and ytd_q3_val != 0.0:
                                q4_values[tag] = fy_val - ytd_q3_val

                    # Stock values for Q4 are taken as point-in-time directly from 10-K
                    record = {
                        'ticker': ticker,
                        'as_of_date': filing_date,
                        'fiscal_end': period_of_report,
                        'accession_number': filing.accession_number,
                        **q4_values
                    }
                    records.append(record)

            except Exception as e:
                self.context.log.error(f"Error parsing filing {filing.accession_number} for {ticker}: {e}")

        if not records:
            return pd.DataFrame()

        df_out = pd.DataFrame(records)
        
        # Ensure all FINANCE_TAGS exist in output dataframe
        for tag in FINANCE_TAGS:
            if tag not in df_out.columns:
                df_out[tag] = np.nan

        columns_order = ['ticker', 'as_of_date', 'fiscal_end', 'accession_number'] + FINANCE_TAGS
        return df_out[columns_order]

    def _map_facts_to_tags(self, facts_df: pd.DataFrame) -> Dict[str, float]:
        """Maps raw XBRL concept rows to standardized FINANCE_TAGS with fuzzy fallback."""

        mapped_results: Dict[str, float] = {tag: np.nan for tag in FINANCE_TAGS}
        
        if facts_df is None or facts_df.empty:
            return mapped_results

        concept_col = 'concept' if 'concept' in facts_df.columns else 'fact'
        value_col = 'val' if 'val' in facts_df.columns else 'value'

        if concept_col not in facts_df.columns or value_col not in facts_df.columns:
            return mapped_results

        clean_df = facts_df.copy()
        clean_df[value_col] = pd.to_numeric(clean_df[value_col], errors='coerce')
        clean_df = clean_df.dropna(subset=[value_col])

        for std_tag in FINANCE_TAGS:
            # 1. Primary Lookup via explicit USGAAP_MAP
            xbrl_concepts = USGAAP_MAP.get(std_tag, [])
            
            # 2. Automated Fallback: Convert camelCase tag (e.g. totalRevenue -> TotalRevenue)
            pascal_case_tag = std_tag[0].upper() + std_tag[1:]
            if pascal_case_tag not in xbrl_concepts:
                xbrl_concepts.append(pascal_case_tag)

            for xbrl_concept in xbrl_concepts:
                matched = clean_df[
                    clean_df[concept_col].str.endswith(f":{xbrl_concept}", na=False) | 
                    (clean_df[concept_col] == xbrl_concept)
                ]
                
                if not matched.empty:
                    val = matched.iloc[-1][value_col]
                    mapped_results[std_tag] = float(val)
                    break  # High-priority match found, move to next FINANCE_TAG

        return mapped_results

    def upsert_fundamentals(self, df: pd.DataFrame) -> None:
        """Saves extracted DataFrame into SQL context storage."""
        if df.empty:
            return
        
        self.context.log.info(f"Upserting {len(df)} financial records into storage...")
        self.context.store.upsert(
            table_name="fundamentals_history",
            dataframe=df,
            primary_keys=['ticker', 'accession_number']
        )

# TODO: Check currency 
# TODO : 10 Q /A K-A 