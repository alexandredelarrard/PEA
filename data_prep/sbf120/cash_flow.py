import pandas as pd 
import numpy as np
from utils.general_cleaning import create_index, handle_accountancy_numbers, \
                                handle_currency, deduce_trend

def cash_flow_annual(inputs, analysis_dict, params={}):
    """
    - Acquisition / investment in property -> Si plus que 100 alors pas bon car grossit par la dette
    - Trend net cash provided par operating -> cash créé par l'activité (si croit, alors good) 
    - investment in property / Net cashed used for activities  -> proportion du cash reinvesti dans l'activité 
    - common stock repurchased / Free cash flow -> si petit alors ok, si grand alors pb car pas investi dans activité
    - comparer cash flow from financing activities (common stock issued) avec net cash provided par operating (si fin >> cash, alors OPM)
    TODO:
    """

    data = inputs['CASH-FLOW-ANNUAL'].copy()
    new_results = {}
    data = create_index(data)
    data = handle_accountancy_numbers(data)
    data = handle_currency(params, data, which_currency="currency")

    # trend of where cash is invested 
    for i, idx_0 in enumerate([0, 0, 1, 2]):
        if data.shape[1] >= idx_0 +1:
            new_results[f"CASH_FROM_OPERATING_ACTIVITIES_{i}"] = data.loc["CASH_FROM_OPERATING_ACTIVITIES"][idx_0]
            
            if data.shape[1] >= idx_0 +2:
                new_results[f"CASH_TREND_FROM_OPERATING_ACTIVITIES_{i}"] = deduce_trend(data.loc["CASH_FROM_OPERATING_ACTIVITIES"][idx_0:])

            if "CAPITAL_EXPENDITURES" not in data.index:
                data.loc["CAPITAL_EXPENDITURES"] = data.loc["CASH_FROM_OPERATING_ACTIVITIES"]*0.2

            if "CAPITAL_EXPENDITURES" in data.index:
                data.loc["CASH_SHARE_INVESTED_ACTIVITY"] = (data.loc["CAPITAL_EXPENDITURES"].abs()) / data.loc["CASH_FROM_OPERATING_ACTIVITIES"]
                new_results[f"CASH_%_INTO_ACTIVITY_{i}"] = data.loc["CASH_SHARE_INVESTED_ACTIVITY"][idx_0]*100
            
                # financing activites cash expenditures
                data.loc["CASH_FREE_CASH_FLOW"] = data.loc["CASH_FROM_OPERATING_ACTIVITIES"] + data.loc["CAPITAL_EXPENDITURES"]
                
                new_results["CASH_FREE_CASH_FLOW_AVG"] = np.mean(data.loc["CASH_FREE_CASH_FLOW"])
                new_results[f"CASH_FREE_CASH_FLOW_{i}"] = data.loc["CASH_FREE_CASH_FLOW"][idx_0]

                if data.shape[1] >= idx_0 +2:
                    new_results[f"CASH_TREND_FREE_CASH_FLOW_{i}"] = deduce_trend(data.loc["CASH_FREE_CASH_FLOW"][idx_0:])

                if "OTHER_INVESTING_CASH_FLOW_ITEMS_TOTAL" in data.index:
                    new_results[f"CASH_%_INTO_ACQUISITION_{i}"] = data.loc["OTHER_INVESTING_CASH_FLOW_ITEMS_TOTAL"][idx_0]*100 /  data.loc["CASH_FROM_OPERATING_ACTIVITIES"][idx_0]
                
            if "ISSUANCE_RETIREMENT_OF_STOCK_NET" in data.index:
                new_results[f"CASH_SHARE_STOCK_BUY_ACTIVITY_CASH_{i}"] = data.loc["ISSUANCE_RETIREMENT_OF_STOCK_NET"][idx_0]*100 / data.loc["CASH_FROM_OPERATING_ACTIVITIES"][idx_0]
            
            if "TOTAL_CASH_DIVIDENDS_PAID" in data.index:
                new_results[f"CASH_%_DIVIDENDS IN ACTIVITY_CASH_{i}"] = data.loc["TOTAL_CASH_DIVIDENDS_PAID"][idx_0] *100/ data.loc["CASH_FROM_OPERATING_ACTIVITIES"][idx_0]
            
            if "CASH_FROM_FINANCING_ACTIVITIES" in data.index:
                new_results[f"CASH_%_FINANCE IN ACTIVITY_CASH_{i}"] = data.loc["CASH_FROM_FINANCING_ACTIVITIES"][idx_0]*100 / data.loc["CASH_FROM_OPERATING_ACTIVITIES"][idx_0]

    analysis_dict.update(new_results)

    return analysis_dict
