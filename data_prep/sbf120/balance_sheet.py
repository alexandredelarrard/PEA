from utils.general_cleaning import create_index, handle_accountancy_numbers, \
                                    handle_currency, deduce_trend, deduce_specific
import pandas as pd 


def balance_sheet(inputs, analysis_dict, params={}):
    """
    TODO: 
    - Current Asset / Current liabilities -> + c'est gd mieux c'est 
    - Evolution Total Liabilities & Shareholders' Equity -> Si augmente alors de plus en 
    plus pour shareholder (dette augmente moins vite que assets)
    """

    data = inputs[f'BALANCE-SHEET-ANNUAL'].copy()
    new_results = {}
    data = create_index(data)
    data = handle_accountancy_numbers(data)
    data = handle_currency(params, data)

    if "TOTAL_CURRENT_ASSETS" not in data.index:
        data.loc["TOTAL_CURRENT_ASSETS"] = data.loc["TOTAL_ASSETS"]
        data.loc["TOTAL_CURRENT_LIABILITIES"] = data.loc["TOTAL_LIABILITIES"]

    # short term assets over debts 
    data.loc["CURRENT_ASSETS_OVER_LIABILITIES"]= data.loc["TOTAL_CURRENT_ASSETS"] / data.loc["TOTAL_CURRENT_LIABILITIES"] 
    new_results["BALANCE_CURRENT_ASSET_LIAB"] = data.loc["CURRENT_ASSETS_OVER_LIABILITIES"][0]
    new_results["BALANCE_TREND_CURRENT_ASSET_LIAB"] = deduce_trend(data.loc["CURRENT_ASSETS_OVER_LIABILITIES"])

    data.loc["SHAREHOLDERS_EQUITY"] = data.loc["TOTAL_ASSETS"] - data.loc["TOTAL_LIABILITIES"] 
    new_results["BALANCE_SHAREHOLDERS_EQUITY"] = data.loc["SHAREHOLDERS_EQUITY"][0]
    new_results["BALANCE_TREND_SHAREHOLDERS_EQUITY"] = deduce_trend(data.loc["SHAREHOLDERS_EQUITY"])

    # debt to asset ratio 
    new_results["DEBT_TO_ASSET_RATIO"] = data.loc["TOTAL_ASSETS"][0] / data.loc["TOTAL_LIABILITIES"][0]

    if "GOODWILL_NET" in data.index:
        new_results["BALANCE_SHARE_GOODWILL_ASSETS"] = data.loc["GOODWILL_NET"][0] / data.loc["TOTAL_ASSETS"][0] 

    analysis_dict.update(new_results)

    return analysis_dict
