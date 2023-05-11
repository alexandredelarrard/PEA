from utils.general_cleaning import create_index, handle_accountancy_numbers, \
                                    handle_currency, deduce_trend, deduce_specific
import pandas as pd 


def balance_sheet(inputs, analysis_dict, params={}):
    """
    TODO: 
    - Current Asset / Current liabilities -> + c'est gd mieux c'est 
    - Evolution Total Liabilities & Shareholders' Equity -> Si augmente alors de plus en 
    plus pour shareholder (dette augmente moins vite que assets)
    - Missing quarters
    """

    data = inputs[f'BALANCE-SHEET-ANNUAL'].copy()
    new_results = {}
    data = create_index(data)
    data = handle_accountancy_numbers(data)
    data = handle_currency(params, data, which_currency="currency")

    if "TOTAL_CURRENT_ASSETS" not in data.index:
        data.loc["TOTAL_CURRENT_ASSETS"] = data.loc["TOTAL_ASSETS"]
        data.loc["TOTAL_CURRENT_LIABILITIES"] = data.loc["TOTAL_LIABILITIES"]

    # short term assets over debts 
    data.loc["CURRENT_ASSETS_OVER_LIABILITIES"]= data.loc["TOTAL_CURRENT_ASSETS"] / data.loc["TOTAL_CURRENT_LIABILITIES"] 
    data.loc["SHAREHOLDERS_EQUITY"] = data.loc["TOTAL_ASSETS"] - data.loc["TOTAL_LIABILITIES"] 

    # balance idx 
    for i, idx_0 in enumerate([0, 0, 1, 2]):
        if data.shape[1] >= idx_0 +1:
            new_results[f"_TOTAL_ASSETS_{i}"] = data.loc["TOTAL_ASSETS"][idx_0]
            new_results[f"_TOTAL_LIABILITIES_{i}"] = data.loc["TOTAL_LIABILITIES"][idx_0]
            new_results[f"BALANCE_%_CURRENT_ASSET / CURRENT_DEBT_{i}"] = data.loc["CURRENT_ASSETS_OVER_LIABILITIES"][idx_0]*100
            new_results[f"BALANCE_SHAREHOLDERS_EQUITY_{i}"] = data.loc["SHAREHOLDERS_EQUITY"][idx_0]

            # debt to asset ratio 
            new_results[f"BALANCE_%_TOTAL_ASSET / TOTAL_LIABILITIES_{i}"] = data.loc["TOTAL_ASSETS"][idx_0]*100 / data.loc["TOTAL_LIABILITIES"][idx_0]

            # cash and short term to repay total debt 
            if "TOTAL_RECEIVABLES_NET" in data.index:
                new_results[f"BALANCE_%_CASH / TOTAL_DEBT_{i}"] = data.loc["TOTAL_RECEIVABLES_NET"][idx_0]*100 / data.loc["TOTAL_DEBT"][idx_0]

            if "GOODWILL_NET" in data.index:
                new_results[f"_GOODWILL_{i}"] = data.loc["GOODWILL_NET"][idx_0]
                new_results[f"BALANCE_%_SHARE_GOODWILL_ASSETS_{i}"] = data.loc["GOODWILL_NET"][idx_0]*100 / data.loc["TOTAL_ASSETS"][idx_0] 

            if data.shape[1] >= idx_0 +1:
                new_results[f"BALANCE_TREND_CURRENT_ASSET / CURRENT_DEBT_{i}"] = deduce_trend(data.loc["CURRENT_ASSETS_OVER_LIABILITIES"][idx_0:])
                new_results[f"BALANCE_TREND_SHAREHOLDERS_EQUITY_{i}"] = deduce_trend(data.loc["SHAREHOLDERS_EQUITY"][idx_0:])

    analysis_dict.update(new_results)

    return analysis_dict
