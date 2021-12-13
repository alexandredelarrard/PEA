import pandas as pd 
import numpy as np
from pandas.core.indexes import period
from utils.general_cleaning import create_index, handle_accountancy_numbers, \
                                   handle_currency, deduce_trend, deduce_specific


def add_ttm(inputs, data, params):

    qtrl = inputs[f'INCOME-STATEMENT-QUARTERLY'].copy()
    qtrl = create_index(qtrl)
    qtrl = handle_accountancy_numbers(qtrl)
    qtrl = handle_currency(params, qtrl)

    qtrl.columns = [pd.to_datetime(x).to_period("Q") for x in qtrl.columns]
    qtrl.columns = qtrl.columns.to_series().astype(str)

    if params["specific"] == "bank":
        qtrl.loc["TOTAL_REVENUE"] = qtrl.loc["INTEREST_INCOME_BANK"] + qtrl.loc["NONINTEREST_INCOME_BANK"]
        qtrl.loc["TOTAL_OPERATING_EXPENSE"] = qtrl.loc["TOTAL_INTEREST_EXPENSE"] + qtrl.loc["LOAN_LOSS_PROVISION"] + abs(data.loc["NONINTEREST_EXPENSE_BANK"])
        qtrl.loc["OPERATING_INCOME"] = qtrl.loc["NET_INCOME_BEFORE_TAXES"]

    if qtrl.shape[1] == 5:
        june_dec = [x for x in qtrl.columns if "Q2" in x or "Q4" in x][:2] # only first 2 
        reste = [x for x in qtrl.columns if "Q2" not in x and "Q4" not in x]
        avg_jun_dec = qtrl.loc["NET_INCOME", june_dec].mean()
        avg_rest = qtrl.loc["NET_INCOME", reste].mean()
         
        if avg_jun_dec/avg_rest > 1.7: # plus de 60% d'un coup -> don't trust it 
            qtrl["TTM"] = qtrl.loc[:,june_dec].sum(axis=1)
        else: 
            qtrl["TTM"] = qtrl.iloc[:,:4].sum(axis=1)
        data = pd.merge(qtrl[["TTM"]], data, left_index=True, right_index=True, how="right")

    elif qtrl.shape[1] == 2:
        qtrl["TTM"] = qtrl.iloc[:,:].sum(axis=1)
        data = pd.merge(qtrl[["TTM"]], data, left_index=True, right_index=True, how="right")

    else:
        print(f"NO QUARTERLY DATA FOR {params['company']}")
        data["TTM"] = data.iloc[:,0]
        qtrl = None

    return data, qtrl


def stock_processing(inputs):

    stock = inputs["STOCK"].copy()

    # round stock evolution per quarter 
    stock.columns = [x.upper().replace(" ", "_") for x in stock.columns]
    stock = stock[["DATE", "CLOSE", "VOLUME"]]

    # round to Quarter 
    stock["DATE"] = pd.to_datetime(stock["DATE"], format="%Y-%m-%d")
    stock["YEAR"] = stock["DATE"].dt.year
    stock["MONTH"] = stock["DATE"].dt.month
    stock["QUARTER"] = stock['DATE'].dt.to_period('Q')

    agg_stock  = stock[["QUARTER", "CLOSE", "VOLUME"]].groupby("QUARTER").mean()

    # trend quarter to quarter 
    for col in ["CLOSE", "VOLUME"]:
        agg_stock[f"{col}_Q_TREND"] = (agg_stock[col] - agg_stock[col].shift(1)) / agg_stock[col].shift(1)
        agg_stock[f"{col}_Y_TREND"] = (agg_stock[col] - agg_stock[col].shift(4)) / agg_stock[col].shift(4)

    return agg_stock


def income_state(inputs, analysis_dict, params={}):
    """
    TODO: 
    - Add operating margin : operating income or loss  / CA (should be around 15%) -> Done
    - tendence de l'operating income sur les X dernières années -> Done
    - tendence de net income sur les X dernières années -> Done
    - d'ou vient l'increase majoritaire sur les gains (taxes plus passes, cout plus bas , etc) ? 
    - est ce que l'operating income decrease vient de la R&D ? 
    - On veut voir l'operating income tendence comparée au CA
    - Aussi la tendence du CA par rapport à la tendence du cout admin 
    - Tendence de l'operating income comparée à la R&D et au CA 
    """

    data = inputs[f'INCOME-STATEMENT-ANNUAL'].copy()

    new_results = {}
    data = create_index(data)
    data = handle_accountancy_numbers(data)
    data = handle_currency(params, data)
    params["specific"] = deduce_specific(data)

    data.columns = [pd.to_datetime(x).to_period("Q") for x in data.columns]

    # stock processing 
    stock = stock_processing(inputs)

    # ad TTM to all years
    # data, qtrl = add_ttm(inputs, data, params)

    if params["specific"] == "bank":
        data.loc["TOTAL_REVENUE"] = data.loc["INTEREST_INCOME_BANK"] + data.loc["NONINTEREST_INCOME_BANK"]
        data.loc["TOTAL_OPERATING_EXPENSE"] = data.loc["TOTAL_INTEREST_EXPENSE"] + data.loc["LOAN_LOSS_PROVISION"] + abs(data.loc["NONINTEREST_EXPENSE_BANK"])
        data.loc["OPERATING_INCOME"] = data.loc["NET_INCOME_BEFORE_TAXES"]

    # Temps T status latest Yearly FINANCES 
    new_results["TOTAL_REVENUE"] = data.loc["TOTAL_REVENUE"][0]
    new_results["NET_INCOME_BEFORE_EXTRA_ITEMS"] = data.loc["NET_INCOME_BEFORE_EXTRA_ITEMS"][0]
    new_results["OPERATING_INCOME"] = data.loc["OPERATING_INCOME"][0]
    
    new_results["NET_PROFIT_MARGIN"] = data.loc["NET_INCOME"][0] / (data.loc["TOTAL_REVENUE"][0]+0.01)
    new_results["TAXES_&_EXTRA_IN_TOTAL_REVENUE"] = (data.loc["TOTAL_OPERATING_EXPENSE"][0] - data.loc["NET_INCOME"][0])/ (data.loc["TOTAL_REVENUE"][0])

    new_results["TOTAL_REVENUE_LONG_TREND"] = deduce_trend(data.loc["TOTAL_REVENUE"])
    new_results["NET_INCOME_BEFORE_EXTRA_ITEMS_LONG_TREND"] =  deduce_trend(data.loc["NET_INCOME_BEFORE_EXTRA_ITEMS"])
    new_results["OP_INCOME_LONG_TREND"] =  deduce_trend(data.loc["OPERATING_INCOME"])
    
    # give trend if revenue increase faster than op income
    new_results["OP_INCOME_VS_TOTAL_REVENUE"] = new_results["OP_INCOME_LONG_TREND"] / new_results["TOTAL_REVENUE_LONG_TREND"]

    if "SELLING_GENERAL_ADMIN_EXPENSES_TOTAL" in data.index:
        data.loc["SHARE_SALES_GENERAL_IN_REVENUE"] = data.loc["SELLING_GENERAL_ADMIN_EXPENSES_TOTAL"] / data.loc["TOTAL_REVENUE"]
        new_results["SALES_GENERAL_IN_REVENUE"] = data.loc["SHARE_SALES_GENERAL_IN_REVENUE"][0]
        new_results["SHARE_SALES_GENERAL_IN_REVENUE_LONG_TREND"] = deduce_trend(data.loc[f"SHARE_SALES_GENERAL_IN_REVENUE"])

    if "RESEARCH__DEVELOPMENT" in data.index:
        data.loc["R&D_SHARE_OPERATING_INCOME"] =  data.loc["RESEARCH__DEVELOPMENT"] / (data.loc["RESEARCH__DEVELOPMENT"] + data.loc["OPERATING_INCOME"])
        new_results["R&D_SHARE_OPERATING_INCOME"] = data.loc["R&D_SHARE_OPERATING_INCOME"][0]
        new_results["R&D_SHARE_OPERATING_INCOME_TREND"] = deduce_trend(data.loc["R&D_SHARE_OPERATING_INCOME"])

    # if isinstance(qtrl, pd.DataFrame):
    #     if qtrl.shape[1] == 5:
    #         for col in ["NET_INCOME_BEFORE_EXTRA_ITEMS", "TOTAL_REVENUE"]:
    #             new_results[f"Q_TO_Q_{col}"] =  (qtrl.loc[col][0] - qtrl.loc[col][4]) / qtrl.loc[col][4] 
            
    # add stock values +1Q
    data = add_stock_to_data(stock, data)

    # stock trend vs income trend
    new_results["STOCK_CLOSE_PLUS_1Q"] = data.loc["STOCK_CLOSE_PLUS_1Q"][0]
    new_results["STOCK_CLOSE_PLUS_1Q_TREND"] = deduce_trend(data.loc["STOCK_CLOSE_PLUS_1Q"])
    new_results["STOCK_VS_INCOME_RATIO_TREND"] = new_results["STOCK_CLOSE_PLUS_1Q_TREND"] / new_results["NET_INCOME_BEFORE_EXTRA_ITEMS_LONG_TREND"]

    # add p/e
    if  "SHARES_OUT_MIL" in analysis_dict.keys():
        nbr_shares = analysis_dict["SHARES_OUT_MIL"]

        data.loc["P/E"] = nbr_shares*data.loc["STOCK_CLOSE_PLUS_1Q"] / data.loc["NET_INCOME_BEFORE_EXTRA_ITEMS"]  
        new_results["TREND_P/E"] = deduce_trend(data.loc["P/E"])

        # new_results["FOREWARD_P/E_TTM"] = data.loc["P/E", "TTM"]
        new_results["FOREWARD_P/E_Y"] = data.loc["P/E", data.columns[0]]

    analysis_dict.update(new_results)

    return analysis_dict


def add_stock_to_data(stock, data):

    stocks_values = []
    missing_quarters = []
    volume_values = []
    for col in data.columns:
        if col == "TTM":
            stocks_values.append(stock.iloc[-1, 0])
            volume_values.append(stock.iloc[-1, 1])
        else:
            if col in stock.index:
                stocks_values.append(stock.loc[col + 1, "CLOSE"])
                volume_values.append(stock.loc[col + 1, "VOLUME"])
            else:
                stocks_values.append(np.nan)
                volume_values.append(np.nan)
    
    data.loc["STOCK_CLOSE_PLUS_1Q"] = stocks_values
    data.loc["STOCK_VOLUME_PLUS_1Q"] = volume_values

    return data
