from datetime import datetime, timedelta
import pandas as pd 
import numpy as np
from utils.general_cleaning import create_index, handle_accountancy_numbers, \
                                   handle_currency, deduce_trend,\
                                   load_currency
from data_prep.sbf120.stock import stock_processing, add_stock_to_data
from data_prep.sbf120.handle_quarters import add_ttm


def deduce_intrinsuc_value(x, duree_vie=10):

    # 10% inflation per year to be conservative
    deduced_rate = lambda x, y: y/ (1 + 0.1)**x
    decote = 0.85

    future_cash_flow = sum([deduced_rate(i, x[0]*(1 + (x[1]/100)*decote**i)**i) for i in range(duree_vie)])
    future_sell = x[2]*decote
    intrinsic_value = future_cash_flow + future_sell

    if intrinsic_value> 0:
        return intrinsic_value
    else: 
        return np.nan


def add_financial_info(new_results, data, analysis_dict, stock, df_currency, idx_0):

    new_results[f"_TOTAL_REVENUE_{idx_0}"] = data.loc["TOTAL_REVENUE"][idx_0]
    new_results[f"_NET_INCOME_BEFORE_EXTRA_ITEMS_{idx_0}"] = data.loc["NET_INCOME_BEFORE_EXTRA_ITEMS"][idx_0]
    new_results[f"_OPERATING_INCOME_{idx_0}"] = data.loc["OPERATING_INCOME"][idx_0]

    # % analysis
    new_results[f"%_NET_PROFIT_MARGIN_{idx_0}"] = data.loc["NET_INCOME"][idx_0]*100 / (data.loc["TOTAL_REVENUE"][idx_0]+0.01)
    new_results[f"%_TAXES_&_EXTRA_IN_TOTAL_REVENUE_{idx_0}"] = (data.loc["OPERATING_INCOME"][idx_0] - data.loc["NET_INCOME"][idx_0])*100/ (data.loc["TOTAL_REVENUE"][idx_0])

    # YOY analysis
    if data.shape[1] >= idx_0 +2:
        new_results[f"%_YOY_TOTAL_REVENUE_{idx_0}"] = (data.loc["TOTAL_REVENUE"][idx_0] - data.loc["TOTAL_REVENUE"][idx_0+1])*100 / (0.1 + np.abs(data.loc["TOTAL_REVENUE"][idx_0+1]))
        new_results[f"%_YOY_NET_INCOME_BEFORE_EXTRA_ITEMS_{idx_0}"] = (data.loc["NET_INCOME_BEFORE_EXTRA_ITEMS"][idx_0] - data.loc["NET_INCOME_BEFORE_EXTRA_ITEMS"][idx_0+1])*100 / (0.1 + np.abs(data.loc["NET_INCOME_BEFORE_EXTRA_ITEMS"][idx_0+1]))
        new_results[f"%_YOY_OPERATING_INCOME_{idx_0}"] = (data.loc["OPERATING_INCOME"][idx_0] -data.loc["OPERATING_INCOME"][idx_0+1])*100/ (0.1 + np.abs(data.loc["OPERATING_INCOME"][idx_0+1]))
    
    # 2_YOY analysis
    if len(data.loc["TOTAL_REVENUE"]) > idx_0 + 3:
        new_results[f"%_YO2Y_TOTAL_REVENUE_{idx_0}"] = (data.loc["TOTAL_REVENUE"][idx_0] - data.loc["TOTAL_REVENUE"][idx_0+2])*100 / np.abs(data.loc["TOTAL_REVENUE"][idx_0+2])
        new_results[f"%_YO2Y_NET_INCOME_BEFORE_EXTRA_ITEMS_{idx_0}"] = (data.loc["NET_INCOME_BEFORE_EXTRA_ITEMS"][idx_0] - data.loc["NET_INCOME_BEFORE_EXTRA_ITEMS"][idx_0+2])*100 / (0.1 + np.abs(data.loc["NET_INCOME_BEFORE_EXTRA_ITEMS"][idx_0+2]))
        new_results[f"%_YO2Y_OPERATING_INCOME_{idx_0}"] = (data.loc["OPERATING_INCOME"][idx_0] -data.loc["OPERATING_INCOME"][idx_0+2])*100/(0.1 + np.abs(data.loc["OPERATING_INCOME"][idx_0+2]))

    # Trend analysis
    if data.shape[1] >= idx_0 +2:
        new_results[f"LONG_TREND_TOTAL_REVENUE_{idx_0}"] = deduce_trend(data.loc["TOTAL_REVENUE"][idx_0:])
        new_results[f"LONG_TREND_NET_INCOME_BEFORE_EXTRA_ITEMS_{idx_0}"] = deduce_trend(data.loc["NET_INCOME_BEFORE_EXTRA_ITEMS"][idx_0:])
        new_results[f"LONG_TREND_OP_INCOME_{idx_0}"] = deduce_trend(data.loc["OPERATING_INCOME"][idx_0:])
        
        # give trend if revenue increase faster than op income
        new_results[f"TREND_OP_INCOME / TREND_TOTAL_REVENUE_{idx_0}"] = new_results[f"LONG_TREND_OP_INCOME_{idx_0}"] / new_results[f"LONG_TREND_TOTAL_REVENUE_{idx_0}"]

    if "SELLING_GENERAL_ADMIN_EXPENSES_TOTAL" in data.index:
        data.loc["SHARE_SALES_GENERAL_IN_REVENUE"] = data.loc["SELLING_GENERAL_ADMIN_EXPENSES_TOTAL"] / data.loc["TOTAL_REVENUE"]
        new_results[f"%_SALES_GENERAL_IN_REVENUE_{idx_0}"] = data.loc["SHARE_SALES_GENERAL_IN_REVENUE"][idx_0]*100

        if data.shape[1] >= idx_0 +2:
            new_results[f"LONG_TREND_SHARE_SALES_GENERAL_IN_REVENUE_{idx_0}"] = deduce_trend(data.loc[f"SHARE_SALES_GENERAL_IN_REVENUE"][idx_0:])

    if "RESEARCH__DEVELOPMENT" in data.index:
        data.loc["R&D_SHARE_OPERATING_INCOME"] =  data.loc["RESEARCH__DEVELOPMENT"] / (data.loc["RESEARCH__DEVELOPMENT"] + data.loc["OPERATING_INCOME"])
        data.loc["R&D_SHARE_OPERATING_INCOME"] = np.where((data.loc["RESEARCH__DEVELOPMENT"] + data.loc["OPERATING_INCOME"]) == 0, 1, data.loc["R&D_SHARE_OPERATING_INCOME"])
        
        new_results[f"_R&D_{idx_0}"] = data.loc["RESEARCH__DEVELOPMENT"][idx_0]
        new_results[f"%_R&D_IN_OPERATING_INCOME_{idx_0}"] = data.loc["R&D_SHARE_OPERATING_INCOME"][idx_0]*100

        if data.shape[1] >= idx_0 +2:
            new_results[f"%_YOY_R&D_{idx_0}"] = (data.loc["RESEARCH__DEVELOPMENT"][idx_0] - data.loc["RESEARCH__DEVELOPMENT"][idx_0+1])*100 /(0.1 + np.abs(data.loc["RESEARCH__DEVELOPMENT"][idx_0+1]))
            new_results[f"LONG_TREND_R&D_SHARE_OPERATING_INCOME_{idx_0}"] = deduce_trend(data.loc["R&D_SHARE_OPERATING_INCOME"][idx_0:])
    
    if data.shape[1] >= idx_0 +2:
        new_results[f"STOCK_LONG_TREND_CLOSE_PLUS_1Q_{idx_0}"] = deduce_trend(data.loc["STOCK_CLOSE_PLUS_1Q"][idx_0:])
        new_results[f"STOCK_% TREND_NET_INCOME_WO_EXTRA - STOCK_TREND_{idx_0}"] = -1*(new_results[f"STOCK_LONG_TREND_CLOSE_PLUS_1Q_{idx_0}"] - new_results[f"LONG_TREND_NET_INCOME_BEFORE_EXTRA_ITEMS_{idx_0}"])*100

    # stock trend vs income trend
    if idx_0 == 0: 
        day_first = datetime.today() 
    else:
        day_first = (data.columns[idx_0] + 1).to_timestamp()

    # STOCK ANALYASIS        
    analysis_dict["NBR_SHARES"] = analysis_dict["PROFILE_MARKET_CAP"]/(stock.iloc[-1,1]*df_currency.iloc[-1,1])
    
    yearly_stock = stock.loc[stock["DATE"].between(day_first - timedelta(days=365), day_first), "CLOSE"]
    yearly_vol = stock.loc[stock["DATE"].between(day_first - timedelta(days=365), day_first), "VOLUME"]
    items = min(91, len(yearly_stock))

    # keep stock between 52 weeks and today 
    if yearly_stock.shape[0] > 0:
        new_results[f"STOCK_%_52_WEEKS_{idx_0}"] = (yearly_stock.iloc[-items] - yearly_stock.iloc[0])*100 / yearly_stock.iloc[0] 
        new_results[f"STOCK_MAX_52_WEEKS_{idx_0}"]  = yearly_stock[items:].max()
        new_results[f"STOCK_MIN_52_WEEKS_{idx_0}"]  = yearly_stock[items:].min()
        new_results[f"STOCK_CLOSE_TODAY_{idx_0}"] = yearly_stock.iloc[-1]

        # VOLUME
        new_results[f"STOCK_VOLUME_52_WEEKS_{idx_0}"] = (yearly_vol.iloc[-items] - yearly_vol.iloc[0])*100 / yearly_vol.iloc[0] 
        new_results[f"STOCK_VOLUME_MAX_52_WEEKS_{idx_0}"] = yearly_vol[items:].max()
        new_results[f"STOCK_VOLUME_MIN_52_WEEKS_{idx_0}"] = yearly_vol[items:].min()
        new_results[f"STOCK_VOLUME_CLOSE_TODAY_{idx_0}"] = yearly_vol.iloc[-1]

        # calculate intrinsic value 
        if f"CASH_TREND_FREE_CASH_FLOW_{idx_0}" in analysis_dict.keys():
            if not pd.isnull(analysis_dict[f"CASH_TREND_FREE_CASH_FLOW_{idx_0}"]) : 
                analysis_dict[f"CASH_TREND_FREE_CASH_FLOW_{idx_0}"] = analysis_dict[f"CASH_TREND_FREE_CASH_FLOW_{idx_0}"].clip(-35, 35)
            else: 
                analysis_dict[f"CASH_TREND_FREE_CASH_FLOW_{idx_0}"] = 0

            new_results[f"MARKET_CAP_{idx_0}"] = analysis_dict["NBR_SHARES"]*new_results[f"STOCK_CLOSE_TODAY_{idx_0}"]
            
            if f"BALANCE_SHAREHOLDERS_EQUITY_{idx_0}" in analysis_dict.keys():
                avg_free_cash = analysis_dict["CASH_FREE_CASH_FLOW_AVG"]
                new_results[f"INTRINSIC_VALUE_{idx_0}"] = deduce_intrinsuc_value([avg_free_cash, analysis_dict[f"CASH_TREND_FREE_CASH_FLOW_{idx_0}"], analysis_dict[f"BALANCE_SHAREHOLDERS_EQUITY_{idx_0}"]])
                new_results[f"DISTANCE MARKET_CAP / INTRINSIC _{idx_0}"] =new_results[f"MARKET_CAP_{idx_0}"]*100 / new_results[f"INTRINSIC_VALUE_{idx_0}"] 

                if "PROFILE_MARKET_CAP" in analysis_dict.keys(): 
                    new_results[f"BALANCE_%_MARKET_CAP / SHAREHOLDERS_EQUITY_{idx_0}"] = new_results[f"MARKET_CAP_{idx_0}"]*100 / analysis_dict[f"BALANCE_SHAREHOLDERS_EQUITY_{idx_0}"]
        
    return new_results


def add_pe_infos(new_results, data, analysis_dict, df_currency, agg_stock, max_quarter):

    today = datetime.today().strftime("%d-%m-%Y")

    # get right currency mapping 
    mapped_currencies = []
    for x in data.columns:
        if x != "TTM":
            curre = df_currency.loc[df_currency["Month"].isin([(x+1).to_timestamp().to_period("M")]), "Close"].values[0]
            mapped_currencies.append(curre)
        else:
            curre = df_currency.loc[df_currency["Month"].isin([pd.to_datetime(today, format="%d-%m-%Y").to_period("M")]), "Close"].values[0]
            mapped_currencies.append(curre)
    
    # get righ stock on results 
    mapped_stock = []
    for x in data.columns:
        if x == "TTM":
            mapped_stock.append(agg_stock.loc[max_quarter, "CLOSE"])
        else:
            if x in agg_stock.index:
                mapped_stock.append(agg_stock.loc[x, "CLOSE"])
            else:
                mapped_stock.append(np.nan)
    
    if "UNUSUAL_EXPENSES" in data.index:
        data.loc["P/E_Q+1"] = list(analysis_dict["NBR_SHARES"]*data.loc["STOCK_CLOSE_PLUS_1Q"]*mapped_currencies / (data.loc["NET_INCOME_BEFORE_EXTRA_ITEMS"] + data.loc["UNUSUAL_EXPENSES"]))
        data.loc["P/E_Q"] = list(analysis_dict["NBR_SHARES"]*mapped_stock*mapped_currencies / (data.loc["NET_INCOME_BEFORE_EXTRA_ITEMS"] + data.loc["UNUSUAL_EXPENSES"]))
    else:
        data.loc["P/E_Q+1"] = analysis_dict["NBR_SHARES"]*data.loc["STOCK_CLOSE_PLUS_1Q"]*mapped_currencies / data.loc["NET_INCOME_BEFORE_EXTRA_ITEMS"]
        data.loc["P/E_Q"] = analysis_dict["NBR_SHARES"]*np.array(mapped_stock)*mapped_currencies / (data.loc["NET_INCOME_BEFORE_EXTRA_ITEMS"])

    for i in range(4):
        try:
            new_results[f"_P/E_+1Q_Y-{i}"] = data.loc["P/E_Q+1"][i]
            new_results[f"_P/E_Y-{i}"] = data.loc["P/E_Q"][i]
        except Exception:
            pass

    new_results["LATEST_DATE_FINANCIALS"] = data.columns[1].strftime("%Y-%m")
    new_results["LATEST_Q_EQUAL_LAST_Y"] = data.columns[1] == max_quarter

    return new_results


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
    df_currency =  load_currency(params, which_currency="currency_stock")
    data = create_index(data)
    data = handle_accountancy_numbers(data)
    data = handle_currency(params, data, which_currency="currency")

    data.columns = [pd.to_datetime(x).to_period("Q") for x in data.columns]

    # stock processing 
    agg_stock, stock = stock_processing(inputs)

    # ad TTM to all years
    data, new_results, max_quarter = add_ttm(inputs, data, params, new_results)

    # add stock values +1Q
    data = add_stock_to_data(agg_stock, stock, data)

    if params["specific"] == "bank":
        if "TOTAL_REVENUE" not in data.index:
            data.loc["TOTAL_REVENUE"] = data.loc["INTEREST_INCOME_BANK"] + data.loc["NONINTEREST_INCOME_BANK"]
        data.loc["OPERATING_INCOME"] = data.loc["NET_INCOME_BEFORE_TAXES"]

    # Temps T status latest Yearly FINANCES 
    for idx_0 in [0, 1, 2, 3]:
        if data.shape[1] >= idx_0 +1:
            new_results = add_financial_info(new_results, data, analysis_dict, stock, df_currency, idx_0)
            
    ### P/E ###
    new_results = add_pe_infos(new_results, data, analysis_dict, df_currency, agg_stock, max_quarter)

    # save results
    analysis_dict.update(new_results)

    return analysis_dict
