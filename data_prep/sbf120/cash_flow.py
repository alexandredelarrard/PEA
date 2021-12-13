import pandas as pd 
from utils.general_cleaning import create_index, handle_accountancy_numbers, \
                                handle_currency, deduce_trend, deduce_specific


def cash_flow_annual(inputs, analysis_dict, params={}, granularity="ANNUAL"):
    """
    - Acquisition / investment in property -> Si plus que 100 alors pas bon car grossit par la dette
    - Trend net cash provided par operating -> cash créé par l'activité (si croit, alors good) 
    - investment in property / Net cashed used for activities  -> proportion du cash reinvesti dans l'activité 
    - common stock repurchased / Free cash flow -> si petit alors ok, si grand alors pb car pas investi dans activité
    - comparer cash flow from financing activities (common stock issued) avec net cash provided par operating (si fin >> cash, alors OPM)
    TODO:
    """

    # 'CASH-FLOW-ANNUAL'
    data = inputs[f'CASH-FLOW-{granularity}'].copy()
    new_results = {}
    data = create_index(data)
    data = handle_accountancy_numbers(data)
    data = handle_currency(params, data)
    params["specific"] = deduce_specific(data)

    new_results["CASH_FROM_OPERATING_ACTIVITIES"] = data.loc["CASH_FROM_OPERATING_ACTIVITIES"][0]
    new_results["CASH_TREND_FROM_OPERATING_ACTIVITIES"] = deduce_trend(data.loc["CASH_FROM_OPERATING_ACTIVITIES"])
    new_results["CASH_TREND_NET_INCOME"] = deduce_trend(data.loc["NET_INCOME_STARTING_LINE"])

    # trend of where cash is invested 
    data.loc["CASH_SHARE_INVESTED_ACTIVITY"] = data.loc["CAPITAL_EXPENDITURES"] /  data.loc["CASH_FROM_OPERATING_ACTIVITIES"]
    new_results["CASH_SHARE_INVESTED_ACTIVITY"] = data.loc["CASH_SHARE_INVESTED_ACTIVITY"][0]
    new_results["CASH_ACQUISITION_OVER_ACTIVITY"] = data.loc["OTHER_INVESTING_CASH_FLOW_ITEMS_TOTAL"][0] /  data.loc["CAPITAL_EXPENDITURES"][0]
    new_results["CASH_TREND_SHARE_INVESTED_ACTIVITY"] = deduce_trend(data.loc["CASH_SHARE_INVESTED_ACTIVITY"])

    # financing activites cash expenditures
    free_cash_flow = data.loc["CASH_FROM_OPERATING_ACTIVITIES"][0] + data.loc["CAPITAL_EXPENDITURES"][0]
    new_results["CASH_SHARE_STOCK_BUY_ACTIVITY_CASH"] = data.loc["ISSUANCE_RETIREMENT_OF_STOCK_NET"][0] / data.loc["CASH_FROM_OPERATING_ACTIVITIES"][0]
    new_results["CASH_SHARE_DIV_PAID_ACTIVITY_CASH"] = data.loc["TOTAL_CASH_DIVIDENDS_PAID"][0] / data.loc["CASH_FROM_OPERATING_ACTIVITIES"][0]
    new_results["CASH_FINANCE_OVER_ACTIVITY_CASH"] = data.loc["CASH_FROM_FINANCING_ACTIVITIES"][0] / data.loc["CASH_FROM_OPERATING_ACTIVITIES"][0]

    analysis_dict.update(new_results)

    return analysis_dict
