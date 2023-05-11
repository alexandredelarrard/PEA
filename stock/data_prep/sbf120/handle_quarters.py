import pandas as pd 
from utils.general_cleaning import create_index, handle_accountancy_numbers, \
                                   handle_currency

def verify_quarter_data(qtrl): 

    corrected= False
    get_wrong_quarters = []
    for i in range(len(qtrl.columns) -1):
        if qtrl.loc["TOTAL_REVENUE"][i] > 1.66*qtrl.loc["TOTAL_REVENUE"][i+1]:
            get_wrong_quarters.append(i)

    if qtrl.shape[1] == 5 and len(get_wrong_quarters) == 2:
        if get_wrong_quarters[1] - get_wrong_quarters[0] == 2:
            for idx in get_wrong_quarters:
                if idx !=4:
                    corrected = True
                    qtrl.iloc[:,idx] = qtrl.iloc[:,idx] - qtrl.iloc[:,idx + 1]

    return qtrl, corrected


def quarter_leverageable(qtrl):
    for col in ["TOTAL_REVENUE", "NET_INCOME", "NET_INCOME_BEFORE_EXTRA_ITEMS", 
                 "OPERATING_INCOME"]:
        if sum(qtrl.loc[col].isnull()) > 0:
            if sum(qtrl.loc[col].isnull()) <= 2 and qtrl.shape[1] == 5:
                qtrl.loc[col].fillna(qtrl.loc[col].median())
            else:
                return True, qtrl
    return False, qtrl


def add_ttm(inputs, data, params, new_results):

    qtrl_corrected = False

    qtrl = inputs['INCOME-STATEMENT-QUARTERLY'].copy()
    qtrl = create_index(qtrl)
    qtrl = handle_accountancy_numbers(qtrl)
    qtrl = handle_currency(params, qtrl, which_currency="currency")

    if qtrl.shape[0] < 10:
        raise Exception(f"QUARTERLY DATA IS OFF FOR {params['company']}")

    if params["specific"] == "bank":
        if "TOTAL_REVENUE" not in qtrl.index:
            qtrl.loc["TOTAL_REVENUE"] = qtrl.loc["INTEREST_INCOME_BANK"] + qtrl.loc["NONINTEREST_INCOME_BANK"]
        qtrl.loc["OPERATING_INCOME"] = qtrl.loc["NET_INCOME_BEFORE_TAXES"]

    # check we just have 1 year + 1 quarter max of data
    qtrl.columns = [pd.to_datetime(x).to_period("Q") for x in qtrl.columns]
    max_quarter= max(qtrl.columns)
    top_quarter_1y_ago = qtrl.columns[0] - 4
    quarters_to_sum = [x for x in qtrl.columns if x > top_quarter_1y_ago]

    if top_quarter_1y_ago in qtrl.columns:
        qtrl = qtrl[quarters_to_sum + [top_quarter_1y_ago]]
    else:
        qtrl = qtrl[quarters_to_sum]

    # can leverage qtrl 
    lever_qtrl, qtrl = quarter_leverageable(qtrl)

    if lever_qtrl:
        print(f"MISSING VALUES IN QTRL {params['company']}")
        qtrl["TTM"] = data.iloc[:,0]
        data = pd.merge(qtrl[["TTM"]], data, left_index=True, right_index=True, how="right")
        return data, new_results
    
    if qtrl.shape[1] == 5:

        qtrl, qtrl_corrected = verify_quarter_data(qtrl)

        if qtrl_corrected:
            print(f"QUARTER CORRECTED {params['company']}")

        qtrl["TTM"] = qtrl[quarters_to_sum].sum(axis=1)
        for col in ["NET_INCOME_BEFORE_EXTRA_ITEMS", "TOTAL_REVENUE", "OPERATING_INCOME"]:
            new_results[f"Q_TO_Q_{col}"] =  (qtrl.loc[col][0] - qtrl.loc[col][5])*100 / qtrl.loc[col][5] 

        data = pd.merge(qtrl[["TTM"]], data, left_index=True, right_index=True, how="right")
    
    elif qtrl.shape[1] == 3: # 3 semesters/ 2 must be summed, last is for trend Q To Q

        qtrl["TTM"] = qtrl[quarters_to_sum].sum(axis=1)

        data = pd.merge(qtrl[["TTM"]], data, left_index=True, right_index=True, how="right")
        for col in ["NET_INCOME_BEFORE_EXTRA_ITEMS", "TOTAL_REVENUE"]:
            new_results[f"Q_TO_Q_{col}"] =  (qtrl.loc[col][0] - qtrl.loc[col][2]) / qtrl.loc[col][2] 

    elif qtrl.shape[1] == 2:
        qtrl["TTM"] = qtrl[quarters_to_sum].sum(axis=1)
        data = pd.merge(qtrl[["TTM"]], data, left_index=True, right_index=True, how="right")

    elif qtrl.shape[1] == 4:
        qtrl["TTM"] = qtrl[quarters_to_sum].sum(axis=1)
        data = pd.merge(qtrl[["TTM"]], data, left_index=True, right_index=True, how="right")

    else:
        print(f"NO QUARTERLY DATA FOR {params['company']}")
        qtrl["TTM"] = data.iloc[:,0]
        data = pd.merge(qtrl[["TTM"]], data, left_index=True, right_index=True, how="right")

    return data, new_results, max_quarter
