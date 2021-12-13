import pandas as pd 
import numpy as np
import os 
import tqdm
from pathlib import Path as pl
from utils.general_cleaning import create_index, deduce_currency

from data_prep.sbf120.balance_sheet import balance_sheet
from data_prep.sbf120.income_statement import income_state

def people_analysis(inputs, analysis_dict):
    """TODO: Men vs Women in board

    Args:
        inputs ([type]): [description]
        analysis_dict ([type]): [description]

    Returns:
        [type]: [description]
    """

    data = inputs["PEOPLE"].copy()
    data = data.loc[~data["NAME"].isnull()]
    data = data.loc[data["POSITION"] != "Independent Director"]
    new_results = {}

    # replace age missing 
    for col in ["AGE", "APPOINTED"]:
        data[col] = data[col].apply(lambda x : str(x).replace("--", "0")).astype(int)
        data[col] = np.where(data[col] == 0, np.nan, data[col])

    is_ceo = data["POSITION"].apply(lambda x: "Chief Executive Officer," in x)
    is_cfo = data["POSITION"].apply(lambda x: "Chief Financial Officer," in x)
    is_coo = data["POSITION"].apply(lambda x: "Chief Operating Officer," in x)

    new_results["CEO_APPOINTED"] = data.loc[is_ceo, "APPOINTED"].mean()
    new_results["C_AVG_APPOINTED"] = data.loc[is_cfo + is_coo + is_ceo, "APPOINTED"].mean()
    new_results["LEADER_AGE_AVG"] = data["AGE"].mean()
    new_results["LEADER_APPOINTED_AVG"] = data["APPOINTED"].mean()

    analysis_dict.update(new_results)

    return analysis_dict


def profile_analysis(inputs, analysis_dict):

    data = inputs['PROFILE'].copy()
    data = create_index(data)
    
    new_results = {}
    new_results["MARKET CAP"] = float(data.loc["MARKET_CAP_MIL"][0].replace(",",""))
    new_results["DESC"] = data.loc["PRESENTATION"][0]
    
    if "SHARES_OUT_MIL" in data.index:
        new_results["SHARES_OUT_MIL"] = float(data.loc["SHARES_OUT_MIL"][0].replace(",",""))

    if "FORWARD_P_E" in data.index:
        if data.loc["FORWARD_P_E"][0] != "--":
            new_results["FORWARD_P_E"] = float(data.loc["FORWARD_P_E"][0].replace(",",""))

    if "RATING" in data.index:
        new_results["RATING"] = float(data.loc["RATING"][0].split("-")[0].replace(" mean rating ", ""))
        new_results["RATING_NBR_ANALYSTS"] = float(data.loc["RATING"][0].split("-")[1].replace(" analysts", "").strip())

    analysis_dict.update(new_results)

    return analysis_dict


def read_files(params):

    inputs = {}

    liste_dates_company = os.listdir(params["base_path"] / pl(params["company"]))
    finance_date = max(liste_dates_company)
    liste_files_date_company = os.listdir(params["base_path"] / pl(params["company"]) / finance_date)

    for file in liste_files_date_company: 
        f = file.replace(".csv", "")
        try:
            inputs[f] = pd.read_csv(params["base_path"] / pl(params["company"]) / pl(finance_date) / file)
        except Exception as e: 
            # print(f"ERROR LOAD DATA : {params['company']} / {f} / {e}")
            pass

    return inputs


def main_analysis_financials(configs_general):

    base_path = configs_general["resources"]["base_path"] / pl("data/extracted_data")
    liste_companies = os.listdir(base_path)
    results_analysis = {}

    for company in tqdm.tqdm(list(set(liste_companies) - set(["currencies"]))):

        results_analysis[company] = {}
        params = {"specific" : "",
                  "company" : company,
                  "base_path" : base_path}
        
        inputs = read_files(params)

        # parameters
        params["currency"] = deduce_currency(company)
      
        # list of analysis
        try: 
            results_analysis[company] = people_analysis(inputs, results_analysis[company])
        except Exception as e:
            print(company, e)
            pass

        try: 
            results_analysis[company] = profile_analysis(inputs, results_analysis[company])
        except Exception as e:
            print(company, e)
            pass

        try:
            results_analysis[company] = income_state(inputs, results_analysis[company], params)
        except Exception as e:
            print(company, e)
            pass

        try: 
            results_analysis[company] = balance_sheet(inputs, results_analysis[company], params)
        except Exception  as e:
            print(company, e)
            pass

    results_analysis[company]["SPECIFIC"] = params["specific"]

    # shape and filter results
    results = pd.DataFrame(results_analysis).sort_index()

    # if more than 90% variables missing, drop company 
    to_drop = results.isnull().sum().loc[results.isnull().sum() > 0.9*results.shape[0]].index
    results = results.drop(to_drop, axis=1)

    second_drop = results.T.loc[(results.T['TOTAL_REVENUE'] < 100)].index
    results = results.drop(second_drop, axis=1)

    pe_drop = results.T.loc[(results.T["TREND_P/E"] > 10000)].index
    results = results.drop(pe_drop, axis=1)

    print(f"Dropped MVS : {len(to_drop)}")
    print(f"Dropped Less 10M CA : {len(second_drop)}")
    print(f"Dropped MISSING PE : {len(pe_drop)}")
    print(f"FINAL shape {results.shape}")

    # POST PROCESSING SANITY CHECKS
    results = results.T

    # keep firms with less margin than revenue (otherwise they live on debt / fund raising)
    results = results.loc[results['NET_PROFIT_MARGIN'] <=1]

    # <0 R&D does not make sense 
    results["R&D_SHARE_OPERATING_INCOME"] = results["R&D_SHARE_OPERATING_INCOME"].clip(0, None)
    
    return results