import pandas as pd 
import numpy as np
import os 
import tqdm
from pathlib import Path as pl
from data_prep.sbf120.cash_flow import cash_flow_annual
from utils.general_cleaning import create_index, deduce_currency, deduce_specific

from data_prep.sbf120.balance_sheet import balance_sheet
from data_prep.sbf120.income_statement import income_state
from data_prep.sbf120.profile import profile_analysis
from data_prep.sbf120.people import people_analysis


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


def post_processing(results):

    # if more than 90% variables missing, drop company 
    to_drop = results.isnull().sum().loc[results.isnull().sum() > 0.9*results.shape[0]].index
    results = results.drop(to_drop, axis=1)

    second_drop = results.T.loc[(results.T['_TOTAL_REVENUE_1'] < 5)].index
    results = results.drop(second_drop, axis=1)

    print(f"Dropped MVS : {len(to_drop)}")
    print(f"Dropped Less 10M CA : {len(second_drop)}")

    # POST PROCESSING SANITY CHECKS
    results = results.T

    # <0 R&D does not make sense 
    for i in range(3):
        results[f"%_R&D_IN_OPERATING_INCOME_{i}"] = results[f"%_R&D_IN_OPERATING_INCOME_{i}"].clip(0, None)

    # # remove firms without stock data f
    no_foreward_pe = results.loc[results["_P/E_Y-0"].isnull()].index
    results = results.loc[~results.index.isin(no_foreward_pe)]
    print(f"Dropped MISSING PE REUTERS info : {no_foreward_pe}")

    # remove firms without cash flow / missing infos on cash flow
    no_foreward_pe = results.loc[(results["CASH_FREE_CASH_FLOW_1"] == 0)&(results["CASH_FREE_CASH_FLOW_0"] == 0)].index
    results = results.loc[~results.index.isin(no_foreward_pe)]
    print(f"Dropped NO FREE CASH FLOW info : {no_foreward_pe}")

    print(f"FINAL shape {results.shape}")

    return results


def main_analysis_financials(configs_general):

    base_path = configs_general["resources"]["base_path"] / pl("data/extracted_data")
    liste_companies = os.listdir(base_path)
    results_analysis = {}

    for company in tqdm.tqdm(list(set(liste_companies) - set(["currencies", "commodities"]))):

        results_analysis[company] = {}
        params = {"specific" : "",
                  "company" : company,
                  "base_path" : base_path}
        
        inputs = read_files(params)
        params["currency_stock"] = deduce_currency(company)

        try: 
            results_analysis[company], currency = profile_analysis(inputs, results_analysis[company], params)
        except Exception as e:
            print(company, e)
            pass

        if currency != "":
            params["currency"] = currency
        else:
            params["currency"] = params["currency_stock"]

        # deduce if bank insur or not 
        try: 
            data = inputs['INCOME-STATEMENT-ANNUAL'].copy()
            data = create_index(data)
            params["specific"] = deduce_specific(data)
        except Exception as e:
            print(company, e)
            pass

        # list of analysis
        try: 
            results_analysis[company] = people_analysis(inputs, results_analysis[company])
        except Exception as e:
            print(company, e)
            pass
 
        try: 
            results_analysis[company] = balance_sheet(inputs, results_analysis[company], params)
        except Exception  as e:
            print(company, e)
            pass

        try: 
            results_analysis[company] = cash_flow_annual(inputs, results_analysis[company], params)
        except Exception  as e:
            print(company, e)
            pass

        try:
            results_analysis[company] = income_state(inputs, results_analysis[company], params)
        except Exception as e:
            print(company, e)
            pass
    
        results_analysis[company]["SPECIFIC"] = params["specific"]

    # shape and filter results
    results = pd.DataFrame(results_analysis).sort_index()

    results = post_processing(results)

    return results
