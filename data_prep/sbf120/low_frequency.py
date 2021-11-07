import pandas as pd 
import numpy as np
import os 
from pathlib import Path as pl
from utils.general_cleaning import create_index, handle_accountancy_numbers

def people_analysis(inputs, analysis_dict):

    data = inputs["PEOPLE"].copy()
    data = data.loc[~data["NAME"].isnull()]
    new_results = {}

    # replace age missing 
    for col in ["AGE", "APPOINTED"]:
        data[col] = data[col].apply(lambda x : str(x).replace("--", "0")).astype(int)
        data[col] = np.where(data[col] == 0, np.nan, data[col])

    new_results["LEADER_AGE_AVG"] = data["AGE"].mean()
    new_results["LEADER_APPOINTED_AVG"] = data["APPOINTED"].mean()

    analysis_dict.update(new_results)

    return analysis_dict


def balance_sheet_annual(inputs, analysis_dict, specific):

    data = inputs['BALANCE-SHEET-ANNUAL'].copy()
    new_results = {}
    data = create_index(data, specific)
    data = handle_accountancy_numbers(data)

    for i, col in enumerate(data.columns[:-1]):

        if specific == "insur":
            data.loc["TOTAL_CURRENT_ASSETS"][i] = data.loc["TOTAL_ASSETS"][i]
            data.loc["CASH_AND_SHORT_TERM_INVESTMENTS"][i] = data.loc["TOTAL_RECEIVABLES_NET"][i]
           
        new_results[f"LATEST_{i}"] = col
        new_results[f"CASH_EQ/ASSETS_{i}"] = data.loc["CASH_AND_SHORT_TERM_INVESTMENTS"][i]*100/ data.loc["TOTAL_CURRENT_ASSETS"][i]
        new_results[f"DEBT/CASH_{i}"] = data.loc["TOTAL_DEBT"][i]*100/ data.loc["CASH_AND_SHORT_TERM_INVESTMENTS"][i]
        
        if "LONG_TERM_INVESTMENTS" in data.index:
            new_results[f"LONG_TERM_INVEST/ASSETS_{i}"] = data.loc["LONG_TERM_INVESTMENTS"][i]*100/ data.loc["TOTAL_ASSETS"][i]
        if "TOTAL_LONG_TERM_DEBT"in data.index:
            new_results[f"LONG_TERM_DEBT/DEBTS_{i}"] = data.loc["TOTAL_LONG_TERM_DEBT"][i]*100/ data.loc["TOTAL_DEBT"][i]

    analysis_dict.update(new_results)

    return analysis_dict


def income_state_annual(inputs, analysis_dict, specific=""):

    data = inputs['INCOME-STATEMENT-ANNUAL'].copy()
    new_results = {}
    data = create_index(data, specific)
    data = handle_accountancy_numbers(data)

    # Temps T status latest Yearly FINANCES 
    for i, col in enumerate(data.columns[:-1]):

        if specific == "bank":
            data.loc["TOTAL_REVENUE"][i] = data.loc["INTEREST_INCOME_BANK"][i] + data.loc["NONINTEREST_INCOME_BANK"][i]
            data.loc["TOTAL_OPERATING_EXPENSE"][i] = data.loc["TOTAL_INTEREST_EXPENSE"][i] + data.loc["LOAN_LOSS_PROVISION"][i] + abs(data.loc["NONINTEREST_EXPENSE_BANK"][i])
            data.loc["TOTAL_REVENUE"][i+1] = data.loc["INTEREST_INCOME_BANK"][i+1] + data.loc["NONINTEREST_INCOME_BANK"][i+1]
            data.loc["TOTAL_OPERATING_EXPENSE"][i+1] = data.loc["TOTAL_INTEREST_EXPENSE"][i+1] + data.loc["LOAN_LOSS_PROVISION"][i] + abs(data.loc["NONINTEREST_EXPENSE_BANK"][i+1])
        
        new_results[f"LATEST_{i}"] = col
        new_results[f"NET_PROFIT_MARGIN_{i}"] = data.loc["NET_INCOME"][i]*100/ data.loc["TOTAL_REVENUE"][i]
        new_results[f"OPERATING_MARGIN_{i}"] = data.loc["TOTAL_OPERATING_EXPENSE"][i]*100/ data.loc["TOTAL_REVENUE"][i]

        # Evolution to Past
        new_results[f"GROWTH_YEAR-1_{i}"] = data.loc["TOTAL_REVENUE"][i]*100 / data.loc["TOTAL_REVENUE"][i+1]
        new_results[f"GROWTH_NET_INCOME_YEAR-1_{i}"] = data.loc["NET_INCOME"][i]*100 / data.loc["NET_INCOME"][i+1]

        if "SELLING_GENERAL_ADMIN_EXPENSES_TOTAL" in data.index:
            new_results[f"SALES_GENERAL_IN_OP_COST_{i}"] = data.loc["SELLING_GENERAL_ADMIN_EXPENSES_TOTAL"][i]*100 / data.loc["TOTAL_OPERATING_EXPENSE"][i]

        if "DEPRECIATION_AMORTIZATION" in data.index:
            new_results[f"DEPRECIATION_AMORTIZATION_IN_OP_COST_{i}"] = data.loc["DEPRECIATION_AMORTIZATION"][i]*100 / data.loc["TOTAL_OPERATING_EXPENSE"][i]
        
        if "RESEARCH__DEVELOPMENT" in data.index:
            new_results[f"R&D_{i}"] = data.loc["RESEARCH__DEVELOPMENT"][i]*100 / data.loc["GROSS_PROFIT"][i]
        
    # 3Y AVG GROWTH
    new_results[f"3Y_AVG_GROWTH_CA"] = 0.33*sum([new_results[f"GROWTH_YEAR-1_{i}"] for i in range(3)])
    new_results[f"3Y_AVG_GROWTH_NET_INCOME"] = 0.33*sum([new_results[f"GROWTH_NET_INCOME_YEAR-1_{i}"] for i in range(3)])

    analysis_dict.update(new_results)

    return analysis_dict


def cash_flow_annual(data, analysis_dict):
    # 'CASH-FLOW-ANNUAL'

    data = create_index(data)
    data = handle_accountancy_numbers(data)

    # Temps T status latest Yearly FINANCES 
    # for i, col in enumerate(data.columns[:-1]):
        # analysis_dict[f"NET_INC/CA_{col}"] = data.loc["NET_INCOME_BEFORE_TAXES"][i]*100/ data.loc["TOTAL_REVENUE"][i]
        # analysis_dict[f"OP_COST/CA{col}"] = data.loc["TOTAL_OPERATING_EXPENSE"][i]*100/ data.loc["TOTAL_REVENUE"][i]

        # analysis_dict[f"DEPRECIATION_AMORTIZATION_IN_OP_COST_{col}"] = data.loc["DEPRECIATION_AMORTIZATION"][i]*100 / data.loc["TOTAL_OPERATING_EXPENSE"][i]
        # analysis_dict[f"SALES_GENERAL_IN_OP_COST_{col}"] = data.loc["SELLING_GENERAL_ADMIN_EXPENSES_TOTAL"][i]*100 / data.loc["TOTAL_OPERATING_EXPENSE"][i]

        # # Evolution to Past
        # analysis_dict[f"EVOL_REVENU_YEAR-1_{col}"] = data.loc["TOTAL_REVENUE"][i]*100 / data.loc["TOTAL_REVENUE"][i+1]
        # analysis_dict[f"EVOL_ENT_INC_YEAR-1_{col}"] = data.loc["NET_INCOME_BEFORE_TAXES"][i]*100 / data.loc["NET_INCOME_BEFORE_TAXES"][i+1]

    # return analysis_dict


def main_analysis_financials(configs_general):

    base_path = configs_general["resources"]["base_path"] / pl("data/extracted_data")
    liste_companies = os.listdir(base_path)

    results_analysis = {}

    for company in liste_companies:
        print(f"ANALYSIS {company}")

        specific = ""
        if company in configs_general["data"]["banks"]:
            specific="bank"
        if company in configs_general["data"]["insur"]:
            specific="insur"

        inputs = {}
        results_analysis[company] = {}

        liste_dates_company = os.listdir(base_path / pl(company))
        finance_date = liste_dates_company[0]
        liste_files_date_company = os.listdir(base_path / pl(company) / finance_date)

        for file in liste_files_date_company: 
                f = file.replace(".csv", "")
                try:
                    inputs[f] = pd.read_csv(base_path / pl(company) / pl(finance_date) / file)
                except Exception as e: 
                    print(f"ERROR LOAD DATA : {company} / {f} / {e}")
                    pass

        # list of analysis
        try: 
            results_analysis[company] = people_analysis(inputs, results_analysis[company])
        except Exception:
            pass

        try: 
            results_analysis[company] = income_state_annual(inputs, results_analysis[company], specific)
        except Exception:
            pass

        try: 
            results_analysis[company] = balance_sheet_annual(inputs, results_analysis[company], specific)
        except Exception:
            pass

    return results_analysis