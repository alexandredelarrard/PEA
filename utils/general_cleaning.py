import pandas as pd 
import re
import numpy as np 
from pathlib import Path as pl
from utils.general_functions import smart_column_parser
from sklearn.linear_model import LinearRegression
import warnings
import os 
import statsmodels.api as sm
warnings.filterwarnings("ignore")

def create_index(data):

    # create index 
    data = data.loc[~data["Unnamed: 0"].isnull()]
    data["Unnamed: 0"] = data["Unnamed: 0"].apply(lambda x : smart_column_parser([x])[0])
    data.index = data["Unnamed: 0"].values
    del data["Unnamed: 0"]

    return data 


def deduce_specific(data):

    if "INTEREST_INCOME_BANK" in data.index or \
        "NET_LOANS" in data.index:
        return "bank"
    
    if "INSURANCE_RECEIVABLES" in data.index or \
        "TOTAL_PREMIUMS_EARNED" in data.index:
        return "insur"

    return ""


def handle_accountancy_numbers(data):
    # create index 
    for col in data.columns:
        data[col] = data[col].apply(lambda x : re.sub('^\((.*?)$', r'-\1', str(x).split(".")[0].replace(",","")))
        data[col] = np.where(data[col] == "--", np.nan, data[col])
        data[col] = pd.to_numeric(data[col], errors='coerce')
    return data 


def check_folder(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def deduce_currency(x):

    if".N" in x:
        return "USD"
    
    if ".PA" in x or ".DE" in x or ".MI" in x \
         or ".MA" in x or ".BR" in x or ".VI" in x \
         or ".LU" in x or ".AS" in x or ".BE" in x \
         or ".MU" in x or ".LS" in x or ".I" in x \
         or".HE" in x or ".MC" in x :
        return 'EUR'
    
    if ".L" in x: # uk
        return "GBP"

    if ".ZU" in x or ".S" in x: # suisse
        return "CHF"

    if ".ST" in x: # suede 
        return "SEK"

    if ".CO" in x: # denmark 
        return "DKK"

    if ".WA" in x: # poland 
        return "PLN"

    if ".OL" in x: # norvege 
        return "NOK"

    if ".TW" in x: # norvege 
        return "TWD"

    if ".HK" in x: # norvege 
        return "HKD"

    if ".KS" in x: # norvege 
        return "KRW"

    if ".AX" in x: # norvege 
        return "AUD"


def handle_currency(params, data):
    """
    convert data to usd currency based 
    """
    currency_map = {}

    # read currency conv 
    if params["currency"] == "USD":
        return data 
    else:
        currency = pd.read_csv(params["base_path"] / pl("currencies") / pl(f"{params['currency']}.csv"))
        currency["Date"] = pd.to_datetime(currency["Date"])
        currency["Month"] = currency["Date"].dt.to_period('M')
        currency = currency[["Month", "Close"]].groupby("Month").mean().reset_index()

        for col in data.columns:
            col_m = pd.to_datetime(col).to_period('M')
            currency_map[col] = currency.loc[currency["Month"] == col_m, "Close"].values[0]

        for col in data.columns:
            data[col] = data[col] * currency_map[col]
        
        return data


def deduce_trend(df):

    diff = pd.DataFrame((df -  df.shift(-1)) /  df.shift(-1)).reset_index()
    diff.columns = ["INDEX", "TARGET"]
    diff["WEIGHTS"] = 1/ np.sqrt(diff.index)

    diff["TARGET"] = np.where(abs(diff["TARGET"]) == np.inf, np.nan, diff["TARGET"])

    if diff.loc[0,"TARGET"] == 0 :
        diff.loc[0, "WEIGHTS"] = 0
    else:
        diff.loc[0, "WEIGHTS"] = 0.5

    diff["WEIGHTS"] = np.where(diff["TARGET"].isnull(), np.nan, diff["WEIGHTS"])

    return (diff["TARGET"]*diff["WEIGHTS"]).sum()/diff["WEIGHTS"].sum()
