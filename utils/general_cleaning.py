import pandas as pd 
import re
import numpy as np 
from pathlib import Path as pl
from utils.general_functions import smart_column_parser
from datetime import datetime
import warnings
import os 
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
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

    if ".ST" in x: # suede 
        return "SEK"

    if ".ZU" in x or ".S" in x: # suisse
        return "CHF"
   
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

    if ".T" in x: # japan 
        return "JPY"


def handle_currency(params, data, which_currency):
    """
    convert data to usd currency based 
    """
    currency_map = {}
    currency = load_currency(params, which_currency)
    quarters = list(data.columns)
    today = datetime.today().strftime("%d-%m-%Y")

    # read currency conv 
    if params["currency"] == "USD":
        return data

    else:
        for col in [today] + quarters:
            if col == today:
                col_m = pd.to_datetime(col, format="%d-%m-%Y").to_period('M')
            else:
                col_m = pd.to_datetime(col).to_period('M')
            currency_map[col] = currency.loc[currency["Month"] == col_m, "Close"].values[0]
            
        for col in quarters:
            data[col] = data[col] * currency_map[col]
        
        return data


def load_currency(params, which_currency = ""):
    if params[which_currency] != "USD":
        currency = pd.read_csv(params["base_path"] / pl("currencies") / pl(f"{params[which_currency]}.csv"))
        currency["Date"] = pd.to_datetime(currency["Date"])
        currency["Month"] = currency["Date"].dt.to_period('M')
        currency = currency[["Month", "Close"]].groupby("Month").mean().reset_index()
    else:
        currency = pd.read_csv(params["base_path"] / pl("currencies") / pl("EUR.csv"))
        currency["Date"] = pd.to_datetime(currency["Date"])
        currency["Month"] = currency["Date"].dt.to_period('M')
        currency = currency[["Month", "Close"]].groupby("Month").mean().reset_index()
        currency["Close"] = 1

    return currency


def deduce_trend(df):

    diff = pd.DataFrame(df).reset_index()
    diff.columns = ["INDEX", "VALUE"]
    diff["WEIGHTS"] = 1/ np.sqrt(diff.index +1)
    diff = diff.loc[~diff["VALUE"].isnull()]
    diff = diff.loc[~diff["VALUE"].isin([np.inf, -np.inf])]

    if diff.shape[0] == 0:
        return np.nan

    good= False
    i = 1
    while i < diff.shape[0]:
        if not good: 
            if diff["VALUE"].values[-i] != 0:
                diff["VALUE"] = diff["VALUE"] / diff["VALUE"].values[-i]
                good= True
            else: 
                i +=1
        else: 
            break
    
    if i == diff.shape[0]:
        return np.nan

    # in case TTM not well calculated and similar to Y-1
    if diff.iloc[0, 1] == diff.iloc[1,1] :
        diff.iloc[0, 2] = 0
    else:
        diff.iloc[0, 2] = 1

    # to weight ; , diff["WEIGHTS"]
    reg = LinearRegression(normalize=True).fit(np.array(diff.index[::-1]).reshape(-1, 1), diff["VALUE"])
    coeff = reg.coef_[0]

    return coeff*100

 