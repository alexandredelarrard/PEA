import pandas as pd 
import re
import numpy as np 
from utils.general_functions import smart_column_parser
import warnings
warnings.filterwarnings("ignore")

def create_index(data, specific):
     # create index 
    data = data.loc[~data["Unnamed: 0"].isnull()]
    data["Unnamed: 0"] = data["Unnamed: 0"].apply(lambda x : smart_column_parser([x])[0])
    data.index = data["Unnamed: 0"].values
    del data["Unnamed: 0"]

    if specific == "bank":
        data.loc["TOTAL_REVENUE", :] = 0
        data.loc["TOTAL_OPERATING_EXPENSE", :] = 0

    if specific in ["bank", "insur"]:
        data.loc["CASH_AND_SHORT_TERM_INVESTMENTS", :] = 0

    return data 

def handle_accountancy_numbers(data):
    # create index 
    for col in data.columns:
        data[col] = data[col].apply(lambda x : re.sub('^\((.*?)$', r'-\1', str(x).split(".")[0].replace(",","")))
        data[col] = np.where(data[col] == "--", np.nan, data[col])
        data[col] = pd.to_numeric(data[col], errors='coerce')
    return data 