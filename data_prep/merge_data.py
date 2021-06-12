import pandas as pd 
import numpy as np 
from pathlib import Path as pl
import glob 
import os 


def load_dividends(path):
    dividends = pd.read_csv(path).reset_index(drop=True)
    dividends["RECORDED_DATE"] = pd.to_datetime(dividends["RECORDED_DATE"], format="%d %B %Y")
    dividends["AMOUNT"] = dividends["AMOUNT"].apply(lambda x : x.replace("EUR", "").replace("USD", "")).astype(float)

    return pd.DataFrame([{"LAST_DIVIDEND_DATE": dividends.loc[0, "RECORDED_DATE"], "LAST_DIVIDEND_AMOUNT" : dividends.loc[0, "AMOUNT"]}])


def merge_reuters_data(configs, datas):

    base_path = pl(configs["resources"]["base_path"])
    savepath = base_path / pl("data/extracted_data/reuters")

    full_data = datas["mapping_reuters"]

    extracted_info = {}
    for company_code in os.listdir(savepath):
        liste_dates = os.listdir(savepath /pl(company_code))
        dividends = load_dividends(savepath / pl(company_code) / pl(max(liste_dates)) / pl("DIVIDENDS.csv"))
        



