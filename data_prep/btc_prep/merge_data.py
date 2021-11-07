import pandas as pd 
import numpy as np 
from pathlib import Path as pl
import glob 
import os 
from datetime import datetime 


def load_dividends(path):
    """TODO: homogeneiser tous les dividendes par années 
    mettre la vairable combien de versement de div par an

    Args:
        path ([type]): [description]

    Returns:
        [type]: [description]
    """

    try:
        dividends = pd.read_csv(path).reset_index(drop=True)
    except Exception:
        return pd.DataFrame([{"LAST_DIVIDEND_DATE": np.nan, "LAST_DIVIDEND_AMOUNT": 0, "PCTG_DIFF_TO_AVG_DIV_AMOUNT": 0, "REGULARITY_DIVIDENDS" : 0, "NBR_DIV_PER_YEAR": 0}])

    dividends = dividends.loc[dividends["AMOUNT"] != "--"]

    if dividends.shape[0]>0:
        dividends["RECORDED_DATE"] = pd.to_datetime(dividends["RECORDED_DATE"], format="%d %b %Y")

        # TODO: check currency
        dividends["AMOUNT"] = dividends["AMOUNT"].apply(lambda x : str(x).replace("EUR", "").replace("USD", "").replace("AUD", "").replace("BRL", "")).astype(float)

        #get latest div
        last_div_date = dividends.loc[0, "RECORDED_DATE"]
        last_div_amount = dividends.loc[0, "AMOUNT"]

        # compared to usual 
        mean_div = dividends["AMOUNT"].mean()
        diff_m_div = (last_div_amount - mean_div)*100/mean_div

        # frequence de div
        years = dividends["RECORDED_DATE"].dt.year.unique()
        frequency_div = len(years)/3
        nbr_per_year = pd.Series(dividends["RECORDED_DATE"].dt.year).value_counts().mean()

        return pd.DataFrame([{"LAST_DIVIDEND_DATE": last_div_date, "LAST_DIVIDEND_AMOUNT": last_div_amount, "PCTG_DIFF_TO_AVG_DIV_AMOUNT": diff_m_div, "REGULARITY_DIVIDENDS" : frequency_div, "NBR_DIV_PER_YEAR": nbr_per_year}])
    else:
        return pd.DataFrame([{"LAST_DIVIDEND_DATE": np.nan, "LAST_DIVIDEND_AMOUNT": 0, "PCTG_DIFF_TO_AVG_DIV_AMOUNT": 0, "REGULARITY_DIVIDENDS" : 0, "NBR_DIV_PER_YEAR": 0}])


def merge_reuters_data(configs, datas):

    base_path = pl(configs["resources"]["base_path"])
    savepath = base_path / pl("data/extracted_data/reuters")

    full_data = datas["mapping_reuters"]

    extracted_info = pd.DataFrame()
    for company_code in os.listdir(savepath):
        liste_dates = os.listdir(savepath /pl(company_code))
        dividends = load_dividends(savepath / pl(company_code) / pl(max(liste_dates)) / pl("DIVIDENDS.csv"))
        dividends["CODE"] = company_code
        extracted_info = pd.concat([extracted_info, dividends], axis=0)

    full_data = pd.merge(full_data, extracted_info, on="CODE", how="left")



