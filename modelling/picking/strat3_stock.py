# https://data.oecd.org/price/inflation-cpi.htm

from re import A
import pandas as pd 
import os
import tqdm
from pathlib import Path as pl

from data_prep.sbf120.stock import stock_processing
from data_prep.sbf120.main_data_prep import read_files

def load_oecd_data(base_path):

    inflation = pd.read_csv(base_path / pl("data/external/inflation_history_oecd.csv"), sep=";")
    rates = pd.read_csv(base_path / pl("data/external/interest_rates_oecd.csv"), sep=";")

    inflation = inflation[["LOCATION", "TIME", "Value"]]
    rates = rates[["LOCATION", "TIME", "Value"]]
    
    rates["TIME"] = pd.to_datetime(rates["TIME"]).dt.to_period("M")
    inflation["TIME"] = pd.to_datetime(inflation["TIME"]).dt.to_period("M")

    return inflation, rates


def leverage_oecd_data(base_path, data, df):

    inflation, rates = load_oecd_data(base_path)

    mapping_location = {"IT": "ITA", "GB": "GBR", "DE": "DEU", 
                        "BE": "BEL", "CH" : "CHN", 
                        "US": "USA", "FR" : "FRA",
                        "CHF" : "CHE", "ES" : "ESP",
                        "LU" : "LUX", "PT" : "ESP", "PL" : "POL", 
                        "IE" : "IRL", "TW" : "CHN", "AT" : "AUT",
                        "KS" : "KOR", "NO" : "NOR", "FI" : "FIN",
                        "DK" : "DNK", "NL" : "NLD", "AUS" : "DEU",
                        "JPY": "CHN",
                        "SE" : "SWE"}

    file_path = base_path / pl("data/extracted_data")
    results_analysis = {}

    for company in tqdm.tqdm(data["REUTERS_CODE"].unique()):

        results_analysis[company] = {}
        params = {"specific" : "",
                  "company" : company,
                  "base_path" : file_path}
        
        try:
            inputs = read_files(params)
            _, stock = stock_processing(inputs)
            stock["DATE"] = stock["DATE"].dt.to_period("M")
            agg_stock = stock[["DATE", "CLOSE", "VOLUME"]].groupby("DATE").mean().reset_index()

            country = data.loc[data["REUTERS_CODE"] == company, "Country"].values[0]
            infl = inflation.loc[inflation["LOCATION"] == mapping_location[country]]
            rte = rates.loc[rates["LOCATION"] == mapping_location[country]]

            comparison_df= pd.merge(agg_stock, infl, left_on="DATE", right_on="TIME", how="inner")
            comparison_df= pd.merge(comparison_df, rte, left_on="DATE", right_on="TIME", how="left", suffixes=("_INFLATION", "_RATE"))
            
            comparison_df = comparison_df[["DATE", "CLOSE", "VOLUME", "Value_INFLATION", "Value_RATE"]]
            comparison_df = comparison_df.loc[~comparison_df["CLOSE"].isnull()]
            results_analysis[company] = {"INFLATION" : comparison_df[["CLOSE", "Value_INFLATION"]].corr().iloc[0,1],
                                        "RATE" : comparison_df[["CLOSE", "Value_RATE"]].corr().iloc[0,1]}

        except Exception:
            pass 

    results = pd.DataFrame(results_analysis).sort_index().T

    # results = pd.merge(results, data[["REUTERS_CODE", "SECTOR", "SUB INDUSTRY", "NAME"]], left_index=True, right_on="REUTERS_CODE", how="left")
    # agg = results[["SUB INDUSTRY", "INFLATION", "RATE"]].groupby("SUB INDUSTRY").mean().sort_values("INFLATION")

    df = pd.merge(df, results, left_index=True, right_index=True, how="left", validate="1:1")

    return df