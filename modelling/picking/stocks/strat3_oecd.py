# https://data.oecd.org/price/inflation-cpi.htm
# https://data.oecd.org/interest/long-term-interest-rates.htm#indicator-chart

from re import A
import pandas as pd 
import os
import tqdm
from pathlib import Path as pl

from data_prep.sbf120.stock import stock_processing
from data_prep.sbf120.main_data_prep import read_files

def load_macroeconomy_data(base_path):

    macro_data = {}

    macro_data["INFLATION"] = pd.read_csv(base_path / pl("data/external/inflation_rates_since_1980.csv"), sep=",")
    macro_data["LT_RATES"] = pd.read_csv(base_path / pl("data/external/long_term_rates_since_1980.csv"), sep=",")
    macro_data["ST_RATES"] = pd.read_csv(base_path / pl("data/external/short_term_rates_since_1980.csv"), sep=",")
    macro_data["PPI_RATES"] = pd.read_csv(base_path / pl("data/external/producer_price_index_since_1980.csv"), sep=",")
    macro_data["UNEMPLOYMENT_RATES"] = pd.read_csv(base_path / pl("data/external/unemployment_rate_since_1980.csv"), sep=",")
    
    #commodities 
    macro_data["GOLD"] = pd.read_csv(base_path / pl("data/extracted_data/commodities/gold.csv"), sep=",")
    macro_data["SILVER"] = pd.read_csv(base_path / pl("data/extracted_data/commodities/silver.csv"), sep=",")
    macro_data["s&p500"] = pd.read_csv(base_path / pl("data/extracted_data/commodities/s&p500.csv"), sep=",")
    macro_data["WHEAT"] = pd.read_csv(base_path / pl("data/extracted_data/commodities/wheat.csv"), sep=",")
    macro_data["BRENT"] = pd.read_csv(base_path / pl("data/extracted_data/commodities/brent.csv"), sep=",")
    macro_data["GAS"] = pd.read_csv(base_path / pl("data/extracted_data/commodities/gas.csv"), sep=",")
    macro_data["SUGAR"] = pd.read_csv(base_path / pl("data/extracted_data/commodities/sugar.csv"), sep=",")
    macro_data["COPPER"] = pd.read_csv(base_path / pl("data/extracted_data/commodities/copper.csv"), sep=",")
    macro_data["COFFEE"] = pd.read_csv(base_path / pl("data/extracted_data/commodities/coffee.csv"), sep=",")
    macro_data["CACAO"] = pd.read_csv(base_path / pl("data/extracted_data/commodities/cacao.csv"), sep=",")
    macro_data["COTTON"] = pd.read_csv(base_path / pl("data/extracted_data/commodities/cotton.csv"), sep=",")
    macro_data["VIX"] = pd.read_csv(base_path / pl("data/extracted_data/commodities/vix.csv"), sep=",")

    for indic in macro_data.keys():
        if len(set(["INDICATOR", "SUBJECT", "FREQUENCY", "Flag Codes", "MEASURE"]).intersection(macro_data[indic].columns)) == 5: 
            macro_data[indic] = macro_data[indic].drop(["INDICATOR", "SUBJECT", "FREQUENCY", "Flag Codes", "MEASURE"], axis=1)
            macro_data[indic]["TIME"] = pd.to_datetime(macro_data[indic]["TIME"], format="%Y-%m").dt.to_period("M")

    return macro_data


def leverage_oecd_data(base_path, data, df):

    macro_data = load_macroeconomy_data(base_path)

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
    inflation = macro_data["INFLATION"]
    rates= macro_data["LT_RATES"]

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