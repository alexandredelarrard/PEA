import pandas as pd
import os
import yfinance as yf
from datetime import datetime
from pathlib import Path as pl
from utils.general_cleaning import check_folder

# Fetch the data
def main_extract_stock(data_mapping_yahoo, sub_loc):

    today = datetime.today().strftime("%Y-%m-%d")
    data_mapping_yahoo = data_mapping_yahoo.loc[~data_mapping_yahoo["YAHOO_CODE"].isnull()]

    for tick in data_mapping_yahoo[["REUTERS_CODE", "YAHOO_CODE"]].values:
        if not os.path.isfile(sub_loc / pl(f"{tick[0]}/{today}/STOCK.csv")):
            tick = tuple(tick)
            extract = yf.download(tick[1], '2010-1-1')
            check_folder(sub_loc / pl(f"{tick[0]}"))
            check_folder(sub_loc / pl(f"{tick[0]}/{today}"))
            extract.to_csv(sub_loc / pl(f"{tick[0]}/{today}/STOCK.csv"))

            if extract.shape[0] == 0:
                print(f"NO STOCK FOUND FOR {tick}")
        
    # extract currency pairs 
    check_folder(sub_loc / pl("currencies"))
    for paires in ["EURUSD", "GBPUSD", "CHFUSD", "SEKUSD", "DKKUSD", \
                    "PLNUSD", "NOKUSD", "KRWUSD", "TWDUSD", "HKDUSD",
                    "AUDUSD"]:
        x_usd = yf.Ticker(f"{paires}=X")
        hist = x_usd.history(period="max")
        hist = hist[["Open", "Close"]].reset_index()
        hist.to_csv(sub_loc  / pl(f"currencies/{paires.replace('USD','')}.csv"), index=False)

# def main_extract_stock_filling(data_mapping_yahoo, sub_loc):

#     today = datetime.today().strftime("%Y-%m-%d")
#     data_mapping_yahoo = pd.read_csv(r"C:\Users\de larrard alexandre\Documents\repos_github\PEA\data\data_for_crawling\mapping_asia_yahoo.csv", sep=";")

#     for i, tick in enumerate(data_mapping_yahoo[["REUTERS_CODE", "YAHOO_CODE"]].values):
#         # if sum(data_mapping_yahoo.loc[data_mapping_yahoo["REUTERS_CODE"] == tick[0], "YAHOO_CODE"].isnull()) > 0:
#         tick = tuple(tick)
#         extract = yf.download(tick[0], '2010-1-1')
#             if extract.shape[0] > 0:
#                 data_mapping_yahoo.loc[data_mapping_yahoo["REUTERS_CODE"] == tick[0], "YAHOO_CODE"] = tick[0]
#                 print(tick[0], tick[0])

#     data_mapping_yahoo.to_csv(r"C:\Users\de larrard alexandre\Documents\repos_github\PEA\data\data_for_crawling\mapping_europe_yahoo_V2.csv", sep=";", index=False)