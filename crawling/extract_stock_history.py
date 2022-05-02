import os
import yfinance as yf
from yahoo_fin.stock_info import *
from datetime import datetime
from pathlib import Path as pl
import tqdm
from utils.general_cleaning import check_folder

# Fetch the data
def main_extract_stock(data, sub_loc, split_size = 25):

    today = datetime.today().strftime("%d-%m-%Y")
    data = data.loc[~data["YAHOO_CODE"].isnull()]
    to_extract = data["YAHOO_CODE"].tolist()

    mapping_yahoo_reuters = data[["REUTERS_CODE", "YAHOO_CODE"]]
    mapping_yahoo_reuters.index = data["YAHOO_CODE"].tolist()
    mapping_yahoo_reuters = mapping_yahoo_reuters["REUTERS_CODE"].to_dict()

    stocks = {}
    missing_ticks = []

    # extract currency history 
    extract_currencies(sub_loc)

    # extract commodities 
    extract_commodities(sub_loc)

    # extract stocks history 
    while len(to_extract)>0:
        splits = int(len(to_extract)/split_size)
        print(f"TO EXTRACT FROM STOCK YAHOO: {len(to_extract)}")
        stocks, to_extract = extract_stocks(stocks, to_extract, splits = splits)

    # save stocks history 
    for company in tqdm.tqdm(stocks.keys()):
        listdir = os.listdir(sub_loc / pl(f"{mapping_yahoo_reuters[company]}"))

        if len(listdir) == 0: 
            check_folder(sub_loc / pl(f"{mapping_yahoo_reuters[company]}"))
            check_folder(sub_loc / pl(f"{mapping_yahoo_reuters[company]}/{today}"))
            missing_ticks.append(mapping_yahoo_reuters[company])
        else:
            max_date = max(listdir)

            stocks[company].to_csv(sub_loc / pl(f"{mapping_yahoo_reuters[company]}/{max_date}/STOCK.csv"))

            if stocks[company].shape[0] == 0:
                missing_ticks.append(mapping_yahoo_reuters[company])

    print(f"NO STOCK FOUND FOR {missing_ticks}")

    return missing_ticks
        

def extract_stocks(stocks, to_extract, splits = 5):

    extracts = {}
    missing_stocks = []

    if splits > 0:
        size_splits = int(len(to_extract)/splits)
    else: 
        extract = yf.download(to_extract, '1980-1-1')
        for company in to_extract:
            stocks, missing_stocks = extract_columns_per_stock(stocks, extract, company, missing_stocks)

    for i in range(splits):
        try:
            if i != splits:
                extracts[i] = yf.download(to_extract[i*size_splits:(i+1)*size_splits], '1980-1-1')
            else:
                extracts[i] = yf.download(to_extract[i*size_splits:], '1980-1-1')
        except Exception:
            pass
   
    for i in range(splits):
        extract = extracts[i]
        if i != splits:
            for company in to_extract[i*size_splits:(i+1)*size_splits]:
                stocks, missing_stocks = extract_columns_per_stock(stocks, extract, company, missing_stocks)
        else:
            for company in to_extract[i*size_splits:]:
                stocks, missing_stocks = extract_columns_per_stock(stocks, extract, company, missing_stocks)

    return stocks, missing_stocks


def extract_columns_per_stock(stocks, extract, company, missing_stocks):

    keep_cols = []
    features = ['Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']
    for col in features:
        keep_cols.append((col, company))
    sub_extract = extract[keep_cols]
    if sub_extract.shape[1] == len(features):
        sub_extract.columns = features
        sub_extract = sub_extract.loc[~sub_extract["Close"].isnull()]
        stocks[company] = sub_extract
    else:
        missing_stocks.append(company)

    return stocks, missing_stocks


def extract_currencies(sub_loc):

    # extract currency pairs 
    check_folder(sub_loc / pl("currencies"))
    for paires in ["EURUSD", "GBPUSD", "CHFUSD", "SEKUSD", "DKKUSD", \
                    "PLNUSD", "NOKUSD", "KRWUSD", "TWDUSD", "HKDUSD",
                    "AUDUSD", "CNYUSD", "JPYUSD"]:
        x_usd = yf.Ticker(f"{paires}=X")
        hist = x_usd.history(period="max")
        hist = hist[["Open", "Close"]].reset_index()
        hist.to_csv(sub_loc  / pl(f"currencies/{paires.replace('USD','')}.csv"), index=False)


def extract_commodities(sub_loc):
    check_folder(sub_loc / pl("commodities"))

    mapping_index = {"SB=F" : "sugar", "OJ=F" : "orange juice", "CT=F" : "cotton",
                    "KC=F" : "coffee", "CC=F" : "cacao", "LE=F" : "cattle", "HE=F" : "hog",
                    "GF=F" : "cattle feed", "ZS=F" : "soy bean", "ZL=F" : "soybean oil",
                    "ZM=F" : "soybean meal", "ZR=F" : "rice", "KE=F" : "wheat", 
                    "ZO=F" : "oat", "ZC=F" : "corn", "B0=F" : "propane", "BZ=F" : "brent",
                    "RB=F" : "gasoline", "NG=F" : "gas", "HO=F" : "heating oil", 
                    "CL=F" : "crude oil", "PA=F" : "palladium", "HG=F" : "copper",
                    "PL=F" : "platinium", "SIL=F" : "micro silver", "SI=F" : "silver",
                    "GC=F" : "gold", "ZT=F" : "2Y US bond", "ZF=F" : "5Y US bond",
                    "ZN=F" : "10Y US bond", "NQ=F" : "nasdaq 100", "ES=F" : "s&p500"}

    for future, value in mapping_index.items():
        x_usd = yf.Ticker(future)
        hist = x_usd.history(period="max")
        hist = hist[["Open", "Close"]].reset_index()
        hist.to_csv(sub_loc  / pl(f"commodities/{value}.csv"), index=False)

    #add vix
    vix = yf.download("^VIX", '1980-1-1')
    vix = vix[["Open", "Close"]].reset_index()
    vix.to_csv(sub_loc  / pl(f"commodities/vix.csv"), index=False)