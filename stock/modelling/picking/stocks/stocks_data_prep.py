import os
from attr import validate 
import pandas as pd 
from pathlib import Path as pl
import tqdm
import numpy as np
from datetime import datetime, timedelta
from data_prep.sbf120.main_data_prep import read_files


def ajouter_date_prediction(df, k_days=5):

    df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")
    max_date = df["Date"].max()
    to_predict = pd.DataFrame([max_date + timedelta(days=x) for x in range(1, k_days+3)], columns=["Date"])

    # remove samedi dimanche
    to_predict["DAY"] = to_predict["Date"].dt.dayofweek
    to_predict = to_predict.loc[~(to_predict["DAY"] >=5)]

    df = pd.concat([df, to_predict], axis=0)
    df["Close"] = df["Close"].fillna(-9999)
    df = df.reset_index(drop=True)

    del df["DAY"]

    return df


def distance_to_mean(df, commo, k_days = 5, distance_time=5, week=1):

    cols = [f"{commo}_{i}D" for i in range(1 + distance_time, distance_time + 5 + 1)]
    df[f"{commo}_0D_{week}W_STD"] = df[cols].std(axis=1) 

    mean_stock = df[cols].mean(axis=1)
    df[f"DISTANCE_{commo}_0D_TO_{week}W_MEAN"] = (df[f"{commo}_{k_days}D"] - mean_stock) / mean_stock

    return df


def kpis_past(df, var, commo, k_days=5):

    for i in range(0, k_days + 381):
        df[f"{commo}_{i}D"] = df[var].shift(i)

    # feature engineering 
    df = df.loc[~df[f"{commo}_{k_days + 380}D"].isnull()]
    df[f"TARGET_{commo}_+{k_days}D"] = ((df[var] - df[f"{commo}_{k_days}D"]) / df[f"{commo}_{k_days}D"])

    #1W
    df = distance_to_mean(df, commo, k_days = k_days, distance_time=5, week=1)

    #2W
    df = distance_to_mean(df, commo, k_days = k_days, distance_time=10, week=2)

    #1M
    df = distance_to_mean(df, commo, k_days = k_days, distance_time=20, week=4)

    #2M
    df = distance_to_mean(df, commo, k_days = k_days, distance_time=40, week=8)

    # Delta 1Y 
    cols_1y = [f"{commo}_{k_days +i}D" for i in range(350, 380)]
    df[f"{commo}_DELTA_1Y_TREND"] = (df[f"{commo}_{k_days}D"] - df[cols_1y].mean(axis=1)) / df[cols_1y].mean(axis=1)

    df = df.rename(columns={var: commo})

    keep_cols = [f"TARGET_{commo}_+{k_days}D", "Date", commo,
                f"{commo}_0D_1W_STD", 
                f"{commo}_0D_2W_STD", 
                f"{commo}_0D_4W_STD", 
                f"{commo}_0D_8W_STD", 
                f"DISTANCE_{commo}_0D_TO_1W_MEAN",
                f"DISTANCE_{commo}_0D_TO_2W_MEAN",
                f"DISTANCE_{commo}_0D_TO_4W_MEAN",
                f"DISTANCE_{commo}_0D_TO_8W_MEAN",
                f"{commo}_DELTA_1Y_TREND"]

    df = df[keep_cols]
    return df


def stock_feature_engineering(df):

    stock = kpis_past(df, "Close", "STOCK", k_days=5)
    volume = kpis_past(df, "Volume", "VOLUME", k_days=5)

    df= df[["Date"]]
    df = df.merge(stock, on="Date", how="left", validate="1:1")
    df = df.merge(volume.drop(["TARGET_VOLUME_+5D"], axis=1), on="Date", how="left", validate="1:1")

    df = df.loc[~df["DISTANCE_STOCK_0D_TO_8W_MEAN"].isnull()]

    for col in ["DISTANCE_VOLUME_0D_TO_2W_MEAN", "DISTANCE_VOLUME_0D_TO_4W_MEAN", "DISTANCE_VOLUME_0D_TO_8W_MEAN"]:
        df[col] = df[col].ffill().bfill()

    for col in ["DISTANCE_VOLUME_0D_TO_2W_MEAN", "DISTANCE_VOLUME_0D_TO_4W_MEAN", "DISTANCE_VOLUME_0D_TO_8W_MEAN"]:
        df.loc[df[col].isin([np.inf, -np.inf]), col] = 0

    for col in df.columns:
        if col not in ["Date"]:
            df[col] = np.where(df[col].isin([np.inf, -np.inf]), 0, df[col])

    if df.shape[0] > 0:
        df["MONTH_DAY"] = df["Date"].dt.day
        df["WEEK_DAY"] = df["Date"].dt.dayofweek
        df["MONTH"] = df["Date"].dt.month
        df["WEEK"] = df["Date"].dt.week
        # df["WEEK_PERIODE"] = df["Date"].dt.to_period('W')
        # df["WEEK_PERIODE"] = list(zip(*df["WEEK_PERIODE"].astype(str).str.split("/")))[0]

    return df


def encode_features(stocks):

    # final data prep 
    stocks["WEEK_DAY"] = stocks["WEEK_DAY"].astype("category")
    stocks["WEEK"] = stocks["WEEK"].astype("category")
    stocks["SECTOR"] = stocks["SECTOR"].astype("category")
    stocks["COUNTRY"] = stocks["Country"].astype("category")
    
    today = datetime.today()
    stocks["TIME_SINCE_TODAY"] = (today - stocks["Date"]).dt.days/365.25
    stocks["WEIGHT"] = 1/(1 + np.exp(-2/(1 + stocks["TIME_SINCE_TODAY"])))

    return stocks


def neutral_market_sector(stocks):

    # full market delta 
    agg_market = stocks[["Date", "TARGET_STOCK_+5D", 
                    "STOCK_DELTA_1Y_TREND",
                    "STOCK_0D_1W_STD", 
                    "STOCK_0D_2W_STD", 
                    "DISTANCE_VOLUME_0D_TO_1W_MEAN", 
                    "DISTANCE_VOLUME_0D_TO_2W_MEAN", 
                    "DISTANCE_VOLUME_0D_TO_4W_MEAN", 
                    "DISTANCE_STOCK_0D_TO_1W_MEAN",
                    "DISTANCE_STOCK_0D_TO_2W_MEAN",
                    "DISTANCE_STOCK_0D_TO_4W_MEAN",
                    "DISTANCE_STOCK_0D_TO_8W_MEAN"]].groupby("Date").median().reset_index()
    
    agg_market.rename(columns={"TARGET_STOCK_+5D" : "MARKET_TARGET_STOCK_+5D",
                            "STOCK_0D_1W_STD" : "MARKET_STOCK_0D_1W_STD", 
                            "STOCK_0D_2W_STD" : "MARKET_STOCK_0D_2W_STD",  
                            "DISTANCE_VOLUME_0D_TO_1W_MEAN" : "MARKET_DISTANCE_VOLUME_0D_TO_1W_MEAN", 
                            "DISTANCE_VOLUME_0D_TO_2W_MEAN" : "MARKET_DISTANCE_VOLUME_0D_TO_2W_MEAN",
                            "DISTANCE_VOLUME_0D_TO_4W_MEAN" : "MARKET_DISTANCE_VOLUME_0D_TO_4W_MEAN", 
                            "DISTANCE_STOCK_0D_TO_1W_MEAN" : "MARKET_DISTANCE_STOCK_0D_TO_1W_MEAN",
                            "DISTANCE_STOCK_0D_TO_2W_MEAN": "MARKET_DISTANCE_STOCK_0D_TO_2W_MEAN",
                            "DISTANCE_STOCK_0D_TO_4W_MEAN": "MARKET_DISTANCE_STOCK_0D_TO_4W_MEAN",
                            "DISTANCE_STOCK_0D_TO_8W_MEAN": "MARKET_DISTANCE_STOCK_0D_TO_8W_MEAN",
                            "STOCK_DELTA_1Y_TREND" : "MARKET_STOCK_DELTA_1Y_TREND"}, 
                    inplace=True)

    agg_sector = stocks[["Date", "SECTOR", "TARGET_STOCK_+5D",  
                    "STOCK_DELTA_1Y_TREND",
                    "STOCK_0D_1W_STD", 
                    "STOCK_0D_2W_STD", 
                    "DISTANCE_VOLUME_0D_TO_1W_MEAN", 
                    "DISTANCE_VOLUME_0D_TO_2W_MEAN", 
                    "DISTANCE_VOLUME_0D_TO_4W_MEAN", 
                    "DISTANCE_STOCK_0D_TO_1W_MEAN",
                    "DISTANCE_STOCK_0D_TO_2W_MEAN",
                    "DISTANCE_STOCK_0D_TO_4W_MEAN",
                    "DISTANCE_STOCK_0D_TO_8W_MEAN"]].groupby(["SECTOR", "Date"]).median().reset_index()
                    
    agg_sector.rename(columns={"TARGET_STOCK_+5D" : "SECTOR_TARGET_STOCK_+5D",
                            "STOCK_0D_1W_STD" : "SECTOR_STOCK_0D_1W_STD", 
                            "STOCK_0D_2W_STD" : "SECTOR_STOCK_0D_2W_STD",  
                            "DISTANCE_VOLUME_0D_TO_1W_MEAN" : "SECTOR_DISTANCE_VOLUME_0D_TO_1W_MEAN", 
                            "DISTANCE_VOLUME_0D_TO_2W_MEAN" : "SECTOR_DISTANCE_VOLUME_0D_TO_2W_MEAN",
                            "DISTANCE_VOLUME_0D_TO_4W_MEAN" : "SECTOR_DISTANCE_VOLUME_0D_TO_4W_MEAN",  
                            "DISTANCE_STOCK_0D_TO_1W_MEAN" : "SECTOR_DISTANCE_STOCK_0D_TO_1W_MEAN",
                            "DISTANCE_STOCK_0D_TO_2W_MEAN": "SECTOR_DISTANCE_STOCK_0D_TO_2W_MEAN",
                            "DISTANCE_STOCK_0D_TO_4W_MEAN": "SECTOR_DISTANCE_STOCK_0D_TO_4W_MEAN",
                            "DISTANCE_STOCK_0D_TO_8W_MEAN": "SECTOR_DISTANCE_STOCK_0D_TO_8W_MEAN",
                            "STOCK_DELTA_1Y_TREND" : "SECTOR_STOCK_DELTA_1Y_TREND"}, 
                    inplace=True)

    stocks = stocks.merge(agg_sector, on=["SECTOR", "Date"], how="left", validate="m:1")
    stocks = stocks.merge(agg_market, on=["Date"], how="left", validate="m:1")

    #################"" stock to sector
    stocks["TARGET_STOCK_+5D_TO_SECTOR"] = stocks["TARGET_STOCK_+5D"] -  stocks["SECTOR_TARGET_STOCK_+5D"]
    
    # features homogeneous to target
    for weeks in [1,2,4,8]:
        stocks[f"STOCK_DISTANCE_TO_SECTOR_{weeks}W_MEAN"] = stocks[f"DISTANCE_STOCK_0D_TO_{weeks}W_MEAN"] - stocks[f"SECTOR_DISTANCE_STOCK_0D_TO_{weeks}W_MEAN"] 

    stocks["STOCK_DELTA_1Y_TREND_TO_SECTOR"] = stocks["STOCK_DELTA_1Y_TREND"] - stocks["SECTOR_STOCK_DELTA_1Y_TREND"] 
    stocks["STOCK_1W_STD_RATIO_TO_SECTOR"] = stocks["STOCK_0D_1W_STD"] / (0.1+ stocks["SECTOR_STOCK_0D_1W_STD"])
    stocks["STOCK_2W_STD_RATIO_TO_SECTOR"] = stocks["STOCK_0D_2W_STD"] / (0.1+ stocks["SECTOR_STOCK_0D_2W_STD"])

    stocks["VOLUME_DISTANCE_TO_SECTOR_1W_MEAN_RATIO"] = stocks["DISTANCE_VOLUME_0D_TO_1W_MEAN"] / (0.1+ stocks["SECTOR_DISTANCE_VOLUME_0D_TO_1W_MEAN"])
    stocks["VOLUME_DISTANCE_TO_SECTOR_2W_MEAN_RATIO"] = stocks["DISTANCE_VOLUME_0D_TO_2W_MEAN"] / (0.1+ stocks["SECTOR_DISTANCE_VOLUME_0D_TO_2W_MEAN"])

    #################"" sector to market 
    stocks["TARGET_SECTOR_+5D_TO_MARKET"] = stocks["SECTOR_TARGET_STOCK_+5D"] - stocks["MARKET_TARGET_STOCK_+5D"]
    for weeks in [1,2,4,8]:
        stocks[f"SECTOR_STOCK_DISTANCE_TO_MARKET_{weeks}W_MEAN"] = stocks[f"SECTOR_DISTANCE_STOCK_0D_TO_{weeks}W_MEAN"] - stocks[f"MARKET_DISTANCE_STOCK_0D_TO_{weeks}W_MEAN"] 

    stocks["SECTOR_STOCK_DELTA_1Y_TREND_TO_MARKET"] = stocks["SECTOR_STOCK_DELTA_1Y_TREND"] - stocks["MARKET_STOCK_DELTA_1Y_TREND"]
    stocks["SECTOR_STOCK_1W_STD_RATIO_TO_MARKET"] = stocks["SECTOR_STOCK_0D_1W_STD"] / (0.1+ stocks["MARKET_STOCK_0D_1W_STD"])
    stocks["SECTOR_STOCK_2W_STD_RATIO_TO_MARKET"] = stocks["SECTOR_STOCK_0D_2W_STD"] / (0.1+ stocks["MARKET_STOCK_0D_1W_STD"])

    stocks["SECTOR_VOLUME_DISTANCE_TO_MARKET_1W_MEAN_RATIO"] = stocks["SECTOR_DISTANCE_VOLUME_0D_TO_1W_MEAN"] / (0.1+ stocks["MARKET_DISTANCE_VOLUME_0D_TO_1W_MEAN"])
    stocks["SECTOR_VOLUME_DISTANCE_TO_MARKET_2W_MEAN_RATIO"] = stocks["SECTOR_DISTANCE_VOLUME_0D_TO_2W_MEAN"] / (0.1+ stocks["MARKET_DISTANCE_VOLUME_0D_TO_2W_MEAN"])

    return stocks


def fill_missing_stock_dates(df):

    df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")

    min_date = df["Date"].min()
    max_date = df["Date"].max()

    days_fill = pd.DataFrame(pd.date_range(min_date, max_date-timedelta(days=1), freq='d'), columns=["Date"])
    days_fill["DAY"] = days_fill["Date"].dt.dayofweek
    days_fill = days_fill.loc[days_fill["DAY"].isin([0,1,2,3,4])]
    del days_fill["DAY"]

    df = df.merge(days_fill, on="Date", how="right", validate="1:1")
    df["Close"] = df["Close"].ffill().bfill()

    return df


def como_corr(df, full_commos, rolling):

    correlation_como_cols = list(set(full_commos.columns) - set(["Date"]))

    full_commos["Date"] = pd.to_datetime(full_commos["Date"], format="%Y-%m-%d")
    df = df.merge(full_commos, on="Date", how="left", validate="1:1")
    df[correlation_como_cols] =  df[correlation_como_cols].bfill().ffill()
    como_corrs = df["STOCK"].rolling(rolling).corr(df[correlation_como_cols])
    como_corrs["Date"] = df["Date"]
    
    df = df.merge(como_corrs, on="Date", how="left", suffixes=("", "_CORRELATION_1Y"))

    return df


def stock_analysis(configs_general, data, full_commos, rolling=360):

    base_path = configs_general["resources"]["base_path"] / pl("data/extracted_data")
    liste_companies = os.listdir(base_path)
    missing=[]
    full_stock = pd.DataFrame()
    
    # in_finance = data.loc[data["SECTOR"].isin(map_sectors["FINANCE"])]["REUTERS_CODE"].unique()

    for company in tqdm.tqdm(list(set(liste_companies) - set(["currencies", "commodities"]))):
        # if company in in_finance:

            params = {"specific" : "",
                    "company" : company,
                    "base_path" : base_path}

            df = read_files(params, 'STOCK')
            if "STOCK" in df.keys():
                if df["STOCK"].shape[0] > 720:
                    df = df["STOCK"]

                    # only since 2000
                    df = df.loc[df["Date"] >= "2000-01-01"]

                    # for missing stock days to have a regular distance yoy
                    df = fill_missing_stock_dates(df)

                    # prediction days to fill
                    df = ajouter_date_prediction(df)
                    df = stock_feature_engineering(df)
                    df["COMPANY"] = company

                    # get commo correlation to stock
                    df = como_corr(df, full_commos, rolling)

                    full_stock = pd.concat([full_stock, df], axis=0)
                else: 
                    missing.append(company)
        
            else: 
                missing.append(company)
    
    print(missing)

    return full_stock, missing