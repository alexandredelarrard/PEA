import pandas as pd 
import numpy as np
import os 
from pathlib import Path as pl

import plotly.express as px  # interactive charts


def stock_processing(inputs):

    stock = inputs.copy()

    # round stock evolution per quarter 
    stock.columns = [x.upper().replace(" ", "_") for x in stock.columns]
    stock = stock[["DATE", "CLOSE", "VOLUME"]]

    # round to Quarter 
    stock["DATE"] = pd.to_datetime(stock["DATE"], format="%Y-%m-%d")
  
    return stock


def read_files(params):

    liste_dates_company = os.listdir(params["base_path"] / pl(params["company"]))
    finance_date = max(liste_dates_company)

    for file in ["STOCK.csv"]: 
        try:
            inputs = pd.read_csv(params["base_path"] / pl(params["company"]) / pl(finance_date) / file)
            inputs = stock_processing(inputs)

        except Exception as e: 
            # print(f"ERROR LOAD DATA : {params['company']} / {f} / {e}")
            pass

    return inputs
    

def load_stocks(data, configs_general):

    base_path = configs_general["resources"]["base_path"] / pl("data/extracted_data")
    liste_companies = data["REUTERS_CODE"].unique()
    all_stocks = pd.DataFrame()

    for company in set(liste_companies):

        sub_data = data.loc[data["REUTERS_CODE"] == company]

        params = {"specific" : "",
                  "company" : company,
                  "base_path" : base_path}
        
        inputs = read_files(params)

        inputs["REUTERS_CODE"] = company 
        inputs  = inputs.merge(sub_data, on="REUTERS_CODE", how="left", validate="m:1")

        all_stocks = pd.concat([all_stocks, inputs], axis=0)

    return all_stocks


def stocks_lt(stocks, since="2010-01-01"):

    small_date = stocks[["DATE", "NAME"]].groupby(["NAME"]).min().reset_index()
    big_date = stocks[["DATE", "NAME"]].groupby(["NAME"]).max().reset_index()

    keep_since = small_date.loc[small_date["DATE"] < since, "NAME"].unique()
    keep_still = big_date.loc[big_date["DATE"] == big_date["DATE"].max(), "NAME"].unique()
    
    to_study = list(set(keep_since).intersection(set(keep_still)))
    sub_stocks = stocks.loc[stocks["NAME"].isin(to_study)]
    sub_stocks = sub_stocks.loc[sub_stocks["DATE"] >= since]

    return sub_stocks


def sector_comparison(sub_stocks):

    #cac agg 
    agg_200 = sub_stocks[["DATE", "CLOSE", "VOLUME"]].groupby("DATE").sum().reset_index()
    agg_200["CLOSE"] =  100*(agg_200["CLOSE"] /  agg_200["CLOSE"].values[0])
    agg_200["VOLUME"] =  100*(agg_200["VOLUME"] /  agg_200["VOLUME"].values[0])
    agg_200.columns = ["DATE", "CLOSE", "VOLUME"]
    agg_200["SECTOR"] = "CAC_200"

    # aggregate_sector 
    for sectors in sub_stocks["SECTOR"].unique():
        stock_sector = sub_stocks.loc[sub_stocks["SECTOR"] == sectors]
        agg_stock_sector = stock_sector[["DATE", "CLOSE", "VOLUME"]].groupby("DATE").sum().reset_index()
        agg_stock_sector["CLOSE"] =  100*(agg_stock_sector["CLOSE"] /  agg_stock_sector["CLOSE"].values[0])
        agg_stock_sector["VOLUME"] =  100*(agg_stock_sector["VOLUME"] /  agg_stock_sector["VOLUME"].values[0])
        agg_stock_sector.columns = ["DATE", "CLOSE", "VOLUME"]
        agg_stock_sector["SECTOR"] = sectors
       
        agg_200 = pd.concat([agg_200, agg_stock_sector], axis=0)

    return agg_200


def comp_per_sector_comparison(sub_stock, sector):

    sub_sub = sub_stock.loc[sub_stock["SECTOR"] == sector]
    sub_sub = sub_sub.reset_index(drop="True")

    #cac agg 
    agg_200 = sub_sub[["DATE", "CLOSE", "VOLUME"]].groupby("DATE").sum().reset_index()
    agg_200["CLOSE"] =  100*(sub_sub["CLOSE"] /  sub_sub["CLOSE"].values[0])
    agg_200["VOLUME"] =  100*(sub_sub["VOLUME"] /  sub_sub["VOLUME"].values[0])
    agg_200.columns = ["DATE", "CLOSE", "VOLUME"]
    agg_200["SECTOR"] = sector.upper()

    # aggregate_sector 
    for name in sub_sub["NAME"].unique():
        stock_sector = sub_stocks.loc[sub_stocks["NAME"] == name]
        agg_stock_sector = stock_sector[["DATE", "CLOSE", "VOLUME"]].groupby("DATE").sum().reset_index()
        agg_stock_sector["CLOSE"] =  100*(agg_stock_sector["CLOSE"] /  agg_stock_sector["CLOSE"].values[0]) 
        agg_stock_sector["VOLUME"] =  100*(agg_stock_sector["VOLUME"] /  agg_stock_sector["VOLUME"].values[0])
        agg_stock_sector.columns = ["DATE", "CLOSE", "VOLUME"]
        agg_stock_sector["SECTOR"] = name.lower()
       
        agg_200 = pd.concat([agg_200, agg_stock_sector], axis=0)

    return agg_200


if __name__ == "__main__":

    # load stock_data
    stocks = load_stocks(data, configs_general)

    # filter stocks to those available since 2010
    sub_stocks = stocks_lt(stocks, since="2017-01-01")

    # sector comparison
    agg_sectors = sector_comparison(sub_stocks)

    #analyse sector increase / decrease
    fig = px.line(agg_sectors, x="DATE", y="VOLUME", color='SECTOR',
                 width=1300, height=800)
    fig.show()

    # sub_sector = comp_per_sector_comparison(sub_stocks, sector="Industrials")
    #analyse sector increase / decrease
    # fig = px.line(sub_sector, x="DATE", y="CLOSE", color='SECTOR',
    #              width=1300, height=800)
    # fig.show()





