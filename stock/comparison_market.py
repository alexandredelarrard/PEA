import pandas as pd 
import numpy as np
import os 
from pathlib import Path as pl

import plotly.express as px  # interactive charts
import yaml
from datetime import datetime

from crawling.extract_stock_history import main_extract_stock

today = datetime.today()

mapping_sectors = {'Hotels & Restaurants' : "consumers", 
                   'Airlines' : "transport", 
                   'Chemicals' : "materials",
                    'Aerospace & Defense' : "defense", 
                    'Industrials' : "industry", 
                    'Information Services' : "information",
                    'Materials' : "materials", 
                    'Insurance' : "financials", 
                    'Consumer Staples' : "consumers",
                    'Health Care Equipment' : "health", 
                    'Banks' : "financials", 
                    'Communication Services': "communication",
                    'Construction and Materials' : "materials",
                    'Application Software' : "information", 
                    'Utilities' : "industry",
                    'Energy' : "energy", 
                    'Apparel, Accessories & Luxury Goods': "luxury", 
                    'Oil & Gas' : "energy",
                    'Asset Management' : "financials", 
                    'Health Care Services' : "health", 
                    'Telecommunications' : "communication",
                    'Automobiles and Parts' : "transport", 
                    'Real Estate' : "real estate",
                    'Electronic Components' : "electronique",
                    'Biotechnology' : "biotech", 
                    'Vinters & Tobacco' : "luxury", 
                    'Semiconductors' : "electronique",
                    'Financials': "financials"
                }

def load_configs(config_yml):
    config_path = "./configs"
    config_yaml = open(pl(config_path) / pl(config_yml), encoding="utf-8")
    configs = yaml.load(config_yaml, Loader=yaml.FullLoader)
    return configs


def stock_processing(inputs):

    stock = inputs.copy()

    # round stock evolution per quarter 
    stock = stock.reset_index()
    stock.columns = [x.upper().replace(" ", "_") for x in stock.columns]
    stock = stock[["DATE", "CLOSE", "VOLUME"]]

    # round to Quarter 
    stock["DATE"] = pd.to_datetime(stock["DATE"], format="%Y-%m-%d")
  
    return stock


def load_stocks(stocks, data, since="2022-08-01"):

    all_stocks = pd.DataFrame()
    data["GAIN"] = 0
    data["STD_CLOSE"] = 0
    data["GAIN_DIV"] = 0
    since = pd.to_datetime(since, format="%Y-%m-%d")

    for company, inputs in stocks.items():

        inputs = stock_processing(inputs)
        inputs["YAHOO_CODE"] = company 
        inputs  = inputs.merge(data.loc[data["YAHOO_CODE"] == company], on="YAHOO_CODE", how="left", validate="m:1")

        inputs = inputs.loc[inputs["DATE"] >= since]
        inputs["CLOSE"] = inputs["CLOSE"] / np.mean(inputs["CLOSE"].values[0:5])
        inputs["VOLUME"] = inputs["VOLUME"] / np.mean(inputs["VOLUME"].values[0:5])

        data.loc[data["YAHOO_CODE"] == company, "GAIN_DIV"] = inputs["CLOSE"].median()*data.loc[data["YAHOO_CODE"] == company, "DIV"].values[0]*((today - since).days / 365)/100
        data.loc[data["YAHOO_CODE"] == company, "GAIN"] = inputs["CLOSE"].values[-1]
        data.loc[data["YAHOO_CODE"] == company, "STD_CLOSE"] = inputs["CLOSE"].std()

        all_stocks = pd.concat([all_stocks, inputs], axis=0)

    data["SHARP"] = ((data["GAIN"] + data["GAIN_DIV"]) -1) / (data["STD_CLOSE"])
    data["AGG_SECTOR"] = data["SECTOR"].map(mapping_sectors)
    data["SCORE"] = data["SHARP"] / np.sqrt(data["PE"])

    data = data.sort_values("SCORE", ascending = 0)

    return all_stocks, data


def sector_comparison(sub_stocks):

    sub_stocks["AGG_SECTOR"] = sub_stocks["SECTOR"].map(mapping_sectors)

    #cac agg 
    agg_200 = sub_stocks[["DATE", "CLOSE", "VOLUME"]].groupby("DATE").mean().reset_index()
    agg_200.columns = ["DATE", "CLOSE", "VOLUME"]
    agg_200["AGG_SECTOR"] = "CAC_200"

    # aggregate_sector 
    for sectors in sub_stocks["AGG_SECTOR"].unique():
        stock_sector = sub_stocks.loc[sub_stocks["AGG_SECTOR"] == sectors]
        agg_stock_sector = stock_sector[["DATE", "CLOSE", "VOLUME"]].groupby("DATE").mean().reset_index()
        agg_stock_sector.columns = ["DATE", "CLOSE", "VOLUME"]
        agg_stock_sector["AGG_SECTOR"] = sectors
       
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
        stock_sector = sub_stock.loc[sub_stock["NAME"] == name]
        agg_stock_sector = stock_sector[["DATE", "CLOSE", "VOLUME"]].groupby("DATE").sum().reset_index()
        agg_stock_sector["CLOSE"] =  100*(agg_stock_sector["CLOSE"] /  agg_stock_sector["CLOSE"].values[0]) 
        agg_stock_sector["VOLUME"] =  100*(agg_stock_sector["VOLUME"] /  agg_stock_sector["VOLUME"].values[0])
        agg_stock_sector.columns = ["DATE", "CLOSE", "VOLUME"]
        agg_stock_sector["SECTOR"] = name.lower()
       
        agg_200 = pd.concat([agg_200, agg_stock_sector], axis=0)

    return agg_200


if __name__ == "__main__":

    # load stock_data
    configs_general = load_configs("configs_pea.yml")
    base_path = configs_general["resources"]["base_path"]
    sub_loc= base_path / pl("data/extracted_data")

    data = pd.read_csv(base_path / pl("data/data_for_crawling/mapping_reuters_yahoo.csv"), sep=";", decimal=",", encoding="latin1")
    data["SECTOR"] = data["SECTOR"].apply(lambda x: x.strip())
    data = data.loc[data["Country"] == "FR"]
    data = data.loc[(data["SBF"] == 1)&(data["ETAT"] == 0)]

    stocks_extract, missing_ticks = main_extract_stock(data, sub_loc=sub_loc, split_size = 100)

    # concatenate all stocks
    stocks, data = load_stocks(stocks_extract, data, since="2022-08-01")

    # sector comparison
    agg_sectors = sector_comparison(stocks)

    #analyse sector increase / decrease
    fig = px.line(agg_sectors, x="DATE", y="CLOSE", color='AGG_SECTOR',
                 width=1300, height=800)
    fig.show()

    # sub_sector = comp_per_sector_comparison(sub_stocks, sector="Industrials")
    #analyse sector increase / decrease
    # fig = px.line(sub_sector, x="DATE", y="CLOSE", color='SECTOR',
    #              width=1300, height=800)
    # fig.show()





