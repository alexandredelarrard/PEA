from logging import raiseExceptions
import pandas as pd 
import numpy as np 
from datetime import datetime


def stock_processing(inputs):

    stock = inputs["STOCK"].copy()

    if stock.shape[0] == 0:
        raise Exception("No stock data available -> dataset is empty")

    # round stock evolution per quarter 
    stock.columns = [x.upper().replace(" ", "_") for x in stock.columns]
    stock = stock[["DATE", "CLOSE", "VOLUME"]]

    # round to Quarter 
    stock["DATE"] = pd.to_datetime(stock["DATE"], format="%Y-%m-%d")
    stock["YEAR"] = stock["DATE"].dt.year
    stock["MONTH"] = stock["DATE"].dt.month
    stock["QUARTER"] = stock['DATE'].dt.to_period('Q')

    agg_stock  = stock[["QUARTER", "CLOSE", "VOLUME"]].groupby("QUARTER").mean()

    # trend quarter to quarter 
    for col in ["CLOSE", "VOLUME"]:
        agg_stock[f"{col}_Q_TREND"] = (agg_stock[col] - agg_stock[col].shift(1)) / agg_stock[col].shift(1)
        agg_stock[f"{col}_Y_TREND"] = (agg_stock[col] - agg_stock[col].shift(4)) / agg_stock[col].shift(4)

    return agg_stock, stock


def add_stock_to_data(agg_stock, stock, data):

    stocks_values = []
    volume_values = []

    for col in data.columns:
        if col == "TTM":
            stocks_values.append(stock.iloc[-1, 1])
            volume_values.append(stock.iloc[-1, 2])
        else:
            if col + 1 in agg_stock.index:
                stocks_values.append(agg_stock.loc[col + 1, "CLOSE"])
                volume_values.append(agg_stock.loc[col + 1, "VOLUME"])

            elif col == pd.to_datetime(datetime.today()).to_period("Q"):
                stocks_values.append(agg_stock.loc[col, "CLOSE"])
                volume_values.append(agg_stock.loc[col, "VOLUME"])

            else:
                stocks_values.append(np.nan)
                volume_values.append(np.nan)
    
    data.loc["STOCK_CLOSE_PLUS_1Q"] = stocks_values
    data.loc["STOCK_VOLUME_PLUS_1Q"] = volume_values

    return data
