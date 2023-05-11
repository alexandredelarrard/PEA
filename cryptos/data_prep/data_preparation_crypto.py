import pandas as pd 
import ssl
import numpy as np
import tqdm
import yfinance as yf 
from utils.general_functions import smart_column_parser

ssl._create_default_https_context = ssl._create_unverified_context

class PrepareCrytpo(object):

    def __init__(self, configs, lags):
        self.configs = configs
        self.hours = range(24)
        self.lags = lags
        self.currencies = self.configs["cryptos_desc"]["Cryptos"]

    def load_share_price_data(self):

        # urls = self.configs["cryptos_desc"]

        datas = {}
        for currency in tqdm.tqdm(self.currencies):
            # url = urls["URL_BASE"] + "Binance_" + currency + urls["URL_END"]
            # datas[currency] = pd.read_csv(url, delimiter=",", skiprows=[0]) 

            pair = f"{currency}-EUR"
            if currency == "STX":
                pair = "STX4847-EUR"
            
            # load intra day data 
            intra_day_data = yf.download(tickers=pair, period="2y", interval="1h")
            intra_day_data = intra_day_data.reset_index()
            intra_day_data["Symbol"] = currency + "EUR"
            intra_day_data.rename(columns={"Volume" : "Volume " + currency, "index" : "Date"}, inplace=True)
            intra_day_data = intra_day_data.drop("Adj Close", axis=1)
            intra_day_data["Date"] = intra_day_data["Date"].dt.strftime("%Y-%m-%d %H:%M:%S")
            intra_day_data["TRADECOUNT"] = 0

            # datas[currency] = pd.concat([datas[currency], intra_day_data], axis=0)
            datas[currency] = intra_day_data

        return datas

    def pre_clean_cols(self, datas, currency):

        df = datas[currency].copy()
        df["DATE"] = pd.to_datetime(df["DATE"], format="%Y-%m-%d %H:%M:%S")
        df = df.sort_values(["DATE"], ascending= 0)
        df = df.drop_duplicates("DATE")

        df["RANGE"] = df["HIGH"] - df["LOW"]
        df.rename(columns={"CLOSE" : f"CLOSE_{currency}",
                            "RANGE" : f"RANGE_{currency}",
                            "TRADECOUNT" : f"TRADECOUNT_{currency}",
                            }, inplace=True)
        return df
    
    
    def prepare_currency_daily(self, full, currency="BTC"):

        def rolling_mean(df, feature, nbr_values, type="mean"):
            return eval(f"df[feature].rolling(nbr_values, min_periods=1, center=True).{type}()")

        # create currency target 
        agg = full[["DATE", f"CLOSE_{currency}"]]
        agg = agg.loc[~agg[f"CLOSE_{currency}"].isnull()]

        # rolling mean distance to 7d, 15d, 30d, 45d
        liste_targets = []
        for avg_mean in self.lags:
            agg[f"CLOSE_{currency}_ROLLING_MEAN_{avg_mean}D"] = rolling_mean(agg, f"CLOSE_{currency}", len(self.hours)*avg_mean, "mean")
            agg[f"CLOSE_{currency}_ROLLING_STD_{avg_mean}D"] = rolling_mean(agg, f"CLOSE_{currency}", len(self.hours)*avg_mean, "std")

            # create target normalized to be able to compare 
            agg[f"TARGET_{currency}_NORMALIZED_{avg_mean}"] = (agg[f"CLOSE_{currency}"] - agg[f"CLOSE_{currency}_ROLLING_MEAN_{avg_mean}D"].shift(-1)) / agg[f"CLOSE_{currency}_ROLLING_STD_{avg_mean}D"].shift(-1)
            liste_targets.append(f"TARGET_{currency}_NORMALIZED_{avg_mean}")

        #tradecount normalized to mean of past 30 days 
        # agg[f"TRADECOUNT_{currency}"] = np.where(agg[f"TRADECOUNT_{currency}"]==0, np.nan, agg[f"TRADECOUNT_{currency}"])
        # agg[f"TRADECOUNT_{currency}_ROLLING_MEAN"] = rolling_mean(agg, f"TRADECOUNT_{currency}", len(self.hours)*avg_mean, "mean")
        # agg[f"TRADECOUNT_{currency}_NORMALIZED"] = (agg[f"TRADECOUNT_{currency}"] - agg[f"TRADECOUNT_{currency}_ROLLING_MEAN"].shift(-1)) / agg[f"TRADECOUNT_{currency}_ROLLING_MEAN"].shift(-1)

        return agg[["DATE", f"CLOSE_{currency}"] +
                    liste_targets]
    

    def distance_to_market(self, full, currency):

        for avg_mean in self.lags:
            full[f"MARKET_NORMALIZED_{avg_mean}"] = 0
            full["TRADECOUNT"] = 0
            for i, currency in enumerate(self.currencies):
                full[f"MARKET_NORMALIZED_{avg_mean}"] += full[f"TARGET_{currency}_NORMALIZED_{avg_mean}"]*full[f"TRADECOUNT_{currency}"]
                full["TRADECOUNT"] += full[f"TRADECOUNT_{currency}"]

            full[f"MARKET_NORMALIZED_{avg_mean}"] = full[f"MARKET_NORMALIZED_{avg_mean}"]/(full["TRADECOUNT"])
        
            for i, currency in enumerate(self.currencies):
                full[f"DIFF_{currency}_TO_MARKET_{avg_mean}"] = full[f"TARGET_{currency}_NORMALIZED_{avg_mean}"] - full[f"MARKET_NORMALIZED_{avg_mean}"]
       
        return full


    def aggregate_crypto_price(self, datas):

        for i, currency in enumerate(self.currencies):

            datas[currency].columns = smart_column_parser(datas[currency].columns)
            df = self.pre_clean_cols(datas, currency)
            
            # data prep
            df = self.prepare_currency_daily(df, currency)

            if i == 0: 
                full = df
            else:
                full = pd.merge(full, df, on="DATE", how="left", validate="1:1")

        # add distances to others 
        full = self.distance_to_market(full, currency)
        
        return full
