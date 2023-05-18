import pandas as pd 
import numpy as np
from datetime import datetime
import logging.config as log_config
import logging
import os
import shutil
import glob
import ssl
import tqdm
import pickle
import yfinance as yf 
pd.options.mode.chained_assignment = None 

from utils.general_functions import smart_column_parser
from utils.config import Config
from dotenv import load_dotenv

ssl._create_default_https_context = ssl._create_unverified_context

class PrepareCrytpo(object):

    def __init__(self):

        load_dotenv("./configs/.env")
        self.hours = range(24)

        # init with app variables 
        self.configs = self.config_init("./configs/main.yml") 
        self.lags = self.configs.load["cryptos_desc"]["LAGS"]
        self.currencies = self.configs.load["cryptos_desc"]["Cryptos"]

        log_config.dictConfig(self.configs.logging)

        self.create_directories()

    def config_init(self, config_path):
        return Config(config_path).read()

    def create_directories(self):

        self.path_dirs = {}
        self.path_dirs["BASE"] = "./data"

        self.path_dirs["CURRENCY"] = "/".join([self.path_dirs["BASE"], "currencies"])
        if not os.path.isdir(self.path_dirs["CURRENCY"]):
            os.mkdir(self.path_dirs["CURRENCY"])

        self.path_dirs["INTERMEDIATE"] = "/".join([self.path_dirs["BASE"], "intermediate"])
        if not os.path.isdir(self.path_dirs["INTERMEDIATE"]):
            os.mkdir(self.path_dirs["INTERMEDIATE"])

        self.path_dirs["PORTFOLIO"] = "/".join([self.path_dirs["BASE"], "portfolio"])
        if not os.path.isdir(self.path_dirs["PORTFOLIO"]):
            os.mkdir(self.path_dirs["PORTFOLIO"])

        self.path_dirs["ORDERS"] = "/".join([self.path_dirs["BASE"], "orders"])
        if not os.path.isdir(self.path_dirs["ORDERS"]):
            os.mkdir(self.path_dirs["ORDERS"])

    def load_share_price_data(self):

        # urls = self.configs["cryptos_desc"]

        history, nbr_days = self.load_datas()

        if nbr_days:
            logging.info(f"History leveraged with nbr _days = {nbr_days}")
            period=f"{nbr_days}d"
        else:
            period="2y"

        datas = {}
        for currency in tqdm.tqdm(self.currencies):

            pair = f"{currency}-EUR"
            if currency == "STX":
                pair = "STX4847-EUR"
            
            # load intra day data 
            intra_day_data = yf.download(tickers=pair, period=period, interval="1h")
            intra_day_data = intra_day_data.reset_index()
            intra_day_data.rename(columns={"Volume" : "Volume " + currency, "index" : "Date", "Datetime" : "Date"}, inplace=True)
            intra_day_data = intra_day_data.drop("Adj Close", axis=1)
            intra_day_data["Date"] = intra_day_data["Date"].dt.strftime("%Y-%m-%d %H:%M:%S")
            intra_day_data.columns = smart_column_parser(intra_day_data.columns)

            if nbr_days:
                datas[currency] = pd.concat([history[currency], intra_day_data], axis=0)

            else:
                datas[currency] = intra_day_data

        return datas

    def pre_clean_cols(self, datas, currency):

        df = datas[currency].copy()
        df["DATE"] = pd.to_datetime(df["DATE"], format="%Y-%m-%d %H:%M:%S")
        df = df.sort_values(["DATE"], ascending= 0)
        df = df.drop_duplicates("DATE")

        df.rename(columns={"CLOSE" : f"CLOSE_{currency}",
                            # "TRADECOUNT" : f"TRADECOUNT_{currency}",
                            }, inplace=True)
        return df
    
    def rolling_mean(self, df, feature, nbr_values, type="mean"):
            return eval(f"df[feature].rolling(int(nbr_values), min_periods=1, center=False).{type}()")

    def normalize_target(self, df, feature, nbr_days=360):
        df[f"{feature}_ROLLING_MEAN"] = self.rolling_mean(df, feature, len(self.hours)*nbr_days, "mean")
        df[f"{feature}_ROLLING_STD"] = self.rolling_mean(df, feature, len(self.hours)*nbr_days, "std")
        
        return (df[feature] - df[f"{feature}_ROLLING_MEAN"]) / df[f"{feature}_ROLLING_STD"]

    def prepare_currency_daily(self, full, currency="BTC"):

        # create currency target 
        agg = full[["DATE", f"CLOSE_{currency}", f"VOLUME_{currency}"]]
        agg["DATE"] = agg["DATE"].dt.round("H")
        agg = agg.drop_duplicates("DATE")

        if currency == "XRP":
            agg.loc[agg["DATE"] == "2021-12-14 21:00:00"] = np.nan
        
        # remove mvs 
        agg.loc[agg[f"CLOSE_{currency}"] <= 0, f"CLOSE_{currency}"] = np.nan
        agg = agg.loc[~agg[f"CLOSE_{currency}"].isnull()]
        agg = agg.sort_values("DATE", ascending=True)

        # normalization over past year
        agg[f"CLOSE_{currency}_NORMALIZED"] = self.normalize_target(agg, f"CLOSE_{currency}", nbr_days=360)
        
        # volume -> sum last 4 hours 
        agg[f"VOLUME_{currency}"] = self.rolling_mean(agg, f"VOLUME_{currency}", 4, "sum")
        agg[f"VOLUME_{currency}_NORMALIZED"] = self.normalize_target(agg, f"VOLUME_{currency}", nbr_days=360)
        
        # rolling mean distance to X.d
        liste_targets = []
        for avg_mean in self.lags:
            if avg_mean not in ["MEAN_LAGS"]:
                # close moments
                agg[f"CLOSE_{currency}_ROLLING_MEAN_{avg_mean}D"] = self.rolling_mean(agg, f"CLOSE_{currency}_NORMALIZED", len(self.hours)*avg_mean, "mean")
                agg[f"CLOSE_{currency}_ROLLING_STD_{avg_mean}D"] = self.rolling_mean(agg, f"CLOSE_{currency}_NORMALIZED", len(self.hours)*avg_mean, "std")

                # volume moments
                agg[f"VOLUME_{currency}_ROLLING_MEAN_{avg_mean}D"] = self.rolling_mean(agg, f"VOLUME_{currency}_NORMALIZED", len(self.hours)*avg_mean, "mean")
                agg[f"VOLUME_{currency}_ROLLING_STD_{avg_mean}D"] = self.rolling_mean(agg, f"VOLUME_{currency}_NORMALIZED", len(self.hours)*avg_mean, "std")

                # distance to past averages for close
                agg[f"TARGET_{currency}_NORMALIZED_{avg_mean}"] = (agg[f"CLOSE_{currency}_NORMALIZED"] - agg[f"CLOSE_{currency}_ROLLING_MEAN_{avg_mean}D"])
                liste_targets.append(f"TARGET_{currency}_NORMALIZED_{avg_mean}")

                # distance to past averages for volume
                agg[f"VOLUME_{currency}_NORMALIZED_{avg_mean}"] = (agg[f"VOLUME_{currency}_NORMALIZED"] - agg[f"VOLUME_{currency}_ROLLING_MEAN_{avg_mean}D"])
                
        agg[f"TARGET_{currency}_NORMALIZED_MEAN_LAGS"] = agg[liste_targets].mean(axis=1)

        agg = agg.sort_values("DATE", ascending=False)

        return agg
    

    def distance_to_market(self, dict_full, currency):

        for k, col in {"BTC" : "CLOSE_BTC_NORMALIZED", 
                       "ETH" : "CLOSE_ETH_NORMALIZED"}.items():
            if  col not in dict_full[currency].columns:
                 dict_full[currency] =  dict_full[currency].merge(dict_full[k][["DATE", col]], on="DATE", how="left", validate="1:1")

        dict_full[currency]["MARKET_NORMALIZED"] = dict_full[currency][["CLOSE_BTC_NORMALIZED", "CLOSE_ETH_NORMALIZED"]].mean(axis=1)
        dict_full[currency] = dict_full[currency].sort_values("DATE", ascending=True)

        to_drop = []
        for avg_mean in self.lags:
            if avg_mean not in ["MEAN_LAGS"]:
                dict_full[currency][f"MARKET_ROLLING_MEAN_{avg_mean}D"] = self.rolling_mean(dict_full[currency], "MARKET_NORMALIZED", len(self.hours)*avg_mean, "mean")
                dict_full[currency][f"MARKET_ROLLING_STD_{avg_mean}D"] = self.rolling_mean(dict_full[currency], "MARKET_NORMALIZED", len(self.hours)*avg_mean, "std")
                
                dict_full[currency][f"MARKET_{currency}_NORMALIZED_{avg_mean}"] = dict_full[currency]["MARKET_NORMALIZED"] - dict_full[currency][f"MARKET_ROLLING_MEAN_{avg_mean}D"]
                dict_full[currency][f"DIFF_{currency}_TO_MARKET_{avg_mean}"] = dict_full[currency][f"CLOSE_{currency}_NORMALIZED"] - dict_full[currency][f"MARKET_ROLLING_MEAN_{avg_mean}D"]
                
                to_drop.append(f"MARKET_ROLLING_MEAN_{avg_mean}D")

        dict_full[currency][f"DIFF_{currency}_TO_MARKET_MEAN_LAGS"] = dict_full[currency][[f"DIFF_{currency}_TO_MARKET_{x}" for x in self.lags if x != "MEAN_LAGS"]].mean(axis=1)
        dict_full[currency] = dict_full[currency].sort_values("DATE", ascending=False)

        return dict_full[currency].drop(to_drop, axis=1)


    def aggregate_crypto_price(self, datas):

        dict_full = {}

        for currency in self.currencies:
            df = self.pre_clean_cols(datas, currency)
            
            # data prep target
            dict_full[currency] = self.prepare_currency_daily(df, currency)

        for currency in self.currencies:

            # add distances to others 
            dict_full[currency] = self.distance_to_market(dict_full, currency)
        
        return dict_full


    def save_prep(self, datas, prepared):

        utcnow = datas["BTC"]["DATE"].max().replace(" ", "_").replace(":","-")

        # save currencies 
        self.remove_files_from_dir(self.path_dirs["CURRENCY"])
        pickle.dump(datas, open("/".join([self.path_dirs["CURRENCY"], f"cryptos_{utcnow}.pkl"]), 'wb'))

        # save prepared data 
        self.remove_files_from_dir(self.path_dirs["INTERMEDIATE"])
        pickle.dump(prepared, open("/".join([self.path_dirs["INTERMEDIATE"], f"prepared_{utcnow}.pkl"]), 'wb'))


    def load_datas(self):
        list_of_files = glob.glob(self.path_dirs["CURRENCY"]+"/*") 
        if len(list_of_files)>0:
            latest_file = max(list_of_files, key=os.path.getctime)
            ti_c = os.path.getctime(latest_file)
            nbr_days = (datetime.today() - datetime.fromtimestamp(ti_c)).days
            return pickle.load(open(latest_file, 'rb')), nbr_days + 20
        else:
            return None, None

    def load_prepared(self):
        list_of_files = glob.glob(self.path_dirs["INTERMEDIATE"]+"/*")

        if len(list_of_files)>0:
            latest_file = max(list_of_files, key=os.path.getctime)
            return pickle.load(open(latest_file, 'rb'))
        else:
            return None

    def remove_files_from_dir(self, folder):

        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
