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
import scipy.ndimage as ndi
from scipy.stats import truncnorm, uniform, norm

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
        self.targets = self.configs.load["cryptos_desc"]["TARGETS"]
        self.lags = self.configs.load["cryptos_desc"]["LAGS"]
        self.currencies = self.configs.load["cryptos_desc"]["Cryptos"]
        self.market_makers_currencies = ["BTC", "ETH", "ADA", "XRP"]

        self.normalization_days = 200

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

        history, nbr_days = self.load_datas()

        if nbr_days and sum([1 for x in self.currencies if x not in list(history.keys())]) < 1:
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

            if nbr_days and sum([1 for x in self.currencies if x not in list(history.keys())]) < 1:
                datas[currency] = pd.concat([history[currency], intra_day_data], axis=0)

            else:
                datas[currency] = intra_day_data

        return datas

    def pre_clean_cols(self, datas, currency):

        df = datas[currency].copy()
        df["DATE"] = pd.to_datetime(df["DATE"], format="%Y-%m-%d %H:%M:%S")
        df = df.sort_values(["DATE"], ascending= 0)
        df = df.drop_duplicates("DATE")

        df.rename(columns={f"VOLUME_{currency}" : "VOLUME"}, inplace=True)

        return df
    
    def rolling_window(self, a, window):
        window = int(window)
        good_shape = a.shape[0]
        reflect = a[-window+1:][::-1]
        a = np.concatenate((a, reflect), axis=0)
        matrix_ = np.lib.stride_tricks.sliding_window_view(a, window)
        return matrix_[:good_shape, :]

    def running_mean_uniform_filter1d(self, x, N):
        N = int(N)
        return ndi.uniform_filter1d(x, N, mode='reflect', origin=-(N//2))

    def normalize_target(self, df, feature, nbr_days=360):
        rollings = self.rolling_window(df[feature], len(self.hours)*nbr_days)
        df[f"{feature}_ROLLING_MEAN"] = np.mean(rollings, axis=1) 
        df[f"{feature}_ROLLING_STD"] = np.std(rollings, axis=1) 
        
        return (df[feature] - df[f"{feature}_ROLLING_MEAN"]) / df[f"{feature}_ROLLING_STD"]

    def deduce_threshold(self, prepared, target, lag, days=180): # doit dépendre de l'historique vu réellement

        # matrix_ = self.rolling_window(prepared[target].ffill(), len(self.hours)*days)
        # normed = np.apply_along_axis(norm.fit, 1, matrix_)
        mu, std = norm.fit(prepared[target].ffill())

        # Fit a normal distribution to the data:
        prepared[f"SEUIL_MEAN_{lag}"] = mu
        prepared[f"SEUIL_STD_{lag}"] = std #normed[:,1]

        return prepared
    
    def trend_fit(self, y, X):
        return np.linalg.lstsq(X, np.log(y), rcond=None)[0]

    def prepare_currency_daily(self, full, currency="BTC"):

        # create currency target 
        agg = full[["DATE", "CLOSE", "VOLUME"]]
        agg["DATE"] = agg["DATE"].dt.round("H")
        agg = agg.drop_duplicates("DATE")

        if currency == "XRP":
            agg.loc[agg["DATE"] == "2021-12-14 21:00:00"] = np.nan

        if currency == "XLM":
            agg.loc[agg["DATE"] == "2021-09-09 00:00:00"] = np.nan

        if currency == "TRX":
            agg.loc[agg["DATE"] == "2021-06-14 08:00:00"] = np.nan
        
        # remove mvs 
        agg.loc[agg["CLOSE"] <= 0, "CLOSE"] = np.nan
        agg = agg.loc[~agg["CLOSE"].isnull()]

        # normalization over past year
        agg["CLOSE_NORMALIZED"] = agg["CLOSE"] # (agg["CLOSE"] - agg["CLOSE"].mean())/ agg["CLOSE"].std() #self.normalize_target(agg, "CLOSE", nbr_days=min(agg.shape[0], self.normalization_days)) #
        
        # volume -> sum last 4 hours 
        agg["VOLUME"] = np.sum(self.rolling_window(agg["VOLUME"], 6), axis=1)
        agg["VOLUME_NORMALIZED"] = agg["VOLUME"]#(agg["VOLUME"] - agg["VOLUME"].mean())/ agg["VOLUME"].std() #self.normalize_target(agg, "VOLUME", nbr_days=min(agg.shape[0], self.normalization_days)) #
        
        # rolling mean distance to X.d
        liste_targets = []
        for avg_mean in self.lags:
            if avg_mean not in ["MEAN_LAGS"]:

                # close moments
                logging.info(f"[{currency}] CLOSE DIST {avg_mean}")
                rollings = self.rolling_window(agg["CLOSE_NORMALIZED"], len(self.hours)*float(avg_mean))
                agg[f"CLOSE_ROLLING_MEAN_{avg_mean}D"] = np.mean(rollings, axis=1)
                agg[f"CLOSE_ROLLING_STD_{avg_mean}D"] = np.std(rollings, axis=1) 

                # trend deduction
                X = range(rollings.shape[1])
                kwargs = {"X" : np.vstack([X, np.ones(len(X))]).T}
                params = np.apply_along_axis(self.trend_fit, 1, rollings, **kwargs)
                agg[f"CLOSE_TREND_{avg_mean}"] = params[:,0]*24 # trend per day

                # volume moments
                rollings = self.rolling_window(agg["VOLUME_NORMALIZED"], len(self.hours)*float(avg_mean))
                agg[f"VOLUME_ROLLING_MEAN_{avg_mean}D"] = np.mean(rollings, axis=1)
                agg[f"VOLUME_ROLLING_STD_{avg_mean}D"] = np.std(rollings, axis=1) 

                # distance to past averages for close
                agg[f"TARGET_NORMALIZED_{avg_mean}"] = (agg["CLOSE_NORMALIZED"] - agg[f"CLOSE_ROLLING_MEAN_{avg_mean}D"].shift(-1))*10 / agg[f"CLOSE_ROLLING_MEAN_{avg_mean}D"].shift(-1)
                liste_targets.append(f"TARGET_NORMALIZED_{avg_mean}")

                # distance to past averages for volume
                agg[f"VOLUME_NORMALIZED_{avg_mean}"] = (agg["VOLUME_NORMALIZED"] - agg[f"VOLUME_ROLLING_MEAN_{avg_mean}D"].shift(-1))*10 / agg[f"VOLUME_ROLLING_MEAN_{avg_mean}D"].shift(-1)
                
            agg = self.deduce_threshold(agg, f"TARGET_NORMALIZED_{avg_mean}", lag=avg_mean, days=self.normalization_days)

        agg["TARGET_NORMALIZED_MEAN_LAGS"] = agg[liste_targets].mean(axis=1)

        return agg
    

    def distance_to_market(self, dict_full, currency):

        for avg_mean in self.lags:
            if avg_mean not in ["MEAN_LAGS"]:

                # average % increase decrease per lag
                for curr in self.market_makers_currencies:
                    df_to_merge = dict_full[curr][["DATE", f"TARGET_NORMALIZED_{avg_mean}"]]
                    df_to_merge = df_to_merge.rename(columns={f"TARGET_NORMALIZED_{avg_mean}" : f"TARGET_{curr}"})
                    dict_full[currency] = dict_full[currency].merge(df_to_merge, on="DATE", how="left", validate="1:1")
                
                cols= [f"TARGET_{x}" for x in self.market_makers_currencies]
                dict_full[currency][f"MARKET_NORMALIZED_{avg_mean}"] = dict_full[currency][cols].mean(axis=1)
                dict_full[currency] = dict_full[currency].drop(cols, axis=1)

                rollings = self.rolling_window(dict_full[currency][f"MARKET_NORMALIZED_{avg_mean}"], len(self.hours)*float(avg_mean))
                dict_full[currency][f"MARKET_ROLLING_MEAN_{avg_mean}D"] = np.mean(rollings, axis=1)
                dict_full[currency][f"MARKET_ROLLING_STD_{avg_mean}D"] = np.std(rollings, axis=1) 
                
                # defined as p.p up or down to market % variation average
                dict_full[currency][f"DIFF_TO_MARKET_{avg_mean}"] = (dict_full[currency][f"TARGET_NORMALIZED_{avg_mean}"] - dict_full[currency][f"MARKET_ROLLING_MEAN_{avg_mean}D"])*10

        dict_full[currency]["DIFF_TO_MARKET_MEAN_LAGS"] = dict_full[currency][[f"DIFF_TO_MARKET_{x}" for x in self.lags if x != "MEAN_LAGS"]].mean(axis=1)
        dict_full[currency] = dict_full[currency].sort_values("DATE", ascending=False)

        dict_full[currency] = dict_full[currency].loc[~dict_full[currency]["CLOSE_NORMALIZED"].isnull()]

        return dict_full[currency]
    

    def data_prep_strats(self, prepared):

        prepared = prepared.sort_values("DATE", ascending = False)

        for day_future in self.targets:

            hours = int(day_future*len(self.hours))
            prepared[f"FUTUR_TARGET_{day_future}"] = prepared["CLOSE_NORMALIZED"].rolling(window=hours, center=True).mean().shift(hours)
            prepared[f"BINARY_FUTUR_TARGET_{day_future}"] = 1*(prepared[f"FUTUR_TARGET_{day_future}"] > prepared["CLOSE_NORMALIZED"])
            prepared[f"DELTA_FUTUR_TARGET_{day_future}"] = (prepared[f"FUTUR_TARGET_{day_future}"] - prepared["CLOSE_NORMALIZED"]) *10 / prepared["CLOSE_NORMALIZED"]

        return prepared


    def aggregate_crypto_price(self, datas):

        dict_full = {}

        for currency in self.currencies:
            df = self.pre_clean_cols(datas, currency)
            
            # data prep target
            dict_full[currency] = self.prepare_currency_daily(df, currency)

        for currency in self.currencies:
            dict_full[currency] = self.distance_to_market(dict_full, currency)
            dict_full[currency] = self.data_prep_strats(dict_full[currency])

            # TODO add distance to S&P
            # TODO add distance to gold
            # TODO distance to VIX ?
        
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
            return pickle.load(open(latest_file, 'rb')), nbr_days + 15
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
