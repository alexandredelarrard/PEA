import pandas as pd 
import numpy as np
import logging
import os
import shutil
import glob
import ssl
import pickle
from typing import Dict
from datetime import timedelta
import scipy.ndimage as ndi
from scipy.stats import truncnorm, uniform, norm

from data_load.data_loading import LoadCrytpo

pd.options.mode.chained_assignment = None 

ssl._create_default_https_context = ssl._create_unverified_context

class PrepareCrytpo(LoadCrytpo):

    def __init__(self):

        LoadCrytpo.__init__(self)

        self.hours = range(24)
        self.obs_per_hour = 60//self.granularity
        
        # init with app variables 
        self.targets = self.configs.load["cryptos_desc"]["TARGETS"]
        self.lags = self.configs.load["cryptos_desc"]["LAGS"]
        self.market_makers_currencies = ["BTC", "ETH", "ADA", "XRP"]

        self.normalization_days = 200


    def pre_clean_cols(self, df):

        df["DATE"] = pd.to_datetime(df["DATE"], format="%Y-%m-%d %H:%M:%S")
        df = df.sort_values(["DATE"], ascending= 0)
        df = df.drop_duplicates("DATE")

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
        rollings = self.rolling_window(df[feature], len(self.hours)*nbr_days*self.obs_per_hour)
        df[f"{feature}_ROLLING_MEAN"] = np.mean(rollings, axis=1) 
        df[f"{feature}_ROLLING_STD"] = np.std(rollings, axis=1) 
        
        return (df[feature] - df[f"{feature}_ROLLING_MEAN"]) / df[f"{feature}_ROLLING_STD"]

    def deduce_threshold(self, prepared, target, lag): # doit dépendre de l'historique vu réellement

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

        logging.info(f"[{currency}] CLOSE DIST")

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
        agg["VOLUME"] = np.sum(self.rolling_window(agg["VOLUME"], 6*self.obs_per_hour), axis=1)
        agg["VOLUME_NORMALIZED"] = agg["VOLUME"]#(agg["VOLUME"] - agg["VOLUME"].mean())/ agg["VOLUME"].std() #self.normalize_target(agg, "VOLUME", nbr_days=min(agg.shape[0], self.normalization_days)) #
        
        # rolling mean distance to X.d
        liste_targets = []
        for avg_mean in self.lags:
            if avg_mean not in ["MEAN_LAGS"]:

                # close moments
                rollings = self.rolling_window(agg["CLOSE_NORMALIZED"], len(self.hours)*float(avg_mean)*self.obs_per_hour)
                agg[f"CLOSE_ROLLING_MEAN_{avg_mean}D"] = np.mean(rollings, axis=1)
                agg[f"CLOSE_ROLLING_STD_{avg_mean}D"] = np.std(rollings, axis=1) 

                # trend deduction
                X = range(rollings.shape[1])
                kwargs = {"X" : np.vstack([X, np.ones(len(X))]).T}
                params = np.apply_along_axis(self.trend_fit, 1, rollings, **kwargs)
                agg[f"CLOSE_TREND_{avg_mean}"] = params[:,0]*len(self.hours)*self.obs_per_hour # trend per day

                # volume moments
                rollings = self.rolling_window(agg["VOLUME_NORMALIZED"], len(self.hours)*float(avg_mean)*self.obs_per_hour)
                agg[f"VOLUME_ROLLING_MEAN_{avg_mean}D"] = np.mean(rollings, axis=1)
                agg[f"VOLUME_ROLLING_STD_{avg_mean}D"] = np.std(rollings, axis=1) 

                # distance to past averages for close
                agg[f"TARGET_NORMALIZED_{avg_mean}"] = (agg["CLOSE_NORMALIZED"] - agg[f"CLOSE_ROLLING_MEAN_{avg_mean}D"].shift(-1))*10 / agg[f"CLOSE_ROLLING_MEAN_{avg_mean}D"].shift(-1)
                liste_targets.append(f"TARGET_NORMALIZED_{avg_mean}")

                # distance to past averages for volume
                agg[f"VOLUME_NORMALIZED_{avg_mean}"] = (agg["VOLUME_NORMALIZED"] - agg[f"VOLUME_ROLLING_MEAN_{avg_mean}D"].shift(-1))*10 / agg[f"VOLUME_ROLLING_MEAN_{avg_mean}D"].shift(-1)
                
        agg["TARGET_NORMALIZED_MEAN_LAGS"] = agg[liste_targets].mean(axis=1)

        return agg
    

    def distance_to_market(self, dict_full, currency):

        logging.info(f"[{currency}] MARKET DIST")

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

                rollings = self.rolling_window(dict_full[currency][f"MARKET_NORMALIZED_{avg_mean}"], len(self.hours)*float(avg_mean)*self.obs_per_hour)
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

            hours = int(day_future*len(self.hours)*self.obs_per_hour)
            prepared[f"FUTUR_TARGET_{day_future}"] = prepared["CLOSE_NORMALIZED"].rolling(window=hours, center=True).mean().shift(hours)
            prepared[f"BINARY_FUTUR_TARGET_{day_future}"] = 1*(prepared[f"FUTUR_TARGET_{day_future}"] > prepared["CLOSE_NORMALIZED"])
            prepared[f"DELTA_FUTUR_TARGET_{day_future}"] = (prepared[f"FUTUR_TARGET_{day_future}"] - prepared["CLOSE_NORMALIZED"]) *10 / prepared["CLOSE_NORMALIZED"]

        return prepared


    def main_prep_crypto_price(self, datas):

        dict_full = {}
        prepared= {}
        dict_prepared = self.load_prepared()

        for currency in self.currencies:

            df = datas[currency].copy()

            if isinstance(dict_prepared, Dict):
                prepared[currency] = dict_prepared[currency]
                max_date = prepared[currency]["DATE"].max()
                lags = [float(x) for x in self.lags]
                min_date = max_date - timedelta(days= 2*(1+int(np.max(lags))))
                df = df.loc[df["DATE"].between(min_date, max_date)]

                logging.info(f"[DATA PREP] Already prepared up to {max_date} will onlu redo last {np.max(lags)} days")
            else:
                prepared[currency] = pd.DataFrame()

            df = self.pre_clean_cols(df)
            
            # data prep target
            dict_full[currency] = self.prepare_currency_daily(df, currency)

        for currency in self.currencies:
            dict_full[currency] = self.distance_to_market(dict_full, currency)
            dict_full[currency] = self.data_prep_strats(dict_full[currency])

            # TODO add distance to S&P
            # TODO add distance to gold
            # TODO distance to VIX ?

            if prepared[currency].shape[0] > 0:
                dict_full[currency] = pd.concat([prepared[currency], dict_full[currency]], axis=0)
                dict_full[currency] = dict_full[currency].drop_duplicates("DATE")

            for lag in self.lags:
                dict_full[currency] = self.deduce_threshold(dict_full[currency], f"TARGET_NORMALIZED_{lag}", lag=lag)
                
        return dict_full


    def save_prepared(self, dict_prepared):

        utcnow = dict_prepared["BTC"]["DATE"].max()

        # save prepared data 
        self.remove_files_from_dir(self.path_dirs["INTERMEDIATE"])
        pickle.dump(dict_prepared, open("/".join([self.path_dirs["INTERMEDIATE"], f"prepared_{utcnow}.pkl".replace(":", "_")]), 'wb'))


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
