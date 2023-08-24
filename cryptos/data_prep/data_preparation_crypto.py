import pandas as pd 
import numpy as np
import logging
import os
import shutil
import glob
import ssl
import pickle
from typing import Dict
from datetime import timedelta, datetime
import math
import scipy.ndimage as ndi
from scipy.stats import truncnorm, uniform, norm
from tqdm import tqdm

from data_load.data_loading import LoadCrytpo

pd.options.mode.chained_assignment = None 

ssl._create_default_https_context = ssl._create_unverified_context

def round_dt(dt, delta):
    return datetime.min + math.floor((dt - datetime.min) / delta) * delta

class PrepareCrytpo(LoadCrytpo):

    def __init__(self):

        LoadCrytpo.__init__(self)

        self.hours = range(24)
        self.obs_per_hour = 60//self.granularity #4 tous les quart d'heure
        self.target_depth = 8
        
        # init with app variables 
        self.lags = range(1, 24*2*self.obs_per_hour)
        self.market_makers_currencies = ["BTC", "ETH", "ADA", "XRP"]

        self.normalization_days = 200
    

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
    
    def prepare_currency_daily(self, df):

        df["DATE"] = pd.to_datetime(df["DATE"], format="%Y-%m-%d %H:%M:%S")
        df = df.sort_values(["DATE"], ascending = False)
        df = df.drop_duplicates("DATE")

        # create currency target 
        agg = df[["DATE", "CLOSE", "VOLUME"]]

        #round the earliest value date to the closest 15 minutes 
        agg.iloc[0,0] = agg.iloc[0,0] + timedelta(minutes=self.granularity//2)
        agg.iloc[0,0] = agg.iloc[0,0] - timedelta(minutes=agg.iloc[0,0].minute % self.granularity, 
                                                seconds=agg.iloc[0,0].second %60)

        agg = agg.drop_duplicates("DATE")

        # remove mvs 
        agg.loc[agg["CLOSE"] <= 0, "CLOSE"] = np.nan
        agg = agg.loc[~agg["CLOSE"].isnull()]

        for i in range(1, self.target_depth+1):
            agg[f"TARGET_CLOSE_{i}"] = agg["CLOSE"].shift(i)

        # rolling mean distance to X.d
        for lag in self.lags:
            agg[f"CLOSE_{lag}"] = agg["CLOSE"].shift(-lag)
            agg[f"VOLUME_{lag}"] = agg["VOLUME"].shift(-lag)

        agg = agg.loc[~agg[f"CLOSE_{lag}"].isnull()]

        return agg
    
    def prepare_features(self, df):

        for means in tqdm(list(range(1, max(self.lags) + 1, 4))):
            df[f"MEAN_CLOSE_{means}"] = df[[f"CLOSE_{x}" for x in range(1, means+1)]].mean(axis=1)
            df[f"STD_CLOSE_{means}"] = df[[f"CLOSE_{x}" for x in range(1, means+1)]].std(axis=1)
            df[f"MEAN_VOLUME_{means}"] = df[[f"VOLUME_{x}" for x in range(1, means+1)]].mean(axis=1)
            df[f"STD_VOLUME_{means}"] = df[[f"VOLUME_{x}" for x in range(1, means+1)]].std(axis=1)
    
            #delta calculations : 
            df[f"DELTA_CLOSE_MEAN_{means}"] = (df["CLOSE"] - df[f"MEAN_CLOSE_{means}"]) / (df[f"MEAN_CLOSE_{means}"])
            df[f"DELTA_VOLUME_MEAN_{means}"] = (df["CLOSE"] - df[f"MEAN_VOLUME_{means}"]) / (df[f"MEAN_VOLUME_{means}"])

        for lag in range(2, 25):
            df[f"DELTA_CLOSE_{lag}"] =(df["CLOSE"] - df[f"CLOSE_{lag}"]) / (df[f"CLOSE_{lag}"])

        return df 
    
    def prepare_target(self, df):

        for i in range(1, self.target_depth+1):
            df[f"DELTA_TARGET_{i}"] =  (df[f"TARGET_CLOSE_{i}"]- df["CLOSE"]) /  df["CLOSE"]
        
        return df

    def distance_to_market(self, dict_full, currency):

        logging.info(f"[{currency}] MARKET DIST")

        for lag in self.lags:

            # average % increase decrease per lag
            for curr in self.market_makers_currencies:
                df_to_merge = dict_full[curr][["DATE", f"CLOSE_TO_{lag}_{self.granularity}"]]
                df_to_merge = df_to_merge.rename(columns={f"CLOSE_TO_{lag}_{self.granularity}" : f"TARGET_{curr}"})
                dict_full[currency] = dict_full[currency].merge(df_to_merge, on="DATE", how="left", validate="1:1")
            
            cols= [f"CLOSE_TO_{x}_{self.granularity}" for x in self.market_makers_currencies]
            dict_full[currency][f"MARKET_TO_{lag}_{self.granularity}"] = dict_full[currency][cols].mean(axis=1)
            dict_full[currency] = dict_full[currency].drop(cols, axis=1)
            
            # defined as p.p up or down to market % variation average
            dict_full[currency][f"CLOSE_TO_MARKET_{lag}_{self.granularity}"] = (dict_full[currency][f"CLOSE_TO_{lag}_{self.granularity}"] - dict_full[currency][f"MARKET_TO_{lag}_{self.granularity}"])*10
        
        dict_full[currency] = dict_full[currency].sort_values("DATE", ascending=False)

        return dict_full[currency]
    

    def distance_to_others(self, datas, dict_full, currency):

        logging.info(f"[{currency}] OTHERS DIST")

        # merge others to data 
        for other in ["GOLD", "S&P", "BRENT"]:
            datas[other] = datas[other][["DATE", "CLOSE"]]
            datas[other] = datas[other].sort_values(["DATE"], ascending= 0)

            for avg_mean in [7, 30, 90]:
                rollings = self.rolling_window(datas[other]["CLOSE"], float(avg_mean))
                datas[other][f"CLOSE_MEAN_{avg_mean}D"] = np.mean(rollings, axis=1)

                datas[other][f"{other}_NORMALIZED_{avg_mean}"] = (datas[other]["CLOSE"] - datas[other][f"CLOSE_MEAN_{avg_mean}D"].shift(-1))*10 / datas[other][f"CLOSE_MEAN_{avg_mean}D"].shift(-1)
        
            dict_full[currency] = dict_full[currency].merge(datas[other].drop("CLOSE", axis=1), on="DATE", how="left", validate="1:1") 

        for other in ["S&P", "GOLD", "BRENT"]:
            for avg_mean in [7, 30, 90]:
                dict_full[currency][f"{other}_NORMALIZED_{avg_mean}"] = dict_full[currency][f"{other}_NORMALIZED_{avg_mean}"].bfill()
                dict_full[currency][f"DIFF_{other}_NORMALIZED_{avg_mean}"] = dict_full[currency][f"TARGET_NORMALIZED_{avg_mean}"] - dict_full[currency][f"{other}_NORMALIZED_{avg_mean}"]

        return dict_full[currency]


    def main_prep_crypto_price(self, datas):

        dict_full = {}
        prepared= {}
        dict_prepared = self.load_prepared()

        for currency in self.currencies:

            df = datas[currency].copy()

            if isinstance(dict_prepared, Dict):
                if currency in dict_prepared.keys():
                    prepared[currency] = dict_prepared[currency]
                    max_date = prepared[currency]["DATE"].max()
                    lags = [float(x) for x in self.lags]
                    min_date = max_date - timedelta(days = (np.max(lags)//self.obs_per_hour)//24)
                    df = df.loc[df["DATE"] >= min_date]

                    logging.info(f"[DATA PREP] Already prepared up to {max_date} will only redo last {np.max(lags)} days")
            else:
                prepared[currency] = pd.DataFrame()

            # data prep target
            logging.info(f"[{currency}] CLOSE DIST")
            df = self.prepare_currency_daily(df)
            df = self.prepare_features(df)
            df = self.prepare_target(df)

            # dropp all close / volume 
            closes = [f"CLOSE_{x}" for x in self.lags]
            vols = [f"VOLUME_{x}" for x in self.lags]
            df = df.drop(closes + vols, axis=1)
            
            dict_full[currency] = df

        for currency in self.currencies:
            # dict_full[currency] = self.distance_to_market(dict_full, currency)
            # dict_full[currency] = self.distance_to_others(datas, dict_full, currency) 

            if prepared[currency].shape[0] > 0:
                dict_full[currency] = pd.concat([prepared[currency], dict_full[currency]], axis=0)
                dict_full[currency] = dict_full[currency].drop_duplicates("DATE")
                dict_full[currency] = dict_full[currency].sort_values("DATE", ascending=False)

        return dict_full


    def save_prepared(self, dict_prepared, last_x_months=None):

        utcnow = dict_prepared["BTC"]["DATE"].max()

        # save prepared data 
        self.remove_files_from_dir(self.path_dirs["INTERMEDIATE"])
        pickle.dump(dict_prepared, open("/".join([self.path_dirs["INTERMEDIATE"], f"prepared_{utcnow}_all.pkl".replace(":", "_")]), 'wb'))

        if last_x_months:
            pivot_date = datetime.today() - timedelta(days=last_x_months*30)
            for currency in self.currencies:
                dict_prepared[currency] = dict_prepared[currency].loc[dict_prepared[currency]["DATE"]>=pivot_date]
            pickle.dump(dict_prepared, open("/".join([self.path_dirs["INTERMEDIATE"], f"prepared_{utcnow}_{last_x_months}_months.pkl".replace(":", "_")]), 'wb'))


    def load_prepared(self, last_x_months=None):

        if last_x_months:
            list_of_files = glob.glob(self.path_dirs["INTERMEDIATE"]+"/*_months.pkl")
        else:
            list_of_files = glob.glob(self.path_dirs["INTERMEDIATE"]+"/*_all.pkl")

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
