import pandas as pd 
from datetime import datetime
import logging
import logging.config as log_config
import os
import shutil
import glob
import ssl
import tqdm
import pickle
import yfinance as yf 

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

    def load_share_price_data(self):

        # urls = self.configs["cryptos_desc"]

        history, nbr_days = self.load_datas()

        if nbr_days:
            print(f"History leveraged with nbr _days = {nbr_days}")
            period=f"{nbr_days}d"
        else:
            period="2y"

        datas = {}
        for currency in tqdm.tqdm(self.currencies):
            # url = urls["URL_BASE"] + "Binance_" + currency + urls["URL_END"]
            # datas[currency] = pd.read_csv(url, delimiter=",", skiprows=[0]) 

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

        # df["RANGE"] = df["HIGH"] - df["LOW"]
        df.rename(columns={"CLOSE" : f"CLOSE_{currency}",
                            # "RANGE" : f"RANGE_{currency}",
                            # "TRADECOUNT" : f"TRADECOUNT_{currency}",
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
            if avg_mean != "MEAN_LAGS":
                agg[f"CLOSE_{currency}_ROLLING_MEAN_{avg_mean}D"] = rolling_mean(agg, f"CLOSE_{currency}", len(self.hours)*avg_mean, "mean")
                agg[f"CLOSE_{currency}_ROLLING_STD_{avg_mean}D"] = rolling_mean(agg, f"CLOSE_{currency}", len(self.hours)*avg_mean, "std")

                # create target normalized to be able to compare 
                agg[f"TARGET_{currency}_NORMALIZED_{avg_mean}"] = (agg[f"CLOSE_{currency}"] - agg[f"CLOSE_{currency}_ROLLING_MEAN_{avg_mean}D"].shift(-1)) / agg[f"CLOSE_{currency}_ROLLING_STD_{avg_mean}D"].shift(-1)
                liste_targets.append(f"TARGET_{currency}_NORMALIZED_{avg_mean}")

        agg[f"TARGET_{currency}_NORMALIZED_MEAN_LAGS"] = agg[liste_targets].mean(axis=1)
        liste_targets.append(f"TARGET_{currency}_NORMALIZED_MEAN_LAGS")

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
            df = self.pre_clean_cols(datas, currency)
            
            # data prep
            df = self.prepare_currency_daily(df, currency)

            if i == 0: 
                full = df
            else:
                full = pd.merge(full, df, on="DATE", how="left", validate="1:1")

        # add distances to others 
        # full = self.distance_to_market(full, currency)
        
        return full


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
            return pickle.load(open(latest_file, 'rb')), nbr_days + 1
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


if __name__ == "__main__":
    data_prep = PrepareCrytpo()