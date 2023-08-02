import pandas as pd 
from datetime import datetime
import logging.config as log_config
import logging
import os
import shutil
import glob
import ssl
import pickle
import yfinance as yf 
import urllib
import zipfile
import numpy as np

from kraken.spot import Market

pd.options.mode.chained_assignment = None 

from utils.general_functions import smart_column_parser
from utils.config import Config
from dotenv import load_dotenv

ssl._create_default_https_context = ssl._create_unverified_context

class LoadCrytpo(object):

    def __init__(self):

        load_dotenv("./configs/.env")
        
        # init with app variables 
        self.granularity = 15
        self.configs = self.config_init("./configs/main.yml") 
        self.currencies = self.configs.load["cryptos_desc"]["Cryptos"]

        log_config.dictConfig(self.configs.logging)

        self.create_directories()

    def config_init(self, config_path):
        return Config(config_path).read()

    def create_directories(self):

        self.path_dirs = {}
        self.path_dirs["BASE"] = "./data"

        self.path_dirs["HISTORY"] = "/".join([self.path_dirs["BASE"], "currencies", "Kraken_OHLCVT"])
        if not os.path.isdir(self.path_dirs["HISTORY"]):
            os.mkdir(self.path_dirs["HISTORY"])

        self.path_dirs["HISTORY_QUARTER"] = "/".join([self.path_dirs["BASE"], "currencies", "Kraken_OHLCVT_Q"])
        if not os.path.isdir(self.path_dirs["HISTORY_QUARTER"]):
            os.mkdir(self.path_dirs["HISTORY_QUARTER"])

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

        self.path_dirs["MODELS"] = "/".join([self.path_dirs["BASE"], "models"])
        if not os.path.isdir(self.path_dirs["MODELS"]):
            os.mkdir(self.path_dirs["MODELS"])

        self.path_dirs["PLOTS"] = "/".join([self.path_dirs["BASE"], "plots"])
        if not os.path.isdir(self.path_dirs["PLOTS"]):
            os.mkdir(self.path_dirs["PLOTS"])
        if not os.path.isdir(self.path_dirs["PLOTS"] + "/shap"):
            os.mkdir(self.path_dirs["PLOTS"] + "/shap")


    def download_history_kraken(self):

        file_path = "data/Kraken_OHLCVT_delta.zip"

        # full history 
        url = "https://drive.google.com/file/d/1YfZxX4HkRFzSWKGdFpLirpDG6XpMU-K7/view?usp=share_link"
        urllib.request.urlretrieve(url, "data/Kraken_OHLCVT_full.zip")

        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(file_path)

    def download_update_kraken(self):
        # TODO : download automatic data from zip files quarterly

        file_path = "data/Kraken_OHLCVT_delta.zip"

        # update history 
        url = "https://googledrive.com/host/uc?export=download&confirm=t&id=1CGtFpsU0Mc1gwtl8f6zudvBEOx81gpmR"
        urllib.request.urlretrieve(url, file_path)

        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(file_path)


    def load_share_price_data(self, currency, period):

        logging.info(f"Days to extract with yahoo = {period}")

        pair = f"{currency}-EUR"
        if currency == "STX":
            pair = "STX4847-EUR"
        
        # load intra day data 
        intra_day_data = yf.download(tickers=pair, period=period, interval= str(self.granularity) + "m")
        intra_day_data = intra_day_data.reset_index()
        intra_day_data.rename(columns={"index" : "Date", "Datetime" : "Date"}, inplace=True)
        intra_day_data = intra_day_data.drop("Adj Close", axis=1)
        intra_day_data["Date"] = pd.to_datetime(intra_day_data["Date"].dt.strftime("%Y-%m-%d %H:%M:%S"), format="%Y-%m-%d %H:%M:%S", utc=False)
        intra_day_data.columns = smart_column_parser(intra_day_data.columns)
        intra_day_data["TRADECOUNT"] = np.nan

        return intra_day_data
    
    
    def load_other_daily(self, pair):

        # "ES=F" = s&p ;  "BZ=F" : "brent"
        hist = yf.download(pair, period="7y", interval="1d")
        hist = hist.reset_index()
        hist.rename(columns={"index" : "Date", "Datetime" : "Date"}, inplace=True)
        hist["Date"] = pd.to_datetime(hist["Date"], format="%Y-%m-%d")
        hist.columns = smart_column_parser(hist.columns)

        return hist


    def load_ohlctv(self, currency):

        kraken = Market()
        intra_day_data = kraken.get_ohlc(pair=f"{currency}EUR", interval=self.granularity)
        intra_day_data = pd.DataFrame(intra_day_data[next(iter(intra_day_data))])
        intra_day_data.columns = smart_column_parser(intra_day_data.columns)
        intra_day_data.columns = ["DATE", "OPEN", "HIGH", "LOW", "CLOSE_1", "CLOSE", "VOLUME", "TRADECOUNT"]
        intra_day_data["DATE"] = pd.to_datetime(intra_day_data["DATE"], unit='s')

        return intra_day_data
    
    def load_datas(self):

        list_of_files = glob.glob(self.path_dirs["CURRENCY"]+"/*.pkl") 
        if len(list_of_files)>0:
            latest_file = max(list_of_files, key=os.path.getctime)
            ti_c = os.path.getctime(latest_file)
            nbr_days = (datetime.today() - datetime.fromtimestamp(ti_c)).days
            return pickle.load(open(latest_file, 'rb')), nbr_days + 60
        else:
            return None, None
        
    def load_history_data(self, currency, root_path="HISTORY"):

        if currency == "BTC":
            currency = "XBT"
        
        if currency == "DOGE":
            currency = "XDG"

        list_of_files = glob.glob(self.path_dirs[root_path]+f"/{currency}EUR_{self.granularity}*.csv") 
        if len(list_of_files)>0:
            latest_file = max(list_of_files, key=os.path.getctime)
            df = pd.read_csv(latest_file, header=None)
            df.columns = ["DATE", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME", "TRADECOUNT"]
            df["DATE"] = pd.to_datetime(df["DATE"], unit='s')
            date_max = df["DATE"].max()
            return df, date_max
        else:
            return None, None
        
    def concatenate_histories(self, currency):
        
        history_data, max_date = self.load_history_data(currency, root_path="HISTORY")
        history_data_q, max_date = self.load_history_data(currency, root_path="HISTORY_QUARTER")

        history = pd.concat([history_data, history_data_q], axis=0)
        history = history.drop_duplicates("DATE")
        history = history.sort_values("DATE", ascending=1)

        period = str((datetime.today() - max_date).days + 1) + "d"

        return history, period

    def main_load_data(self):

        datas= {}

        for currency in self.currencies:

            history_data, period = self.concatenate_histories(currency)
            data = self.load_share_price_data(currency, period)

            datas[currency] = pd.concat([history_data, data], axis=0)

        # other currencies
        datas["S&P"] = self.load_other_daily(pair="ES=F")
        datas["BRENT"] = self.load_other_daily(pair="BZ=F")
        datas["GOLD"] = self.load_other_daily(pair="GC=F")

        return datas


    def save_datas(self, datas):

        utcnow = datas["BTC"]["DATE"].max()

        # save currencies 
        self.remove_files_from_dir(self.path_dirs["CURRENCY"])
        pickle.dump(datas, open("/".join([self.path_dirs["CURRENCY"], f"cryptos_{utcnow}.pkl".replace(":", "_")]), 'wb'))

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
