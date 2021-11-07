import pandas as pd 
import ssl
import tqdm
from datetime import datetime
ssl._create_default_https_context = ssl._create_unverified_context

# For example: BTC/USD data

def load_share_price_data(configs):

    urls = configs["cryptos_desc"]

    datas = {}
    for CRYPTO in tqdm.tqdm(urls["Cryptos"]):
        url = urls["URL_BASE"] + CRYPTO + urls["URL_END"]
        datas[CRYPTO] = pd.read_csv(url, delimiter=",", skiprows=[0]) 

    return datas


# import yfinance as yf 
# data = yf.download(tickers="BTC-EUR", period="1000d", interval="1h")

if __name__ == "__main__":
    datas = load_share_price_data()
