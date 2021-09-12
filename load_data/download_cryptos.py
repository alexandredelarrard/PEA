import pandas as pd 
import ssl
import tqdm
from datetime import datetime
ssl._create_default_https_context = ssl._create_unverified_context

# For example: BTC/USD data

urls = {"Cryptos": ["BTC", "ETH", "LTC", "NEO", "BNB", "XRP", "LINK", "EOS", "TRX", 
                    "ETC", "XLM", "ZEC",  "ADA", "QTUM", "DASH", "XMR", "BAT", "BTT", "USDC",
                    "TUSD",  "MATIC", "PAX", "CELR", "ONE", "DOT", "UNI", "ICP", "SOL", "VET",
                    "FIL", "AAVE", "MKR", "ICX", "CVC", "SC", "LRC"],
        "URL_BASE" : "https://www.cryptodatadownload.com/cdd/Binance_",
        "URL_END" : "USDT_1h.csv"}


def load_share_price_data():
    datas = {}
    for CRYPTO in tqdm.tqdm(urls["Cryptos"]):
        url = urls["URL_BASE"] + CRYPTO + urls["URL_END"]
        datas[CRYPTO] = pd.read_csv(url, delimiter=",", skiprows=[0]) 
    return datas


# import yfinance as yf 
# data = yf.download(tickers="LRC-EUR", period="1d", interval="5m")

if __name__ == "__main__":
    datas = load_share_price_data()
