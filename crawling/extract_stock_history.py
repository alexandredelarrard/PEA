import pandas as pd
import yfinance as yf
from datetime import datetime
from pathlib import Path as pl
from tqdm import tqdm

# Fetch the data
def main_extract_stock(data_mapping_yahoo, sub_loc):

    today = datetime.today().strftime("%Y-%m-%d")
    data_mapping_yahoo = data_mapping_yahoo.loc[~data_mapping_yahoo["YAHOO_CODE"].isnull()]

    for tick in tqdm(data_mapping_yahoo.iloc[:, 1:3].values):
        tick = tuple(tick)
        extract = yf.download(tick[1], '2010-1-1')
        extract.to_csv(sub_loc / pl(f"{tick[0]}/{today}/STOCK.csv"))