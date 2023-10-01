import pandas as pd
import logging 
import numpy as np
from datetime import datetime, timedelta
import warnings

from data_prep.data_preparation_crypto import PrepareCrytpo 
from modelling.training_strategy2 import TrainingStrat2

warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")


if __name__ == "__main__":
    data_prep = PrepareCrytpo()
    analysis = False

    logging.info("Starting data / strategy execution")
    datas = data_prep.main_load_data()

    # data preparation 
    dict_prepared = data_prep.main_prep_crypto_price(datas)

    # # save datas
    data_prep.save_prepared(dict_prepared, last_x_months=18)

    # training models
    training_step = TrainingStrat2(path_dirs=data_prep.path_dirs, since=2017, oof_start_data=360)  
    training_step.train_all_currencies(dict_prepared)
    
    btc["RDATE"] = btc["DATE"].dt.round("D")
    btc = btc[["RDATE", "CLOSE"]].groupby("RDATE").median().reset_index()

    a = already_there.loc[already_there["sentiment"] != 0][["date", "sentiment"]]
    a["RDATE"] = a["date"].dt.round("D")
    a = a[["RDATE", "sentiment"]].groupby("RDATE").mean().reset_index()

    c = a.merge(btc, how="left")
    c = c.loc[~c["CLOSE"].isnull()]