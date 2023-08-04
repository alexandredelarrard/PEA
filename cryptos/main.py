import pandas as pd
import logging 
import numpy as np
from datetime import datetime, timedelta
import warnings

from data_prep.data_preparation_crypto import PrepareCrytpo 
from strategy.strategie_2 import Strategy2
from data_prep.kraken_portfolio import OrderKraken 
from modelling.training_strategy2 import TrainingStrat2

warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")


def training(data_prep, dict_prepared):

    args = {"lag" : "2"}
    final ={}
    dict = {}

    for target in [f"NEG_BINARY_FUTUR_TARGET_{args['lag']}",
                   f"POS_BINARY_FUTUR_TARGET_{args['lag']}"]:  
        
        data_prep.configs.strategie["strategy_2"]["TARGET"] = target
        strat = Strategy2(configs=data_prep.configs, 
                        path_dirs=data_prep.path_dirs,
                        start_date=datetime.utcnow() - timedelta(days=90),
                        end_date=datetime.utcnow())
        
        final[strat.target] = {}
        for curr in data_prep.currencies:
            prepared = dict_prepared[curr].copy()
            args["currency"] = curr 
            dict[curr] = strat.main_strategy(prepared, mode="TRAINING", args=args)
            final[strat.target][curr] = np.mean(strat.final_metric)

    return final, dict


def main(data_prep, dict_prepared):

    args = {"lag" : "2"}

    # strategy deduce buy / sell per currency
    strat = Strategy2(configs=data_prep.configs, 
                    path_dirs=data_prep.path_dirs,
                    start_date=datetime.utcnow() - timedelta(days=120),
                    end_date=datetime.utcnow())
    
    import matplotlib.pyplot as plt
    for curr in data_prep.currencies:
        prepared = dict_prepared[curr].copy()
        args["currency"] = curr 
        a, b = strat.main_strategy(prepared, mode="PREDICTING", args=args)

        fig, ax = plt.subplots(figsize=(20,10))
        a[["DATE", "PREDICTION_POS_BINARY_FUTUR_TARGET_0.5", "PREDICTION_NEG_BINARY_FUTUR_TARGET_0.5"]].set_index("DATE").plot(title=curr, ax=ax)
        a[["DATE", "CLOSE"]].set_index("DATE").plot(ax=ax, style="--", secondary_y =True)
        plt.show()

    # kraken portfolio
    kraken = OrderKraken(configs=data_prep.configs, paths=data_prep.path_dirs)
    df_init = kraken.get_current_portolio() 
    current_price = kraken.get_latest_price()


if __name__ == "__main__":
    data_prep = PrepareCrytpo()

    logging.info("Starting data / strategy execution")
    datas = data_prep.main_load_data()

    # data preparation 
    dict_prepared = data_prep.main_prep_crypto_price(datas)

    # # save datas
    data_prep.save_prepared(dict_prepared)

    # training models
    training_step = TrainingStrat2(configs=data_prep.configs, 
                                    path_dirs=data_prep.path_dirs)  
    
    training_step.main_training(dict_prepared["BTC"], args={"currency" : "BTC", "oof_days" : 120})
