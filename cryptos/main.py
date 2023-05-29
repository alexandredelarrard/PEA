import pandas as pd
import logging 
import numpy as np
from datetime import datetime, timedelta
import warnings

from data_prep.data_preparation_crypto import PrepareCrytpo 
from strategy.strategie_0 import Strategy0
from strategy.strategie_1 import Strategy1 
from strategy.strategie_2 import Strategy2
from data_prep.kraken_portfolio import OrderKraken 
from trading.kraken_trading import TradingKraken

warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

def prepare_data(data_prep):

    logging.info("Starting data / strategy execution")
    datas = data_prep.main_load_data()

    # data preparation 
    dict_prepared = data_prep.main_prep_crypto_price(datas)

    # save datas
    data_prep.save_datas(datas)
    data_prep.save_prepared(dict_prepared)

    return dict_prepared


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

    df_init =  strat.allocate_cash(dict_prepared, df_init, 
                                   current_price, 
                                   args=args)
    
    _, moves_prepared = strat.main_strategy_analysis_currencies(dict_prepared, 
                                                                 df_init=None, 
                                                                 args=args)

    # pass orders if more than 0
    orders_infos=pd.DataFrame()
    if moves_prepared.shape[0]>0:
        trading = TradingKraken(configs=data_prep.configs)
        trading.cancel_orders()
        
        if trading.validate_trading_conditions(dict_prepared, df_init):
            futur_orders = trading.validate_orders(df_init, moves_prepared)

            if len(futur_orders)>0:
                logging.info(f"[TRADING][ORDERS TO SEND] {futur_orders}")
                # passed_orders = trading.pass_orders(orders=futur_orders)

                # if len(passed_orders)>0:
                #     orders_infos = trading.get_orders_info(list_id_orders=passed_orders)
                
    # save trades and portfolio positions
    trades, overall = kraken.get_past_trades()
    pnl_over_time = kraken.pnl_over_time(trades)

    # save all after clearing infos first in portfolio
    data_prep.remove_files_from_dir(kraken.path_dirs["PORTFOLIO"])
    
    kraken.save_orders(orders_infos) 
    kraken.save_df_init(df_init)
    kraken.save_trades(trades)
    kraken.save_global_portfolio(overall)
    kraken.save_pnl(pnl_over_time)

    logging.info("Finished data / strategy execution")

if __name__ == "__main__":
    data_prep = PrepareCrytpo()
    dict_prepared = prepare_data(data_prep)
    final, dict = training(data_prep, dict_prepared)    
    # main(data_prep, dict_prepared)

    # NEG = 0.8143231579259071
    # POS = 0.8144980723086368

    # np.mean(list(final["NEG_BINARY_FUTUR_TARGET_2"].values()))
    # np.mean(list(final["POS_BINARY_FUTUR_TARGET_2"].values()))

    # import matplotlib.pyplot as plt
    # for curr in data_prep.currencies:
    #     fig, ax = plt.subplots(figsize=(20,10))
    #     dict[curr][-4000:][["DATE", "CLOSE"]].set_index(["DATE"]).plot(ax = ax, title = curr)
    #     dict[curr][-4000:][["DATE", "POS_BINARY_FUTUR_TARGET_2"]].set_index(["DATE"]).plot(ax = ax, style="--", secondary_y =True)

