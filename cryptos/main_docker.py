import pandas as pd
import logging 
import numpy as np
from datetime import datetime, timedelta
import warnings

from data_prep.data_preparation_crypto import PrepareCrytpo 
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
    data_prep.save_prepared(dict_prepared, last_x_months=18)

    return dict_prepared


def main(data_prep, dict_prepared):

    args = {}

    # strategy deduce buy / sell per currency
    strat = Strategy2(configs=data_prep.configs, 
                    path_dirs=data_prep.path_dirs,
                    start_date=datetime.utcnow() - timedelta(days=90),
                    end_date=datetime.utcnow())

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
    # main(data_prep, dict_prepared)