import pandas as pd
import logging 
from datetime import datetime, timedelta
import warnings

from data_prep.data_preparation_crypto import PrepareCrytpo 
from strategy.strategie_0 import Strategy0
from strategy.strategie_1 import Strategy1 
from strategy.strategie_2 import Strategy2
from data_prep.kraken_portfolio import OrderKraken 
from trading.kraken_trading import TradingKraken

warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

def main():

    args = {"lag" : "1"}

    # load data
    data_prep = PrepareCrytpo()
    # self.configs = data_prep.configs

    logging.info("Starting data / strategy execution")
    datas = data_prep.main_load_data()

    # data preparation 
    dict_prepared = data_prep.main_prep_crypto_price(datas)
    
    # kraken portfolio
    kraken = OrderKraken(configs=data_prep.configs, paths=data_prep.path_dirs)
    df_init = kraken.get_current_portolio() 
    current_price = kraken.get_latest_price()

    # strategy deduce buy / sell per currency
    strat = Strategy0(configs=data_prep.configs, 
                        start_date=datetime.utcnow() - timedelta(minutes=30),
                        end_date=datetime.utcnow())
    
    # results = {}
    # for curr in data_prep.currencies:
    #     a, b = strat.main_strategy(dict_prepared, currency=curr)
    #     results[curr] = strat.final_metric
    
    df_init =  strat.allocate_cash(dict_prepared, df_init, 
                                   current_price, 
                                   args=args)
    
    _, moves_prepared = strat.main_strategy_analysis_currencies(dict_prepared, 
                                                                 df_init, 
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

    data_prep.save_datas(datas)
    data_prep.save_prepared(dict_prepared)

    logging.info("Finished data / strategy execution")

if __name__ == "__main__":
    main()