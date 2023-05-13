import pandas as pd
import time
import logging 
from datetime import datetime, timedelta

from data_prep.data_preparation_crypto import PrepareCrytpo 
from strategy.strategie_1 import MainStrategy 
from data_prep.kraken_portfolio import OrderKraken 
from trading.kraken_trading import TradingKraken

def main():

    # load data
    data_prep = PrepareCrytpo()

    logging.info("Starting data / strategy execution")
    datas = data_prep.load_share_price_data()

    # data preparation 
    prepared = data_prep.aggregate_crypto_price(datas)
    
    # kraken portfolio
    kraken = OrderKraken(configs=data_prep.configs, paths=data_prep.path_dirs)
    df_init = kraken.get_current_portolio() 

    # strategy deduce buy / sell per currency
    strat = MainStrategy(configs=data_prep.configs, 
                        start_date=datetime.utcnow() - timedelta(hours=2),
                        end_date=datetime.utcnow(), 
                        lag="MEAN_LAGS", 
                        fees_buy=0.015, 
                        fees_sell=0.026)
    df_init =  strat.allocate_cash(df_init)
    _, moves_prepared = strat.main_strategy_1_anaysis_currencies(prepared, df_init)

    # pass orders if more than 0
    if moves_prepared.shape[0]>0:
        trading = TradingKraken(configs=data_prep.configs)
        trading.cancel_orders()
        futur_orders = trading.validate_orders(df_init, moves_prepared)

        if len(futur_orders)>0:
            passed_orders = trading.pass_orders(orders=futur_orders)

            if len(passed_orders)>0:
                time.sleep(30)
                orders_infos = trading.get_orders_info(list_id_orders=passed_orders)
            
    # save trades and portfolio positions
    trades = kraken.get_past_trades(prepared)
    pnl_over_time = kraken.pnl_over_time(trades, prepared)

    # save all after clearing infos first in portfolio
    data_prep.remove_files_from_dir(kraken.path_dirs["PORTFOLIO"])
    
    kraken.save_orders(orders_infos)
    kraken.save_df_init(df_init)
    kraken.save_trades(trades)
    kraken.save_pnl(pnl_over_time)
    data_prep.save_prep(datas, prepared)

    logging.info("Finished data / strategy execution")

# if __name__ == "__main__":
#     main()