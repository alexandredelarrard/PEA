import warnings
import streamlit as st
import logging

from data_prep.data_preparation_crypto import PrepareCrytpo     
from strategy.strategie_1 import Strategy1 
from strategy.strategie_0 import Strategy0
from strategy.strategie_2 import Strategy2
from data_prep.kraken_portfolio import OrderKraken 
from UI.web_app import MainApp as App

warnings.filterwarnings("ignore")
st.set_page_config(layout="wide")

def main():

    #### initiate app
    data_prep = PrepareCrytpo()
    app = App(st=st, configs=data_prep.configs)
    inputs = app.get_sidebar_inputs()

    logging.info("starting app")
    kraken = OrderKraken(configs = app.configs, paths=data_prep.path_dirs)
    dict_prepared = data_prep.load_prepared()
    
    if app.state.done_once == False:
        
        logging.info("Starting run once")

        # kraken portfolio
        app.state.df_init = kraken.load_df_init() 
        app.state.trades = kraken.load_trades()
        app.state.portfolio = kraken.load_global_portfolio()
        app.state.pnl_over_time = kraken.load_pnl()
        app.prepare_display_portfolio(app.state.df_init, app.state.pnl_over_time, app.state.trades)

        app.state.done_once=True

    if app.state.submitted:

        tab1, tab2, tab3 = app.st.tabs(["COIN_Backtest", "PNL_Backtest", "Portfolio"])

        Strategy = eval(inputs["strategie"])
        strat = Strategy(configs=app.configs, 
                         start_date=inputs['start_date'], 
                         end_date=inputs['end_date'], 
                         path_dirs=data_prep.path_dirs)

        args = {"currency" : inputs["currency"]}
        
        if inputs["init_file"]:
                inputs["init_file"] = app.state.df_init
        else:
            inputs["init_file"] = None

        # display results / analysis
        with tab1:
            prepared = dict_prepared[inputs["currency"]].copy()
            prepared_currency, pnl_currency = strat.main_strategy(prepared, 
                                                                df_init=inputs["init_file"],
                                                                args=args)
            app.display_backtest(dict_prepared, inputs, pnl_currency, prepared_currency, app.state.trades)
        
        with tab2:
            pnl_prepared, _ = strat.main_strategy_analysis_currencies(dict_prepared.copy(), 
                                                                    df_init=inputs["init_file"],
                                                                    args=args)
            app.display_market(pnl_prepared)

        with tab3:
            app.display_portfolio(app.state.portfolio, kraken.get_open_orders())

        logging.info("Finished tables creation")

if __name__ == "__main__":
    main()