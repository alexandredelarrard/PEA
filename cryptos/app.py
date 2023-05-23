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
    dict_prepared = data_prep.load_prepared()

    kraken = OrderKraken(configs = app.configs, paths=data_prep.path_dirs)
    
    if app.state.done_once == False:

        logging.info("Starting run once")
        app.state.dict_prepared = dict_prepared
        
        # kraken portfolio
        app.state.df_init = kraken.load_df_init() 
        app.state.trades = kraken.load_trades()
        app.state.portfolio = kraken.load_global_portfolio()
        app.state.pnl_over_time = kraken.load_pnl()
        app.prepare_display_portfolio(app.state.df_init, app.state.pnl_over_time, app.state.trades)

        app.state.done_once=True

    if app.state.submitted:

        tab1, tab2, tab3 = app.st.tabs(["COIN_Backtest", "PNL_Backtest", "Portfolio"])
        prepared = app.state.dict_prepared[inputs["currency"]].copy()

        Strategy = eval(inputs["strategie"])
        strat = Strategy(configs=app.configs, start_date=inputs['start_date'], end_date=inputs['end_date'], dict_prepared=dict_prepared)

        if inputs["strategie"] == "Strategy0":
        #     best = strat.deduce_parameters(app.state.dict_prepared)
            st.write(f"Returned lag/futur/seuil = {strat.parameters}")

        args = {"lag" : inputs["lag"],
                "currency" : inputs["currency"]}
        
        with tab2:
            pnl_prepared, _ = strat.main_strategy_analysis_currencies(app.state.dict_prepared, 
                                                                        df_init=inputs["init_file"],
                                                                        args=args)
            app.display_market(pnl_prepared)

        # display results / analysis
        with tab1:
            if inputs["init_file"]:
                inputs["init_file"] = app.state.df_init
            else:
                inputs["init_file"] = None

            prepared_currency, pnl_currency = strat.main_strategy(prepared, 
                                                                    df_init=inputs["init_file"],
                                                                    args=args)
            
            pnl_currency = strat.strategy_lags_comparison(prepared,
                                                            df_init=inputs["init_file"],
                                                            args=args)
            
            app.display_backtest(inputs, pnl_currency, prepared_currency, app.state.trades)
        
        with tab3:
            app.display_portfolio(app.state.portfolio, kraken.get_open_orders())

        logging.info("Finished tables creation")

if __name__ == "__main__":
    main()