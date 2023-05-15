import warnings
import streamlit as st
import logging

from data_prep.data_preparation_crypto import PrepareCrytpo 
from strategy.strategie_1 import MainStrategy 
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
    prepared = data_prep.load_prepared()
    
    if app.state.done_once == False:

        logging.info("Starting run once")

        app.state.prepared = prepared
        # kraken portfolio
        kraken = OrderKraken(configs = app.configs, paths=data_prep.path_dirs)
        app.state.df_init = kraken.load_df_init() 
        app.state.trades = kraken.load_trades()
        app.state.pnl_over_time = kraken.load_pnl()

        app.prepare_display_portfolio(app.state.df_init, app.state.pnl_over_time)

        app.state.done_once=True

    if app.state.submitted:

        tab1, tab2, tab3 = app.st.tabs(["COIN_Backtest", "PNL_Backtest", "Portfolio"])

        strat = MainStrategy(configs=app.configs, 
                            start_date=inputs["start_date"], 
                            end_date=inputs["end_date"])

        # display results / analysis
        with tab1:
            if inputs["init_file"]:
                inputs["init_file"] = app.state.df_init
                df_init = strat.allocate_cash(app.state.prepared, inputs["init_file"])
            else:
                inputs["init_file"] = None

            prepared_currency, pnl_currency = strat.main_strategy_1(app.state.prepared, currency=inputs["currency"], 
                                                                    lag=inputs["lag"], df_init=inputs["init_file"])
            
            sub_prepare = app.state.prepared.loc[app.state.prepared["DATE"].between(strat.start_date, strat.end_date)]
            pnl_currency = strat.strategy_1_lags_comparison(sub_prepare, currency = inputs["currency"], 
                                                            df_init=inputs["init_file"])
            
            app.display_backtest(inputs, pnl_currency, prepared_currency)

        with tab2:
            pnl_prepared, moves_prepared = strat.main_strategy_1_anaysis_currencies(app.state.prepared, 
                                                                                    lag=inputs["lag"],
                                                                                    df_init=inputs["init_file"])
            app.display_market(pnl_prepared, moves_prepared)

        with tab3:
            app.display_portfolio(app.state.trades)

        logging.info("Finished tables creation")

if __name__ == "__main__":
    main()