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
    
    if app.state.done_once == False:
        
        logging.info("Starting run once")

        # kraken portfolio
        app.state.df_init = kraken.load_df_init() 
        app.state.trades = kraken.load_trades()
        app.state.portfolio = kraken.load_global_portfolio()
        app.state.pnl_over_time = kraken.load_pnl()
        dict_prepared = data_prep.load_prepared(last_x_months=18)
        # app.prepare_display_portfolio(app.state.df_init, app.state.pnl_over_time, app.state.trades)

        # display results / analysis
        logging.info(f"start date {inputs['start_date']} & end date {inputs['end_date']}")
        # app.state.pnl_prepared = pnl_prepared
        # app.state.moves = moves
        app.state.done_once=True
        app.state.dict_prepared = dict_prepared

        logging.info("finished run once")

    if app.state.submitted:

        tab1, tab2 = app.st.tabs(["COIN_Backtest", "FULL_Backtest"]) 
        
        if inputs["init_file"]:
                inputs["init_file"] = app.state.df_init
        else:
            inputs["init_file"] = None

        Strategy = eval(inputs["strategie"])
        strat = Strategy(configs=app.configs, 
                         start_date=inputs['start_date'], 
                         end_date=inputs['end_date'], 
                         path_dirs=data_prep.path_dirs)
        pnl_prepared, moves = strat.main_strategy_analysis_currencies(app.state.dict_prepared.copy(), 
                                                                        df_init=inputs["init_file"])
        
        with tab1:
            app.display_market(pnl_prepared)    
    
        with tab2:
            currency = inputs['currency']
            pnl_currency = pnl_prepared[["DATE", f"PNL_{currency}", f"PNL_BASELINE_{currency}"]]
            prepared_currency = moves.loc[moves["CURRENCY"] == currency][["DATE", "PRICE", "REAL_BUY_SELL", "PREDICTION_BNARY_TARGET_UP", "PREDICTION_BNARY_TARGET_DOWN"]].rename(columns={"PRICE" : "CLOSE"})
            prepared_currency["PREDICTION_BNARY_TARGET_DOWN"] = -1*prepared_currency["PREDICTION_BNARY_TARGET_DOWN"]
            app.display_backtest(app.state.dict_prepared, inputs, pnl_currency, prepared_currency, app.state.trades)

        # with tab3:
        #     app.display_portfolio(app.state.portfolio, kraken.get_open_orders())

        logging.info("Finished tables creation")

if __name__ == "__main__":
    main()