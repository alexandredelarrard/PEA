import warnings
import pandas as pd
import streamlit as st

from data_prep.data_preparation_crypto import PrepareCrytpo 
from strategy.strategie_1 import MainStrategy 
from data_prep.kraken_portfolio import OrderKraken 
from UI.web_app import MainApp as App

warnings.filterwarnings("ignore")
st.set_page_config(layout="wide")

def main():

    lags = [7, 15, 30, 45]

    #### initiate app
    app = App(st=st, lags=lags)
    inputs = app.get_sidebar_inputs()

    if not app.state.done_once:

        data_prep = PrepareCrytpo(configs = app.configs, lags=lags)
        datas = data_prep.load_share_price_data()

        # data preparation 
        app.state.prepared = data_prep.aggregate_crypto_price(datas)
        app.state.done_once=True
        
        # kraken portfolio
        kraken = OrderKraken(configs = app.configs)
        app.state.df_init = kraken.get_current_portolio() 
        app.state.trades = kraken.get_past_trades(app.state.prepared)
        app.state.pnl_over_time = kraken.pnl_over_time(app.state.trades, app.state.prepared)

    if app.state.submitted:

        tab1, tab2, tab3 = st.tabs(["Coin", "Portfolio", "Trading"])

        kraken = OrderKraken(configs = app.configs)
        strat = MainStrategy(configs=app.configs, 
                            start_date=inputs["start_date"], 
                            end_date=inputs["end_date"], 
                            lag=inputs["lag"], 
                            fees_buy=inputs["fees_buy"], 
                            fees_sell=inputs["fees_sell"])

        # display results / analysis
        with tab1:
            if inputs["init_file"]:
                inputs["init_file"] = app.state.df_init
            else:
                inputs["init_file"] = None
            prepared_currency, pnl_currency = strat.main_strategie_1(app.state.prepared, currency = inputs["currency"])
            pnl_currency = strat.strategy_1_lags_comparison(app.state.prepared, currency = inputs["currency"], lags=lags)
            app.display_backtest(inputs, pnl_currency, prepared_currency)

        with tab2:
            app.display_portfolio(app.state.df_init, app.state.trades, app.state.pnl_over_time)

        with tab3:
            pnl_prepared, moves_prepared = strat.main_strategy_1_anaysis_currencies(app.state.prepared)
            app.display_market(pnl_prepared, moves_prepared)

if __name__ == "__main__":
    main()