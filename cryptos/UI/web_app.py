from datetime import date, datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import altair as alt
import pandas as pd 

from dotenv import load_dotenv
from utils.general_functions import load_configs

class MainApp(object):

    def __init__(self, st, lags):

        load_dotenv("./configs/.env")

        # init with app variables 
        self.configs = load_configs("configs.yml") 

        self.st = st
        self.currencies = self.configs["cryptos_desc"]["Cryptos"]
        self.state = self.init_state()
        self.lags = lags

    def init_state(self):

        state = self.st.session_state

        if "submitted" not in state:
            state.submitted = False

        if "done_once" not in state:
            state.done_once = False

        if "prepared" not in state:
            state.prepared = None

        if "df_init" not in state:
            state.df_init = None

        if "trades" not in state:
            state.trades = None

        if "pnl_over_time" not in state:
            state.pnl_over_time = None

        return state

    def get_sidebar_inputs(self):

        inputs = {"currency" : None,
                "lags" : None,
                "start_date" : None,
                "end_date" : None,
                'fees' : None,
                "button" : None,
                "init_file" : None}
        
        self.st.title("Crypto portfolio")

        self.st.sidebar.header("What initial values to backtest / prospect from ?")

        form = self.st.sidebar.form("my_form")

        inputs["init_file"] = form.checkbox('Use Kraken current position')

        inputs["currency"] = form.selectbox('Select crypto', np.sort(self.currencies))
        inputs["start_date"] = form.date_input("When to start the backtest ?", date(2023,1,1))
        inputs["end_date"] = form.date_input("When to end the backtest ?", date.today())
        
        inputs["lag"] = form.selectbox('Select lags target', self.lags)
        # inputs["fees_buy"] = form.slider("Select transaction fees in % for buy", 0, 15, 1.4)
        inputs["fees_buy"] = 1.4/100

        # inputs["fees_sell"] = form.slider("Select transaction fees in % for sell", 0, 15, 2.6)
        inputs["fees_sell"] = 2.6/100

        inputs["button"] = form.form_submit_button("Run Analysis", on_click=lambda: self.state.update(submitted=True))

        return inputs


    def display_backtest(self, inputs, pnl_currency, prepared_currency):
        
        fig, ax = plt.subplots(figsize=(20,10)) 
        prepared_currency[["DATE", f"REAL_BUY_SELL"]].set_index(["DATE"]).plot(ax = ax)
        prepared_currency[["DATE", f"TARGET_{inputs['currency']}_NORMALIZED_{inputs['lag']}"]].set_index(["DATE"]).plot(ax = ax)
        prepared_currency[["DATE", f"CLOSE_{inputs['currency']}"]].set_index(["DATE"]).plot(ax = ax, secondary_y =True)
        self.st.pyplot(fig)

        max_date = self.state.prepared["DATE"].max()
        value = self.state.prepared.loc[self.state.prepared["DATE"] == max_date, f"CLOSE_{inputs['currency']}"].values[0]
        self.st.header(f"SPOT PRICE {inputs['currency']} || {max_date} || {value:.2f}")
        self.st.line_chart(self.state.prepared[["DATE", f"CLOSE_{inputs['currency']}"]].set_index("DATE"))

        self.st.header(f"PNL for currency : {inputs['currency']}")
        self.st.line_chart(pnl_currency.set_index("DATE"))


    def display_portfolio(self, df_init, trades, pnl_over_time):

        def color_survived(val):
            return ['color: green']*len(val) if val.MARGIN_EUR >0 else ['color: red']*len(val)

        col1, col2= self.st.columns([4,2])
        
        # display historical trades (+/- value sur positions ouvertes)
        col1.header(f"Trades history")
        col1.dataframe(trades.style.apply(color_survived, axis=1), use_container_width=True)

        # composition of portfolio on latest date
        df_init = df_init[self.currencies + ["CASH"]]
        df_init = df_init.astype(float)
        labels = list(df_init.columns)

        get_price = self.state.prepared.iloc[0:3]

        for currency in self.currencies:
            df_init[currency] = df_init[currency]*get_price["CLOSE_" + currency].bfill()[0]
        
        fig1, ax1 = plt.subplots()
        ax1.pie(df_init.iloc[0], labels=labels, autopct='%1.1f%%',
                shadow=True, startangle=90)
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        col2.header(f"Portfolio composition = {df_init.iloc[0].sum():.2f}")
        col2.pyplot(fig1)
        
        # display split PNL over time 
        pnl_over_time = pnl_over_time.loc[pnl_over_time["DATE"] > "2022-12-01"]

        self.st.header("Split Portfolio PNL evolution")
        split_pnl = pd.DataFrame(columns=["DATE", "PNL", "CURRENCY"])

        for currency in self.currencies:
            sub_pnl = pnl_over_time[["DATE", f"PNL_{currency}"]]
            sub_pnl["CURRENCY"] = currency 
            sub_pnl.rename(columns={f"PNL_{currency}": "PNL"}, inplace=True)
            split_pnl =pd.concat([split_pnl, sub_pnl], axis=0)

        chart = alt.Chart(split_pnl).mark_line(point=alt.OverlayMarkDef(color="red")).encode(
            x='DATE',
            y=alt.Y('PNL', scale=alt.Scale(domain=[split_pnl["PNL"].min()*0.95, 
                                                    split_pnl["PNL"].max()*1.05])),
            color="CURRENCY"
        ).interactive()
        self.st.altair_chart(chart, theme="streamlit", use_container_width=True)

        # display global PNL
        self.st.header("Overall Portfolio PNL evolution")
        chart = alt.Chart(pnl_over_time).mark_line(point=alt.OverlayMarkDef(color="red")).encode(
            x='DATE',
            y=alt.Y('PNL_PORTFOLIO', scale=alt.Scale(domain=[pnl_over_time["PNL_PORTFOLIO"].min()*0.95, 
                                                            pnl_over_time["PNL_PORTFOLIO"].max()*1.05]))
        ).interactive()
        self.st.altair_chart(chart, theme="streamlit", use_container_width=True)

        
    def display_market(self,  pnl_prepared, moves_prepared):

        col1, col2 = self.st.columns([4,2])

        # dates to buy or sell per crypto (still in df init) 
        moves_prepared = moves_prepared.loc[moves_prepared["REAL_BUY_SELL"] !=0]
        moves_prepared = moves_prepared[["DATE", "REAL_BUY_SELL", "CURRENCY"]]
        col1.dataframe(moves_prepared, use_container_width=True)

        # split PNL
        self.st.header("Split Portfolio PNL evolution")
        split_pnl = pd.DataFrame(columns=["DATE", "PNL", "CURRENCY"])

        for currency in self.currencies:
            sub_pnl = pnl_prepared[["DATE", f"PNL_{currency}"]]
            sub_pnl["CURRENCY"] = currency 
            sub_pnl.rename(columns={f"PNL_{currency}": "PNL"}, inplace=True)
            split_pnl =pd.concat([split_pnl, sub_pnl], axis=0)

        chart = alt.Chart(split_pnl).mark_line(point=alt.OverlayMarkDef(color="red")).encode(
            x='DATE',
            y=alt.Y('PNL', scale=alt.Scale(domain=[split_pnl["PNL"].min()*0.95, 
                                                    split_pnl["PNL"].max()*1.05])),
            color="CURRENCY"
        ).interactive()
        self.st.altair_chart(chart, theme="streamlit", use_container_width=True)

        # overall PNL
        self.st.header("Overall Portfolio PNL evolution")
        chart = alt.Chart(pnl_prepared).mark_line(point=alt.OverlayMarkDef(color="red")).encode(
            x='DATE',
            y=alt.Y('PNL_PORTFOLIO', scale=alt.Scale(domain=[pnl_prepared["PNL_PORTFOLIO"].min()*0.95, 
                                                            pnl_prepared["PNL_PORTFOLIO"].max()*1.05]))
        ).interactive()
        self.st.altair_chart(chart, theme="streamlit", use_container_width=True)

