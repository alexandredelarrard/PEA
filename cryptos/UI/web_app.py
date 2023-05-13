from datetime import date
import numpy as np
import matplotlib.pyplot as plt
import altair as alt
import pandas as pd 

class MainApp(object):

    def __init__(self, st, configs):
        
        self.configs = configs
        self.st = st
        self.currencies = self.configs.load["cryptos_desc"]["Cryptos"]
        self.lags = self.configs.load["cryptos_desc"]["LAGS"]

        self.init_state()

    def init_state(self):

        self.state = self.st.session_state

        if "submitted" not in self.state:
            self.state.submitted = False

        if "prepared" not in self.state:
            self.state.prepared = None

        if "prepared" not in self.state:
            self.state.pnl_over_time= None 

        if "prepared" not in self.state:
            self.state.trades= None 

        if "prepared" not in self.state:
            self.state.df_init= None 

        if "prepared" not in self.state:
            self.state.fig1 = None

        if "prepared" not in self.state:
            self.state.chart1 = None

        if "prepared" not in self.state:
            self.state.chart2 = None

        if "prepared" not in self.state:
            self.state.total_portfolio = None

        if "prepared" not in self.state:
            self.state.done_once = False

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
        inputs["start_date"] = form.date_input("When to start the backtest ?", date(2023,3,1))
        inputs["end_date"] = form.date_input("When to end the backtest ?", date.today())
        
        inputs["lag"] = form.selectbox('Select lags target', self.lags)
        # inputs["fees_buy"] = form.slider("Select transaction fees in % for buy", 0, 15, 1.4)
        inputs["fees_buy"] = 1.5/100

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

    
    def prepare_display_portfolio(self, df_init, pnl_over_time):

        df_init = df_init[self.currencies + ["CASH"]]
        df_init = df_init.astype(float)
        labels = list(df_init.columns)

        get_price = self.state.prepared.iloc[0:3]

        for currency in self.currencies:
            df_init[currency] = df_init[currency]*get_price["CLOSE_" + currency].bfill()[0]

        self.state.total_portfolio = df_init.iloc[0].sum().round(2)
        
        self.state.fig1, ax1 = plt.subplots()
        ax1.pie(df_init.iloc[0], labels=labels, autopct='%1.1f%%',
                shadow=True, startangle=90)
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        # chart 1 
        pnl_over_time = pnl_over_time.loc[pnl_over_time["DATE"] > "2022-12-01"]
        split_pnl = pd.DataFrame(columns=["DATE", "PNL", "CURRENCY"])

        for currency in self.currencies:
            sub_pnl = pnl_over_time[["DATE", f"PNL_{currency}"]]
            sub_pnl["CURRENCY"] = currency 
            sub_pnl.rename(columns={f"PNL_{currency}": "PNL"}, inplace=True)
            split_pnl =pd.concat([split_pnl, sub_pnl], axis=0)

        self.state.chart1 = alt.Chart(split_pnl).mark_line(point=alt.OverlayMarkDef(color="red")).encode(
            x='DATE',
            y=alt.Y('PNL', scale=alt.Scale(domain=[split_pnl["PNL"].min()*0.95, 
                                                    split_pnl["PNL"].max()*1.05])),
            color="CURRENCY"
        ).interactive()

        # chart 2 
        self.state.chart2 = alt.Chart(pnl_over_time).mark_line(point=alt.OverlayMarkDef(color="red")).encode(
            x='DATE',
            y=alt.Y('PNL_PORTFOLIO', scale=alt.Scale(domain=[pnl_over_time["PNL_PORTFOLIO"].min()*0.95, 
                                                            pnl_over_time["PNL_PORTFOLIO"].max()*1.05]))
        ).interactive()


    def display_portfolio(self, trades):

        def color_survived(val):
            return ['color: green']*len(val) if val.MARGIN_EUR >0 else ['color: red']*len(val)

        col1, col2= self.st.columns([4,2])
        
        # display historical trades (+/- value sur positions ouvertes)
        col1.header(f"Trades history")
        col1.dataframe(trades.style.apply(color_survived, axis=1), use_container_width=True)

        # composition of portfolio on latest date
        col2.header(f"Portfolio composition = {self.state.total_portfolio:.2f}")
        col2.pyplot(self.state.fig1)
        
        # display split PNL over time 
        self.st.header("Split Portfolio PNL evolution")
        self.st.altair_chart(self.state.chart1, theme="streamlit", use_container_width=True)

        # display global PNL
        self.st.header("Overall Portfolio PNL evolution")
        self.st.altair_chart(self.state.chart2, theme="streamlit", use_container_width=True)

        
    def display_market(self,  pnl_prepared, moves_prepared):

        # col1, col2 = self.st.columns([4,2])

        # dates to buy or sell per crypto (still in df init) 
        # moves_prepared = moves_prepared.loc[moves_prepared["REAL_BUY_SELL"] !=0]
        # moves_prepared = moves_prepared[["DATE", "REAL_BUY_SELL", "CURRENCY"]]
        # col1.dataframe(moves_prepared, use_container_width=True)

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

