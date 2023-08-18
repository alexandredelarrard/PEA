from datetime import date, timedelta, datetime
import numpy as np
import matplotlib.pyplot as plt
import altair as alt
import pandas as pd 

class MainApp(object):

    def __init__(self, st, configs):
        
        self.configs = configs
        self.st = st
        self.currencies = self.configs.load["cryptos_desc"]["Cryptos"]
        self.strategies = self.configs.load["cryptos_desc"]["STRATEGIES"]

        self.init_state()

    def init_state(self):

        self.state = self.st.session_state

        if "submitted" not in self.state:
            self.state.submitted = False

        if "dict_prepared" not in self.state:
            self.state.dict_prepared = None

        if "pnl_over_time" not in self.state:
            self.state.pnl_over_time= None 

        if "trades" not in self.state:
            self.state.trades= None 

        if "df_init" not in self.state:
            self.state.df_init= None 

        if "fig1" not in self.state:
            self.state.fig1 = None

        if "chart1" not in self.state:
            self.state.chart1 = None

        if "chart2" not in self.state:
            self.state.chart2 = None

        if "total_portfolio" not in self.state:
            self.state.total_portfolio = None

        if "done_once" not in self.state:
            self.state.done_once = False

        if "orders" not in self.state:
            self.state.orders = None

        if "portfolio" not in self.state:
            self.state.portfolio = None

    def get_sidebar_inputs(self):

        inputs = {"currency" : None,
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
        inputs["start_date"] = form.date_input("When to start the backtest ?", date.today() - timedelta(days=366))
        inputs["end_date"] = form.date_input("When to end the backtest ?", date.today() + timedelta(days=1))
        
        inputs["strategie"] = form.selectbox('Select strategie to use', self.strategies)
        inputs["button"] = form.form_submit_button("Run Analysis", on_click=lambda: self.state.update(submitted=True))

        return inputs


    def display_backtest(self, dict_prepared, inputs, pnl_currency, prepared_currency, trades):

        trades = trades.copy()
        real_trade = pd.concat([trades[["TIME_BUY", "ASSET"]], trades.loc[~trades["TIME_SOLD"].isnull()][["TIME_SOLD", "ASSET"]]], axis=0)
        real_trade["KRAKEN_BUY_SELL"] = np.where(real_trade["TIME_BUY"].isnull(), -1.2, 1.2)
        for col in["TIME_BUY", "TIME_SOLD"]:
            real_trade[col] = pd.to_datetime(real_trade[col], format="%Y-%m-%d %H:%M%S")

        real_trade["DATE"] = np.where(real_trade["TIME_BUY"].isnull(), real_trade["TIME_SOLD"], real_trade["TIME_BUY"])
        real_trade["DATE"] = real_trade["DATE"].dt.round("H")
        real_trade = real_trade.loc[real_trade["ASSET"] == inputs['currency']][["DATE", "KRAKEN_BUY_SELL"]].drop_duplicates()
        real_trade = prepared_currency[["DATE"]].merge(real_trade, on="DATE", how="left", validate="1:1")
        real_trade["KRAKEN_BUY_SELL"].fillna(0, inplace=True)

        self.st.header(f"BUY/SELL {inputs['currency']}")
        
        fig, ax = plt.subplots(figsize=(20,10))
        real_trade[["DATE", "KRAKEN_BUY_SELL"]].set_index("DATE").plot(ax = ax, color="red", style="--")
        prepared_currency[["DATE", "REAL_BUY_SELL"]].set_index(["DATE"]).plot(ax = ax)
        prepared_currency[["DATE", "PREDICTION_BNARY_TARGET_UP"]].set_index(["DATE"]).plot(ax = ax)
        prepared_currency[["DATE", "CLOSE"]].set_index(["DATE"]).plot(ax = ax, color = "green", style="--", secondary_y =True)
        self.st.pyplot(fig)

        prepared = dict_prepared[inputs['currency']]
        max_date = prepared["DATE"].max()
        value = prepared.loc[prepared["DATE"] == max_date, "CLOSE"].values[0]
        self.st.header(f"SPOT PRICE {inputs['currency']} || {max_date} || {value:.3f}")
        self.st.line_chart(prepared[["DATE", "CLOSE"]].set_index("DATE"))

        self.st.header(f"PNL for currency : {inputs['currency']}")
        self.st.line_chart(pnl_currency.set_index("DATE"))

    
    def prepare_display_portfolio(self, df_init, pnl_over_time, trades):

        labels = list(df_init.columns)
        self.state.total_portfolio = df_init.loc["COIN_VALUE"].sum().round(2)
        
        self.state.fig1, ax1 = plt.subplots()
        ax1.pie(df_init.loc["COIN_VALUE"], labels=labels, autopct='%1.1f%%',
                shadow=True, startangle=90)
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        # chart 1 
        trades = trades.copy()
        trades["TIME_SOLD"]= trades["TIME_SOLD"].fillna(datetime.now()).dt.round("D").dt.strftime("%Y-%m-%d")
        trades["TOTAL_EARNED"] = trades["EARNED_MARGIN"].fillna(0) + trades["TO_EARN_MARGIN"] - trades["FEE_SELL"].fillna(0) - trades["FEE_BUY"].fillna(0)
        self.state.chart1 = pd.pivot_table(index="TIME_SOLD", columns="ASSET", values="TOTAL_EARNED", aggfunc=sum, data=trades)
        self.state.chart1 = self.state.chart1.fillna(0)

        # chart 2 
        self.state.chart2 = alt.Chart(pnl_over_time).mark_line(point=alt.OverlayMarkDef(color="red")).encode(
            x='DATE',
            y=alt.Y('PNL_PORTFOLIO', scale=alt.Scale(domain=[pnl_over_time["PNL_PORTFOLIO"].min()*0.9, 
                                                            pnl_over_time["PNL_PORTFOLIO"].max()*1.1]))
        ).interactive()


    def display_portfolio(self, portfolio, orders):

        def color_trades(val):
            return ['color: green']*len(val) if val.NET_MARGIN_PERCENT > 0 else ['color: red']*len(val)

        def color_orders(val):
            return ['color: green']*len(val) if val.SIDE == "buy" else ['color: red']*len(val)

        col1, col2= self.st.columns([4,2])
        
        # display historical trades (+/- value sur positions ouvertes)
        if isinstance(portfolio, pd.DataFrame):
            if portfolio.shape[0]>0:
                col1.header(f"Trades history")
                col1.dataframe(portfolio.style.apply(color_trades, axis=1), use_container_width=True)

        # composition of portfolio on latest date
        col2.header(f"Portfolio composition = {self.state.total_portfolio:.2f}")
        col2.pyplot(self.state.fig1)

        # orders 
        if isinstance(orders, pd.DataFrame):
            if orders.shape[0]>0:
                col1.header("Orders history")
                self.st.dataframe(orders.style.apply(color_orders, axis=1), use_container_width=True)
                
        # display split PNL over time 
        self.st.header("Trades")
        self.st.bar_chart(self.state.chart1, use_container_width=True)

        # display global PNL
        self.st.header("Overall Portfolio PNL evolution")
        self.st.altair_chart(self.state.chart2, theme="streamlit", use_container_width=True)

        
    def display_market(self, pnl_prepared):

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
        a = pnl_prepared[["DATE", "PNL_PORTFOLIO"]]
        a["HUE"] = "STRATEGY"
        b = pnl_prepared[["DATE", "PNL_PORTFOLIO_BASELINE"]].rename(columns={"PNL_PORTFOLIO_BASELINE": "PNL_PORTFOLIO"})
        b["HUE"] = "BASELINE"
        c = pd.concat([a, b], axis=0)

        self.st.header("Overall Portfolio PNL evolution")
        chart = alt.Chart(c).mark_line(point=alt.OverlayMarkDef(color="red")).encode(
            x='DATE',
            y=alt.Y("PNL_PORTFOLIO", scale=alt.Scale(domain=[c["PNL_PORTFOLIO"].min()*0.9, 
                                                            c["PNL_PORTFOLIO"].max()*1.1])),
            color="HUE"
        ).interactive()
        self.st.altair_chart(chart, theme="streamlit", use_container_width=True)
