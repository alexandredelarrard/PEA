
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy.stats import norm


class MainStrategy(object):

    def __init__(self, configs, start_date, end_date=None, lag=15, fees_buy=0.015, fees_sell=0.026):

        self.configs = configs
        self.currencies = self.configs["cryptos_desc"]["Cryptos"]

        self.start_date =  pd.to_datetime(start_date, format = "%Y-%m-%d")
        self.end_date = end_date
        if self.end_date !=None:
            self.end_date = pd.to_datetime(end_date , format = "%Y-%m-%d")

        self.lag = lag
        self.fees_buy = fees_buy
        self.fees_sell = fees_sell

    def deduce_threshold(self, prepared, target):
        # Fit a normal distribution to the data:
        mu, std = norm.fit(prepared.loc[~prepared[target].isnull(), target][:1000])
        return mu - 1.95*std, mu + 1.95*std

    def main_strategie_1(self, 
                         prepared, 
                         currency = "BTC",
                         df_init=None):
        
        seuil_down, seuil_up = self.deduce_threshold(prepared, f"TARGET_{currency}_NORMALIZED_{self.lag}")
        
        if isinstance(df_init, pd.DataFrame):
            prepared["CASH"] = float(df_init[f"CASH_{currency}"].values[0])
            prepared["CURRENCY"] = float(df_init[currency].values[0])
        else:
            prepared["CASH"] = 100
            prepared["CURRENCY"] = 0

        if self.end_date == None:
            self.end_date = datetime.today()

        prepared = prepared.loc[prepared["DATE"].between(self.start_date, self.end_date)]
        prepared = prepared.sort_values("DATE", ascending= True)

        prepared[f"BUY_HOLD_SELL_{currency}"] = np.where(prepared[f"TARGET_{currency}_NORMALIZED_{self.lag}"] > seuil_up, -1,
                                    np.where(prepared[f"TARGET_{currency}_NORMALIZED_{self.lag}"] < seuil_down, 1, 0))
        prepared["REAL_BUY_SELL"] = 0

        for i in prepared.index:
            
            if ((prepared.loc[i, f"BUY_HOLD_SELL_{currency}"]==-1)&(prepared.loc[i, "CURRENCY"]>0)):
                prepared.loc[i:, "CASH"] += (1-self.fees_sell)*prepared.loc[i, f"CLOSE_{currency}"]*prepared.loc[i, "CURRENCY"]
                prepared.loc[i:, "CURRENCY"] = 0
                prepared.loc[i,"REAL_BUY_SELL"] = -1

            if ((prepared.loc[i, f"BUY_HOLD_SELL_{currency}"]==1)&(prepared.loc[i, "CASH"]>0)):
                prepared.loc[i:, "CURRENCY"] += ((1-self.fees_buy)*prepared.loc[i, "CASH"])/prepared.loc[i, f"CLOSE_{currency}"]
                prepared.loc[i:, "CASH"] = 0
                prepared.loc[i,"REAL_BUY_SELL"] = 1
            
        prepared["PNL"] = prepared["CASH"] + prepared["CURRENCY"]*prepared[f"CLOSE_{currency}"]

        pnl = prepared[["DATE", "PNL"]].groupby("DATE").mean().reset_index()
        return prepared, pnl


    def main_strategy_1_anaysis_currencies(self, 
                                            prepared,
                                            df_init=None):
        
        dict_moves = {}
        dict_pnl = {}

        for currency in self.currencies:
            dict_moves[currency], dict_pnl[currency] = self.main_strategie_1(prepared, currency = currency, df_init=df_init)
        
        #aggregate all PNLs
        for i, currency in enumerate(self.currencies):
            dict_pnl[currency].rename(columns={"PNL" : f"PNL_{currency}"}, inplace= True)

            if i == 0:
                pnl_prepared = dict_pnl[currency]
            else:
                pnl_prepared = pnl_prepared.merge(dict_pnl[currency], on="DATE", how="left", validate= "1:1")

        coins = [x for x in pnl_prepared.columns if "PNL_" in x]
        pnl_prepared["PNL_PORTFOLIO"] = pnl_prepared[coins].sum(axis=1)

        # aggregate all biy / hold / sell positions
        for i, currency in enumerate(self.currencies):
            dict_moves[currency] = dict_moves[currency][["DATE", "REAL_BUY_SELL"]]
            dict_moves[currency]["CURRENCY"] = currency

            if i == 0:
                moves_prepared = dict_moves[currency]
            else:
                moves_prepared = pd.concat([moves_prepared, dict_moves[currency]], axis=0)

        return pnl_prepared, moves_prepared
    
    def strategy_1_lags_comparison(self, 
                                   prepared, 
                                   currency = "BTC",
                                   lags = [7, 15, 30, 45]):
        
        init_lag = self.lag

        for i, lag in enumerate(lags):
            self.lag = lag 
            _, pnl = self.main_strategie_1(prepared, currency = currency)
            pnl.rename(columns={"PNL": f"PNL_{lag}"}, inplace=True)

            if i == 0:
                result = pnl
            else: 
                result = result.merge(pnl, on="DATE", how="left", validate="1:1")
        
        self.lag = init_lag

        return result 
        

    def cash_available(self, prepared, df_init):
        
        cash_start_value = df_init["CASH"].values[0]
        cash_min_prop = 4
        cash_max_prop = 24
        cash_proportion = {}

        for currency in self.currencies:
            cash_proportion[currency] = 1/len(self.currencies)
        
        # simulate over past 3 months
        pnl,_= self.main_strategy_1_anaysis_currencies(prepared)

        return df_init

        

        