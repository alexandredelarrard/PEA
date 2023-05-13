
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import tqdm
import random
from scipy.stats import truncnorm, uniform, norm


class MainStrategy(object):

    def __init__(self, configs, start_date, end_date=None, lag=15, fees_buy=0.015, fees_sell=0.026):

        self.configs = configs
        self.currencies = self.configs.load["cryptos_desc"]["Cryptos"]
        self.lags = self.configs.load["cryptos_desc"]["LAGS"]

        self.start_date =  pd.to_datetime(start_date, format = "%Y-%m-%d")
        self.end_date = end_date
        if self.end_date !=None:
            self.end_date = pd.to_datetime(end_date , format = "%Y-%m-%d")

        self.lag = lag
        self.fees_buy = fees_buy
        self.fees_sell = fees_sell

    def deduce_threshold(self, prepared, target):
        # Fit a normal distribution to the data:
        mu, std = norm.fit(prepared.loc[~prepared[target].isnull(), target][:3000])
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
        prepared["AMOUNT"] = 0

        for i in prepared.index:
            
            if ((prepared.loc[i, f"BUY_HOLD_SELL_{currency}"]==-1)&(prepared.loc[i, "CURRENCY"]>0)):
                prepared.loc[i:, "AMOUNT"] = (1-self.fees_sell)*prepared.loc[i, f"CLOSE_{currency}"]*prepared.loc[i, "CURRENCY"]
                prepared.loc[i:, "CASH"] +=  prepared.loc[i:, "AMOUNT"]
                prepared.loc[i:, "CURRENCY"] = 0
                prepared.loc[i, "REAL_BUY_SELL"] = -1
                
            if ((prepared.loc[i, f"BUY_HOLD_SELL_{currency}"]==1)&(prepared.loc[i, "CASH"]>0)):
                prepared.loc[i:, "CURRENCY"] += ((1-self.fees_buy)*prepared.loc[i, "CASH"])/prepared.loc[i, f"CLOSE_{currency}"]
                prepared.loc[i:, "AMOUNT"] = -1*prepared.loc[i, "CASH"]
                prepared.loc[i:, "CASH"] = 0
                prepared.loc[i, "REAL_BUY_SELL"] = 1
            
        prepared["PNL"] = prepared["CASH"] + prepared["CURRENCY"]*prepared[f"CLOSE_{currency}"]
        pnl = prepared[["DATE", "PNL"]].groupby("DATE").mean().reset_index()

        return prepared, pnl


    def main_strategy_1_anaysis_currencies(self, 
                                            prepared,
                                            df_init=None,
                                            deduce_moves=True):
        
        moves_prepared = None
        dict_moves = {}

        for i, currency in enumerate(self.currencies):
            dict_moves[currency], dict_pnl = self.main_strategie_1(prepared, currency = currency, df_init=df_init)
            dict_pnl.rename(columns={"PNL" : f"PNL_{currency}"}, inplace= True)

            if i == 0:
                pnl_prepared = dict_pnl
            else:
                pnl_prepared = pnl_prepared.merge(dict_pnl, on="DATE", how="left", validate= "1:1")

        coins = [x for x in pnl_prepared.columns if "PNL_" in x]
        pnl_prepared["PNL_PORTFOLIO"] = pnl_prepared[coins].sum(axis=1, numeric_only=True)

        if deduce_moves:
            # aggregate all biy / hold / sell positions
            for i, currency in enumerate(self.currencies):
                dict_moves[currency] = dict_moves[currency][["DATE", "REAL_BUY_SELL", "AMOUNT", f"CLOSE_{currency}"]]
                dict_moves[currency]["CURRENCY"] = currency
                dict_moves[currency].rename(columns={f"CLOSE_{currency}" : "PRICE"}, inplace=True)

                if i == 0:
                    moves_prepared = dict_moves[currency]
                else:
                    moves_prepared = pd.concat([moves_prepared, dict_moves[currency]], axis=0)
        
            moves_prepared = moves_prepared.loc[moves_prepared["REAL_BUY_SELL"] !=0]
            moves_prepared = moves_prepared[["DATE", "REAL_BUY_SELL", "CURRENCY", "AMOUNT", "PRICE"]]

        return pnl_prepared, moves_prepared
    
    def strategy_1_lags_comparison(self, 
                                   prepared, 
                                   currency = "BTC"):
        
        init_lag = self.lag

        for i, lag in enumerate(self.lags):
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
        
        vect_cash = []
        date_range = pd.date_range(prepared["DATE"].min(), prepared["DATE"].max())

        # simulate over 3 months random in 2 years
        start_index = random.randint(1, len(date_range) - 101)
        self.start_date = pd.to_datetime(date_range[start_index], format="%Y-%m-%d")
        self.end_date = pd.to_datetime(date_range[start_index + 100], format="%Y-%m-%d")

        # initialization
        for currency in self.currencies:
            vect_cash.append(1/len(self.currencies))
            df_init[f"CASH_{currency}"] = 1/len(self.currencies)
            df_init[currency] = 0

        pnl,_= self.main_strategy_1_anaysis_currencies(prepared, df_init=df_init, deduce_moves=False)
        sharp = (pnl.iloc[-1, -1] -1)/pnl["PNL_PORTFOLIO"].std()

        sharps = []
        all_vects=[vect_cash]
        pnls = [pnl.iloc[-1, -1]]

        for i in tqdm.tqdm(range(100)):

            # random new value 
            new_vect = truncnorm.rvs(a=0.04, b=0.25, loc=vect_cash, scale=0.07, size=len(vect_cash))
            new_vect = new_vect/sum(new_vect)
            rand = uniform.rvs(size=1)[0]

            for i, currency in enumerate(self.currencies):
                df_init[f"CASH_{currency}"] = new_vect[i]

            pnl,_= self.main_strategy_1_anaysis_currencies(prepared, df_init=df_init, deduce_moves=False)
            new_sharp = (pnl.iloc[-1, -1] - 1)/pnl["PNL_PORTFOLIO"].std()

            ratio = new_sharp / sharp

            if ratio > rand:
                vect_cash= new_vect
                sharp = new_sharp
                pnls.append(pnl.iloc[-1, -1]) 
            else:
                pnls.append(pnls[-1])

            sharps.append(new_sharp)
            all_vects.append(vect_cash)

        return pnls[-1], sharps[-1], all_vects[-1], self.start_date, self.end_date


    def portfolio_currency_balance(self, prepared, df_init):
        # MH strategy to find out best sharp ratio 

        
        results = {"PNL":{}, "SHARP":{}, "CASH" : {}, "START" : {}, "END" : {}}

        for i in range(10):
            results["PNL"][i], results["SHARP"][i], results["CASH"][i], results["START"][i], results["END"][i] = self.cash_available(prepared, df_init)

    def allocate_cash(self, df_init):
        cash_start_value = float(df_init["CASH"].values[0])
        for currency in self.currencies:
            df_init[f"CASH_{currency}"] = max(0, int((1/len(self.currencies))*cash_start_value) - 1)
        return df_init