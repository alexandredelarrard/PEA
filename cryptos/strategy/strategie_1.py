
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import tqdm
from scipy.stats import truncnorm, uniform, norm


class MainStrategy(object):

    def __init__(self, configs, start_date, end_date=None):

        self.configs = configs
        self.currencies = self.configs.load["cryptos_desc"]["Cryptos"]
        self.lags = self.configs.load["cryptos_desc"]["LAGS"]

        self.start_date =  pd.to_datetime(start_date, format = "%Y-%m-%d")
        self.end_date = end_date
        if self.end_date !=None:
            self.end_date = pd.to_datetime(end_date , format = "%Y-%m-%d")
        else:
            self.end_date = datetime.today()

        self.fees_buy = 0.015
        self.fees_sell = 0.015
        self.back_step_months= 3

    def deduce_threshold(self, prepared, target):
        # Fit a normal distribution to the data:
        mu, std = norm.fit(prepared.loc[~prepared[target].isnull(), target][:3000])
        return mu - 1.95*std, mu + 1.95*std

    def execute_strategie_1(self, 
                         sub_prepared, 
                         currency = "BTC",
                         df_init=None):
        
        if isinstance(df_init, pd.DataFrame):
            sub_prepared["CASH"] = df_init.loc["CASH_TO_ALLOCATE", currency]
            sub_prepared["CURRENCY"] = df_init.loc["BALANCE", currency]
        else:
            sub_prepared["CASH"] = 100
            sub_prepared["CURRENCY"] = 0

        sub_prepared["REAL_BUY_SELL"] = 0
        sub_prepared["AMOUNT"] = 0

        for i in sub_prepared.index:

            lag_i = sub_prepared.loc[i, "LAG"]

            if lag_i != "MIX_MATCH":
                tentative_buy_sell = np.where(sub_prepared.loc[i, f"TARGET_{currency}_NORMALIZED_{lag_i}"] > sub_prepared.loc[i, f"SEUIL_UP_{lag_i}"], -1,
                                    np.where(sub_prepared.loc[i, f"TARGET_{currency}_NORMALIZED_{lag_i}"] < sub_prepared.loc[i, f"SEUIL_DOWN_{lag_i}"], 1, 0))
            else: 
                tentative_buy_sell= 0
                for lag_i_ in list(set(self.lags) - set(["MIX_MATCH"])):
                    tentative_buy_sell += np.where(sub_prepared.loc[i, f"TARGET_{currency}_NORMALIZED_{lag_i_}"] > sub_prepared.loc[i, f"SEUIL_UP_{lag_i_}"], -1,
                                    np.where(sub_prepared.loc[i, f"TARGET_{currency}_NORMALIZED_{lag_i_}"] < sub_prepared.loc[i, f"SEUIL_DOWN_{lag_i_}"], 1, 0))
                tentative_buy_sell = np.where(tentative_buy_sell>=1, 1, 
                                     np.where(tentative_buy_sell<=-1, -1, 0)) 

            if ((tentative_buy_sell==-1)&(sub_prepared.loc[i, "CURRENCY"]>0)):
                sub_prepared.loc[i:, "AMOUNT"] = (1-self.fees_sell)*sub_prepared.loc[i, f"CLOSE_{currency}"]*sub_prepared.loc[i, "CURRENCY"]
                sub_prepared.loc[i:, "CASH"] +=  sub_prepared.loc[i:, "AMOUNT"]
                sub_prepared.loc[i:, "CURRENCY"] = 0
                sub_prepared.loc[i, "REAL_BUY_SELL"] = -1
                
            if ((tentative_buy_sell==1)&(sub_prepared.loc[i, "CASH"]>0)):
                sub_prepared.loc[i:, "CURRENCY"] += ((1-self.fees_buy)*sub_prepared.loc[i, "CASH"])/sub_prepared.loc[i, f"CLOSE_{currency}"]
                sub_prepared.loc[i:, "AMOUNT"] = -1*sub_prepared.loc[i, "CASH"]
                sub_prepared.loc[i:, "CASH"] = 0
                sub_prepared.loc[i, "REAL_BUY_SELL"] = 1
            
        sub_prepared["PNL"] = sub_prepared["CASH"] + sub_prepared["CURRENCY"]*sub_prepared[f"CLOSE_{currency}"]
        pnl = sub_prepared[["DATE", "PNL"]].groupby("DATE").mean().reset_index()

        return sub_prepared, pnl
    

    def main_strategy_1(self, prepared, currency = "BTC",
                        df_init=None, lag=None):
        
        for l in self.lags:
            prepared[f"SEUIL_DOWN_{l}"], prepared[f"SEUIL_UP_{l}"] = self.deduce_threshold(prepared, f"TARGET_{currency}_NORMALIZED_{l}")
        
        keep_cols = ["DATE", f"CLOSE_{currency}"] + [f'TARGET_{currency}_NORMALIZED_{x}' for x in self.lags] \
                    + [f'SEUIL_UP_{x}' for x in self.lags] + [f'SEUIL_DOWN_{x}' for x in self.lags]
        date_condition = prepared["DATE"].between(self.start_date, self.end_date)
        
        final_prepared = prepared.loc[date_condition][keep_cols].reset_index(drop=True)
        final_prepared["LAG"] = lag

        final_prepared = final_prepared.sort_values("DATE", ascending= True)
        prepared = prepared.sort_values("DATE", ascending= True)

        # if lag == None:

        #     date_range = pd.date_range(start=self.start_date, end=self.end_date, 
        #                                freq=f"{self.futur_step_days}D")
        #     for test_date in date_range:

        #         start_date_test = test_date - timedelta(days=self.back_step_days)
        #         end_date_test = test_date

        #         sub_prepared = prepared.loc[prepared["DATE"].between(start_date_test, end_date_test)].reset_index(drop=True)
        #         pnls = self._lags_comparison(sub_prepared[keep_cols], currency)

        #         if pnls.loc[~pnls["PNL_7"].isnull()].shape[0]>0:
        #             best_lag = pnls.iloc[-1:, 1:].idxmax(axis=1).values[0].replace("PNL_","")
        #         else:
        #             best_lag = "MEAN_LAGS"

        #         condition_final = final_prepared["DATE"].between(test_date, 
        #                                                     test_date + timedelta(days=self.futur_step_days))
        #         final_prepared.loc[condition_final, "LAG"] = best_lag
        
        return self.execute_strategie_1(final_prepared, 
                                        currency=currency, 
                                        df_init=df_init)


    def _lags_comparison(self, prepared, currency = "BTC"):
        
        for i, lag in enumerate(self.lags):
            prepared["LAG"] = lag
            _, pnl = self.execute_strategie_1(prepared, currency=currency, df_init=None)
            pnl.rename(columns={"PNL": f"PNL_{lag}"}, inplace=True)

            if i == 0:
                result = pnl
            else: 
                result = result.merge(pnl, on="DATE", how="left", validate="1:1")

        return result 
    

    def strategy_1_lags_comparison(self, prepared, currency="BTC", df_init=None):

        for i, lag in enumerate(self.lags):

            _, pnl = self.main_strategy_1(prepared, currency = currency,
                        df_init=df_init, lag=lag)
            pnl.rename(columns={"PNL": f"PNL_{lag}"}, inplace=True)

            if i == 0:
                result = pnl
            else: 
                result = result.merge(pnl, on="DATE", how="left", validate="1:1")

        return result 


    def main_strategy_1_anaysis_currencies(self, prepared, df_init=None,
                                            lag="15", deduce_moves=True):
        
        moves_prepared = None
        dict_moves = {}

        for i, currency in enumerate(self.currencies):
            dict_moves[currency], dict_pnl = self.main_strategy_1(prepared, currency = currency, df_init=df_init, lag=lag)
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
    

    def allocate_cash(self, prepared, df_init, lag="7"):

        cash_start_value = float(df_init.loc["BALANCE", "CASH"])
        
        # get the PNL for each currency of the past 3 months
        tampon_start = self.start_date
        self.start_date = tampon_start - timedelta(days=int(self.back_step_months*30.5))
        pnl_prepared, _ = self.main_strategy_1_anaysis_currencies(prepared, lag=lag, deduce_moves=False)
        
        # deduce proportion of portfolio in theory needed 
        # idea is that next 3 months will be the same (or close)
        pnls = (pnl_prepared.iloc[-1, 1:-1] - 100)/(pnl_prepared.iloc[-1, -1] - 100*len(self.currencies))
        gain_cash = 1 + (pnls - pnls.mean())*4
        gain_cash = gain_cash/gain_cash.sum()

        # get price_value_ each coin 
        prices = prepared.iloc[0]
        df_init.loc["BALANCE"] = df_init.loc["BALANCE"].astype(float)
        df_init.loc["COIN_VALUE"] = 0
        df_init.loc["TARGET_PERCENTAGE"] = 0
        for currency in self.currencies:
            df_init.loc["COIN_VALUE", currency] = df_init.loc["BALANCE", currency]*prices[f"CLOSE_{currency}"]
            df_init.loc["TARGET_PERCENTAGE", currency] = gain_cash[f"PNL_{currency}"]
        df_init.loc["ACTUAL_PERCENTAGE"] = df_init.loc["COIN_VALUE"]/df_init.loc["COIN_VALUE"].sum()

        tampon_percentage = np.where(df_init.loc["ACTUAL_PERCENTAGE"]>df_init.loc["TARGET_PERCENTAGE"], 0, df_init.loc["TARGET_PERCENTAGE"])
        tampon_percentage = tampon_percentage/tampon_percentage.sum()
        df_init.loc["CASH_TO_ALLOCATE"] = tampon_percentage*cash_start_value
        df_init.loc["CASH_TO_ALLOCATE"] = (df_init.loc["CASH_TO_ALLOCATE"].astype(float) - 0.4).round(0)
        self.start_date = tampon_start

        return df_init