
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


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

        self.fees_buy = 0.0015
        self.fees_sell = 0.0015
        self.back_step_months = 4

        self.picked_strategie = None


    def main_strategy_1_analysis_currencies(self, 
                                           dict_prepared, 
                                           df_init=None,
                                           deduce_moves=True, 
                                           args = {}):
        
        moves_prepared = None
        dict_moves = {}

        for i, currency in enumerate(self.currencies):
            args["currency"] = currency
            dict_moves[currency], dict_pnl = self.picked_strategie(dict_prepared[currency], 
                                                                    df_init=df_init, 
                                                                    args=args)
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


    def allocate_cash(self, dict_prepared, df_init, current_price, args):

        cash_start_value = float(df_init.loc["BALANCE", "CASH"])
        
        # get the PNL for each currency of the past 2 months
        tampon_start = self.start_date
        self.start_date = tampon_start - timedelta(days=int(self.back_step_months*30.5))
        pnl_prepared, _ = self.main_strategy_1_anaysis_currencies(dict_prepared,
                                                                  df_init=None,
                                                                  deduce_moves=False,
                                                                  args=args)
        
        # deduce proportion of portfolio in theory needed 
        # idea is that next 3 months will be the same (or close)
        pnls = (pnl_prepared.iloc[-2, 1:-1] - 100)/(pnl_prepared.iloc[-2, -1] - 100*len(self.currencies))
        gain_cash = 1 + (pnls - pnls.mean())*4
        gain_cash = gain_cash/gain_cash.sum()
        gain_cash = gain_cash.clip(0, 2*(1/len(self.currencies))) # verrou surete
        gain_cash = gain_cash/gain_cash.sum()
        
        # get price_value_ each coin 
        df_init.loc["BALANCE"] = df_init.loc["BALANCE"].astype(float)
        df_init.loc["COIN_VALUE"] = 0
        df_init.loc["TARGET_PERCENTAGE"] = 0
        for currency in self.currencies:
            df_init.loc["COIN_VALUE", currency] = df_init.loc["BALANCE", currency]*float(current_price[currency])
            df_init.loc["TARGET_PERCENTAGE", currency] = gain_cash[f"PNL_{currency}"]
        df_init.loc["ACTUAL_PERCENTAGE"] = df_init.loc["COIN_VALUE"]/df_init.loc["COIN_VALUE"].sum()
        
        tampon_percentage = np.where(df_init.loc["ACTUAL_PERCENTAGE"]>df_init.loc["TARGET_PERCENTAGE"], 0, df_init.loc["TARGET_PERCENTAGE"])
        tampon_percentage = tampon_percentage/tampon_percentage.sum()
        df_init.loc["CASH_TO_ALLOCATE"] = tampon_percentage*cash_start_value
        df_init.loc["CASH_TO_ALLOCATE"] = (df_init.loc["CASH_TO_ALLOCATE"].astype(float) - 0.4).round(0)
        self.start_date = tampon_start

        assert df_init.loc["CASH_TO_ALLOCATE"].sum() <= cash_start_value
        assert abs(df_init.loc["ACTUAL_PERCENTAGE"].sum() - 1)< 0.001
        assert abs(df_init.loc["TARGET_PERCENTAGE"].sum() - 1)< 0.001

        df_init.loc["COIN_VALUE", "CASH"] = df_init.loc["BALANCE", "CASH"] # useful for app shape

        return df_init