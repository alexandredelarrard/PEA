import abc
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


class MainStrategy(object):

    def __init__(self, configs, start_date, end_date=None, path_dirs=""):

        self.configs = configs
        self.path_dirs = path_dirs
        self.hours = range(24)

        self.currencies = self.configs.load["cryptos_desc"]["Cryptos"]

        self.start_date =  pd.to_datetime(start_date, format = "%Y-%m-%d")
        self.end_date = end_date
        if self.end_date !=None:
            self.end_date = pd.to_datetime(end_date , format = "%Y-%m-%d")
        else:
            self.end_date = datetime.today()

        self.fees_buy = 0.0015
        self.fees_sell = 0.0015
        self.back_step_months = 4

        #   - news feeds -> sentiments strengths 
        #   - twitter feeds 
        #   - etoro feeds ? 
        #   - spread bid / ask + tradecount std / min / max TODO
        #   - % of all coins traded over past X times 
        #   - remaining liquidity / coins to create (offer) TODO
        #   - Coin tenure / descriptors (clustering ?) / employees ? / state, etc.
        # - RSI TODO
        # - spread bid ask over time 
        # - tradecount over time
        # - news sentiments extract 
        # - coin trends (volume tradecount var + news attraction) -> for new coins only 
        # - put / call futurs baught same time -> price on vol 


    @abc.abstractmethod
    def condition(self):
        pass

    def execute_strategie(self, 
                         sub_prepared, 
                         currency = "BTC",
                         variable_to_use = None,
                         df_init=None):
        
        if isinstance(df_init, pd.DataFrame):
            sub_prepared["CASH"] = df_init.loc["CASH_TO_ALLOCATE", currency]
            sub_prepared["CURRENCY"] = df_init.loc["BALANCE", currency]
        else:
            sub_prepared["CASH"] = 100
            sub_prepared["CURRENCY"] = 0

        sub_prepared["REAL_BUY_SELL"] = 0
        sub_prepared["AMOUNT"] = 0
        sub_prepared["BUY_PRICE"] = -1

        market = sub_prepared[["DATE", "CLOSE", "CASH", "CURRENCY"]]

        max_index = max(sub_prepared.index)
        for i in sub_prepared.index:

            if i == max_index:
                market.loc[i:, "CURRENCY"] += ((1-self.fees_buy)*market.loc[i, "CASH"])/market.loc[i, "CLOSE"]
                market.loc[i:, "CASH"] = 0

            tentative_buy_sell = self.condition(sub_prepared, i, variable_to_use, args={"currency" : currency, "max_index" : max_index})

            if ((tentative_buy_sell==-1)&(sub_prepared.loc[i, "CURRENCY"]>0)):
                sub_prepared.loc[i:, "AMOUNT"] = (1-self.fees_sell)*sub_prepared.loc[i, "CLOSE"]*sub_prepared.loc[i, "CURRENCY"]
                sub_prepared.loc[i:, "CASH"] +=  sub_prepared.loc[i:, "AMOUNT"]
                sub_prepared.loc[i:, "CURRENCY"] = 0
                sub_prepared.loc[i, "REAL_BUY_SELL"] = -1
                sub_prepared.loc[i:, "BUY_PRICE"] = -1
                
            if ((tentative_buy_sell==1)&(sub_prepared.loc[i, "CASH"]>0)):
                sub_prepared.loc[i:, "CURRENCY"] += ((1-self.fees_buy)*sub_prepared.loc[i, "CASH"])/sub_prepared.loc[i, "CLOSE"]
                sub_prepared.loc[i:, "AMOUNT"] = -1*sub_prepared.loc[i, "CASH"]
                sub_prepared.loc[i:, "CASH"] = 0
                sub_prepared.loc[i, "REAL_BUY_SELL"] = 1
                sub_prepared.loc[i:, "BUY_PRICE"] = sub_prepared.loc[i, "CLOSE"]
            
        sub_prepared["PNL"] = sub_prepared["CASH"] + sub_prepared["CURRENCY"]*sub_prepared["CLOSE"]
        market["PNL_BASELINE"] = market["CASH"] + market["CURRENCY"]*market["CLOSE"]

        pnl = sub_prepared[["DATE", "PNL"]].groupby("DATE").mean().reset_index()
        pnl_market = market[["DATE", "PNL_BASELINE"]].groupby("DATE").mean().reset_index()
        pnl = pnl.merge(pnl_market, on="DATE", how="left", validate="1:1")

        sub_prepared = sub_prepared[["DATE", "CLOSE", "REAL_BUY_SELL", "AMOUNT", "BUY_PRICE", 
                                     "CASH", "PREDICTION_BNARY_TARGET_DOWN", 
                                     "PREDICTION_BNARY_TARGET_UP", "DELTA_CLOSE_MEAN_25", 
                                     "PNL"]]

        return sub_prepared, pnl
    

    def main_strategy_analysis_currencies(self, 
                                           dict_prepared, 
                                           df_init=None,
                                           deduce_moves=True, 
                                           args = {}):
        
        moves_prepared = None
        dict_moves = {}

        for i, currency in enumerate(self.currencies):
            args["currency"] = currency
            dict_moves[currency], dict_pnl = self.main_strategy(dict_prepared[currency].copy(), 
                                                                df_init=df_init, 
                                                                args=args)
            dict_pnl.rename(columns={"PNL" : f"PNL_{currency}", "PNL_BASELINE" : f"PNL_BASELINE_{currency}"}, inplace= True)

            if i == 0:
                pnl_prepared = dict_pnl
            else:
                pnl_prepared = pnl_prepared.merge(dict_pnl, on="DATE", how="left", validate= "1:1")

        coins = [x for x in pnl_prepared.columns if "PNL_" in x and "PNL_BASELINE" not in x]
        base = [x for x in pnl_prepared.columns if "PNL_BASELINE" in x]
        for coin in coins: 
            pnl_prepared[coin] = pnl_prepared[coin].bfill()
        for coin in base: 
            pnl_prepared[coin] = pnl_prepared[coin].bfill()
        pnl_prepared["PNL_PORTFOLIO"] = pnl_prepared[coins].sum(axis=1, numeric_only=True)
        pnl_prepared["PNL_PORTFOLIO_BASELINE"] = pnl_prepared[base].sum(axis=1, numeric_only=True)

        if deduce_moves:
            # aggregate all biy / hold / sell positions
            for i, currency in enumerate(self.currencies):
                dict_moves[currency] = dict_moves[currency][["DATE", "REAL_BUY_SELL", "AMOUNT", "CLOSE", "PREDICTION_BNARY_TARGET_UP", "PREDICTION_BNARY_TARGET_DOWN"]]
                dict_moves[currency]["CURRENCY"] = currency
                dict_moves[currency].rename(columns={"CLOSE" : "PRICE"}, inplace=True)

                if i == 0:
                    moves_prepared = dict_moves[currency]
                else:
                    moves_prepared = pd.concat([moves_prepared, dict_moves[currency]], axis=0)
        
            # moves_prepared = moves_prepared.loc[moves_prepared["REAL_BUY_SELL"] !=0]
            moves_prepared = moves_prepared[["DATE", "REAL_BUY_SELL", "CURRENCY", "AMOUNT", "PRICE", "PREDICTION_BNARY_TARGET_UP", "PREDICTION_BNARY_TARGET_DOWN"]]

        return pnl_prepared, moves_prepared


    def allocate_cash(self, dict_prepared, df_init, current_price, args):

        cash_start_value = float(df_init.loc["BALANCE", "CASH"])
        
        # get the PNL for each currency of the past 2 months
        tampon_start = self.start_date
        self.start_date = tampon_start - timedelta(days=int(self.back_step_months*30.5))
        pnl_prepared, _ = self.main_strategy_analysis_currencies(dict_prepared,
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