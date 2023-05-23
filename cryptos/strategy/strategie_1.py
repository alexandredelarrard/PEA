
import numpy as np
import pandas as pd

from strategy.main_strategie import MainStrategy


class Strategy1(MainStrategy):

    def __init__(self, configs, start_date, end_date =None, dict_prepared={}):

        MainStrategy.__init__(self, configs, start_date, end_date)

        self.seuils = 1.8

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

        for i in sub_prepared.index:

            lag_i = sub_prepared.loc[i, "LAG"]

            tentative_buy_sell = np.where(sub_prepared.loc[i, f"{variable_to_use}_{lag_i}"] > sub_prepared.loc[i, f"SEUIL_UP_{lag_i}"], -1,
                                    np.where(sub_prepared.loc[i, f"{variable_to_use}_{lag_i}"] < sub_prepared.loc[i, f"SEUIL_DOWN_{lag_i}"], 1, 0))

            if ((tentative_buy_sell==-1)&(sub_prepared.loc[i, "CURRENCY"]>0)):
                sub_prepared.loc[i:, "AMOUNT"] = (1-self.fees_sell)*sub_prepared.loc[i, "CLOSE"]*sub_prepared.loc[i, "CURRENCY"]
                sub_prepared.loc[i:, "CASH"] +=  sub_prepared.loc[i:, "AMOUNT"]
                sub_prepared.loc[i:, "CURRENCY"] = 0
                sub_prepared.loc[i, "REAL_BUY_SELL"] = -1
                
            if ((tentative_buy_sell==1)&(sub_prepared.loc[i, "CASH"]>0)):
                sub_prepared.loc[i:, "CURRENCY"] += ((1-self.fees_buy)*sub_prepared.loc[i, "CASH"])/sub_prepared.loc[i, "CLOSE"]
                sub_prepared.loc[i:, "AMOUNT"] = -1*sub_prepared.loc[i, "CASH"]
                sub_prepared.loc[i:, "CASH"] = 0
                sub_prepared.loc[i, "REAL_BUY_SELL"] = 1
            
        sub_prepared["PNL"] = sub_prepared["CASH"] + sub_prepared["CURRENCY"]*sub_prepared["CLOSE"]
        pnl = sub_prepared[["DATE", "PNL"]].groupby("DATE").mean().reset_index()

        return sub_prepared, pnl
    

    def main_strategy(self, prepared,
                        df_init=None, 
                        args = {}):
        
        lag = args["lag"]
        currency = args["currency"]
        prepared = prepared.copy()
        variable_to_use = "TARGET_NORMALIZED"

        prepared[f"SEUIL_UP_{lag}"] = prepared[f"SEUIL_MEAN_{lag}"] + 2*prepared[f"SEUIL_STD_{lag}"]
        prepared[f"SEUIL_DOWN_{lag}"] = prepared[f"SEUIL_MEAN_{lag}"] - 2*prepared[f"SEUIL_STD_{lag}"]
        
        date_condition = prepared["DATE"].between(self.start_date, self.end_date)
        final_prepared = prepared.loc[date_condition].reset_index(drop=True)
        final_prepared["LAG"] = lag

        final_prepared = final_prepared.sort_values("DATE", ascending= True)
        prepared = prepared.sort_values("DATE", ascending= True)
        
        return self.execute_strategie(final_prepared, 
                                        currency=currency, 
                                        variable_to_use=variable_to_use,
                                        df_init=df_init)
