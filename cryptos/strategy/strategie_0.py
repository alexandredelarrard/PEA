
import numpy as np
import pandas as pd
import logging 

from strategy.main_strategie import MainStrategy


class Strategy0(MainStrategy):

    def __init__(self, configs, start_date, end_date =None, path_dirs=""):

        MainStrategy.__init__(self, configs, start_date, end_date, path_dirs)

        self.parameters = {'BTC': {'LAG': '0.5', 'TARGET': 1, 'PERCENTAGE': -0.8}, 
                           'ETH': {'LAG': '0.5', 'TARGET': 1.0, 'PERCENTAGE': -0.7}, 
                           'XRP': {'LAG': '0.5', 'TARGET': 1.0, 'PERCENTAGE': -0.7}, 
                           'ADA': {'LAG': '1', 'TARGET': 1.0, 'PERCENTAGE': -0.9}, 
                           'DOGE': {'LAG': '1', 'TARGET': 1.0, 'PERCENTAGE': -0.9}, 
                           'SOL': {'LAG': '1', 'TARGET': 1, 'PERCENTAGE': -0.9}, 
                           'ICP': {'LAG': '1', 'TARGET': 1, 'PERCENTAGE': -0.9}, 
                           'SAND': {'LAG': '1', 'TARGET': 1.0, 'PERCENTAGE': -0.9}, 
                           'TRX': {'LAG': '0.5', 'TARGET': 1.0, 'PERCENTAGE': -0.7}, 
                           'XLM': {'LAG': '1', 'TARGET': 1.0, 'PERCENTAGE': -1}, 
                           'STX': {'LAG': '0.5', 'TARGET': 1.0, 'PERCENTAGE': -0.7},
                           'DOT': {'LAG': '0.5', 'TARGET': 1.0, 'PERCENTAGE': -0.7},
                           'DAI': {'LAG': '1', 'TARGET': 1.0, 'PERCENTAGE': -0.8},
                           'LTC': {'LAG': '1', 'TARGET': 1.0, 'PERCENTAGE': -0.8},
                           'LINK': {'LAG': '1', 'TARGET': 1.0, 'PERCENTAGE': -0.8}}

    def condition(self, sub_prepared, i, lag_i, variable_to_use, args):
        futur = int(self.parameters[args["currency"]]["TARGET"]*24)
        seuil = self.parameters[args["currency"]]["PERCENTAGE"]
        tentative_buy_sell = np.where(sub_prepared.loc[min(args["max_index"], i + futur), "REAL_BUY_SELL"] == 1, -1,
                            np.where(sub_prepared.loc[i, f"{variable_to_use}_{lag_i}"] < seuil, 1, 0)) # perte de 8%
        return tentative_buy_sell
    

    def main_strategy(self, prepared,
                        df_init=None, 
                        args = {}):
        
        currency = args["currency"]
        lag = args["lag"] #self.parameters[currency]["LAG"]
        prepared = prepared.copy()
        variable_to_use = "TARGET_NORMALIZED"
        
        date_condition = prepared["DATE"].between(self.start_date, self.end_date)
        final_prepared = prepared.loc[date_condition].reset_index(drop=True)
        final_prepared["LAG"] = lag

        final_prepared[f"SEUIL_UP_{lag}"] = final_prepared[f"SEUIL_MEAN_{lag}"] + 2*final_prepared[f"SEUIL_STD_{lag}"]
        final_prepared[f"SEUIL_DOWN_{lag}"] = final_prepared[f"SEUIL_MEAN_{lag}"] - 2*final_prepared[f"SEUIL_STD_{lag}"]

        final_prepared = final_prepared.sort_values("DATE", ascending= True)
        prepared = prepared.sort_values("DATE", ascending= True)
        
        return self.execute_strategie(final_prepared, 
                                        currency=currency, 
                                        variable_to_use=variable_to_use,
                                        df_init=df_init)
