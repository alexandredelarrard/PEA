
import numpy as np
import pandas as pd

from strategy.main_strategie import MainStrategy


class Strategy1(MainStrategy):

    def __init__(self, configs, start_date, end_date =None, path_dirs=""):

        MainStrategy.__init__(self, configs, start_date, end_date, path_dirs)
        self.seuils = 1.8

    def condition(self, sub_prepared, i, lag_i, variable_to_use, args={}):

        take_profit = (sub_prepared.loc[i, "BUY_PRICE"] >0)&((sub_prepared.loc[i, "CLOSE"] - sub_prepared.loc[i, "BUY_PRICE"])/sub_prepared.loc[i, "BUY_PRICE"] >= 0.03)
        stop_loss = (sub_prepared.loc[i, "BUY_PRICE"] >0)&((sub_prepared.loc[i, "CLOSE"] - sub_prepared.loc[i, "BUY_PRICE"])/sub_prepared.loc[i, "BUY_PRICE"] < -0.03)
       
        a = np.where(take_profit|stop_loss, -1,
            np.where(sub_prepared.loc[i, f"{variable_to_use}_{lag_i}"] < sub_prepared.loc[i, f"SEUIL_DOWN_{lag_i}"], 1, 0))
        return a 

    def main_strategy(self, prepared,
                        df_init=None, 
                        args = {}):
        
        lag = args["lag"]
        currency = args["currency"]
        prepared = prepared.copy()
        variable_to_use = "TARGET_NORMALIZED"

        prepared[f"SEUIL_UP_{lag}"] = prepared[f"SEUIL_MEAN_{lag}"] + 1.9*prepared[f"SEUIL_STD_{lag}"]
        prepared[f"SEUIL_DOWN_{lag}"] = prepared[f"SEUIL_MEAN_{lag}"] - 1.9*prepared[f"SEUIL_STD_{lag}"]
        
        date_condition = prepared["DATE"].between(self.start_date, self.end_date)
        final_prepared = prepared.loc[date_condition].reset_index(drop=True)
        final_prepared["LAG"] = lag

        final_prepared = final_prepared.sort_values("DATE", ascending= True)
        prepared = prepared.sort_values("DATE", ascending= True)
        
        return self.execute_strategie(final_prepared, 
                                        currency=currency, 
                                        variable_to_use=variable_to_use,
                                        df_init=df_init)
