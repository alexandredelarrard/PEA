
import numpy as np
import pandas as pd
import logging 

from strategy.main_strategie import MainStrategy


class Strategy0(MainStrategy):

    def __init__(self, configs, start_date, end_date =None, dict_prepared={}):

        MainStrategy.__init__(self, configs, start_date, end_date)

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
                           'STX': {'LAG': '0.5', 'TARGET': 1.0, 'PERCENTAGE': -0.7}}
        # self.parameters = self.deduce_parameters(dict_prepared)

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
        sub_prepared["TO_SELL"] = 0

        if sub_prepared.shape[0]>0:
            max_index = max(sub_prepared.index)
            for i in sub_prepared.index:

                lag_i = sub_prepared.loc[i, "LAG"]

                futur = int(self.parameters[currency]["TARGET"]*24)
                seuil = self.parameters[currency]["PERCENTAGE"]
                tentative_buy_sell = np.where(sub_prepared.loc[min(max_index, i + futur), "REAL_BUY_SELL"] == 1, -1,
                                    np.where(sub_prepared.loc[i, f"{variable_to_use}_{lag_i}"] < seuil, 1, 0)) # perte de 8%

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
        
        currency = args["currency"]
        lag = self.parameters[currency]["LAG"]
        # lag= args["lag"]
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

    def deduce_parameters(self, dict_prepared):

        dict_all = {}
        columns = ["PERCENTAGE", "LAG", "TARGET", "NUMBER_OBS", "MEDIAN_FUTUR", "GAINS"]

        for currency in self.currencies:
            prepared = dict_prepared[currency]
            shape_0 = prepared.shape[0]
            results = []

            for lag in [0.5, 1, 2]:
                for target in [0.5, 1, 2]:
                    for percentage in range(-15, -6):
                        sub = prepared.loc[prepared[f"TARGET_NORMALIZED_{lag}"] < percentage/10]
                        nbr = sub.shape[0]/shape_0
                        avg = sub[f"DELTA_FUTUR_TARGET_{target}"].median()
                        results.append([percentage, lag, target, nbr, avg, avg*nbr])

            results = pd.DataFrame(results, columns=columns)
            best_results = results.loc[results["GAINS"] == results["GAINS"].max()]

            dict_all[currency] = {"LAG" : best_results["LAG"].values[0], 
                                 "TARGET" : best_results["TARGET"].values[0],
                                 "PERCENTAGE" : best_results["PERCENTAGE"].values[0]/10}
        return dict_all