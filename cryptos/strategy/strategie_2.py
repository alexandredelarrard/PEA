
import numpy as np
import pickle
import logging
import warnings
import os
import glob
from scipy.stats import norm

from strategy.main_strategie import MainStrategy
from modelling.training_strategy2 import TrainingStrat2

warnings.filterwarnings("ignore")

class Strategy2(MainStrategy):

    def __init__(self, configs, start_date, end_date =None, path_dirs=""):

        MainStrategy.__init__(self, configs, start_date, end_date, path_dirs)

        self.configs = self.configs.strategie["strategy_2"]
        self.configs_lgbm = self.configs["parameters"]
        self.target = self.configs["TARGET"]
        self.since_year = 2017
        self.trehshold = {"BTC" : {"UP" : 0.1, "DOWN" : 0.05},
                          "ETH" : {"UP" : 0.1, "DOWN" : 0.05},
                          "XRP" : {"UP" : 0.1, "DOWN" : 0.05},
                          "ADA" : {"UP" : 0.1, "DOWN" : 0.05},
                          "DOGE" : {"UP" : 0.1, "DOWN" : 0.05},
                          "SOL" : {"UP" : 0.1, "DOWN" : 0.05},
                          "TRX" : {"UP" : 0.1, "DOWN" : 0.05},
                          "XLM" : {"UP" : 0.1, "DOWN" : 0.05},
                          "DOT" : {"UP" : 0.1, "DOWN" : 0.05},
                          "LTC" : {"UP" : 0.1, "DOWN" : 0.05},
                          "DAI" : {"UP" : 0.1, "DOWN" : 0.05},
                          "LINK" : {"UP" : 0.1, "DOWN" : 0.05},
                          "ICP" : {"UP" : 0.1, "DOWN" : 0.05},
                          "SAND" : {"UP" : 0.1, "DOWN" : 0.05}}


    def data_prep(self, prepared):

        #filter year
        prepared = prepared.loc[prepared["DATE"].dt.year >= self.since_year]

        # prepared date features 
        prepared["HOUR"] = prepared["DATE"].dt.hour
        prepared["WEEK_DAY"] = prepared["DATE"].dt.dayofweek
        prepared["DAY_OF_YEAR"] = prepared["DATE"].dt.dayofyear
        prepared["DAY"] = prepared["DATE"].dt.day
        prepared["MONTH"] = prepared["DATE"].dt.month

        for col in self.configs["categorical_features"]:
            prepared[col] = prepared[col].astype("category")
        
        prepared = prepared.sort_values("DATE", ascending = False)

        return prepared
    

    def condition(self, sub_prepared, i, variable_to_use, args={}):
       
        take_profit = (sub_prepared.loc[i, "BUY_PRICE"] >0)&((sub_prepared.loc[i, "CLOSE"] - sub_prepared.loc[i, "BUY_PRICE"])/sub_prepared.loc[i, "BUY_PRICE"] >= 0.06)
        stop_loss = (sub_prepared.loc[i, "BUY_PRICE"] >0)&((sub_prepared.loc[i, "CLOSE"] - sub_prepared.loc[i, "BUY_PRICE"])/sub_prepared.loc[i, "BUY_PRICE"] < -0.03)
        sell = (sub_prepared.loc[i, f"PREDICTION_{variable_to_use}_DOWN"] >= 0.2)&(sub_prepared.loc[i, "DELTA_CLOSE_MEAN_25"] > 0.04)&(sub_prepared.loc[i, f"PREDICTION_{variable_to_use}_UP"] < 0.02) #self.trehshold[args["currency"]]["DOWN"]
        
        a = np.where((sub_prepared.loc[i, f"PREDICTION_{variable_to_use}_UP"] >= 0.2)&(sub_prepared.loc[i, "DELTA_CLOSE_MEAN_25"] < -0.04)&(sub_prepared.loc[i, f"PREDICTION_{variable_to_use}_DOWN"] < 0.02), #self.trehshold[args["currency"]]["UP"], 
                     1,
            np.where(sell|take_profit|stop_loss, -1, 0))#min(args["max_index"], i)
        
        return a 


    def main_strategy(self, prepared, df_init=None, args={}):

        currency = args["currency"]
        prepared = self.data_prep(prepared)

        date_condition = prepared["DATE"].between(self.start_date, self.end_date)
        final_prepared = prepared.loc[date_condition].reset_index(drop=True)

        for target in ["BNARY_TARGET_DOWN",
                       "BNARY_TARGET_UP"]:
            model, _ = self.load_model(currency, target)
            final_prepared = self.predicting(model, final_prepared, target)

        final_prepared = final_prepared.sort_values("DATE", ascending= True)

        return self.execute_strategie(final_prepared, 
                                    currency=currency, 
                                    variable_to_use="BNARY_TARGET",
                                    df_init=df_init)
   

    def predicting(self, model, prepared, target):

        prepared[f"PREDICTION_{target}"] = model.predict_proba(
            prepared[model.feature_name_],
            categorical_feature=list(self.configs["categorical_features"])
        )[:, 1]
        
        return prepared
    

    def load_model(self, currency, target):

        if target not in ["BNARY_TARGET_DOWN",
                          "BNARY_TARGET_UP"]:
            logging.warning(f"model need to be either {currency}_BNARY_TARGET_DOWN or {currency}_BNARY_TARGET_UP")

        liste_files = glob.glob("/".join([self.path_dirs["MODELS"], f"{currency}_{target}_*.pickle.dat"]))
        if len(liste_files)>0:
            latest_file = max(liste_files, key=os.path.getctime)
            logging.info(latest_file)
            return pickle.load(open(latest_file, "rb"))
        else:
            logging.warning(f"No model found for {target}")
    