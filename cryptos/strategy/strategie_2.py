
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
        self.final_metric = []
        self.since_year = 2017


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

        # filter target 
        prepared = prepared.loc[~prepared[self.target].isnull()]

        return prepared
    

    def condition(self, sub_prepared, i, lag_i, variable_to_use, args={}):
       
        take_profit = (sub_prepared.loc[i, "BUY_PRICE"] >0)&((sub_prepared.loc[i, "CLOSE"] - sub_prepared.loc[i, "BUY_PRICE"])/sub_prepared.loc[i, "BUY_PRICE"] >= 0.05)
        stop_loss = (sub_prepared.loc[i, "BUY_PRICE"] >0)&((sub_prepared.loc[i, "CLOSE"] - sub_prepared.loc[i, "BUY_PRICE"])/sub_prepared.loc[i, "BUY_PRICE"] < -0.04)
        sell = sub_prepared.loc[i, f"PREDICTION_NEG_{variable_to_use}_{lag_i}"] >= sub_prepared.loc[i, f"SEUIL_PREDICTION_NEG_{variable_to_use}_{lag_i}"]
        
        a = np.where(sub_prepared.loc[min(args["max_index"], i), f"PREDICTION_POS_{variable_to_use}_{lag_i}"] >= sub_prepared.loc[min(args["max_index"], i), f"SEUIL_PREDICTION_POS_{variable_to_use}_{lag_i}"], 1,
            np.where(sell|take_profit|stop_loss, -1, 0))
        
        return a 


    def main_strategy(self, prepared, df_init=None, args={}):

        currency = args["currency"]
        self.target_lag = args["lag"]
        
        prepared = self.data_prep(prepared)

        date_condition = prepared["DATE"].between(self.start_date, self.end_date)
        final_prepared = prepared.loc[date_condition].reset_index(drop=True)
        final_prepared["LAG"] = self.target_lag

        seuils={}
        for target in [f"POS_BINARY_FUTUR_TARGET_{self.target_lag}",
                        f"NEG_BINARY_FUTUR_TARGET_{self.target_lag}"]:
            model, _ = self.load_model(currency, target)
            final_prepared = self.predicting(model, final_prepared, target)
            seuils = self.define_thresholds(final_prepared, target, seuils)
            # final_prepared = self.postprocess(final_prepared, target)
        
        for k, v in seuils.items():
            final_prepared[k] = v

        final_prepared = final_prepared.sort_values("DATE", ascending= True)

        return self.execute_strategie(final_prepared, 
                                    currency=currency, 
                                    variable_to_use="BINARY_FUTUR_TARGET",
                                    df_init=df_init)
   

    def predicting(self, model, prepared, target):

        prepared[f"PREDICTION_{target}"] = model.predict_proba(
            prepared[model.feature_name_],
            categorical_features=list(self.configs["categorical_features"])
        )[:, 1]
        
        return prepared
    
    def postprocess(self, prepared, target):
        prepared[f"PREDICTION_{target}"] = np.where((prepared[f"TARGET_NORMALIZED_{self.past_lags}"] < self.conditional_filter_down)&("POS" in target), prepared[f"PREDICTION_{target}"],
                                            np.where((prepared[f"TARGET_NORMALIZED_{self.past_lags}"] > self.conditional_filter_up)&("NEG" in target), prepared[f"PREDICTION_{target}"], 0))
        return prepared

    def define_thresholds(self, prepared, target, seuils):
        target = f"PREDICTION_{target}"
        if "POS" in target:
            qt  = 0.95
        else:
            qt = 0.95
        seuils[f"SEUIL_{target}"] = np.quantile(prepared.loc[~prepared[target].isnull(), target], qt) # achat
        return seuils

    def load_model(self, currency, target):

        if target not in [f"POS_BINARY_FUTUR_TARGET_{self.target_lag}" , f"NEG_BINARY_FUTUR_TARGET_{self.target_lag}"]:
            logging.warning(f"model need to be either POS_BINARY_FUTUR_TARGET_{self.target_lag} or NEG_BINARY_FUTUR_TARGET_{self.target_lag}")

        liste_files = glob.glob("/".join([self.path_dirs["MODELS"], f"{currency}_{target}_*.pickle.dat"]))
        if len(liste_files)>0:
            latest_file = max(liste_files, key=os.path.getctime)
            logging.info(latest_file)
            return pickle.load(open(latest_file, "rb"))
        else:
            logging.warning(f"No model found for {target}")
    