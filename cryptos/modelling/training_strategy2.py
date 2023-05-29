
import pandas as pd
import numpy as np
import pickle
import logging
import warnings
import shap
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from pathlib import Path

from modelling.modelling_lightgbm import TrainModel

from utils.general_functions import function_weight

warnings.filterwarnings("ignore")

class TrainingStrat2(object):

    def __init__(self, configs, path_dirs="", up=0.1, down=-0.1):

        self.path_dirs = path_dirs
        self.configs = configs
        self.configs_lgbm = configs["parameters"]
        self.target = configs["TARGET"]
        self.final_metric = None
        self.obs_per_hour = 1
        self.conditional_filter_up=up
        self.conditional_filter_down=down

        if "NEG" in self.target and "monotone_constraints" in self.configs["parameters"]:
            self.configs["parameters"]["monotone_constraints"]  = list(-1*np.array(self.configs["parameters"]["monotone_constraints"]))


    def main_training(self, prepared, args={}):

        currency = args["currency"]
        self.target_lag = args["lag"]

        if "POS" in self.target:
            prepared = prepared.loc[prepared["TARGET_NORMALIZED_2"] < self.conditional_filter_down]
        if "NEG" in self.target:
            prepared = prepared.loc[prepared["TARGET_NORMALIZED_2"] > self.conditional_filter_up]

        results, model, test_ts = self.cross_validation(prepared)

        if model:
            self.shap_values(model, test_ts, currency)

        oof_start_data = datetime.today() - timedelta(days=180)
        logging.info(f"TRAIN full model test start date = {oof_start_data}")
        prepared = self.add_weights_for_training(prepared)
        model = self.tm.train_on_set(prepared.loc[prepared["DATE"] < oof_start_data])

        self.save_model(model, currency, oof_start_data)

    
    def add_weights_for_training(self, df):

        # create weight on time
        df["DIFF"] = (df["DATE"].max() - df["DATE"]).dt.days
        df["WEIGHT"] = function_weight()(df["DIFF"])

        if df[self.target].nunique() == 2:
            df["WEIGHT"] = np.where(df[self.target] == 0, df["WEIGHT"], 3*df["WEIGHT"])

        return df
            
    def training(self, train_ts, test_ts):

        train_ts = train_ts.copy()
        test_ts = test_ts.copy()

        # add weights
        train_ts= self.add_weights_for_training(train_ts)

        # train on train_ts and predict on test_ts
        model = self.tm.train_on_set(train_ts, test_ts)
        x_val = self.tm.test_on_set(model, test_ts)

        return x_val, model
    

    def cross_validation(self, prepared):

        self.tm = TrainModel(data=prepared, configs=self.configs)
        folds = self.tm.time_series_fold(start = prepared["DATE"].min(), 
                                        end = prepared["DATE"].max(),
                                        k_folds = self.configs["n_splits"],
                                        total_test_days = 360,
                                        test_days = 360)
        total_test, models = pd.DataFrame(), {}

        for k, tuple_time in folds.items():

            logging.info(f"[fold {k}] START TEST DATE = {tuple_time[0]}")
            condition_train = prepared["DATE"].between(tuple_time[0], prepared["DATE"].max()) # tuple_time[1]
            train_ts = prepared.loc[prepared["DATE"] < tuple_time[0]]
            test_ts = prepared.loc[condition_train]

            logging.info(test_ts[self.target].mean())

            if test_ts.shape[0]>0 and train_ts.shape[0]>0:
                x_val, models[k] = self.training(train_ts, test_ts)
                x_val["FOLD"] = k
                
                # concatenate all test_errors
                total_test = pd.concat([total_test, x_val], axis=0).reset_index(drop=True)

        self.final_metric = self.tm.metric
    
        if models: # calculate overall aux
            logging.info(f"FINAL METRIC IS {np.mean(self.final_metric)} +- {np.std(self.final_metric)}")
            return total_test, models[k], x_val
        
        else:
            logging.info(f"NOT ENOUGH TRAINING DATA SHAPE = {prepared.shape[0] // 365*len(self.hours)*self.obs_per_hour}")
            return None, None, None

    def shap_values(self, train_model, test_ts, currency):

        valid_x = test_ts[self.tm.feature_name]
        shap_values = shap.TreeExplainer(train_model).shap_values(valid_x)

        if self.tm.params["parameters"]["objective"] == "binary":
            shap_values = shap_values[1]
        
        # feature importance
        file_name = f"SHAPE_{currency}.png"
        path = self.path_dirs["PLOTS"] + "/shap/"
        Path(path).mkdir(parents=True, exist_ok=True)
        full_path =  Path(path + file_name)

        shap.summary_plot(shap_values, valid_x)
        plt.savefig(str(full_path.absolute()))
        plt.show()


    def save_model(self, model, currency, oof_start_data):

        oof_start_data = oof_start_data.strftime("%Y-%m-%d")
        file_name = f"{currency}_{self.target}_{oof_start_data}.pickle.dat"
        lgb_path_dump = Path("/".join([self.path_dirs["MODELS"], file_name]))

        pickle.dump([model, self.configs], open(str(lgb_path_dump.absolute()), "wb"))
