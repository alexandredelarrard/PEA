
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
from data_load.data_loading import LoadCrytpo

from utils.general_functions import function_weight

warnings.filterwarnings("ignore")

class TrainingStrat2(LoadCrytpo):

    def __init__(self, path_dirs="", since=2017):

        LoadCrytpo.__init__(self)

        self.since_year = since
        self.path_dirs = path_dirs

        self.configs = self.configs.strategie["strategy_2"]
        self.configs_lgbm = self.configs["parameters"]
        self.target = self.configs["TARGET"]
        self.final_metric = None
        self.obs_per_hour = 60//self.granularity


    def data_prep(self, prepared):

        #filter year
        prepared = prepared.loc[prepared["DATE"].dt.year >= self.since_year]

        # prepared date features 
        prepared["HOUR"] = prepared["DATE"].dt.hour
        prepared["WEEK_DAY"] = prepared["DATE"].dt.dayofweek
        prepared["DAY_OF_YEAR"] = prepared["DATE"].dt.dayofyear
        prepared["DAY"] = prepared["DATE"].dt.day
        prepared["MONTH"] = prepared["DATE"].dt.month

        if self.configs["categorical_features"]:
            for col in self.configs["categorical_features"]:
                prepared[col] = prepared[col].astype("category")
        
        prepared = prepared.sort_values("DATE", ascending = False)

        # filter target 
        prepared = prepared.loc[~prepared[self.target].isnull()]

        return prepared


    def main_training(self, prepared, args={}):

        currency = args["currency"]
        self.oof_start_data = datetime.today() - timedelta(days=args["oof_days"])

        prepared = self.data_prep(prepared)
        results, model, test_ts = self.cross_validation(prepared.loc[prepared["DATE"] < self.oof_start_data])

        # if model:
        #     self.shap_values(model, test_ts, currency)
        
        # logging.info(f"TRAIN full model until OOF date = {self.oof_start_data}")
        # prepared = self.add_weights_for_training(prepared)
        # model = self.tm.train_on_set(prepared.loc[prepared["DATE"] < self.oof_start_data])

        # self.save_model(model, currency, self.oof_start_data)

        return results, model

    
    def add_weights_for_training(self, df):

        # create weight on time
        df["DIFF"] = (df["DATE"].max() - df["DATE"]).dt.days
        df["WEIGHT"] = function_weight()(df["DIFF"])

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
    
    def predicting(self, model, data):

        data["PREDICTION_" + self.target] = model.predict_proba(data[self.configs["FEATURES"]],
                                                categorical_feature=self.configs["categorical_features"])[:,1]
    
        return data
    
    def cross_validation(self, prepared):

        self.tm = TrainModel(data=prepared, configs=self.configs)
        folds = self.tm.time_series_fold(start = prepared["DATE"].min(), 
                                        end = prepared["DATE"].max(),
                                        k_folds = self.configs["n_splits"],
                                        total_test_days = self.configs["total_test_days"])
        total_test, models = pd.DataFrame(), {}

        for k, tuple_time in folds.items():

            logging.info(f"[fold {k}] START TEST DATE = {tuple_time[0]}")
            condition_train = prepared["DATE"].between(tuple_time[0], tuple_time[1])
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
