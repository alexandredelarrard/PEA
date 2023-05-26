
import pandas as pd
import numpy as np
import logging
import warnings
import shap
import matplotlib.pyplot as plt
from pathlib import Path

from strategy.main_strategie import MainStrategy
from modelling.modelling_lightgbm import TrainModel

from utils.general_functions import function_weight

warnings.filterwarnings("ignore")

class Strategy2(MainStrategy):

    def __init__(self, configs, start_date, end_date =None, path_dirs="", dict_prepared={}):

        MainStrategy.__init__(self, configs, start_date, end_date, path_dirs)

        self.configs = self.configs.strategie["strategy_2"]
        self.configs_lgbm = self.configs["parameters"]
        self.target = self.configs["TARGET"]
        self.final_metric = None
        self.obs_per_hour = 1
        self.hours = range(24)

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

    def data_prep(self, prepared):

        prepared["HOUR"] = prepared["DATE"].dt.hour
        prepared["WEEK_DAY"] = prepared["DATE"].dt.dayofweek
        prepared["DAY_OF_YEAR"] = prepared["DATE"].dt.dayofyear
        prepared["DAY"] = prepared["DATE"].dt.day
        prepared["MONTH"] = prepared["DATE"].dt.month

        for col in self.configs["categorical_features"]:
            prepared[col] = prepared[col].astype("category")
        
        prepared = prepared.sort_values("DATE", ascending = False)

        for day_future in self.targets:

            hours = int(day_future*len(self.hours)*self.obs_per_hour)
            prepared[f"FUTUR_TARGET_{day_future}"] = prepared["CLOSE_NORMALIZED"].rolling(window=hours, center=True).mean().shift(hours)
            prepared[f"POS_BINARY_FUTUR_TARGET_{day_future}"] = 1*(prepared[f"FUTUR_TARGET_{day_future}"] >= prepared["CLOSE_NORMALIZED"]*1.05)
            prepared[f"NEG_BINARY_FUTUR_TARGET_{day_future}"] = 1*(prepared["CLOSE_NORMALIZED"]*0.95 >= prepared[f"FUTUR_TARGET_{day_future}"])
            prepared[f"DELTA_FUTUR_TARGET_{day_future}"] = (prepared[f"FUTUR_TARGET_{day_future}"] - prepared["CLOSE_NORMALIZED"]) *10 / prepared["CLOSE_NORMALIZED"]

        return prepared


    def main_strategy(self, dict_prepared, currency, df_init=None):

        prepared = dict_prepared[currency]
        prepared = prepared.loc[prepared["DATE"].dt.year >= 2018]
        prepared = self.data_prep(prepared)
        prepared = prepared.loc[~prepared[self.target].isnull()]

        # import seaborn as sns 
        # sns.regplot(x="CLOSE_TREND_45", y="DELTA_FUTUR_TARGET_1", data=prepared)

        results, model, test_ts = self.cross_validation(prepared)
        self.shap_values(model, test_ts, currency)

        return results, model, test_ts
    
    
    def add_weights_for_training(self, df):

        # create weight on time
        df["DIFF"] = (df["DATE"].max() - df["DATE"]).dt.days
        df["WEIGHT"] = function_weight()(df["DIFF"])

        return df
            

    def training(self, train_ts, test_ts):

        train_ts = train_ts.copy()
        test_ts = test_ts.copy()

        # add weights
        train_ts = self.add_weights_for_training(train_ts)

        # train on train_ts and predict on test_ts
        model = self.tm.train_on_set(train_ts, test_ts)
        x_val = self.tm.test_on_set(model, test_ts)

        return x_val, model


    def cross_validation(self, prepared):

        self.tm = TrainModel(data=prepared, configs=self.configs)
        folds = self.tm.time_series_fold(start = prepared["DATE"].min(), 
                                        end = prepared["DATE"].max(),
                                        k_folds = self.configs["n_splits"],
                                        total_test_days = 300,
                                        test_days = 180)
        total_test, models = pd.DataFrame(), {}

        for k, tuple_time in folds.items():

            logging.info(f"[fold {k}] START TEST DATE = {tuple_time[0]}")
            condition_train = prepared["DATE"].between(tuple_time[0], prepared["DATE"].max()) # tuple_time[1]
            train_ts = prepared.loc[prepared["DATE"] < tuple_time[0]]
            test_ts = prepared.loc[condition_train]

            logging.info(test_ts[self.target].mean())

            if test_ts.shape[0]>0:
                x_val, models[k] = self.training(train_ts, test_ts)
                x_val["FOLD"] = k
                
                # concatenate all test_errors
                total_test = pd.concat([total_test, x_val], axis=0).reset_index(drop=True)

        # logging.info("TRAIN full model")
        # prepared = self.add_weights_for_training(prepared)
        # model = self.tm.train_on_set(prepared)

        self.final_metric = self.tm.metric
      
        # calculate overall aux
        logging.info(f"FINAL METRIC IS {np.mean(self.final_metric)} +- {np.std(self.final_metric)}")

        return total_test, models[k-1], test_ts
    

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


    def predicting(self):
        return 0
    
    def load_model(self):
        return 0
    
    def save_model(self):
        return 0
