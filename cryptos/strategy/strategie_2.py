
import pandas as pd
import logging

from strategy.main_strategie import MainStrategy
from modelling.modelling_lightgbm import TrainModel
from utils.general_functions import function_weight

class Strategy2(MainStrategy):

    def __init__(self, configs, start_date, end_date =None):

        MainStrategy.__init__(self, configs, start_date, end_date)

        self.targets = [0.125, 0.25, 0.5, 1, 2, 3, 4]
        self.picked_strategie = self.main_strategy_2
        self.configs = self.configs.strategie["strategy_2"]
        self.configs_lgbm = self.configs["parameters"]
        self.target = self.configs["TARGET"]

        # - probabilité de hausse next 0.25, 0.5, 1, 2, 3, 4, 5, 10, 15 jours (modelling) -> not BTC / ETH becasue is market 
        #   - diff to mean 0.25, 0.5, etc. 
        #   - std 0.25, etc. 
        #   - diff to market 
        #   - diff to s&p500 TODO
        #   - diff to gold TODO
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
        # - diff to market index -> buy the lower one, sell the higher one 

    def data_prep(self, prepared):

        prepared = prepared.sort_values("DATE", ascending = False)

        for day_future in self.targets:

            hours = int(day_future*24)
            prepared[f"TARGET_PLUS_{day_future}"] = prepared["CLOSE_NORMALIZED"].shift(hours)
            prepared[f"BINARY_TARGET_PLUS_{day_future}"] = 1*(prepared[f"TARGET_PLUS_{day_future}"] > prepared["CLOSE_NORMALIZED"])

            #date infos 
            prepared["HOUR"] = prepared["DATE"].dt.hour + hours 
            prepared["WEEK_DAY"] = prepared["DATE"].dt.dayofweek
            prepared["DAY"] = prepared["DATE"].dt.day
            prepared["MONTH"] = prepared["DATE"].dt.month

            for col in self.configs["categorical_features"]:
                prepared[col] = prepared[col].astype("category")

        return prepared


    def main_strategy_2(self, prepared):

        prepared = dict_prepared["ETH"]
        prepared = self.data_prep(prepared)
        prepared = prepared.loc[~prepared[self.target].isnull()]

        results, model = self.cross_validation(prepared)

        return pd.DataFrame(), pd.DataFrame()
    
    
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
                                        end = prepared["DATE"].max())
        
        total_test, models = pd.DataFrame(), {}

        for k, tuple_time in folds.items():

            condition_train = prepared["DATE"].between(tuple_time[0], tuple_time[1])
            train_ts = prepared.loc[prepared["DATE"] < tuple_time[0]]
            test_ts = prepared.loc[condition_train]

            x_val, models[k] = self.training(train_ts, test_ts)
            x_val["FOLD"] = k
            
            # concatenate all test_errors
            total_test = pd.concat([total_test, x_val], axis=0).reset_index(drop=True)

        logging.info("TRAIN full model")
        model = self.tm.train_on_set(prepared)

        return total_test, model
    

    def predicting(self):
        return 0
    
    def load_model(self):
        return 0
    
    def save_model(self):
        return 0
