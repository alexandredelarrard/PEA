#
# Copyright (c) 2023 by Boston Consulting Group. All rights reserved
#
# -*- coding: utf-8 -*-

import logging
import warnings

import lightgbm as lgb
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn import metrics
from sklearn.model_selection import KFold, StratifiedKFold

warnings.filterwarnings("ignore")
matplotlib.use("Agg")
plt.ioff()

# parameter for custom loss
alpha = 3


def evaluation_metric(true, pred):
    """
    We use absolute percentage error to evaluate model performance.
    """

    return abs(true - pred) * 100 / true


def evaluation_auc(true, pred):
    """
    Calculate ROC AUC for binary target

    Args:
        true ([int]): [binary target to predict ]
        pred ([float]): [predicted probability of the model]

    Returns:
        [float]: [AUC]

    """

    fpr, tpr, _ = metrics.roc_curve(true, pred, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    return auc


def linear_exponential_loss(train_data, preds):
    if type(preds) == np.ndarray:
        error = train_data - preds
    else:
        error = train_data - preds.get_label()
    # 1st derivative of loss function
    grad = 2 / alpha * (np.exp(alpha * error) - 1)

    # 2nd derivative of loss function
    hess = 2 * np.exp(alpha * error)
    return grad, hess


def linear_exponential_eval(train_data, preds):
    if type(preds) == np.ndarray:
        error = train_data - preds
    else:
        error = train_data - preds.get_label()
    loss = (2 / alpha**2) * (np.exp(alpha * error) - alpha * error - 1)
    return "linear_exponential", loss.mean(), False


class TrainModel:
    """
    Wrapper around LightGBM model responsible for
        * fitting the LightGBM model on in put data.
        * evaluating model performance on a tests dataset.
        * Training on the overall dataset if no test set is given

    Args:
        object {[type]} --

    """

    def __init__(self, configs, data):
        self.data = data
        self.target_name = configs["TARGET"].upper()
        self.feature_name = [x.upper() for x in configs["FEATURES"]]
        self.params = configs
        self.weight = None
        self.total_test = pd.DataFrame()
        self.shap_values = None
        self.valid_x = None
        self._debug = configs["debug"]

        if "WEIGHT" in configs.keys():
            self.weight = configs["WEIGHT"]
        if "weight" in configs.keys():
            self.weight = configs["weight"]

    def calculate_shap(self, model, test_data):

        self.valid_x = test_data[self.feature_name]
        self.shap_values = shap.TreeExplainer(model).shap_values(self.valid_x)

        if self._debug:
            shap.summary_plot(self.shap_values, self.valid_x)

    def evaluate_model(self, test_data):
        """
        Plot feature importances and log error metrics.
        """

        # evaluate results based on evaluation metrics
        if self.params["parameters"]["objective"] in ["regression", "none"]:
            logging.info("_" * 60)
            for error in ["PREDICTION_" + self.target_name]:
                if error in test_data.columns:
                    test_data["ABS_BIAS_" + error] = abs(test_data[self.target_name] - test_data[error])
                    test_data["BIAS_" + error] = test_data[self.target_name] - test_data[error]
                    test_data["ERROR_" + error] = evaluation_metric(test_data[self.target_name], test_data[error])

                    logging.info(
                        "MAPE {2} : {0:.2f} +/- {1:.2f} % || ABS_BIAS {3:.3f} || BIAS {4:.3f}".format(
                            np.mean(test_data["ERROR_" + error].loc[test_data[self.target_name] > 0]),
                            np.std(test_data["ERROR_" + error].loc[test_data[self.target_name] > 0]),
                            error,
                            test_data["ABS_BIAS_" + error].mean(),
                            test_data["BIAS_" + error].mean(),
                        )
                    )
            logging.info("_" * 60)
        else:
            logging.info("_" * 60)
            for error in ["PREDICTION_" + self.target_name]:
                if error in test_data.columns:
                    logging.info(
                        "AUC {1} : {0:.4f} %".format(
                            evaluation_auc(test_data[self.target_name], test_data[error]),
                            error,
                        )
                    )
            logging.info("_" * 60)

    def train_on_set(self, train_data, test_data=None, init_score=None):
        """
        Creates LightGBM model instance and trains on a train dataset. If a tests set is provided,
        we validate on this set and use early stopping to avoid over-fitting.
        """

        if isinstance(test_data, pd.DataFrame):
            if self.weight:
                train_weight = train_data[self.weight]
                test_weight = test_data[self.weight]
            else:
                train_weight = None
                test_weight = None

            if init_score:
                train_init_bias = train_data[init_score]
                test_init_bias = test_data[init_score]
            else:
                train_init_bias = None
                test_init_bias = None

            # model training and prediction of val
            # have an idea of the error rate and use early stopping round
            train_data = lgb.Dataset(
                train_data[self.feature_name],
                label=train_data[self.target_name],
                weight=train_weight,
                init_score=train_init_bias,
                categorical_feature=self.params["categorical_features"],
            )

            val_data = lgb.Dataset(
                test_data[self.feature_name],
                label=test_data[self.target_name],
                weight=test_weight,
                init_score=test_init_bias,
                categorical_feature=self.params["categorical_features"],
            )

            model = lgb.train(
                self.params["parameters"],
                num_boost_round=self.params["parameters"]["num_iteration"],
                train_set=train_data,
                valid_sets=[train_data, val_data],
                valid_names=["data_train", "data_valid"],
                callbacks=[lgb.log_evaluation(self.params["parameters"]["verbose"])],
            )

        else:
            if "early_stopping_round" in self.params["parameters"]:
                self.params["parameters"].pop("early_stopping_round", None)

            if self.weight:
                sample_weight = train_data[self.weight]
            else:
                sample_weight = None

            if init_score:
                init_bias = train_data[init_score]
            else:
                init_bias = None

            model = lgb.LGBMRegressor(
                n_estimators=self.params["parameters"]["num_iteration"],
                learning_rate=self.params["parameters"]["learning_rate"],
                max_depth=self.params["parameters"]["max_depth"],
                subsample=self.params["parameters"]["subsample"],
                reg_lambda=self.params["parameters"]["lambda_l1"],
                colsample_bytree=self.params["parameters"]["colsample_bytree"],
                min_child_samples=self.params["parameters"]["min_data_in_leaf"],
                n_jobs=-self.params["parameters"]["n_jobs"],
                random_state=self.params["seed"],
            )

            model.fit(
                train_data[self.feature_name],
                train_data[self.target_name],
                sample_weight=sample_weight,
                init_score=init_bias,
                categorical_feature=self.params["categorical_features"],
            )

        return model

    def test_on_set(self, model, test_data, init_score=None):
        """
        Takes model and tests dataset as input, computes predictions for the tests dataset, and evaluates metric
        on the predictions. Returns tests dataset with added columns for pediction and metrics.
        """

        test_data["PREDICTION_" + self.target_name] = model.predict(
            test_data[self.feature_name],
            categorical_feature=self.params["categorical_features"],
        )
        if init_score:
            test_data["PREDICTION_" + self.target_name] = (
                test_data["PREDICTION_" + self.target_name] + test_data[init_score]
            )

        test_data["ERROR_MODEL"] = evaluation_metric(
            test_data[self.target_name],
            test_data["PREDICTION_" + self.target_name],
        )

        # self.evaluate_model(model, test_data)

        return test_data

    def modelling_cross_validation(self, data=None, init_score=None):
        """
        Fits model using k-fold cross-validation on a train set.
        """

        if not isinstance(data, pd.DataFrame):
            data = self.data.reset_index(drop=True)
        else:
            data = data.reset_index(drop=True)

        if self.params["parameters"]["objective"] == "binary":
            kf = StratifiedKFold(
                n_splits=self.params["n_splits"],
                random_state=self.params["seed"],
                shuffle=True,
            )
        else:
            kf = KFold(
                n_splits=self.params["n_splits"],
                random_state=self.params["seed"],
                shuffle=True,
            )

        total_test = pd.DataFrame()
        for train_index, val_index in kf.split(data.index, data.index):
            train_data = data.loc[train_index]
            test_data = data.loc[val_index]

            model = self.train_on_set(train_data, test_data, init_score)
            x_val = self.test_on_set(model, test_data, init_score)

            # concatenate all test_errors
            total_test = pd.concat([total_test, x_val], axis=0).reset_index(drop=True)

        self.calculate_shap(model, total_test)
        self.total_test = total_test

        logging.info("TRAIN full model")
        model = self.train_on_set(data, init_score=init_score)

        return total_test, model
