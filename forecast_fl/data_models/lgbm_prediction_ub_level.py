#
# Copyright (c) 2023 by Boston Consulting Group. All rights reserved
#
# -*- coding: utf-8 -*-

import logging
import os
from datetime import timedelta
from pathlib import Path
from typing import Dict

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import shap
from forecast_fl.data_models.modelling_lightgbm import TrainModel

plt.ioff()
matplotlib.use("Agg")


def get_all_features(X_train_dates_ub, lgbm_config):
    target = lgbm_config["TARGET"]
    dynamic_features = list(X_train_dates_ub.filter(regex=rf"{target}_SHIFTED").columns)
    lgbm_config["FEATURES"] = lgbm_config["STATIC_FEATURES"] + dynamic_features
    return lgbm_config


def split_train_test(X_train_dates_ub, split_percentage):

    nbr_days = (X_train_dates_ub["DATE"].max() - X_train_dates_ub["DATE"].min()).days
    split_date = X_train_dates_ub["DATE"].max() - timedelta(days=int(nbr_days * split_percentage))

    train = X_train_dates_ub.loc[X_train_dates_ub["DATE"] < split_date].reset_index(drop=True)
    test = X_train_dates_ub.loc[X_train_dates_ub["DATE"] >= split_date].reset_index(drop=True)

    return train, test


def evaluate_error(x_val, target, ub_n):

    x_val[f"PREDICTION_{target}"] = x_val[f"PREDICTION_{target}"].clip(0, None)
    x_val["ERROR_PREDICTION_UB_SMOOTH"] = abs(x_val[target] - x_val[f"PREDICTION_{target}"]) / x_val[target]
    x_val["ERROR_PREDICTION_UB_REAL"] = abs(x_val["TARGET"] - x_val[f"PREDICTION_{target}"]) / x_val["TARGET"]

    avg_error_ub_smooth = x_val.loc[x_val[target] >= 1, "ERROR_PREDICTION_UB_SMOOTH"].mean()
    avg_error_ub_real = x_val.loc[x_val["TARGET"] >= 1, "ERROR_PREDICTION_UB_REAL"].mean()

    logging.info(
        f"UB NAME = {ub_n}; MAPE on Test SET"
        + f"\n SMOOTH UB ERROR = {avg_error_ub_smooth}"
        + f"\n REAL UB ERROR= {avg_error_ub_real} \n"
    )

    return x_val


def save_shap_feature_importance(base_path, ub_n, shap_values, valid_x):

    # feature importance
    file_name = f"SHAPE_ub_{ub_n}.png"
    path = base_path / str(ub_n) / "plots" / "shap"
    path.mkdir(parents=True, exist_ok=True)
    full_path = path / file_name

    plt.figure()
    shap.summary_plot(shap_values, valid_x)
    plt.savefig(str(full_path.absolute()))
    plt.close()


def save_time_series(base_path, ub_n, pdv, target, sub_pdv):

    file_name = f"LGBM_TEST_plot_time_series_ub_{ub_n}.png"
    path = base_path / str(ub_n) / "plots" / "time_series"
    path.mkdir(parents=True, exist_ok=True)
    full_path = path / file_name

    plt.figure()
    sub_pdv.set_index("DATE")[[target, f"PREDICTION_{target}", "TARGET"]].plot(
        figsize=(15, 10), alpha=0.5, title=f"UB pred {ub_n} / example for granularity = {pdv}"
    )
    plt.savefig(str(full_path.absolute()))
    plt.close()


def train_lgbm_model_per_ub(
    X_train_dates_ub: pd.DataFrame,
    ub_n: str,
    config_ub_model: Dict,
    base_path: Path,
    split_percentage: float = 0.15,
):

    lgbm_config = config_ub_model["config_ub_lgbm"]
    target = lgbm_config["TARGET"]

    # get all features list
    lgbm_config = get_all_features(X_train_dates_ub, lgbm_config)

    # split train / test
    train, test = split_train_test(X_train_dates_ub, split_percentage)

    ################ TRAIN /TEST ESTIMATION ERROR ON OOS
    tm = TrainModel(data=train, configs=lgbm_config)
    train_model = tm.train_on_set(train)
    x_val = tm.test_on_set(train_model, test)

    x_val = evaluate_error(x_val, target, ub_n)

    ################## TRAIN ON FULL DATA
    tm = TrainModel(data=X_train_dates_ub, configs=lgbm_config)
    model_lgbm_ub = tm.train_on_set(X_train_dates_ub)
    preds_all = tm.test_on_set(model_lgbm_ub, X_train_dates_ub)

    training_results_cols = [
        "DATE",
        "COD_SITE",
        "UB_CODE",
        "META_UB",
        "TARGET",
        target,
        f"PREDICTION_{target}",
        "ERROR_PREDICTION_UB_SMOOTH",
        "ERROR_PREDICTION_UB_REAL",
    ]

    x_val = x_val[training_results_cols]
    x_val["MODEL"] = config_ub_model["model_name"]

    if config_ub_model["save_plots"]:

        valid_x = test[tm.feature_name]
        shap_values = shap.TreeExplainer(train_model).shap_values(valid_x)
        save_shap_feature_importance(base_path, ub_n, shap_values, valid_x)

        # time series for top 1 pdv
        pdv = preds_all["COD_SITE"].value_counts().index[0]
        sub_pdv = preds_all.loc[preds_all["COD_SITE"] == pdv]
        save_time_series(base_path, ub_n, pdv + "_full", target, sub_pdv)

        # time series for top 1 pdv
        pdv = x_val["COD_SITE"].value_counts().index[0]
        sub_pdv = x_val.loc[x_val["COD_SITE"] == pdv]
        save_time_series(base_path, ub_n, pdv, target, sub_pdv)

    return x_val, model_lgbm_ub
