#
# Copyright (c) 2023 by Boston Consulting Group. All rights reserved
#
# -*- coding: utf-8 -*-

import matplotlib

matplotlib.use("Agg")
import logging
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

plt.ioff()
from pathlib import Path

from forecast_fl.data_models.modelling_lightgbm import TrainModel
from forecast_fl.utils.general_functions import function_weight


def add_weights_for_training(df, target):
    """PREDICTION is here the ts forecast from previous step
    No weights on target = NULL since we assign the TS predict value
    wight almost null for diff between Prophet &

    Args:
        df (_type_): _description_
        target (_type_): _description_

    Returns:
        _type_: _description_
    """

    # create weight on time
    df["DIFF"] = (df["DS"].max() - df["DS"]).dt.days
    f = function_weight()
    df["WEIGHT"] = f(df["DIFF"])

    # reduce weight when prophet too far from target -> target probably wrong
    std_diff = (df["PREDICTION"] - df[target]).std()

    df["WEIGHT"] = np.where(
        abs(df["PREDICTION"] - df[target]).between(std_diff, 1.5 * std_diff),
        0.75 * df["WEIGHT"],
        np.where(
            abs(df["PREDICTION"] - df[target]).between(1.5 * std_diff, 2.5 * std_diff),
            0.5 * df["WEIGHT"],
            np.where(abs(df["PREDICTION"] - df[target]) > 2.5 * std_diff, 0.25 * df["WEIGHT"], df["WEIGHT"]),
        ),
    )

    # outliers -> almost don't learn on it
    df["WEIGHT"] = np.where(df[target].isnull(), 0, df["WEIGHT"])

    return df


def handle_missing_output(df):
    return np.where(df["Y"].isnull(), df["PREDICTION"], df["Y"]).astype(float)


def evaluate_error(x_val, target, model_name):

    x_val[f"ERROR_PREDICTION_{model_name}"] = abs(x_val[target] - x_val[f"PREDICTION_{model_name}"]) / x_val[target]
    x_val["ERROR_PREDICTION_LGBM"] = abs(x_val[target] - x_val["PREDICTION_LGB"]) / x_val[target]

    avg_error_prophet = x_val.loc[x_val[target] >= 1, f"ERROR_PREDICTION_{model_name}"].median()
    avg_error_lgbm = x_val.loc[x_val[target] >= 1, "ERROR_PREDICTION_LGBM"].median()

    x_val["MODEL"] = "_".join(["LGBM", model_name])

    return x_val, avg_error_prophet, avg_error_lgbm


def column_renaming(x_val, parameters_from_config):

    if parameters_from_config["log_transfo"]:
        prediction_cols = [x for x in x_val.columns if "PREDICTION" in x]
        for log_col in ["Y"] + prediction_cols:
            x_val[log_col] = np.exp(x_val[log_col]) - 1

    x_val = x_val.rename(
        columns={
            "Y": parameters_from_config["target"],
            "DS": "DATE",
            "PREDICTION_Y": "PREDICTION_LGB",
            "PREDICTION": f"PREDICTION_PROPHET",
        }
    )

    x_val["PREDICTION_LGB"] = x_val["PREDICTION_LGB"].clip(0, None)
    return x_val


def save_shap_feature_importance(base_path, ub_n, shap_values, valid_x, granularity):

    # feature importance
    file_name = f"SHAPE_ub_{ub_n}_codesite_{granularity}.png"
    path = base_path / str(ub_n) / "plots" / "shap"
    path.mkdir(parents=True, exist_ok=True)
    full_path = path / file_name

    plt.figure()
    shap.summary_plot(shap_values, valid_x)
    plt.savefig(str(full_path.absolute()))
    plt.close()


def lgbm_prediction(
    full_ts: pd.DataFrame,
    train_ts: pd.DataFrame,
    test_ts: pd.DataFrame,
    ub_n: str,
    granularity: str,
    lgbm_config: Dict,
    parameters_from_config: Dict,
    base_path: Path,
):
    """train model TrainModel based on lgbm"""

    # hyper parameters
    target = parameters_from_config["target"]
    model_name = parameters_from_config["model_name"]
    debug = parameters_from_config["debug"]
    save_plots = parameters_from_config["save_plots"]

    train_ts = train_ts.copy()
    test_ts = test_ts.copy()

    # handle outleirs Y predictions on OOS
    train_ts["Y"] = handle_missing_output(train_ts)
    test_ts["Y"] = handle_missing_output(test_ts)
    full_ts["Y"] = handle_missing_output(full_ts)

    # add weights
    train_ts = add_weights_for_training(train_ts, target="Y")
    full_ts = add_weights_for_training(full_ts, target="Y")

    # train on train_ts and predict on test_ts
    tm = TrainModel(data=train_ts, configs=lgbm_config)
    _, train_model = tm.modelling_cross_validation()  # init_score="PREDICTION"
    x_val = tm.test_on_set(train_model, test_ts)  # , init_score="PREDICTION"

    # renaming and put to exp if log_transformed first
    x_val = column_renaming(x_val, parameters_from_config)

    # error and renaming of columns
    x_val, avg_error_prophet, avg_error_lgbm = evaluate_error(x_val, target, model_name)
    logging.info(
        f"UB NAME = {ub_n}; site {granularity} MAPE on Test SET"
        + f"\n {model_name} = {avg_error_prophet}"
        + f"\n LGBM= {avg_error_lgbm} \n"
    )

    # Keep only relevant columns to compute metrics and save results later
    x_val = x_val[
        [
            "DATE",
            "META_UB",
            "COD_SITE",
            "MODEL",
            "TEMPERATURE",
            target,
            "PREDICTION_PROPHET",
            "PREDICTION_LGB",
            "ERROR_PREDICTION_PROPHET",
            "ERROR_PREDICTION_LGBM",
        ]
    ]

    # train model on entire dataset
    tm = TrainModel(data=full_ts, configs=lgbm_config)
    lgbm_model = tm.train_on_set(full_ts)  # , init_score="PREDICTION"

    # handle model analysis
    if save_plots:

        # save summary plot
        valid_x = test_ts[tm.feature_name]
        shap_values = shap.TreeExplainer(train_model).shap_values(valid_x)
        save_shap_feature_importance(base_path, ub_n, shap_values, valid_x, granularity)

        # save time series
        file_name = f"LGBM_TEST_plot_time_series_ub_{ub_n}_codesite_{granularity}.png"
        path = base_path / str(ub_n) / "plots" / "time_series"
        path.mkdir(parents=True, exist_ok=True)
        full_path = path / file_name

        plt.figure()
        x_val.set_index("DATE")[[target, f"PREDICTION_{model_name}", "PREDICTION_LGB"]].plot(
            figsize=(15, 10),
            alpha=0.5,
            title=f"{ub_n}_codesite_{granularity}, \n error rate {model_name} = {avg_error_prophet} LGBM= {avg_error_lgbm}",
        )
        plt.savefig(str(full_path.absolute()))
        plt.close()

    if debug and tm.valid_x is not None:
        for name in tm.valid_x.columns:

            shap.dependence_plot(
                name,
                tm.shap_values,
                tm.valid_x,
                interaction_index=None,
                show=False,
            )
            file_name = f"plot_shape_variable_{name}_ub_{ub_n}_codesite_{granularity}.png"
            path = base_path / str(ub_n) / "plots" / "shape"
            path.mkdir(parents=True, exist_ok=True)
            plt.savefig(str(path.absolute()) + file_name)

    return x_val, lgbm_model
