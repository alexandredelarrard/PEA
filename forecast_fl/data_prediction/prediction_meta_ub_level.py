#
# Copyright (c) 2023 by Boston Consulting Group. All rights reserved
#
# -*- coding: utf-8 -*-

import logging
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from forecast_fl.data_prediction.utils_models import find_model_from_name
from forecast_fl.utils.config import Config
from tqdm import tqdm


def get_model_name(
    config: Config,
    ub_n: str,
    site_granularity: str,
    prediction_horizon: int,
) -> Optional[str]:
    """Attempts to find pre-trained model for given unité de besoin, point de vente.
    Behaviour is the following: List all models matching the UB x PDV. Removes if pred date - horizon > last train
    date (no leakage). If multiple models available, takes the last one based on last train date and
    then on last run date

    Args:
        config_load: Config
        trained_model_folder_str: Directory of the saved models
        ub_n (str): META UB number of given model
        site_granularity (str): Code of PDV model, cluster model or BASE model to load depending on defined granularity
        prediction_date (pd.Timestamp): Date of prediction. Used to check overfitting
        prediction_horizon (int): Prediction horizon to predict (diff between latest features date and prediction date in days)

    Warnings:
        Careful of folder structure. If changed, this function could be obsoletet. If no model, function returns None

    """

    # Title filters to find right model
    elements_to_filter_in_title = [
        "trained_model_until_",
        "ub_" + ub_n,
        config.load.prediction_mode["tree_model"],
        config.load.prediction_mode["model_name"],
        str(prediction_horizon) + "j",
    ]

    site_prefix = ""
    if config.load["prediction_granularity"] == "PDV":
        site_prefix = "codesite_"
    elif config.load["prediction_granularity"] == "CLUSTER":
        site_prefix = "cluster_"

    elements_to_filter_in_title.append(site_prefix + site_granularity.lower())

    if config.load.prediction_mode["log_transfo"]:
        elements_to_filter_in_title.append("log_transfo")

    return elements_to_filter_in_title


def compute_prediction(
    prediction_input: pd.DataFrame, model_file_name: Optional[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    if model_file_name:
        prediction_input = prediction_input.reset_index(drop=True)

        lgbm_model, ts_model, config_loaded = pickle.load(open(model_file_name, "rb"))

        ts_prediction_input = prediction_input[["DATE"]].rename(columns={"DATE": "ds"})
        preds_ts = ts_model.predict(ts_prediction_input)["yhat"].values
        prediction_input["PREDICTION"] = preds_ts

        preds_lgbm = lgbm_model.predict(
            prediction_input[config_loaded.load["config_lgbm"]["FEATURES"]],
            categorical_feature=config_loaded.load["config_lgbm"]["categorical_features"],
        )

        prediction_input["PREDICTION_TS"] = preds_ts
        prediction_input["PREDICTION_LGBM"] = preds_lgbm
        prediction_input["PREDICTION"] = preds_lgbm

    return prediction_input


def looping_prediction(
    prediction_input,
    date_to_predict,
    prediction_horizon,
    meta_ub,
    granularity,
    config: Config,
    paths_by_directory: Dict,
):

    sub_prediction_input = prediction_input.loc[
        (prediction_input["DATE"] == date_to_predict)
        & (prediction_input["META_UB"] == meta_ub)
        & (prediction_input["COD_SITE"] == granularity)
    ]

    if not sub_prediction_input.empty:

        # Find pretrained META UB model fitting the right horizon
        elements_to_filter_in_title = get_model_name(
            config=config,
            ub_n=meta_ub,
            site_granularity=str(granularity),
            prediction_horizon=prediction_horizon,
        )

        model_file_name = find_model_from_name(
            paths_by_directory["model"], elements_to_filter_in_title, date_to_predict, prediction_horizon
        )

        if model_file_name:
            logging.info(f"PREDICTING META UB LEVEL: Found model - using {model_file_name} for prediction")

            # Compute prediction for date x meta_ub x pdv/cluster/base
            sub_prediction_output = compute_prediction(sub_prediction_input, model_file_name)

            sub_prediction_output["PREDICTION_HORIZON"] = prediction_horizon
            return sub_prediction_output

        else:
            logging.critical(f"PREDICTING META UB: No model found for {meta_ub} in site {granularity}")

    else:
        logging.critical(
            f"PREDICTION: Exit as no data available for META code {meta_ub} "
            + f"and {config.load['prediction_granularity']} {granularity}"
        )

    return pd.DataFrame()


def predict_meta_ub_level(
    prediction_input: pd.DataFrame,
    config: Config,
    paths_by_directory: Dict,
    tuple_meta_ub_granularity: List,
):
    """
    Loop over each [Date, Meta UB, Code Site] to:
        - extract right Meta UB model
        - Use the model to predict Meta UB target
        - Save each Meta UB prediction in a dict which keys are dates to predict

    Args:
        prediction_input_ub: prediction input with features created at Meta UB level
        configs (config class): contains parameters in the config
        paths_by_directory (dict): dictionnary with paths where to save each type of results
        tuple_meta_ub_granularity: Tuples [META_UB, COD_SITE] to iterate on

    Returns:
        dict containing all (META_UB x COD_SITE) predictions which keys are all dates to predict

    """
    predictions_meta_ub_raw = {}

    for date_to_predict in tqdm(prediction_input["DATE"].unique()):

        # Deduce from each date prediction horizon if it's j+2, j+3...
        prediction_horizon = (date_to_predict - config.load["histo_sales_end_date"]).days
        output_pred_meta_ub = pd.DataFrame()

        for (meta_ub, granularity) in tqdm(tuple_meta_ub_granularity):
            sub_prediction_output = looping_prediction(
                prediction_input,
                date_to_predict,
                prediction_horizon,
                meta_ub,
                granularity,
                config,
                paths_by_directory,
            )

            if sub_prediction_output.shape[0] > 0:
                output_pred_meta_ub = pd.concat([output_pred_meta_ub, sub_prediction_output])

        # transform global outputs to be in kg not log
        if config.load.parameters_training_model["log_transfo"] and not output_pred_meta_ub.empty:
            output_pred_meta_ub["PREDICTION"] = (np.exp(output_pred_meta_ub["PREDICTION"]) - 1).round(2).clip(0, None)
            output_pred_meta_ub["PREDICTION_TS"] = np.exp(output_pred_meta_ub["PREDICTION_TS"]) - 1
            output_pred_meta_ub["PREDICTION_LGBM"] = (
                output_pred_meta_ub["PREDICTION"] - output_pred_meta_ub["PREDICTION_TS"]
            )

            logged_lags_cols = [x for x in output_pred_meta_ub.columns if "Y-HORIZON-" in x]
            for col in logged_lags_cols:
                output_pred_meta_ub[col] = np.exp(output_pred_meta_ub[col]) - 1

        predictions_meta_ub_raw[date_to_predict] = output_pred_meta_ub

    return predictions_meta_ub_raw
