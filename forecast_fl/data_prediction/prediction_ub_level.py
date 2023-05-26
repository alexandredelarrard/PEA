#
# Copyright (c) 2023 by Boston Consulting Group. All rights reserved
#
# -*- coding: utf-8 -*-

import logging
import pickle
from typing import Dict, Optional

import pandas as pd
from forecast_fl.data_prediction.utils_models import find_model_from_name
from forecast_fl.utils.config import Config
from tqdm import tqdm


def get_model_name(
    ub_n: str,
    prediction_horizon: int,
    prediction_granularity: str,
) -> Optional[str]:
    """Attempts to find pre-trained model for given UB
    Behaviour is the following: List all models matching the UB. Removes if pred date - horizon > last train
    date (no leakage). If multiple models available, takes the last one based on last train date and
    then on last run date

    Args:
        config_load: Config
        trained_model_folder_str: Directory of the saved models
        ub_n (str): META UB number of given model
        prediction_date (pd.Timestamp): Date of prediction. Used to check overfitting
        prediction_horizon (int): Prediction horizon to predict (diff between latest features date and prediction date in days)

    Warnings:
        Careful of folder structure. If changed, this function could be obsoletet. If no model, function returns None

    """

    granularity_site = "pdvs"
    if prediction_granularity == "BASE":
        granularity_site = "bases"

    # Title filters to find right model
    elements_to_filter_in_title = [
        "until_",
        "ub_" + ub_n,
        "UB_LGBM",
        str(prediction_horizon) + "j",
        granularity_site,
    ]

    return elements_to_filter_in_title


def prediction_from_model_ub(
    config: Config, sub_prediction_input_ub: pd.DataFrame, model_file_name: str, prediction_horizon: int
) -> pd.DataFrame:
    """preform prediction from model found at ub level

    Args:
        sub_prediction_input_ub (pd.DataFrame): input dataframe for prediction
        model_file_name (str): path of the model to load
        prediction_horizon (int): horizon of prediction

    Returns:
        pd.DataFrame: prediction dataframe
    """

    prediction_ub_formatted = sub_prediction_input_ub[["DATE", "META_UB", "UB_CODE", "COD_SITE"]].copy()

    # Load pretrained model and predict
    model_lgbm_ub, _ = pickle.load(open(model_file_name, "rb"))

    prediction_ub_formatted["PREDICTION_LGBM_LEVEL_UB"] = model_lgbm_ub.predict(
        sub_prediction_input_ub[model_lgbm_ub.feature_name_],
        categorical_features=config.load.parameters_training_ub_model.config_ub_lgbm["categorical_features"],
    )
    prediction_ub_formatted["PREDICTION_HORIZON"] = prediction_horizon

    return prediction_ub_formatted


def predict_ub_level(
    prediction_input_ub: pd.DataFrame,
    top_ubs: pd.DataFrame,
    config: Config,
    paths_by_directory: Dict,
):
    """
    Loop over each [Date, UB Code] to:
        - extract right UB model
        - Use the model to predict UB smoothed target
        - Save each UB prediction in a dict which keys are dates to predict

    Args:
        prediction_input_ub: prediction input with features created at UB level
        top_ubs: Filtered UBs in scope
        configs (config class): contains parameters in the config
        paths_by_directory (dict): dictionnary with paths where to save each type of results

    Returns:
        dict containing all UBs predictions which keys are all dates to predict

    """

    predictions_ub_raw = {}

    prediction_granularity = config.load["prediction_granularity"]

    # Iterate over all dates to predict
    for date_to_predict in tqdm(prediction_input_ub["DATE"].unique()):

        prediction_ub_dated_output = pd.DataFrame()
        prediction_horizon = (date_to_predict - config.load["histo_sales_end_date"]).days

        # Iterate over each ub code to find the right model
        for ub_code in tqdm(top_ubs.UB_CODE.unique()):

            sub_prediction_input_ub = prediction_input_ub[
                (prediction_input_ub["DATE"] == date_to_predict) & (prediction_input_ub["UB_CODE"] == ub_code)
            ]

            if not sub_prediction_input_ub.empty:
                # Deduce from each date prediction horizon if it's j+2, j+3...

                logging.info(f"Prediction horizon defined as J+{prediction_horizon}")
                logging.info(f"PREDICTING UB {ub_code}  for date {date_to_predict} Horizon : +{prediction_horizon}J")

                # Find pretrained UB model fitting the right horizon
                # check no unrealistic horizon to check on
                if prediction_horizon <= 5:

                    elements_to_filter_in_title = get_model_name(
                        ub_n=ub_code,
                        prediction_horizon=prediction_horizon,
                        prediction_granularity=prediction_granularity,
                    )

                    model_file_name = find_model_from_name(
                        paths_by_directory["ub_model"],
                        elements_to_filter_in_title,
                        prediction_date=date_to_predict,
                        prediction_horizon=prediction_horizon,
                    )

                else:
                    model_file_name = None

                # Feature engineering only when a pretrained model is found
                if model_file_name:
                    logging.info(
                        f"PREDICTING UB LEVEL: Found model - using {model_file_name} for prediction of ub {ub_code}"
                    )

                    prediction_ub_formatted = prediction_from_model_ub(
                        config, sub_prediction_input_ub, model_file_name, prediction_horizon
                    )

                    prediction_ub_dated_output = pd.concat([prediction_ub_dated_output, prediction_ub_formatted])
                else:
                    logging.critical(f"PREDICTING UB: No model found for UB {ub_code}")

            else:
                logging.critical(f"PREDICTION: Exit as no data available for UB code {ub_code} ")

        # Return predictions of all (UB x COD_SITE) for each date to predict
        predictions_ub_raw[date_to_predict] = prediction_ub_dated_output

    return predictions_ub_raw
