#
# Copyright (c) 2023 by Boston Consulting Group. All rights reserved
#
# -*- coding: utf-8 -*-
import datetime
import logging
import os
import re
from typing import List

import pandas as pd


def find_model_from_name(
    trained_model_folder_str: str,
    elements_to_filter_in_title: List,
    prediction_date: datetime.datetime,
    prediction_horizon: int,
):

    run_folders = [f.name for f in os.scandir(trained_model_folder_str) if f.is_dir()]

    available_models = {}

    # Extract
    # Iterates over run timestamps
    for run in run_folders:
        try:
            run_ts = datetime.datetime.strptime(run, "%d_%m_%Y_%H_%M_%S")
        except ValueError:
            continue
        candidates = [m for m in os.listdir(os.path.join(trained_model_folder_str, run))]
        for element_to_filter in elements_to_filter_in_title:
            candidates = list(
                filter(
                    lambda file: element_to_filter in file,
                    candidates,
                )
            )
        if not candidates:
            continue

        # Extracts latest train date
        for c in candidates:
            latest_train_date = pd.to_datetime(
                re.search(r"(\d+-\d+-\d+)", c)[0], format="%d-%m-%Y"
            )  # TODO: change date format unconsistant
            if latest_train_date <= prediction_date - pd.Timedelta(
                days=prediction_horizon
            ):  # Remove overfitting  # Remove overfitting
                available_models[(latest_train_date, run_ts)] = os.path.join(
                    trained_model_folder_str, run, c
                )  # Adds in available model if no overfitting
            else:
                logging.info(
                    f"Candidate model removed due to overfitting, found hist end date {latest_train_date} "
                    f"when trying to predict {prediction_date}"
                )
                continue

    if available_models:
        last_model_in_dic = sorted(available_models)[-1]
        model = available_models[last_model_in_dic]
        logging.info(
            "Multiple eligible models found. Taking last one available based first "
            "on latest train date and then on latest run date if multiple trained on same date"
        )
        logging.info(f"Candidate available and returned : {model}")
        return model
    else:
        return None


def check_model_is_not_overfitting(model_file_name, date_to_predict, ub_log_flag: str = ""):
    """

    Args:
        model_file_name (str): model file name
        date_to_predict (datetime.date): date to predict
        ub_log_flag (str): prefix to add in loggings in case of a prediction at UB level, empty by default

    Returns: None, only flags in logs when the model chosen is overfitting

    """
    logging.info(f"Found model - using {model_file_name} for prediction")

    # check history is before date to predict from model title
    history_end = re.search(r"(\d+-\d+-\d+)", model_file_name).group(1)

    if len(history_end) == 10:  # checks correct format of date str
        history_end = pd.to_datetime(history_end, format="%d-%m-%Y")  # TODO : parametrize date format
        if date_to_predict <= history_end:
            logging.error(
                f"{ub_log_flag}[MODEL OVERFITTING] Model is fitted on date to predict, \
                            results won't be representative of the futur behavior of the model"
            )  # TODO : throw error ?
    else:
        logging.warning(f"{ub_log_flag}[MODEL] No end date in model title, BEWARE OF THE MODEL VALIDITY !")
