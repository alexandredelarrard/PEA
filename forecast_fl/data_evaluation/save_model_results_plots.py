#
# Copyright (c) 2023 by Boston Consulting Group. All rights reserved
#
# -*- coding: utf-8 -*-

import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import arrow
import pandas as pd
from forecast_fl.utils.config import Config


def create_and_clean_saving_directories(output_save_path: str) -> Dict:
    """
    Create saving directories for training results and models and prediction, and post processing results

    Args:
        parameters_from_config (Box): configuration file (training or prediction configs are passed depending on train/predict pipeline)
        output_save_path (str): Path of where the output_save_path folder holds.

    Returns:
        path_by_directory (Dict): paths to each folders

    """

    # Architecture is as follows:
    # INTERMEDIATE:
    # TRAIN:
    #     - training/
    #         - model_training_results/
    #             - TIMESTAMPED_FOLDER/
    #               - ub_n
    #                 - plots/
    #                     - shap/
    #                     - time_series/
    #                 - results/
    #         - saved_trained_models/
    #             - TIMESTAMPED_FOLDER/
    # PREDICT:
    #     - prediction_results/
    #         - model_prediction_results/
    # POST_PROCESSING
    #     - post_processed_results/

    root_path_save = output_save_path
    path_by_directory = {}
    today = datetime.now().date()

    if not os.path.isdir(root_path_save):
        os.mkdir(root_path_save)
    if not os.path.isdir("/".join([root_path_save, "intermediate"])):
        os.mkdir("/".join([root_path_save, "intermediate"]))
    path_by_directory["intermediate"] = "/".join([root_path_save, "intermediate"])

    # First layer folder structure
    training_path = "/".join([root_path_save, "training"])
    prediction_path = "/".join([root_path_save, "prediction_results"])

    if not os.path.isdir(training_path):
        os.mkdir(training_path)
    if not os.path.isdir(prediction_path):
        os.mkdir(prediction_path)

    # Second layer folder structure
    if not os.path.isdir("/".join([training_path, "saved_trained_models"])):
        os.mkdir("/".join([training_path, "saved_trained_models"]))
    if not os.path.isdir("/".join([training_path, "model_train_results"])):
        os.mkdir("/".join([training_path, "model_train_results"]))
    if not os.path.isdir("/".join([training_path, "saved_trained_ub_models"])):
        os.mkdir("/".join([training_path, "saved_trained_ub_models"]))
    if not os.path.isdir("/".join([training_path, "ub_model_train_results"])):
        os.mkdir("/".join([training_path, "ub_model_train_results"]))

    path_by_directory["model"] = "/".join([training_path, "saved_trained_models"])
    path_by_directory["ub_model"] = "/".join([training_path, "saved_trained_ub_models"])
    path_by_directory["train_results"] = "/".join([training_path, "model_train_results"])
    path_by_directory["ub_train_results"] = "/".join([training_path, "ub_model_train_results"])

    if not os.path.isdir("/".join([prediction_path, "model_prediction_results"])):
        os.mkdir("/".join([prediction_path, "model_prediction_results"]))
    if not os.path.isdir("/".join([prediction_path, "ub_model_prediction_results"])):
        os.mkdir("/".join([prediction_path, "ub_model_prediction_results"]))
    path_by_directory["prediction_results"] = "/".join([prediction_path, "model_prediction_results"])
    path_by_directory["ub_prediction_results"] = "/".join([prediction_path, "ub_model_prediction_results"])

    # keep it along the entire code
    path_by_directory["root_path_save"] = root_path_save

    return path_by_directory


def write_training_results(
    df_list: List[pd.DataFrame],
    sheet_list: List[str],
    path: Path,
    config: Config,
    date_to_save: str,
) -> None:

    path.mkdir(parents=True, exist_ok=True)

    for dataframe, sheet in zip(df_list, sheet_list):
        model_name = str(config.load.parameters_training_model["model_name"])
        tree_model = str(config.load.parameters_training_model["tree_model"])
        granularity = str(config.load["prediction_granularity"])
        horizon = str(config.load["prediction_horizon"])

        file_name = f"{date_to_save}_{sheet}_{tree_model}_{model_name}_{granularity}_{horizon}_j.csv"
        file_path = path / file_name

        dataframe.to_csv(str(file_path.absolute()), index=False, sep=";")


def write_prediction_results(
    df_list: List[pd.DataFrame],
    sheet_list: List[str],
    date_to_save: str,
    config: Config,
    path: Path,
) -> None:

    path.mkdir(parents=True, exist_ok=True)

    model_name = str(config.load.prediction_mode["model_name"])
    tree_model = str(config.load.prediction_mode["tree_model"])
    date_min = config.load["prediction_date_min"].strftime("%Y-%m-%d")
    date_max = config.load["prediction_date_max"].strftime("%Y-%m-%d")
    granularity = str(config.load["prediction_granularity"])

    file_name = f"{date_to_save}_{tree_model}_{model_name}_{granularity}_from_{date_min}_to_{date_max}.xlsx"
    file_path = path / file_name

    logging.info(f"Writing results into {str(file_path.absolute())}")
    with pd.ExcelWriter(str(file_path.absolute()), engine="auto") as writer:
        for dataframe, sheet in zip(df_list, sheet_list):
            dataframe.to_excel(writer, sheet_name=sheet, index=False)
