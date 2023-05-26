#
# Copyright (c) 2023 by Boston Consulting Group. All rights reserved
#
# -*- coding: utf-8 -*-

import logging
import os
import re
from pathlib import Path as pl
from typing import Dict, List

import numpy as np
import pandas as pd
from forecast_fl.utils.config import Config


def load_configs(config_path, config_name) -> Dict:
    config_yaml = open(pl(config_path) / pl(config_name), encoding="utf-8")
    configs = Config(config_yaml).read()
    return configs


def smart_column_parser(col_names: List) -> List:
    """Rename columns to upper case and remove space and
    non conventional elements

    Args:
        col_names (List): original column names

    Returns:
        List: clean column names
    """

    new_list = []
    for var in col_names:
        var = str(var).replace("/", " ")  # replace space by underscore
        new_var = re.sub("[ ]+", "_", str(var))  # replace space by underscore
        new_var = re.sub("[^A-Za-z0-9_]+", "", new_var)  # only alphanumeric characters and underscore are allowed
        new_var = re.sub("_$", "", new_var)  # variable name cannot end with underscore
        new_var = new_var.upper()  # all variables should be upper case
        new_list.append(new_var)
    return new_list


def save_data_clean(df, configs, file_to_save):

    # save to csv
    BASE_PATH = pl(configs["BASE_PATH"])

    if not os.path.isdir(BASE_PATH / "clean_data"):
        os.mkdir(BASE_PATH / "clean_data")

    file_path = str(BASE_PATH / pl(configs["Clean_data"][file_to_save]))
    df.to_csv(file_path, index=False)


def loader_raw_data(
    configs: Dict,
    key_path: str,
    sep=";",
    encoding="latin1",
    sheet_name="",
) -> pd.DataFrame:
    """
    general raw data loader csv files
    """

    file_path = os.path.join(configs.load.BASE_PATH, configs.load["Full_data"][key_path])

    if file_path.endswith("csv"):
        df = pd.read_csv(
            file_path,
            sep=sep,
            encoding=encoding,
            dtype=configs.dtypes.dtypes[key_path],
        )

    elif file_path.endswith("xlsx"):
        if sheet_name != "":
            df = pd.read_excel(file_path, sheet_name=sheet_name)
        else:
            df = pd.read_excel(file_path)

    # DATA RENAMING
    if key_path in configs.renaming.datasets.keys():
        df = df[list(configs.renaming.datasets[key_path].keys())]
        df = df.rename(columns=configs.renaming.datasets[key_path])

    # format column names to be upper capital
    df.columns = smart_column_parser(df.columns)

    if "*" in file_path:
        logging.info("INTO COMPUTE")
        df = df.compute()

    logging.info(f"DATA: Loaded {key_path}")

    return df


def loader_clean_data(configs: Dict, key_path: str) -> pd.DataFrame:
    """
    load cleaned data, csv file
    :param configs:
    :param key_path:
    :return:
    """
    file_path = os.path.join(configs["resources"]["BASE_PATH"], configs["Clean_data"][key_path])
    return pd.read_csv(file_path, encoding="utf_8")


def function_weight():
    return lambda x: 1 - 1 / (1 + np.exp(-1.5 * (x / 365 - 3)))
