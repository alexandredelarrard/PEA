#
# Copyright (c) 2023 by Boston Consulting Group. All rights reserved
#
# -*- coding: utf-8 -*-

import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from forecast_fl.utils.config import Config


def filtering_dates_and_retrieving_holidays(df: pd.DataFrame, configs: Config) -> Tuple[pd.DataFrame, pd.DataFrame]:

    """
    Filters input data based on  histo_sales_start_date and histo_sales_end_date and retrieve special_holidays data
    Args:
        df: input data
        histo_sales_start_date: first date that should be available in df
        histo_sales_end_date: last date that should be available in df
        train: option to adapt function if a train or predict pipeline is running
        prediction_end_date: last day to predict, only used in case of a prediction pipeline

    Returns:
        sub_df: filtered data to chosen dates
        special_holidays: format holidays for prophet

    """

    # filter dataset on start and end dates
    history_mask = df["DATE"].between(configs.load["histo_sales_start_date"], configs.load["histo_sales_end_date"])
    sub_df = df[history_mask]

    logging.info(f"Filtered dates min / max in dataframe: {sub_df['DATE'].min()} - {sub_df['DATE'].max()}'")

    if not df.empty:
        special_holidays = sub_df[["DATE", "FERIER"]].drop_duplicates()
        logging.info(f"TRAIN DATA SHAPE on date range is : shape= {sub_df.shape}")
        return sub_df, special_holidays

    logging.critical(f"TRAIN DATA is empty on date range : shape= {sub_df.shape}")
    return pd.DataFrame(), pd.DataFrame()


def _weight_sku_in_meta_ub_cod_site(df: pd.DataFrame):
    """
    Get weight of UB CODE in META UB
    """

    weight = (
        df[["QTE_VTE", "META_UB", "UB_CODE", "COD_SITE"]]
        .groupby(["UB_CODE", "COD_SITE", "META_UB"])
        .sum()
        .reset_index()
    )

    weight_meta_ub = df[["QTE_VTE", "COD_SITE", "META_UB"]].groupby(["META_UB", "COD_SITE"]).sum()
    weight = weight.merge(
        weight_meta_ub,
        on=["META_UB", "COD_SITE"],
        how="left",
        suffixes=("_UB_CODE_COD_SITE", "_META_UB_COD_SITE"),
    )
    weight["WEIGHT"] = weight["QTE_VTE_UB_CODE_COD_SITE"] / weight["QTE_VTE_META_UB_COD_SITE"]

    return weight[["UB_CODE", "COD_SITE", "WEIGHT"]]


######################################" AGGREGATE TO RIGHT GRANULARITY


def aggregate_features_meta_ub_level(
    df: pd.DataFrame,
    parameters_from_config: Dict,
    train: bool = True,
):
    """
    Aggregate features from UB level to Meta UB level per [DATE, META UB, COD_SITE] with the following steps :

        - Some features will only need aggregation if we are in train mode especially : PRIX_UNITAIRE, TARGET and QTE_VTE
            - TARGET and QTE_VTE are aggregated using sum
            - PRIX_UNITAIRE is aggregated using a weighted average depending on past sales of UBs inside each Tuple META UB, COD_SITE

        - Other features related to date (temperature) will be aggregated using np.mean (as we will average same value)
        - Clipping some values on meteo features
        - Changing features type related to Date into category features
        - Target is transformed to log(1+target) to improve modelling

    Args:
        df: input data
        parameters_from_config: config parameters
        train:

    Returns:
        Aggregated version of input df at Meta UB level

    """

    target = parameters_from_config["target"]
    agg_features = ["DATE", "META_UB", "COD_SITE", "FERIER"]
    specific_train_aggregate = {}

    if train:
        # get weight per ub
        ub_weight_in_meta_ub = _weight_sku_in_meta_ub_cod_site(df)

        # keep only the UB with at least 0.1% volume of the meta_ub
        df = df.merge(ub_weight_in_meta_ub, on=["UB_CODE", "COD_SITE"], how="right", validate="m:1")

        specific_train_aggregate = {
            target: sum,
            "QTE_VTE": sum,
        }

    df = df.sort_values(agg_features)
    aggregation_dict = {
        "TEMPERATURE": np.mean,
        "NEBULOSITE": np.mean,
        "PRECIPITATIONS": np.mean,
        "PRESSION": np.mean,
        "HOLIDAYS": np.mean,
        "IS_SEASONAL": np.max,
        "DIST_TO_DAY_OFF": np.mean,
    }

    aggregation_dict.update(specific_train_aggregate)
    agg_df = (
        df[agg_features + list(aggregation_dict.keys())].groupby(agg_features).aggregate(aggregation_dict).reset_index()
    )

    if train:
        for col in [
            "PRIX_UNITAIRE_RETRAITE",
            "VAR_PRIX_UNITAIRE_RETRAITE_LW",
            "VAR_PRIX_UNITAIRE_RETRAITE_LM",
            "VAR_PRIX_UNITAIRE_RETRAITE_LY",
        ]:
            agg_df[col] = (
                df[agg_features + [col, "WEIGHT"]]
                .groupby(agg_features)
                .apply(lambda x: np.average(x[col], weights=x["WEIGHT"]))
                .values
            )

    agg_df = agg_df.sort_values(agg_features)
    agg_df = agg_df.reset_index(drop=True)

    return agg_df


######################################" FEATURE ENGINEERING FOR LGBM"


def clip_strange_values(df):
    # clip strange values
    df["PRECIPITATIONS"] = df["PRECIPITATIONS"].clip(None, 20)
    df["TEMPERATURE"] = df["TEMPERATURE"].clip(-10, 30)
    df["PRESSION"] = df["PRESSION"] / 100000
    return df


def create_date_variables(df):
    # WEEK_DAY
    df["WEEK_DAY"] = df["DATE"].dt.dayofweek.astype("category")
    df["MONTH"] = df["DATE"].dt.month.astype("category")
    df["MONTH_DAY"] = df["DATE"].dt.day.astype("category")
    return df


def transform_target(df, target, log_transfo):
    if log_transfo and target in df.columns:
        df[target] = np.log(1 + df[target])
    return df


def add_shift_features(df, config, prediction_horizon, target):

    for j in range(prediction_horizon + config.load.config_lgbm["rolling_sum_days"]):
        df[f"Y-HORIZON-{j}"] = df.groupby(["META_UB", "COD_SITE"])[target].shift(j + prediction_horizon)

    df["TEMP_MEAN_3_D"] = df.groupby(["META_UB", "COD_SITE"], as_index=False)["TEMPERATURE"].shift(prediction_horizon)[
        "TEMPERATURE"
    ]
    df["TEMP_MEAN_3_D"] = (
        df.groupby(["META_UB", "COD_SITE"], as_index=False)["TEMP_MEAN_3_D"]
        .rolling(3)
        .mean(skipna=True)["TEMP_MEAN_3_D"]
    )

    # change name of the feature (more than 7D)
    df["Y_MEAN_7D"] = df[[x for x in df.columns if "Y-" in x]].mean(axis=1, skipna=True)
    df["TREND-HORIZON-7"] = ((df["Y-HORIZON-0"] - df["Y-HORIZON-7"]) / df["Y-HORIZON-7"]).clip(-1, 1)
    df["Y_STD_7D"] = df[[x for x in df.columns if "Y-" in x]].std(axis=1, skipna=True)

    return df


def add_price_features(df, prediction_horizon):

    # shift prix unitaire per Meta UB
    # Price effect is diluted since all UB are aggregated between each other
    df["PRIX_UNITAIRE_RETRAITE"] = df["PRIX_UNITAIRE_RETRAITE"].round(2)

    for col in [
        "PRIX_UNITAIRE_RETRAITE",
        "VAR_PRIX_UNITAIRE_RETRAITE_LW",
        "VAR_PRIX_UNITAIRE_RETRAITE_LM",
        "VAR_PRIX_UNITAIRE_RETRAITE_LY",
    ]:
        df[f"{col}_J-HORIZON"] = df.groupby(["META_UB", "COD_SITE"])[col].shift(prediction_horizon)

    return df


def add_ratio_to_date_features(df, config, target, prediction_horizon):

    shift_by_type = {"WEEK": 7, "MONTH": 30, "YEAR": 365}
    df["MEAN_TARGET"] = (
        df.groupby(["META_UB", "COD_SITE"], as_index=False)[target]
        .rolling(window=config.load.config_lgbm["rolling_sum_days"])
        .quantile(quantile=0.75)[target]
    )
    for type_shift, shift in shift_by_type.items():
        df[f"MEAN_TARGET_DELAYED_{type_shift}"] = df.groupby(["META_UB", "COD_SITE"], as_index=False)[
            "MEAN_TARGET"
        ].shift(shift)

        df[f"RATIO_TARGET_DELAYED_{type_shift}"] = np.where(
            (df["MEAN_TARGET"] == 0) | (df[f"MEAN_TARGET_DELAYED_{type_shift}"] == 0),
            np.nan,
            df["MEAN_TARGET"] / df[f"MEAN_TARGET_DELAYED_{type_shift}"],
        )

        df[f"RATIO_TARGET_DELAYED_{type_shift}-HORIZON"] = df.groupby(["META_UB", "COD_SITE"])[
            f"RATIO_TARGET_DELAYED_{type_shift}"
        ].shift(prediction_horizon)

    return df


def feature_engineering_meta_ub_cod_site(df: pd.DataFrame, config: Config, prediction_horizon: int):
    """
    Feature engineering of cleaned dataframe to prepare dataframe for train / predict related to Target feature
        - Creates all lagged features : Y-HORIZON-X
        - Trend features
        - Std features
        - lagged price
        - Target ratios comparing last rolling_sum_days target to last week, month and year
        - Applying specific target adjustment for train mode

    Args:
        df (pd.DataFrame): cleaned dataframe
        parameters_from_config (dict): configuration file (training or prediction configs are passed depending on train/predict pipeline)
        rolling_sum_days (int): number of days on which to calculate the rolling mean to get trend/std features of sales
        prediction_horizon (int): Horizon of modelling. If None, taken from config

    Returns:
        df (pd.DataFrame): feature engineered dataframe

    """

    logging.info("FEATURE ENGINEERING: Adding features needed for modelling")

    parameters_from_config = config.load["parameters_training_model"]
    target = parameters_from_config["target"]
    log_transfo = parameters_from_config["log_transfo"]

    if prediction_horizon is None:
        prediction_horizon = parameters_from_config["prediction_horizon"]
    assert prediction_horizon >= 1, "Modify prediction_horizon value in training config to be at least 1"

    ############################## CLIP STRANGE VALUES
    df = clip_strange_values(df)

    df = create_date_variables(df)

    ############################## TRANSFORM TARGET
    df = transform_target(df, target, log_transfo)

    ##############################  ADD price features
    df = add_price_features(df, prediction_horizon)

    ##############################  ADD shift features
    df = add_shift_features(df, config, prediction_horizon, target)

    # Ratio TARGET Prior Year, Month and Week
    df = add_ratio_to_date_features(df, config, target, prediction_horizon)

    return df
