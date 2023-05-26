#
# Copyright (c) 2023 by Boston Consulting Group. All rights reserved
#
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from forecast_fl.data_preparation.feature_engineering import (
    filtering_dates_and_retrieving_holidays,
)
from forecast_fl.utils.config import Config
from forecast_fl.utils.general_functions import function_weight


def create_weights(X_train_dates_ub):

    X_train_dates_ub["DIFF"] = (X_train_dates_ub["DATE"] - X_train_dates_ub["DATE"].max()).dt.days
    f = function_weight()
    X_train_dates_ub["WEIGHT"] = f(X_train_dates_ub["DIFF"])

    return X_train_dates_ub


def encode_dates(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """
    Encode dates in weight model, keeps month, day, week
    """

    df[date_col] = pd.to_datetime(df[date_col], format="%Y-%m-%d")
    df["MONTH"] = df[date_col].dt.month.astype("category")
    df["WEEK_DAY"] = df[date_col].dt.dayofweek.astype("category")
    df["MONTH_DAY"] = df[date_col].dt.day.astype("category")

    return df


def lgbm_cols_to_category(X: pd.DataFrame) -> pd.DataFrame:
    """
    Convert object typed columns to category for LGBM automatic handling of categorical features
    """

    for c in X.columns:
        col_type = X[c].dtype
        if col_type == "object" or col_type.name == "category":
            X[c] = X[c].astype("category")
        # convert Day, Month to categories
        elif c == "MONTH_DAY" or c == "MONTH" or c == "WEEK":
            X[c] = X[c].astype("category")
    return X


def compute_rolling_target(df_input, config):

    rolling_days = config.load.parameters_training_ub_model["smooth_target_by_x_rolling_days"]
    target_lgbm = config.load.parameters_training_ub_model.config_ub_lgbm["TARGET"]

    # Compute rolling target (lissage signal)
    df_input[target_lgbm] = (
        df_input.sort_values(["DATE"], ascending=True)
        .groupby(["UB_CODE", "COD_SITE"])["TARGET"]
        .transform(lambda x: x.rolling(rolling_days, 1).mean())
    )
    return df_input


def compute_lagged_features(df_input, target_lgbm, prediction_horizon, lag_nb_days):

    # Compute lagged rolling target
    for j in list(range(prediction_horizon, prediction_horizon + lag_nb_days + 1)):
        df_input[f"{target_lgbm}_SHIFTED_HORIZON_{j-prediction_horizon}_D"] = (
            df_input.sort_values(["DATE"], ascending=True).groupby(["UB_CODE", "COD_SITE"])[f"{target_lgbm}"].shift(j)
        )

    for days in list(set([7, lag_nb_days])):
        cols_to_mean = [f"{target_lgbm}_SHIFTED_HORIZON_{j}_D" for j in range(days)]
        df_input[f"{target_lgbm}_SHIFTED_MEAN_{days}D"] = df_input[cols_to_mean].mean(axis=1, skipna=True)

    return df_input


def compute_ratio_date_features(df_input, prediction_horizon, lag_nb_days):

    shift_by_type = {"WEEK": 7, "MONTH": 30, "YEAR": 365}
    df_input["MEAN_TARGET"] = (
        df_input.groupby(["UB_CODE", "COD_SITE"], as_index=False)["TARGET"]
        .rolling(window=lag_nb_days)
        .quantile(quantile=0.75)["TARGET"]
    )

    for type_shift, shift in shift_by_type.items():
        df_input[f"MEAN_TARGET_DELAYED_{type_shift}"] = df_input.groupby(["UB_CODE", "COD_SITE"], as_index=False)[
            "MEAN_TARGET"
        ].shift(shift)

        df_input[f"RATIO_TARGET_DELAYED_{type_shift}"] = np.where(
            (df_input["MEAN_TARGET"] == 0) | (df_input[f"MEAN_TARGET_DELAYED_{type_shift}"] == 0),
            np.nan,
            df_input["MEAN_TARGET"] / df_input[f"MEAN_TARGET_DELAYED_{type_shift}"],
        )

        df_input[f"RATIO_TARGET_DELAYED_{type_shift}-HORIZON"] = df_input.groupby(["UB_CODE", "COD_SITE"])[
            f"RATIO_TARGET_DELAYED_{type_shift}"
        ].shift(prediction_horizon)

    return df_input


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

    return df.drop(
        ["VAR_PRIX_UNITAIRE_RETRAITE_LW", "VAR_PRIX_UNITAIRE_RETRAITE_LM", "VAR_PRIX_UNITAIRE_RETRAITE_LY"], axis=1
    )


def get_timeseries_features_rolling_7_D(
    df_input: pd.DataFrame, config: Config, prediction_horizon: int, lag_nb_days: int = 7
) -> pd.DataFrame:

    # LGBM model at UB predicts rolling sales
    target_lgbm = config.load.parameters_training_ub_model.config_ub_lgbm["TARGET"]

    # ROLLING TARGET
    df_input = compute_rolling_target(df_input, config)

    # Ratio TARGET Prior Year, Month and Week
    df_input = compute_ratio_date_features(df_input, prediction_horizon, lag_nb_days=14)

    # RATIO PRICES
    df_input = add_price_features(df_input, prediction_horizon)

    # lagged features
    df_input = compute_lagged_features(df_input, target_lgbm, prediction_horizon, lag_nb_days=lag_nb_days)

    return df_input.drop(
        [
            "MNT_VTE",
            "QTE_VTE",
            "VOCATION",
            "CODE_BASE",
            "NOM_PDV",
            "UB_NOM",
            "POIDS_UC_MAPPING",
            "PRIX_UNITAIRE_RETRAITE",
            "RATIO_TARGET_DELAYED_YEAR",
            "RATIO_TARGET_DELAYED_MONTH",
            "RATIO_TARGET_DELAYED_WEEK",
            "PRESSION",
            "HOLIDAYS",
            "IS_SEASONAL",
            "TYPE_UB",
            "MEAN_TARGET",
        ],
        axis=1,
    )


def ub_feature_engineering(df_ts_features: pd.DataFrame, ub_n: str, config: Config) -> pd.DataFrame:

    sub_df = df_ts_features[df_ts_features["UB_CODE"] == ub_n]

    # Train LGB on period before backtest March-August 2022
    input_df_ub, _ = filtering_dates_and_retrieving_holidays(df=sub_df, configs=config)

    # Create weight on date & data prep
    input_df_ub = create_weights(input_df_ub)
    input_df_ub = encode_dates(input_df_ub, "DATE")
    input_df_ub = lgbm_cols_to_category(input_df_ub)

    return input_df_ub
