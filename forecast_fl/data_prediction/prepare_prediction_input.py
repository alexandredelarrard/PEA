#
# Copyright (c) 2023 by Boston Consulting Group. All rights reserved
#
# -*- coding: utf-8 -*-

from typing import Dict

import pandas as pd
from forecast_fl.data_preparation.feature_engineering import (
    aggregate_features_meta_ub_level,
    feature_engineering_meta_ub_cod_site,
)
from forecast_fl.data_preparation.prepare_histo import (
    _enrich_histo_data,
    _jours_feriers,
    _vacances_scolaires,
    clean_meteo_data,
)
from forecast_fl.utils.config import Config


def create_prediction_input(df_input: pd.DataFrame, config: Config) -> pd.DataFrame:
    """
    Initialises input dataframe for prediction pipeline.
    Includes dates to predict for given meta unité de besoin and point de vente

    Args:
        df_input (pd.DataFrame): processed historical data used for training
        prediction_date_min (datetime.date): first date to predict
        prediction_date_max (datetime.date): last date to predict

    Returns:
        prediction input containing all tuples to be predicted [DATE, UB_CODE, COD_SITE] and relative infos

    """

    prediction_date_min = config.load["prediction_date_min"]
    prediction_date_max = config.load["prediction_date_max"]

    # create all combinaison to perform predictions
    dates_to_predict = pd.date_range(
        prediction_date_min,
        prediction_date_max,
        freq="D",
    )

    prediction_input = (
        df_input[["COD_SITE", "META_UB", "UB_NOM", "UB_CODE", "TYPE_UB", "POIDS_UC_MAPPING", "IS_SEASONAL"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    full_to_predict = pd.DataFrame()
    for date in dates_to_predict:
        prediction_input["DATE"] = date
        full_to_predict = pd.concat([full_to_predict, prediction_input], axis=0)

    return full_to_predict


def prepare_prediction_data(
    prediction_input_df: pd.DataFrame,
    datas: Dict,
) -> pd.DataFrame:
    """
    Enrich prediction data with meteo, vacances scolaires, jours feries

    Args:
        prediction_input_df: input prediction to be enriched
        datas: datas where meteorological data is included

    Returns:
        prediction input enriched with features related to dates

    """
    meteo = datas["METEO"].copy()
    meteo = clean_meteo_data(meteo)

    # add meteo data
    prediction_input_df = _enrich_histo_data(prediction_input_df, meteo)

    # add holidays & off days
    prediction_input_df = _vacances_scolaires(prediction_input_df)

    # add off days
    prediction_input_df = _jours_feriers(prediction_input_df)

    return prediction_input_df


def prepare_rolling_info_prediction_data(prediction_input: pd.DataFrame, train_df: pd.DataFrame, config: Config):
    """
    Create specific features to META UB model into prediction input. Included :
        - Features aggregation from UB to META UB level
        - Features related to Target : lagged target, average values, stds per META_UB x COD_SITE

            - For features related to Target :
                - Lagged features are created for day = histo_sales_end_date +1 day with a prediction_horizon set to 1 and are used for all dates to predict as those features are related to prediction_horizon and not explicitly the date itself.

                For example : feature target_prediction_horizon with histo_sales_end_date = 2022-09-01 and prediction_date = [2022-09-03 -> (prediction_horizon will be 2 days), 2022-09-04 -> (prediction_horizon will be 3 days)]
                - for dates :

                    - 2022-09-03 : target-prediction_horizon will be the target for date (2022-09-03 - 2 (prediction_horizon) days) which is 2022-09-01
                    - 2022-09-04 : target-prediction_horizon will be the target for date (2022-09-04 - 3 (prediction_horizon) days) which is 2022-09-01

    Args:
        prediction_input (pd.DataFrame): input prediction to be enriched
        train_df: historical data filtered before histo_sales_end_date
        configs (config class): contains parameters in the config
        histo_sales_end_date (datetime.date): last date to have in historical data to create shifted features

    Returns:
        prediction input enriched with features aggregated at Meta UB level related to target (lagged, average, std target...)

    """

    # reduce to 1 year of data for feature engineering purpose since 1 year lagged are used
    filtered_train_df = train_df[train_df["DATE"] >= config.load["histo_sales_end_date"] - pd.Timedelta(385, "d")]

    train_df_aggregated = aggregate_features_meta_ub_level(
        filtered_train_df.sort_values(["DATE"]), config.load["prediction_mode"], train=True
    )

    prediction_input_aggregated = aggregate_features_meta_ub_level(
        prediction_input.sort_values(["DATE"]), config.load["prediction_mode"], train=False
    )

    # Create lagged features
    lagged_features_df = train_df_aggregated[["COD_SITE", "META_UB"]].drop_duplicates()
    next_day = config.load["histo_sales_end_date"] + pd.Timedelta(1, "d")
    lagged_features_df["DATE"] = next_day
    lagged_features_df = pd.concat([train_df_aggregated, lagged_features_df]).reset_index(drop=True)

    lagged_features_df = feature_engineering_meta_ub_cod_site(
        df=lagged_features_df,
        config=config,
        prediction_horizon=1,
    )

    lagged_features = list(
        set(config.load.config_lgbm.FEATURES) - set(prediction_input_aggregated.columns) - set(["PREDICTION"])
    )
    lagged_features_df = lagged_features_df[lagged_features_df["DATE"] == next_day][
        ["META_UB", "COD_SITE"] + lagged_features
    ]
    prediction_input_aggregated = prediction_input_aggregated.merge(
        lagged_features_df, on=["META_UB", "COD_SITE"], how="left", validate="m:1"
    )

    return prediction_input_aggregated
