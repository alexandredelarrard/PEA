#
# Copyright (c) 2023 by Boston Consulting Group. All rights reserved
#
# -*- coding: utf-8 -*-

import pandas as pd
from forecast_fl.data_preparation.feature_engineering_ub_model import (
    encode_dates,
    get_timeseries_features_rolling_7_D,
    lgbm_cols_to_category,
)
from forecast_fl.utils.config import Config


def create_prediction_input_ub(
    prediction_input: pd.DataFrame,
    train_df: pd.DataFrame,
    config: Config,
) -> pd.DataFrame:
    """
    Create specific features to UB model into prediction input. Included :
        - Features related to stores characteristics
        - Features related to Target : lagged target, average values, stds
            - For features related to Target :
                - Lagged features are created for day = histo_sales_end_date +1 day with
                a prediction_horizon set to 1 and are used for all dates to predict as those
                features are related to prediction_horizon and not explicitly the date itself.
                - For example : feature target_prediction_horizon
                - histo_sales_end_date = 2022-09-01
                - prediction_date =[2022-09-03 -> (prediction_horizon will be 2 days), 2022-09-04
                    -> (prediction_horizon will be 3 days)]
                - For dates :
                    - 2022-09-03 : target-prediction_horizon will be the target for date
                        (2022-09-03 - 2 (prediction_horizon) days) which is 2022-09-01
                    - 2022-09-04 : target-prediction_horizon will be the target for date
                        (2022-09-04 - 3 (prediction_horizon) days) which is 2022-09-01

            - Encode dates
            - Transform columns to category columns if needed

    Args:
        prediction_input (pd.DataFrame): input prediction to be enriched
        train_df: historical data filtered before histo_sales_end_date
        histo_sales_end_date (datetime.date): last date to have in historical data to create shifted features
        configs (config class): contains parameters in the config

    Returns:
        prediction input enriched with features related to target (lagged, average, std target...) and stores
        characteristics

    """

    histo_sales_end_date = config.load["histo_sales_end_date"]

    # Retrieve features from prediction input at meta ub level (month, day, week, temperature,...)
    prediction_input_ub = prediction_input[
        ["DATE", "COD_SITE", "META_UB", "UB_CODE", "PRECIPITATIONS", "TEMPERATURE", "NEBULOSITE", "PRESSION"]
    ]

    # get PDV characteristics
    pdv_characteristics = train_df[["COD_SITE", "SURFACE_TOTALE", "NB_CAISSES"]].drop_duplicates()
    prediction_input_ub = prediction_input_ub.merge(pdv_characteristics, on=["COD_SITE"], how="left", validate="m:1")

    # Create lagged features
    lagged_features_df = train_df[["COD_SITE", "UB_CODE"]].drop_duplicates()
    next_day = histo_sales_end_date + pd.Timedelta(1, "d")
    lagged_features_df["DATE"] = next_day
    lagged_features_df = pd.concat([train_df, lagged_features_df]).reset_index(drop=True)

    lagged_features_df = get_timeseries_features_rolling_7_D(
        df_input=lagged_features_df, config=config, prediction_horizon=1
    )

    # get all remaining columns to extract in static features
    static_features = list(config.load.parameters_training_ub_model.config_ub_lgbm.STATIC_FEATURES)
    dynamic_features = list(
        lagged_features_df.filter(
            regex=rf"{config.load.parameters_training_ub_model.config_ub_lgbm['TARGET']}_SHIFTED"
        ).columns
    )
    remaining_features = set(static_features) - set(prediction_input_ub.columns)
    remaining_features = list(remaining_features.intersection(set(lagged_features_df.columns)))

    # filter lagged features df on it
    lagged_features_df = lagged_features_df[lagged_features_df["DATE"] == next_day][
        ["UB_CODE", "COD_SITE"] + dynamic_features + remaining_features
    ]
    prediction_input_ub = prediction_input_ub.merge(
        lagged_features_df, on=["UB_CODE", "COD_SITE"], how="left", validate="m:1"
    )

    # Get features related to date
    prediction_input_ub = encode_dates(prediction_input_ub, "DATE")
    prediction_input_ub = lgbm_cols_to_category(prediction_input_ub)

    return prediction_input_ub
