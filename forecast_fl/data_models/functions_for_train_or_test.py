#
# Copyright (c) 2023 by Boston Consulting Group. All rights reserved
#
# -*- coding: utf-8 -*-

import logging
from datetime import timedelta
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from forecast_fl.data_models.lgbm_prediction import lgbm_prediction
from forecast_fl.data_models.prophet_prediction import prophet_predict
from forecast_fl.data_postprocessing.split_predictions_from_meta_ub_to_ub import (
    split_predictions_from_meta_ub_to_ub,
)
from forecast_fl.data_preparation.feature_engineering import (
    filtering_dates_and_retrieving_holidays,
)
from forecast_fl.utils.config import Config


def create_tuples_to_model(
    df: pd.DataFrame,
    config,
) -> List[Tuple[str, str]]:
    """
    Retrieves list of PdVs and Meta UBs configurations defined to run.
    Each configuration will lead to training a model an eventuaelly use it to predict
    Ensure we have at least > min_non_zero_observations observations to train on

    Args:
        df (pd.DataFrame): cleaned dataframe used to train / predict
        config (Dict): configuration file (all load configurations available)

    Returns:
        tuple_meta_ub_granularity: List of models to train or predict (e.g, (banane bio, 1599))

    """

    min_non_zero_observations = config.load.parameters_training_model["min_non_zero_observations"]

    # create tuples (meta_ub, group)
    tuple_meta_ub_granularity = df[["META_UB", "COD_SITE"]].drop_duplicates().reset_index(drop=True)

    if config.load["objective"] == "training":
        count_tuples_obs = (
            df.loc[df["TARGET"] > 0][["META_UB", "COD_SITE", "DATE"]]
            .groupby(["META_UB", "COD_SITE"])
            .size()
            .reset_index()
        )
        count_tuples_obs = count_tuples_obs.loc[count_tuples_obs[0] > min_non_zero_observations]
        tuple_meta_ub_granularity = list(zip(count_tuples_obs.META_UB, count_tuples_obs.COD_SITE))
    else:
        tuple_meta_ub_granularity = df[["META_UB", "COD_SITE"]].drop_duplicates().reset_index(drop=True)
        tuple_meta_ub_granularity = list(zip(tuple_meta_ub_granularity.META_UB, tuple_meta_ub_granularity.COD_SITE))

    return tuple_meta_ub_granularity


def split_train_test_predict(sub_df, target, split_percentage=0.15):

    nbr_days = (sub_df["DATE"].max() - sub_df["DATE"].min()).days
    split_date = sub_df["DATE"].max() - timedelta(days=int(nbr_days * split_percentage))

    full_predict = (
        sub_df.rename(
            columns={
                "DATE": "ds",
                target: "y",
            }
        )
        .reset_index(drop=True)
        .copy()
    )

    return full_predict, split_date


def handle_outliers_and_zeros(full_predict, prophet_config):
    """Handle outliers value close to 0
    2 cases happened :
    - 0 since this is a seasonal product, which means the 0 is likely to be good
    - 0 because of OOS -> this is an outleir

    The logic is to say that for a given month if 50% of the data is very small, then we assume the
    data is good
    If median is higher than the outlier percentile, then we assume any 0 is an outlier -> set to nan

    Args:
        full_predict (_type_): df_input dataframe at META UB level
        prophet_config (_type_): parameters of prophet

    Returns:
        _type_: _description_

    """

    # ensure that for seasonal products, 0 remains 0, otherwise it is a nan
    # full_predict["IS_ZERO"] = full_predict["META_UB"].map(mapping_is_zero_nan_meta_ub)
    full_predict["y"] = np.where(
        (full_predict["IS_SEASONAL"] == 0) & (full_predict["y"] == 0), np.nan, full_predict["y"]
    )

    # ensure 0 or nan for periodic products
    outliers_percentile_value = np.percentile(full_predict["y"], prophet_config["outliers"])

    full_predict["month"] = full_predict["ds"].dt.month
    agg = full_predict[["month", "y"]].groupby("month").median().reset_index()
    month_at_zero = agg.loc[agg["y"] <= outliers_percentile_value].month.tolist()
    month_remaining = agg.loc[agg["y"] > outliers_percentile_value].month.tolist()

    full_predict["y"] = np.where(
        full_predict["month"].isin(month_at_zero) & (full_predict["y"] == 0),
        -1,
        np.where(full_predict["month"].isin(month_remaining) & (full_predict["y"] == 0), np.nan, full_predict["y"]),
    )
    del full_predict["month"]

    return full_predict


def backtest_meta_ub_to_ub_level(
    results_lgbm: pd.DataFrame,
    df_input: pd.DataFrame,
    config: Config,
    meta_ub_n: str,
    base_path: Path,
    granularity: str,
):

    target = config.load.parameters_training_model["target"]

    results_lgbm["PREDICTION_HORIZON"] = config.load["prediction_horizon"]
    results_lgbm["PREDICTION"] = results_lgbm["PREDICTION_LGB"]

    train_df = df_input.loc[df_input["META_UB"] == meta_ub_n]
    top_ubs = train_df[["COD_SITE", "META_UB", "UB_CODE"]].drop_duplicates()

    train_df, _ = filtering_dates_and_retrieving_holidays(df=train_df, configs=config)

    # Split META UB predictions to UB CODE
    prediction_output_ub_pdv = split_predictions_from_meta_ub_to_ub(
        prediction_output=results_lgbm, df=train_df, top_ubs=top_ubs, config=config
    )

    # merge with ground truth
    past_ub_target = train_df[["DATE", "COD_SITE", "META_UB", "UB_CODE", target]]

    prediction_output_ub_pdv = prediction_output_ub_pdv.merge(
        past_ub_target,
        on=["DATE", "COD_SITE", "META_UB", "UB_CODE"],
        how="left",
        validate="1:1",
    )

    # save time series per UB with sell during test phase
    for ub_code in top_ubs["UB_CODE"].unique():
        sub_ub_df = prediction_output_ub_pdv.loc[prediction_output_ub_pdv["UB_CODE"] == ub_code].set_index("DATE")
        if sub_ub_df[target].sum() > 0:

            # calculate error
            sub_ub_df["ERROR_PREDICTION_UB"] = (
                abs(sub_ub_df[target] - sub_ub_df["PREDICTION_UB_USING_PROPORTION"]) / sub_ub_df[target]
            )
            avg_error_ub = sub_ub_df.loc[sub_ub_df[target] > 0, f"ERROR_PREDICTION_UB"].median() * 100

            file_name = f"Time_Serie_ub_{ub_code}_codesite_{granularity}.png"
            path = base_path / str(meta_ub_n) / "plots" / "ub_code_backtest"
            path.mkdir(parents=True, exist_ok=True)
            full_path = path / file_name

            plt.figure()
            title = f"UB / PDV = {ub_code} / {granularity}, MEDIAN error % = {avg_error_ub:0.3f}"
            sub_ub_df[["PREDICTION_UB_USING_PROPORTION", target]].plot(title=title)
            plt.savefig(str(full_path.absolute()))
            plt.close()


def model_single_ub_granularity(
    sub_df: pd.DataFrame,
    special_holidays: pd.DataFrame,
    ub_n: str,
    granularity: str,
    config: Config,
    path: Path,
):
    """Train and save model for a given PdV/Base/Cluster and Meta UB when enough observations are available

    Args:
        sub_df (pd.DataFrame): cleaned dataframe filtered on Meta UB x PdV/Base/Cluster (without feature engineering)
        special_holidays (pd.DataFrame): whether a given date is Ferier or not (Date x Ferier)
        ub_n (str): Name of Meta UB to model
        granularity (str): Name of PdV/Cluster/Base to model
        configs (dict): configuration file (all configurations available)
        paths_by_directory (dict): saving paths

    Returns:
        results_lgbm (pd.DataFrame): trained results of LGB model
        lgbm_model (Booster): trained lgbm model
        model_ts: trained time series model (Prophet or Holt Winter)

    """

    # hyperparameters
    parameters_from_config = config.load.parameters_training_model
    target = parameters_from_config["target"]

    results_lgbm = pd.DataFrame()

    granularity_desc = ""
    if config.load["prediction_granularity"] == "PDV":
        granularity_desc = "site"
    elif config.load["prediction_granularity"] == "CLUSTER":
        granularity_desc = "cluster"

    logging.info(f"PROCESSING META UB {ub_n} in {granularity_desc} {granularity}")

    full_predict, split_date = split_train_test_predict(
        sub_df, target, split_percentage=parameters_from_config.train_test_proportion
    )

    # handle outliers -> remove for fit /
    # but take care of products with seasonalities at 0 for some time
    full_predict = handle_outliers_and_zeros(full_predict, config.load.config_prophet)

    # Trains only if enough data point
    n_observations = full_predict.loc[full_predict["y"] > 0].shape[0]
    if n_observations > parameters_from_config.min_non_zero_observations:

        # TS Prediction
        df_with_ts_prediction, train_ds_prophet, test_ds_prophet, ts_model = prophet_predict(
            full_predict=full_predict,
            split_date=split_date,
            special_holidays=special_holidays,
            ub_n=ub_n,
            granularity=granularity,
            configs=config,
            base_path=path,
        )

        # LGBM Layer
        results_lgbm, lgbm_model = lgbm_prediction(
            full_ts=df_with_ts_prediction,
            train_ts=train_ds_prophet,
            test_ts=test_ds_prophet,
            ub_n=ub_n,
            granularity=granularity,
            lgbm_config=config.load["config_lgbm"],
            parameters_from_config=parameters_from_config,
            base_path=path,
        )

        return results_lgbm, lgbm_model, ts_model

    else:
        logging.error(
            f"Found only {n_observations} non zeros observations for {ub_n}, {granularity_desc} {granularity}, skips training as this does not meet the minimum number of observations ({parameters_from_config.min_non_zero_observations})"
        )
        return results_lgbm, None, None
