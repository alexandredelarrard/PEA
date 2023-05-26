#
# Copyright (c) 2023 by Boston Consulting Group. All rights reserved
#
# -*- coding: utf-8 -*-

import datetime
import logging
from typing import Dict

import pandas as pd


def _add_missing_ub_code_per_site(sub_df, target, histo_sales_end_date):

    meta_ub_code_site = sub_df[["COD_SITE", "META_UB", "UB_CODE"]].drop_duplicates()
    sub_df_filtered = sub_df[sub_df["DATE"] == histo_sales_end_date]

    missing_meta_ub_cod_site = pd.merge(
        meta_ub_code_site, sub_df_filtered, on=["COD_SITE", "META_UB", "UB_CODE"], how="outer", indicator=True
    )
    missing_meta_ub_cod_site = missing_meta_ub_cod_site[missing_meta_ub_cod_site["_merge"] == "left_only"].drop(
        columns=["_merge"]
    )
    missing_meta_ub_cod_site[target] = 0
    missing_meta_ub_cod_site["DATE"] = histo_sales_end_date

    sub_df = pd.concat([sub_df, missing_meta_ub_cod_site])

    return sub_df


def _deduction_sale(df_proportion_ub_in_meta_ub, config):
    # Gets proportion for split. Gets median of sales per ub over period, and then converts it to shares
    # This method does not consider 0, and should be used carefully. This is a preferred method so that
    # The sum of share sums to 1

    logging.debug("Not considering 0 for median")
    rolling_days = config.load.prediction_mode["ub_proportion_periode"]
    target = config.load.prediction_mode["target"]
    method = config.load.prediction_mode["ub_split_method"]

    if method == "MEAN_SALES":
        df_proportion_ub_in_meta_ub[f"MEAN_{target}_{rolling_days}d"] = (
            df_proportion_ub_in_meta_ub.sort_values(["DATE"], ascending=True)
            .groupby(["META_UB", "COD_SITE", "UB_CODE"])[f"{target}"]
            .transform(lambda x: x.rolling(rolling_days, 1).mean())
        )

    else:
        df_proportion_ub_in_meta_ub[f"MEAN_{target}_{rolling_days}d"] = (
            df_proportion_ub_in_meta_ub.sort_values(["DATE"], ascending=True)
            .groupby(["META_UB", "COD_SITE", "UB_CODE"])[f"{target}"]
            .transform(lambda x: x.rolling(rolling_days, 1).median())
        )

    meta_ub_sales_median_agg = (
        df_proportion_ub_in_meta_ub.groupby(["DATE", "META_UB", "COD_SITE"], as_index=False)[
            f"MEAN_{target}_{rolling_days}d"
        ]
        .sum()
        .rename(columns={f"MEAN_{target}_{rolling_days}d": f"MEAN_{target}_{rolling_days}d_META_UB"})
    )
    df_proportion_ub_in_meta_ub = df_proportion_ub_in_meta_ub.merge(
        meta_ub_sales_median_agg,
        on=["DATE", "META_UB", "COD_SITE"],
        how="left",
        validate="m:1",
    )

    df_proportion_ub_in_meta_ub["WEIGHT_UB_IN_META_UB"] = (
        df_proportion_ub_in_meta_ub[f"MEAN_{target}_{rolling_days}d"]
        / df_proportion_ub_in_meta_ub[f"MEAN_{target}_{rolling_days}d_META_UB"]
    )

    return df_proportion_ub_in_meta_ub


def _get_proportion_ub_per_date(
    config: Dict,
    df: pd.DataFrame,
):
    """
    Computing proportions of UB inside its Meta UB per COD_SITE. To do that:
        - We compute sales average or median (depending on ub_split_method) sales of each UB per COD_SITE over last
        rolling_days days (1)
        - We compute sum of these average or median sales per META_UB and COD_SITE (2)
        - Dividing (1) by (2) gives us the proportion of UB inside META_UB for each COD_SITE

    Args:
        config_prediction: config containing parameters to compute proportions such as ub_proportion_periode,
        df (pd.DataFrame): train df, history sales df
        histo_sales_end_date (datetime.date): last date that should be avaialable in historical data

    Returns:
        lagged (by prediction horizon) ubs' average proportion on rolling_days we want to moke a prediction on.

    Notes:
        If method = MEAN_SALES, we take the average sales over rolling period excluding zeros. Then, we compute the
        share over the average mean.
    """

    # target
    target = config.load.prediction_mode["target"]
    sub_df = df[["DATE", "META_UB", "UB_CODE", target, "COD_SITE"]]

    # Add (UB_CODE, COD_SITE) without any sales on histo sales end date to be able to have their proportion even if they had no sales on one day
    sub_df = _add_missing_ub_code_per_site(sub_df, target, config.load["histo_sales_end_date"])

    sub_df = sub_df.groupby(["DATE", "META_UB", "UB_CODE", "COD_SITE"], as_index=False)[target].sum()

    meta_ub_sales = (
        sub_df.groupby(["DATE", "META_UB", "COD_SITE"], as_index=False)[target]
        .sum()
        .rename(columns={target: target + "_META_UB"})
    )

    df_proportion_ub_in_meta_ub = sub_df.merge(
        meta_ub_sales,
        on=["DATE", "META_UB", "COD_SITE"],
        how="left",
        validate="m:1",
    )

    # deduce proportion of each UB in Meta UB
    df_proportion_ub_in_meta_ub = _deduction_sale(df_proportion_ub_in_meta_ub, config)

    df_proportion_ub_in_meta_ub = df_proportion_ub_in_meta_ub[
        [
            "DATE",
            "COD_SITE",
            "META_UB",
            "UB_CODE",
            "WEIGHT_UB_IN_META_UB",
        ]
    ]

    df_proportion_ub_in_meta_ub["WEIGHT_UB_IN_META_UB"] = df_proportion_ub_in_meta_ub["WEIGHT_UB_IN_META_UB"].fillna(0)

    return df_proportion_ub_in_meta_ub


def _compute_prediction_at_ub_level(
    df_proportion_ub_in_meta_ub: pd.DataFrame,
    prediction_output_ub: pd.DataFrame,
) -> pd.DataFrame:

    prediction_output_ub["COD_SITE"] = prediction_output_ub["COD_SITE"].astype(str)
    prediction_output_ub = prediction_output_ub.rename(columns={"PREDICTION": "PREDICTION_META_UB"})

    df_proportion_ub_in_meta_ub["COD_SITE"] = df_proportion_ub_in_meta_ub["COD_SITE"].astype(str)
    prediction_output_ub["COD_SITE"] = prediction_output_ub["COD_SITE"].astype(str)

    # complete with missing date for prospective dates
    max_date = df_proportion_ub_in_meta_ub["DATE"].max()
    max_proportion = df_proportion_ub_in_meta_ub.loc[df_proportion_ub_in_meta_ub["DATE"] == max_date]
    max_date_predict = prediction_output_ub["DATE"].max()

    if max_date_predict > max_date:
        for add_date in pd.date_range(max_date + datetime.timedelta(days=1), max_date_predict):
            max_proportion["DATE"] = add_date
            df_proportion_ub_in_meta_ub = pd.concat([df_proportion_ub_in_meta_ub, max_proportion], axis=0)

    prediction_output_ub_with_weight = prediction_output_ub.merge(
        df_proportion_ub_in_meta_ub,
        on=["DATE", "COD_SITE", "META_UB", "UB_CODE"],
        how="left",
        validate="m:1",
    )

    # for futur days of prediction with no weights to predict
    prediction_output_ub_with_weight = prediction_output_ub_with_weight.sort_values("DATE", ascending=True)
    cols_to_aggregate = ["COD_SITE", "META_UB", "UB_CODE"]
    prediction_output_ub_with_weight["WEIGHT_UB_IN_META_UB"] = (
        prediction_output_ub_with_weight[["DATE", "WEIGHT_UB_IN_META_UB"] + cols_to_aggregate]
        .groupby(cols_to_aggregate)["WEIGHT_UB_IN_META_UB"]
        .ffill()
    )

    prediction_output_ub_with_weight = prediction_output_ub_with_weight[
        prediction_output_ub_with_weight["DATE"].isin(prediction_output_ub.DATE.unique())
    ]
    prediction_output_ub_with_weight[f"PREDICTION_UB_USING_PROPORTION"] = (
        prediction_output_ub_with_weight["PREDICTION_META_UB"]
        * prediction_output_ub_with_weight["WEIGHT_UB_IN_META_UB"]
    )

    return prediction_output_ub_with_weight


def split_predictions_from_meta_ub_to_ub(
    prediction_output: pd.DataFrame, df: pd.DataFrame, top_ubs: pd.DataFrame, config: Dict
) -> pd.DataFrame:
    """
    Main function to split from meta ub prediction to ub prediction
        - Retrieve proportion of UB inside Meta UB per location (PDV or BASE)
        - Compute prediction at UB level, using prediction at Meta UB level multiplied by proportion of UB inside Meta UB

    Args:
        prediction_output (meta ub prediction output dataframe):
        df : sales history dataframe == train df:
        top_ubs (pd.DataFrame):
        config (configs): Config object

    Returns:
        Predictions at UB level

    """

    if not prediction_output.empty:

        # Adding UBs to the prediction output from top ubs
        prediction_output_ub = prediction_output[
            ["DATE", "META_UB", "PREDICTION", "COD_SITE", "PREDICTION_HORIZON"]
        ].merge(top_ubs, on=["META_UB", "COD_SITE"], how="left", validate="m:m")

        logging.info("POSTPROCESSING: Computing proportion of UB in Meta UB for each PdV")
        df_proportion_ub_in_meta_ub = _get_proportion_ub_per_date(
            config=config,
            df=df,
        )

        logging.info("POSTPROCESSING: Evaluating prediction at UB level for each PdV")
        prediction_output_ub = _compute_prediction_at_ub_level(
            df_proportion_ub_in_meta_ub=df_proportion_ub_in_meta_ub,
            prediction_output_ub=prediction_output_ub,
        )

    else:
        prediction_output_ub = pd.DataFrame(
            columns=[
                "DATE",
                "COD_SITE",
                "META_UB",
                "UB_CODE",
                "PREDICTION_UB_USING_PROPORTION",
                "PREDICTION_META_UB",
                "WEIGHT_UB_IN_META_UB",
                "PREDICTION_HORIZON",
            ]
        )
        logging.warning(
            "POSTPROCESSING: WARNING NO PREDICTIONS FOUND AT LEVEL META UBxCOD_SITE, PREDICTION DATA IS EMPTY"
        )

    return prediction_output_ub
