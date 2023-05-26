#
# Copyright (c) 2023 by Boston Consulting Group. All rights reserved
#
# -*- coding: utf-8 -*-

import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from forecast_fl.utils.config import Config


def split_predictions_from_cluster_to_pdv(
    prediction_output: pd.DataFrame,
    df_input_pdv_level: pd.DataFrame,
    clustering: pd.DataFrame,
    config: Config,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split prediction from cluster to PdV level by getting weights of PdV in cluster (via NAIVE or MODEL methodology)
    and computing prediction
    """

    if config.load["prediction_granularity"] == "CLUSTER":

        prediction_output["CLUSTER"] = prediction_output["COD_SITE"]
        prediction_output_pdv = prediction_output[
            ["DATE", "META_UB", "PREDICTION", "CLUSTER", "PREDICTION_LGBM", "PREDICTION_TS", "PREDICTION_HORIZON"]
        ].merge(clustering, on=["CLUSTER", "META_UB"], how="left", validate="m:m")

        logging.info(f"POSTPROCESSING: Computing proportion via Naive method of PdV in cluster")
        # Get proportion of PdV in cluster based on historical sales
        pdv_sales_proportion = _get_proportion_pdv_per_date_naive(
            config=config,
            df_input_pdv_level=df_input_pdv_level,
            clustering_mapping=clustering,
        )

        # Compute prediction at PdV level using historical proportion and cluster level predictions
        logging.info("POSTPROCESSING: Evaluating prediction at PdV level")
        prediction_output_pdv = _compute_prediction_at_pdv_level(
            pdv_sales_proportion=pdv_sales_proportion,
            prediction_output_pdv=prediction_output_pdv,
        )
        prediction_output_cluster = prediction_output
    else:
        prediction_output_pdv = prediction_output
        prediction_output_cluster = pd.DataFrame()

    return prediction_output_pdv, prediction_output_cluster


def _compute_prediction_at_pdv_level(
    pdv_sales_proportion: pd.DataFrame, prediction_output_pdv: pd.DataFrame
) -> pd.DataFrame:
    """Computes prediction at PdV level using cluster level prediction and weight of PdV in cluster
    (obtained either via NAIVE method of weight model)
    """

    # Rename
    prediction_output_pdv = prediction_output_pdv.rename(columns={"PREDICTION": "PREDICTION_CLUSTER"})

    pdv_sales_proportion["COD_SITE"] = pdv_sales_proportion["COD_SITE"].astype(str)
    prediction_output_pdv["COD_SITE"] = prediction_output_pdv["COD_SITE"].astype(str)
    prediction_output_pdv = prediction_output_pdv.merge(
        pdv_sales_proportion,
        on=["META_UB", "COD_SITE"],
        how="left",
        validate="m:1",
    )

    prediction_output_pdv["PREDICTION"] = (
        prediction_output_pdv["PREDICTION_CLUSTER"] * prediction_output_pdv["WEIGHT_PDV_IN_CLUSTER"]
    )
    return prediction_output_pdv


def _get_proportion_pdv_per_date_naive(
    config: Dict,
    df_input_pdv_level: pd.DataFrame,
    clustering_mapping: pd.DataFrame,
):
    """Get proportion of sales of each PdV in its cluster."""

    histo_sales_end_date = config.load["histo_sales_end_date"]
    target = config.load.prediction_mode["cluster_target_pdv_proportion_in_cluster"]
    rolling_days = config.load.prediction_mode["cluster_take_proportion_on_x_rolling_days"]
    metric = config.load.prediction_mode["cluster_take_proportion_based_on"]
    sub_df = df_input_pdv_level[["DATE", "META_UB", "COD_SITE", target]]
    sub_df = (
        sub_df.groupby(["DATE", "META_UB", "COD_SITE"], as_index=False)
        .sum()
        .reset_index()[["DATE", "COD_SITE", "META_UB", f"{target}"]]
    )

    sub_df = sub_df.merge(clustering_mapping, on=["COD_SITE", "META_UB"])

    # Add (PDV, COD_SITE) without any sales on histo sales end date to be able to have their proportion even if they
    # had no sales on one day
    meta_ub_code_site = sub_df[["COD_SITE", "META_UB", "CLUSTER"]].drop_duplicates()
    sub_df_filtered = sub_df[sub_df["DATE"] == histo_sales_end_date]
    missing_meta_ub_cod_site = pd.merge(
        meta_ub_code_site, sub_df_filtered, on=["COD_SITE", "META_UB", "CLUSTER"], how="outer", indicator=True
    )
    missing_meta_ub_cod_site = missing_meta_ub_cod_site[missing_meta_ub_cod_site["_merge"] == "left_only"].drop(
        columns=["_merge"]
    )
    missing_meta_ub_cod_site[target] = 0
    missing_meta_ub_cod_site["DATE"] = histo_sales_end_date
    sub_df = pd.concat([sub_df, missing_meta_ub_cod_site])

    # Get median volume of sales per PDV over last rolling days.
    sub_df = sub_df.sort_values(["DATE"], ascending=True)
    sub_df[f"MEDIAN_{target}_{rolling_days}d"] = sub_df.groupby(["META_UB", "COD_SITE"])[target].transform(
        lambda x: x.rolling(rolling_days, 1).median()
    )

    # Get mean volume of sales per PDV over last rolling days (excluding 0).
    sub_df[f"{target}_excl_0"] = np.where(sub_df[f"{target}"] == 0, np.nan, sub_df[f"{target}"])
    sub_df[f"MEAN_{target}_{rolling_days}d"] = sub_df.groupby(["META_UB", "COD_SITE"])[f"{target}_excl_0"].transform(
        lambda x: x.rolling(rolling_days, 1).mean()
    )

    # Get volume of clusters over same period
    concat_df_gb_cluster = sub_df.groupby(["DATE", "META_UB", "CLUSTER"], as_index=False).sum()
    concat_df_gb_cluster = concat_df_gb_cluster.rename(
        {
            f"MEAN_{target}_{rolling_days}d": f"MEAN_{target}_{rolling_days}d_CLUSTER",
            f"MEDIAN_{target}_{rolling_days}d": f"MEDIAN_{target}_{rolling_days}d_CLUSTER",
        },
        axis=1,
    )

    # Compute proportion using pdv volumes and cluster volumes over last 20 days
    df_proportion_pdv_in_cluster = sub_df.merge(
        concat_df_gb_cluster[
            [
                "DATE",
                "META_UB",
                "CLUSTER",
                f"MEAN_{target}_{rolling_days}d_CLUSTER",
                f"MEDIAN_{target}_{rolling_days}d_CLUSTER",
            ]
        ],
        on=["DATE", "META_UB", "CLUSTER"],
        validate="m:1",
    )
    df_proportion_pdv_in_cluster[f"PROPORTION_{target}_MEAN_{rolling_days}d"] = (
        df_proportion_pdv_in_cluster[f"MEAN_{target}_{rolling_days}d"]
        / df_proportion_pdv_in_cluster[f"MEAN_{target}_{rolling_days}d_CLUSTER"]
    )
    df_proportion_pdv_in_cluster[f"PROPORTION_{target}_MEDIAN_{rolling_days}d"] = (
        df_proportion_pdv_in_cluster[f"MEDIAN_{target}_{rolling_days}d"]
        / df_proportion_pdv_in_cluster[f"MEDIAN_{target}_{rolling_days}d_CLUSTER"]
    )

    df_proportion_pdv_in_cluster = df_proportion_pdv_in_cluster[
        df_proportion_pdv_in_cluster["DATE"] == histo_sales_end_date
    ]

    df_proportion_pdv_in_cluster = df_proportion_pdv_in_cluster[
        [
            "COD_SITE",
            "META_UB",
            f"PROPORTION_{target}_{metric}_{rolling_days}d",
        ]
    ]
    df_proportion_pdv_in_cluster = df_proportion_pdv_in_cluster.rename(
        columns={f"PROPORTION_{target}_{metric}_{rolling_days}d": "WEIGHT_PDV_IN_CLUSTER"}
    )
    df_proportion_pdv_in_cluster["WEIGHT_PDV_IN_CLUSTER"] = df_proportion_pdv_in_cluster[
        "WEIGHT_PDV_IN_CLUSTER"
    ].fillna(0)

    return df_proportion_pdv_in_cluster
