#
# Copyright (c) 2023 by Boston Consulting Group. All rights reserved
#
# -*- coding: utf-8 -*-

from datetime import timedelta
from typing import Dict

import pandas as pd


def adjust_target_for_base_with_order_day(
    config: Dict, df_input: pd.DataFrame, taget_variable: str = "TARGET"
) -> pd.DataFrame:

    if config.load["prediction_granularity"] == "BASE":
        df_input[taget_variable] = (
            df_input[["COD_SITE", "UB_CODE", taget_variable]].groupby(["COD_SITE", "UB_CODE"])[taget_variable].shift(-1)
        )

        # for Base the target needs to be adjusted by the direct prorata
        # 100% means all is ordered to base, so no adjustement
        # 0% means all is ordered to direct so no orders to face for Base
        df_input[taget_variable] = df_input[taget_variable] * (df_input["RATIO_NOT_DIRECT"] / 100)

    return df_input


def handle_direct_proportion(
    config: Dict, df_input: pd.DataFrame, datas: Dict, df_to_export: pd.DataFrame
) -> pd.DataFrame:

    nbr_days_for_proportion = 7

    if config.load["prediction_granularity"] == "BASE":

        orders_pdv = datas["COMMANDES_PDV"]
        orders_pdv["DATE"] = pd.to_datetime(orders_pdv["DATE"], format="%Y-%m-%d")
        date_max = df_input["DATE"].max()

        # sales per ub past X days
        sub_input = df_input.loc[
            df_input["DATE"].between(date_max - timedelta(days=nbr_days_for_proportion - 1), date_max)
        ]
        liste_days_sub_input = sub_input["DATE"].unique()

        # orders per ub past X days
        sub_orders_pdv = orders_pdv.loc[
            orders_pdv["DATE"].between(date_max - timedelta(days=nbr_days_for_proportion - 1), date_max)
        ]
        liste_days_sub_orders = sub_orders_pdv["DATE"].unique()

        intersect_days = list(set(liste_days_sub_input).intersection(set(liste_days_sub_orders)))

        # calculate orders & sales
        sales_per_ub = (
            sub_input.loc[sub_input["DATE"].isin(intersect_days)][["UB_CODE", "TARGET"]]
            .groupby("UB_CODE")
            .sum()
            .reset_index()
        )
        orders_per_ub = (
            sub_orders_pdv.loc[sub_orders_pdv["DATE"].isin(intersect_days)][["UB_CODE", "POIDS_COMMANDE"]]
            .groupby("UB_CODE")
            .sum()
            .reset_index()
        )

        sales_and_orders_per_ub = sales_per_ub.merge(orders_per_ub, on="UB_CODE", how="left", validate="1:1")
        sales_and_orders_per_ub = sales_and_orders_per_ub.fillna(0)

        # calculate ratio of base compared to direct
        sales_and_orders_per_ub["RATIO_NOT_DIRECT"] = (
            (sales_and_orders_per_ub["POIDS_COMMANDE"] * 100 / sales_and_orders_per_ub["TARGET"]).round(0).clip(0, 100)
        )
        # sales_orders_per_ub["RATIO_DIRECT"].fillna(0, inplace=True)

        # merge ratio to final export prediction
        df_to_export = df_to_export.merge(
            sales_and_orders_per_ub[["UB_CODE", "RATIO_NOT_DIRECT"]], on="UB_CODE", how="left", validate="m:1"
        )

        # merge ratio to final historical data
        df_input = df_input.merge(
            sales_and_orders_per_ub[["UB_CODE", "RATIO_NOT_DIRECT"]], on="UB_CODE", how="left", validate="m:1"
        )

    else:
        df_to_export["RATIO_NOT_DIRECT"] = 100
        df_input["RATIO_NOT_DIRECT"] = 100

    return df_to_export, df_input
