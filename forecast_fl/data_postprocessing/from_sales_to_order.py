#
# Copyright (c) 2023 by Boston Consulting Group. All rights reserved
#
# -*- coding: utf-8 -*-

from datetime import timedelta
from typing import Dict

import numpy as np
import pandas as pd
from forecast_fl.data_postprocessing.direct_proportion_base import (
    adjust_target_for_base_with_order_day,
)


def reconstruct_stock_rolling_missing(
    stock_and_commandes: pd.DataFrame, config: Dict, max_deduction_days: int = 15
) -> pd.DataFrame:

    # do not deduce more than max_deduction_days days of stock foreward otherwise -> no stock
    for _ in range(max_deduction_days):
        # create dedicated features
        stock_and_commandes["TARGET_J-1"] = (
            stock_and_commandes[["COD_SITE", "UB_CODE", "TARGET"]].groupby(["COD_SITE", "UB_CODE"])["TARGET"].shift(1)
        )

        stock_and_commandes["TARGET_J+1"] = (
            stock_and_commandes[["COD_SITE", "UB_CODE", "TARGET"]].groupby(["COD_SITE", "UB_CODE"])["TARGET"].shift(-1)
        )

        stock_and_commandes["POIDS_COMMANDE_EN_J-2"] = (
            stock_and_commandes[["COD_SITE", "UB_CODE", "POIDS_COMMANDE"]]
            .groupby(["COD_SITE", "UB_CODE"])["POIDS_COMMANDE"]
            .shift(2)
        )
        stock_and_commandes["POIDS_COMMANDE_EN_J-1"] = (
            stock_and_commandes[["COD_SITE", "UB_CODE", "POIDS_COMMANDE"]]
            .groupby(["COD_SITE", "UB_CODE"])["POIDS_COMMANDE"]
            .shift(1)
        )
        stock_and_commandes["POIDS_TOTAL_STOCK_J-1"] = (
            stock_and_commandes[["COD_SITE", "UB_CODE", "POIDS_TOTAL_STOCK"]]
            .groupby(["COD_SITE", "UB_CODE"])["POIDS_TOTAL_STOCK"]
            .shift(1)
        )

        if config.load["prediction_granularity"] == "BASE":
            # TODO: update based on order date when data available
            # everything is for end of day so same as J+1 for stock
            stock_and_commandes["PREDICTED_STOCK_J"] = (
                (
                    stock_and_commandes["POIDS_TOTAL_STOCK_J-1"]
                    - stock_and_commandes["SPOILAGE_RATE_BASE"] * stock_and_commandes["TARGET_J-1"]
                ).clip(0, None)
                - stock_and_commandes["TARGET_J-1"]  # store sales of J+1
                + stock_and_commandes["POIDS_COMMANDE_EN_J-1"]  # J-1 because we work with delivery date not order date
            ).clip(0, None)

        else:
            stock_and_commandes["PREDICTED_STOCK_J"] = (
                (
                    stock_and_commandes["POIDS_TOTAL_STOCK_J-1"]
                    - stock_and_commandes["SPOILAGE_RATE_PDV"] * stock_and_commandes["TARGET_J-1"]
                ).clip(0, None)
                + -stock_and_commandes["TARGET_J-1"]
                + stock_and_commandes["POIDS_COMMANDE_EN_J-1"]  # date of reception
            ).clip(0, None)

        # clip stocks to be betwenn 0 and 6 sales day max
        stock_and_commandes["POIDS_TOTAL_STOCK"] = np.where(
            pd.isnull(stock_and_commandes["POIDS_TOTAL_STOCK"]),
            stock_and_commandes["PREDICTED_STOCK_J"].clip(0, None),
            stock_and_commandes["POIDS_TOTAL_STOCK"],
        )

    stock_and_commandes["POIDS_TOTAL_STOCK"] = stock_and_commandes["POIDS_TOTAL_STOCK"].astype(float).round(1)

    return stock_and_commandes


def fill_missing_target(stock: pd.DataFrame) -> pd.DataFrame:

    # replace future target with predictions
    stock["TARGET"] = np.where(stock["TARGET"].isnull(), stock["PREDICTION_UB_POSTPROCESSED"], stock["TARGET"])

    # fill in target when missing based on J-7 data then average of round 5 days
    stock["TARGET-7"] = stock[["COD_SITE", "UB_CODE", "TARGET"]].groupby(["COD_SITE", "UB_CODE"])["TARGET"].shift(7)

    stock["TARGET_MEAN_5D"] = (
        stock[["COD_SITE", "UB_CODE", "TARGET"]]
        .groupby(["COD_SITE", "UB_CODE"])["TARGET"]
        .transform(lambda x: x.rolling(5, 1, center=True).mean())
    )

    stock["TARGET"] = np.where(
        (stock["TARGET"].isnull()) & (~stock["TARGET-7"].isnull()) & (~stock["TARGET_MEAN_5D"].isnull()),
        0.5 * (stock["TARGET_MEAN_5D"] + stock["TARGET-7"]),
        np.where(
            (stock["TARGET"].isnull()) & (~stock["TARGET-7"].isnull()),
            stock["TARGET-7"],
            np.where(
                (stock["TARGET"].isnull()) & (~stock["TARGET_MEAN_5D"].isnull()),
                stock["TARGET_MEAN_5D"],
                stock["TARGET"],
            ),
        ),
    )

    return stock.drop(["TARGET-7", "TARGET_MEAN_5D"], axis=1)


def fill_missing_commandes(stock: pd.DataFrame) -> pd.DataFrame:
    stock["POIDS_COMMANDE"] = stock["POIDS_COMMANDE"].fillna(0)
    return stock


def adjust_end_of_day_stock_for_base(stock: pd.DataFrame) -> pd.DataFrame:

    # adjust stock for base
    # STOCK is end of day stock for Base so put it as early next day
    stock["POIDS_TOTAL_STOCK"] = (
        stock[["COD_SITE", "UB_CODE", "POIDS_TOTAL_STOCK"]]
        .groupby(["COD_SITE", "UB_CODE"])["POIDS_TOTAL_STOCK"]
        .shift(1)
    )
    return stock


def adjust_to_reception_order_for_pdv(stock: pd.DataFrame) -> pd.DataFrame:

    # adjust past orders to be homogeneous with base
    # orders moving from date of order to date of reception
    stock["POIDS_COMMANDE"] = (
        stock[["COD_SITE", "UB_CODE", "POIDS_COMMANDE"]].groupby(["COD_SITE", "UB_CODE"])["POIDS_COMMANDE"].shift(1)
    )
    return stock


def add_pdv_orders(datas: Dict, stock: pd.DataFrame) -> pd.DataFrame:
    # add orders info for comparison purpose
    orders_pdv = datas["COMMANDES_PDV"]
    agg_orders = orders_pdv[["DATE", "UB_CODE", "POIDS_COMMANDE"]].groupby(["DATE", "UB_CODE"]).sum().reset_index()
    agg_orders = agg_orders.rename(columns={"POIDS_COMMANDE": "COMMANDES_PDV"})

    stock = stock.merge(agg_orders, on=["DATE", "UB_CODE"], how="left", validate="1:1")

    # for base we want to have pdv orders as past sales for base
    stock["TARGET"] = np.where(stock["COMMANDES_PDV"].isnull(), stock["TARGET"], stock["COMMANDES_PDV"])

    return stock


def handle_direct_pdv(
    stock: pd.DataFrame, max_deduction_days: int, minimum_base_proportion: int = 0.25
) -> pd.DataFrame:
    """
    If proportion of orders is significantly lower than the one of sales then we deduce this is direct
    only sales predictions will be shared

    Args:
        stock (pd.DataFrame): stock pre adjusted to PDV X ub level

    Returns:
        pd.DataFrame: stock adjusted
    """

    date_max = stock["DATE"].max()

    has_command_last_x_days = stock.loc[stock["DATE"] >= date_max - timedelta(days=max_deduction_days)]
    has_command_last_x_days = has_command_last_x_days.groupby(["COD_SITE", "UB_CODE"]).sum().reset_index()
    has_command_last_x_days["PROPORTION_NOT_TO_DIRECT"] = has_command_last_x_days["POIDS_COMMANDE"] / (
        0.01 + has_command_last_x_days["TARGET"]
    )

    has_command_last_x_days = has_command_last_x_days[["COD_SITE", "UB_CODE", "PROPORTION_NOT_TO_DIRECT", "TARGET"]]

    stock = stock.merge(
        has_command_last_x_days,
        on=["COD_SITE", "UB_CODE"],
        how="left",
        validate="m:1",
        suffixes=("", f"_LAST_{max_deduction_days}_D"),
    )

    # correct poids commande UB
    stock["POIDS_COMMANDES_UB"] = np.where(
        (stock["PROPORTION_NOT_TO_DIRECT"] < minimum_base_proportion)
        & (stock[f"TARGET_LAST_{max_deduction_days}_D"] > 0),
        np.nan,
        stock["POIDS_COMMANDES_UB"],
    )

    return stock


def deduce_futur_orders_base(stock, min_date):

    stock["ADDITIONAL_STOCK"] = stock["TARGET"] * stock["MIN_PROPORTION_STOCK_BASE"]

    # for fridays we want more stock on the morning
    # stock["ADDITIONAL_STOCK"] = np.where(stock["DATE"].dt.dayofweek == 4, stock["ADDITIONAL_STOCK"]*1.25, stock["ADDITIONAL_STOCK"])

    stock["STOCK_TARGET"] = stock[["POIDS_TOTAL_STOCK", "ADDITIONAL_STOCK"]].max(axis=1)

    # reset stocks to recalculate based on new orders
    stock["POIDS_TOTAL_STOCK"] = np.where(
        stock["DATE"] > min_date,  # strict is important since min_date stock is available
        stock["STOCK_TARGET"],
        stock["POIDS_TOTAL_STOCK"],
    )

    stock["POIDS_COMMANDES_UB"] = (
        (
            stock["TARGET"]
            + stock["TARGET_J+1"] * stock["MIN_PROPORTION_STOCK_BASE"]
            - (stock["POIDS_TOTAL_STOCK"] - stock["SPOILAGE_RATE_BASE"] * stock["TARGET"]).clip(0, None)
        )
        .astype(float)
        .round(0)
        .clip(0, None)
    )

    stock["POIDS_COMMANDE"] = np.where(
        stock["DATE"] >= min_date,
        stock["POIDS_COMMANDES_UB"],
        stock["POIDS_COMMANDE"],
    )

    return stock


def deduce_futur_orders_pdv(stock, min_date):

    stock["ADDITIONAL_STOCK"] = stock["TARGET"] * stock["MIN_PROPORTION_STOCK_PDV"]
    stock["STOCK_TARGET"] = np.where(
        stock["POIDS_TOTAL_STOCK"].isnull(), np.nan, stock[["POIDS_TOTAL_STOCK", "ADDITIONAL_STOCK"]].max(axis=1)
    )

    # reset stocks to recalculate based on new orders
    stock["POIDS_TOTAL_STOCK"] = np.where(
        stock["DATE"] >= min_date,
        stock["STOCK_TARGET"],
        stock["POIDS_TOTAL_STOCK"],
    )

    stock["POIDS_COMMANDES_UB"] = (
        (
            stock["TARGET"]
            + stock["TARGET_J+1"] * stock["MIN_PROPORTION_STOCK_PDV"]
            - (stock["POIDS_TOTAL_STOCK"] - stock["SPOILAGE_RATE_PDV"] * stock["TARGET"]).clip(0, None)
        )
        .astype(float)
        .round(0)
        .clip(0, None)
    )

    # since never going to have predicted orders for J+4 with no J+5 predictions
    # we then order based on J+4 sales not J+5
    stock["POIDS_COMMANDES_UB_J+4"] = (
        (
            stock["TARGET"] * (1 + stock["MIN_PROPORTION_STOCK_PDV"])
            - (stock["POIDS_TOTAL_STOCK"] - stock["SPOILAGE_RATE_PDV"] * stock["TARGET"]).clip(0, None)
        )
        .astype(float)
        .round(0)
        .clip(0, None)
    )

    # ensure no order to give if no stock
    stock["POIDS_COMMANDE"] = np.where(
        (stock["DATE"] >= min_date) & (stock["POIDS_TOTAL_STOCK"].isnull()), np.nan, stock["POIDS_COMMANDE"]
    )

    # ensure at J+4 we deduce the right value
    stock["POIDS_COMMANDES_UB"] = np.where(
        (stock["DATE"] >= min_date) & (~stock["POIDS_COMMANDES_UB"].isnull()),
        stock["POIDS_COMMANDES_UB"],
        np.where(
            (stock["DATE"] >= min_date)
            & (stock["POIDS_COMMANDES_UB"].isnull())
            & (~stock["POIDS_COMMANDES_UB_J+4"].isnull()),
            stock["POIDS_COMMANDES_UB_J+4"],
            stock["POIDS_COMMANDE"],
        ),
    )

    return stock


def from_sales_to_order(config, stock, datas):

    max_deduction_days = config.load.prediction_mode.max_deduction_days_stock  # 30
    minimum_base_proportion = (
        config.load.prediction_mode.minimum_base_proportion
    )  # 0.25 # 25% of UB provided by Base to be considered for preco commande
    min_date = config.load.prediction_date_min

    stock = fill_missing_target(stock)
    stock = fill_missing_commandes(stock)

    if config.load["prediction_granularity"] == "BASE":
        stock = adjust_target_for_base_with_order_day(config, stock, taget_variable="TARGET")
        stock = adjust_end_of_day_stock_for_base(stock)
        stock = add_pdv_orders(datas, stock)
    else:
        stock = adjust_to_reception_order_for_pdv(stock)

    stock = reconstruct_stock_rolling_missing(stock, config, max_deduction_days)

    if config.load["prediction_granularity"] == "BASE":
        stock = deduce_futur_orders_base(stock, min_date=min_date)
    else:
        stock = deduce_futur_orders_pdv(stock, min_date=min_date)

    # no precos of order if > 75% sales are not ordered to base
    if config.load["prediction_granularity"] != "BASE":
        stock = handle_direct_pdv(stock, max_deduction_days, minimum_base_proportion)

    return stock
