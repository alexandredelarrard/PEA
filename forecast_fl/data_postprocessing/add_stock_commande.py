#
# Copyright (c) 2023 by Boston Consulting Group. All rights reserved
#
# -*- coding: utf-8 -*-

import logging
from typing import Dict

import numpy as np
import pandas as pd
from forecast_fl.data_preparation.prepare_histo import _fill_missing_days


##################### LOAD STOCKS
def load_stocks(datas: Dict, config: Dict, top_ubs: pd.DataFrame) -> pd.DataFrame:

    if config.load["prediction_granularity"] == "BASE":
        prdt_std = datas["PRODUIT_STD_INFO"]
        prdt_std = prdt_std.loc[prdt_std["UB_CODE"].isin(top_ubs.UB_CODE.unique())]
        prdt_std = prdt_std.drop_duplicates("PRODUIT_STD_CODE")

        stock = datas["STOCK_BASE"].copy()
        stock = stock.merge(prdt_std, how="left", on=["PRODUIT_STD_CODE"], validate="m:1")
        stock["COD_SITE"] = "BASE_" + stock["COD_SITE"]
        stock = stock.rename(columns={"DATE_STOCK": "DATE"})
    else:
        mapping = datas["UB_MAPPING"][["UB_CODE", "POIDS_UC"]]

        stock = datas["STOCK_PDV"].copy()
        stock = stock.rename(columns={"DATE_INV": "DATE"})
        stock = stock.merge(mapping, on="UB_CODE", how="left", validate="m:1")
        stock["POIDS_TOTAL_STOCK"] = stock["POIDS_UC"] * stock["QTE_INV"]
        stock = stock.loc[stock["UB_CODE"] != "00000000"]

    stock["DATE"] = pd.to_datetime(stock["DATE"], format="%Y-%m-%d")
    stock = stock.groupby(["DATE", "COD_SITE", "UB_CODE"], as_index=False).POIDS_TOTAL_STOCK.sum()
    stock = stock.loc[stock["UB_CODE"].isin(top_ubs.UB_CODE.unique())]

    return stock


def load_commandes(datas: Dict, config: Dict, top_ubs: pd.DataFrame) -> pd.DataFrame:

    if config.load["prediction_granularity"] == "BASE":
        commandes = datas["COMMANDES_BASE"].copy()
        commandes["COD_SITE"] = "BASE_" + commandes["COD_SITE"]
        # commandes["POIDS_COMMANDE"] *= 100  # data is in quintal not in kg -> No longer the case
        # commandes["POIDS_COMMANDE"] = np.where(
        #     commandes["PRHT"] > 100, commandes["POIDS_COMMANDE"] / 100, commandes["POIDS_COMMANDE"]
        # )
    else:
        commandes = datas["COMMANDES_PDV"].copy()

    commandes["DATE"] = pd.to_datetime(commandes["DATE"], format="%Y-%m-%d")
    commandes = commandes.groupby(["DATE", "COD_SITE", "UB_CODE"], as_index=False).POIDS_COMMANDE.sum()
    commandes = commandes.loc[commandes["UB_CODE"].isin(top_ubs.UB_CODE.unique())]

    return commandes


def aggregate_stock_commande_target(
    df_to_export: pd.DataFrame,
    stock: pd.DataFrame,
    commandes: pd.DataFrame,
    df_input: pd.DataFrame,
    top_ubs: pd.DataFrame,
) -> pd.DataFrame:

    # ensure all values are present on granularity DATExPDVxUB
    my_dates = pd.date_range(
        min(stock["DATE"].min(), commandes["DATE"].min()),
        max(df_to_export["DATE"].max(), commandes["DATE"].max()),
        freq="D",
    )
    dates_df = pd.DataFrame(my_dates, columns=["DATE"])
    crossings = pd.DataFrame(dates_df["DATE"].unique()).merge(top_ubs.drop("META_UB", axis=1), how="cross")
    crossings.columns = ["DATE", "COD_SITE", "UB_CODE"]

    idx = pd.MultiIndex.from_frame(crossings)
    commandes = commandes.set_index(["DATE", "COD_SITE", "UB_CODE"]).reindex(idx)
    commandes = commandes.reset_index()

    # merge with stock & target
    stock_and_commandes = commandes.merge(stock, on=["DATE", "COD_SITE", "UB_CODE"], how="left", validate="1:1")
    stock_and_commandes = stock_and_commandes.merge(
        df_input[["DATE", "COD_SITE", "UB_CODE", "TARGET", "RATIO_NOT_DIRECT"]],
        on=["DATE", "COD_SITE", "UB_CODE"],
        how="left",
        validate="1:1",
    )

    return stock_and_commandes


def extract_stock_and_reconstruct_missing_stock(
    df_to_export: pd.DataFrame, datas: Dict, df_input: pd.DataFrame, top_ubs: pd.DataFrame, config: Dict
):
    """
    The stocks extraction and reconstruction follows the logic below :
        - Aggregate Stock and Order data at UB level x COD_SITE
        - If the stock (per UB x COD_SITE) is available for the day (prediction_date_min - 1 day) we use it -> existing_stock
        - Else : The following steps show how to reconstruct the Stock for the day (prediction_date_min - 1 day)
        - For each UB x COD_SITE, we look into the latest stock available.
        - From that stock we will reconstruct the past stock using this formula Stock Day = Stock Day-1 - Sales Day-1 + Orders Day-2
        - If Sales Day-1 and Orders Day-2 are missing, we will use the hypothesis Sales Day-1 = Orders Day-2, the forumla above will be then Stock Day = Stock Day-1
        - If Sales Day-1 and Orders Day-2 are missing Stock can't be reconstructed and other Stock depending on it as well -> no recommended orders will be done for this UBx COD_SITE, only predicted sales will be provided
        - If there is no stock available to reconstruct further days stocks the formula can't be used as well -> no recommended orders will be computed for this UBx COD_SITE, only predicted sales will be provided

    Args:
        - datas (dict): containing stock and orders historical data
        - prediction_granularity (str): BASE or PDV to indicates which stoch and order data to use
        - df (pd.DatFrame): historical sales data used to reconstruct stock when it's missing
        - top_ubs (pd.DataFrame): UBs in scope used to filter stock and orders data
        - prediction_date_min datetime.date): first date to predict, the only date where we can recommend orders (PRECO_COMMANDES) to place. On next days, we can only provide predicted sales

    Returns:
        Latest (weight) stock extracted and reconstructed delayed by one day to match prediction_date_min per UBx COD_SITE

    """

    # load stock
    stock = load_stocks(datas, config, top_ubs)

    # load commandes
    commandes = load_commandes(datas, config, top_ubs)

    # pre process stock / commandes
    if (not stock.empty) and (not commandes.empty):

        logging.info("MERGE STOCK WITH ORDERS & SALES")
        stock_and_commandes = aggregate_stock_commande_target(df_to_export, stock, commandes, df_input, top_ubs)

        # Add missing days
        stock_and_commandes = _fill_missing_days(raw_data=stock_and_commandes)

    else:
        if stock.empty:
            logging.warning(f"STOCK TABLE IS EMPTY, NO ORDER RECOMMENDATION POSSIBLE")
        if commandes.empty:
            logging.warning(f"ORDER TABLE IS EMPTY, NO ORDER RECOMMENDATION POSSIBLE")

        stock_and_commandes = pd.DataFrame(
            columns=["DATE", "COD_SITE", "UB_CODE", "POIDS_COMMANDE", "POIDS_TOTAL_STOCK", "TARGET", "RATIO_NOT_DIRECT"]
        )

    return stock_and_commandes[
        ["DATE", "COD_SITE", "UB_CODE", "POIDS_COMMANDE", "POIDS_TOTAL_STOCK", "TARGET", "RATIO_NOT_DIRECT"]
    ]
