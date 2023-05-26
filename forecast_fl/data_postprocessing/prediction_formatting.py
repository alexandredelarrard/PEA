#
# Copyright (c) 2023 by Boston Consulting Group. All rights reserved
#
# -*- coding: utf-8 -*-

import logging
from datetime import datetime
from typing import Dict

import numpy as np
import pandas as pd
from forecast_fl.data_postprocessing.add_family_rate_proportion import add_family_rate
from forecast_fl.data_postprocessing.add_stock_commande import (
    extract_stock_and_reconstruct_missing_stock,
)
from forecast_fl.data_postprocessing.direct_proportion_base import (
    adjust_target_for_base_with_order_day,
)
from forecast_fl.data_postprocessing.final_business_rules import (
    business_rules_post_processing,
)
from forecast_fl.data_postprocessing.from_sales_to_order import from_sales_to_order
from forecast_fl.data_preparation.prepare_histo import (
    _fill_missing_days,
    _jours_feriers,
)


def add_historical_target_value_to_prediction(df_to_export: pd.DataFrame, df_input: pd.DataFrame, config: Dict):
    """
    Add historical sales data to perform sanity checks

    Args:

        df_to_export (pd.DataFrame): Raw result of post processing
        df_input (pd.DataFrame): Historical sales data

    Returns:
        Raw result of post processing with historical sales (last 10 days and 28 days before)

    """

    df_input = df_input.copy()
    prediction_date_max = config.load["prediction_date_max"]
    target = config.load.prediction_mode.target

    df_input = _fill_missing_days(df_input, max_date=prediction_date_max)
    df_input = adjust_target_for_base_with_order_day(config=config, df_input=df_input, taget_variable="TARGET")

    cols_to_keep = []
    for day in list(range(2, 11)) + [28]:
        df_input[f"QTE_VTE_POIDS_J-{day}"] = df_input.groupby(["UB_CODE", "COD_SITE"])[target].shift(day)
        cols_to_keep += [f"QTE_VTE_POIDS_J-{day}"]

    df_to_export = df_to_export.merge(
        df_input[["DATE", "UB_CODE", "COD_SITE"] + cols_to_keep], on=["DATE", "UB_CODE", "COD_SITE"], how="left"
    )

    return df_to_export


def data_export_renaming(df_to_export: pd.DataFrame, config: Dict) -> pd.DataFrame:

    logging.debug("Renaming columns")
    df_to_export.rename(
        columns={
            "DATE": "DATE_PREDICTION",
            "UB_CODE": "CODE_UB",
            "UB_NOM": "NOM_UB",
            "POIDS_COMMANDES_UB": "QTE_PRECO_COMMANDES_POIDS",
            "POIDS_COMMANDES_UB_EN_UC": "QTE_PRECO_COMMANDES",
            "PREDICTION_UB_POSTPROCESSED_EN_UC": "QTE_VTE",
            "PREDICTION_UB_POSTPROCESSED": "QTE_VTE_POIDS",
            "META_UB": "MACRO_UB_NOM",
            "PREDICTION_META_UB": "QTE_VTE_POIDS_MACRO_UB",
            "POIDS_TOTAL_STOCK_EN_UC": "POIDS_STOCK_J-1",
            "PREDICTION_UB_USING_PROPORTION": "PREDICTION_FROM_META_UB_VENTILATION",
            "PREDICTION_LGBM_LEVEL_UB": "PREDICTION_FROM_DIRECT_UB",
        },
        inplace=True,
    )

    logging.debug("Deducing missing columns")
    df_to_export["DATE_EXECUTION"] = datetime.now().date()
    df_to_export["DATE_HISTORIQUE_MAX"] = config.load["histo_sales_end_date"]

    df_to_export["UNITE_UB"] = np.where(df_to_export["TYPE_UB"] == "PIECE", "UC", "KG")
    df_to_export["MACRO_UB_CODE"] = np.nan
    df_to_export["QTE_VTE_POIDS_OBSERVE"] = np.nan
    df_to_export["QTE_VTE_OBSERVE"] = np.nan

    return df_to_export


def enrich_data_to_export(df_to_export, df_input):

    historical_sales_cols = [col for col in df_to_export.columns if "QTE_VTE_POIDS_J" in col]

    keep_cols = [
        "DATE",
        "META_UB",
        "COD_SITE",
        "UB_CODE",
        "PREDICTION_HORIZON",
        "WEIGHT_UB_IN_META_UB",
        "PREDICTION_META_UB",
        "PREDICTION_UB_USING_PROPORTION",
        "PREDICTION_LGBM_LEVEL_UB",
        "PREDICTION_UB_POSTPROCESSED",
        "POIDS_COMMANDES_UB",
        "POIDS_TOTAL_STOCK",
        "RATIO_NOT_DIRECT",
    ] + historical_sales_cols
    df_to_export = df_to_export[keep_cols]

    # round results up
    df_to_export["PREDICTION_UB_POSTPROCESSED"] = df_to_export["PREDICTION_UB_POSTPROCESSED"].astype(float).round(0)
    df_to_export["PREDICTION_META_UB"] = df_to_export["PREDICTION_META_UB"].astype(float).round(0)

    # add type of UB nom / poids unitaire
    ub_desc = df_input[["UB_CODE", "UB_NOM", "META_UB", "TYPE_UB", "POIDS_UC_MAPPING"]].drop_duplicates()
    df_to_export = df_to_export.merge(ub_desc, on=["META_UB", "UB_CODE"], how="left")

    return df_to_export


def from_weight_to_uc_orders(df_to_export):

    historical_sales_cols = [col for col in df_to_export.columns if "QTE_VTE_POIDS_J" in col]

    mapping_columns_sales_uc_to_weight = {
        col.replace("QTE_VTE_POIDS_J", "QTE_VTE_J"): col for col in historical_sales_cols
    }

    mapping_columns_sales_uc_to_weight.update(
        {
            "PREDICTION_UB_POSTPROCESSED_EN_UC": "PREDICTION_UB_POSTPROCESSED",
            "POIDS_TOTAL_STOCK_EN_UC": "POIDS_TOTAL_STOCK",
            "POIDS_COMMANDES_UB_EN_UC": "POIDS_COMMANDES_UB",
        }
    )

    for col_uc, col_weight in mapping_columns_sales_uc_to_weight.items():
        df_to_export[col_uc] = np.where(
            df_to_export["TYPE_UB"] == "PIECE",
            df_to_export[col_weight] / df_to_export["POIDS_UC_MAPPING"],
            df_to_export[col_weight],
        )
        df_to_export[col_uc] = df_to_export[col_uc].astype(float).round(0)

    return df_to_export


def merge_stock_and_predictions(stock: pd.DataFrame, df_to_export: pd.DataFrame) -> pd.DataFrame:

    # merge to have all required info to deduce order
    stock = stock.merge(
        df_to_export[["DATE", "COD_SITE", "UB_CODE", "PREDICTION_UB_POSTPROCESSED", "RATIO_NOT_DIRECT"]],
        on=["DATE", "COD_SITE", "UB_CODE"],
        how="left",
        validate="1:1",
        suffixes=("_HISTORY", "_FUTUR"),
    )

    stock["RATIO_NOT_DIRECT"] = stock["RATIO_NOT_DIRECT_HISTORY"].fillna(0) + stock["RATIO_NOT_DIRECT_FUTUR"].fillna(0)

    return stock.drop(["RATIO_NOT_DIRECT_HISTORY", "RATIO_NOT_DIRECT_FUTUR"], axis=1)


def enrich_with_orders(stock: pd.DataFrame, df_to_export: pd.DataFrame) -> pd.DataFrame:

    # merge deductions back to df_to_export
    df_to_export = df_to_export.merge(
        stock[["DATE", "COD_SITE", "UB_CODE", "POIDS_COMMANDES_UB", "POIDS_TOTAL_STOCK"]],
        on=["DATE", "COD_SITE", "UB_CODE"],
        how="left",
        validate="1:1",
    )

    return df_to_export


def format_result_dataframe(
    df_to_export: pd.DataFrame, df_input: pd.DataFrame, datas: Dict, top_ubs: pd.DataFrame, config: Dict
) -> pd.DataFrame:
    """
    Formats the df_to_export to the right format, including renaming of columns and addition of columns.

    Args:
        df_to_export (pd.DataFrame): Raw result of post processing

    Returns:
        Formatted df

    Warnings:
        This function must be aligned with the post processing step

    """

    if not df_to_export.empty:

        # Fromat prediction unit to UC or KG
        logging.info(
            "POSTPROCESSING : Extract existing stock, reconstruct it for missing stock on first day to predict"
        )

        df_to_export = add_historical_target_value_to_prediction(df_to_export, df_input, config)

        # add business rules to final sales prediction (top up 0,X days)
        df_to_export = business_rules_post_processing(df_to_export)

        stock = extract_stock_and_reconstruct_missing_stock(
            df_to_export,
            datas,
            df_input,
            top_ubs,
            config,
        )

        stock = add_family_rate(config=config, datas=datas, stock=stock)

        if not stock.empty:

            # merge to stock to have all informations
            stock = merge_stock_and_predictions(stock, df_to_export)

            # deduce orders from stock and past / futur sales
            # VERY IMPORTANT FUNCTION
            stock = from_sales_to_order(config, stock, datas)

            # merge back to df export data
            df_to_export = enrich_with_orders(stock, df_to_export)

        else:
            df_to_export["POIDS_TOTAL_STOCK"] = np.nan
            df_to_export["POIDS_COMMANDES_UB"] = np.nan

        # Need to shift from pdv sales to pdv orders 1 day before
        # orders are calculated as so in from_sales_to_order function
        if config.load["prediction_granularity"] == "BASE":
            df_to_export = adjust_target_for_base_with_order_day(
                config, df_to_export, taget_variable="PREDICTION_UB_POSTPROCESSED"
            )

        # pre filter dataframe
        df_to_export = enrich_data_to_export(df_to_export, df_input)

        # compute sales and orders by unit from weight column
        df_to_export = from_weight_to_uc_orders(df_to_export)

        # add if ferier or not / dimanche or not -> useful to cumulate precos when needed
        df_to_export = _jours_feriers(df_to_export, date_feature="DATE")
        df_to_export["FERIER"] = np.where(df_to_export["DATE"].dt.weekday == 6, "DIMANCHE", df_to_export["FERIER"])
        df_to_export["FERIER"] = np.where(df_to_export["FERIER"] == "None", np.nan, df_to_export["FERIER"])

        # final renaming and shaping
        df_to_export = data_export_renaming(df_to_export, config)

    return df_to_export
