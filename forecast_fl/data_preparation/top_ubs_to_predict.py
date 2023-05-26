#
# Copyright (c) 2023 by Boston Consulting Group. All rights reserved
#
# -*- coding: utf-8 -*-

import logging
from datetime import timedelta
from typing import Dict

import pandas as pd


def filter_to_ub_to_predict_directly(df: pd.DataFrame, target: str, nbr_months_ub_prop: int = 12):
    """Keep UB bringing at least 1 euros per day on average for past X months per PDV

    Args:
        df (pd.DataFrame): ub X granularity observations
        target (str): _description_
        min_proportion (float): _description_
        nbr_months_ub_prop (int, optional): nbr months to look for porpotion. Defaults to 12.

    Returns:
        _type_: _description_
    """

    df = df[["DATE", "META_UB", "COD_SITE", "UB_CODE", target]]

    # keep last 12 months
    last_x_months = df["DATE"].max() - timedelta(days=int(nbr_months_ub_prop * 31))
    sub_df = df[df["DATE"] >= last_x_months]

    # keep UBs with at least 1 euros sell per day per PDV
    nbr_pdvs = sub_df["COD_SITE"].nunique()
    liste_ub = sub_df[["UB_CODE", "MNT_VTE"]].groupby("UB_CODE").sum().reset_index()
    liste_ub = liste_ub.loc[liste_ub["MNT_VTE"] > nbr_pdvs * nbr_months_ub_prop * 31]

    return liste_ub["UB_CODE"].unique()


############ filter meta ubs!


def keep_significant_meta_ub_per_pdv(df: pd.DataFrame, config_prediction: Dict, target: str = "TARGET") -> pd.DataFrame:

    nbr_months_ub_prop = 12

    # keep last 18 months
    last_x_months = df["DATE"].max() - timedelta(days=int(30.5 * nbr_months_ub_prop))
    sub_df = df[df["DATE"] >= last_x_months]

    # FILTER to meaningful META_UB x PDV
    meta_ubs_to_keep = filter_to_meta_ub_granularity(sub_df=sub_df, nbr_months_ub_prop=nbr_months_ub_prop)
    df = pd.merge(df, meta_ubs_to_keep, on=["META_UB", "COD_SITE"], how="right", validate="m:1")

    return df


def filter_to_meta_ub_granularity(sub_df: pd.DataFrame, nbr_months_ub_prop: int) -> pd.DataFrame:
    """

    Args :
        df (pd.DataFrame): sales data to be transformed to get processed data

    Return:
        sales data filtered on META UBs per Granularity level

    """

    minimum_cash_to_median = 1 / 3
    minimum_observations_to_periode = 1 / 3

    # remove META_UB x PDV pairs when less than 3 times all stores median sell per Meta UB
    df_meta_pdv = (
        sub_df[["DATE", "META_UB", "COD_SITE", "MNT_VTE"]].groupby(["DATE", "META_UB", "COD_SITE"]).sum().reset_index()
    )
    agg_meta_pdv = df_meta_pdv.groupby(["META_UB", "COD_SITE"]).median(numeric_only=True).reset_index()
    meta_ub_median = df_meta_pdv.groupby("META_UB").median(numeric_only=True).reset_index()
    agg_meta_pdv = agg_meta_pdv.merge(
        meta_ub_median, on="META_UB", how="left", validate="m:1", suffixes=("", "_MEDIAN")
    )
    agg_meta_pdv["TO_REMOVE"] = agg_meta_pdv["MNT_VTE_MEDIAN"] * minimum_cash_to_median > agg_meta_pdv["MNT_VTE"]

    # remove META_UB x PDV pairs when too few obs
    df_nbr_obs = sub_df[["COD_SITE", "META_UB"]].drop_duplicates().shape[0]
    sub_df = sub_df[["DATE", "COD_SITE", "META_UB"]].drop_duplicates().copy()
    nbr_sell_per_pdv_meta = sub_df.groupby(["COD_SITE", "META_UB"]).count().reset_index()
    nbr_sell_per_pdv_meta = nbr_sell_per_pdv_meta.fillna(0)
    nbr_sell_per_pdv_meta["TO_REMOVE"] = nbr_sell_per_pdv_meta["DATE"] < int(
        minimum_observations_to_periode * 30.5 * nbr_months_ub_prop
    )

    # merge the 2 constraints
    agg_meta_pdv = agg_meta_pdv.merge(
        nbr_sell_per_pdv_meta,
        on=["META_UB", "COD_SITE"],
        how="left",
        validate="1:1",
        suffixes=("_TOO_LOW_CASH", "_TOO_LOW_OBSERVATIONS"),
    )
    agg_meta_pdv["TO_REMOVE"] = agg_meta_pdv["TO_REMOVE_TOO_LOW_CASH"] | agg_meta_pdv["TO_REMOVE_TOO_LOW_OBSERVATIONS"]

    # 35% of days will have
    logging.info(
        f"REMOVED {sum(agg_meta_pdv['TO_REMOVE'])} / {df_nbr_obs} pairs (META_UB, PDV) since less than "
        + f"{int(minimum_observations_to_periode*30.5*nbr_months_ub_prop)} days sold over past {nbr_months_ub_prop} months"
        + f" And less than {minimum_cash_to_median} times median sales per day on similar Meta UB"
    )

    return agg_meta_pdv[~agg_meta_pdv["TO_REMOVE"]][["META_UB", "COD_SITE"]]


############ filter meta ub


def filter_to_ub_granularity(sub_df: pd.DataFrame, target: str, min_proportion: float) -> pd.DataFrame:
    """

    Args :
        df (pd.DataFrame): sales data to be transformed to get processed data

    Return:
        sales data filtered on Top UBs

    """

    # volume sold in x months per cod_site / ub
    sub_df = sub_df.groupby(["META_UB", "COD_SITE", "UB_CODE"], as_index=False)[target].sum()
    meta_ub_sales = sub_df.groupby(["META_UB", "COD_SITE"], as_index=False)[target].sum()

    # calculate proportion of ub in meta ub
    sub_df = sub_df.merge(
        meta_ub_sales, on=["META_UB", "COD_SITE"], how="right", validate="m:1", suffixes=("", "_META_UB")
    )
    sub_df[f"PROPORTION_{target}"] = sub_df[target] / sub_df[target + "_META_UB"]

    # get what proportion the sum of proportion a pari ub, pdv gets
    sub_df = sub_df.sort_values(["COD_SITE", "META_UB", f"PROPORTION_{target}"], ascending=False)
    sub_df["CUMSUM_WEIGHT"] = (
        sub_df[[f"PROPORTION_{target}", "COD_SITE", "META_UB"]].groupby(["COD_SITE", "META_UB"]).cumsum()
    )
    sub_df["CUMSUM_WEIGHT_REMAINING"] = sub_df["CUMSUM_WEIGHT"] - (1 - min_proportion)

    # keep all pairs (ub_n, pdv) such that sum reaches 1- min_proportion
    # or a mono ub reaches min_proportion
    cumsum_condition = sub_df["CUMSUM_WEIGHT_REMAINING"] <= min_proportion * 0.2
    minimal_prop = sub_df["PROPORTION_MNT_VTE"] > min_proportion  # some meta ub are only sold by 1 / 2 ub
    top_ubs = sub_df.loc[cumsum_condition | minimal_prop].drop_duplicates()

    return top_ubs


def keep_ubs(df: pd.DataFrame, top_ubs: pd.DataFrame) -> pd.DataFrame:

    nbr_pairs = df[["COD_SITE", "UB_CODE"]].drop_duplicates().shape[0]
    df = df.merge(top_ubs, on=["META_UB", "COD_SITE", "UB_CODE"], how="right", validate="m:1")
    after_nbr_pairs = df[["COD_SITE", "UB_CODE"]].drop_duplicates().shape[0]
    logging.info(f"FILTERING to UB,PDV pairs, moving from {nbr_pairs} to {after_nbr_pairs} pairs")

    return df.drop(["META_UB", "IN_COMMANDE_PDV"], axis=1)


def get_top_ub(df: pd.DataFrame, config_prediction: Dict, target: str = "TARGET") -> pd.DataFrame:
    """
    Extract all UBs (unité de besoin) of each Meta UB (meta unité de besoin) which cumulated proportions make up at least (100 - min_proportion) % of Meta UB sales per store over given year. These UBs are considered to be our Top UBs and will be the one the modelling and the prediction is done on. For example :
        - Meta UB : banana
            - UBs :
            - Banana type 1 : 30%
            - Banana type 2 : 8%
            - Banana type 3 : 6%
            - Banana type 4 : 55%

        - If min_proportion=0.1, Top UBs will be UBs which cumulated proportions make up at least 90% which are:
        - UB name : cumulated proportion
            - Banana type 4 : 55%
            - Banana type 1 : 85%
            - Banane type 2: 93%

    Banana type 3 will be excluded from the rest of the modelling and prediction parts

    Args:
        df (pd.DataFrame): Input sales dataframe
        config_prediction (Box): Config file - prediction section

    Returns:
        Top UB dataframe for all Meta UB

    """

    min_proportion = config_prediction["top_ub_min_sales_proportion"]
    df = df[["DATE", "META_UB", "COD_SITE", "UB_CODE", target]]

    # keep last 18 months / 9 months / 2 months
    top_ubs_X_months = pd.DataFrame()
    for nbr_months, min_proportion_it in [(2, 0), (9, min_proportion), (18, min_proportion)]:
        last_x_months = df["DATE"].max() - timedelta(days=int(30.5 * nbr_months))
        sub_df = df[df["DATE"] >= last_x_months]

        # FILTER to meaningful UB x PDV for past 2 / 9 / 18 months
        selected_ubs = filter_to_ub_granularity(sub_df=sub_df, target=target, min_proportion=min_proportion_it)
        top_ubs_X_months = pd.concat([top_ubs_X_months, selected_ubs], axis=0)

    top_ubs_X_months = top_ubs_X_months[["COD_SITE", "META_UB", "UB_CODE"]].drop_duplicates()

    # FILTER to meaningful UB x PDV for entire history
    if sub_df["DATE"].min() > df["DATE"].min():
        logging.info(
            "Historique plus grand que 18 mois - xtrait les UB"
            + f"importantes du passé 18mois min = {sub_df['DATE'].min()} vs all min {df['DATE'].min()}"
        )

        top_ubs_full_history = filter_to_ub_granularity(sub_df=df, target=target, min_proportion=min_proportion)
        additional_ubs = (
            top_ubs_full_history.merge(
                top_ubs_X_months[["COD_SITE", "UB_CODE"]],
                on=["COD_SITE", "UB_CODE"],
                validate="1:1",
                how="outer",
                indicator=True,
            )
            .query('_merge=="left_only"')
            .drop("_merge", axis=1)
        )
        additional_ubs = additional_ubs.loc[additional_ubs["PROPORTION_MNT_VTE"] > min(min_proportion, 0.1)]
        top_ubs = pd.concat([top_ubs_X_months, additional_ubs], axis=0)

    else:
        top_ubs = top_ubs_X_months

    return top_ubs[["COD_SITE", "META_UB", "UB_CODE"]].drop_duplicates()
