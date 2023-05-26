#
# Copyright (c) 2023 by Boston Consulting Group. All rights reserved
#
# -*- coding: utf-8 -*-

import gc

import numpy as np
import pandas as pd


def _deduced_price_from_sales(raw_data: pd.DataFrame, smoothing_price: float) -> pd.DataFrame:

    df_cadencier_aggregated = (
        raw_data[["DATE", "UB_CODE", "TYPE_UB", "UB_NOM", "POIDS_UC_MAPPING", "PRIX_UNITAIRE_DEDUCED"]]
        .groupby(["DATE", "UB_CODE", "TYPE_UB", "UB_NOM"])
        .median()
        .reset_index()
    )

    # PRIX unitaire is good for PIECE but wrong for VRAC, need to smooth it for realistic purpose
    df_cadencier_aggregated["PRIX_UNITAIRE_DEDUCED"] = np.where(
        df_cadencier_aggregated["PRIX_UNITAIRE_DEDUCED"].between(0.1, 60),
        df_cadencier_aggregated["PRIX_UNITAIRE_DEDUCED"],
        np.nan,
    )
    df_cadencier_aggregated["PRIX_UNITAIRE_DEDUCED_ROLLED"] = (
        df_cadencier_aggregated[["UB_CODE", "PRIX_UNITAIRE_DEDUCED"]]
        .groupby(["UB_CODE"])["PRIX_UNITAIRE_DEDUCED"]
        .transform(lambda x: x.rolling(smoothing_price, min_periods=1, center=True).median())
    )
    df_cadencier_aggregated["PRIX_UNITAIRE_DEDUCED_ROLLED"] = (
        df_cadencier_aggregated[["UB_CODE", "PRIX_UNITAIRE_DEDUCED_ROLLED"]]
        .groupby("UB_CODE")["PRIX_UNITAIRE_DEDUCED_ROLLED"]
        .bfill()
        .ffill()
    )

    df_cadencier_aggregated["PRIX_UNITAIRE_RETRAITE"] = df_cadencier_aggregated["PRIX_UNITAIRE_DEDUCED_ROLLED"]

    # # edge cases
    # manual_corrections = {
    #     "00695072": 3.5,
    #     "00004386": 1.79,
    #     "00935007": 5.99,
    #     "00200195": 3,
    #     "10002865": 2.99,
    #     "00545221": 5.99,
    #     "00545204": 5.99,
    #     "00003008": 2.99,
    #     "10002709": 3.3,
    #     "00952002": 4,
    #     "00920007": 6,
    #     "10003060": 5,
    #     "00695071": 4.99,
    #     "00940009": 8,
    #     "0000555023": 3.2,
    #     "00003120": 8,
    #     "00420005": 6.5,
    #     "00760046": 9,
    # }
    # for k, v in manual_corrections.items():
    #     df_cadencier_aggregated.loc[df_cadencier_aggregated["UB_CODE"] == k, "PRIX_UNITAIRE_RETRAITE"] = v

    # price can only be between 10 cents and 60 euros
    df_cadencier_aggregated["PRIX_UNITAIRE_RETRAITE"] = np.where(
        df_cadencier_aggregated["PRIX_UNITAIRE_RETRAITE"].between(0.1, 60),
        df_cadencier_aggregated["PRIX_UNITAIRE_RETRAITE"],
        np.nan,
    )

    return df_cadencier_aggregated


def _deduce_price_from_commande_pdv(
    commandes_pdv: pd.DataFrame, raw_data: pd.DataFrame, smoothing_price: float
) -> pd.DataFrame:

    commandes_pdv["DATE"] = pd.to_datetime(commandes_pdv["DATE"], format="%Y-%m-%d")
    commandes_pdv = (
        commandes_pdv[["DATE", "COD_SITE", "UB_CODE", "PRIX_UNITAIRE"]]
        .groupby(["DATE", "COD_SITE", "UB_CODE"])
        .median()
        .reset_index()
    )
    commandes_pdv["PRIX_UNITAIRE"] = np.where(
        commandes_pdv["PRIX_UNITAIRE"].between(0.1, 60), commandes_pdv["PRIX_UNITAIRE"], np.nan
    )

    pdv_price = (
        raw_data[["DATE", "COD_SITE", "UB_CODE", "UB_NOM"]]
        .drop_duplicates()
        .merge(commandes_pdv, on=["DATE", "COD_SITE", "UB_CODE"], how="left", validate="1:1")
    )

    pdv_price["PPTTC_RETRAITE"] = (
        pdv_price[["COD_SITE", "UB_CODE", "PRIX_UNITAIRE"]]
        .groupby(["COD_SITE", "UB_CODE"])["PRIX_UNITAIRE"]
        .transform(lambda x: x.rolling(smoothing_price, min_periods=1, center=True).median())
    )

    return pdv_price


def _fill_missing_price_median(pdv_price):

    median_ub_per_day = (
        pdv_price[["DATE", "UB_CODE", "PPTTC_RETRAITE"]].groupby(["DATE", "UB_CODE"]).median().reset_index()
    ).rename(columns={"PPTTC_RETRAITE": "PRIX_UNITAIRE_MEDIAN_UB_PER_DAY"})

    pdv_price = pdv_price.merge(median_ub_per_day, on=["DATE", "UB_CODE"], how="left", validate="m:1")

    pdv_price["PPTTC_RETRAITE"] = np.where(
        pdv_price["PPTTC_RETRAITE"].isnull(),
        pdv_price["PRIX_UNITAIRE_MEDIAN_UB_PER_DAY"],
        pdv_price["PPTTC_RETRAITE"],
    )

    # 3) complete mvs with median of UB price
    median_ub = (pdv_price[["UB_CODE", "PPTTC_RETRAITE"]].groupby("UB_CODE").median().reset_index()).rename(
        columns={"PPTTC_RETRAITE": "PRIX_UNITAIRE_MEDIAN_UB"}
    )

    pdv_price = pdv_price.merge(median_ub, on=["UB_CODE"], how="left", validate="m:1")

    pdv_price["PPTTC_RETRAITE"] = np.where(
        pdv_price["PPTTC_RETRAITE"].isnull(),
        pdv_price["PRIX_UNITAIRE_MEDIAN_UB"],
        pdv_price["PPTTC_RETRAITE"],
    )

    return pdv_price


def _fill_missing_price_with_sales_median(pdv_price, df_cadencier_aggregated):

    pdv_price = pdv_price.merge(
        df_cadencier_aggregated[["DATE", "UB_CODE", "PRIX_UNITAIRE_RETRAITE", "PRIX_UNITAIRE_DEDUCED"]],
        on=["DATE", "UB_CODE"],
        how="left",
        validate="m:1",
        suffixes=("_MEDIAN", ""),
    )

    median_prix_unitaire = (
        pdv_price[["UB_CODE", "PRIX_UNITAIRE_RETRAITE"]].groupby("UB_CODE").median().reset_index()
    ).rename(columns={"PRIX_UNITAIRE_RETRAITE": "PRIX_UNITAIRE_RETRAITE_MEDIAN"})
    pdv_price = pdv_price.merge(median_prix_unitaire, on=["UB_CODE"], how="left", validate="m:1")

    # 5) For remaining missing price per UB -> 234 remaining with low sell volume complete with :
    # - PRIX_UNITAIRE_RETRAITE
    # - Median price per meta_ub
    pdv_price["PRIX_UNITAIRE_RETRAITE"] = np.where(
        pdv_price["PRIX_UNITAIRE_RETRAITE"].between(
            0.5 * pdv_price["PRIX_UNITAIRE_RETRAITE_MEDIAN"], 2 * pdv_price["PRIX_UNITAIRE_RETRAITE_MEDIAN"]
        ),
        pdv_price["PRIX_UNITAIRE_RETRAITE"],
        pdv_price["PRIX_UNITAIRE_RETRAITE_MEDIAN"],
    )

    return pdv_price


def _retreate_price_outliers(pdv_price, commandes_pdv):

    # 6) If price difference between PPTTC_RETRAITE & PRIX_UNITAIRE_RETRAITE is < 5 % in median,
    # then past can be completed by PRIX_UNITAIRE_RETRAITE
    # instead of filling past with median, we fill with variations from sales

    ubs_low_diff = pdv_price.loc[pdv_price["DATE"].between(commandes_pdv["DATE"].min(), commandes_pdv["DATE"].max())][
        ["UB_CODE", "PRIX_UNITAIRE_RETRAITE", "PPTTC_RETRAITE"]
    ]
    ubs_low_diff = ubs_low_diff.groupby("UB_CODE", group_keys=False).apply(
        lambda x: np.median(abs(x["PRIX_UNITAIRE_RETRAITE"] - x["PPTTC_RETRAITE"]) / x["PPTTC_RETRAITE"])
    )
    ubs_to_compleate = ubs_low_diff.loc[ubs_low_diff <= 0.08].index

    pdv_price["PPTTC_RETRAITE"] = np.where(
        (pdv_price["DATE"] < commandes_pdv["DATE"].min()) & (pdv_price["UB_CODE"].isin(ubs_to_compleate)),
        pdv_price["PRIX_UNITAIRE_RETRAITE"],
        pdv_price["PPTTC_RETRAITE"],
    )
    pdv_price["PRIX_UNITAIRE_RETRAITE"] = np.where(
        pdv_price["PPTTC_RETRAITE"].isnull(), pdv_price["PRIX_UNITAIRE_RETRAITE"], pdv_price["PPTTC_RETRAITE"]
    )
    return pdv_price


def add_prix_hebdomadaire(raw_data: pd.DataFrame, commandes_pdv: pd.DataFrame) -> pd.DataFrame:
    """
    Processes unit price to compute the target. This is a temporary solution used because we do not have a correct
    unit price for VRAC products in the data.
    It uses historical recommended prices per the warehouse to its dedicated stores (BASE): (prix_hebdomadaire and
    prix_hebdomadaire_hp).
    Following processes are applied in order to get final unit price (PRIX_UNITAIRE_RETRAITE):
        - Retreive recommended prices from prix_hebdomadaire and prix_hebdomadaire_hp on UBs in scope ->
        df_cadencier_aggregated
        - Get the average recommended prices per date at UB level (UB_CODE) and use this price for the week coming in
        case of missing prices (usually prices are recommended one or twice a week) -> Weekly price
        - Feature PRIX_UNITAIRE_CADENCIER for each DATE x UB_CODE  will be Weekly price when available, else it will be
        data point from other table of recommended prices prix_hebdomadaire_hp
        - Clean this feature to :
            - remove negative or null prices
            - prices > clip_ratio * median price per UB
            - prices < median price per UB / clip_ratio
            - computes linear interpolation for recommended prices within the limit of limit_interpolation_days
        - For remaining dates without recommended prices, another logic is in place with 3 options:
            - MEDIAN FILLING : when UB doesn't have enough recommended price observations (min_observations_hebdo) or
            very little variationin the recommended price (less than std_ratio_min). MEDIAN FILLING consists in using
            median recommended price for UBs under this filling method for remaining dates without prices
            - ROLLING FILLING : for remaining UBs in cadencier. ROLLING FILLING consists in using unit price variations
            smoothed offseted by the median value from recommended prices for UBs under this filling method for
            remaining dates without prices -> PRIX_UNITAIRE_SMOOTH_7J_OFFSETED. We decided to use this method, because
            we noticed that variations in initial prices (PRIX_UNITAIRE) values of prices did not make sens however,
            variations matched recommended prices variations for overlapping dates.

    Once all these rules are followed we get the processed price PRIX_UNITAIRE_HEBDO_RETRAITE that will be used for
    UBs of type VRAC. For other UBs (PIECE): the initial price (PRIX_UNITAIRE) is correct.
    Args:
        raw_data (pd.DataFrame): raw sales data
        prix_hebdomadaire (pd.DataFrame): Prix hebdo data with promo
        prix_hebdomadaire_hp (pd.DataFrame): Prix hebdo data without promo but more history
        limit_interpolation_days (int): Maximum gap in data for linear interpolation

    Returns:
        pd.DataFrame: updated raw data dataframe with corrected unit price
    """

    smoothing_price = 20  # nbr days to smooth price to reduce too strange variations in data

    raw_data["PRIX_UNITAIRE_DEDUCED"] = raw_data["MNT_VTE"] / raw_data["QTE_VTE"]

    # Init aggregation dateframe
    df_cadencier_aggregated = _deduced_price_from_sales(raw_data, smoothing_price=smoothing_price)

    #############################"" commande pdv prix
    # 1) create all price info to be used for recreating the final signal of price per PDV X UB

    pdv_price = _deduce_price_from_commande_pdv(commandes_pdv, raw_data, smoothing_price=smoothing_price)

    # 2) complete mvs with median of UB price per day
    pdv_price = _fill_missing_price_median(pdv_price)

    # 4 ) retreive and calculate median of prix_unitaire_retraite -> will be used to recreate variations
    pdv_price = _fill_missing_price_with_sales_median(pdv_price, df_cadencier_aggregated)

    # if price too low or too high, replace value
    pdv_price = _retreate_price_outliers(pdv_price, commandes_pdv)

    # free memory after intensive memory allocation
    gc.collect()

    # FINAL CONCLUSION
    raw_data = raw_data.merge(
        pdv_price[["DATE", "COD_SITE", "UB_CODE", "PRIX_UNITAIRE_RETRAITE"]],
        on=["DATE", "COD_SITE", "UB_CODE"],
        how="left",
        validate="m:1",
    )

    median_per_meta_ub = (
        raw_data[["META_UB", "PRIX_UNITAIRE_RETRAITE"]].groupby("META_UB").median().reset_index()
    ).rename(columns={"PRIX_UNITAIRE_RETRAITE": "PRIX_UNITAIRE_RETRAITE_MEDIAN"})
    raw_data = raw_data.merge(median_per_meta_ub, on=["META_UB"], how="left", validate="m:1")

    raw_data["PRIX_UNITAIRE_RETRAITE"] = np.where(
        raw_data["PRIX_UNITAIRE_RETRAITE"].isnull(),
        raw_data["PRIX_UNITAIRE_RETRAITE_MEDIAN"],
        raw_data["PRIX_UNITAIRE_RETRAITE"],
    ).round(2)

    return raw_data.drop(["PRIX_UNITAIRE", "PRIX_UNITAIRE_RETRAITE_MEDIAN", "PRIX_UNITAIRE_DEDUCED"], axis=1)


def add_price_variations(raw_data: pd.DataFrame) -> pd.DataFrame:

    median_ub_price_per_day = (
        raw_data[["DATE", "COD_SITE", "UB_CODE", "PRIX_UNITAIRE_RETRAITE"]]
        .groupby(["DATE", "COD_SITE", "UB_CODE"])
        .median()
        .reset_index()
    )

    median_ub_price_per_day["PRIX_UNITAIRE_RETRAITE_LISSE"] = median_ub_price_per_day.groupby(["COD_SITE", "UB_CODE"])[
        "PRIX_UNITAIRE_RETRAITE"
    ].transform(lambda x: x.rolling(7, min_periods=1, center=True).median())

    # price last week
    median_ub_price_per_day["PRIX_UNITAIRE_RETRAITE_LW"] = median_ub_price_per_day.groupby(["COD_SITE", "UB_CODE"])[
        "PRIX_UNITAIRE_RETRAITE_LISSE"
    ].shift(7)

    # price last month
    median_ub_price_per_day["PRIX_UNITAIRE_RETRAITE_LM"] = median_ub_price_per_day.groupby(["COD_SITE", "UB_CODE"])[
        "PRIX_UNITAIRE_RETRAITE_LISSE"
    ].shift(30)

    # price last year
    median_ub_price_per_day["PRIX_UNITAIRE_RETRAITE_LY"] = median_ub_price_per_day.groupby(["COD_SITE", "UB_CODE"])[
        "PRIX_UNITAIRE_RETRAITE_LISSE"
    ].shift(365)

    # % diff
    for col in ["PRIX_UNITAIRE_RETRAITE_LY", "PRIX_UNITAIRE_RETRAITE_LM", "PRIX_UNITAIRE_RETRAITE_LW"]:
        median_ub_price_per_day["VAR_" + col] = (
            (
                (median_ub_price_per_day["PRIX_UNITAIRE_RETRAITE_LISSE"] - median_ub_price_per_day[col])
                * 100
                / median_ub_price_per_day[col]
            )
            .round(0)
            .clip(-100, 300)
        )

    median_ub_price_per_day = median_ub_price_per_day[
        [
            "DATE",
            "COD_SITE",
            "UB_CODE",
            "VAR_PRIX_UNITAIRE_RETRAITE_LW",
            "VAR_PRIX_UNITAIRE_RETRAITE_LM",
            "VAR_PRIX_UNITAIRE_RETRAITE_LY",
        ]
    ]

    raw_data = raw_data.merge(median_ub_price_per_day, on=["DATE", "COD_SITE", "UB_CODE"], how="left", validate="1:1")

    return raw_data
