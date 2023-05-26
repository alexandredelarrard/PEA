#
# Copyright (c) 2023 by Boston Consulting Group. All rights reserved
#
# -*- coding: utf-8 -*-

import logging
from datetime import datetime, timedelta
from typing import List

import numpy as np
import pandas as pd
from jours_feries_france import JoursFeries
from vacances_scolaires_france import SchoolHolidayDates


def _remove_unrealistic_data(raw_data: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    """
    Args:
        raw_data (pd.DataFrame): sales data to be cleaned in order to get processed data
        mapping (pd.DataFrame): custom mapping UB (UB_CODE) to meta UB (banane bio, tomate ronde)
    Returns:
        - raw_data (pd.DataFrame): raw sales data with following transformations :
        - Price feature added
        - Price feature added
        - Filtered on UBs in our scope (from mapping)
        - Negative or null sales removed
    """
    raw_data["DATE"] = pd.to_datetime(raw_data["DATE"], format="%Y-%m-%d")
    raw_data["PRIX_UNITAIRE"] = np.where(
        raw_data["QTE_VTE"] == 0,
        np.nan,
        raw_data["MNT_VTE"] / raw_data["QTE_VTE"],
    )
    raw_data = raw_data.loc[raw_data["UB_CODE"].isin(mapping["UB_CODE"].unique())]

    shape = raw_data.shape[0]
    raw_data = raw_data.loc[raw_data["MNT_VTE"] >= 0]
    raw_data = raw_data.loc[raw_data["QTE_VTE"] >= 0]
    logging.info(f"Removed {shape - raw_data.shape[0]} / {shape} observations with negative sell / quantity")

    return raw_data


def _clean_redundant_ub(raw_data: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and select UB to perfom modeling on
    Some UB have different references and then are grouped together based on pre validated mapping with Base.
    This is listed and saved in the mapping file in column REPLACE_UB.
    We tried to ensure UBs ordered to the base are the same than those sold by stores.

    Args:
        raw_data (DataFrame): input sales
        mapping (DataFrame): mapping files between UBs

    Returns:
        DataFrame: raw_data
    """

    raw_data = raw_data.merge(
        mapping[["UB_CODE", "META_UB", "IN_COMMANDE_PDV", "REPLACE_UB"]], on="UB_CODE", how="left", validate="m:1"
    )
    raw_data["UB_CODE"] = np.where(
        raw_data["REPLACE_UB"].isin([np.nan, "OK"]), raw_data["UB_CODE"], raw_data["REPLACE_UB"]
    )
    raw_data = (
        raw_data.groupby(["DATE", "META_UB", "UB_CODE", "COD_SITE"])
        .aggregate({"MNT_VTE": sum, "PRIX_UNITAIRE": np.median, "QTE_VTE": sum, "IN_COMMANDE_PDV": np.min})
        .reset_index()
    )

    return raw_data


def _fill_missing_days(raw_data: pd.DataFrame, max_date="") -> pd.DataFrame:
    """
    Add missing dates for each tuple [unit col, granularity col] usually : [UB_CODE, COD_SITE] to have in the processed
    data all dates for [UB_CODE, COD_SITE]. For new added dates :
        - Missing information such as sales, quantity sold, weight sold will be set to nan
        - Missing information such as Prices (PRIX_UNITAIRE) will be filled forward and backward to always have a value

    Args:
        raw_data (pd.DataFrame): sales data to be cleaned in order to get processed data
        unit_col (str): column to indicate product level of filling missing
        granularity_col (str):  column to indicate location level of filling missing
        max_date (str): set max date to fill missing days
        post_process (bool): indicates if function is used in post processing

    Returns:
        Sales data with all dates for each tuple [unit col, granularity col] chosen
    """

    if max_date == "":
        max_date = raw_data["DATE"].max()

    my_dates = pd.date_range(raw_data["DATE"].min(), max_date, freq="D")
    dates_df = pd.DataFrame(my_dates, columns=["DATE"])

    unit_granularity_df = raw_data[["COD_SITE", "UB_CODE"]].drop_duplicates().reset_index(drop=True)
    crossings = pd.DataFrame(dates_df["DATE"].unique()).merge(unit_granularity_df, how="cross")
    crossings.columns = ["DATE", "COD_SITE", "UB_CODE"]

    idx = pd.MultiIndex.from_frame(crossings)
    raw_data = raw_data.set_index(["DATE", "COD_SITE", "UB_CODE"]).reindex(idx)
    raw_data = raw_data.reset_index()

    return raw_data


def _filter_meaningful_ub(raw_data: pd.DataFrame) -> List:
    """
    Removes UBs with less than X observation in total over the past 3/4 years
    X observation : at least 10 days per year for at least 5% of stores

    Args:
          df (pd.DataFrame): sales data to be cleaned in order to get processed data

    Returns:
        List: list of ub to drop
    """

    raw_data["DATE"] = pd.to_datetime(raw_data["DATE"], format="%Y-%m-%d")
    nbr_pdv = len(raw_data["COD_SITE"].unique())
    nbr_days = (raw_data["DATE"].max() - raw_data["DATE"].min()).days

    # at least 10 days per year for at least 5% of pdvs
    # otherwise no prediction
    nbr_observations_min = (max(nbr_days * 1.5 / 52, 1)) * (max(nbr_pdv * 0.05, 1))

    # too few observations
    counting_values = raw_data["UB_CODE"].value_counts()
    too_few_obs_ub = counting_values.loc[counting_values < nbr_observations_min].index
    logging.info(f"nbr UB with too few observations is {len(too_few_obs_ub)} less than {nbr_observations_min}")

    # remove when all values are null
    agg_sales_per_ub_pdv = (
        raw_data[["UB_CODE", "COD_SITE", "MNT_VTE"]]
        .groupby(["COD_SITE", "UB_CODE"], as_index=False)
        .aggregate({"MNT_VTE": sum})
    )
    agg_sales_per_ub_pdv["TUPLE_UB_PDV"] = list(zip(agg_sales_per_ub_pdv.UB_CODE, agg_sales_per_ub_pdv.COD_SITE))

    liste_ub_to_drop = agg_sales_per_ub_pdv.loc[agg_sales_per_ub_pdv["MNT_VTE"] <= 0, "TUPLE_UB_PDV"].unique()
    logging.info(f"nbr UB with no sell is {len(liste_ub_to_drop)}")

    return list(set(list(too_few_obs_ub) + list(liste_ub_to_drop)))


def _enrich_histo_data(raw_data: pd.DataFrame, meteo: pd.DataFrame) -> pd.DataFrame:
    """
    Adds meteorological data to sales data
    Args:
          raw_data (pd.DataFrame): sales data to be cleaned in order to get processed data
          meteo (pd.DataFrame): cleaned historical and forecasted meteorological data
    Returns:
        Sales data enriched with meteorological data
    """
    meteo["DATE"] = pd.to_datetime(meteo["DATE"].astype(str).str[:10], format="%Y-%m-%d")
    meteo = meteo.groupby("DATE").mean().reset_index()

    # merge ouverture and df
    raw_data = raw_data.merge(meteo, on=["DATE"], how="left", validate="m:1")

    return raw_data


def _merge_ub_mapping(df: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    """
    Adds custom mapping info and deduce when a UB is VRAC or PIECE
    Args:
        df (pd.DataFrame): sales data to be cleaned in order to get processed data
        mapping (pd.DataFrame): custom mapping UB (UB_CODE) to meta UB (banane bio, tomate ronde)
    Returns:
        Sales data with custom mapping info and UB type
    """

    # prepare mapping file
    mapping = mapping.rename(columns={"POIDS_UC": "POIDS_UC_MAPPING"})

    mapping["UB_CODE"] = np.where(mapping["REPLACE_UB"].isin([np.nan, "OK"]), mapping["UB_CODE"], mapping["REPLACE_UB"])
    mapping = mapping.drop(["SOUS_FAMILLE", "REPLACE_UB", "IN_COMMANDE_PDV"], axis=1)
    mapping = mapping.drop_duplicates("UB_CODE")

    # Merge on UB_CODE for mapping
    df = df.merge(mapping, on="UB_CODE", how="left", validate="m:1")

    # Deduce vrac
    df["TYPE_UB"] = df["UB_NOM"].apply(lambda x: "VRAC" if "VRAC" in str(x) else "PIECE")

    return df


def _jours_feriers(df: pd.DataFrame, date_feature="DATE") -> pd.DataFrame:
    """
    Args:
        df (pd.DataFrame): sales data to be cleaned in order to get processed data
    Returns:
        sales data with flags for public holiday, special days (e.g. valentine's day) and 3-days or 4-days week-ends
    """

    min_year = df[date_feature].min().year
    max_year = datetime.today().year + 1

    liste_days_pacques = []
    for year in range(min_year, max_year):
        liste_days_pacques.append(JoursFeries.lundi_paques(year).strftime("%Y-%m-%d"))

    liste_days_pentecote = []
    for year in range(min_year, max_year):
        liste_days_pentecote.append(JoursFeries.lundi_pentecote(year).strftime("%Y-%m-%d"))

    liste_days_ascension = []
    for year in range(min_year, max_year):
        liste_days_ascension.append(JoursFeries.ascension(year).strftime("%Y-%m-%d"))

    liste_days_jour_noel = []
    for year in range(min_year, max_year):
        liste_days_jour_noel.append(JoursFeries.jour_noel(year).strftime("%Y-%m-%d"))

    liste_days_onze_novembre = []
    for year in range(min_year, max_year):
        liste_days_onze_novembre.append(JoursFeries.onze_novembre(year).strftime("%Y-%m-%d"))

    liste_days_toussaint = []
    for year in range(min_year, max_year):
        liste_days_toussaint.append(JoursFeries.toussaint(year).strftime("%Y-%m-%d"))

    liste_days_assomption = []
    for year in range(min_year, max_year):
        liste_days_assomption.append(JoursFeries.assomption(year).strftime("%Y-%m-%d"))

    liste_days_quatorze_juillet = []
    for year in range(min_year, max_year):
        liste_days_quatorze_juillet.append(JoursFeries.quatorze_juillet(year).strftime("%Y-%m-%d"))

    liste_days_premier_mai = []
    for year in range(min_year, max_year):
        liste_days_premier_mai.append(JoursFeries.premier_mai(year).strftime("%Y-%m-%d"))

    liste_days_huit_mai = []
    for year in range(min_year, max_year):
        liste_days_huit_mai.append(JoursFeries.huit_mai(year).strftime("%Y-%m-%d"))

    liste_days_premier_janvier = []
    for year in range(min_year, max_year):
        liste_days_premier_janvier.append(JoursFeries.premier_janvier(year).strftime("%Y-%m-%d"))

    all_days_off = (
        liste_days_pacques
        + liste_days_pentecote
        + liste_days_premier_janvier
        + liste_days_huit_mai
        + liste_days_premier_mai
        + liste_days_quatorze_juillet
        + liste_days_assomption
        + liste_days_toussaint
        + liste_days_onze_novembre
        + liste_days_jour_noel
        + liste_days_ascension
    )

    # CREATE DISTANCE VARIABLE -> necessary this way since
    # if prediction is for 31/12 then we won't know next day is christmas
    # so we need to do it date by date ...
    df["DIST_TO_DAY_OFF"] = -5

    for i in range(-3, 4):
        new_days_off = []
        for day_off in all_days_off:
            new_days_off.append(pd.to_datetime(day_off, format="%Y-%m-%d") + timedelta(days=i))
        df.loc[df[date_feature].isin(new_days_off), "DIST_TO_DAY_OFF"] = i

    # days off as string
    df["FERIER"] = np.where(
        df[date_feature].isin(liste_days_onze_novembre),
        "ARMISTICE 1 WW",
        np.where(
            df[date_feature].isin(liste_days_quatorze_juillet),
            "FETE NATIONALE",
            np.where(
                df[date_feature].isin(liste_days_jour_noel),
                "NOEL",
                np.where(
                    df[date_feature].isin(liste_days_assomption),
                    "ASSOMPTION",
                    np.where(
                        df[date_feature].isin(liste_days_premier_mai),
                        "TRAVAIL",
                        np.where(
                            df[date_feature].isin(liste_days_huit_mai),
                            "ARMISTICE 2 WW",
                            np.where(
                                df[date_feature].isin(liste_days_toussaint),
                                "TOUSSAIN",
                                np.where(
                                    df[date_feature].isin(liste_days_premier_janvier),
                                    "SAINT SYLVESTRE",
                                    np.where(
                                        df[date_feature].isin(liste_days_pacques),
                                        "PACQUES",
                                        np.where(
                                            df[date_feature].isin(liste_days_ascension),
                                            "ASCENSION",
                                            np.where(
                                                df[date_feature].isin(liste_days_pentecote),
                                                "PENTECOTE",
                                                "None",
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )

    return df


def _vacances_scolaires(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds french school holidays flag with right zone for our stores
    Args:
          df (pd.DataFrame): sales data to be cleaned in order to get processed data
    Returns:
        Sales data with a flag for french school holidays dates
    """

    all_holidays = []

    min_year = df["DATE"].min().year
    max_year = datetime.today().year

    # ADD holidays infos
    d = SchoolHolidayDates()
    for year in range(min_year, max_year + 1):
        all_holidays = all_holidays + list(d.holidays_for_year_and_zone(year, "B").keys())

    df["HOLIDAYS"] = (df["DATE"].isin(all_holidays)) * 1
    return df


def clean_meteo_data(meteo: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans meteo data using following steps:
        - Transform dates into right format
        - Round precipitations and temperature
        - Convert and round pression
        - Forward fill missing datas

    Args:
        meteo (pd.DataFrame): historical and forecasted meteorological data
    Returns:
        cleaned historical and forecasted meteorological data
    """

    meteo["DATE"] = pd.to_datetime(meteo["DATE"], format="%Y-%m-%d")

    # handle precipitations
    meteo["PRECIPITATIONS_3H"] = meteo["PRECIPITATIONS_3H"].fillna(0)
    pluie = meteo[["DATE", "PRECIPITATIONS_3H"]].groupby("DATE").sum().reset_index()
    pluie["PRECIPITATIONS"] = pluie["PRECIPITATIONS_3H"].round(0)

    # handle pression and temp
    temp = meteo.loc[meteo["TIME"].astype(str) == "12:00:00"][["DATE", "TEMPERATURE", "PRESSION", "NEBULOSITE"]]
    temp["PRESSION"] = (temp["PRESSION"] / 100).round(0) * 100
    temp["TEMPERATURE"] = temp["TEMPERATURE"].round(0)

    df = pd.merge(
        pluie[["DATE", "PRECIPITATIONS"]],
        temp,
        on="DATE",
        how="left",
        validate="1:1",
    )

    for col in ["PRESSION", "TEMPERATURE", "PRECIPITATIONS", "NEBULOSITE"]:
        df[col] = df[col].ffill()

    return df


def check_sql_max_available_date(df: pd.DataFrame, histo_sales_end_date: datetime) -> None:

    current_max_date_sql = df["DATE"].max()
    horizon = (current_max_date_sql - histo_sales_end_date).days

    if histo_sales_end_date <= current_max_date_sql:
        logging.warning(
            f"SQL max date is {current_max_date_sql} - we can predict J+{horizon} with Max chosen historical date {histo_sales_end_date}"
        )
    else:
        logging.warning(
            f"SQL max date {current_max_date_sql} instead of {histo_sales_end_date} \n can only predict J+{horizon}"
        )
        histo_sales_end_date = current_max_date_sql

    return histo_sales_end_date, horizon


def create_target_for_prediction(raw_data: pd.DataFrame) -> pd.DataFrame:

    # filter realistic values
    raw_data["TARGET"] = (raw_data["MNT_VTE"] * raw_data["POIDS_UC_MAPPING"]) / raw_data["PRIX_UNITAIRE_RETRAITE"]
    raw_data = raw_data.drop(["YEAR_SELL", "META_UB_FOR_PDV"], axis=1)

    return raw_data


############### BASE SPECIFIC ################


def pre_aggregation_base(raw_data: pd.DataFrame, pdv_info: pd.DataFrame):
    """
    Aggregate raw_data so that function _fill_missing_days is optimized in the case of a BASE model for +100 PDVs, else
    it will generate an out of memory error.
    Aggregate amount sold (MNT_VTE), quantity sold (QTE_VTE) using sum over stores to
    reconstitute BASE level.
    Concerning Unit Price (PRIX_UNITAIRE) : it will be aggregated using a weighted average on amount sold per store
    (COD_SITE) over its warehouse (COD_BASE)
    Args:
        raw_data: (pd.DataFrame), raw data containing sales info at pdv x ub level per day

    Returns: an aggregated raw data with all values of COD_SITE = BASE.
    """

    logging.info(f"Agregating at BASE level from {raw_data.COD_SITE.nunique()} PDVs")
    raw_data = raw_data.merge(pdv_info[["COD_SITE", "CODE_BASE"]], on=["COD_SITE"], how="left", validate="m:1")
    raw_data["COD_SITE"] = "BASE_" + raw_data["CODE_BASE"]

    raw_data = (
        raw_data.groupby(["DATE", "UB_CODE", "UB_NOM", "META_UB", "META_UB_FOR_PDV", "COD_SITE", "TYPE_UB"])
        .aggregate(
            {
                "MNT_VTE": sum,
                "QTE_VTE": sum,
                "PRIX_UNITAIRE_RETRAITE": np.median,
                "IS_SEASONAL": np.median,
                "POIDS_UC_MAPPING": np.median,
            }
        )
        .reset_index()
    )

    raw_data["PRIX_UNITAIRE_RETRAITE"] = np.where(
        raw_data["PRIX_UNITAIRE_RETRAITE"] == 0, np.nan, raw_data["PRIX_UNITAIRE_RETRAITE"]
    )

    return raw_data


############### CLUSTER SPECIFIC ################


def pre_aggregation_cluster(raw_data_cluster: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate raw_data in the case of a model by clusters for +100 PDVs. Features such as :
        - Target (TARGET), sales (MNT_VTE), quantity sold (QTE_VTE) will be aggregated using sum
        - Relative to dates will be aggregated using "first" method it does not change

    Args:
        raw_data_cluster: (pd.DataFrame), raw data containing sales info at pdv x ub level per day

    Returns:
        raw_data_cluster_level: (pd.DataFrame), aggregated raw data (cluster x ub level per day) with all values of COD_SITE = cluster pdv belongs to

    """

    logging.info(f"Agregating at Cluster level from {raw_data_cluster.COD_SITE.nunique()} PDVs")
    weight = (
        raw_data_cluster.groupby(["DATE", "UB_CODE", "CLUSTER"], as_index=False)
        .MNT_VTE.sum()
        .rename(columns={"MNT_VTE": "MNT_VTE_CLUSTER"})
    )
    raw_data_cluster = raw_data_cluster.merge(weight, on=["DATE", "UB_CODE", "CLUSTER"], validate="m:1")
    raw_data_cluster["WEIGHT"] = raw_data_cluster["MNT_VTE"] / (0.1 + raw_data_cluster["MNT_VTE_CLUSTER"])

    for weighted_col in [
        "PRIX_UNITAIRE_RETRAITE",
        "VAR_PRIX_UNITAIRE_RETRAITE_LW",
        "VAR_PRIX_UNITAIRE_RETRAITE_LM",
        "VAR_PRIX_UNITAIRE_RETRAITE_LY",
    ]:
        raw_data_cluster[weighted_col] = raw_data_cluster[weighted_col] * raw_data_cluster["WEIGHT"]

    raw_data_cluster_level = (
        raw_data_cluster.groupby(["DATE", "CLUSTER", "META_UB", "UB_CODE", "UB_NOM", "TYPE_UB", "FERIER"])
        .aggregate(
            {
                "MNT_VTE": sum,
                "QTE_VTE": sum,
                "PRIX_UNITAIRE_RETRAITE": sum,
                "IS_SEASONAL": np.median,
                "POIDS_UC_MAPPING": np.median,
                "PRECIPITATIONS": np.median,
                "TEMPERATURE": np.median,
                "PRESSION": np.median,
                "NEBULOSITE": np.median,
                "HOLIDAYS": np.median,
                "DIST_TO_DAY_OFF": np.median,
                "TARGET": sum,
                "VAR_PRIX_UNITAIRE_RETRAITE_LW": sum,
                "VAR_PRIX_UNITAIRE_RETRAITE_LM": sum,
                "VAR_PRIX_UNITAIRE_RETRAITE_LY": sum,
            }
        )
        .reset_index()
    )

    raw_data_cluster_level = raw_data_cluster_level.rename(columns={"CLUSTER": "COD_SITE"})

    for col in ["TARGET", "PRIX_UNITAIRE_RETRAITE"]:
        raw_data_cluster_level[col] = np.where(
            raw_data_cluster_level["PRIX_UNITAIRE_RETRAITE"] == 0, np.nan, raw_data_cluster_level[col]
        )

    return raw_data_cluster_level
