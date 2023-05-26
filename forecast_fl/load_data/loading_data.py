#
# Copyright (c) 2023 by Boston Consulting Group. All rights reserved
#
# -*- coding: utf-8 -*-

import datetime
import logging

import numpy as np
import pandas as pd
from box import Box
from forecast_fl.utils.config import Config
from forecast_fl.utils.general_functions import loader_raw_data


def load_meta_ub_mapping(config: Box) -> pd.DataFrame:
    return loader_raw_data(config, "meta_ub_mapping")


def load_prix_hebdomadaire(config: Box) -> pd.DataFrame:
    return loader_raw_data(config, "prix_hebdomadaire")


def generate_pdv_clustering(config: Box, datas) -> pd.DataFrame:
    pdv_clustering = datas["PDV_INFO_SQL"][["COD_SITE", "VOCATION", "CODE_BASE"]]
    pdv_clustering = pdv_clustering[
        pdv_clustering.CODE_BASE.isin(config.load.clustering_pdvs_types.code_base_to_include)
    ]
    pdv_clustering["CLUSTER"] = pdv_clustering.VOCATION.map(config.load.clustering_pdvs_types.mapping)
    if pdv_clustering.shape[0] != pdv_clustering[["COD_SITE", "CLUSTER"]].drop_duplicates().shape[0]:
        logging.error("DATA: Error while creating clustering file. Duplicates found.")
    return pdv_clustering[["COD_SITE", "CLUSTER"]].drop_duplicates()


def load_unit_price_cadencier_hors_promo(config: Box) -> pd.DataFrame:
    """
    Temporary function to load Cadencier data from processing. Ple

    Args:
        config: Config file

    Returns:
        Unit price data

    """

    key = "prix_cadencier_hp_full"
    df_prix_unitaire = loader_raw_data(config, key)

    # Post-Processes
    price_col_name = config.renaming.datasets[key]["PPTC"]
    datecolname = config.renaming.datasets[key]["JourDebutSemaine"]

    # Tries to casts format float "X,x" to "Y.y"
    df_prix_unitaire[price_col_name] = df_prix_unitaire[price_col_name].str.replace(",", ".").astype(float)

    # Cats date column
    df_prix_unitaire[datecolname] = pd.to_datetime(df_prix_unitaire[datecolname])

    return df_prix_unitaire


def get_sales_histo_query(
    query: str,
    clustering: pd.DataFrame,
    meta_ub_mapping: pd.DataFrame,
    config: Config,
) -> str:
    """Builds query of historical sales data in SQL database by filtering on PdVs and UBs of interest

    Args:
        query (str): SQL query without the WHERE clause to filter on
        clustering (pd.DataFrame): Clustering mapping assigning each PdV to a cluster
        meta_ub_mapping (pd.DataFrame): Meta UB mapping assigning each UB_CODE to a META_UB
    Returns:
        Complete SQL query for historical sales data
    """

    prediction_granularity = config.load["prediction_granularity"]
    specific_meta_ubs = config.load["specific_meta_ubs"]
    specific_pdvs = config.load["specific_pdvs"]
    objective = config.load["objective"]
    prediction_date_max = config.load["prediction_date_max"]
    select_prediction_past_X_months = config.load.prediction_mode.select_prediction_past_X_months
    select_training_past_X_years = config.load.parameters_training_model.select_training_past_X_years

    # WHERE CONDITION 1 - Filter PdVs of interest
    # Modeling for specific PdVs requires adding PdV we want to query
    if prediction_granularity == "PDV":
        if len(specific_pdvs) > 0:
            query += f" WHERE vg.CODE_PDV IN ("
            logging.info(f"Querying information for {len(specific_pdvs)} PdVs : {specific_pdvs}")
            for pdv in specific_pdvs:
                query += f"{pdv},"
            query = query[:-1]
            query += ")"

    # Modeling for specific clusters requires finding associated PdVs to query
    elif prediction_granularity == "CLUSTER":
        if len(specific_pdvs) > 0:
            query += f" WHERE vg.CODE_PDV IN ("
            logging.info(
                f'Querying information for {len(clustering["COD_SITE"].unique())} PdVs treated as {len(clustering["CLUSTER"].unique())} clusters.'
            )
            for pdv in clustering["COD_SITE"].unique():
                query += f"{pdv},"
            query = query[:-1]
            query += ")"
            logging.info(query)
        else:
            logging.info("Querying information for all PdVs with cluster approach.")
    # Modeling for BASE with query all PdVs no WHERE condition required in SQL query
    elif prediction_granularity == "BASE":
        logging.info(f"Querying information for BASE PdVs")
    else:
        raise NotImplementedError("Prediction granularity can only be PDV, CLUSTER or BASE.")

    # WHERE CONDITION 2 - Filter UBs of interest
    # Modeling for specific Meta UBs requires finding associated UB_CODE
    if " WHERE " in query:
        query += " AND "
    else:
        query += " WHERE "

    query += "vg.CODE_PRGE IN ("
    if len(specific_meta_ubs) > 0:
        logging.info(f"Querying information for {len(specific_meta_ubs)} META UB: {specific_meta_ubs}")
        ub_to_query = meta_ub_mapping[meta_ub_mapping["META_UB"].isin(specific_meta_ubs)]["UB_CODE"]
    else:
        ub_to_query = list(
            set(
                meta_ub_mapping["UB_CODE"].tolist()
                + meta_ub_mapping.loc[~meta_ub_mapping["REPLACE_UB"].isin([np.nan, "OK"]), "REPLACE_UB"].tolist()
            )
        )
        logging.info(f"Querying information for {len(ub_to_query)} UBS (all in mapping)")

    for ub in ub_to_query:
        query += f"{ub},"
    query = query[:-1]
    query += ")"

    # extract between dates
    # 18 months for prediction, up to max_date for training
    if pd.isnull(prediction_date_max):
        logging.warning("Prediction mode : no prediction_date_max setted up will take today as so")
        prediction_date_max = datetime.datetime.today()
    date_max = "".join(str(prediction_date_max.date()).split("-"))

    if " WHERE " in query:
        query += " AND "
    else:
        query += " WHERE "

    if objective == "predicting":
        date_min = "".join(
            str((prediction_date_max - pd.Timedelta(int(select_prediction_past_X_months * 30.5), "D")).date()).split(
                "-"
            )
        )
        query += f" DATE_VENTE BETWEEN '{date_min}' AND '{date_max}'"

        logging.info(f"PREDICTING mode : DATE_VENTE BETWEEN '{date_min}' AND '{date_max}'")
    else:
        date_min = "".join(
            str((prediction_date_max - pd.Timedelta(int(12 * select_training_past_X_years * 30.5), "D")).date()).split(
                "-"
            )
        )
        query += f" DATE_VENTE BETWEEN '{date_min}' AND '{date_max}'"

        logging.info(f"TRAINING mode : DATE_VENTE BETWEEN '{date_min}' AND '{date_max}'")

    logging.info(f"FINAL QUERY {query}")
    return query
