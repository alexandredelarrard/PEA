#
# Copyright (c) 2023 by Boston Consulting Group. All rights reserved
#
# -*- coding: utf-8 -*-

import hashlib
import logging

import numpy as np
import pandas as pd
from box import Box
from forecast_fl.data_exporter.utils_database_sql import (
    add_columns_in_table_sql,
    create_table_preco,
)
from forecast_fl.postgres_db.setup_db import load_database


class PreconisationSQLExporter:
    """
    Class exporting preconisations into database
    It is responsible to check the format of the exported table into the table, and includes the following
    functionalities:
    1) Data export
    2) Data checks
    3) Table update (delta mode)

    With some log monitoring

    Args:
        XXX

    """

    def __init__(self, config: Box):
        self.config = config
        self.table_path = f"{self.config.database.table_path.output_preconisations.table_name}"
        self.database = load_database()
        self.primary_keys = ["COD_SITE", "CODE_UB", "DATE_HISTORIQUE_MAX", "DATE_PREDICTION"]
        self.id_column = "MACRO_UB_CODE"

        self.table_schema = [
            {"name": "MACRO_UB_CODE", "type": str},
            {"name": "DATE_HISTORIQUE_MAX", "type": pd.datetime},
            {"name": "DATE_PREDICTION", "type": pd.datetime},
            {"name": "DATE_EXECUTION", "type": pd.datetime},
            {"name": "PREDICTION_HORIZON", "type": int},
            {"name": "MACRO_UB_NOM", "type": str},
            {"name": "COD_SITE", "type": str},
            {"name": "CODE_UB", "type": str},
            {"name": "NOM_UB", "type": str},
            {"name": "UNITE_UB", "type": str},
            {"name": "POIDS_STOCK_J-1", "type": int},
            {"name": "QTE_PRECO_COMMANDES", "type": int},
            {"name": "QTE_PRECO_COMMANDES_POIDS", "type": int},
            {"name": "QTE_VTE", "type": int},
            {"name": "QTE_VTE_POIDS", "type": int},
            {"name": "PREDICTION_FROM_META_UB_VENTILATION", "type": float},
            {"name": "PREDICTION_FROM_DIRECT_UB", "type": float},
            {"name": "QTE_VTE_POIDS_OBSERVE", "type": int},
            {"name": "QTE_VTE_OBSERVE", "type": int},
            {"name": "QTE_VTE_POIDS_MACRO_UB", "type": int},
            {"name": "QTE_VTE_J-2", "type": int},
            {"name": "QTE_VTE_J-3", "type": int},
            {"name": "QTE_VTE_J-4", "type": int},
            {"name": "QTE_VTE_J-5", "type": int},
            {"name": "QTE_VTE_J-6", "type": int},
            {"name": "QTE_VTE_J-7", "type": int},
            {"name": "QTE_VTE_J-8", "type": int},
            {"name": "QTE_VTE_J-9", "type": int},
            {"name": "QTE_VTE_J-10", "type": int},
            {"name": "QTE_VTE_J-28", "type": int},
            {"name": "FERIER", "type": str},
            {"name": "RATIO_NOT_DIRECT", "type": float},
        ]

        create_table_preco(self.database)
        add_columns_in_table_sql(self.database, table_name="preco", list_columns=self.table_schema)

    def export(self, df_preconisations_for_export: pd.DataFrame, df_input: pd.DataFrame) -> None:
        """
        Exports the table into the database.

        Args:
            df_preconisations_for_export (pd.DataFrame): Post-processed dataframe holding preconisations, to export

        Returns:

        """

        cols = [dic["name"] for dic in self.table_schema]
        df_preconisations_for_export = df_preconisations_for_export[cols]

        # create unique key per obs
        logging.info("CREATE PRIMIRY KEY ID")
        df_preconisations_for_export = self._create_primary_key(df_preconisations_for_export)

        # delete redundant records
        logging.info(f"Cleaning records based on primary key {', '.join(self.primary_keys)}")
        self._delete_existing_records_in_table(df_preconisations_for_export)

        # save into sql
        logging.info(f"WRITING IN SQL table {self.table_path}")
        df_preconisations_for_export.to_sql(
            self.table_path, con=self.database.engine, if_exists="append", chunksize=1000, index=False
        )

        # fill in missing past real target infos
        precos_with_target = self._select_past_records_to_complete_with_target(df_input)
        logging.info(f"WRITING IN SQL table {self.table_path} observed TARGET for past values to monitor predictions")
        self._export_past_targets_to_preco(precos_with_target)

        return df_preconisations_for_export

    def _create_primary_key(self, df_preconisations_for_export):
        id_df = df_preconisations_for_export[self.primary_keys].copy().astype(str)
        id_df["KEY"] = id_df.sum(axis=1)
        df_preconisations_for_export[self.id_column] = id_df["KEY"].apply(
            lambda x: hashlib.sha1(x.encode("utf-8")).hexdigest()
        )
        return df_preconisations_for_export

    def _delete_existing_records_in_table(self, df_preconisations_for_export: pd.DataFrame) -> None:

        min_date_predict = df_preconisations_for_export["DATE_PREDICTION"].min().strftime("%Y-%m-%d")
        precos = pd.read_sql(
            f"SELECT * FROM {self.table_path} WHERE DATE_PREDICTION >= '{min_date_predict}'", con=self.database.engine
        )

        id_already_in_sql = list(
            set(df_preconisations_for_export[self.id_column]).intersection(set(precos[self.id_column]))
        )
        if len(id_already_in_sql) > 0:
            logging.info(f"REMOVING {len(id_already_in_sql)} observations from table SQL {self.table_path}")
        else:
            logging.info("NO RECORDS TO REMOVE")

        # Segment records_id into chunks of 1000 elements from records_id
        chunk_size = int(len(id_already_in_sql) / 1000)
        if chunk_size > 0:
            chunked_records_id = np.array_split(id_already_in_sql, chunk_size)
        elif len(id_already_in_sql) > 0:
            chunked_records_id = [id_already_in_sql]
        else:
            chunked_records_id = []

        for chunked_record in chunked_records_id:
            query = f"DELETE FROM {self.table_path} WHERE {self.id_column} in {tuple(chunked_record)}"

            logging.debug(f"Query to update records : {query}")
            with self.database.engine.connect() as connection:
                connection.execute(query)

    def _select_past_records_to_complete_with_target(self, df_input: pd.DataFrame) -> pd.DataFrame:

        sub_input = df_input[["DATE", "COD_SITE", "UB_CODE", "TARGET", "POIDS_UC_MAPPING", "TYPE_UB"]].rename(
            columns={"DATE": "DATE_PREDICTION", "UB_CODE": "CODE_UB", "TARGET": "QTE_VTE_POIDS_OBSERVE"}
        )

        logging.info("Loading preco records with no Target Value")
        precos_with_target = pd.read_sql(
            f"SELECT DATE_PREDICTION, COD_SITE,CODE_UB FROM {self.table_path} WHERE QTE_VTE_POIDS_OBSERVE IS NULL "
            f"AND DATE_PREDICTION <= '{df_input.DATE.max().strftime('%Y-%m-%d')}'",
            con=self.database.engine,
        )

        logging.info("Adding Target for records in preco table")
        precos_with_target["DATE_PREDICTION"] = pd.to_datetime(precos_with_target["DATE_PREDICTION"], format="%Y-%m-%d")
        precos_with_target = precos_with_target.drop_duplicates().merge(
            sub_input,
            on=["DATE_PREDICTION", "COD_SITE", "CODE_UB"],
            how="left",
            validate="1:1",
        )

        precos_with_target = precos_with_target[precos_with_target["QTE_VTE_POIDS_OBSERVE"].notna()]
        precos_with_target["QTE_VTE_POIDS_OBSERVE"] = precos_with_target["QTE_VTE_POIDS_OBSERVE"].astype(int)
        precos_with_target["QTE_VTE_OBSERVE"] = np.where(
            precos_with_target["TYPE_UB"] == "PIECE",
            precos_with_target["QTE_VTE_POIDS_OBSERVE"] / precos_with_target["POIDS_UC_MAPPING"],
            precos_with_target["QTE_VTE_POIDS_OBSERVE"],
        )

        precos_with_target["QTE_VTE_OBSERVE"] = precos_with_target["QTE_VTE_OBSERVE"].astype(float).round(0)

        return precos_with_target

    def _export_past_targets_to_preco(self, precos_with_target: pd.DataFrame) -> None:
        records_to_update = list(precos_with_target.itertuples(index=False))
        with self.database.engine.connect() as connection:
            for record_to_update in records_to_update:
                query = (
                    f"UPDATE {self.table_path} SET QTE_VTE_POIDS_OBSERVE={record_to_update.QTE_VTE_POIDS_OBSERVE}, "
                    f"QTE_VTE_OBSERVE={record_to_update.QTE_VTE_OBSERVE} "
                    f"WHERE DATE_PREDICTION='{record_to_update.DATE_PREDICTION.strftime('%Y-%m-%d')}' AND "
                    f"CODE_UB='{record_to_update.CODE_UB}' AND "
                    f"COD_SITE='{record_to_update.COD_SITE}' "
                )

                connection.execute(query)
