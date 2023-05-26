#
# Copyright (c) 2022 by Boston Consulting Group. All rights reserved
#
# -*- coding: utf-8 -*-
import datetime
import logging
import os
import pickle

import forecast_fl.load_data.loading_data as ld
import numpy as np
import pandas as pd
from forecast_fl.postgres_db.setup_db import load_database
from forecast_fl.steps.step import Step


class DataLoader(Step):
    """
    Data Ingestion step of pipeline. Reponsible for reading all necessary data
    for the indicated PDVs & Méta UB depending on modes.
    Load the cube that will be cleaned in the next step.

    Args:
        config_path (str): Config path
        input_data_path (str): Path to the base input folder
        steps_output_data_path (str): Path to the base output folder
        specific_pdvs (list): List of PDVs to filter in the query,
        specific_meta_ubs (list): List of meta ubs to filter in the query,
        prediction_granularity (str): BASE/CLUSTER/PDV

    """

    def __init__(
        self,
        config_path: str,
        input_data_path: str,
        steps_output_data_path: str,
        specific_pdvs: list,
        specific_meta_ubs: list,
        prediction_granularity: str,
        objective: str = "training",
        prediction_date_max: datetime.date = None,
    ):
        super().__init__(config_path=config_path, steps_output_data_path=steps_output_data_path)

        # only for loading static files from input data path
        # different from steps_input_data_path
        if self.aml_run_id == "":
            input_data_path = os.getenv("BASE_PATH", input_data_path)

        self.input_data_path = input_data_path
        self.config.load["BASE_PATH"] = self.input_data_path

        self.config.load["specific_pdvs"] = self.check_specific_pdvs(specific_pdvs)
        self.config.load["specific_meta_ubs"] = self.check_specific_meta_ubs(specific_meta_ubs)
        self.config.load["prediction_granularity"] = self.check_prediction_granularity(prediction_granularity)
        self.config.load["objective"] = self.check_objective(objective)
        self.config.load["prediction_date_max"] = self.check_prediction_date_max(prediction_date_max)

    def run(self):
        """
        Runs loading flow of all necessary data to create needed to create the input of the model.

        Returns:
            datas (dict): containing all needed dataframes for algorithm training or prediction mode
            key: data information type (str)
            value: data (pd.DataFrame)

        """

        logging.info("Loading data")

        # Load database SQLServer connection
        self.database = load_database()

        # load all datas
        datas = self.data_loading()

        # save all datas
        self.save_output(datas)

        return datas

    def data_loading(self):
        """
        Load data needed to perform training and predictions, loading will differ based on parameters enterd :
            - prediction granularity:

                - BASE : No filter is applied to stores (PDVs) thus loading data for all available stores in the sales db
                - CLUSTER : Filtering on the Bases chosen in the config and using all available stores for each Base
                - PDV : Filtering on the list of PDVs entered if not empty, else loading data for all available stores in the sales db

            - specific_meta_ubs: If not empty filters on UBs matching Meta UBs specified, else loading data for all UBs available in the sales db
            - objective: if the objective is "predict", only loads 18 months history instead of 4 years history on sales data

        Returns:
            datas (dict) : containing all needed dataframes for algorithm training or prediction mode. Consists in a dictionary containing :
                - historical sales per date, point de vente, unité de besoin
                - sous famille information
                - point de ventes characteristics (m2, # caisses)
                - unité de besoin information (name)
                - historical meteorological data and forecast
                - custom mapping UB (UB_CODE) to meta UB (banane bio, tomate ronde)
                - clustering mapping assigning a PdV to a Clustering (for now based on PdV VOCATION)
                - historical ordering from BASE useful to compute our BASE order recommendation
                - historical stock from BASE useful to compute our BASE order recommendation
                - historical ordering from Stores (PDVs) useful to compute our PDV order recommendation

        """

        if self.status.offline:
            logging.info(f"Data inputs found from {self.input_data_path}")
            logging.info(f"Available files : {len(os.listdir(self.input_data_path))}")

        # load data
        datas = {}

        ################## MAPPING file manually created
        datas["UB_MAPPING"] = ld.load_meta_ub_mapping(self.config)
        datas["UB_MAPPING"]["UB_CODE"] = datas["UB_MAPPING"]["UB_CODE"].astype(int).astype(str).str.zfill(8)

        # different mapping for BASE and PDV, since much fewer sell per PDV
        if self.config.load["prediction_granularity"] == "PDV":
            if "META_UB_FOR_PDV" in datas["UB_MAPPING"].columns:
                datas["UB_MAPPING"]["META_UB"] = datas["UB_MAPPING"]["META_UB_FOR_PDV"]
                logging.info(f"{len(datas['UB_MAPPING']['META_UB'].unique())} META_UB available for prediction")
                logging.info(f"{len(datas['UB_MAPPING']['UB_CODE'].unique())} UB available in total for prediction")
            else:
                logging.critical("no columns META_UB_FOR_PDV in mapping, please check version")

        ################## PDV_INFO_SQL QUERY
        with open(
            os.path.join(self.config.queries.query_directory, self.config.queries.pdv_info),
            "r",
        ) as query:
            query_str = query.read()
            logging.info("Query PDV_INFO_SQL started")
            datas["PDV_INFO_SQL"] = pd.read_sql(query_str, con=self.database.engine)
        datas["PDV_INFO_SQL"]["CODE_BASE"] = datas["PDV_INFO_SQL"]["CODE_BASE"].astype(int).astype(str).str.zfill(3)

        ##################  PDV_CLUSTERING INFO
        datas["PDV_CLUSTERING"] = ld.generate_pdv_clustering(config=self.config, datas=datas)

        with open(
            os.path.join(self.config.queries.query_directory, self.config.queries.historical_pdv),
            "r",
        ) as query:

            query_str = ld.get_sales_histo_query(
                query.read(),
                datas["PDV_CLUSTERING"],
                datas["UB_MAPPING"],
                config=self.config,
            )

            logging.info("Query HISTO SQL started")
            datas["HISTO_SQL"] = pd.read_sql(query_str, con=self.database.engine)

        ################## requete SQL pour extraire les familles / ub liés à la liste de meta ub
        with open(
            os.path.join(self.config.queries.query_directory, self.config.queries.sous_famille),
            "r",
        ) as query:
            query_str = query.read()
            logging.info("Query SOUS_FAMILLE_SQL started")
            datas["SOUS_FAMILLE_SQL"] = pd.read_sql(query_str, con=self.database.engine)

        ################## requete SQL pour extraire les infos des produits standards
        with open(
            os.path.join(self.config.queries.query_directory, self.config.queries.produit_std_info),
            "r",
        ) as query:
            query_str = query.read()
            logging.info("Query PRODUIT_STD_INFO started")
            datas["PRODUIT_STD_INFO"] = pd.read_sql(query_str, con=self.database.engine)

        if self.config.load["objective"] == "predicting":

            if pd.isnull(self.config.load["prediction_date_max"]):
                logging.warning(
                    "Prediction mode : no prediction_date_max setted up will take most recent date in historical data"
                )
                prediction_date_max = datas["HISTO_SQL"].DATE.max()
                date_min = prediction_date_max - pd.Timedelta(40, "D")
                date_max = prediction_date_max
            else:
                date_min = (self.config.load["prediction_date_max"] - pd.Timedelta(40, "D")).date()
                date_max = self.config.load["prediction_date_max"].date()

            date_condition = f"BETWEEN '{date_min.strftime('%Y-%m-%d')}' AND '{date_max.strftime('%Y-%m-%d')}' "

            logging.info(f"EXTRACTING STOCKS AND ORDERS WITH DATE BETWEEN '{date_min}' AND '{date_max}'")

            ###################"" SQL request to extract base stock
            with open(
                os.path.join(self.config.queries.query_directory, self.config.queries.stock_base),
                "r",
            ) as query:
                query_str = query.read()
                query_str = query_str + " WHERE DATE_STOCK " + date_condition
                logging.info("Query STOCK_BASE started")
                datas["STOCK_BASE"] = pd.read_sql(query_str, con=self.database.engine)
            datas["STOCK_BASE"]["PRODUIT_STD_CODE"] = (
                datas["STOCK_BASE"]["PRODUIT_STD_CODE"].astype(int).astype(str).str.zfill(8)
            )

            ###################"" SQL request to extract pdv stock
            with open(
                os.path.join(self.config.queries.query_directory, self.config.queries.stock_pdv),
                "r",
            ) as query:
                query_str = query.read()
                query_str = query_str + " WHERE DATE_INV " + date_condition
                logging.info("Query STOCK_PDV started")
                datas["STOCK_PDV"] = pd.read_sql(query_str, con=self.database.engine)
            datas["STOCK_PDV"]["UB_CODE"] = datas["STOCK_PDV"]["UB_CODE"].fillna(0).astype(int).astype(str).str.zfill(8)

            ##################### SQL request to extract base orders
            with open(
                os.path.join(self.config.queries.query_directory, self.config.queries.commandes_base),
                "r",
            ) as query:
                query_str = query.read()
                query_str += "WHERE DATE " + date_condition

                logging.info("Query COMMANDES_BASE started")
                datas["COMMANDES_BASE"] = pd.read_sql(query_str, con=self.database.engine)

            datas["COMMANDES_BASE"]["PRODUIT_STD_CODE"] = (
                datas["COMMANDES_BASE"]["PRODUIT_STD_CODE"].astype(int).astype(str).str.zfill(8)
            )

            ###################"" SQL request to extract familles
            with open(
                os.path.join(self.config.queries.query_directory, self.config.queries.rate_familles),
                "r",
            ) as query:
                query_str = query.read()
                datas["RATE_FAMILLES"] = pd.read_sql(query_str, con=self.database.engine)

        else:
            if pd.isnull(self.config.load["prediction_date_max"]):
                date_max = datas["HISTO_SQL"].DATE.max().strftime("%Y-%m-%d")
            else:
                date_max = self.config.load["prediction_date_max"].strftime("%Y-%m-%d")
            date_condition = f" <= '{date_max}'"

        ##################### SQL request to extract stores orders
        with open(
            os.path.join(self.config.queries.query_directory, self.config.queries.commandes_pdv),
            "r",
        ) as query:
            query_str = query.read()
            query_str = query_str + " WHERE DATE " + date_condition

            ub_to_query = ub_to_query = list(
                set(
                    datas["UB_MAPPING"]["UB_CODE"].tolist()
                    + datas["UB_MAPPING"]
                    .loc[~datas["UB_MAPPING"]["REPLACE_UB"].isin([np.nan, "OK"]), "REPLACE_UB"]
                    .tolist()
                )
            )
            ubs = ""
            for ub in ub_to_query:
                ubs += f"{ub},"

            query_str += f"AND CODE_PRGE IN ({ubs[:-1]})"
            logging.info("Query COMMANDES_PDV started")
            datas["COMMANDES_PDV"] = pd.read_sql(query_str, con=self.database.engine)

        datas["COMMANDES_PDV"]["PRODUIT_STD_CODE"] = (
            datas["COMMANDES_PDV"]["PRODUIT_STD_CODE"].astype(int).astype(str).str.zfill(8)
        )

        ##################### METEO
        logging.info("Query METEO started")
        datas["METEO"] = pd.read_sql("meteo", con=self.database.engine)

        logging.info("Data Loading finished")

        return datas

    def save_output(self, datas) -> None:
        """
        Saves data cube for later use in other steps in a pickle file.

        Args:
            datas (dict) : containing all needed dataframes for algorithm training or prediction mode

        Returns:
            None

        """
        save_path = "/".join(
            [
                self.paths_by_directory["intermediate"],
                f"{self.aml_run_id}_raw_datas_{self.config.load['objective']}.pickle.dat",
            ]
        )
        pickle.dump(datas, open(save_path, "wb"))

    @classmethod
    def load_output(cls, steps_input_data_path, objective, aml_run_id):
        """
        Class method to load data saved when using the class DataLoader

        Args:
            output_data_path (str): path to overall saving directory
            objective (str): to laod either predict or train raw_data as loading steps depend on the mode chosen
            aml_run_id (str): run id useful for aml pipeline to mutualize files for each job of the pipeline, when
        running in local this parameter is an empty string

        Returns:
            raw_data (DataFrame): containing all needed dataframes for algorithm training or prediction mode

        """

        load_path = "/".join([steps_input_data_path, f"{aml_run_id}_raw_datas_{objective}.pickle.dat"])
        return pickle.load(open(load_path, "rb"))
