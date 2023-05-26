#
# Copyright (c) 2022 by Boston Consulting Group. All rights reserved
#
# -*- coding: utf-8 -*-

import gc
import logging
import pickle

import forecast_fl.data_preparation.prepare_histo as prep
import pandas as pd
from forecast_fl.data_preparation.price_retreatement import (
    add_price_variations,
    add_prix_hebdomadaire,
)
from forecast_fl.data_preparation.top_ubs_to_predict import (
    get_top_ub,
    keep_significant_meta_ub_per_pdv,
    keep_ubs,
)
from forecast_fl.steps.step import Step


class DataProcessing(Step):
    """
    Data Processing step of pipeline. Reponsible for cleaning the data and creating the cube
    used either for prediction or for training

    Args:
        config_path (str): Config file
        input_data_path (str): Path to the base input folder
        steps_output_data_path (str): Path to the base output folder
        datas (Dict): contains all necessary raw data read in the previous step,
        prediction_granularity (str): prediction granularity used to aggregate data at right level (PDV, CLUSTER OR BASE),

    """

    def __init__(
        self,
        config_path: str,
        steps_input_data_path: str,
        steps_output_data_path: str,
        prediction_granularity: str,
        objective: str,
    ):
        super().__init__(
            config_path=config_path,
            steps_input_data_path=steps_input_data_path,
            steps_output_data_path=steps_output_data_path,
        )
        self.config.load["prediction_granularity"] = self.check_prediction_granularity(prediction_granularity)
        self.config.load["objective"] = self.check_objective(objective)

    def run(self, datas):
        """
        Creates and clean raw data to compute our input data following the steps in the class function
        preprocessing_data

        Args:
            datas (Dict): raw input data saved by the class DataLoader

        Returns:
            df_input (DataFrame): clean data used to for training or prediction

        """

        logging.info("Cleaning data")

        df_input = self.preprocessing_data(datas)

        # save intermediate results
        self.save_output(df_input)

        if self.config.load["prediction_granularity"] == "CLUSTER":

            del df_input
            gc.collect()

            # create cluster shaped data input
            df_input_cluster = self.preprocessing_data_cluster(datas)

            # save cluster intermediate results
            self.save_output(df_input_cluster, cluster_mode="CLUSTER")

    def preprocessing_data(self, datas):
        """
        Main function for data cleaning. Cleaning and merging of various data sources to obtain model dataframe input. Steps detailed below and in comments:
            -Removing unrealistic data :
                - filtering out negative or null sales
                - filtering on UBs mapped

            - Mapping UBs to Meta UB
            - Filtering meaningful UBs in each Meta UB making more than x% (using the parameter top_ub_min_sales_proportion in the config) of sales in each Meta UB.
            - If location granularity is set to BASE: Agregation of sales data from stores of each Base
            - Filling missing days for each UB with null sales data
            - Adding other Meta UB used information (weight and price)
            - Processing of price using Warehouse recommended prices
            - Adding stores characteristics (# cash register, ...)
            - Adding meteo data
            - Adding national and school holidays
            - Computing the TARGET for our training
            - If location granularity is set to CLUSTER: Agregation of sales data from stores over each CLUSTER depending on their vocation (mapping is available in the config: clustering_pdvs_types)

        Args:
            datas (dict): Dictionnary with datas, initiated in load_data
            config (Dict): Config objects

        Returns:
            Cleaned dataframe for training and prediction
            Cleaned dataframe for training and prediction aggregated at CLUSTER level

        """

        raw_data = datas["HISTO_SQL"]
        meteo = datas["METEO"]
        pdv_info = datas["PDV_INFO_SQL"]
        mapping = datas["UB_MAPPING"]
        commandes_pdv = datas["COMMANDES_PDV"]

        # clean raw_data
        raw_data = prep._remove_unrealistic_data(raw_data, mapping)

        # Add META_UB mapping
        logging.info("CLEAN: Add Meta ub and remove UB not in mapping")
        raw_data = prep._clean_redundant_ub(raw_data, mapping)
        logging.info(f"CLEAN: {len(raw_data['META_UB'].unique())} META_UB in raw data")

        if self.config.load["prediction_granularity"] == "PDV":
            logging.info("[PDV] Meta_UB filtering: keep only relevant Meta_UB per PdV")
            raw_data = keep_significant_meta_ub_per_pdv(
                df=raw_data, config_prediction=self.config.load.prediction_mode, target="MNT_VTE"
            )

        # Filter to meaningful UB only
        if not self.config.load.prediction_mode.ub_only_in_command_base:
            nbr_ubs = len(raw_data["UB_CODE"].unique())
            nbr_pdvs = len(raw_data["COD_SITE"].unique())
            logging.info(f"CLEAN: Filter to meaningful UB from {nbr_ubs} / NBR PDV = {nbr_pdvs}")
            top_ubs = get_top_ub(df=raw_data, config_prediction=self.config.load.prediction_mode, target="MNT_VTE")
            raw_data = keep_ubs(raw_data, top_ubs)
            logging.info(f"CLEAN: keeping {len(top_ubs['UB_CODE'].unique())}/{nbr_ubs} UB overall")

        else:
            raw_data = raw_data.loc[raw_data["IN_COMMANDE_PDV"] == 1]
            nbr_ubs = len(raw_data["UB_CODE"].unique())
            nbr_pdvs = len(raw_data["COD_SITE"].unique())
            raw_data = raw_data.drop(["META_UB", "IN_COMMANDE_PDV"], axis=1)
            logging.info(f"CLEAN: KEEP meaningful UB from COMMANDE PDV {nbr_ubs} / NBR PDV = {nbr_pdvs}")

        # Fill missing days
        logging.info(f"CLEAN: Fill missing days for aggregated {self.config.load['prediction_granularity']}")
        raw_data = prep._fill_missing_days(raw_data)

        # merge to have all ub infos
        raw_data = prep._merge_ub_mapping(raw_data, mapping)

        # Adding prix_hebdomadaire
        logging.info("CLEAN: Add Prix Hebdo")
        raw_data = add_prix_hebdomadaire(raw_data=raw_data, commandes_pdv=commandes_pdv)

        # Check if aggregation at Base level is needed
        if self.config.load["prediction_granularity"] == "BASE":
            logging.info("[BASE] AGGREGATION: Aggregate data at BASE level")
            raw_data = prep.pre_aggregation_base(raw_data, pdv_info)

        # Adding price_variations
        logging.info("CLEAN: Add Prix variation")
        raw_data = add_price_variations(raw_data=raw_data)

        # Merge PDV Information
        logging.info("CLEAN: Merge PDV INFO")
        raw_data = raw_data.merge(pdv_info, how="left", on="COD_SITE")

        # Add meteo data
        logging.info("CLEAN: Add meteo")
        meteo = prep.clean_meteo_data(meteo)
        raw_data = prep._enrich_histo_data(raw_data, meteo)

        # Add holidays
        logging.info("CLEAN: Add vacances")
        raw_data = prep._vacances_scolaires(raw_data)

        # Add special dates
        logging.info("CLEAN: Add jours feries")
        raw_data = prep._jours_feriers(raw_data)

        # Create target
        raw_data = prep.create_target_for_prediction(raw_data=raw_data)

        logging.info(f"CLEANED: Raw dataframe shape = {raw_data.shape}")
        logging.info(
            f"{raw_data[['META_UB', 'COD_SITE']].drop_duplicates().shape} pairs META_UB X PDV in total to model"
        )

        return raw_data

    def preprocessing_data_cluster(self, datas):

        # load clustering mapping data
        clustering = datas["PDV_CLUSTERING"]
        raw_data = self.load_output(
            steps_input_data_path=self.paths_by_directory["intermediate"], aml_run_id=self.aml_run_id
        )

        # Add cluster information
        clustering["COD_SITE"] = clustering["COD_SITE"].apply(lambda x: str(x))
        logging.info(f"[CLUSTER] CLEANED: Mapping PdV x Cluster contains {len(clustering['COD_SITE'].unique())} PdVs")
        raw_data_cluster = raw_data.merge(clustering, on="COD_SITE", how="left", validate="m:1")

        logging.info("[CLUSTER] AGGREGATION: Aggregate data at CLUSTER level")
        raw_data_cluster = prep.pre_aggregation_cluster(raw_data_cluster)
        logging.critical(f"[CLUSTER] CLEANED: Raw dataframe shape = {raw_data.shape}")

        return raw_data_cluster

    def save_output(self, df_input: pd.DataFrame, cluster_mode="") -> None:
        """
        Saves processed data cube for later use in training and prediction.
        When prediction granularity is set to CLUSTER training and prediction need both processed data at
        CLUSTER level and PDV level.
        When prediction granularity is set to BASE or PDV, processed data is saved at PDV/CLUSTER level

        Args:
            df_input: pd.DataFrame
            cluster_mode: additional mode to be able to save the processed data cube at cluster level

        Returns:
            None

        """

        cluster_str = ""
        if cluster_mode == "CLUSTER":
            cluster_str = "_cluster"

        logging.info("Saving processed data")
        save_path = "/".join(
            [
                self.paths_by_directory["intermediate"],
                f"{self.aml_run_id}_processed_data_{self.config.load['objective']}{cluster_str}.pickle.dat",
            ]
        )

        pickle.dump(df_input, open(save_path, "wb"))

    @classmethod
    def load_output(cls, steps_input_data_path, aml_run_id, objective="training", cluster_mode=""):

        """
        Class method to load data saved when using the class DataProcessing

        Args:
            root_path_save (str): path to overall saving directory
            aml_run_id (str): run id useful for aml pipeline to mutualize files for each job of the pipeline, when
            running in local this parameter is an empty string
            objective (str): to laod either predict or train processed data as processing steps depend on the mode chosen
            cluster_mode (str): to laod cluster processed data

        Return:
            raw_data (DataFrame): containing all needed dataframes for algorithm training or prediction mode

        """
        cluster_str = ""
        if cluster_mode == "CLUSTER":
            cluster_str = "_cluster"

        load_path = "/".join(
            [steps_input_data_path, f"{aml_run_id}_processed_data_{objective}{cluster_str}.pickle.dat"]
        )
        return pickle.load(open(load_path, "rb"))
