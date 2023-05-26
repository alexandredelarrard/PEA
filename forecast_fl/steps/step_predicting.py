#
# Copyright (c) 2022 by Boston Consulting Group. All rights reserved
#
# -*- coding: utf-8 -*-

import datetime
import logging
import pickle
from typing import Dict, List

import pandas as pd
from forecast_fl.data_models.functions_for_train_or_test import create_tuples_to_model
from forecast_fl.data_prediction.prediction_meta_ub_level import predict_meta_ub_level
from forecast_fl.data_prediction.prediction_ub_level import predict_ub_level
from forecast_fl.data_prediction.prepare_prediction_input import (
    create_prediction_input,
    prepare_prediction_data,
    prepare_rolling_info_prediction_data,
)
from forecast_fl.data_prediction.prepare_prediction_input_ub import (
    create_prediction_input_ub,
)
from forecast_fl.data_preparation.prepare_histo import check_sql_max_available_date
from forecast_fl.steps.step import Step


class PredictingStep(Step):
    """
    Prediction steps using processed data:
        - Prepares prediction input
        - Predicts sales

    Args:
        config_path (str): Config object
        paths_by_directory (Dict): Dictionary holding pathes
        prediction_granularity (str): Granularity of prediction (PDV, BASE, CLUSTER, etc...)
        prediction_date_min (datetime.date) : Minimum date of prediction (included)
        prediction_date_max (datetime.date): Maximum date of prediction (included)
        histo_sales_end_date (datetime.date): Latest date to filter df_input for prediction
        specific_pdvs (List): Specific list of PDVs to iterate on for prediction when prediction granularity is set
        to PDV
        specific_meta_ubs (List): Specific list of Meta UBs to iterate on for prediction

    """

    def __init__(
        self,
        config_path: str,
        steps_input_data_path: str,
        steps_output_data_path: str,
        prediction_granularity: str,
        specific_pdvs: List,
        specific_meta_ubs: List,
        prediction_date_min: datetime.date,
        prediction_date_max: datetime.date,
        histo_sales_end_date: datetime.date,
    ):

        super().__init__(
            config_path=config_path,
            steps_input_data_path=steps_input_data_path,
            steps_output_data_path=steps_output_data_path,
        )
        self.config.load["prediction_granularity"] = self.check_prediction_granularity(prediction_granularity)
        self.config.load["specific_pdvs"] = self.check_specific_pdvs(specific_pdvs)
        self.config.load["specific_meta_ubs"] = self.check_specific_meta_ubs(specific_meta_ubs)
        self.config.load["prediction_date_max"] = self.check_prediction_date_max(prediction_date_max)
        self.config.load["prediction_date_min"] = self.check_prediction_date_min(prediction_date_min)
        self.config.load["histo_sales_end_date"] = self.check_histo_sales_end_date(histo_sales_end_date)
        self.config.load["objective"] = "predicting"

        if self.config.load["prediction_granularity"] == "PDV":
            self.config.load["config_lgbm"] = self.config.load["config_lgbm_pdv"]
        else:
            self.config.load["config_lgbm"] = self.config.load["config_lgbm_base"]

    def run(self, df_input: pd.DataFrame, df_input_cluster: pd.DataFrame, datas: Dict) -> None:
        """
        Predict pipeline for given date in configuration file. This is the main function for prediction. Steps are the following:
            - Creates prediction input using prediction dates to predict for ubs in scope useful for UB and Meta UB predictions
            - Creates features for UB level prediction input
            - Looping over each date to predict UB level sales
            - Creates features for Meta UB level prediction input using:

                - df_input (aggregated at PDV or BASE) if prediction_granularity chosen is either PDV or BASE
                - df_input_cluster (aggregated at CLUSTER) if prediction_granularity chosen is CLUSTER

            - Looping over each date to predict and over each Tuple (location_granularity, meta_ub) to predict Meta UB sales
            - Saves in one pickle file both :

                - predictions_raw (Dict): Meta UB predictions (raw) for each date to predict
                - predictions_ub_raw (Dict): UB predictions (raw) for each date to predict


        Args:
            df_input (pd.DataFrame): processed input at PDV or BASE level from DataProcessing class saved output
            df_input_cluster (pd.DataFrame): processed input at CLUSTER level from DataProcessing class saved output
            datas (Dict): raw datas from DataLoader class saved output

        Returns:
            None

        """

        self.config.load["histo_sales_end_date"] = min(self.config.load["histo_sales_end_date"], df_input["DATE"].max())

        logging.info(
            f"PREDICTION: Starting predictions with parameters : \n"
            + f"histo_sales_end_date = {self.config.load['histo_sales_end_date']} \n"
            + f"prediction_date_min = {self.config.load['prediction_date_min']} \n"
            + f"prediction_date_max = {self.config.load['prediction_date_max']}"
        )

        # Check horizon histo_sales_end_date is available in df_input
        self.config.load["histo_sales_end_date"], horizon = check_sql_max_available_date(
            df_input, self.config.load["histo_sales_end_date"]
        )

        # Filter
        df_input = df_input[df_input["DATE"] <= self.config.load["histo_sales_end_date"]]
        top_ubs = df_input[["COD_SITE", "META_UB", "UB_CODE"]].drop_duplicates()

        # prepare prediction data (use to pass config_prediction.load.prediction_mode["prediction_granularity"],)
        prediction_input = create_prediction_input(df_input, config=self.config)

        # add weather & holidays infos on dates to predict
        logging.info("PREDICTION: Creating prediction inputs")
        prediction_input = prepare_prediction_data(prediction_input, datas)

        if self.config.load["prediction_granularity"] == "CLUSTER":
            logging.info("PREDICTION: Creating prediction inputs at CLUSTER level")
            df_input_cluster = df_input_cluster[df_input_cluster["DATE"] <= self.config.load["histo_sales_end_date"]]

            prediction_input_cluster = create_prediction_input(df_input_cluster, config=self.config)
            prediction_input_cluster = prepare_prediction_data(prediction_input_cluster, datas)

        ##################### UB prediction
        if eval(f"self.config.load.prediction_mode.{self.config.load['prediction_granularity']}.w_model_ub") > 0:
            logging.info("PREDICTION: Creating prediction inputs for ub model")
            prediction_input_ub = create_prediction_input_ub(
                prediction_input=prediction_input,
                train_df=df_input,
                config=self.config,
            )

            logging.info("PREDICTION: Prediction UB level")
            predictions_ub_raw = predict_ub_level(
                prediction_input_ub=prediction_input_ub,
                top_ubs=top_ubs,
                config=self.config,
                paths_by_directory=self.paths_by_directory,
            )
        else:
            predictions_ub_raw = {}

        ##################### META UB prediction
        # prepare rolling features for prediction
        if eval(f"self.config.load.prediction_mode.{self.config.load['prediction_granularity']}.w_model_meta_ub") > 0:
            logging.info("PREDICTION: Creating prediction inputs for META_ub model")
            if self.config.load["prediction_granularity"] == "CLUSTER":
                prediction_input_meta_ub = prepare_rolling_info_prediction_data(
                    prediction_input_cluster,
                    df_input_cluster,
                    config=self.config,
                )
                tuple_meta_ub_granularity = create_tuples_to_model(
                    df=df_input_cluster,
                    config=self.config,
                )
            else:
                prediction_input_meta_ub = prepare_rolling_info_prediction_data(
                    prediction_input,
                    df_input,
                    config=self.config,
                )
                # create Meta ub / granularity pairs for modeling
                tuple_meta_ub_granularity = create_tuples_to_model(
                    df=df_input,
                    config=self.config,
                )

            logging.info("PREDICTION: Prediction META UB level")
            predictions_meta_ub_raw = predict_meta_ub_level(
                prediction_input=prediction_input_meta_ub,
                config=self.config,
                paths_by_directory=self.paths_by_directory,
                tuple_meta_ub_granularity=tuple_meta_ub_granularity,
            )
        else:
            predictions_meta_ub_raw = {}

        logging.info("SAVING: saving predictions")
        self.save_output(predictions_meta_ub_raw, predictions_ub_raw)

    def save_output(self, predictions_meta_ub_raw: pd.DataFrame, predictions_ub_raw: pd.DataFrame):
        """
        Saves predictions made at Meta UB and UB level to be processed in the postprocessing step.

        Args:
            predictions_meta_ub_raw (pd.DataFrame) lgbm meta ub model output predictions for dates chosen
            predictions_ub_raw (pd.DataFrame) : lgbm ub model output predictions for dates chosen

        Returns:
            None

        """

        logging.info("Saving predictions")
        saving_path = "/".join(
            [self.paths_by_directory["intermediate"], f"{self.aml_run_id}_raw_predictions_meta_ub_and_ub.pickle.dat"]
        )
        pickle.dump(
            [predictions_meta_ub_raw, predictions_ub_raw],
            open(saving_path, "wb"),
        )

    @classmethod
    def load_output(cls, steps_input_data_path, aml_run_id):
        """
        Class method to load data saved when using the class PredictingStep

        Args:
            root_path_save (str): path to overall saving directory
            aml_run_id (str): run id useful for aml pipeline to mutualize files for each job of the pipeline, when
            running in local this parameter is an empty string

        Returns:
            prediction_data (DataFrame): containing all predictions made at UB and Meta UB level

        """

        return pickle.load(
            open(
                "/".join([steps_input_data_path, f"{aml_run_id}_raw_predictions_meta_ub_and_ub.pickle.dat"]),
                "rb",
            )
        )
