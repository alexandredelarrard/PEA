#
# Copyright (c) 2022 by Boston Consulting Group. All rights reserved
#
# -*- coding: utf-8 -*-

import datetime
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from forecast_fl.data_evaluation.evaluate_results import compute_metrics
from forecast_fl.data_evaluation.save_model_results_plots import write_training_results
from forecast_fl.data_models.lgbm_prediction_ub_level import train_lgbm_model_per_ub
from forecast_fl.data_preparation.feature_engineering_ub_model import (
    get_timeseries_features_rolling_7_D,
    ub_feature_engineering,
)
from forecast_fl.data_preparation.top_ubs_to_predict import (
    filter_to_ub_to_predict_directly,
)
from forecast_fl.steps.step import Step
from forecast_fl.utils.config import Config
from forecast_fl.utils.ml_flow import MlFlowITM
from lightgbm.sklearn import LGBMRegressor
from tqdm import tqdm


class TrainingUbModelStep(Step):
    """
    Training UB model step of pipeline. Responsible for training the model given the input data, for the given UB per PDVs or Base.

    Args:
        config_path (str): Config file
        steps_input_data_path (str): Path to the base input folder
        steps_output_data_path (str): Path to the base output folder
        prediction_granularity (str): Training granularity (cluster, méta UB, pdv, etc...)
        prediction_horizon (int): Train model to predict future sales after prediction_horizon days
        (e.g. if prediction_horizon=2 model will be trained to predict sales 2 days in the future)
        specific_pdvs (List): Train models on specific list of PDVs to iterate on, only used when prediction granularity
        is set to PDV
        For BASE and CLUSTER, all PDVs are aggregated
        specific_meta_ubs: Train models on specific list of Meta UB to iterate on
        histo_sales_start_date (datetime.date): Filter earliest date data to train model on
        histo_sales_end_date (datetime.date): Filter lastest date data to train model on

    """

    def __init__(
        self,
        config_path: Config,
        steps_input_data_path: str,
        steps_output_data_path: str,
        prediction_granularity: str,
        prediction_horizon: int,
        specific_pdvs: List,
        specific_meta_ubs: List,
        histo_sales_start_date: datetime.date,
        histo_sales_end_date: datetime.date,
    ):
        super().__init__(
            config_path=config_path,
            steps_input_data_path=steps_input_data_path,
            steps_output_data_path=steps_output_data_path,
        )
        self.config.load["prediction_granularity"] = self.check_prediction_granularity(prediction_granularity)
        self.config.load["prediction_horizon"] = self.check_prediction_horizon(prediction_horizon)
        self.config.load["specific_pdvs"] = self.check_specific_pdvs(specific_pdvs)
        self.config.load["specific_meta_ubs"] = self.check_specific_meta_ubs(specific_meta_ubs)
        self.config.load["histo_sales_start_date"] = self.check_histo_sales_start_date(histo_sales_start_date)
        self.config.load["histo_sales_end_date"] = self.check_histo_sales_end_date(histo_sales_end_date)
        self.config.load["objective"] = "training"

        if self.status.aml_pipeline_run:
            self.ml_flow_itm = MlFlowITM(
                self.config,
                self.steps_output_data_path,
                self.date_to_save,
                name_experience="UB_TRAINING",
            )

        if self.config.load["prediction_granularity"] == "CLUSTER":
            logging.info(
                "CLUSTER Mode is only working for Meta UB training, for UB level training PDV granularity will be used"
            )

    def run(self, df_input: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:  # type: ignore
        """
        Runs training flow of the forecast model UB. Using processed data:
            - Performs feature engineering
            - Creating features (delayed target, averaged target, trends, std...)
            - Looping over each UB code to :

                - Train a specific model
                - Evaluatea  specific model
                - Save a specific model for later use

        Args:
            df_input (pd.DataFrame): Cleaned pandas' data frame with features from DataProcessing

        Return:
            models_results_df:  results of UB model on training set

        """

        # parameters
        models_results_df = pd.DataFrame()
        config_ub_model = self.config.load.parameters_training_ub_model
        target = config_ub_model.config_ub_lgbm["TARGET"]
        self.config.load["histo_sales_end_date"] = min(self.config.load["histo_sales_end_date"], df_input["DATE"].max())

        # filter modeling to relevent UB
        ubs_to_model = filter_to_ub_to_predict_directly(df=df_input, target="MNT_VTE", nbr_months_ub_prop=18)

        # filter to ub to model
        df_input = df_input.loc[df_input["UB_CODE"].isin(ubs_to_model)]

        # create time dependent features
        df_ts_features = get_timeseries_features_rolling_7_D(
            df_input=df_input, config=self.config, prediction_horizon=self.config.load["prediction_horizon"]
        )

        # Iterate modelling for each ub in available meta ub
        for ub_it_nb, ub_n in enumerate(tqdm(ubs_to_model)):

            logging.info(f"TRAINING UB MODEL: Training UB {ub_n}")
            train_df_ub = ub_feature_engineering(df_ts_features, ub_n, config=self.config)

            if train_df_ub.empty:
                logging.info(
                    f"TRAINING UB MODEL: Input dataframe for ub {ub_n} is empty, got shape {train_df_ub.shape}"
                )
            else:
                results_lgbm_ub, model_lgbm_ub = train_lgbm_model_per_ub(
                    X_train_dates_ub=train_df_ub,
                    ub_n=ub_n,
                    config_ub_model=config_ub_model,
                    base_path=self.results_path,
                    split_percentage=self.config.load.parameters_training_model.train_test_proportion,
                )

                if model_lgbm_ub is not None:
                    _, summary = compute_metrics(
                        models_results=results_lgbm_ub,
                        target=target,
                        minimum_sales=1.1,
                        prediction_str=f"PREDICTION_{target}",
                        ub_level=True,
                        meta_ub_level=False,
                    )

                if self.status.aml_pipeline_run:
                    if model_lgbm_ub is not None:

                        self.ml_flow_itm.instanciate_ml_flow()
                        mlflow_run = self.ml_flow_itm.start_run_ml_flow()
                        logging.info(f"MLFlow Active run_id: {mlflow_run.info.run_id}")

                        self.ml_flow_itm.log_metrics(summary)
                        self.ml_flow_itm.tracking_lgbm(train_df_ub, model_lgbm_ub, ub_it_nb, ub_n)
                        local_path = str(self.get_local_artifacts_path(ub_n).absolute())
                        self.ml_flow_itm.ml_flow_artifacts(local_path, ub_it_nb, ub_n)
                        self.ml_flow_itm.end_run_ml_flow()

                nb_sites = train_df_ub["COD_SITE"].nunique()
                self.save_lgbm_model_per_ub(model_lgbm_ub, ub_n, nb_sites)
                self.metrics_handler(models_results_df=models_results_df, config_ub_model=config_ub_model, ub_n=ub_n)

                models_results_df = pd.concat([models_results_df, results_lgbm_ub])

        self.metrics_handler(models_results_df=models_results_df, config_ub_model=config_ub_model, ub_n=None)

        return models_results_df

    def metrics_handler(
        self, models_results_df: pd.DataFrame, config_ub_model: Dict[str, Any], ub_n: Optional[str]
    ) -> None:
        """
        compute metrics over training set and saves results localy or on Azure ML
        Args:
            models_results_df (pd.DataFrame): results data to compute metrics on
        Return:
            None
        """

        # Compute Train metrics
        logging.info("Train metrics")
        models_results, summary_metrics = compute_metrics(
            models_results=models_results_df,
            target=config_ub_model.config_ub_lgbm["TARGET"],
            prediction_str=f"PREDICTION_{config_ub_model.config_ub_lgbm['TARGET']}",
            minimum_sales=1.1,
            ub_level=True,
            meta_ub_level=False,
        )

        # Save Train results in excel & saves model
        logging.info("Save train results")
        self.save_results(
            summary_metrics=summary_metrics,
            models_results=models_results,
            ub_n=ub_n,
        )

    def save_lgbm_model_per_ub(self, model_lgbm_ub: LGBMRegressor, ub_n: str, nb_sites: int) -> None:

        train_end_date = self.config.load["histo_sales_end_date"].strftime("%d-%m-%Y")

        granularity_suffix = "pdvs"
        if self.config.load["prediction_granularity"] == "BASE":
            granularity_suffix = "bases"

        file_name = f"UB_LGBM_trained_on_{nb_sites}_{granularity_suffix}_until_{train_end_date}_ub_{ub_n}_{self.config.load['prediction_horizon']}j.pickle.dat"
        self.models_path.mkdir(parents=True, exist_ok=True)
        path = self.models_path / file_name

        pickle.dump([model_lgbm_ub, self.config], open(str(path.absolute()), "wb"))

    def save_results(
        self,
        ub_n: Optional[str],
        summary_metrics: pd.DataFrame,
        models_results: pd.DataFrame,
    ) -> None:
        """
        Saves summary metric locally or on Azure Blob Storage.
        Logs the model into ML Flow as well # TODO

        Args:
            summary_metrics (pd.DataFrame): DataFrame holding a summary metric
            models_results (pd.DataFrame): DataFrame holding models training results

        Returns:
            Nothing

        """

        logging.debug(f"Writing results metric into {self.steps_output_data_path} basepath")

        path = self.get_local_artifacts_path(ub_n)
        path.mkdir(parents=True, exist_ok=True)

        write_training_results(
            df_list=[summary_metrics, models_results],
            sheet_list=[
                "summary_metrics",
                "models_results",
            ],
            path=path,
            config=self.config,
            date_to_save=self.date_to_save,
        )
        return None

    @classmethod
    def load_output(cls):
        pass

    @property
    def folder_train_results_name(self) -> str:
        horizon = str(self.config.load["prediction_horizon"]) + "j"
        return "_".join(
            [
                self.date_to_save,
                self.config.load.parameters_training_model["model_name"],
                self.config.load["prediction_granularity"],
                horizon,
            ]
        )

    @property
    def results_path(self) -> Path:
        return Path(self.paths_by_directory["ub_train_results"]) / self.folder_train_results_name

    @property
    def models_path(self) -> Path:
        return Path(self.paths_by_directory["ub_model"]) / self.date_to_save

    def get_local_artifacts_path(self, ub_n: Optional[str]) -> Path:
        if ub_n is None:
            path = self.results_path
        else:
            path = self.results_path / ub_n
        return path
