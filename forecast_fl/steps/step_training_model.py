#
# Copyright (c) 2022 by Boston Consulting Group. All rights reserved
#
# -*- coding: utf-8 -*-

import datetime
import logging
import pickle
from pathlib import Path
from typing import List, Optional

import pandas as pd
from forecast_fl.data_evaluation.evaluate_results import (
    compute_metrics,
    reformat_results,
)
from forecast_fl.data_evaluation.save_model_results_plots import write_training_results
from forecast_fl.data_models.functions_for_train_or_test import (
    backtest_meta_ub_to_ub_level,
    create_tuples_to_model,
    model_single_ub_granularity,
)
from forecast_fl.data_preparation.feature_engineering import (
    aggregate_features_meta_ub_level,
    feature_engineering_meta_ub_cod_site,
    filtering_dates_and_retrieving_holidays,
)
from forecast_fl.steps.step import Step
from forecast_fl.utils.config import Config
from forecast_fl.utils.ml_flow import MlFlowITM
from tqdm import tqdm


class TrainingModelStep(Step):
    """
    Training model step of pipeline. Responsible for training the model given the input data, for the given
    Meta UB x PDVs or cluster or Base, depending on the mode.

    Args:
        config (config class): Config file
        input_data_path (str): Path to the base input folder
        steps_output_data_path (str): Path to the base output folder
        paths_by_directory (Dict): Dict of relative folder paths
        prediction_granularity (str): Training granularity (cluster, méta UB, pdv, etc...)
        prediction_horizon (int): Train model to predict future sales after prediction_horizon days
        (e.g. if prediction_horizon=2 model will be trained to predict sales 2 days in the future)
        steps_output_data_path (str): path to overall saving directory
        specific_pdvs (List): Train models on specific list of PDVs to iterate on, only used when prediction granularity is set to PDV
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

        if self.config.load["prediction_granularity"] == "PDV":
            self.config.load["config_lgbm"] = self.config.load["config_lgbm_pdv"]
        else:
            self.config.load["config_lgbm"] = self.config.load["config_lgbm_base"]

        if self.status.aml_pipeline_run:
            self.ml_flow_itm = MlFlowITM(
                self.config,
                self.steps_output_data_path,
                self.date_to_save,
                name_experience="META_UB_TRAINING",
            )

    def run(self, df_input: pd.DataFrame) -> pd.DataFrame:
        """
        Runs training flow of the forecast model. Using processed data:
            - Performs feature engineering
            - Create Tuples of (granularity_site, meta_ub) to iterate on
            - Aggregate existing features at Meta UB level (meteo, holidays...)
            - Creating features (delayed target, averaged target, trends, std...)
            - Looping over each Tuple (granularity_site, meta_ub) to :

                - Train a specific model per Tuple
                - Evaluate a specific model per Tuple
                - Save a specific model per Tuple for later use

        Args:
            df_input (pd.DataFrame): Cleaned pandas' data frame with features from DataProcessing

        Returns:
            None

        """

        logging.info("Start training")
        self.config.load["histo_sales_end_date"] = min(self.config.load["histo_sales_end_date"], df_input["DATE"].max())

        # hyperparameters
        parameters_from_config = self.config.load.parameters_training_model

        # Filter on dates to match scope required
        filtered_df, special_holidays = filtering_dates_and_retrieving_holidays(df=df_input, configs=self.config)

        # Create all (pdv/base/cluster, meta ub) tuples to iterate on
        tuple_meta_ub_granularity = create_tuples_to_model(df=filtered_df, config=self.config)

        # Aggregate features from ub to meta_ub level
        # & remove UB with less than 0.1% of meta ub volume
        filtered_df = aggregate_features_meta_ub_level(
            df=filtered_df, parameters_from_config=parameters_from_config, train=True
        )

        # Feature engineering : Add lagged features
        filtered_df = feature_engineering_meta_ub_cod_site(
            df=filtered_df, config=self.config, prediction_horizon=self.config.load["prediction_horizon"]
        )

        # Iterate modelling for each product granularity x shop granularity
        models_results_df = pd.DataFrame()
        for (meta_ub_n, granularity) in tqdm(tuple_meta_ub_granularity):

            logging.info(f"TRAINING META UB = {meta_ub_n}")

            # Filter meta ub on right meta_ub x pdv/cluster/base id
            sub_df = filtered_df.loc[(filtered_df["META_UB"] == meta_ub_n) & (filtered_df["COD_SITE"] == granularity)]

            if sub_df.empty:
                logging.critical(
                    f"TRAINING: Exit as no data available for UB code {meta_ub_n} and {self.config.load['prediction_granularity']} {granularity}"
                )
            else:
                results_lgbm, model_lgb, model_ts = model_single_ub_granularity(
                    sub_df=sub_df,
                    special_holidays=special_holidays,
                    ub_n=meta_ub_n,
                    granularity=str(granularity),
                    config=self.config,
                    path=self.results_path,
                )

                # BACKTEST UB MODELS
                if model_lgb is not None:
                    backtest_meta_ub_to_ub_level(
                        results_lgbm=results_lgbm,
                        df_input=df_input,
                        config=self.config,
                        meta_ub_n=meta_ub_n,
                        base_path=self.results_path,
                        granularity=str(granularity),
                    )

                    _, summary = compute_metrics(
                        models_results=results_lgbm,
                        target=self.config.load.parameters_training_model["target"],
                        minimum_sales=1.1,
                    )

                if model_lgb is None or model_ts is None:
                    logging.warning(f"Model {meta_ub_n} {granularity} skipped because of too few data points")
                    continue

                if self.status.aml_pipeline_run:
                    if model_lgb is not None:

                        self.ml_flow_itm.instanciate_ml_flow()
                        mlflow_run = self.ml_flow_itm.start_run_ml_flow()
                        logging.info(f"MLFlow Active run_id: {mlflow_run.info.run_id}")

                        self.ml_flow_itm.log_metrics(summary)
                        self.ml_flow_itm.tracking_lgbm(sub_df, model_lgb, meta_ub_n, granularity)
                        self.ml_flow_itm.tracking_prophet(sub_df, model_ts, meta_ub_n, granularity)
                        local_path = str(self.get_local_artifacts_path(meta_ub_n).absolute())
                        self.ml_flow_itm.ml_flow_artifacts(local_path, meta_ub_n, granularity)
                        self.ml_flow_itm.end_run_ml_flow()

                logging.debug(f"Saving product granularity {meta_ub_n} for shop granularity {granularity}")
                self.save_models(
                    lgbm_model=model_lgb,
                    ts_model=model_ts,
                    ub_n=meta_ub_n,
                    granularity=str(granularity),
                )
                self.save_metrics(models_results_df=results_lgbm, ub_n=meta_ub_n)

                models_results_df = pd.concat([models_results_df, results_lgbm])

        # save results and calculations / data & predictions
        self.save_metrics(models_results_df=models_results_df, ub_n=None)

        return models_results_df

    def save_metrics(self, models_results_df: pd.DataFrame, ub_n: Optional[str]) -> None:
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
            target=self.config.load.parameters_training_model["target"],
            minimum_sales=1,
        )

        reformatted_results = reformat_results(models_results=models_results)

        # Save Train results in excel & saves model
        logging.debug(f"Writing results metric into {self.steps_output_data_path} basepath")

        path = self.results_path
        if ub_n is not None:
            path = self.results_path / ub_n

        path.mkdir(parents=True, exist_ok=True)

        # TODO: specific_meta_ubs should be taken from CLI or config by default
        write_training_results(
            df_list=[summary_metrics, models_results, reformatted_results],
            sheet_list=[
                "summary_metrics",
                "models_results",
                "reformatted_models_results",
            ],
            path=path,
            config=self.config,
            date_to_save=self.date_to_save,
        )

    def save_models(
        self,
        lgbm_model,
        ts_model,
        ub_n,
        granularity,
    ):
        """
        Saving models for later use, using below parameters to save specific name for the model path needed to
        filter effectively a specific model based on those criterias and additional one related to the class arguments

        Args:
            lgbm_model: LightGBM meta ub model to save
            ts_model: Time series meta ub model to save
            ub_n (str): Meta UB name
            granularity (str): granularity of saving specific PDV code, CLUSTER name, BASE code

        Return:
            None

        """

        parameters_from_config = self.config.load["parameters_training_model"]

        # hyper parameters
        log_transfo = parameters_from_config["log_transfo"]
        model = parameters_from_config["model_name"]
        tree_model = parameters_from_config["tree_model"]
        prediction_granularity = self.config.load["prediction_granularity"]

        if prediction_granularity == "PDV":
            granularity_desc = "_".join(["codesite", granularity])
        elif prediction_granularity == "CLUSTER":
            granularity_desc = "_".join(["cluster", granularity])
        else:
            # no need to add prefix "base" as it's already in the granularity
            granularity_desc = granularity.lower()

        if log_transfo:
            log = "log_transfo"
        else:
            log = ""

        train_end_date = self.config.load["histo_sales_end_date"].strftime("%d-%m-%Y")

        # save lgbm
        file_name = f"{tree_model}_{model}_trained_model_until_{train_end_date}_{log}_ub_{ub_n}_{granularity_desc}_{self.config.load['prediction_horizon']}j.pickle.dat"
        self.models_path.mkdir(parents=True, exist_ok=True)
        lgb_path_dump = self.models_path / file_name

        pickle.dump([lgbm_model, ts_model, self.config], open(str(lgb_path_dump.absolute()), "wb"))

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
        return Path(self.paths_by_directory["train_results"]) / self.folder_train_results_name

    @property
    def models_path(self) -> Path:
        return Path(self.paths_by_directory["model"]) / self.date_to_save

    def get_local_artifacts_path(self, ub_n: Optional[str]) -> Path:
        if ub_n is None:
            path = self.results_path
        else:
            path = self.results_path / ub_n
        return path
