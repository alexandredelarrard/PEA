# -*- coding: utf-8 -*-
import logging
from pathlib import Path
from random import SystemRandom
from string import ascii_uppercase, digits
from time import sleep
from typing import Optional

from azure.core.exceptions import ServiceRequestError, ServiceResponseError
from forecast_fl.utils.azure import CloudStatus
from mlflow import (
    create_experiment,
    end_run,
    get_experiment,
    log_artifacts,
    log_metric,
    log_param,
    start_run,
)
from mlflow.lightgbm import DEFAULT_AWAIT_MAX_SLEEP_SECONDS as LGBM_AWAIT
from mlflow.lightgbm import log_model as lgbm_log_model
from mlflow.prophet import DEFAULT_AWAIT_MAX_SLEEP_SECONDS as PROPHET_AWAIT
from mlflow.prophet import log_model as prophet_log_model


class MlFlowITM:
    """
    Dedicated class to handle ML flow artifacts
    """

    def __init__(self, config, steps_output_data_path, date_to_save, name_experience) -> None:
        self.steps_output_data_path = steps_output_data_path
        self.config = config
        self.prediction_granularity = self.config.load["prediction_granularity"]
        self.prediction_horizon = self.config.load["prediction_horizon"]
        self.date_to_save = date_to_save
        self.name_experience = name_experience

        self.status = CloudStatus()
        self.random_hash = "".join(SystemRandom().choice(ascii_uppercase + digits) for _ in range(4))
        self.retries = 3

    def instanciate_ml_flow(self):

        # Get the MLFlow Experiment
        self.mlflow_experiment_id = create_experiment(name=self.experiment_name)
        self.mlflow_experiment = get_experiment(experiment_id=self.mlflow_experiment_id)
        if self.mlflow_experiment is not None:
            logging.info(
                f"Created MLFlow Experiment {self.experiment_name} with id {self.mlflow_experiment_id} and object {self.mlflow_experiment}"
            )

    def start_run_ml_flow(self):
        return start_run(experiment_id=self.mlflow_experiment_id)

    def end_run_ml_flow(self):
        return end_run()

    def get_model_name(self, model_class: str, meta_ub_n: str, granularity: str) -> str:
        horizon = str(self.prediction_horizon) + "j"
        return "_".join(
            [
                model_class,
                meta_ub_n,
                granularity,
                self.date_to_save,
                self.prediction_granularity,
                horizon,
            ]
        ).replace(" ", "_")

    def tracking_lgbm(self, sub_df, model_lgb, meta_ub_n, granularity):
        for _ in range(self.retries):
            try:
                mlflow_path = f"{self.mlflow_experiment.artifact_location}/{meta_ub_n}/{granularity}"
                lgbm_log_model(
                    model_lgb,
                    artifact_path=f"{mlflow_path}/lgbm/",
                    conda_env="environment.yml",
                    code_paths=None,
                    registered_model_name=self.get_model_name(
                        model_class="lgbm", meta_ub_n=meta_ub_n, granularity=granularity
                    ),
                    signature=None,
                    input_example=sub_df.head(),
                    await_registration_for=LGBM_AWAIT,
                    pip_requirements=None,
                    extra_pip_requirements=None,
                )
            except ServiceRequestError as e:
                logging.error(f"Error saving mlflow model going to retry {e}")
                sleep(10)
            except ServiceResponseError as e:
                logging.error(f"Error connecting mlflow model going to retry {e}")
                sleep(10)
            except Exception as e:
                logging.error(f"Error overall mlflow model going to retry {e}")
                sleep(10)
            break

    def tracking_prophet(self, sub_df, model_ts, meta_ub_n, granularity):

        for _ in range(self.retries):
            try:
                mlflow_path = f"{self.mlflow_experiment.artifact_location}/{meta_ub_n}/{granularity}"
                prophet_log_model(
                    model_ts,
                    artifact_path=f"{mlflow_path}/prophet/",
                    conda_env="environment.yml",
                    code_paths=None,
                    registered_model_name=self.get_model_name(
                        model_class="prophet", meta_ub_n=meta_ub_n, granularity=granularity
                    ),
                    signature=None,
                    input_example=sub_df.head(),
                    await_registration_for=PROPHET_AWAIT,
                    pip_requirements=None,
                    extra_pip_requirements=None,
                )
            except ServiceRequestError as e:
                logging.error(f"Error saving mlflow model going to retry {e}")
                sleep(10)
            except ServiceResponseError as e:
                logging.error(f"Error connecting mlflow model going to retry {e}")
                sleep(10)
            except Exception as e:
                logging.error(f"Error overall mlflow model going to retry {e}")
                sleep(10)
            break

    def ml_flow_artifacts(self, local_path, meta_ub_n, granularity):

        for _ in range(self.retries):
            try:
                mlflow_path = f"{self.mlflow_experiment.artifact_location}/{meta_ub_n}/{granularity}"
                log_artifacts(local_dir=local_path, artifact_path=mlflow_path)
            except ServiceRequestError as e:
                logging.error(f"Error saving mlflow model going to retry {e}")
                sleep(10)
            except ServiceResponseError as e:
                logging.error(f"Error connecting mlflow model going to retry {e}")
                sleep(10)
            except Exception as e:
                logging.error(f"Error overall mlflow model going to retry {e}")
                sleep(10)
            break

    def log_metrics(self, summary) -> None:

        metrics_dict = summary.to_dict("records")[0]
        metrics = [
            "TARGET",
            "PREDICTION_PROPHET",
            "PREDICTION_LGB",
            "ERROR_PREDICTION_PROPHET",
            "ERROR_PREDICTION_LGBM",
            "MAPE",
            "ACCURACY",
            "SMAPE",
            "S_ACCURACY",
        ]

        for key, value in metrics_dict.items():
            if type(value) == float and key in metrics:
                log_metric(key, value)
            else:
                log_param(str(key), value)

    @property
    def folder_train_results_name(self) -> str:
        horizon = str(self.prediction_horizon) + "j"
        return "_".join(
            [
                self.date_to_save,
                self.config.load.parameters_training_model["model_name"],
                self.prediction_granularity,
                horizon,
            ]
        )

    @property
    def experiment_name(self) -> str:
        horizon = str(self.prediction_horizon) + "j"
        return "_".join(
            [
                self.name_experience,
                self.date_to_save,
                self.config.load.parameters_training_model["model_name"],
                self.prediction_granularity,
                horizon,
                self.random_hash,
            ]
        )

    def log_files(self, path_string: str) -> None:
        path = Path(path_string)
        paths = path.glob("**/*")
        files = [f for f in paths if f.is_file()]
        logging.info(f"Found following files {files} in path {path_string}")
