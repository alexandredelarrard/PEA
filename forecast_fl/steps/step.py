#
# Copyright (c) 2023 by Boston Consulting Group. All rights reserved
#
# -*- coding: utf-8 -*-
import datetime
import logging
from abc import abstractmethod
from warnings import simplefilter

import pandas as pd
from dotenv import load_dotenv
from forecast_fl.data_evaluation.save_model_results_plots import (
    create_and_clean_saving_directories,
)
from forecast_fl.utils.azure import CloudStatus, get_run_id
from forecast_fl.utils.config import Config
from forecast_fl.utils.string import camel_to_snake

simplefilter("ignore", category=RuntimeWarning)


class Step:
    """Abstract class outlining what it means to be a Step
    A valid Step has a `run` method to run the code it contains, as well as a valid name and generic methods to save
    and load results.
    """

    def __init__(self, config_path: str, steps_input_data_path: str = "", steps_output_data_path: str = "") -> None:

        load_dotenv()

        self.status = CloudStatus()
        self.aml_run_id = get_run_id()
        self.steps_output_data_path = steps_output_data_path
        self.steps_input_data_path = steps_input_data_path
        self.date_to_save = datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S")

        # initialize config
        self.config_init(config_path)

        # handle saving directories
        self.create_saving_directories()

        logging.config.dictConfig(self.config.logging)

    def config_init(self, config_path):
        self.config = Config(config_path).read()

    def create_saving_directories(self):
        self.paths_by_directory = create_and_clean_saving_directories(self.steps_output_data_path)

    def check_specific_pdvs(self, specific_pdvs):
        specific_pdvs = specific_pdvs.replace('"', "").split(",") if specific_pdvs != "all" else []
        specific_pdvs = list({pdv.strip().lower() for pdv in specific_pdvs})
        return specific_pdvs

    def check_specific_meta_ubs(self, specific_meta_ubs):
        specific_meta_ubs = specific_meta_ubs.replace('"', "").split(",") if specific_meta_ubs != "all" else []
        specific_meta_ubs = list({meta_ub.strip().lower() for meta_ub in specific_meta_ubs})
        return specific_meta_ubs

    def check_prediction_granularity(self, prediction_granularity):
        prediction_granularity = prediction_granularity.replace('"', "").strip().upper()
        assert prediction_granularity in (
            "BASE",
            "PDV",
            "CLUSTER",
        ), f"PREDICTION GRANULARITY SHOULD BE EITHER BASE, CLUSTER OR PDV Got {prediction_granularity}"
        return prediction_granularity

    def check_prediction_date_max(self, prediction_date_max):
        prediction_date_max = pd.to_datetime("/".join(str(prediction_date_max).split("-")[::-1]), format="%d/%m/%Y")
        return prediction_date_max

    def check_prediction_date_min(self, prediction_date_min):
        prediction_date_min = pd.to_datetime("/".join(str(prediction_date_min).split("-")[::-1]), format="%d/%m/%Y")
        return prediction_date_min

    def check_objective(self, objective):
        objective = objective.lower().strip()
        assert objective in (
            "training",
            "predicting",
        ), f"Objective variable must be either training or predicting Got {objective}"
        return objective

    def check_histo_sales_start_date(self, histo_sales_start_date):
        histo_sales_start_date = pd.to_datetime(
            "/".join(str(histo_sales_start_date).split("-")[::-1]), format="%d/%m/%Y"
        )
        return histo_sales_start_date

    def check_histo_sales_end_date(self, histo_sales_end_date):
        histo_sales_end_date = pd.to_datetime("/".join(str(histo_sales_end_date).split("-")[::-1]), format="%d/%m/%Y")
        return histo_sales_end_date

    def check_prediction_horizon(self, prediction_horizon):
        prediction_horizon = int(prediction_horizon)
        return prediction_horizon

    def check_min_day_stock(self, min_day_stock):
        min_day_stock = float(min_day_stock)
        if min_day_stock > 2:
            logging.warning(f"PLEASE BE CAREFUL TOPUP MORE THAN 2 DAYS OF STOCK : {min_day_stock}")
        return min_day_stock

    def check_broken_rate(self, broken_rate):
        broken_rate = float(broken_rate)
        if broken_rate > 0.1:
            logging.warning(f"PLEASE BE CAREFUL BROKEN RATE LARGER THAN 10% WITH VALUE : {broken_rate}")
        return broken_rate

    @property
    def name(self) -> str:
        """The name of the step instance"""
        return self._name

    @abstractmethod
    def run(self, *args, **kwargs):
        """Runs the step"""
        raise NotImplementedError()

    @abstractmethod
    def save_output(self, *args, **kwargs) -> None:
        """save the output(s) of the step"""
        pass

    @classmethod
    @abstractmethod
    def load_output(cls, *args, **kwargs):
        """Loads the output(s) of the step"""
        pass

    @classmethod
    def get_name(cls):
        """Returns a standardized name pattern for pipeline objects"""
        return camel_to_snake(cls.__name__)
