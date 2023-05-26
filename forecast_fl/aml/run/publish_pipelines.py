#
# Copyright (c) 2023 by Boston Consulting Group. All rights reserved
#
# -*- coding: utf-8 -*-

import logging
import os
import sys

sys.path.append(os.getcwd())  # Sets working directory to run from root
from dotenv import load_dotenv
from forecast_fl.aml.pipelines.generate_preconisation_pipeline import (
    main_pipeline_prediction,
)
from forecast_fl.aml.pipelines.training_model_pipeline import (
    main_pipeline_model_meta_ub,
)
from forecast_fl.aml.pipelines.training_ub_model_pipeline import main_pipeline_model_ub
from forecast_fl.aml.utils.general_functions import (
    define_environment_azure_ml,
    get_blob_storage,
    get_workspace,
)
from forecast_fl.aml.utils.variables_pipelines import DATA_INPUTS, DATA_INPUTS_FL360
from mlflow import set_tracking_uri

# Sets working directory to run from root
module_logger = logging.getLogger(__name__)

if __name__ == "__main__":

    load_dotenv(".env")
    workspace = get_workspace()
    env = define_environment_azure_ml()
    set_tracking_uri(workspace.get_mlflow_tracking_uri())

    # Data connections
    # We have one datastore for inputs and outputs, as container is easier to manipulate and control
    blob_datastore = get_blob_storage(workspace, DATA_INPUTS, DATA_INPUTS_FL360)

    main_pipeline_prediction(workspace, blob_datastore, env)
    module_logger.info("PREDICTION PIPELINE PUBLISHED")

    main_pipeline_model_meta_ub(workspace, blob_datastore, env)
    module_logger.info("meta ub training PIPELINE PUBLISHED")

    main_pipeline_model_ub(workspace, blob_datastore, env)
    module_logger.info("UB training PIPELINE PUBLISHED")
