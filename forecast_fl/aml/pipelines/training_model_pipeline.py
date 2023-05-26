#
# Copyright (c) 2023 by Boston Consulting Group. All rights reserved
#
# -*- coding: utf-8 -*-

import logging
import os
import sys

sys.path.append(os.getcwd())

from azureml.core import Dataset, RunConfiguration
from azureml.data import OutputFileDatasetConfig
from azureml.pipeline.core import Pipeline, PipelineEndpoint, PipelineParameter
from forecast_fl.aml.steps.load_raw_data_step import get_data_loading_from_db_step
from forecast_fl.aml.steps.process_data_step import get_data_processing_step
from forecast_fl.aml.steps.train_models_step import get_step_training_models
from forecast_fl.aml.utils.variables_pipelines import (
    AML_ALLOW_REUSE,  # If False, rebuilds docker image if parameters and code has not changed
)
from forecast_fl.aml.utils.variables_pipelines import (
    DATA_INPUTS,
    MODEL_TRAINING_PIPELINE_BASENAME,
    MODEL_TRAINING_PIPELINE_DESCRIPTION,
    MODEL_TRAINING_PIPELINE_ENDPOINT_NAME,
    PIPELINE_OUTPUTS,
)


def create_model_training_pipeline(workspace, blob_datastore, env) -> Pipeline:
    """
    Function that triggers model training for specifics PDV and Meta UB, given the mode.

    Returns:
        Azure ML pipeline
    """

    # file config
    source_dir = "./"

    # link to ml flow
    # mlflow.set_tracking_uri(workspace.get_mlflow_tracking_uri())

    # aml config
    AML_CPU_CLUSTER_NAME = os.environ["AML_CPU_CLUSTER_NAME"]

    dataset_input = Dataset.File.from_files(path=(blob_datastore, DATA_INPUTS + "/mapping_files"))
    input_data_mounted = dataset_input.as_named_input("Data_static_files_input")

    data_loading_output = OutputFileDatasetConfig(
        destination=(blob_datastore, PIPELINE_OUTPUTS + "/outputs")
    )  # It passes a reference in the blob

    data_processing_output = OutputFileDatasetConfig(
        destination=(blob_datastore, PIPELINE_OUTPUTS + "/outputs")
    )  # It passes a reference in the blob

    data_training_output = OutputFileDatasetConfig(
        destination=(blob_datastore, PIPELINE_OUTPUTS + "/outputs")
    )  # It passes a reference in the blob

    # Defines compute target
    cpu_cluster = workspace.compute_targets[AML_CPU_CLUSTER_NAME]
    cpu_cluster.add_identity(identity_type="SystemAssigned")

    # Add run environment - making sure we have all the dependencies we will need
    aml_run_config = RunConfiguration()
    aml_run_config.target = cpu_cluster
    aml_run_config.environment = env

    # Add relevant secrets in the AML Keyvault
    # prep_secrets_for_aml(workspace)

    # Pipeline parameters
    training_mode_param = PipelineParameter(name="prediction_mode", default_value="PDV")
    specific_pdvs = PipelineParameter(name="specific_pdvs", default_value="all")
    specific_meta_ubs = PipelineParameter(name="specific_meta_ubs", default_value="all")
    training_horizon = PipelineParameter(name="prediction_horizon", default_value=2)
    histo_sales_end_date = PipelineParameter(name="histo_sales_end_date", default_value="2022-10-01")
    histo_sales_start_date = PipelineParameter(name="histo_sales_start_date", default_value="2018-01-01")

    # Get differents steps
    data_loading_step = get_data_loading_from_db_step(
        input_data_mounted,
        data_loading_output,
        source_dir,
        run_config=aml_run_config,
        allow_reuse=AML_ALLOW_REUSE,
        prediction_mode_param=training_mode_param,
        specific_pdvs=specific_pdvs,
        specific_meta_ubs=specific_meta_ubs,
        objective="training",
        prediction_date_max_param="",
    )

    data_processing_step = get_data_processing_step(
        data_loading_output,
        data_processing_output,
        source_dir,
        run_config=aml_run_config,
        allow_reuse=AML_ALLOW_REUSE,
        objective="training",
        prediction_mode_param=training_mode_param,
    )

    training_step = get_step_training_models(
        data_processing_output,
        data_training_output,
        source_dir,
        run_config=aml_run_config,
        allow_reuse=AML_ALLOW_REUSE,
        training_mode_param=training_mode_param,
        specific_pdvs=specific_pdvs,
        specific_meta_ubs=specific_meta_ubs,
        training_horizon=training_horizon,
        histo_sales_end_date=histo_sales_end_date,
        histo_sales_start_date=histo_sales_start_date,
    )

    training_pipeline = Pipeline(
        workspace=workspace,
        steps=[data_loading_step, data_processing_step, training_step],
        description="FL model training pipeline",
    )

    logging.info("ScriptRunConfig: \n {}".format(training_step))

    # à partir de là, publier la pipeline
    # pipeline.publish & exposée via une API

    return training_pipeline


def main_pipeline_model_meta_ub(workspace, blob_datastore, env):

    model_training_pipeline = create_model_training_pipeline(workspace, blob_datastore, env)

    model_training_pipeline.validate()

    published_pipeline = model_training_pipeline.publish(
        name=MODEL_TRAINING_PIPELINE_BASENAME,
        description=MODEL_TRAINING_PIPELINE_DESCRIPTION,
        version="1.1",
    )

    try:
        pipeline_endpoint = PipelineEndpoint.get(workspace, name=MODEL_TRAINING_PIPELINE_ENDPOINT_NAME)
        pipeline_endpoint.add_default(published_pipeline)

    except Exception as e:
        logging.error(e)
        PipelineEndpoint.publish(
            workspace,
            name=MODEL_TRAINING_PIPELINE_ENDPOINT_NAME,
            pipeline=published_pipeline,
            description=MODEL_TRAINING_PIPELINE_DESCRIPTION,
        )
