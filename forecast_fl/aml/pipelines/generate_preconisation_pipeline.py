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
from forecast_fl.aml.steps.preconisation_export_in_db_step import (
    get_preconisation_export_in_db_step,
)
from forecast_fl.aml.steps.prediction_step import get_prediction_azure_ml_step
from forecast_fl.aml.steps.process_data_step import get_data_processing_step
from forecast_fl.aml.utils.variables_pipelines import (
    AML_ALLOW_REUSE,  # If False, rebuilds docker image if parameters and code has not changed
)
from forecast_fl.aml.utils.variables_pipelines import (
    DATA_INPUTS,
    MODEL_PREDICTION_PIPELINE_BASENAME,
    MODEL_PREDICTION_PIPELINE_DESCRIPTION,
    MODEL_PREDICTION_PIPELINE_ENDPOINT_NAME,
    PIPELINE_OUTPUTS,
)


def preconisation_generation_pipeline(workspace, blob_datastore, env) -> Pipeline:
    """
    Function that triggers the preconisations generation from the trained model.

    Returns:
        Azure ML pipeline
    """

    source_dir = "./"

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

    data_prediction_output = OutputFileDatasetConfig(
        destination=(blob_datastore, PIPELINE_OUTPUTS + "/outputs")
    )  # It passes a reference in the blob

    data_export_output = OutputFileDatasetConfig(
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
    prediction_mode_param = PipelineParameter(name="prediction_mode", default_value="PDV")
    prediction_date_min = PipelineParameter(
        name="prediction_date_min", default_value="2022-08-03"
    )  # DateFormat : %Y-%m-%d
    prediction_date_max = PipelineParameter(
        name="prediction_date_max", default_value="2022-08-03"
    )  # DateFormat : %Y-%m-%d

    histo_sales_end_date = PipelineParameter(
        name="histo_sales_end_date", default_value="2022-08-01"
    )  # DateFormat : %Y-%m-%d
    specific_pdvs = PipelineParameter(name="specific_pdvs", default_value="all")
    specific_meta_ubs = PipelineParameter(name="specific_meta_ubs", default_value="all")
    min_day_stock = PipelineParameter(name="min_day_stock", default_value="0.5")
    broken_rate = PipelineParameter(name="broken_rate", default_value="0.07")

    # Get differents steps
    data_loading_step = get_data_loading_from_db_step(
        input_data_mounted,
        data_loading_output,
        source_dir,
        run_config=aml_run_config,
        allow_reuse=AML_ALLOW_REUSE,
        prediction_mode_param=prediction_mode_param,
        specific_pdvs=specific_pdvs,
        specific_meta_ubs=specific_meta_ubs,
        objective="predicting",
        prediction_date_max_param=prediction_date_max,
    )

    data_processing_step = get_data_processing_step(
        data_loading_output,
        data_processing_output,
        source_dir,
        run_config=aml_run_config,
        allow_reuse=AML_ALLOW_REUSE,
        objective="predicting",
        prediction_mode_param=prediction_mode_param,
    )

    predict_step = get_prediction_azure_ml_step(
        data_processing_output,
        data_prediction_output,
        source_dir,
        run_config=aml_run_config,
        allow_reuse=AML_ALLOW_REUSE,
        prediction_mode_param=prediction_mode_param,
        prediction_date_min_param=prediction_date_min,
        prediction_date_max_param=prediction_date_max,
        histo_sales_end_date_param=histo_sales_end_date,
        specific_pdvs=specific_pdvs,
        specific_meta_ubs=specific_meta_ubs,
    )

    formatting_and_exporting_step = get_preconisation_export_in_db_step(
        data_prediction_output,
        data_export_output,
        source_dir,
        run_config=aml_run_config,
        allow_reuse=AML_ALLOW_REUSE,
        prediction_mode_param=prediction_mode_param,
        prediction_date_min_param=prediction_date_min,
        prediction_date_max_param=prediction_date_max,
        histo_sales_end_date_param=histo_sales_end_date,
        min_day_stock=min_day_stock,
        broken_rate=broken_rate,
    )

    pipeline = Pipeline(
        workspace=workspace,
        steps=[data_loading_step, data_processing_step, predict_step, formatting_and_exporting_step],
        description="FL model predicting, postprocessing & final result export pipelines",
    )

    logging.info("ScriptRunConfig: \n {}".format(predict_step))

    return pipeline


def main_pipeline_prediction(workspace, blob_datastore, env):

    model_prediction_pipeline = preconisation_generation_pipeline(workspace, blob_datastore, env)
    model_prediction_pipeline.validate()

    published_pipeline = model_prediction_pipeline.publish(
        name=MODEL_PREDICTION_PIPELINE_BASENAME,
        description=MODEL_PREDICTION_PIPELINE_DESCRIPTION,
        version="1.1",
    )

    try:
        pipeline_endpoint = PipelineEndpoint.get(workspace, name=MODEL_PREDICTION_PIPELINE_ENDPOINT_NAME)
        pipeline_endpoint.add_default(published_pipeline)

    except Exception:
        PipelineEndpoint.publish(
            workspace,
            name=MODEL_PREDICTION_PIPELINE_ENDPOINT_NAME,
            pipeline=published_pipeline,
            description=MODEL_PREDICTION_PIPELINE_DESCRIPTION,
        )
