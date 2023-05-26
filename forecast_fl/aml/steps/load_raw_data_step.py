#
# Copyright (c) 2023 by Boston Consulting Group. All rights reserved
#
# -*- coding: utf-8 -*-

from azureml.core import RunConfiguration
from azureml.data import OutputFileDatasetConfig
from azureml.data.dataset_consumption_config import DatasetConsumptionConfig
from azureml.pipeline.steps import PythonScriptStep


def get_data_loading_from_db_step(
    input_data: DatasetConsumptionConfig,
    output_data: OutputFileDatasetConfig,
    source_dir: str,
    run_config: RunConfiguration,
    allow_reuse: bool,
    prediction_mode_param: str,
    specific_pdvs: str,
    specific_meta_ubs: str,
    prediction_date_max_param: str,
    objective: str,
) -> PythonScriptStep:
    """
    Instantiates the step responsible for loading the data from the models, given the parameters.

    Args:
        input_data (DatasetConsumptionConfig): Input data from container holding mapping tables
        output_data (OutputFileDatasetConfig): Output data (blob) holding output structure of pipelines
        source_dir (str): source directory used with the code
        run_config (RunConfiguration): AML run configuration
        allow_reuse (allow_reuse): Whether to use cached step & data functionnality
        prediction_mode_param (str): mode of prediction. CLUSTER PDV or BASE

    Returns:
        PythonScriptStep

    """
    return PythonScriptStep(
        script_name="cli.py",
        name="Data loading from database",
        arguments=[
            "forecast_fl",
            "step-load-data",
            "--input-data-path",
            input_data.as_mount(),
            "--steps-output-data-path",
            output_data,
            "--prediction-granularity",
            prediction_mode_param,
            "--specific-pdvs-list",
            specific_pdvs,
            "--specific-meta-ub-list",
            specific_meta_ubs,
            "--prediction-date-max",
            prediction_date_max_param,
            "--objective",
            objective,
        ],  # TODO: could be put in get_command() function somewhere
        source_directory=source_dir,
        runconfig=run_config,
        allow_reuse=allow_reuse,
    )
