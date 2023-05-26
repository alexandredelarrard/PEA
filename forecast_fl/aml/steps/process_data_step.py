#
# Copyright (c) 2023 by Boston Consulting Group. All rights reserved
#
# -*- coding: utf-8 -*-

from azureml.core import RunConfiguration
from azureml.data import OutputFileDatasetConfig
from azureml.pipeline.steps import PythonScriptStep


def get_data_processing_step(
    step_input_data: OutputFileDatasetConfig,
    output_data: OutputFileDatasetConfig,
    source_dir: str,
    run_config: RunConfiguration,
    allow_reuse: bool,
    objective: str,
    prediction_mode_param: str,
) -> PythonScriptStep:
    """
    Instantiates the step responsible for processing the data, after data load.

    Args:
        step_input_data (OutputFileDatasetConfig): Input data from previous run
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
        name="Data processing for train &/or predict",
        arguments=[
            "forecast_fl",
            "step-process-data",
            "--steps-input-data-path",
            step_input_data.as_input(),
            "--steps-output-data-path",
            output_data,
            "--objective",
            objective,
            "--prediction-granularity",
            prediction_mode_param,
        ],
        source_directory=source_dir,
        runconfig=run_config,
        allow_reuse=allow_reuse,
    )
