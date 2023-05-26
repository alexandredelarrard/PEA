#
# Copyright (c) 2023 by Boston Consulting Group. All rights reserved
#
# -*- coding: utf-8 -*-

from azureml.core import RunConfiguration
from azureml.data import OutputFileDatasetConfig
from azureml.pipeline.steps import PythonScriptStep


def get_preconisation_export_in_db_step(
    step_input_data: OutputFileDatasetConfig,
    output_data: OutputFileDatasetConfig,
    source_dir: str,
    run_config: RunConfiguration,
    allow_reuse: bool,
    prediction_mode_param: str,
    prediction_date_min_param: str,
    prediction_date_max_param: str,
    histo_sales_end_date_param: str,
    min_day_stock: str,
    broken_rate: str,
) -> PythonScriptStep:
    """
    Instantiates the step responsible for the export of preconisations from blob to data db

    Args:
        input_data (OutputFileDatasetConfig): Input data from container holding mapping tables
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
        name="Formatting and export preconisations into db",
        arguments=[
            "forecast_fl",
            "step-format-and-export-into-db",
            "--steps-input-data-path",
            step_input_data.as_input(),
            "--steps-output-data-path",
            output_data,
            "--prediction-granularity",
            prediction_mode_param,
            "--prediction-date-min",
            prediction_date_min_param,
            "--prediction-date-max",
            prediction_date_max_param,
            "--histo-sales-end-date",
            histo_sales_end_date_param,
            "--min-day-stock",
            min_day_stock,
            "--broken-rate",
            broken_rate,
        ],
        source_directory=source_dir,
        runconfig=run_config,
        allow_reuse=allow_reuse,
    )
