#
# Copyright (c) 2023 by Boston Consulting Group. All rights reserved
#
# -*- coding: utf-8 -*-

from azureml.core import RunConfiguration
from azureml.data import OutputFileDatasetConfig
from azureml.data.dataset_consumption_config import DatasetConsumptionConfig
from azureml.pipeline.steps import PythonScriptStep


def get_step_training_ub_models(
    step_input_data: OutputFileDatasetConfig,
    output_data: OutputFileDatasetConfig,
    source_dir: str,
    run_config: RunConfiguration,
    allow_reuse: bool,
    training_mode_param: str,
    specific_pdvs: str,
    specific_meta_ubs: str,
    training_horizon: int,
    histo_sales_end_date: str,
    histo_sales_start_date: str,
) -> PythonScriptStep:
    """
    Instantiates the step responsible for the UB models trainings

    Args:
        input_data (DatasetConsumptionConfig): Input data from container holding mapping tables
        step_input_data (DatasetConsumptionConfig): Input data from previous run
        output_data (OutputFileDatasetConfig): Output data (blob) holding output structure of pipelines
        source_dir (str): source directory used with the code
        run_config (RunConfiguration): AML run configuration
        allow_reuse (allow_reuse): Whether to use cached step & data functionnality
        training_mode_param (str): Mode of run, CLUSTER PDV or BASE
        specific_pdvs (str): specifc pdvs to train model on
        specific_meta_ubs (str): specifc meta ubs to train model on
        training_horizon (str): train model to predict j+ horizon sales
        histo_sales_end_date (str): max date in sales data
        histo_sales_start_date (str): min date in sales data

    Returns:
        PythonScriptStep

    """
    return PythonScriptStep(
        script_name="cli.py",
        name="Model training ub",
        arguments=[
            "forecast_fl",
            "step-training-ub-model",
            "--steps-input-data-path",
            step_input_data.as_input(),
            "--steps-output-data-path",
            output_data,
            "--prediction-granularity",
            training_mode_param,
            "--specific-pdvs-list",
            specific_pdvs,
            "--specific-meta-ub-list",
            specific_meta_ubs,
            "--prediction-horizon",
            training_horizon,
            "--histo-sales-end-date",
            histo_sales_end_date,
            "--histo-sales-start-date",
            histo_sales_start_date,
        ],
        source_directory=source_dir,
        runconfig=run_config,
        allow_reuse=allow_reuse,
    )
