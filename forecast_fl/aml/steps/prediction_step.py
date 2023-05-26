#
# Copyright (c) 2023 by Boston Consulting Group. All rights reserved
#
# -*- coding: utf-8 -*-

from azureml.core import RunConfiguration
from azureml.data import OutputFileDatasetConfig
from azureml.pipeline.steps import PythonScriptStep


def get_prediction_azure_ml_step(
    step_input_data: OutputFileDatasetConfig,
    output_data: OutputFileDatasetConfig,
    source_dir: str,
    run_config: RunConfiguration,
    allow_reuse: bool,
    prediction_mode_param: str,
    prediction_date_min_param: str,
    prediction_date_max_param: str,
    histo_sales_end_date_param: str,
    specific_pdvs: str,
    specific_meta_ubs: str,
) -> PythonScriptStep:
    """
    Instantiates the step responsible for the models prediction from trained model

    Args:
        step_input_data (OutputFileDatasetConfig): Input data from previous run
        output_data (OutputFileDatasetConfig): Output data (blob) holding output structure of pipelines
        source_dir (str): source directory used with the code
        run_config (RunConfiguration): AML run configuration
        allow_reuse (allow_reuse): Whether to use cached step & data functionnality
        prediction_mode_param (str): mode of prediction. CLUSTER PDV or BASE
        prediction_date_min_param (str): Minimum date for prediction. Format %Y-%m-%d
        prediction_date_max_param (str): Maximum date for prediction. Format %Y-%m-%d
        histo_sales_end_date_param (str): Latest date in features df for prediction.
        Will determine horizon of prediction

    Returns:
        PythonScriptStep

    """

    return PythonScriptStep(
        script_name="cli.py",
        name="Predictions of preco / demand at level UB and META UB",
        arguments=[
            "forecast_fl",
            "step-predicting",
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
            "--specific-pdvs-list",
            specific_pdvs,
            "--specific-meta-ub-list",
            specific_meta_ubs,
        ],
        source_directory=source_dir,
        runconfig=run_config,
        allow_reuse=allow_reuse,
    )
