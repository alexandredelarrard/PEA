#
# Copyright (c) 2023 by Boston Consulting Group. All rights reserved
#
# -*- coding: utf-8 -*-

import logging
import warnings

import click
import pandas as pd
from forecast_fl.constants.command_line_interface import (
    BROKEN_RATE_ARGS,
    BROKEN_RATE_KWARGS,
    CONFIG_ARGS,
    CONFIG_KWARGS,
    HISTO_SALES_END_DATE_ARGS,
    HISTO_SALES_END_DATE_KWARGS,
    HISTO_SALES_START_DATE_ARGS,
    HISTO_SALES_START_DATE_KWARGS,
    INPUT_DATA_PATH_ARGS,
    INPUT_DATA_PATH_KWARGS,
    MIN_DAY_STOCK_ARGS,
    MIN_DAY_STOCK_KWARGS,
    OBJECTIVE_ARGS,
    OBJECTIVE_KWARGS,
    PREDICTION_DATE_MAX_ARGS,
    PREDICTION_DATE_MAX_KWARGS,
    PREDICTION_DATE_MIN_ARGS,
    PREDICTION_DATE_MIN_KWARGS,
    PREDICTION_GRANULARITY_ARGS,
    PREDICTION_GRANULARITY_KWARGS,
    PREDICTION_HORIZON_ARGS,
    PREDICTION_HORIZON_KWARGS,
    SPECIFIC_META_UB_ARGS,
    SPECIFIC_META_UB_KWARGS,
    SPECIFIC_PDVS_ARGS,
    SPECIFIC_PDVS_KWARGS,
    STEPS_INPUT_DATA_PATH_ARGS,
    STEPS_INPUT_DATA_PATH_KWARGS,
    STEPS_OUTPUT_DATA_PATH_ARGS,
    STEPS_OUTPUT_DATA_PATH_KWARGS,
)
from forecast_fl.steps.step_data_loading import DataLoader
from forecast_fl.steps.step_data_processing import DataProcessing
from forecast_fl.steps.step_export_preconisations_in_db import PreconisationExporter
from forecast_fl.steps.step_predicting import PredictingStep
from forecast_fl.steps.step_training_model import TrainingModelStep
from forecast_fl.steps.step_training_ub_model import TrainingUbModelStep
from forecast_fl.utils.cli_helper import SpecialHelpOrder

warnings.filterwarnings("ignore")


@click.group(cls=SpecialHelpOrder)
def cli():
    """FL PIPELINE STEPS **********************************************
    ********************************************************************
    These steps may be long to run, you can run them separately if needed.
    Keep in mind that dependencies may exist between them. A safe choice
    is to re-run first steps regularly to make sure intermediate data
    is up-to-date.
    *********************************************************************
    """


@cli.command(
    help="Load raw data",
    help_priority=1,
)
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*INPUT_DATA_PATH_ARGS, **INPUT_DATA_PATH_KWARGS)
@click.option(*SPECIFIC_PDVS_ARGS, **SPECIFIC_PDVS_KWARGS)
@click.option(*SPECIFIC_META_UB_ARGS, **SPECIFIC_META_UB_KWARGS)
@click.option(*STEPS_OUTPUT_DATA_PATH_ARGS, **STEPS_OUTPUT_DATA_PATH_KWARGS)
@click.option(*PREDICTION_GRANULARITY_ARGS, **PREDICTION_GRANULARITY_KWARGS)
@click.option(*PREDICTION_DATE_MAX_ARGS, **PREDICTION_DATE_MAX_KWARGS)
@click.option(*OBJECTIVE_ARGS, **OBJECTIVE_KWARGS)
def step_load_data(
    config_path,
    input_data_path,
    specific_pdvs,
    specific_meta_ubs,
    steps_output_data_path,
    prediction_granularity,
    prediction_date_max,
    objective,
):

    logging.debug(f"Starting load data with argument {prediction_granularity}, {specific_meta_ubs}, {specific_pdvs}")

    step = DataLoader(
        config_path=config_path,
        input_data_path=input_data_path,
        steps_output_data_path=steps_output_data_path,
        specific_pdvs=specific_pdvs,
        specific_meta_ubs=specific_meta_ubs,
        prediction_granularity=prediction_granularity,
        prediction_date_max=prediction_date_max,
        objective=objective,
    )

    step.run()


@cli.command(
    help="Process raw data",
    help_priority=2,
)
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*PREDICTION_GRANULARITY_ARGS, **PREDICTION_GRANULARITY_KWARGS)
@click.option(*STEPS_INPUT_DATA_PATH_ARGS, **STEPS_INPUT_DATA_PATH_KWARGS)
@click.option(*STEPS_OUTPUT_DATA_PATH_ARGS, **STEPS_OUTPUT_DATA_PATH_KWARGS)
@click.option(*OBJECTIVE_ARGS, **OBJECTIVE_KWARGS)
def step_process_data(config_path, prediction_granularity, steps_input_data_path, steps_output_data_path, objective):

    step = DataProcessing(
        config_path=config_path,
        steps_input_data_path=steps_input_data_path,
        steps_output_data_path=steps_output_data_path,
        prediction_granularity=prediction_granularity,
        objective=objective,
    )

    datas = DataLoader.load_output(
        steps_input_data_path=step.paths_by_directory["intermediate"],
        objective=step.config.load["objective"],
        aml_run_id=step.aml_run_id,
    )

    step.run(datas=datas)


@cli.command(
    help="Run the training of the model",
    help_priority=3,
)
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*STEPS_INPUT_DATA_PATH_ARGS, **STEPS_INPUT_DATA_PATH_KWARGS)
@click.option(*STEPS_OUTPUT_DATA_PATH_ARGS, **STEPS_OUTPUT_DATA_PATH_KWARGS)
@click.option(*SPECIFIC_PDVS_ARGS, **SPECIFIC_PDVS_KWARGS)
@click.option(*SPECIFIC_META_UB_ARGS, **SPECIFIC_META_UB_KWARGS)
@click.option(*PREDICTION_GRANULARITY_ARGS, **PREDICTION_GRANULARITY_KWARGS)
@click.option(*PREDICTION_HORIZON_ARGS, **PREDICTION_HORIZON_KWARGS)
@click.option(*HISTO_SALES_START_DATE_ARGS, **HISTO_SALES_START_DATE_KWARGS)
@click.option(*HISTO_SALES_END_DATE_ARGS, **HISTO_SALES_END_DATE_KWARGS)
def step_training_model(
    config_path,
    steps_input_data_path,
    steps_output_data_path,
    specific_pdvs,
    specific_meta_ubs,
    prediction_granularity,
    prediction_horizon,
    histo_sales_end_date,
    histo_sales_start_date,
):

    logging.info("Starts step training")

    step = TrainingModelStep(
        config_path=config_path,
        steps_input_data_path=steps_input_data_path,
        steps_output_data_path=steps_output_data_path,
        specific_pdvs=specific_pdvs,
        specific_meta_ubs=specific_meta_ubs,
        prediction_granularity=prediction_granularity,
        prediction_horizon=prediction_horizon,
        histo_sales_end_date=histo_sales_end_date,
        histo_sales_start_date=histo_sales_start_date,
    )

    df_input = DataProcessing.load_output(
        steps_input_data_path=step.paths_by_directory["intermediate"],
        aml_run_id=step.aml_run_id,
        cluster_mode=step.config.load["prediction_granularity"],
    )

    step.run(df_input=df_input)


@cli.command(
    help="Run the training of the model",
    help_priority=4,
)
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*STEPS_INPUT_DATA_PATH_ARGS, **STEPS_INPUT_DATA_PATH_KWARGS)
@click.option(*STEPS_OUTPUT_DATA_PATH_ARGS, **STEPS_OUTPUT_DATA_PATH_KWARGS)
@click.option(*SPECIFIC_PDVS_ARGS, **SPECIFIC_PDVS_KWARGS)
@click.option(*SPECIFIC_META_UB_ARGS, **SPECIFIC_META_UB_KWARGS)
@click.option(*PREDICTION_GRANULARITY_ARGS, **PREDICTION_GRANULARITY_KWARGS)
@click.option(*PREDICTION_HORIZON_ARGS, **PREDICTION_HORIZON_KWARGS)
@click.option(*HISTO_SALES_START_DATE_ARGS, **HISTO_SALES_START_DATE_KWARGS)
@click.option(*HISTO_SALES_END_DATE_ARGS, **HISTO_SALES_END_DATE_KWARGS)
def step_training_ub_model(
    config_path,
    steps_input_data_path,
    steps_output_data_path,
    specific_pdvs,
    specific_meta_ubs,
    prediction_granularity,
    prediction_horizon,
    histo_sales_end_date,
    histo_sales_start_date,
):

    logging.info("Starts step training UB")

    step = TrainingUbModelStep(
        config_path=config_path,
        steps_input_data_path=steps_input_data_path,
        steps_output_data_path=steps_output_data_path,
        specific_pdvs=specific_pdvs,
        specific_meta_ubs=specific_meta_ubs,
        prediction_granularity=prediction_granularity,
        prediction_horizon=prediction_horizon,
        histo_sales_end_date=histo_sales_end_date,
        histo_sales_start_date=histo_sales_start_date,
    )

    df_input = DataProcessing.load_output(
        steps_input_data_path=step.paths_by_directory["intermediate"], aml_run_id=step.aml_run_id
    )

    step.run(df_input=df_input)


@cli.command(
    help="Run the prediction of the model",
    help_priority=5,
)
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*STEPS_INPUT_DATA_PATH_ARGS, **STEPS_INPUT_DATA_PATH_KWARGS)
@click.option(*STEPS_OUTPUT_DATA_PATH_ARGS, **STEPS_OUTPUT_DATA_PATH_KWARGS)
@click.option(*SPECIFIC_PDVS_ARGS, **SPECIFIC_PDVS_KWARGS)
@click.option(*SPECIFIC_META_UB_ARGS, **SPECIFIC_META_UB_KWARGS)
@click.option(*PREDICTION_GRANULARITY_ARGS, **PREDICTION_GRANULARITY_KWARGS)
@click.option(*PREDICTION_DATE_MIN_ARGS, **PREDICTION_DATE_MIN_KWARGS)
@click.option(*PREDICTION_DATE_MAX_ARGS, **PREDICTION_DATE_MAX_KWARGS)
@click.option(*HISTO_SALES_END_DATE_ARGS, **HISTO_SALES_END_DATE_KWARGS)
def step_predicting(
    config_path,
    steps_input_data_path,
    steps_output_data_path,
    specific_pdvs,
    specific_meta_ubs,
    prediction_granularity,
    prediction_date_min,
    prediction_date_max,
    histo_sales_end_date,
):

    logging.info("Start step predicting")

    step = PredictingStep(
        config_path=config_path,
        steps_input_data_path=steps_input_data_path,
        steps_output_data_path=steps_output_data_path,
        specific_pdvs=specific_pdvs,
        specific_meta_ubs=specific_meta_ubs,
        prediction_granularity=prediction_granularity,
        prediction_date_min=prediction_date_min,
        prediction_date_max=prediction_date_max,
        histo_sales_end_date=histo_sales_end_date,
    )

    logging.info("Loads processing & raw data results")
    datas = DataLoader.load_output(
        steps_input_data_path=step.paths_by_directory["intermediate"],
        objective=step.config.load["objective"],
        aml_run_id=step.aml_run_id,
    )
    df_input = DataProcessing.load_output(
        steps_input_data_path=step.paths_by_directory["intermediate"],
        objective=step.config.load["objective"],
        aml_run_id=step.aml_run_id,
    )

    df_input_cluster = pd.DataFrame()
    if prediction_granularity == "CLUSTER":
        df_input_cluster = DataProcessing.load_output(
            steps_input_data_path=step.paths_by_directory["intermediate"],
            objective=step.config.load["objective"],
            aml_run_id=step.aml_run_id,
            cluster_mode=step.config.load["prediction_granularity"],
        )

    step.run(df_input=df_input, df_input_cluster=df_input_cluster, datas=datas)


@cli.command(
    help="Format and exports run into database",
    help_priority=6,
)  # TODO: could add extra argument like UB, date, etc...
@click.option(*CONFIG_ARGS, **CONFIG_KWARGS)
@click.option(*STEPS_INPUT_DATA_PATH_ARGS, **STEPS_INPUT_DATA_PATH_KWARGS)
@click.option(*STEPS_OUTPUT_DATA_PATH_ARGS, **STEPS_OUTPUT_DATA_PATH_KWARGS)
@click.option(*PREDICTION_GRANULARITY_ARGS, **PREDICTION_GRANULARITY_KWARGS)
@click.option(*HISTO_SALES_END_DATE_ARGS, **HISTO_SALES_END_DATE_KWARGS)
@click.option(*PREDICTION_DATE_MIN_ARGS, **PREDICTION_DATE_MIN_KWARGS)
@click.option(*PREDICTION_DATE_MAX_ARGS, **PREDICTION_DATE_MAX_KWARGS)
@click.option(*MIN_DAY_STOCK_ARGS, **MIN_DAY_STOCK_KWARGS)
@click.option(*BROKEN_RATE_ARGS, **BROKEN_RATE_KWARGS)
def step_format_and_export_into_db(
    config_path,
    steps_input_data_path,
    steps_output_data_path,
    prediction_granularity,
    histo_sales_end_date,
    prediction_date_min,
    prediction_date_max,
    min_day_stock,
    broken_rate,
):

    logging.info("Starts formatting and export step of preco from post_processing")

    step = PreconisationExporter(
        config_path=config_path,
        steps_input_data_path=steps_input_data_path,
        steps_output_data_path=steps_output_data_path,
        prediction_granularity=prediction_granularity,
        histo_sales_end_date=histo_sales_end_date,
        prediction_date_min=prediction_date_min,
        prediction_date_max=prediction_date_max,
        min_day_stock=min_day_stock,
        broken_rate=broken_rate,
    )

    logging.info("Loads processing & raw data results")
    datas = DataLoader.load_output(
        steps_input_data_path=step.paths_by_directory["intermediate"],
        objective=step.config.load["objective"],
        aml_run_id=step.aml_run_id,
    )
    df_input = DataProcessing.load_output(
        steps_input_data_path=step.paths_by_directory["intermediate"],
        objective=step.config.load["objective"],
        aml_run_id=step.aml_run_id,
    )

    df_input_cluster = pd.DataFrame()
    if prediction_granularity == "CLUSTER":
        df_input_cluster = DataProcessing.load_output(
            steps_input_data_path=step.paths_by_directory["intermediate"],
            objective=step.config.load["objective"],
            aml_run_id=step.aml_run_id,
            cluster_mode=step.config.load["prediction_granularity"],
        )

    predictions_raw, predictions_ub_raw = PredictingStep.load_output(
        steps_input_data_path=step.paths_by_directory["intermediate"], aml_run_id=step.aml_run_id
    )

    step.run(
        df_input=df_input,
        df_input_cluster=df_input_cluster,
        datas=datas,
        predictions_raw=predictions_raw,
        predictions_ub_raw=predictions_ub_raw,
    )
