#
# Copyright (c) 2023 by Boston Consulting Group. All rights reserved
#
# -*- coding: utf-8 -*-

"""
A list of constants for CLIs. Typically arguments that are often used in CLI steps. Centralizing their definition allows
to have consistent naming in all our CLIs.
"""
from forecast_fl.utils.cli_helper import CLIDate

CONFIG_ARGS = ("--config", "-c", "config_path")
CONFIG_KWARGS = {
    "default": "configs/main.yml",
    "show_default": True,
    "help": (
        "The path to configuration folder for the run. "
        "This corresponds by default to the nboa configuration included into `config/`. "
        "The config is recursively created / merged with the help of the `OmegaConf` python library."
    ),
}
INPUT_DATA_PATH_ARGS = ("--input-data-path", "input_data_path")
INPUT_DATA_PATH_KWARGS = {
    "type": str,
    "default": "data/",
    "required": False,
    "show_default": True,
    "help": "Path to the bae path input data folder(local or azure)",
}

STEPS_INPUT_DATA_PATH_ARGS = ("--steps-input-data-path", "steps_input_data_path")
STEPS_INPUT_DATA_PATH_KWARGS = {
    "type": str,
    "default": "./saved_results",
    "required": False,
    "show_default": True,
    "help": "Path to the base path of pipelines output data folder, from previous steps (local or azure)",
}

STEPS_OUTPUT_DATA_PATH_ARGS = ("--steps-output-data-path", "steps_output_data_path")
STEPS_OUTPUT_DATA_PATH_KWARGS = {
    "type": str,
    "default": "./saved_results",
    "required": False,
    "show_default": True,
    "help": "Path to the base path of pipelines output data folder (local or azure), in which to save the results",
}

SPECIFIC_PDVS_ARGS = ("--specific-pdvs-list", "specific_pdvs")
SPECIFIC_PDVS_KWARGS = {
    "type": str,
    "default": "all",
    "required": False,
    "show_default": True,
    "help": "List with all PDVs to extract data from, if an empty is given; "
    "no filter on PDVs will be used to query the data",
}

SPECIFIC_META_UB_ARGS = ("--specific-meta-ub-list", "specific_meta_ubs")
SPECIFIC_META_UB_KWARGS = {
    "type": str,
    "default": "all",
    "required": False,
    "show_default": True,
    "help": "List with all meta ubs to train or predict, if an empty list is given; "
    "prediction or train will be made on all meta ub defined",
}

PREDICTION_GRANULARITY_ARGS = ("--prediction-granularity", "prediction_granularity")
PREDICTION_GRANULARITY_KWARGS = {
    "type": str,
    "default": "PDV",
    "required": False,
    "show_default": True,
    "help": "Path to the base path of pipelines output data folder (local or azure), "
    "choose between either BASE, PDV or CLUSTER",
}

PREDICTION_HORIZON_ARGS = ("--prediction-horizon", "prediction_horizon")
PREDICTION_HORIZON_KWARGS = {
    "type": int,
    "default": 2,
    "required": False,
    "show_default": True,
    "help": "Path to the base path of pipelines output data folder (local or azure), "
    "choose between either BASE, PDV or CLUSTER",
}

PREDICTION_DATE_MIN_ARGS = ("--prediction-date-min", "prediction_date_min")
PREDICTION_DATE_MIN_KWARGS = {
    "type": CLIDate(),
    "required": True,
    "show_default": True,
    "help": "Minimum date for which to predict (included), format %Y-%m-%d",
}

PREDICTION_DATE_MAX_ARGS = ("--prediction-date-max", "prediction_date_max")
PREDICTION_DATE_MAX_KWARGS = {
    "type": CLIDate(),
    "default": "",
    "required": True,
    "show_default": True,
    "help": "Maximum date for which to predict (included), format %Y-%m-%d",
}

HISTO_SALES_END_DATE_ARGS = ("--histo-sales-end-date", "histo_sales_end_date")
HISTO_SALES_END_DATE_KWARGS = {
    "type": CLIDate(),
    "required": True,
    "show_default": True,
    "help": "Latest date in features dataframe, format %Y-%m-%d",
}

HISTO_SALES_START_DATE_ARGS = ("--histo-sales-start-date", "histo_sales_start_date")
HISTO_SALES_START_DATE_KWARGS = {
    "type": CLIDate(),
    "required": True,
    "show_default": True,
    "help": "Latest date in features dataframe, format %Y-%m-%d",
}

OBJECTIVE_ARGS = ("--objective", "objective")
OBJECTIVE_KWARGS = {
    "type": str,
    "default": "training",
    "required": True,
    "show_default": True,
    "help": "if predict is the objective set it to predict",
}

MIN_DAY_STOCK_ARGS = ("--min-day-stock", "min_day_stock")
MIN_DAY_STOCK_KWARGS = {
    "type": str,
    "default": "0.5",
    "required": True,
    "show_default": True,
    "help": "proportion to topup on UB prediction",
}

BROKEN_RATE_ARGS = ("--broken-rate", "broken_rate")
BROKEN_RATE_KWARGS = {
    "type": str,
    "default": "0.07",
    "required": True,
    "show_default": True,
    "help": "proportion product broken or picked by ITM internal stores (bakery, etc.)",
}
