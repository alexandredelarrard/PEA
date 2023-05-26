#
# Copyright (c) 2023 by Boston Consulting Group. All rights reserved
#
# -*- coding: utf-8 -*-

import logging
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def mean_absolute_percentage_error(y_true, y_pred):
    """Calculates MAPE given y_true and y_pred"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.median(np.abs((y_true - y_pred) / (y_true))) * 100


def _mape_with_minimum_sales(
    models_results: pd.DataFrame,
    prediction_str: str,
    minimum_sales: float = 1,
    target: str = "TARGET",
) -> pd.DataFrame:
    """Computes model metrics (MAPE, Accuracy) for each data, point de vente, meta unité de besoin"""

    models_results["MAPE"] = 100 * abs(models_results[target] - models_results[prediction_str]) / models_results[target]
    models_results["ACCURACY"] = 100 - models_results["MAPE"]

    models_results["MAPE"] = np.where(models_results[target] >= minimum_sales, models_results["MAPE"], np.nan)

    models_results["ACCURACY"] = np.where(
        models_results[target] >= minimum_sales,
        models_results["ACCURACY"],
        np.nan,
    )

    return models_results


def _symmetric_mape_with_minimum_sales(
    models_results: pd.DataFrame,
    prediction_str: str,
    minimum_sales: float = 1,
    target: str = "TARGET",
) -> pd.DataFrame:
    """Computes model metrics (SMAPE, Accuracy resulting from smape) for each data, point de vente, meta unité de besoin"""
    models_results["SMAPE"] = (
        100
        * abs(models_results[target] - models_results[prediction_str])
        / ((abs(models_results[target]) + abs(models_results[prediction_str])) / 2)
    )
    models_results["S_ACCURACY"] = 100 - models_results["SMAPE"]

    models_results["SMAPE"] = np.where(models_results[target] >= minimum_sales, models_results["SMAPE"], np.nan)

    models_results["S_ACCURACY"] = np.where(
        models_results[target] >= minimum_sales,
        models_results["S_ACCURACY"],
        np.nan,
    )

    return models_results


def _summarize_metrics(
    models_results: pd.DataFrame,
    meta_ub_level: bool = True,
    ub_level: bool = False,
) -> pd.DataFrame:
    """Aggregates metrics (MAPE, Accuracy) per point de vente and meta ub for all dates"""

    if meta_ub_level:
        col_meta_ub = ["META_UB"]
    else:
        col_meta_ub = []

    if ub_level:
        col_ub = ["UB_CODE"]
    else:
        col_ub = []
    summary_metrics = models_results.groupby(["MODEL", "COD_SITE"] + col_meta_ub + col_ub, as_index=False).mean()
    return summary_metrics


def compute_metrics(
    models_results: pd.DataFrame,
    target: str,
    minimum_sales: float = 1,
    prediction_str: str = "PREDICTION_LGB",
    ub_level: bool = False,
    meta_ub_level: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Gets detailed and aggregated (by PdV x Meta UB) model metrics (MAPE, Accuracy, Symmetric MAPE, Symmetric Accuracy)"""

    summary_metrics = pd.DataFrame()
    if not models_results.empty:
        # mape
        models_results = _mape_with_minimum_sales(
            models_results=models_results,
            minimum_sales=minimum_sales,
            target=target,
            prediction_str=prediction_str,
        )
        # smape
        models_results = _symmetric_mape_with_minimum_sales(
            models_results=models_results,
            minimum_sales=minimum_sales,
            target=target,
            prediction_str=prediction_str,
        )

        summary_metrics = _summarize_metrics(
            models_results=models_results,
            ub_level=ub_level,
            meta_ub_level=meta_ub_level,
        )
    return models_results, summary_metrics


def reformat_results(models_results: pd.DataFrame, target: str = "TARGET") -> pd.DataFrame:
    """Retrieves model prediction and target per date, meta unité de besoin"""

    reformatted_results = pd.DataFrame(columns=["DATE", "META_UB", target])
    if not models_results.empty:
        for model in models_results.MODEL.unique():
            temp_reformated_results = models_results.loc[models_results["MODEL"] == model]
            temp_reformated_results = temp_reformated_results.rename(
                columns={"PREDICTION_LGB": "_".join(["PREDICTION_LGB", model])}
            )
            temp_reformated_results = temp_reformated_results[
                ["DATE", "META_UB", target, "_".join(["PREDICTION_LGB", model])]
            ]
            reformatted_results = reformatted_results.merge(
                temp_reformated_results,
                on=["DATE", "META_UB", target],
                how="outer",
            )
    return reformatted_results


def plot(
    df: pd.DataFrame,
    variables_to_plot: List[str],
    title: str,
    paths_by_directory,
    ub_n,
    granularity,
    model_name,
) -> None:
    fig, ax = plt.subplots()
    ax.set_title(f"{title}")
    for variable_to_plot in variables_to_plot:
        ax.plot(df["DATE"], df[variable_to_plot])
    fig.autofmt_xdate()
    plt.savefig(
        paths_by_directory["time_series"] + f"/plot_timeseries_model_{model_name}_ub_{ub_n}_codesite_{granularity}.png"
    )
    plt.show()


def compute_mape_for_existing_data(
    df,
    prediction_output,
    configs,
    ub_level=False,
    prediction_str="PREDICTION",
):
    if not prediction_output.empty:
        ub_level_col = []
        summary_prediction_output = pd.DataFrame()
        if "DATE" in prediction_output.columns and prediction_output["DATE"].max() <= df["DATE"].max():
            if ub_level:
                logging.info("POSTPROCESSING: Computing metrics at Pdv x UB level")
                ub_level_col = ["UB_CODE"]
            else:
                logging.info("POSTPROCESSING: Computing metrics at Pdv x Meta UB level")
            target_per_meta_ub_pdv = df.groupby(["DATE", "META_UB", "COD_SITE"] + ub_level_col, as_index=False)[
                configs.load.prediction_mode["target"]
            ].sum()

            prediction_output = prediction_output.merge(
                target_per_meta_ub_pdv,
                how="left",
                on=["DATE", "META_UB", "COD_SITE"] + ub_level_col,
                validate="1:1",
            )
            prediction_output["MODEL"] = "_".join(
                [
                    configs.load.prediction_mode["tree_model"],
                    configs.load.prediction_mode["model_name"],
                ]
            )
            if ub_level:
                target_per_meta_ub_pdv = (
                    df.groupby(["DATE", "META_UB", "COD_SITE"], as_index=False)[configs.load.prediction_mode["target"]]
                    .sum()
                    .rename(
                        columns={
                            configs.load.prediction_mode["target"]: "_".join(
                                [
                                    configs.load.prediction_mode["target"],
                                    "META_UB",
                                ]
                            )
                        }
                    )
                )

                prediction_output = prediction_output.merge(
                    target_per_meta_ub_pdv,
                    how="left",
                    on=["DATE", "META_UB", "COD_SITE"],
                    validate="m:1",
                )

            prediction_output, summary_prediction_output = compute_metrics(
                models_results=prediction_output,
                target=configs.load.prediction_mode["target"],
                minimum_sales=1,
                prediction_str=prediction_str,
                ub_level=ub_level,
            )
        else:
            logging.warning(
                "RESULTS: Summary metrics will not be calculated as date to "
                "predict is greater than most recent date in train set"
            )
    else:
        logging.warning("RESULTS: EMPTY DATA, NO METRICS TO COMPUTE")
        summary_prediction_output = pd.DataFrame()
    return prediction_output, summary_prediction_output
