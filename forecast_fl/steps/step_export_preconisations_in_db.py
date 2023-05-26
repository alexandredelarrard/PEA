#
# Copyright (c) 2022 by Boston Consulting Group. All rights reserved
#
# -*- coding: utf-8 -*-
import datetime
import logging
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from forecast_fl.data_evaluation.evaluate_results import compute_mape_for_existing_data
from forecast_fl.data_evaluation.save_model_results_plots import (
    write_prediction_results,
)
from forecast_fl.data_exporter.preconisation_sql_exporter import (
    PreconisationSQLExporter,
)
from forecast_fl.data_postprocessing.direct_proportion_base import (
    adjust_target_for_base_with_order_day,
    handle_direct_proportion,
)
from forecast_fl.data_postprocessing.final_predictions_ensembling import (
    final_predictions,
)
from forecast_fl.data_postprocessing.prediction_formatting import (
    format_result_dataframe,
)
from forecast_fl.data_postprocessing.split_from_cluster_to_pdv import (
    split_predictions_from_cluster_to_pdv,
)
from forecast_fl.data_postprocessing.split_predictions_from_meta_ub_to_ub import (
    split_predictions_from_meta_ub_to_ub,
)
from forecast_fl.steps.step import Step

pd.options.mode.chained_assignment = None


class PreconisationExporter(Step):
    """
    Post processing using prediction data:
        - Postprocess raw predictions
        - Save post process predictions
        - Export predictions in SQL database

    Args:
        config (str): Config file
        prediction_granularity (str): Mode of run (Base, PDV, Cluster)
        steps_input_data_path (str): Path to the base input folder
        steps_output_data_path (str): Path to the base output folder
        post_processed_result_sheet_name (str): Sheet_name of output of post processing holding the result to export
        prediction_date_min (datetime.date) : Minimum date of prediction (included)
        prediction_date_max (datetime.date): Maximum date of prediction (included)
        histo_sales_end_date (datetime.date): Latest date to filter df_input for prediction

    """

    def __init__(
        self,
        config_path: str,
        steps_input_data_path: str,
        steps_output_data_path: str,
        prediction_granularity: str,
        histo_sales_end_date: datetime.date,
        prediction_date_min: datetime.date,
        prediction_date_max: datetime.date,
        min_day_stock: str,
        broken_rate: str,
        save: bool = True,
    ):

        super().__init__(
            config_path=config_path,
            steps_input_data_path=steps_input_data_path,
            steps_output_data_path=steps_output_data_path,
        )
        self.config.load["prediction_granularity"] = self.check_prediction_granularity(prediction_granularity)
        self.config.load["prediction_date_max"] = self.check_prediction_date_max(prediction_date_max)
        self.config.load["prediction_date_min"] = self.check_prediction_date_min(prediction_date_min)
        self.config.load["histo_sales_end_date"] = self.check_histo_sales_end_date(histo_sales_end_date)
        self.config.load.prediction_mode["min_days_stock"] = self.check_min_day_stock(min_day_stock)
        self.config.load.prediction_mode["broken_rate"] = self.check_broken_rate(broken_rate)
        self.config.load["objective"] = "predicting"
        self.save = save

    def run(self, predictions_raw, predictions_ub_raw, df_input, df_input_cluster, datas):
        """
        Post process predictions following the steps below :
            - Split Meta UB predictions from CLUSTER x Meta UB to PDV x Meta UB when prediction_granularity is set to CLUSTER
            - Split Meta UB predictions to UB predictions
            - Create final prediction at UB level : UB predictions from UB models and Meta UB models splited to UB using past proportion
            - Compute metrics when predictions are during dates occuring in the past
            - Format predictions results to match SQL db
            - Export predictions to SQL db

        Args:
            predictions_raw (dict): Meta UB level predictions from PredictingStep class saved output
            predictions_ub_raw (dict): UB level predictions from PredictingStep class saved output
            df_input (pd.DataFrame): processed input at PDV or BASE level from DataProcessing class saved output
            df_input_cluster (pd.DataFrame): processed input at CLUSTER level from DataProcessing class saved output
            datas (Dict): raw datas from DataLoader class saved output

        Returns:
            df_data_to_export (pd.DataFrame): formatedgit p data exported to SQL

        """

        top_ubs = df_input[["COD_SITE", "META_UB", "UB_CODE"]].drop_duplicates()
        clustering = top_ubs.merge(datas["PDV_CLUSTERING"], how="left", on=["COD_SITE"])[
            ["CLUSTER", "COD_SITE", "META_UB"]
        ].drop_duplicates()

        # update to last available date in history
        self.config.load["histo_sales_end_date"] = min(self.config.load["histo_sales_end_date"], df_input["DATE"].max())

        logging.info("POSTPROCESSING: postprocess prediction made")
        train_df = df_input[df_input["DATE"] <= self.config.load["histo_sales_end_date"]]

        # ensure input is valide
        predictions_ub_raw_all_dates, predictions_raw_all_dates = self.input_check(predictions_ub_raw, predictions_raw)

        # Split Cluster predictions to PDV
        prediction_output_meta_ub_pdv, prediction_output_meta_ub_cluster = split_predictions_from_cluster_to_pdv(
            prediction_output=predictions_raw_all_dates,
            df_input_pdv_level=train_df,
            clustering=clustering,
            config=self.config,
        )

        # Split META UB predictions to UB CODE
        prediction_output_ub_pdv = split_predictions_from_meta_ub_to_ub(
            prediction_output=prediction_output_meta_ub_pdv,
            df=train_df,
            top_ubs=top_ubs,
            config=self.config,
        )

        # prediction ensembling from the 2 predictions
        prediction_output_ub_pdv_postprocessed = final_predictions(
            prediction_output_ub_pdv, predictions_ub_raw_all_dates, config=self.config
        )

        df_to_export, df_input = handle_direct_proportion(
            config=self.config, df_input=df_input, datas=datas, df_to_export=prediction_output_ub_pdv_postprocessed
        )

        # Format predictions processed before exporting in SQL
        df_data_to_export = format_result_dataframe(
            df_to_export=df_to_export,
            df_input=df_input,
            datas=datas,
            top_ubs=top_ubs,
            config=self.config,
        )

        df_input = adjust_target_for_base_with_order_day(config=self.config, df_input=df_input, taget_variable="TARGET")

        # compute metrics if mode backtest
        logging.info("POSTPROCESSING: computing accuracy metrics")
        (
            prediction_output_meta_ub_cluster,
            summary_pred_output_cluster,
            prediction_output_meta_ub_pdv,
            summary_pred_output_meta_ub_pdv,
            prediction_output_pdv_ub,
            summary_pred_output_pdv_ub,
        ) = self._compute_metrics_on_prediction_results(
            prediction_output_meta_ub_cluster=prediction_output_meta_ub_cluster,
            prediction_output_meta_ub_pdv=prediction_output_meta_ub_pdv,
            prediction_output_ub_pdv_postprocessed=prediction_output_ub_pdv_postprocessed,
            df=df_input,
            df_cluster=df_input_cluster,
        )

        self.save_output(
            prediction_output_cluster=prediction_output_meta_ub_cluster,
            summary_pred_output_cluster=summary_pred_output_cluster,
            prediction_output_pdv=prediction_output_meta_ub_pdv,
            summary_pred_output_pdv=summary_pred_output_meta_ub_pdv,
            prediction_output_pdv_ub=prediction_output_pdv_ub,
            summary_pred_output_pdv_ub=summary_pred_output_pdv_ub,
        )

        if self.save:
            exporter = PreconisationSQLExporter(self.config)
            assert not df_data_to_export.empty, "DATA TO EXPORT TO SQL IS EMPTY, NO PREDICTIONS WERE MADE"
            exporter.export(df_preconisations_for_export=df_data_to_export, df_input=df_input)
            logging.info("Exported results into sql table")

        return df_data_to_export

    def input_check(self, predictions_ub_raw, predictions_raw):

        if not bool(predictions_ub_raw):
            logging.warning(
                "POSTPROCESSING: WARNING NO PREDICTIONS FOUND AT LEVEL UBxCOD_SITE, PREDICTION DATA IS EMPTY"
            )
            predictions_ub_raw_all_dates = pd.DataFrame(
                columns=["DATE", "COD_SITE", "META_UB", "UB_CODE", "PREDICTION_LGBM_LEVEL_UB", "PREDICTION_HORIZON"]
            )
        else:
            predictions_ub_raw_all_dates = pd.concat(predictions_ub_raw.values())

            if predictions_ub_raw_all_dates.empty:
                predictions_ub_raw_all_dates = pd.DataFrame(
                    columns=["DATE", "COD_SITE", "META_UB", "UB_CODE", "PREDICTION_LGBM_LEVEL_UB", "PREDICTION_HORIZON"]
                )

        if not bool(predictions_raw):
            logging.warning(
                "POSTPROCESSING: WARNING NO PREDICTIONS FOUND AT LEVEL META UBxCOD_SITE, PREDICTION DATA IS EMPTY"
            )
            predictions_raw_all_dates = pd.DataFrame(
                columns=[
                    "DATE",
                    "COD_SITE",
                    "META_UB",
                    "UB_CODE",
                    "PREDICTION_UB_USING_PROPORTION",
                    "PREDICTION",
                    "PREDICTION_LGBM",
                    "PREDICTION_TS",
                    "PREDICTION_HORIZON",
                ]
            )
        else:
            predictions_raw_all_dates = pd.concat(predictions_raw.values())
            if predictions_raw_all_dates.empty:
                predictions_raw_all_dates = pd.DataFrame(
                    columns=[
                        "DATE",
                        "COD_SITE",
                        "META_UB",
                        "UB_CODE",
                        "PREDICTION_UB_USING_PROPORTION",
                        "PREDICTION",
                        "PREDICTION_LGBM",
                        "PREDICTION_TS",
                        "PREDICTION_HORIZON",
                    ]
                )

        return predictions_ub_raw_all_dates, predictions_raw_all_dates

    def _compute_metrics_on_prediction_results(
        self,
        prediction_output_meta_ub_cluster,
        prediction_output_meta_ub_pdv,
        prediction_output_ub_pdv_postprocessed,
        df,
        df_cluster,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Computes metrics on postprocessed prediction results when predicted dates occured in the past

        Args:
            prediction_output_meta_ub_cluster (pd.DataFrame): Raw result
            prediction_output_meta_ub_pdv (pd.DataFrame): Splitted at Shop level result
            prediction_output_ub_pdv_postprocessed (pd.DataFrame): Splitted at UB level result
            df (pd.DataFrame): Input dataFrame

        Returns:
            Treated dataframes to save prediction_output_cluster, summary_pred_output_cluster, prediction_output_pdv,
            summary_pred_output_pdv, prediction_output_pdv_ub, summary_pred_output_pdv_ub,

        """
        # TODO: Add metrics to cluster predictions with option in compute_mape_for_existing_data
        if self.config.load["prediction_granularity"] == "CLUSTER":
            prediction_output_cluster, summary_pred_output_cluster = compute_mape_for_existing_data(
                df_cluster,
                prediction_output_meta_ub_cluster,
                self.config,
                ub_level=False,
            )
        else:
            prediction_output_cluster, summary_pred_output_cluster = (
                pd.DataFrame(),
                pd.DataFrame(),
            )

        (prediction_output_meta_ub_pdv, summary_pred_output_pdv,) = compute_mape_for_existing_data(
            df,
            prediction_output_meta_ub_pdv,
            self.config,
            ub_level=False,
        )

        (prediction_output_ub_pdv_postprocessed, summary_pred_output_pdv_ub,) = compute_mape_for_existing_data(
            df,
            prediction_output_ub_pdv_postprocessed,
            self.config,
            ub_level=True,
            prediction_str="PREDICTION_UB_POSTPROCESSED",
        )

        return (
            prediction_output_cluster,
            summary_pred_output_cluster,
            prediction_output_meta_ub_pdv,
            summary_pred_output_pdv,
            prediction_output_ub_pdv_postprocessed,
            summary_pred_output_pdv_ub,
        )

    def save_output(
        self,
        prediction_output_cluster: pd.DataFrame,
        summary_pred_output_cluster: pd.DataFrame,
        prediction_output_pdv: pd.DataFrame,
        summary_pred_output_pdv: pd.DataFrame,
        prediction_output_pdv_ub: pd.DataFrame,
        summary_pred_output_pdv_ub: pd.DataFrame,
    ) -> None:
        """
        Saves prediction outputs and summary of metrics when predictions are done for past.
        Export formated prediction results to SQL

        Args:
            df_result_formatted (pd.DataFrame): data to export to SQL at pdv/base x ub level
            prediction_output_cluster (pd.DataFrame): prediction results at cluster meta ub level
            summary_pred_output_cluster (pd.DataFrame): metrics summary prediction results at cluster meta ub level
            prediction_output_pdv (pd.DataFrame): prediction results at pdv/base x meta ub level
            summary_pred_output_pdv (pd.DataFrame): metrics summary prediction results at pdv/base x meta ub level
            prediction_output_pdv_ub (pd.DataFrame): prediction results at pdv/base x ub level
            summary_pred_output_pdv_ub (pd.DataFrame): metrics summary prediction results at pdv/base x ub level

        Return:
            None

        """

        results_path = Path(self.paths_by_directory["prediction_results"])
        df_list = [
            prediction_output_cluster,
            summary_pred_output_cluster,
            prediction_output_pdv,
            summary_pred_output_pdv,
            prediction_output_pdv_ub,
            summary_pred_output_pdv_ub,
        ]
        sheet_list = [
            "prediction_output_cluster",
            "summary_pred_output_cluster",
            "prediction_output_pdv",
            "summary_pred_output_pdv",
            "prediction_output_pdv_ub",
            "summary_pred_output_pdv_ub",
        ]

        write_prediction_results(
            df_list=df_list,
            sheet_list=sheet_list,
            path=results_path,
            config=self.config,
            date_to_save=self.date_to_save,
        )

    @classmethod
    def load_output(cls):
        pass
