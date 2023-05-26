#
# Copyright (c) 2023 by Boston Consulting Group. All rights reserved
#
# -*- coding: utf-8 -*-

import logging
from typing import Dict

import numpy as np
import pandas as pd


def merge_ensemble_predictions(
    predictions_ub_raw_all_dates: pd.DataFrame, prediction_output_ub_pdv: pd.DataFrame
) -> pd.DataFrame:
    prediction_ub_pdv = predictions_ub_raw_all_dates.merge(
        prediction_output_ub_pdv,
        on=["DATE", "COD_SITE", "META_UB", "UB_CODE", "PREDICTION_HORIZON"],
        how="outer",
        validate="1:1",
    )
    return prediction_ub_pdv


def deduce_final_prediction_if_only_one(
    predictions_ub_raw_all_dates: pd.DataFrame, prediction_output_ub_pdv: pd.DataFrame, prediction_ub_pdv: pd.DataFrame
) -> pd.DataFrame:

    # if only meta ub to ub predictions available then return meta ub ventilation
    if predictions_ub_raw_all_dates.empty and not prediction_output_ub_pdv.empty:
        prediction_ub_pdv["PREDICTION_UB_POSTPROCESSED"] = (
            prediction_ub_pdv["PREDICTION_UB_USING_PROPORTION"].round(2).clip(0, None)
        )
        return prediction_ub_pdv

    # if only lgbm ub predictions available then return it
    if prediction_output_ub_pdv.empty and not predictions_ub_raw_all_dates.empty:
        prediction_ub_pdv["PREDICTION_UB_POSTPROCESSED"] = (
            prediction_ub_pdv["PREDICTION_LGBM_LEVEL_UB"].round(2).clip(0, None)
        )
        return prediction_ub_pdv

    return prediction_ub_pdv


def weighted_ensembling_predictions(prediction_ub_pdv: pd.DataFrame, config: Dict) -> pd.DataFrame:

    # final prediction -> if too large difference between both models, keep lowest
    weight_meta_ub = eval(f"config.load.prediction_mode.{config.load['prediction_granularity']}.w_model_meta_ub")
    weight_ub = eval(f"config.load.prediction_mode.{config.load['prediction_granularity']}.w_model_ub")

    meta_ub_preds = prediction_ub_pdv["PREDICTION_UB_USING_PROPORTION"].round(2).clip(0, None)
    direct_ub_preds = prediction_ub_pdv["PREDICTION_LGBM_LEVEL_UB"].round(2).clip(0, None)

    if weight_ub > 0 and weight_meta_ub > 0:
        prediction_ub_pdv["PREDICTION_UB_POSTPROCESSED"] = np.where(
            meta_ub_preds.isnull(),
            direct_ub_preds,
            np.where(
                direct_ub_preds.isnull(),
                meta_ub_preds,
                weight_meta_ub * meta_ub_preds + weight_ub * direct_ub_preds,
            ),
        )
    elif weight_ub == 0:
        prediction_ub_pdv["PREDICTION_UB_POSTPROCESSED"] = meta_ub_preds
    elif weight_meta_ub == 0:
        prediction_ub_pdv["PREDICTION_UB_POSTPROCESSED"] = direct_ub_preds
    else:
        logging.critical("BOTH WEIGHTS FOR META UB AND DIRECT UB ARE = 0, predictions will all be 0 !!!")

    return prediction_ub_pdv


def final_predictions(
    prediction_output_ub_pdv: pd.DataFrame, predictions_ub_raw_all_dates: pd.DataFrame, config: Dict
) -> pd.DataFrame:

    prediction_ub_pdv = merge_ensemble_predictions(predictions_ub_raw_all_dates, prediction_output_ub_pdv)

    prediction_ub_pdv = deduce_final_prediction_if_only_one(
        predictions_ub_raw_all_dates, prediction_output_ub_pdv, prediction_ub_pdv
    )

    if (not prediction_output_ub_pdv.empty) and (not predictions_ub_raw_all_dates.empty):
        prediction_ub_pdv = weighted_ensembling_predictions(prediction_ub_pdv, config)

    return prediction_ub_pdv
