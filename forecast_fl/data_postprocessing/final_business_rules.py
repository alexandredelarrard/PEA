#
# Copyright (c) 2023 by Boston Consulting Group. All rights reserved
#
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd


def business_rules_post_processing(df_data_to_export: pd.DataFrame) -> pd.DataFrame:
    """Business rules to ensure volume of sales is sufficiently high to avoid any Out of stock

    Args:
        df_data_to_export (pd.DataFrame): predictions of sales

    Returns:
        pd.DataFrame: updated predictions sales with top up
    """

    if not df_data_to_export.empty:

        # BR 1 no sales proposed if less than 0 KG sales
        df_data_to_export["PREDICTION_UB_POSTPROCESSED"] = np.where(
            df_data_to_export["PREDICTION_UB_POSTPROCESSED"] < 0, 0, df_data_to_export["PREDICTION_UB_POSTPROCESSED"]
        )

    return df_data_to_export
