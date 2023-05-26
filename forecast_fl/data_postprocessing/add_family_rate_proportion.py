#
# Copyright (c) 2023 by Boston Consulting Group. All rights reserved
#
# -*- coding: utf-8 -*-

from datetime import timedelta
from typing import Dict

import pandas as pd


def add_family_rate(config: Dict, datas: Dict, stock: pd.DataFrame):

    broken_rate = config.load.prediction_mode.broken_rate
    min_days_stock = config.load.prediction_mode.min_days_stock
    granularity = config.load["prediction_granularity"]

    if granularity == "CLUSTER":
        granularity = "PDV"

    familles = datas["SOUS_FAMILLE_SQL"]
    rates = datas["RATE_FAMILLES"]
    rates["LIB_FAM"] = rates["LIB_FAM"].apply(lambda x: x.strip().upper())

    for col in [f"SPOILAGE_RATE_{granularity}", f"MIN_PROPORTION_STOCK_{granularity}"]:
        rates[col] = rates[col].apply(lambda x: str(x).replace(",", ".")).astype(float)

    rates = rates.drop_duplicates("LIB_FAM")  # 2 aromatiques

    familles = familles.merge(rates, left_on="SOUS_FAMILLE", right_on="LIB_FAM", how="left", validate="m:1")
    familles = familles[
        ["UB_CODE", f"SPOILAGE_RATE_{granularity}", f"MIN_PROPORTION_STOCK_{granularity}", "MIN_THRESHOLD_STOCK"]
    ]
    familles = familles.drop_duplicates("UB_CODE")

    # merge with df_export
    stock = stock.merge(familles, on="UB_CODE", how="left", validate="m:1")

    # fill missing values
    stock[f"SPOILAGE_RATE_{granularity}"] = stock[f"SPOILAGE_RATE_{granularity}"].fillna(broken_rate)
    stock[f"MIN_PROPORTION_STOCK_{granularity}"] = stock[f"MIN_PROPORTION_STOCK_{granularity}"].fillna(min_days_stock)

    return stock
