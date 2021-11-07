import pandas as pd 
import numpy as np 
import re

import yaml
from pathlib import Path as pl
import matplotlib.pyplot as plt
from typing import Dict
import seaborn as sns 
sns.set_style(style="darkgrid")
from modelling.utils.modelling_lightgbm import TrainModel

from utils.general_functions import smart_column_parser


def modelling_assessment(full_clean : pd.DataFrame, 
                        configs_model: Dict):
    """Train model based on input data full_clean. 
    Model is trained on a stratified K-fold basis based on the features defined 
    in configs_whisky_modelling.yml file. 
    We are training a lightgbm classifier with logloss to minimize. 

    Args:
        full_clean (pd.DataFrame): [cleaned whisky input]
        configs_model (Dict): [hyperparameters and features of the model]

    Returns:
        [type]: [predictions with the model]
    """                        
    model_lgb_reg = TrainModel(configs=configs_model["regression_model"], data=full_clean)
    predictions, model = model_lgb_reg.modelling_cross_validation()
    return  model_lgb_reg, predictions, model


def main_btc_training(full, configs_model):
    model_lgb_reg, predictions, model = modelling_assessment(full, configs_model)