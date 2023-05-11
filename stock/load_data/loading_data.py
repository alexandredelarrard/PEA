import pandas as pd 
import os
from pathlib import Path as pl
from typing import Dict
from glob import glob
from utils.general_functions import smart_column_parser


def load_mapping_reuters(configs : Dict) -> pd.DataFrame:
    """Come from https://www.reuters.com/

    Args:
        configs (Dict): [description]

    Returns:
        pd.DataFrame: [description]
    """

    base_path = pl(configs["resources"]["base_path"])
    mapping = pd.read_csv(base_path / pl(configs["Internal_data"]["reuters"]), sep=";")
    mapping.columns = smart_column_parser(mapping.columns)
    mapping["NAME"] = mapping["NAME"].apply(lambda x : x.upper().strip())

    return mapping