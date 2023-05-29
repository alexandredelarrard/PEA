import time
import re
from typing import List
import numpy as np
import os
import yaml
from pathlib import Path as pl

def load_configs(config_yml):
    config_path = "./configs"
    config_yaml = open(pl(config_path) / pl(config_yml), encoding="utf-8")
    configs = yaml.load(config_yaml, Loader=yaml.FullLoader)
    return configs


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f s' % \
                  (method.__name__, (te - ts)))
        return result

    return timed


def smart_column_parser(col_names: List) -> List:
    """Rename columns to upper case and remove space and 
    non conventional elements

    Args:
        col_names (List): original column names

    Returns:
        List: clean column names
    """
    
    new_list = []
    for var in col_names:
        var = str(var).replace("/", " ")  # replace space by underscore
        new_var = re.sub("[ ]+", "_", str(var))  # replace space by underscore
        new_var = re.sub(
            "[^A-Za-z0-9_]+", "", new_var
        )  # only alphanumeric characters and underscore are allowed
        new_var = re.sub("_$", "", new_var)  # variable name cannot end with underscore
        new_var = new_var.upper()  # all variables should be upper case
        new_list.append(new_var)
    return new_list


def weight_history(df, date_name, k=4):
    """
    weight decay to train on most up to date data with an exponential decay
    k=4 means that we have a weight of 4 for recent data, 1 for 1 year later and 0.25 2 years later
    k=2 means that we have a weight of 2 for recent data, 1 for 1 year later and 0.5 2 years later
    """
    return k*np.exp((df[date_name] - df[date_name].max()).dt.days/(1075/k))
    

def check_save_path(save_path):
    if not os.path.exists(save_path):
        os.mkdir(save_path)

def function_weight():
    return lambda x: 1 - 1 / (1 + np.exp(-1 * (x / (365) - 5)))