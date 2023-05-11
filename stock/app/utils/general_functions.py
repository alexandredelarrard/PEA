import time
import re
import os
import pandas as pd
from typing import List


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
        new_var = re.sub("[ ]+", "_", str(var))  # replace space by underscore
        new_var = re.sub(
            "[^A-Za-z0-9_]+", "", new_var
        )  # only alphanumeric characters and underscore are allowed
        new_var = re.sub("_$", "", new_var)  # variable name cannot end with underscore
        new_var = new_var.upper()  # all variables should be upper case
        new_list.append(new_var)
    return new_list
