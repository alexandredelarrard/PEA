import pandas as pd 
import yaml
from pathlib import Path as pl
import load_data.loading_data as  ld

import data.crawling.extract_cotations as ex_rt

def load_configs(config_path):
    config_yaml = open(pl(config_path) / pl("configs.yml"))
    configs = yaml.load(config_yaml, Loader=yaml.FullLoader)
    return configs


if __name__ == "__main__" : 

    proxy=False
    configs = load_configs('config')
    base_path = pl(configs["resources"]["base_path"])

    datas = {"mapping_reuters" : ld.load_mapping_reuters(configs)}

    savepath = base_path / pl("data/extracted_data/reuters")
    ex_rt.main_extract(configs, datas, proxy=False, cores=2, sub_loc=savepath)
