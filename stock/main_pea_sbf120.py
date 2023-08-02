import pandas as pd
import yaml
from pathlib import Path as pl
from datetime import datetime

from crawling.extract_pe import main_extract_stock_pe

today = datetime.today().strftime("%Y-%m-%d")

def load_configs(config_yml):
    config_path = "./configs"
    config_yaml = open(pl(config_path) / pl(config_yml), encoding="utf-8")
    configs = yaml.load(config_yaml, Loader=yaml.FullLoader)
    return configs


if __name__ == "__main__" : 

    configs_general = load_configs("configs_pea.yml")
    base_path = configs_general["resources"]["base_path"]
    sub_loc= base_path / pl("data/extracted_data")

    data = pd.read_csv(base_path / pl("data/data_for_crawling/mapping_reuters_yahoo.csv"), sep=";")
    data["SECTOR"] = data["SECTOR"].apply(lambda x: x.strip())
    data = data.loc[data["Country"] == "FR"]

    # extract latest data SBF120
    missing_ticks = main_extract_stock(data, sub_loc=sub_loc, split_size = 50)
    # missing_ticks_pe = main_extract_stock_pe(data)

    # 
  