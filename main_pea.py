import pandas as pd 
import yaml
import os
from pathlib import Path as pl
from crawling.extract_cotations import main_extract_financials
from crawling.extract_stock_history import main_extract_stock
from data_prep.sbf120.low_frequency import main_analysis_financials
from datetime import datetime
today = datetime.today().strftime("%Y-%m-%d")

def load_configs(config_yml):
    config_path = "./configs"
    config_yaml = open(pl(config_path) / pl(config_yml), encoding="utf-8")
    configs = yaml.load(config_yaml, Loader=yaml.FullLoader)
    return configs


def main_extract(configs_general, extract=False):

    base_path = configs_general["resources"]["base_path"]
    sub_loc=configs_general["resources"]["base_path"] / pl("data/extracted_data")

    datas = {}
    datas["data_mapping_reuters"] = pd.read_csv(base_path / pl("data/data_for_crawling/reuters_mapping.csv"), sep=";")
    datas["data_mapping_yahoo"] = pd.read_csv(base_path / pl("data/data_for_crawling/mapping_reuters_yahoo.csv"), sep=";")

    if extract: 
        main_extract_financials(configs_general, datas["data_mapping_reuters"], sub_loc=sub_loc)

    main_extract_stock(configs_general, datas["data_mapping_yahoo"], sub_loc=sub_loc)

    return datas

if __name__ == "__main__" : 
    configs_general = load_configs("configs_pea.yml")
    base_path = configs_general["resources"]["base_path"]
    
    # extraction
    main_extract(configs_general, True)
    
    # analysis 
    results = main_analysis_financials(configs_general)
    results = pd.DataFrame(results).sort_index()
    if not os.path.exists(base_path /  pl(f"data/results/")):
        os.mkdir(base_path /  pl(f"data/results/"))
    results.to_csv(base_path /  pl(f"data/results/{today}.csv"))


# test mvs 
# 4 -> Pas de R&D (ABIO.PA, AIRF.PA, WLN.PA, SGOB.PA)
# 8 -> Pas R&D ni DEPRECIATION_AMORTIZATION_IN_OP_COST (ACCP.PA, LVMH.PA, FNAC.PA)
# 5 -> 
# 24 ->  