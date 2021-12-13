import pandas as pd
import yaml
from pathlib import Path as pl
from datetime import datetime

from crawling.extract_cotations import main_extract_financials
from crawling.extract_stock_history import main_extract_stock
from data_prep.sbf120.main_data_prep import main_analysis_financials
# from data_prep.sbf120.stock import stocks_analysis

from modelling.picking.strat2_sector import company_neighbors

today = datetime.today().strftime("%Y-%m-%d")


def load_configs(config_yml):
    config_path = "./configs"
    config_yaml = open(pl(config_path) / pl(config_yml), encoding="utf-8")
    configs = yaml.load(config_yaml, Loader=yaml.FullLoader)
    return configs


def main_financial_strats(final_inputs):
    return 0


if __name__ == "__main__" : 
    """TODO : 
        - differencier trend quarter de annual -> comment utiliser les deux ? 
        - reflechir comment utiliser past years financials pourprediction autre que trend
        - lier quarter / annual indicators au cours 
        - créer des indicateurs de cours interessant (volatilité, lié au volume, etc.)
        - ameliorer text descriptif pour rapprochement secteur 
        - caractériser chaque secteur par distance au mot desc (eg: luxe, tech, vaccin 2.0, etc.)
        - strategie autour de la tete dirigeante 
        - strategie autour des news (extraire en plus)
    """

    configs_general = load_configs("configs_pea.yml")
    base_path = configs_general["resources"]["base_path"]
    sub_loc=configs_general["resources"]["base_path"] / pl("data/extracted_data")

    datas = {"data_mapping_yahoo" : pd.read_csv(base_path / pl("data/data_for_crawling/mapping_reuters_yahoo.csv"), sep=";")}
    
    # extraction
    # main_extract_stock(datas["data_mapping_yahoo"], sub_loc=sub_loc)
    # main_extract_financials(configs_general, datas["data_mapping_yahoo"], cores=3, sub_loc=sub_loc)

    final_inputs = {}

    # analysis 
    final_inputs["results"] = main_analysis_financials(configs_general)
    final_inputs["neighbors"] = company_neighbors(final_inputs, n_neighbors=26)

    # scoring 
    # scoring = main_financial_strats(final_inputs)

    # if not os.path.exists(base_path /  pl(f"data/results/")):
    #     os.mkdir(base_path /  pl(f"data/results/"))
    # scoring.to_csv(base_path /  pl(f"data/results/{today}.csv"))

   