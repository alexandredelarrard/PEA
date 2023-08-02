import pandas as pd
import yaml
from pathlib import Path as pl
from datetime import datetime

from crawling.extract_stock_history import main_extract_stock

today = datetime.today().strftime("%Y-%m-%d")

def load_configs(config_yml):
    config_path = "./configs"
    config_yaml = open(pl(config_path) / pl(config_yml), encoding="utf-8")
    configs = yaml.load(config_yaml, Loader=yaml.FullLoader)
    return configs


if __name__ == "__main__" : 
    """TODO : 
        - PROFILES: strategie autour de la tete dirigeante 
        - NEWS: strategie autour des news (extraire en plus)
        - Refaire extraction en fonction des nouvelles infos ou companies (pas tout refaire chaque fois)
        - REVIEW INTRINSIC VALUE
        - DATA PREP : add quarter data for cash & balance
        - DATA PREP -> take into account non op income (bias the ranking )
        - PREDICT NEXT QUARTER EVOLUTION BASED ON UNDERLYING COMO EVOLUTION
        - STOCK: predict which stock is good opp as it is cheap, give range where to sell 

    4 métriques principales :
        - acces aux financements 
        - barrières à l'entrée 
        - bonne équipe d'exectution 
        - croissance / produit qui se vend bien 
    """

    full = False
    configs_general = load_configs("configs_pea.yml")
    base_path = configs_general["resources"]["base_path"]
    sub_loc= base_path / pl("data/extracted_data")

    data = pd.read_csv(base_path / pl("data/data_for_crawling/mapping_reuters_yahoo.csv"), sep=";", encoding="latin1")
    data["SECTOR"] = data["SECTOR"].apply(lambda x: x.strip())
    data = data.loc[data["Country"] == "FR"]
    data = data.loc[(data["SBF"] == 1)&(data["ETAT"] == 0)]

    stocks, missing_ticks = main_extract_stock(data, sub_loc=sub_loc, split_size = 100)
    
   