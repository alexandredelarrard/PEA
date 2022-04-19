import pandas as pd
import yaml
from pathlib import Path as pl
from datetime import datetime

from data_prep.sbf120.merge_all_infos import final_data_prep
from data_prep.sbf120.main_data_prep import main_analysis_financials

from modelling.picking.strat2_sector import closest_per_sector
from modelling.picking.strat3_stock import leverage_oecd_data
from modelling.picking.strat4_modelling import post_data_prep, modelling_pe
from modelling.picking.strat5_financials import business_rules

from crawling.extract_stock_history import main_extract_stock
from crawling.extract_cotations import main_extract_financials

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

    configs_general = load_configs("configs_pea.yml")
    base_path = configs_general["resources"]["base_path"]
    sub_loc= base_path / pl("data/extracted_data")

    data = pd.read_csv(base_path / pl("data/data_for_crawling/mapping_reuters_yahoo.csv"), sep=";")
    data["SECTOR"] = data["SECTOR"].apply(lambda x: x.strip())
    
    # Data extraction
    # missing_fi = main_extract_financials(configs_general, data, cores=4, sub_loc=sub_loc)
    missing_ticks = main_extract_stock(data, sub_loc=sub_loc, split_size = 25)

    # Data Preparation
    final_inputs = {}
    final_inputs["results"] = main_analysis_financials(configs_general)
    final_inputs["neighbors"] = closest_per_sector(final_inputs, data, n_neighbors=16)

    final = final_data_prep(data, final_inputs)
    df = post_data_prep(final)

    # add FOREWARD P/E prediction
    results, models, predictions = modelling_pe(df, configs_general)
    df = pd.merge(df, predictions, left_index=True, right_index=True, how="left")

    # Business rules to identify high potential companies
    df = business_rules(df)

    # oecd data 
    # df = leverage_oecd_data(base_path, data, df)

    # df.to_csv(r"C:\Users\de larrard alexandre\Documents\repos_github\PEA\data\results\financials_V2.csv", sep=";")
